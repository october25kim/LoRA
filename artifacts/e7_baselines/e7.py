#!/usr/bin/env python3
"""E7 (EXPLORATORY / POST HOC, not preregistered): label-free lambda-selection baselines vs U1, compute cost, small 3/4-task merges.
Reuses E6 (artifacts/e6_decoder2) code UNMODIFIED via import (Enc, get_data, load_ad, bm.delta, bm.score, bm.compact) and the E6
calibration split (lam_confirm_e6.split, verbatim). Writes ONLY into artifacts/e7_baselines/. See PLAN_E7.md.
Stages:
  a1        per pair (P,R0,S1 = 273): singles + merged(G7) forward on the <=200 cal inputs per task -> entropies, agreement, timings, peak mem
  holdtime  30 random pairs: held-out lambda tuning on the 2x1000 held-out train examples (G4) -> timings (+ check lam_selected reproduces)
  c         30 random triples + 15 random quads: TA lambda*(sum dW), G7, held-out + eval forward -> per-lambda scores, cal entropy/agreement
  a2        AdaMerging-style gradient optimisation of task-wise (lam1, lam2) on cal entropy, pop P (then R0, S1 if time), eval on test split
"""
import argparse, json, math, os, sys, time, itertools
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E6D = HERE.parent / "e6_decoder2"
sys.path.insert(0, str(E6D))
import e6                                   # noqa: E402  (unmodified; importing only defines things + reads prereg_e6.json)
import lam_confirm_e6 as LC                 # noqa: E402  (verbatim split/score)
bm = e6.bm
G4 = [0.3, 0.5, 0.7, 1.0]; G7 = G4 + [1.3, 1.5, 2.0]
BB = "qwen15"; BS = e6.BS
LOGF = HERE / "e7.log"


def log(m):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(m)
    print(s, flush=True)
    with open(LOGF, "a") as f: f.write(s + "\n")


def status(stage, done, total, extra=""):
    (HERE / "STATUS.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} stage={stage} done={done}/{total} {extra}\n")


def done_keys(p):
    return {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def jl_append(p, rec):
    with open(p, "a") as f: f.write(json.dumps(rec) + "\n")


def is_reg(t): return e6.metric_name(t) == "spearman"


def entropy(logits):
    """mean Shannon entropy (nats) of softmax(logits) and the same divided by log(C)."""
    z = logits.astype(np.float64); z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True)); p = np.exp(lp)
    H = -(p * lp).sum(1)
    return float(H.mean()), float(H.mean() / math.log(logits.shape[1]))


def agree(t, merged_logits, single_compact):
    return LC.score(bm.compact(t, merged_logits), single_compact)


_DATA, _CAL = {}, {}
def data(t):
    if t not in _DATA: _DATA[t] = e6.get_data(t)
    return _DATA[t]


def cal_data(t):
    """the <=200 E6 calibration (unlabeled) eval inputs of task t (same boolean mask as lam_confirm_e6.split), one dict per task
    (Enc caches batches by id(data))."""
    if t not in _CAL:
        d = data(t)["eval"]; n = len(d["labels"]); cal = LC.split(BB, t, n); idx = np.where(cal)[0]
        _CAL[t] = (cal, idx, {"ids": [d["ids"][i] for i in idx], "tt": [d["tt"][i] for i in idx], "labels": d["labels"][idx]})
    return _CAL[t]


_SINGLE = {}
def single_npz(k):
    if k not in _SINGLE: _SINGLE[k] = dict(np.load(E6D / "preds" / f"single_{k}.npz"))
    return _SINGLE[k]


_S0 = {}
def single_hold(k):
    s = e6.seed_of(k)
    if s not in _S0: _S0[s] = json.loads((E6D / f"s{s}" / "stage0.json").read_text())
    return _S0[s]["per_task"][e6.task_of(k)]["hold"]["main"]


class Timer:
    def __init__(self, torch): self.torch = torch
    def __enter__(self):
        self.torch.cuda.synchronize(); self.torch.cuda.reset_peak_memory_stats(); self.t0 = time.time(); return self
    def __exit__(self, *a):
        self.torch.cuda.synchronize(); self.s = time.time() - self.t0; self.peak = self.torch.cuda.max_memory_allocated() / 2**30


class LazyD:
    def __init__(self, ad, dev): self.ad, self.dev = ad, dev
    def __getitem__(self, n): return bm.delta(self.ad, n, self.dev)


def pops_pairs(pops=("P", "R0", "S1")):
    PO = json.loads((E6D / "populations_e6.json").read_text())
    return [(p, tuple(x.split("__"))) for p in pops for x in PO["populations"][p]], PO["lora_layers"]


# ------------------------------------------------------------------ stage a1
def st_a1(a):
    torch = bm.setup_torch(); dev = torch.device("cuda")
    pairs, names = pops_pairs(); out = HERE / "a1_entropy.jsonl"; done = done_keys(out)
    enc = e6.Enc(dev, names); ADS = {}
    ad = lambda k: ADS[k] if k in ADS else ADS.setdefault(k, e6.load_ad(k))   # (setdefault alone would reload eagerly)
    for t in sorted({e6.task_of(k) for _, pr in pairs for k in pr}): enc.batches(cal_data(t)[2], BS)   # prebuild padded cal batches (untimed)
    for i, (pop, (x, y)) in enumerate(pairs):
        key = e6.pid(x, y)
        if key in done: continue
        status("a1", i, len(pairs), key)
        rec = {"id": key, "pop": pop, "t1": x, "t2": y, "timing": {}, "peak_gb": {}, "n_fwd": {}}
        Z = dict(np.load(E6D / "preds" / f"pair_{key}.npz"))
        ks = [(x, e6.task_of(x)), (y, e6.task_of(y))]
        for k, _ in ks: ad(k)                     # adapter loading (disk + sha256) is shared by all rules -> excluded from timings
        # singles on cal (needed by U1/M3 only)
        sl = {}
        with Timer(torch) as T:
            for k, t in ks:
                enc.set_delta_fn(lambda n: bm.delta(ad(k), n, dev))
                sl[k] = enc.predict(cal_data(t)[2], ad(k)["head"], BS)
        rec["timing"]["singles_cal_s"] = T.s; rec["peak_gb"]["singles_cal"] = T.peak
        rec["n_fwd"]["singles_cal_examples"] = int(sum(len(cal_data(t)[1]) for _, t in ks)); rec["n_fwd"]["weight_writes_singles"] = 2
        # consistency: recomputed single cal argmax vs cached E6 single eval preds
        rec["single_cal_match"] = [float(np.mean(bm.compact(t, sl[k]) == single_npz(k)["eval"][cal_data(t)[1]])) if not is_reg(t)
                                   else float(np.max(np.abs(bm.compact(t, sl[k]) - single_npz(k)["eval"][cal_data(t)[1]]))) for k, t in ks]
        d1, d2 = LazyD(ad(x), dev), LazyD(ad(y), dev)
        for lam in G7:
            with Timer(torch) as T:
                enc.set_delta_fn(lambda n: lam * (d1[n] + d2[n]))
                ml = {k: enc.predict(cal_data(t)[2], ad(k)["head"], BS) for k, t in ks}
            rec["timing"][f"merged_cal_lam{lam}_s"] = T.s; rec["peak_gb"][f"merged_cal_lam{lam}"] = T.peak
            for j, (k, t) in enumerate(ks, 1):
                cal, idx, cd = cal_data(t)
                if is_reg(t):
                    rec[f"ent{j}_{lam}"] = None; rec[f"entn{j}_{lam}"] = None
                else:
                    rec[f"ent{j}_{lam}"], rec[f"entn{j}_{lam}"] = entropy(ml[k])
                rec[f"agree{j}_{lam}"] = agree(t, ml[k], single_npz(k)["eval"][idx])            # recomputed (check vs E6 cached)
                cm = Z[f"TA_{k}_lam{lam}"][idx]; rm = bm.compact(t, ml[k])
                rec[f"match{j}_{lam}"] = float(np.mean(rm == cm)) if not is_reg(t) else float(np.max(np.abs(rm - cm)))
        rec["n_fwd"]["merged_cal_examples_per_lam"] = int(sum(len(cal_data(t)[1]) for _, t in ks))
        rec["ncal"] = [int(len(cal_data(t)[1])) for _, t in ks]
        jl_append(out, rec)
        log(f"a1 {i+1}/{len(pairs)} {key}: t_single={rec['timing']['singles_cal_s']:.2f}s t_lam1={rec['timing']['merged_cal_lam1.0_s']:.2f}s "
            f"match={[rec[f'match1_{l}'] for l in G4]} ent={[rec.get(f'ent1_{l}') for l in G4]}")
        if len(ADS) > 8: ADS.clear()
    status("a1", len(pairs), len(pairs), "DONE")


# ------------------------------------------------------------------ stage holdtime
def st_holdtime(a):
    torch = bm.setup_torch(); dev = torch.device("cuda")
    pairs, names = pops_pairs(); rng = np.random.default_rng(20260929)
    sub = [pairs[i] for i in sorted(rng.choice(len(pairs), size=a.n, replace=False))]
    out = HERE / "holdtime.jsonl"; done = done_keys(out)
    recs = {r["pair"]: r for p in ("P", "R0", "S1") for r in LC.jl(E6D / f"pair_results_{p}.jsonl")}
    enc = e6.Enc(dev, names)
    for t in sorted({e6.task_of(k) for _, pr in sub for k in pr}): enc.batches(data(t)["hold"], BS)       # prebuild (untimed)
    for i, (pop, (x, y)) in enumerate(sub):
        key = e6.pid(x, y)
        if key in done: continue
        status("holdtime", i, len(sub), key)
        A, Bd = e6.load_ad(x), e6.load_ad(y); d1, d2 = LazyD(A, dev), LazyD(Bd, dev)
        rec = {"id": key, "pop": pop, "timing": {}, "peak_gb": {}}
        sel = {}
        for lam in G4:
            with Timer(torch) as T:
                enc.set_delta_fn(lambda n: lam * (d1[n] + d2[n]))
                sc = {k: bm.score(e6.task_of(k), enc.predict(data(e6.task_of(k))["hold"], ad["head"], BS), data(e6.task_of(k))["hold"]["labels"])["main"]
                      for k, ad in ((x, A), (y, Bd))}
            rec["timing"][f"hold_lam{lam}_s"] = T.s; rec["peak_gb"][f"hold_lam{lam}"] = T.peak
            sel[lam] = 0.5 * (sc[x] / single_hold(x) + sc[y] / single_hold(y))
            rec[f"hold_norm_{lam}"] = sel[lam]; rec[f"hold_norm_absdiff_vs_e6_{lam}"] = abs(sel[lam] - recs[key][f"TA_hold_norm_lam{lam}"])
        rec["lam_sel"] = G4[int(np.argmax([sel[l] for l in G4]))]; rec["lam_sel_e6"] = recs[key]["lam_selected"]
        rec["n_hold_examples_per_lam"] = int(len(data(e6.task_of(x))["hold"]["labels"]) + len(data(e6.task_of(y))["hold"]["labels"]))
        jl_append(out, rec); log(f"holdtime {i+1}/{len(sub)} {key}: {sum(rec['timing'].values()):.1f}s lam_sel={rec['lam_sel']} (e6 {rec['lam_sel_e6']})")
    status("holdtime", len(sub), len(sub), "DONE")


# ------------------------------------------------------------------ stage c (3-task / 4-task)
def tuples_c():
    PO = json.loads((E6D / "populations_e6.json").read_text()); V = PO["valid_tasks"]
    rng = np.random.default_rng(20260930); out = []
    for m, n in ((3, 30), (4, 15)):
        seen = set()
        while len([o for o in out if len(o) == m]) < n:
            ts = sorted(rng.choice(len(V), size=m, replace=False).tolist()); ss = rng.integers(0, 2, size=m).tolist()
            tup = tuple(f"{V[t]}@s{s}" for t, s in zip(ts, ss))
            if tup in seen: continue
            seen.add(tup); out.append(tup)
    return out, PO["lora_layers"]


def st_c(a):
    torch = bm.setup_torch(); dev = torch.device("cuda")
    tups, names = tuples_c(); out = HERE / "c_multitask.jsonl"; done = done_keys(out)
    pdir = HERE / "preds_c"; pdir.mkdir(exist_ok=True)
    enc = e6.Enc(dev, names)
    for i, tup in enumerate(tups):
        key = "__".join(tup)
        if key in done: continue
        status("c", i, len(tups), key)
        ads = {k: e6.load_ad(k) for k in tup}; D = [LazyD(ads[k], dev) for k in tup]
        rec = {"id": key, "m": len(tup), "keys": list(tup), "tasks": [e6.task_of(k) for k in tup], "timing": {}}; preds = {}
        tp = time.time()
        for lam in G7:
            enc.set_delta_fn(lambda n: lam * sum(d[n] for d in D))
            hn = []
            for j, k in enumerate(tup, 1):
                t = e6.task_of(k); dd = data(t)
                lh = enc.predict(dd["hold"], ads[k]["head"], BS); le = enc.predict(dd["eval"], ads[k]["head"], BS)
                h = bm.score(t, lh, dd["hold"]["labels"])["main"]; hn.append(h / single_hold(k))
                cal, idx, _ = cal_data(t); te = ~cal; sc = bm.compact(t, le); lab = dd["eval"]["labels"]; se = single_npz(k)["eval"]
                rec[f"hold{j}_{lam}"] = h; rec[f"test{j}_{lam}"] = LC.score(sc[te], lab[te]); rec[f"calacc{j}_{lam}"] = LC.score(sc[cal], lab[cal])
                rec[f"agree{j}_{lam}"] = LC.score(sc[cal], se[cal])
                rec[f"ent{j}_{lam}"], rec[f"entn{j}_{lam}"] = (None, None) if is_reg(t) else entropy(le[cal])
                preds[f"TA_{k}_lam{lam}"] = sc
            rec[f"hold_norm_{lam}"] = float(np.mean(hn))
        for j, k in enumerate(tup, 1):
            t = e6.task_of(k); cal, _, _ = cal_data(t); se = single_npz(k)["eval"]; lab = data(t)["eval"]["labels"]
            rec[f"single_test{j}"] = LC.score(se[~cal], lab[~cal]); rec[f"single_cal{j}"] = LC.score(se[cal], lab[cal])
        rec["secs"] = time.time() - tp
        np.savez_compressed(pdir / f"c_{key}.npz", **preds); jl_append(out, rec)
        log(f"c {i+1}/{len(tups)} {key}: {rec['secs']:.0f}s hold_norm={[round(rec[f'hold_norm_{l}'],4) for l in G4]}")
        del ads, D; torch.cuda.empty_cache()
    status("c", len(tups), len(tups), "DONE")


# ------------------------------------------------------------------ stage a2 (AdaMerging-style gradient, task-wise lambdas)
def st_a2(a):
    torch = bm.setup_torch(); dev = torch.device("cuda")
    pairs, names = pops_pairs(tuple(a.pops.split(","))); out = HERE / "a2_adamerge.jsonl"; done = done_keys(out)
    enc = e6.Enc(dev, names); m = enc.m
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.config.use_cache = False
    for p in m.parameters(): p.requires_grad_(False)
    STEPS, LR, MB, INIT, LO, HI = a.steps, a.lr, a.mb, a.init, 0.0, 2.5
    state = {"lora": None, "lam": None}

    def hook(name):
        def f(mod, inp, outp):
            if state["lora"] is None: return outp
            xx = inp[0]; add = 0
            for j, (A, B, s) in enumerate(state["lora"][name]):
                add = add + state["lam"][j] * s * ((xx @ A.T) @ B.T)
            return outp + add.to(outp.dtype)
        return f
    hs = [enc.lin[n].register_forward_hook(hook(n)) for n in names]

    def batch(t, idx):
        cd = cal_data(t)[2]; ids = [cd["ids"][i] for i in idx]; L = max(len(v) for v in ids)
        ii = np.full((len(ids), L), enc.pad, dtype=np.int64); am = np.zeros_like(ii)
        for r, v in enumerate(ids): ii[r, :len(v)] = v; am[r, :len(v)] = 1
        return torch.from_numpy(ii).to(dev), torch.from_numpy(am).to(dev)

    for i, (pop, (x, y)) in enumerate(pairs):
        key = e6.pid(x, y)
        if key in done: continue
        status("a2", i, len(pairs), key)
        ks = [x, y]; ads = {k: e6.load_ad(k) for k in ks}; tasks = [e6.task_of(k) for k in ks]
        cls = [j for j, t in enumerate(tasks) if not is_reg(t)]
        rec = {"id": key, "pop": pop, "steps": STEPS, "lr": LR, "mb": MB, "init": INIT, "clamp": [LO, HI], "entropy_tasks": [tasks[j] for j in cls]}
        enc.set_delta(None)                                                   # base weights; adapters applied as LoRA branches (bf16)
        state["lora"] = {n: [(ads[k]["layers"][n][0].to(dev, torch.bfloat16), ads[k]["layers"][n][1].to(dev, torch.bfloat16), ads[k]["scaling"]) for k in ks] for n in names}
        lam = torch.full((2,), INIT, device=dev, dtype=torch.float32, requires_grad=True); state["lam"] = lam
        opt = torch.optim.Adam([lam], lr=LR); rng = np.random.default_rng(20260929 + i); hist = []
        m.train()
        with Timer(torch) as T:
            for step in range(STEPS):
                opt.zero_grad(set_to_none=True); tot = 0.0
                for j in cls:
                    k, t = ks[j], tasks[j]; n = len(cal_data(t)[1])
                    ii, am = batch(t, rng.choice(n, size=min(MB, n), replace=False))
                    h = m(input_ids=ii, attention_mask=am).last_hidden_state
                    hl = h[torch.arange(h.shape[0], device=dev), am.sum(1) - 1].float()
                    lg = hl @ ads[k]["head"]["weight"].to(dev).float().T
                    lp = torch.log_softmax(lg, -1); H = -(lp.exp() * lp).sum(-1).mean() / len(cls)
                    H.backward(); tot += float(H)
                opt.step()
                with torch.no_grad(): lam.clamp_(LO, HI)
                hist.append([tot] + lam.detach().cpu().tolist())
        m.eval()
        rec["opt_s"] = T.s; rec["opt_peak_gb"] = T.peak; rec["hist"] = hist; rec["lam"] = lam.detach().cpu().tolist()
        rec["n_fwd_bwd_examples"] = int(STEPS * sum(min(MB, len(cal_data(tasks[j])[1])) for j in cls))
        state["lora"] = None; state["lam"] = None; torch.cuda.empty_cache()
        l1, l2 = rec["lam"]; D1, D2 = LazyD(ads[x], dev), LazyD(ads[y], dev)
        with Timer(torch) as T2:
            enc.set_delta_fn(lambda n: l1 * D1[n] + l2 * D2[n])
            for j, k in enumerate(ks, 1):
                t = tasks[j - 1]; cal = cal_data(t)[0]; le = enc.predict(data(t)["eval"], ads[k]["head"], BS)
                sc = bm.compact(t, le); lab = data(t)["eval"]["labels"]
                rec[f"test{j}"] = LC.score(sc[~cal], lab[~cal]); rec[f"agree{j}"] = LC.score(sc[cal], single_npz(k)["eval"][cal])
        rec["eval_s"] = T2.s
        jl_append(out, rec); log(f"a2 {i+1}/{len(pairs)} {key}: lam={[round(v,3) for v in rec['lam']]} H {hist[0][0]:.3f}->{hist[-1][0]:.3f} opt {rec['opt_s']:.0f}s peak {rec['opt_peak_gb']:.1f}GB")
        del ads; torch.cuda.empty_cache()
    status("a2", len(pairs), len(pairs), "DONE")
    for h in hs: h.remove()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--pops", default="P"); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mb", type=int, default=16); ap.add_argument("--init", type=float, default=0.3)
    a = ap.parse_args(); log(f"=== e7.py {sys.argv}")
    {"a1": st_a1, "holdtime": st_holdtime, "c": st_c, "a2": st_a2}[a.stage](a)


if __name__ == "__main__":
    main()
