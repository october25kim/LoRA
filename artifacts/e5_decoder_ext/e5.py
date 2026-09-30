#!/usr/bin/env python3
"""E5 ("raise acceptance odds") GPU-side experiments on the E4b Qwen2.5-0.5B adapters (PREREG_E5.md, prereg_e5.json).
Directory: artifacts/e5_accept/ on ubuntu-4070. Reads E4b (artifacts/e4b_decoder) and E3 (artifacts/e3_baselines/code) READ-ONLY:
e4b.py, e1b.py, e3.py, merges.py are imported UNMODIFIED (no bytecode); nothing is written outside artifacts/e5_accept/.
Stages:
  smoke  pipeline test on THROWAWAY pilot adapters (pilot/lr1e-4 sst2 x pilot/lr1e-3 rte; NOT study adapters), singles set to 1.0,
         output to smoke/ (deleted/ignored; no study numbers)
  e5a    method comparison on population P (91 mixed-seed cross-task pairs): e3.run() with the E5 grids (TA, PICO_TA, GATE, FORCEGATE,
         TIES, PICO_GATE), pairs processed in the pre-registered seeded order, deadline-guarded  -> e5a/results.jsonl, e5a/preds/
  e5b    TA at the extended lambdas {1.3, 1.5, 2.0} (held-out + eval, E4b stage-2 arithmetic) for P, then R0, then S1 -> e5b/pair_ext_<pop>.jsonl
  e5d    lemma quantities (paper/verify_lemma.py lemma_stats, unmodified) on every theta*=30 FAIL layer of P/R0/S1/S2 and on the forced
         k=1 direction of every layer of P -> e5d/
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ART = HERE.parent
E4B = ART / "e4b_decoder"
E3C = ART / "e3_baselines" / "code"
PR5 = json.loads((HERE / "prereg_e5.json").read_text())
LOGF = HERE / "run_e5.log"
sys.path.insert(0, str(E4B))
import e4b as E                                   # noqa: E402  unmodified E4b module (Enc, load_ad, get_data, singles, bm helpers)
bm = E.bm
bm.LOGF = LOGF; E.LOGF = LOGF                     # never log into the E4b directory
DEADLINE = float(os.environ.get("E5_DEADLINE_EPOCH", "0"))


def log(m):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(m)
    print(s, flush=True)
    with open(LOGF, "a") as f: f.write(s + "\n")


def past_deadline(): return DEADLINE > 0 and time.time() > DEADLINE


def timing_add(k, s):
    p = HERE / "timing_e5.json"; d = json.loads(p.read_text()) if p.exists() else {}
    d[k] = d.get(k, 0.0) + s; p.write_text(json.dumps(d, indent=1))


def check_hashes():
    for fn in ("prereg_e5.sha256", "code_e5.sha256"):
        for ln in (HERE / fn).read_text().strip().splitlines():
            h, f = ln.split()
            assert bm.sha256(HERE / f) == h, f"HASH MISMATCH {f} ({fn}) -> refusing to run"
    assert E.verify_predictors(), "E4b predictors changed"


# ------------------------------------------------------------------ E5a grids (prereg_e5.json is authoritative; built from it)
def grids():
    G = PR5["e5a"]["grids"]
    return {
        "TA": [{"lam": l} for l in G["TA_lam"]],
        "PICO_TA": [{"c": c} for c in G["PICO_c"]],
        "GATE": [{"theta": 30.0, "beta": 1.0, "lam": l} for l in G["TA_lam"]],
        "FORCEGATE": [{"force_k": 1, "beta": 1.0, "lam": l} for l in G["TA_lam"]],
        "TIES": [{"k": G["TIES_k"], "lam": l} for l in G["TIES_lam"]],
        "PICO_GATE": [{"theta": 30.0, "c": c} for c in G["PICO_c"]],
    }


def install_pico_gate(M):
    """PICO_GATE (E5 exploratory-in-family arm): apply the hard theta*=30 gate to task 2 (B2 <- (I - S S^T) B2, dW2 <- (I - S S^T) dW2 on
    FAIL layers; S = merges.gate_shared_basis, the E1b/E3 definition), then run the unmodified Pico calibration + rescaling (PICO_TA) on the
    gated pair. Pairs without a FAIL layer: identical to PICO_TA. Implemented as a wrapper around merges.merge_model (merges.py unmodified)."""
    orig = M.merge_model
    if getattr(orig, "_e5_wrapped", False): return

    def mm(method, hp, names, dW, fac=None, info=None, thr_cache=None, task_keys=None):
        if method != "PICO_GATE":
            return orig(method, hp, names, dW, fac, info=info, thr_cache=thr_cache, task_keys=task_keys)
        dW2, fac2, nfail = {}, {}, 0
        for n in names:
            Sd, _ = M.gate_shared_basis(fac[0][n][0], fac[1][n][0], hp.get("theta", 30.0))
            if Sd is None:
                dW2[n] = dW[1][n]; fac2[n] = fac[1][n]
            else:
                nfail += 1; B2, A2 = fac[1][n]
                dW2[n] = dW[1][n] - Sd @ (Sd.T @ dW[1][n]); fac2[n] = (B2 - Sd @ (Sd.T @ B2), A2)
        inf = info if info is not None else {}
        out = orig("PICO_TA", {"c": hp["c"]}, names, [dW[0], dW2], [fac[0], fac2], info=inf)
        inf["gate_n_fail_layers"] = nfail
        return out
    mm._e5_wrapped = True
    M.merge_model = mm


class DeadlineList(list):
    """pair list whose iteration stops (cleanly, between pairs) once the E5 deadline has passed."""
    def __iter__(self):
        for x in list.__iter__(self):
            if past_deadline():
                log(f"e5 deadline reached before {x}: no further pairs started"); return
            yield x


def make_setting(e3, pairs, name, smoke=False):
    S = e3.Setting(); S.name = name
    S.pairs = DeadlineList(pairs); S.tasks = sorted({k for p in pairs for k in p})
    PO = json.loads((E4B / "populations_e4b.json").read_text())
    S.names = PO["lora_layers"]; S.brev = E.BREV; S.base = E.BASE
    if smoke:
        S.single = {k: {"hold": 1.0, "eval": 1.0} for k in S.tasks}
    else:
        sg = E.singles(); S.single = {k: sg[k] for k in S.tasks}
    aux = json.loads((E4B / "predictors_aux_e4b.json").read_text())
    S.ads, S.data, cache = {}, {}, {}
    for k in S.tasks:
        ad = E.load_ad(k)
        if not smoke: assert ad["sha256"] == aux["adapter_sha256"][k], k
        t = E.task_of(k)
        if t not in cache:
            d = E.get_data(t); cache[t] = {"hold": d["hold"], "eval": d["eval"], "hold_idx_sha256": d["hold_idx_sha256"]}
        assert cache[t]["hold_idx_sha256"] == ad["meta"]["hold_idx_sha256"]
        S.ads[k] = ad; S.data[k] = cache[t]
    S.score = lambda k, lg, y: float(bm.score(E.task_of(k), lg, y)["main"])
    S.compact = lambda k, lg: bm.compact(E.task_of(k), lg)
    S.bs = E.BS
    return S


def e5a_order(pairs):
    perm = np.random.default_rng(PR5["e5a"]["order_seed"]).permutation(len(pairs))
    return [pairs[i] for i in perm]


def st_e5a(a, smoke=False):
    import torch
    sys.path.insert(0, str(E3C))
    import e3
    install_pico_gate(e3.M)
    out = HERE / ("smoke" if smoke else "e5a"); out.mkdir(exist_ok=True)
    e3.OUT = out; e3.LOGF = out / "run.log"
    G = grids()
    e3.GRIDS = G; e3.TA_LAMS = list(PR5["e5a"]["grids"]["TA_lam"])
    e3.PRIMARY = list(PR5["e5a"]["family"]); e3.EXPLORATORY = []
    e3.Enc = E.Enc
    bm.setup_torch()
    if smoke:
        pdir = E4B / "pilot"
        remap = {"sst2@s0": pdir / "lr0.0001" / "sst2", "rte@s1": pdir / "lr0.001" / "rte"}
        E.adapter_dir = lambda k: remap[k]
        pairs = [("sst2@s0", "rte@s1")]
    else:
        PO = json.loads((E4B / "populations_e4b.json").read_text())
        pairs = e5a_order([tuple(p.split("__")) for p in PO["populations"]["P"]])
        (out / "pair_order.json").write_text(json.dumps([f"{x}__{y}" for x, y in pairs], indent=0))
    S = make_setting(e3, pairs, "e5a_smoke" if smoke else "e5a_P", smoke=smoke)
    ns = argparse.Namespace(device=a.device, mem_frac=a.mem_frac, methods=None)
    t0 = time.time()
    e3.run(ns, S)
    if not smoke: timing_add("e5a_s", time.time() - t0)
    log(f"e5a{' SMOKE' if smoke else ''} run finished [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ E5b: TA at the extended lambdas (E4b stage-2 arithmetic, unchanged)
def st_e5b(a, smoke=False):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if a.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    LX = PR5["e5b"]["lambda_ext"]
    PO = json.loads((E4B / "populations_e4b.json").read_text())
    names = PO["lora_layers"]; single = E.singles()
    aux = json.loads((E4B / "predictors_aux_e4b.json").read_text())
    outd = HERE / "e5b"; outd.mkdir(exist_ok=True); pdir = outd / "preds"; pdir.mkdir(exist_ok=True)
    enc = E.Enc(dev, names); ads, data = {}, {}
    for pop in (a.pops or PR5["e5b"]["populations_order"]):
        ref = {}
        for l in (E4B / f"pair_results_{pop}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l); ref.setdefault(r["pair"], r)
        jl = outd / f"pair_ext_{pop}.jsonl"
        done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()} if jl.exists() else set()
        pairs = [tuple(p.split("__")) for p in PO["populations"][pop]]
        if a.limit: pairs = pairs[:a.limit]
        for i, (x, y) in enumerate(pairs):
            key = E.pid(x, y)
            if key in done: continue
            if past_deadline(): log(f"e5b deadline reached before {pop} {key}: stopping"); return
            tp = time.time()
            for k in (x, y):
                if k not in ads:
                    ads[k] = E.load_ad(k); assert ads[k]["sha256"] == aux["adapter_sha256"][k], k
                t = E.task_of(k)
                if t not in data: data[t] = E.get_data(t)
            d1 = {n: bm.delta(ads[x], n, dev) for n in names}; d2 = {n: bm.delta(ads[y], n, dev) for n in names}
            rec = {"pair": key, "t1": x, "t2": y, "population": pop}; preds = {}

            def evaluate(lam, which):
                enc.set_delta({n: lam * (d1[n] + d2[n]) for n in names}); o = {}
                for k in (x, y):
                    t = E.task_of(k); lg = enc.predict(data[t][which], ads[k]["head"], E.BS)
                    o[k] = bm.score(t, lg, data[t][which]["labels"])["main"]
                    if which == "eval": preds[f"TA_{k}_lam{lam}"] = bm.compact(t, lg)
                return o
            nm = lambda sc, w: 0.5 * (sc[x] / single[x][w] + sc[y] / single[y][w])
            lams = list(LX)
            if pop == "P" and i < PR5["e5b"]["sanity_pairs"]: lams = [1.0] + lams     # exact-reproduction sanity vs E4b stage 2
            for lam in lams:
                h = evaluate(lam, "hold"); e = evaluate(lam, "eval")
                if lam == 1.0 and 1.0 not in LX:
                    r0 = ref[key]
                    rec["sanity_lam1.0_maxabs"] = max(abs(h[x] - r0["TA_hold_t1_lam1.0"]), abs(h[y] - r0["TA_hold_t2_lam1.0"]),
                                                      abs(e[x] - r0["TA_eval_t1_lam1.0"]), abs(e[y] - r0["TA_eval_t2_lam1.0"]))
                    preds.pop(f"TA_{x}_lam1.0", None); preds.pop(f"TA_{y}_lam1.0", None)
                    continue
                rec[f"TA_hold_norm_lam{lam}"] = nm(h, "hold"); rec[f"TA_eval_D_lam{lam}"] = 1 - nm(e, "eval")
                for j, k in ((1, x), (2, y)): rec[f"TA_hold_t{j}_lam{lam}"] = h[k]; rec[f"TA_eval_t{j}_lam{lam}"] = e[k]
            rec["secs"] = time.time() - tp
            np.savez_compressed(pdir / f"pair_{key}.npz", **preds)
            with open(jl, "a") as f: f.write(json.dumps(rec) + "\n")
            timing_add(f"e5b_{pop}_s", rec["secs"])
            log(f"e5b {pop} {key}: " + " ".join(f"l{l}: hold {rec[f'TA_hold_norm_lam{l}']:.4f} D {rec[f'TA_eval_D_lam{l}']:.4f}" for l in LX)
                + (f" sanity {rec['sanity_lam1.0_maxabs']:.1e}" if "sanity_lam1.0_maxabs" in rec else "") + f" [{rec['secs']:.0f}s]")
            del d1, d2; torch.cuda.empty_cache()
        log(f"e5b {pop} done")


# ------------------------------------------------------------------ E5d: lemma quantities on Qwen gate layers
def st_e5d(a):
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(HERE)); import verify_lemma as VL    # copy of paper/verify_lemma.py (hash in code_e5.sha256)
    dev = torch.device(a.device)
    PO = json.loads((E4B / "populations_e4b.json").read_text()); names = PO["lora_layers"]
    import pandas as pd
    L = pd.read_csv(E4B / "predictors_layers_e4b.csv")
    fail = L[L.theta_min_A_deg < 30.0]
    outd = HERE / "e5d"; outd.mkdir(exist_ok=True)
    ads = {}
    def ad(k):
        if k not in ads: ads[k] = E.load_ad(k)
        return ads[k]
    rows = []; t0 = time.time()
    popof = {p: k for k, v in PO["populations"].items() for p in v}
    def one(pairkey, n, mode):
        x, y = pairkey.split("__")
        A1, B1 = ad(x)["layers"][n]; A2, B2 = ad(y)["layers"][n]
        B1 = B1.to(dev).double(); B2 = B2.to(dev).double(); A1 = A1.to(dev).double(); A2 = A2.to(dev).double()
        dW1 = ad(x)["scaling"] * B1 @ A1; dW2 = ad(y)["scaling"] * B2 @ A2
        U1 = bm.orth(B1); U2 = bm.orth(B2)
        P, S, _ = torch.linalg.svd(U1.T @ U2, full_matrices=False); S = S.clamp(0, 1)
        idx = (S > math.cos(math.radians(30.0))) if mode == "gate30" else torch.arange(len(S), device=dev) < 1
        U, _ = torch.linalg.qr(U1 @ P[:, idx], mode="reduced")
        st = VL.lemma_stats(dW1, dW2, U)
        X2n2 = st["norm_X2"] ** 2
        cos = st["cos_X1_X2"]
        return {"pair": pairkey, "population": popof.get(pairkey), "layer": n, "mode": mode, "theta_min_deg": float(torch.rad2deg(torch.arccos(S.max()))),
                "k": st["k"], "c": st["c"], "cos_X1_X2": cos, "abs_cos": abs(cos) if cos == cos else float("nan"), "mu_equiv": st["mu_equiv"],
                "eps_rel_residual": st["eps_rel_residual"],
                "frac_edit_energy_not_coef_change": 1.0 - cos ** 2 if cos == cos else float("nan"),     # ||R||^2/||X2||^2 = 1 - cos^2(phi)
                "edit_size_rel_to_merge": st["edit_size_rel_to_merge"], "share_dW2_energy_removed": st["share_dW2_energy_removed"],
                "share_dW1_energy_in_U": st["share_dW1_energy_in_U"], "frac_entries_sign_flipped_by_gate": st["frac_entries_sign_flipped_by_gate"],
                "identity_residual_max": max(st["identity_residuals"]), "regime": st["regime"]}
    for r in fail.itertuples():
        rows.append(one(r.pair, r.layer, "gate30"))
    log(f"e5d gate30 layers done: {len(rows)} [{time.time() - t0:.0f}s]")
    if PR5["e5d"]["forced_all_layers_P"]:
        for p in PO["populations"]["P"]:
            for n in names: rows.append(one(p, n, "forced_k1"))
        log(f"e5d forced_k1 P done [{time.time() - t0:.0f}s]")
    pd.DataFrame(rows).to_csv(outd / "lemma_layers_e5d.csv", index=False, float_format="%.8g")
    timing_add("e5d_s", time.time() - t0)
    log(f"e5d done: {len(rows)} rows [{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--mem-frac", type=float, default=0.9)
    ap.add_argument("--pops", nargs="*", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    log(f"=== e5.py {sys.argv}")
    if a.stage == "smoke":
        st_e5a(a, smoke=True); return
    check_hashes()
    if a.stage == "e5a": st_e5a(a)
    elif a.stage == "e5b": st_e5b(a)
    elif a.stage == "e5d": st_e5d(a)
    else: raise SystemExit("unknown stage")


if __name__ == "__main__":
    main()
