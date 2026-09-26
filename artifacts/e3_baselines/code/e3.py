#!/usr/bin/env python3
"""E3: strong LoRA-merge baselines vs task arithmetic (TA), hard theta*-gate vs Pico soft B-space reweighting.

  --setting e1   PILOT (exploratory): 7 prateeky2806 Hub adapters, 21 pairs; data/singles/adapters exactly as E1
                 (artifacts/e1_predictive, read-only: cache/*.pt, stage0.json; all-zero token_type_ids already in the cache).
  --setting e1b  CONFIRMATORY per PREREG_E3.md: E1b adapters/data/singles (artifacts/e1b_confirmatory, read-only), valid tasks
                 from its stage0.json, tuning on the E1b 1,000 held-out TRAIN examples per task, final on the E1b eval sets.
Stages: --stage run (resumable, one jsonl line per pair) | analyze (tables, stats, report) | all.
Every method: grid of <= 8 configs, selected per pair by the mean normalized held-out score (first max in grid order wins),
then ONE evaluation on the final eval set at the selected config.  normalized score = mean over the two tasks of merged/single.
"""
import argparse, hashlib, itertools, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True           # never write __pycache__ into the read-only E1/E1b directories we import from
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import merges as M  # noqa: E402

TA_LAMS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
E1_LAMS = [0.3, 0.5, 0.7, 1.0]
GRIDS = {
    # ---- primary family (each compared with TA) ----
    "TA": [{"lam": l} for l in TA_LAMS],
    "TIES": [{"k": 10, "lam": l} for l in (1.5, 2.0, 2.5)] + [{"k": 20, "lam": l} for l in (1.0, 1.5, 2.0)]
            + [{"k": 30, "lam": l} for l in (1.0, 1.5)],
    "DARE_TA": [{"p": p, "lam": l, "seed": 0} for p in (0.5, 0.9) for l in (0.5, 0.7, 1.0, 1.3)],
    "DARE_TIES": [{"p": p, "k": 20, "lam": l, "seed": 0} for p in (0.5, 0.9) for l in (0.7, 1.0, 1.5, 2.0)],
    "TSVM": [{"kmode": m, "r": 8, "lam": l} for m in ("lora", "official") for l in (0.5, 0.7, 1.0, 1.3)],
    "KNOTS": [{"topK": K, "lam": l, "mask": "ties", "merging_type": "mean"} for K in (20, 100) for l in (1.0, 1.4, 1.8, 2.2)],
    "PICO_TA": [{"c": c} for c in (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6)],
    "GATE": [{"theta": 30.0, "beta": 1.0, "lam": l} for l in TA_LAMS],
    # ---- exploratory (not in the Holm family) ----
    "PICO_TIES": [{"k": 20, "c": c} for c in (0.75, 1.0, 1.25, 1.5)],
    "SOFTGATE": [{"theta": 30.0, "beta": b, "lam": l} for b in (0.25, 0.5, 0.75) for l in (0.7, 1.0)],
}
GRIDS_V1 = GRIDS                                       # pilot (E1) grids, as registered in PREREG_E3.md
# ---- E1b grids: PREREG_E3_AMENDMENTS.md A2 (wider coefficient ranges incl. 1.5/2.0; forced gate; 8 configs per primary method)
TA2_LAMS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3]
GRIDS_V2 = {
    "TA": [{"lam": l} for l in TA2_LAMS],
    "TIES": [{"k": 10, "lam": l} for l in (1.5, 2.5)] + [{"k": 20, "lam": l} for l in (0.7, 1.0, 1.5, 2.0)]
            + [{"k": 30, "lam": l} for l in (1.0, 1.5)],
    "DARE_TA": [{"p": p, "lam": l, "seed": 0} for p in (0.5, 0.9) for l in (0.5, 0.7, 1.0, 1.4)],
    "DARE_TIES": [{"p": p, "k": 20, "lam": l, "seed": 0} for p in (0.5, 0.9) for l in (0.7, 1.0, 1.5, 2.0)],
    "TSVM": [{"kmode": m, "r": 8, "lam": l} for m in ("lora", "official") for l in (0.5, 0.7, 1.0, 1.5)],
    "KNOTS": [{"topK": K, "lam": l, "mask": "ties", "merging_type": "mean"} for K in (20, 100) for l in (0.7, 1.0, 1.5, 2.0)],
    "PICO_TA": [{"c": c} for c in (0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)],
    "GATE": [{"theta": 30.0, "beta": 1.0, "lam": l} for l in TA2_LAMS],
    "FORCEGATE": [{"force_k": 1, "beta": 1.0, "lam": l} for l in TA2_LAMS],
    "PICO_TIES": [{"k": 20, "c": c} for c in (0.75, 1.0, 1.25, 1.5)],
    "SOFTGATE": [{"force_k": 1, "beta": b, "lam": l} for b in (0.25, 0.5, 0.75) for l in (0.7, 1.0)],
}
PRIMARY_V1 = ["TIES", "DARE_TA", "DARE_TIES", "TSVM", "KNOTS", "PICO_TA", "GATE"]
PRIMARY_V2 = ["TIES", "DARE_TA", "DARE_TIES", "TSVM", "KNOTS", "PICO_TA", "GATE", "FORCEGATE"]
PRIMARY = list(PRIMARY_V1)
EXPLORATORY = ["PICO_TIES", "SOFTGATE"]
MERGE_FN = {"SOFTGATE": "GATE", "FORCEGATE": "GATE"}


def use_grids(setting):
    """e1 -> registered pilot grids (v1); e1b -> amendment-A2 grids (v2) incl. FORCEGATE."""
    global GRIDS, PRIMARY, TA_LAMS
    if setting == "e1b":
        GRIDS, PRIMARY, TA_LAMS = GRIDS_V2, list(PRIMARY_V2), list(TA2_LAMS)
    else:
        GRIDS, PRIMARY, TA_LAMS = GRIDS_V1, list(PRIMARY_V1), [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
SCALE_KEY = lambda m: "c" if m.startswith("PICO") else "lam"

OUT = LOGF = None


def log(msg):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(msg)
    print(s, flush=True)
    if LOGF:
        with open(LOGF, "a") as f:
            f.write(s + "\n")


def jdump(o, p):
    Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def nonscale(method, hp):
    k = SCALE_KEY(method)
    return tuple(sorted((a, b) for a, b in hp.items() if a != k))


# ----------------------------------------------------------------------------- settings
class Setting:
    pass


def setup_e1(args):
    import torch
    S = Setting(); S.name = "e1"
    S.src = Path(args.repo_root) / "artifacts" / "e1_predictive"
    sys.path.insert(0, str(S.src))
    import e1 as base_mod                                   # read-only import (no bytecode written)
    s0 = json.loads((S.src / "stage0.json").read_text())
    S.tasks = list(base_mod.TASKS); S.pairs = list(itertools.combinations(S.tasks, 2))
    S.names = s0["lora_layers"]; S.brev = s0["base_model"]["revision"]; S.base = base_mod.BASE
    S.single = {t: {"hold": s0["singles"][t]["sel"]["accuracy"], "eval": s0["singles"][t]["val"]["accuracy"]} for t in S.tasks}
    S.data = {t: {"hold": torch.load(S.src / "cache" / f"{t}_sel.pt", weights_only=False),
                  "eval": torch.load(S.src / "cache" / f"{t}_val.pt", weights_only=False)} for t in S.tasks}
    S.ads = {t: base_mod.load_adapter(t) for t in S.tasks}
    for t in S.tasks:
        assert S.ads[t]["sha256"] == s0["adapters"][t]["safetensors_sha256"], f"{t}: adapter differs from E1 stage0"
    S.score = lambda t, lg, y: float((lg.argmax(-1) == y).mean())
    S.compact = lambda t, lg: lg.argmax(-1).astype(np.int8)
    S.bs = 128                                              # E1 batch size (exact reproduction of E1 numerics)
    S.eval_name = "full GLUE validation"
    S.predictors = S.src / "predictors.csv"
    S.ref_pairs = S.src / "pair_results.csv"
    return S


def setup_e1b(args):
    import torch
    S = Setting(); S.name = "e1b"
    S.src = Path(args.repo_root) / "artifacts" / "e1b_confirmatory"
    sys.path.insert(0, str(S.src))
    import e1b as bm                                        # read-only import; its cache/log globals are pointed at OUR dir
    bm.OUT = OUT; bm.LOGF = OUT / "e1b_import.log"; bm.ADIR = S.src / "adapters"
    s0 = json.loads((S.src / "stage0.json").read_text())
    if not s0.get("stage0_pass"):
        raise SystemExit("E1b stage0 did not pass -> E3 not run (PREREG_E3 sec. 2)")
    if s0.get("FORCE_VALID_SMOKE_TEST"):
        raise SystemExit("E1b stage0 is a smoke-test artefact")
    S.tasks = list(s0["valid_tasks"]); S.pairs = list(bm.pairs_of(S.tasks))
    S.names = s0["lora_layers"]; S.brev = bm.BREV; S.base = bm.BASE
    S.single = {t: {"hold": s0["per_task"][t]["hold"]["main"], "eval": s0["per_task"][t]["eval"]["main"]} for t in S.tasks}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.base, revision=S.brev)
    S.ads, S.data = {}, {}
    for t in S.tasks:
        ad = bm.load_adapter(t)
        assert ad["sha256"] == s0["per_task"][t]["adapter_sha256"], f"{t}: adapter changed since E1b stage0"
        cp = S.src / "cache" / f"{t}.pt"
        d = torch.load(cp, weights_only=False) if cp.exists() else bm.get_data(t, tok)
        assert d["hold_idx_sha256"] == ad["meta"]["hold_idx_sha256"], f"{t}: held-out indices differ from training meta"
        S.ads[t] = ad; S.data[t] = {"hold": d["hold"], "eval": d["eval"]}
    S.score = lambda t, lg, y: float(bm.score(t, lg, y)["main"])
    S.compact = bm.compact
    S.bs = 256                                              # E1b batch size
    S.eval_name = "E1b eval sets (<=10k seeded subsets)"
    S.predictors = S.src / "predictors.csv"
    S.ref_pairs = S.src / "pair_results.csv"
    bm.setup_torch()                                        # TF32 off, as E1b
    return S


class Enc:
    """BERT encoder with LoRA-covered linear layers overwritten by W0 + delta; per-task classifier heads. Padding/ordering
    identical to E1/E1b (length-sorted, stable argsort, right padding, attention mask); batches cached per data dict."""

    def __init__(self, dev, names, base, brev):
        from transformers import BertModel
        self.dev = dev
        self.m = BertModel.from_pretrained(base, revision=brev).to(dev).eval()
        mods = dict(self.m.named_modules())
        self.lin = {n: mods[n] for n in names}
        self.W0 = {n: l.weight.detach().clone() for n, l in self.lin.items()}
        self._bc = {}

    def set_scaled(self, base, scale):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_(self.W0[n] + scale * base[n])      # == E1: W0 + (lam * (dW1 + dW2)) when base = dW1 + dW2

    def batches(self, data, bs):
        import torch
        key = (id(data), bs)
        if key not in self._bc:
            ids, tt = data["ids"], data["tt"]
            order = np.argsort([len(x) for x in ids], kind="stable"); out = []
            for i in range(0, len(ids), bs):
                idx = order[i:i + bs]; L = max(len(ids[j]) for j in idx)
                ii = np.zeros((len(idx), L), dtype=np.int64); ti = np.zeros_like(ii); am = np.zeros_like(ii)
                for r, j in enumerate(idx):
                    l = len(ids[j]); ii[r, :l] = ids[j]; ti[r, :l] = tt[j]; am[r, :l] = 1
                out.append((idx, torch.from_numpy(ii), torch.from_numpy(ti), torch.from_numpy(am)))
            self._bc[key] = (data, out)
        return self._bc[key][1]

    def predict(self, data, head, bs):
        import torch
        n = len(data["ids"])
        W = head["weight"].to(self.dev); b = head["bias"].to(self.dev)
        out = np.empty((n, W.shape[0]), dtype=np.float32)
        with torch.inference_mode():
            for idx, ii, ti, am in self.batches(data, bs):
                pooled = self.m(input_ids=ii.to(self.dev), token_type_ids=ti.to(self.dev), attention_mask=am.to(self.dev)).pooler_output
                out[idx] = (pooled @ W.T + b).float().cpu().numpy()
        return out


# ----------------------------------------------------------------------------- run
def run(args, S):
    import torch
    dev = torch.device(args.device)
    if dev.type == "cuda" and args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    enc = Enc(dev, S.names, S.base, S.brev)
    jl = OUT / "results.jsonl"
    done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()} if jl.exists() else set()
    pdir = OUT / "preds"; pdir.mkdir(exist_ok=True)
    methods = [m for m in (["TA"] + PRIMARY + EXPLORATORY) if (not args.methods or m in args.methods)]
    thr_cache = {}
    gl = {"grids": {m: GRIDS[m] for m in methods}, "grid_sizes": {m: len(GRIDS[m]) for m in methods}, "setting": S.name,
          "tasks": S.tasks, "n_pairs": len(S.pairs), "bs": S.bs, "mem_frac": args.mem_frac,
          "merges_sha256": sha256(HERE / "merges.py"), "e3_sha256": sha256(Path(__file__))}
    jdump(gl, OUT / "run_config.json")
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for (a, b) in S.pairs:
        key = f"{a}-{b}"
        if key in done:
            continue
        tp = time.time()
        ada, adb = S.ads[a], S.ads[b]
        dW = [{n: ad["scaling"] * (ad["layers"][n][1].to(dev) @ ad["layers"][n][0].to(dev)) for n in S.names} for ad in (ada, adb)]
        fac = [{n: (ad["scaling"] * ad["layers"][n][1].to(dev), ad["layers"][n][0].to(dev)) for n in S.names} for ad in (ada, adb)]
        memo, preds, recs, extra = {}, {}, {}, {}
        tim = {m: {"merge_s": 0.0, "hold_s": 0.0, "eval_s": 0.0, "n_hold_evals": 0, "n_eval_evals": 0} for m in methods}
        infos = {}

        def build(method, hp):
            fn = MERGE_FN.get(method, method)
            h = dict(hp); h[SCALE_KEY(method)] = 1.0
            info = {}
            sync(); t0 = time.time()
            base = M.merge_model(fn, h, S.names, dW, fac, info=info, thr_cache=thr_cache, task_keys=[a, b])
            sync(); tim[method]["merge_s"] += time.time() - t0
            eff = (fn, nonscale(method, hp))
            if fn == "GATE" and not hp.get("force_k") and info.get("gate_n_fail_layers", 1) == 0:
                eff = ("TA", ())                             # gate == TA exactly (bitwise) when no layer FAILs
            return base, eff, info

        def evaluate(method, base, eff, scale, which, tag=None):
            mk = (eff, float(scale), which)
            if mk in memo:
                return memo[mk], True
            enc.set_scaled(base, scale)
            sync(); t0 = time.time(); out = {}
            for t in (a, b):
                lg = enc.predict(S.data[t][which], S.ads[t]["head"], S.bs)
                out[t] = S.score(t, lg, S.data[t][which]["labels"])
                if which == "eval" and tag:
                    preds[f"{tag}__{t}"] = S.compact(t, lg)
            sync(); dt = time.time() - t0
            tim[method]["hold_s" if which == "hold" else "eval_s"] += dt
            tim[method]["n_hold_evals" if which == "hold" else "n_eval_evals"] += 1
            memo[mk] = out
            return out, False

        norm = lambda sc, which: 0.5 * (sc[a] / S.single[a][which] + sc[b] / S.single[b][which])
        for m in methods:
            grid = GRIDS[m]; rows = []
            groups = {}
            for i, hp in enumerate(grid):
                groups.setdefault(nonscale(m, hp), []).append(i)
            res = [None] * len(grid)
            best = None                                            # (idx, base, eff) of the current first-max config
            for g, idxs in groups.items():
                base, eff, info = build(m, grid[idxs[0]])
                infos.setdefault(m, {}).update(info)
                for i in idxs:
                    sc, cached = evaluate(m, base, eff, grid[i][SCALE_KEY(m)], "hold")
                    res[i] = {"hp": grid[i], "hold": sc, "hold_norm": norm(sc, "hold"), "reused": cached}
                jj = int(np.argmax([r["hold_norm"] if r is not None else -np.inf for r in res]))
                if best is None or best[0] != jj:
                    best = (jj, base, eff) if jj in idxs else best
                del base
            j = int(np.argmax([r["hold_norm"] for r in res]))       # first max in grid order
            assert best is not None and best[0] == j
            _, base, eff = best; best = None
            sc, cached = evaluate(m, base, eff, grid[j][SCALE_KEY(m)], "eval", tag=m)
            if m == "TA":                                         # sanity: TA at the E1/E1b grid lambdas on the eval set
                for lam in E1_LAMS:
                    extra[f"TA_eval_lam{lam}"] = evaluate("TA", base, eff, lam, "eval", tag=f"TA_lam{lam}")[0]
                ta4 = [res[TA_LAMS.index(l)]["hold_norm"] for l in E1_LAMS]
                l4 = E1_LAMS[int(np.argmax(ta4))]
                extra["TA4_lam_selected"] = l4; extra["TA4_eval"] = extra[f"TA_eval_lam{l4}"]
                extra["TA4_score"] = norm(extra["TA4_eval"], "eval")
            if m == "PICO_TA":                                    # paper-default Pico (c = 1, no tuning); same base
                pc, _ = evaluate("PICO_TA", base, eff, 1.0, "eval", tag="PICO_TA_c1")
                extra["PICO_TA_c1_eval"] = pc; extra["PICO_TA_c1_score"] = norm(pc, "eval")
            recs[m] = {"grid": res, "sel_idx": j, "sel_hp": grid[j], "sel_hold_norm": res[j]["hold_norm"],
                       "eval": sc, "score": norm(sc, "eval"), "eval_reused": cached, "info": infos.get(m, {}),
                       "at_boundary": j in (0, len(grid) - 1)}
            del base
            torch.cuda.empty_cache() if dev.type == "cuda" else None
        rec = {"pair": key, "t1": a, "t2": b, "single": {a: S.single[a], b: S.single[b]}, "methods": recs, "extra": extra,
               "timing": tim, "secs": time.time() - tp}
        np.savez_compressed(pdir / f"pair_{key}.npz", **preds)
        with open(jl, "a") as f:
            f.write(json.dumps(rec, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
        log(f"{key}: TA={recs['TA']['score']:.4f}(lam {recs['TA']['sel_hp']['lam']}) " +
            " ".join(f"{m}={recs[m]['score']:.4f}" for m in methods if m != "TA") + f" [{rec['secs']:.0f}s]")
        del dW, fac
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("run done")


# ----------------------------------------------------------------------------- analysis
def holm(ps):
    names = list(ps); order = sorted(names, key=lambda k: ps[k]); m = len(order); run_ = 0.0; out = {}
    for i, k in enumerate(order):
        run_ = max(run_, min(1.0, (m - i) * ps[k])); out[k] = run_
    return out


def task_block_boot(gain_by_pair, tasks, nrep=2000, seed=0):
    rng = np.random.default_rng(seed); K = len(tasks); ix = {t: i for i, t in enumerate(tasks)}
    G = np.full((K, K), np.nan)
    for (x, y), g in gain_by_pair.items():
        G[ix[x], ix[y]] = G[ix[y], ix[x]] = g
    means, skipped = [], 0
    for _ in range(nrep):
        s = rng.integers(0, K, K)
        vals = [G[s[i], s[j]] for i in range(K) for j in range(i + 1, K) if s[i] != s[j]]
        distinct = {tuple(sorted((s[i], s[j]))) for i in range(K) for j in range(i + 1, K) if s[i] != s[j]}
        if len(distinct) < 4:
            skipped += 1; continue
        means.append(float(np.nanmean(vals)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], skipped


def signflip(g, nrep=10000, seed=0):
    rng = np.random.default_rng(seed); g = np.asarray(g, float); obs = g.mean()
    fl = rng.choice([-1.0, 1.0], size=(nrep, len(g)))
    perm = (fl * g).mean(1)
    p1 = (1 + np.sum(perm >= obs - 1e-12)) / (nrep + 1)
    p2 = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (nrep + 1)
    return float(p1), float(p2)


def analyze(args, S_name, tasks, pairs):
    import pandas as pd
    recs = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines() if l.strip()]
    order = {f"{x}-{y}": i for i, (x, y) in enumerate(pairs)}
    recs.sort(key=lambda r: order[r["pair"]])
    methods = [m for m in ["TA"] + PRIMARY + EXPLORATORY if m in recs[0]["methods"]]
    rows = []
    for r in recs:
        row = {"pair": r["pair"], "t1": r["t1"], "t2": r["t2"]}
        for m in methods:
            row[m] = r["methods"][m]["score"]
            row[f"{m}__sel"] = json.dumps(r["methods"][m]["sel_hp"])
        row["TA4_E1grid"] = r["extra"].get("TA4_score"); row["PICO_TA_c1"] = r["extra"].get("PICO_TA_c1_score")
        row["gate_n_fail_layers"] = r["methods"].get("GATE", {}).get("info", {}).get("gate_n_fail_layers")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "method_by_pair.csv", index=False, float_format="%.6f")
    pk = [(r["t1"], r["t2"]) for r in recs]
    A = {"setting": S_name, "n_pairs": len(recs), "tasks": tasks, "label": "PILOT (exploratory)" if S_name == "e1" else "E3 confirmatory (PREREG_E3.md)"}
    cols = methods + ["TA4_E1grid", "PICO_TA_c1"]
    per = {}
    for m in cols:
        if m == "TA" or df[m].isna().any():
            continue
        g = (df[m] - df["TA"]).values
        ci, sk = task_block_boot(dict(zip(pk, g)), tasks)
        p1, p2 = signflip(g)
        per[m] = {"mean_score": float(df[m].mean()), "mean_gain_vs_TA": float(g.mean()), "median_gain": float(np.median(g)),
                  "task_block_95ci": ci, "boot_skipped": sk, "signflip_p_one_sided": p1, "signflip_p_two_sided": p2,
                  "wins": int((g > 1e-12).sum()), "ties": int((np.abs(g) <= 1e-12).sum()), "losses": int((g < -1e-12).sum())}
    hp_ = holm({m: per[m]["signflip_p_one_sided"] for m in PRIMARY if m in per})
    for m in hp_:
        per[m]["holm_p"] = hp_[m]
        per[m]["beats_TA_rule"] = bool(per[m]["mean_gain_vs_TA"] >= 0.005 and hp_[m] < 0.05)
    A["TA"] = {"mean_score": float(df["TA"].mean())}
    A["per_method"] = per
    # secondary
    sec = {}
    if "GATE" in df and "PICO_TA" in df:
        d = (df["GATE"] - df["PICO_TA"]).values
        ci, _ = task_block_boot(dict(zip(pk, d)), tasks); p1, p2 = signflip(d)
        sec["gate_minus_pico"] = {"mean": float(d.mean()), "task_block_95ci": ci, "signflip_p_two_sided": p2,
                                  "gate_wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()), "pico_wins": int((d < -1e-12).sum())}
    if "FORCEGATE" in df and "PICO_TA" in df:
        d = (df["FORCEGATE"] - df["PICO_TA"]).values
        ci, _ = task_block_boot(dict(zip(pk, d)), tasks); p1, p2 = signflip(d)
        sec["forcegate_minus_pico"] = {"mean": float(d.mean()), "task_block_95ci": ci, "signflip_p_two_sided": p2,
                                       "forcegate_wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
                                       "pico_wins": int((d < -1e-12).sum())}
    if "GATE" in df:
        fm = df["gate_n_fail_layers"].fillna(0) > 0
        sec["n_pairs_with_fail_layer"] = int(fm.sum())
        if fm.sum() > 0:
            d = (df.loc[fm, "GATE"] - df.loc[fm, "TA"]).values
            p1, p2 = signflip(d) if len(d) > 1 else (float("nan"), float("nan"))
            sec["gate_minus_TA_on_fail_pairs"] = {"n": int(fm.sum()), "mean": float(d.mean()), "values": d.tolist(), "signflip_p_two_sided": p2}
    A["secondary"] = sec
    # selection diagnostics
    A["boundary_rate"] = {m: float(np.mean([r["methods"][m]["at_boundary"] for r in recs])) for m in methods}
    A["selected_configs"] = {m: pd.Series([json.dumps(r["methods"][m]["sel_hp"]) for r in recs]).value_counts().to_dict() for m in methods}
    # timing
    tm = {m: {k: float(sum(r["timing"][m][k] for r in recs)) for k in ("merge_s", "hold_s", "eval_s", "n_hold_evals", "n_eval_evals")} for m in methods}
    A["timing"] = tm; A["total_wall_s"] = float(sum(r["secs"] for r in recs))
    # sanity vs reference TA results
    san = {}
    if S_name == "e1" and Path(args.ref_pairs).exists():
        R = pd.read_csv(args.ref_pairs).set_index("pair"); mism = []
        for r in recs:
            for lam in E1_LAMS:
                for t in (r["t1"], r["t2"]):
                    ref = R.loc[r["pair"], f"val_acc_{t}_lam{lam}"]; ours = r["extra"][f"TA_eval_lam{lam}"][t]
                    if abs(ref - ours) > 1e-9: mism.append((r["pair"], lam, t, ref, ours))
            if R.loc[r["pair"], "lam_selected"] != r["extra"]["TA4_lam_selected"]:
                mism.append((r["pair"], "lam_selected", R.loc[r["pair"], "lam_selected"], r["extra"]["TA4_lam_selected"]))
        san = {"compared": len(recs) * len(E1_LAMS) * 2 + len(recs), "mismatches": mism, "exact": len(mism) == 0}
    elif S_name == "e1b" and Path(args.ref_pairs).exists():
        R = pd.read_csv(args.ref_pairs).set_index("pair"); mism = []
        for r in recs:
            if r["pair"] not in R.index: continue
            for lam in E1_LAMS:
                for t in (r["t1"], r["t2"]):
                    c = f"TA_eval_{t}_lam{lam}"
                    if c in R.columns and abs(R.loc[r["pair"], c] - r["extra"][f"TA_eval_lam{lam}"][t]) > 1e-9:
                        mism.append((r["pair"], lam, t, float(R.loc[r["pair"], c]), r["extra"][f"TA_eval_lam{lam}"][t]))
        san = {"mismatches": mism, "exact": len(mism) == 0}
    A["sanity_TA_reproduction"] = san
    # exploratory: do weight-space predictors forecast method gains?
    try:
        from scipy.stats import spearmanr
        P = pd.read_csv(args.predictors).set_index("pair")
        ex = {}
        for m in per:
            g = (df[m] - df["TA"]).values
            ex[m] = {c: float(spearmanr(P.loc[df.pair, c].values, g).statistic) for c in ("O_A", "O_B", "tv_cosine") if c in P.columns}
        ex["TA_loss_D"] = {c: float(spearmanr(P.loc[df.pair, c].values, 1 - df["TA"].values).statistic) for c in ("O_A", "O_B", "tv_cosine") if c in P.columns}
        A["exploratory_predictor_vs_gain_spearman"] = ex
    except Exception as e:
        A["exploratory_predictor_vs_gain_spearman"] = f"failed: {e}"
    jdump(A, OUT / "analysis.json")
    write_report(A, df, methods, recs)
    return A


def write_report(A, df, methods, recs):
    L = []
    title = "E3 PILOT on E1 Hub adapters (EXPLORATORY; not confirmatory)" if A["setting"] == "e1" else "E3 on E1b adapters (pre-registered, PREREG_E3.md)"
    L.append(f"# {title}\n\nGenerated {time.strftime('%Y-%m-%d %H:%M %Z')}. {A['n_pairs']} pairs; tasks {', '.join(A['tasks'])}.\n")
    L.append("Score = normalized pair score = mean over the two tasks of merged/single on the final eval set, each method at its own "
             "held-out-selected config (grid <= 8). Gain = method - TA (TA grid of 8 lambdas). 95% CI: task-block bootstrap (2000). "
             f"p: pair-level sign-flip permutation (10000), one-sided (gain > 0); Holm over the {len(PRIMARY)} primary methods.\n")
    L.append("| method | grid | mean score | mean gain vs TA (pp) | 95% task-block CI (pp) | W/T/L | p (1-sided) | Holm p | rule met | boundary sel. | GPU s (merge / hold / eval) |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---|---:|---|")
    t = A["timing"]
    L.append(f"| TA | {len(GRIDS['TA'])} | {A['TA']['mean_score']:.4f} | - | - | - | - | - | - | {A['boundary_rate']['TA']:.2f} | {t['TA']['merge_s']:.0f} / {t['TA']['hold_s']:.0f} / {t['TA']['eval_s']:.0f} |")
    for m in [x for x in PRIMARY + EXPLORATORY + ["TA4_E1grid", "PICO_TA_c1"] if x in A["per_method"]]:
        s = A["per_method"][m]; tt = t.get(m)
        tstr = f"{tt['merge_s']:.0f} / {tt['hold_s']:.0f} / {tt['eval_s']:.0f}" if tt else "(reuses TA / PICO runs)"
        gsz = len(GRIDS[m]) if m in GRIDS else (4 if m == "TA4_E1grid" else 1)
        L.append(f"| {m}{' (exploratory)' if m not in PRIMARY else ''} | {gsz} | {s['mean_score']:.4f} | {100 * s['mean_gain_vs_TA']:+.2f} | "
                 f"[{100 * s['task_block_95ci'][0]:+.2f}, {100 * s['task_block_95ci'][1]:+.2f}] | {s['wins']}/{s['ties']}/{s['losses']} | "
                 f"{s['signflip_p_one_sided']:.4f} | {s.get('holm_p', float('nan')):.4f} | {s.get('beats_TA_rule', '-')} | "
                 f"{A['boundary_rate'].get(m, float('nan')):.2f} | {tstr} |")
    L.append("\n## Secondary\n")
    L.append("```\n" + json.dumps(A["secondary"], indent=1) + "\n```\n")
    L.append("## Sanity: TA reproduction of the reference pipeline\n")
    san = A["sanity_TA_reproduction"]
    L.append(f"exact = {san.get('exact')}, mismatches = {san.get('mismatches')}\n")
    L.append("## Method x pair (normalized score)\n")
    cols = methods + ["TA4_E1grid", "PICO_TA_c1", "gate_n_fail_layers"]
    L.append("| pair | " + " | ".join(cols) + " |"); L.append("|---|" + "---:|" * len(cols))
    for _, r in df.iterrows():
        L.append(f"| {r.pair} | " + " | ".join((f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    L.append("\n## Selected configs (counts over pairs)\n")
    for m, v in A["selected_configs"].items():
        L.append(f"- {m}: " + "; ".join(f"{k} x{c}" for k, c in v.items()))
    L.append("\n## Exploratory: Spearman(weight-space predictor, gain vs TA) and (predictor, TA loss 1-score)\n")
    L.append("```\n" + json.dumps(A.get("exploratory_predictor_vs_gain_spearman"), indent=1) + "\n```")
    L.append(f"\nTotal wall for the run stage: {A['total_wall_s'] / 60:.1f} min.")
    fn = "PILOT_E1_REPORT.md" if A["setting"] == "e1" else "E3_E1B_REPORT.md"
    (OUT / fn).write_text("\n".join(L) + "\n")


def main():
    global OUT, LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["e1", "e1b"], required=True)
    ap.add_argument("--stage", default="all", choices=["run", "analyze", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=str(Path.home() / "Desktop/Workspace/LoRA"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.22)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    OUT = Path(args.out).resolve(); OUT.mkdir(parents=True, exist_ok=True); LOGF = OUT / "run.log"
    src = Path(args.repo_root) / "artifacts" / ("e1_predictive" if args.setting == "e1" else "e1b_confirmatory")
    assert "e3" in str(OUT) and not str(OUT).startswith(str(src)), "E3 must never write into the E1/E1b directories"
    args.predictors = str(src / "predictors.csv"); args.ref_pairs = str(src / "pair_results.csv")
    log(f"=== e3.py {sys.argv}")
    use_grids(args.setting)
    if args.setting == "e1b":
        pre = HERE / "PREREG_E3.md"; hs = HERE / "PREREG_E3.sha256"
        assert pre.exists() and hs.exists(), "PREREG_E3.md + sha256 required before any E1b run"
        assert hs.read_text().split()[0] == sha256(pre), "PREREG_E3.md changed after hashing"
        am, ah = HERE / "PREREG_E3_AMENDMENTS.md", HERE / "PREREG_E3_AMENDMENTS.sha256"
        assert am.exists() and ah.exists() and ah.read_text().split()[0] == sha256(am), "amendments file missing/changed"
    if args.stage in ("run", "all"):
        S = setup_e1(args) if args.setting == "e1" else setup_e1b(args)
        if args.max_pairs:
            S.pairs = S.pairs[:args.max_pairs]
        jdump({"tasks": S.tasks, "pairs": [f"{a}-{b}" for a, b in S.pairs]}, OUT / "pairs.json")
        run(args, S)
    if args.stage in ("analyze", "all"):
        pj = json.loads((OUT / "pairs.json").read_text())
        pairs = [tuple(p.split("-", 1)) for p in pj["pairs"]]
        A = analyze(args, args.setting, pj["tasks"], pairs)
        log("analysis: " + json.dumps({m: round(v["mean_gain_vs_TA"] * 100, 3) for m, v in A["per_method"].items()}))


if __name__ == "__main__":
    main()
