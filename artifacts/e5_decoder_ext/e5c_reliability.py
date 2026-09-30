#!/usr/bin/env python3
"""E5c (pre-registered, PREREG_E5.md sec. 4c): evaluation-sampling reliability of merge loss D for E4a (RoBERTa) and E4b (Qwen), CPU only.
Uses the saved per-example eval predictions (preds/pair_*.npz, preds/single_*.npz) and the ORIGINAL held-out-selected TA lambda (never
reselected). Metric functions are imported unmodified from analysis/noise_ceiling.py (E1b/E1c reliability analysis) for comparability.
 (i)  bootstrap (R=1000, seed 20260927): resample eval examples within each task (indices shared by single/merged predictions and by all
      pairs containing the task); per-pair SE of D; independent-replicate rank correlation r and Spearman-Brown 2r/(1+r) (= E1b/E1c method)
 (ii) true split-half (S=200 random disjoint halves per task, seed 20260928): D on each half; Spearman over pairs; Spearman-Brown
 (iii) eval-sampling distribution of rho(O_A, D*) and rho(tv_cosine, D*) over the R bootstrap replicates (SD, 2.5/97.5 pct, P(rho>=0.4))
 (iv) sampling reliability ratio 1 - mean(SE^2)/Var(D) and sampling attenuation ceiling sqrt(SB)
Usage: e5c_reliability.py <root with e4a/ and e4b/ (pair_results_*.jsonl, predictors_*.csv, preds/)> <out dir>"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/workspace/lora-paper/analysis")
from noise_ceiling import bootstrap_scores, fast_rho, sb, metric   # noqa: E402
R, SEED, S_SPLIT, SEED_SPLIT = 1000, 20260927, 200, 20260928


def load_jsonl(p):
    out, seen = [], set()
    for l in Path(p).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["pair"] not in seen: seen.add(r["pair"]); out.append(r)
    return out


def tk(k): return k.split("@")[0]


def run_pop(root, exp, pop, rng, rngs):
    recs = load_jsonl(root / exp / f"pair_results_{pop}.jsonl")
    PR = pd.read_csv(root / exp / f"predictors_{exp}.csv").set_index("pair")
    pdir = root / exp / "preds"
    keys = sorted({k for r in recs for k in (r["t1"], r["t2"])})
    single = {}
    for k in keys:
        z = np.load(pdir / f"single_{k}.npz"); single[k] = (np.asarray(z["eval"]), np.asarray(z["eval_labels"]))
    merged = []
    for r in recs:
        z = np.load(pdir / f"pair_{r['pair']}.npz"); lam = r["lam_selected"]
        merged.append({k: np.asarray(z[f"TA_{k}_lam{lam}"]) for k in (r["t1"], r["t2"])})
    # exact recomputation of D from the saved predictions
    Drec = np.array([1 - 0.5 * sum(metric(tk(k), merged[j][k], single[k][1], np.arange(len(single[k][1]))) /
                                   metric(tk(k), single[k][0], single[k][1], np.arange(len(single[k][1]))) for k in (r["t1"], r["t2"]))
                     for j, r in enumerate(recs)])
    Dorig = np.array([r["D"] for r in recs])
    # (i) bootstrap, two independent replicate sets a/b
    out = {}
    for tag in ("a", "b"):
        idx = {k: rng.integers(0, len(single[k][1]), size=(R, len(single[k][1])), dtype=np.int32) for k in keys}
        ss = {k: bootstrap_scores(tk(k), single[k][0], single[k][1], idx[k]) for k in keys}
        D = np.empty((R, len(recs)))
        for j, r in enumerate(recs):
            v = [bootstrap_scores(tk(k), merged[j][k], single[k][1], idx[k]) / ss[k] for k in (r["t1"], r["t2"])]
            D[:, j] = 1 - 0.5 * (v[0] + v[1])
        out[tag] = D
    Da, Db = out["a"], out["b"]
    se = Da.std(0, ddof=1)
    rs = np.array([fast_rho(Da[i], Db[i]) for i in range(R)])
    # (ii) true split-half
    sh = []
    for _ in range(S_SPLIT):
        halves = {k: rngs.permutation(len(single[k][1])) for k in keys}
        Dh = np.empty((2, len(recs)))
        for h in (0, 1):
            ix = {k: np.sort(halves[k][h::2]) for k in keys}
            sc = {k: metric(tk(k), single[k][0], single[k][1], ix[k]) for k in keys}
            for j, r in enumerate(recs):
                Dh[h, j] = 1 - 0.5 * sum(metric(tk(k), merged[j][k], single[k][1], ix[k]) / sc[k] for k in (r["t1"], r["t2"]))
        sh.append(fast_rho(Dh[0], Dh[1]))
    sh = np.array(sh)
    # (iii) rho(predictor, D*) over bootstrap replicates
    rho_dist = {}
    for c in ("O_A", "tv_cosine"):
        x = PR.loc[[r["pair"] for r in recs], c].values.astype(float)
        rb = np.array([fast_rho(x, Da[i]) for i in range(R)])
        rho_dist[c] = {"rho_observed": fast_rho(x, Dorig), "boot_mean": float(rb.mean()), "boot_sd": float(rb.std(ddof=1)),
                       "boot_p2.5": float(np.percentile(rb, 2.5)), "boot_p97.5": float(np.percentile(rb, 97.5)), "P_rho_ge_0.4": float((rb >= 0.4).mean())}
    varD = float(Dorig.var(ddof=1))
    res = {"experiment": exp, "population": pop, "n_pairs": len(recs), "D_recompute_max_abs_diff": float(np.abs(Drec - Dorig).max()),
           "D_sd_between_pairs": math.sqrt(varD), "mean_pair_se": float(se.mean()), "median_pair_se": float(np.median(se)),
           "p95_pair_se": float(np.percentile(se, 95)), "max_pair_se": float(se.max()),
           "sampling_reliability_1_minus_meanSE2_over_varD": float(1 - np.mean(se ** 2) / varD),
           "indep_replicate_r_mean": float(np.nanmean(rs)), "indep_replicate_SB": sb(float(np.nanmean(rs))),
           "split_half_r_mean": float(np.nanmean(sh)), "split_half_r_p2.5": float(np.nanpercentile(sh, 2.5)), "split_half_r_p97.5": float(np.nanpercentile(sh, 97.5)),
           "split_half_SB": sb(float(np.nanmean(sh))), "sampling_attenuation_ceiling_sqrt_SB": math.sqrt(max(0.0, sb(float(np.nanmean(sh))))),
           "rho_under_eval_resampling": rho_dist}
    per_pair = pd.DataFrame({"experiment": exp, "population": pop, "pair": [r["pair"] for r in recs], "D": Dorig, "D_boot_se": se,
                             "D_boot_lo": np.percentile(Da, 2.5, 0), "D_boot_hi": np.percentile(Da, 97.5, 0)})
    return res, per_pair


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED); rngs = np.random.default_rng(SEED_SPLIT)
    allres, pp = [], []
    for exp in ("e4a", "e4b"):
        for pop in ("P", "R0", "S1"):
            t0 = time.time(); r, p = run_pop(root, exp, pop, rng, rngs); allres.append(r); pp.append(p)
            print(f"{exp} {pop}: n={r['n_pairs']} recompute {r['D_recompute_max_abs_diff']:.1e} SE mean {r['mean_pair_se']:.4f} split-half r {r['split_half_r_mean']:.3f} SB {r['split_half_SB']:.3f} "
                  f"rho(O_A) sd {r['rho_under_eval_resampling']['O_A']['boot_sd']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    (out / "e5c_reliability.json").write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "R": R, "S_split": S_SPLIT,
                                                          "seeds": [SEED, SEED_SPLIT], "results": allres}, indent=1))
    pd.concat(pp).to_csv(out / "e5c_pair_bootstrap.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
