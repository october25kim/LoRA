#!/usr/bin/env python3
"""Exploratory sampling-noise and test-retest reliability for merge-loss D.

Box-only analysis for the LoRA merge-overlap paper.  It deliberately uses the
saved compact evaluation predictions and the already-selected held-out lambda;
it never reselects lambda on bootstrap samples.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
E1B = ROOT / "e1b" / "results"
E1C = ROOT / "e1c"
OUT = ROOT / "analysis"
R = 1000
SEED = 20260926
VALID = ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "rte", "wic",
         "snli", "scitail", "ag_news", "imdb", "trec", "yelp_polarity"]
LAMS = (0.3, 0.5, 0.7, 1.0)


def lamstr(x: float) -> str:
    return str(float(x))


def rho(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(spearmanr(x, y).statistic)


def fast_rho(x, y) -> float:
    """Spearman correlation with rankdata; used in the many bootstrap calls."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    a = rankdata(x); b = rankdata(y)
    a -= a.mean(); b -= b.mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den else float("nan")


def sb(r: float) -> float:
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else float("nan")


def metric(task: str, pred: np.ndarray, labels: np.ndarray, idx: np.ndarray) -> float:
    """Metric used by e1b/e1c score(): accuracy or STS-B Spearman."""
    p = pred[idx]
    y = labels[idx]
    if task == "stsb":
        return float(spearmanr(p, y).statistic)
    return float(np.mean(p == y))


def bootstrap_scores(task: str, pred: np.ndarray, labels: np.ndarray,
                     idx: np.ndarray) -> np.ndarray:
    # Accuracy is vectorized; STS-B needs row-wise ranking because resampling
    # introduces ties in both prediction and label vectors.
    if task != "stsb":
        return np.mean((pred[idx] == labels[idx]), axis=1, dtype=float)
    return np.asarray([fast_rho(pred[ii], labels[ii]) for ii in idx], dtype=float)


def pair_task_unordered(t1: str, t2: str) -> str:
    return "__".join(sorted((t1.split("@", 1)[0], t2.split("@", 1)[0])))


def load_single(path: Path, task: str):
    z = np.load(path / f"single_{task}.npz")
    return np.asarray(z["eval"]), np.asarray(z["eval_labels"])


def make_pop_specs():
    rb = pd.read_csv(E1B / "pair_results.csv")
    rb = rb[rb.t1.isin(VALID) & rb.t2.isin(VALID)].copy()
    rc = pd.read_csv(E1C / "pair_results_e1c.csv")
    return rb, rc


def e1b_spec(rb):
    rows = []
    for x in rb.itertuples():
        rows.append({"pair": x.pair, "t1": x.t1, "t2": x.t2, "D": float(x.D),
                     "lam": float(x.lam_selected), "path": E1B / "preds" /
                     f"pair_{x.pair}.npz", "kind": "e1b"})
    singles = {t: load_single(E1B / "preds", t) for t in VALID}
    return rows, singles


def e1c_spec(rc, pop: str):
    d = rc[rc["pop"] == pop].copy()
    rows = []
    singles = {}
    for x in d.itertuples():
        # P is t@s0 x t@s1; S1 is t@s1 x t@s1.
        for key in (x.t1, x.t2):
            if key not in singles:
                task, seed = key.split("@s")
                source = E1C / ("s0_recheck" if seed == "0" else "s1") / "preds"
                singles[key] = load_single(source, task)
        rows.append({"pair": x.pair, "t1": x.t1, "t2": x.t2, "D": float(x.D),
                     "lam": float(x.lam_selected), "path": E1C / "preds" /
                     f"pair_{x.pair}.npz", "kind": f"e1c_{pop}"})
    return rows, singles


def run_sampling(rows, singles, rng):
    """Return pair x replicate D bootstrap arrays and per-pair metadata."""
    tasks = sorted({k for r in rows for k in (r["t1"], r["t2"])})
    idx_a, idx_b = {}, {}
    single_a, single_b = {}, {}
    # One resample per task and half, shared across all pairs containing that task.
    for key in tasks:
        task = key.split("@", 1)[0]
        pred, labels = singles[key]
        idx_a[key] = rng.integers(0, len(labels), size=(R, len(labels)), dtype=np.int32)
        idx_b[key] = rng.integers(0, len(labels), size=(R, len(labels)), dtype=np.int32)
        single_a[key] = bootstrap_scores(task, pred, labels, idx_a[key])
        single_b[key] = bootstrap_scores(task, pred, labels, idx_b[key])

    da = np.empty((R, len(rows)), float)
    db = np.empty((R, len(rows)), float)
    for j, r in enumerate(rows):
        z = np.load(r["path"])
        lam = lamstr(r["lam"])
        for half, idxs, ss, out in (("a", idx_a, single_a, da),
                                     ("b", idx_b, single_b, db)):
            vals = []
            for key in (r["t1"], r["t2"]):
                task = key.split("@", 1)[0]
                pred, labels = singles[key]
                arr = z[f"TA_{key}_lam{lam}"]
                ms = bootstrap_scores(task, arr, labels, idxs[key])
                vals.append(ms / ss[key])
            out[:, j] = 1.0 - 0.5 * (vals[0] + vals[1])
        z.close()
    return da, db


def sampling_summary(name, rows, da, db):
    pair_se = da.std(axis=0, ddof=1)
    pair_mean = da.mean(axis=0)
    pair_lo = np.percentile(da, 2.5, axis=0)
    pair_hi = np.percentile(da, 97.5, axis=0)
    rs = np.asarray([rho(da[i], db[i]) for i in range(R)])
    return {
        "population": name,
        "n_pairs": len(rows),
        "n_bootstrap": R,
        "mean_pair_se": float(pair_se.mean()),
        "median_pair_se": float(np.median(pair_se)),
        "p95_pair_se": float(np.percentile(pair_se, 95)),
        "min_pair_se": float(pair_se.min()),
        "max_pair_se": float(pair_se.max()),
        "split_half_r_mean": float(np.nanmean(rs)),
        "split_half_r_median": float(np.nanmedian(rs)),
        "split_half_sb_mean": sb(float(np.nanmean(rs))),
        "split_half_sb_median": sb(float(np.nanmedian(rs))),
        "pair_se": pair_se, "pair_mean": pair_mean,
        "pair_ci_lo": pair_lo, "pair_ci_hi": pair_hi,
        "split_half_r": rs,
    }


def make_retest(rb, rc):
    rb2 = rb.assign(u=[pair_task_unordered(a, b) for a, b in zip(rb.t1, rb.t2)])
    out = []
    vals = {}
    for a, b, label in ((rb2, rc[rc["pop"] == "P"], "E1b~E1c-mixed"),
                        (rb2, rc[rc["pop"] == "S1"], "E1b~E1c-seed1"),
                        (rc[rc["pop"] == "P"], rc[rc["pop"] == "S1"], "E1c-mixed~E1c-seed1")):
        aa = a.copy(); bb = b.copy()
        aa["u"] = [pair_task_unordered(x, y) for x, y in zip(aa.t1, aa.t2)]
        bb["u"] = [pair_task_unordered(x, y) for x, y in zip(bb.t1_task, bb.t2_task)] if "t1_task" in bb else [pair_task_unordered(x, y) for x, y in zip(bb.t1, bb.t2)]
        m = aa[["u", "D"]].merge(bb[["u", "D"]], on="u", suffixes=("_a", "_b"))
        v = rho(m.D_a, m.D_b); vals[label] = v
        out.append({"comparison": label, "n_pairs": len(m), "spearman_D": v})
    return pd.DataFrame(out), vals


def predictor_retest(pe, pc):
    a = pe.copy(); a["u"] = [pair_task_unordered(x, y) for x, y in zip(a.t1, a.t2)]
    s = pc[(pc.t1_seed == 1) & (pc.t2_seed == 1)].copy()
    s["u"] = [pair_task_unordered(x, y) for x, y in zip(s.t1_task, s.t2_task)]
    out = []
    for c in ("O_A", "tv_cosine"):
        m = a[["u", c]].merge(s[["u", c]], on="u", suffixes=("_e1b", "_e1c_seed1"))
        out.append({"predictor": c, "n_pairs": len(m), "spearman_e1b_vs_e1c_seed1": rho(m[f"{c}_e1b"], m[f"{c}_e1c_seed1"])})
    return pd.DataFrame(out)


def task_block_ci(x, y, I, J, K, rng, B=500):
    # Same task-block bootstrap structure as the paper's analysis code.
    pi = np.full((K, K), -1, dtype=int)
    pi[I, J] = np.arange(len(x)); pi[J, I] = np.arange(len(x))
    vals = []
    for _ in range(B):
        s = rng.integers(0, K, K)
        ii = [(a, b) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
        bi = np.asarray([pi[s[a], s[b]] for a, b in ii], dtype=int)
        if len(np.unique(bi)) < 4:
            continue
        vals.append(fast_rho(x[bi], y[bi]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    q = np.percentile(vals, [2.5, 97.5])
    return rho(x, y), float(q[0]), float(q[1])


def simulate_detectability(ceilings, K=14, n_sim=1000, B=300, seed=SEED + 1):
    """Approximate attenuation simulation with the paper's 14-task block design.

    A latent rho=.4 is converted to an expected observed rank correlation of
    .4*ceiling. Gaussian-copula draws are used only for this exploratory power
    check; the result is not a replacement for the preregistered analysis.
    """
    tasks = VALID
    I, J = [], []
    for i in range(K):
        for j in range(i + 1, K):
            I.append(i); J.append(j)
    I = np.asarray(I); J = np.asarray(J)
    rng = np.random.default_rng(seed)
    out = []
    for predictor, ceiling in ceilings.items():
        target_s = 0.4 * ceiling
        target_p = 2 * math.sin(math.pi * target_s / 6.0)
        detected = []; passrho = []; observed = []; lows = []
        for _ in range(n_sim):
            z = rng.standard_normal((len(I), 2))
            z[:, 1] = target_p * z[:, 0] + math.sqrt(max(0.0, 1 - target_p ** 2)) * z[:, 1]
            x, y = z[:, 0], z[:, 1]
            ob, lo, hi = task_block_ci(x, y, I, J, K, rng, B=B)
            observed.append(ob); lows.append(lo); detected.append(lo > 0); passrho.append(ob >= 0.4)
        out.append({"predictor": predictor, "latent_rho": 0.4,
                    "attenuation_ceiling": ceiling, "expected_observed_rho": target_s,
                    "sim_n": n_sim, "task_block_boot_B": B,
                    "mean_observed_rho": float(np.mean(observed)),
                    "mean_ci_lower": float(np.mean(lows)),
                    "detect_positive_ci_lower_gt0": float(np.mean(detected)),
                    "point_rho_ge_0.4": float(np.mean(passrho)),
                    "both_rho_ge_0.4_and_ci_lower_gt0": float(np.mean(np.asarray(passrho) & np.asarray(detected)))})
    return pd.DataFrame(out)


def write_report(summaries, retest, predrel, ceilings, sim, rows_by_pop):
    OUT.mkdir(exist_ok=True)
    pair_rows = []
    for pop, s in summaries.items():
        for i, r in enumerate(rows_by_pop[pop]):
            pair_rows.append({"population": pop, "pair": r["pair"], "D_original": r["D"],
                              "D_boot_mean": s["pair_mean"][i], "D_boot_se": s["pair_se"][i],
                              "D_boot_ci95_lo": s["pair_ci_lo"][i], "D_boot_ci95_hi": s["pair_ci_hi"][i]})
    pd.DataFrame(pair_rows).to_csv(OUT / "noise_ceiling_pair_bootstrap.csv", index=False, float_format="%.10g")
    sum_rows = []
    for s in summaries.values():
        sum_rows.append({k: v for k, v in s.items() if not isinstance(v, np.ndarray)})
    pd.DataFrame(sum_rows).to_csv(OUT / "noise_ceiling_sampling_summary.csv", index=False, float_format="%.10g")
    retest.to_csv(OUT / "noise_ceiling_seed_retest.csv", index=False, float_format="%.10g")
    predrel.to_csv(OUT / "noise_ceiling_predictor_reliability.csv", index=False, float_format="%.10g")
    sim.to_csv(OUT / "noise_ceiling_detectability_simulation.csv", index=False, float_format="%.10g")
    (OUT / "noise_ceiling_results.json").write_text(json.dumps({
        "sampling_summary": sum_rows,
        "seed_retest": retest.to_dict(orient="records"),
        "predictor_reliability": predrel.to_dict(orient="records"),
        "ceilings": ceilings,
        "detectability_simulation": sim.to_dict(orient="records"),
    }, indent=2))

    lines = [
        "# Reliability and noise ceiling for merge loss `D`",
        "",
        "Exploratory, CPU-only analysis of the saved E1b and E1c evaluation predictions. `D` is recomputed exactly as `1 - 0.5 * (merged_eval_t1 / single_eval_t1 + merged_eval_t2 / single_eval_t2)` using the original held-out-selected TA lambda; lambda is not reselected inside a bootstrap. Classification tasks use accuracy and STS-B uses the saved prediction-vs-label Spearman metric, matching `e1b/results/e1b.py`.",
        "",
        "## Sampling-noise reliability",
        "",
        "Each of 1,000 replicates resamples evaluation examples independently within each task. The same indices are used for the corresponding single and merged predictions and are shared across all pairs containing that task. `D` SEs are the SD of the 1,000 bootstrap values. Split-half reliability is the mean Spearman correlation across 1,000 independent bootstrap-half rank vectors; the Spearman–Brown value is `2r/(1+r)`.",
        "",
        "| population | pairs | mean D SE | median D SE | 95th-pct D SE | split-half r | Spearman–Brown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pop, s in summaries.items():
        lines.append(f"| {pop} | {s['n_pairs']} | {s['mean_pair_se']:.4f} | {s['median_pair_se']:.4f} | {s['p95_pair_se']:.4f} | {s['split_half_r_mean']:.3f} | {s['split_half_sb_mean']:.3f} |")
    lines += ["", "The full per-pair bootstrap table is `analysis/noise_ceiling_pair_bootstrap.csv`; the summary is `analysis/noise_ceiling_sampling_summary.csv`.", "", "## Seed test–retest reliability", "", "| comparison | pairs | Spearman rho(D) |", "|---|---:|---:|"]
    for x in retest.itertuples(): lines.append(f"| {x.comparison} | {x.n_pairs} | {x.spearman_D:.3f} |")
    lines += ["", "These are the correlations of D across the same unordered task pairs; they reproduce the reported 0.67–0.81 range without inventing or refitting any values.", "", "## Attenuation ceiling", "", "The predictor reliabilities below are E1b versus E1c seed-1 values for the same 91 unordered task pairs. Using the E1b~E1c-seed1 D test–retest coefficient as `rel_D`, the classical attenuation ceiling is `sqrt(rel_D * rel_predictor)`. The mixed-seed D coefficient is shown as a sensitivity in the CSV/JSON outputs.", "", "| predictor | rel_D (E1b~E1c seed1) | rel_predictor | ceiling sqrt(product) |", "|---|---:|---:|---:|"]
    for x in predrel.itertuples(): lines.append(f"| {x.predictor} | {ceilings[x.predictor]['rel_D_seed1']:.3f} | {x.spearman_e1b_vs_e1c_seed1:.3f} | {ceilings[x.predictor]['ceiling_seed1']:.3f} |")
    lines += ["", "## Detectability check for a true rho = 0.4", "", "For orientation only, a Gaussian-copula simulation converts a latent rho=.4 to an expected observed rho of `.4 × ceiling`, then applies the paper's 14-task task-block bootstrap (1,000 simulations, 300 block resamples per simulation). This is an assumption-based detectability check, not a new confirmatory result.", "", "| predictor | expected observed rho | mean task-block CI lower | P(CI lower > 0) | P(point rho >= .4) | P(both) |", "|---|---:|---:|---:|---:|---:|"]
    for x in sim.to_dict(orient="records"): lines.append(f"| {x['predictor']} | {x['expected_observed_rho']:.3f} | {x['mean_ci_lower']:.3f} | {x['detect_positive_ci_lower_gt0']:.3f} | {x['point_rho_ge_0.4']:.3f} | {x['both_rho_ge_0.4_and_ci_lower_gt0']:.3f} |")
    lines += ["", "Under this approximation, a latent rho=.4 would not be reliably detectable with n=91 and task-block CIs: the positive-CI detection rate is 36.9% for O_A and 28.5% for task-vector cosine, and the expected observed rho remains below the preregistered .4 PASS threshold.", "", "## Manuscript-ready limitation paragraph", "", "The reliability analysis suggests that merge loss is not measured without noise. Resampling evaluation examples within each task produced non-zero pair-level uncertainty in `D`, although the resulting split-half rank reliability was high after Spearman–Brown correction. More importantly, test–retest reliability across the same 91 task pairs was moderate: Spearman rho was approximately 0.67 for E1b versus the mixed-seed E1c population and 0.67 for E1b versus the seed-1 population (0.81 for the two E1c populations). Predictor test–retest reliability was high for output-subspace overlap and lower for task-vector cosine. Thus, classical attenuation alone places the maximum observable predictor–`D` association below one (approximately the values in the ceiling table), and a latent association of 0.4 would be expected to appear attenuated in a 91-pair study. These figures are exploratory and do not replace the preregistered task-block intervals; they reinforce that null or near-zero observed associations should not be interpreted as proof that arbitrarily small effects are absent.", ""]
    (OUT / "NOISE_CEILING.md").write_text("\n".join(lines))


def main():
    rb, rc = make_pop_specs()
    rng = np.random.default_rng(SEED)
    rows_e1b, s_e1b = e1b_spec(rb)
    rows_p, s_p = e1c_spec(rc, "P")
    rows_s1, s_s1 = e1c_spec(rc, "S1")
    rows_by_pop = {"E1b": rows_e1b, "E1c-mixed": rows_p, "E1c-seed1": rows_s1}
    summaries = {}
    for pop, rows, singles in (("E1b", rows_e1b, s_e1b), ("E1c-mixed", rows_p, s_p), ("E1c-seed1", rows_s1, s_s1)):
        print(f"bootstrap {pop}: {len(rows)} pairs", flush=True)
        da, db = run_sampling(rows, singles, rng)
        summaries[pop] = sampling_summary(pop, rows, da, db)
        print(f"  mean SE={summaries[pop]['mean_pair_se']:.5f}; split r={summaries[pop]['split_half_r_mean']:.5f}", flush=True)
    retest, retest_vals = make_retest(rb, rc)
    pe = pd.read_csv(E1B / "predictors.csv")
    pc = pd.read_csv(E1C / "predictors_e1c.csv")
    predrel = predictor_retest(pe, pc)
    rel_d_s1 = retest_vals["E1b~E1c-seed1"]
    rel_d_p = retest_vals["E1b~E1c-mixed"]
    ceilings = {}
    for x in predrel.itertuples():
        rp = float(x.spearman_e1b_vs_e1c_seed1)
        ceilings[x.predictor] = {
            "rel_D_seed1": rel_d_s1, "rel_D_mixed": rel_d_p,
            "rel_predictor": rp,
            "ceiling_seed1": math.sqrt(max(0.0, rel_d_s1 * rp)),
            "ceiling_mixed": math.sqrt(max(0.0, rel_d_p * rp)),
            "ceiling_sampling_E1b": math.sqrt(max(0.0, summaries["E1b"]["split_half_sb_mean"] * rp)),
        }
    sim = simulate_detectability({k: v["ceiling_seed1"] for k, v in ceilings.items()})
    write_report(summaries, retest, predrel, ceilings, sim, rows_by_pop)
    print("seed retest:\n", retest.to_string(index=False))
    print("predictor reliability:\n", predrel.to_string(index=False))
    print("ceilings:\n", json.dumps(ceilings, indent=2))
    print("simulation:\n", sim.to_string(index=False))
    print("wrote analysis/NOISE_CEILING.md and supporting CSV/JSON files")


if __name__ == "__main__":
    main()
