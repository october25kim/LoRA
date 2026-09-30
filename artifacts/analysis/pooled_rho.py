#!/usr/bin/env python3
"""EXPLORATORY pooled summary of predictor-vs-merge-loss Spearman rho across three pair populations that share the same
14 tasks:  R0 = E1b (both adapters seed 0), P = E1c primary (mixed seed: alphabetically-first task seed 0, other task seed 1),
S1 = E1c secondary (both adapters seed 1).

NOT pre-registered. Written 2026-09-26 (KST) after all E1b and E1c results were known. It changes no verdict.

Estimand: rho_bar = mean over the three populations of Spearman rho(predictor, D) (equal weights).
Uncertainty: a task-block bootstrap that resamples the 14 TASKS JOINTLY across populations (one draw of task indices per
replicate, applied to all three populations; construction copied from e1b.stage3 / e1c_analysis.pop_stats: 2,000 reps,
numpy default_rng(0), keep pairs of distinct resampled indices, skip replicates with < 4 distinct pairs, percentile 95% CI).
Because the three populations share tasks, test sets and (for P) the very same adapters as R0 or S1, they are NOT independent;
the joint resampling of tasks keeps that dependence inside every replicate, but the pooled interval is still not a
confirmatory quantity. A joint task-label permutation (10,000 relabelings, default_rng(0), same permutation applied to the
D matrix of every population) gives a descriptive p-value.
Sensitivity analyses: (i) Fisher-z average; (ii) rho between the per-task-pair MEAN predictor and MEAN D over the three
populations (averaging reduces seed and evaluation noise in D); (iii) leave-one-task-out of rho_bar.
Sanity check: with the same RNG stream, the per-population bootstrap CIs must reproduce E1b analysis.json and
E1c analysis_e1c.json exactly (they use the same task order and the same draws).

Run:  /workspace/lora-paper/e1/.venv/bin/python analysis/pooled_rho.py
Outputs: analysis/pooled_rho.json, analysis/pooled_rho.md
"""
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
E1B = ROOT / "e1b" / "results"
E1C = ROOT / "e1c"
OUT = ROOT / "analysis"

PREDICTORS = ["O_A", "tv_cosine", "mean_theta_min_A_deg", "min_theta_min_A_deg", "sign_conflict_top20",
              "sign_conflict_all", "norm_ratio"]          # O_B == O_A and null_z_O_A is monotone in O_A: omitted
PRIMARY = ["O_A", "tv_cosine"]
NBOOT, NPERM = 2000, 10000


def spearman(x, y):                                    # identical to e1b.spearman
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    tasks = json.loads((E1B / "stage0.json").read_text())["valid_tasks"]
    pr0 = pd.read_csv(E1B / "predictors.csv"); rr0 = pd.read_csv(E1B / "pair_results.csv")
    r0 = pr0.merge(rr0[["pair", "D"]], on="pair"); r0["t1_task"] = r0.t1; r0["t2_task"] = r0.t2
    pe = pd.read_csv(E1C / "predictors_e1c.csv"); re_ = pd.read_csv(E1C / "pair_results_e1c.csv")
    pops = {"R0": r0}
    for p in ("P", "S1"):
        d = re_[re_["pop"] == p][["pair", "D"]]
        pops[p] = pe.merge(d, on="pair")
    for k, df in pops.items():
        assert len(df) == 91, (k, len(df))
    return tasks, pops


def design(df, tasks):
    K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    I = np.array([tix[x] for x in df.t1_task]); J = np.array([tix[x] for x in df.t2_task])
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))
    assert (PI[~np.eye(K, dtype=bool)] >= 0).all()
    return I, J, PI


def main():
    tasks, pops = load(); K = len(tasks); names = list(pops)
    D = {p: pops[p]["D"].values.astype(float) for p in names}
    des = {p: design(pops[p], tasks) for p in names}
    # joint bootstrap draws (same construction and RNG as e1b.stage3)
    brng = np.random.default_rng(0); boots = {p: [] for p in names}; skipped = 0
    for _ in range(NBOOT):
        s = brng.integers(0, K, K); ii = [(a, b) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
        bis = {p: np.array([des[p][2][s[a], s[b]] for a, b in ii], dtype=np.int64) for p in names}
        if len(np.unique(bis["R0"])) < 4: skipped += 1; continue
        for p in names: boots[p].append(bis[p])
    perms = np.argsort(np.random.default_rng(0).random((NPERM, K)), axis=1)
    maps = {p: des[p][2][perms[:, des[p][0]], perms[:, des[p][1]]] for p in names}

    res = {}
    for c in PREDICTORS:
        X = {p: pops[p][c].values.astype(float) for p in names}
        rho = {p: spearman(X[p], D[p]) for p in names}
        rbar = float(np.mean([rho[p] for p in names]))
        zbar = float(np.tanh(np.mean([np.arctanh(rho[p]) for p in names])))
        bs = {p: np.array([spearman(X[p][bi], D[p][bi]) for bi in boots[p]]) for p in names}
        bbar = np.nanmean(np.vstack([bs[p] for p in names]), axis=0)
        rp = {p: np.array([spearman(X[p], D[p][m]) for m in maps[p]]) for p in names}
        pbar = np.mean(np.vstack([rp[p] for p in names]), axis=0)
        p1 = float((1 + np.sum(pbar >= rbar - 1e-12)) / (1 + NPERM)); p2 = float((1 + np.sum(np.abs(pbar) >= abs(rbar) - 1e-12)) / (1 + NPERM))
        loto = {}
        for t in tasks:
            vals = []
            for p in names:
                I, J, _ = des[p]; m = (I != tasks.index(t)) & (J != tasks.index(t))
                vals.append(spearman(X[p][m], D[p][m]))
            loto[t] = float(np.mean(vals))
        # sensitivity (ii): average predictor and D per unordered task pair, then one rho
        key = lambda df: [tuple(sorted((a, b))) for a, b in zip(df.t1_task, df.t2_task)]
        tab = {p: pd.DataFrame({"u": key(pops[p]), "x": X[p], "d": D[p]}).set_index("u") for p in names}
        u_order = list(tab["R0"].index)
        xm = np.mean([tab[p].loc[u_order, "x"].values for p in names], axis=0)
        dm = np.mean([tab[p].loc[u_order, "d"].values for p in names], axis=0)
        r0df = pops["R0"]; I0, J0, PI0 = des["R0"]
        rho_mean = spearman(xm, dm)
        bs_mean = np.array([spearman(xm[bi], dm[bi]) for bi in boots["R0"]])  # R0 row order == u_order
        res[c] = {
            "rho_per_population": rho,
            "per_population_boot_ci95": {p: [float(np.nanpercentile(bs[p], 2.5)), float(np.nanpercentile(bs[p], 97.5))] for p in names},
            "rho_bar": rbar, "rho_bar_boot_ci95": [float(np.nanpercentile(bbar, 2.5)), float(np.nanpercentile(bbar, 97.5))],
            "rho_bar_boot_sd": float(np.nanstd(bbar, ddof=1)),
            "rho_bar_perm_p_one_sided": p1, "rho_bar_perm_p_two_sided": p2,
            "fisher_z_mean_rho": zbar,
            "loto_rho_bar": loto, "loto_rho_bar_range": [float(min(loto.values())), float(max(loto.values()))],
            "rho_of_pairmeans": rho_mean,
            "rho_of_pairmeans_boot_ci95": [float(np.nanpercentile(bs_mean, 2.5)), float(np.nanpercentile(bs_mean, 97.5))],
        }
    # sanity: per-population CIs vs the pre-registered analysis files
    AE = json.loads((E1B / "analysis.json").read_text()); AC = json.loads((E1C / "analysis_e1c.json").read_text())
    ref = {("R0", "O_A"): AE["verdicts"]["H1"]["boot_ci95"], ("R0", "tv_cosine"): AE["verdicts"]["H2"]["boot_ci95"]}
    for p in ("P", "S1"):
        for c in PRIMARY: ref[(p, c)] = AC[p]["all_predictor_stats"][c]["boot_ci95"]
    sanity = {f"{p}:{c}": max(abs(a - b) for a, b in zip(res[c]["per_population_boot_ci95"][p], v)) for (p, c), v in ref.items()}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S KST"), "status": "EXPLORATORY (not pre-registered; written after all results)",
           "populations": {"R0": "E1b, both adapters seed 0", "P": "E1c primary, mixed seed", "S1": "E1c secondary, both seed 1"},
           "K_tasks": K, "tasks": tasks, "n_pairs_per_population": 91, "boot_reps_used": len(boots["R0"]), "boot_skipped": skipped,
           "n_perm": NPERM, "inputs_sha256": {str(f.relative_to(ROOT)): sha(f) for f in [E1B / "predictors.csv", E1B / "pair_results.csv", E1C / "predictors_e1c.csv", E1C / "pair_results_e1c.csv"]},
           "sanity_max_abs_diff_per_population_ci_vs_prereg_files": sanity, "results": res}
    (OUT / "pooled_rho.json").write_text(json.dumps(out, indent=1))
    L = ["# Pooled Spearman rho across R0 (E1b), P and S1 (E1c) — EXPLORATORY", "",
         f"Generated {out['generated']} by `analysis/pooled_rho.py`. Not pre-registered; written after all E1b/E1c results. The three populations share all 14 tasks, the evaluation sets and, for P, the adapters themselves (P's t1 adapters are R0 adapters, P's t2 adapters are S1 adapters), so they are not independent. The bootstrap resamples tasks jointly across populations ({len(boots['R0'])} replicates).", "",
         "| predictor | ρ R0 | ρ P | ρ S1 | **ρ̄ (mean)** | joint task-block 95% CI | joint perm. p (1-sided / 2-sided) | Fisher-z mean | LOTO ρ̄ range | ρ(pair-mean predictor, pair-mean D) [95% CI] |",
         "|---|---:|---:|---:|---:|---|---|---:|---|---|"]
    for c in PREDICTORS:
        r = res[c]; rp = r["rho_per_population"]
        L.append(f"| {c} | {rp['R0']:.3f} | {rp['P']:.3f} | {rp['S1']:.3f} | **{r['rho_bar']:.3f}** | [{r['rho_bar_boot_ci95'][0]:.3f}, {r['rho_bar_boot_ci95'][1]:.3f}] | {r['rho_bar_perm_p_one_sided']:.3f} / {r['rho_bar_perm_p_two_sided']:.3f} | {r['fisher_z_mean_rho']:.3f} | [{r['loto_rho_bar_range'][0]:.3f}, {r['loto_rho_bar_range'][1]:.3f}] | {r['rho_of_pairmeans']:.3f} [{r['rho_of_pairmeans_boot_ci95'][0]:.3f}, {r['rho_of_pairmeans_boot_ci95'][1]:.3f}] |")
    L += ["", f"Sanity check (per-population CIs from the joint draws vs the pre-registered analysis files; max |Δ|): {json.dumps(sanity)}"]
    (OUT / "pooled_rho.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
