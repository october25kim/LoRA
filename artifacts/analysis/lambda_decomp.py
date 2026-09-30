#!/usr/bin/env python3
"""lambda-decomposition of task-arithmetic merge loss (exploratory analysis C1).

For every adapter pair we split the pre-registered pair loss D (TA at the lambda selected on held-out TRAIN
examples) into
    D_selected = D_oracle + overshoot,
    D_oracle   = min_lambda D_lambda   (best lambda chosen on the evaluation split itself; descriptive/optimistic),
    overshoot  = D_selected - D_oracle (>= 0; the loss attributable to lambda selection),
and we also keep D_lambda at every fixed lambda of the grid. Each weight-space predictor is then correlated
(Spearman) with each target using the same uncertainty machinery as the E1 pilot (e1.py, stage3):
  * task-block bootstrap: resample the K tasks with replacement, keep every induced pair of distinct resampled
    indices (duplicates kept), skip a replicate with < 4 distinct pairs; 2000 reps; percentile 95% CI;
  * task-label permutation: permute task labels of the D matrix with the predictor fixed; 10000 reps;
    two-sided p = (1 + #{|rho_perm| >= |rho_obs|}) / (1 + nperm), one-sided (>=) likewise;
    exact enumeration over all K! relabelings when K <= 8.
RNG: one numpy default_rng(seed) stream; the 2000 bootstrap draws are consumed first, then the 10000 permutations
(shared by all predictor/target cells). With seed 0 this reproduces the E1 bootstrap exactly, and the E1
permutation stream of the primary predictor O_A exactly (sanity check reported as `reproduction_check`).

Input schemas supported (auto-detected), so the script can be rerun unchanged on the E1b confirmatory output:
  * E1  (e1/artifacts/e1_predictive):  pair_results.csv with val_D_lam{lam}, lam_selected, D
  * E1b (e1b/ mirror of artifacts/e1b_confirmatory): pair_results.csv with TA_eval_D_lam{lam}, lam_selected, D
predictors.csv must contain pair, t1, t2 and the predictor columns. Task order is taken from stage0.json
(`valid_tasks` for E1b, `final_tasks` for E1) when present, else from first appearance in the pair list.

Usage:
  python lambda_decomp.py --in /workspace/lora-paper/e1/artifacts/e1_predictive --tag e1
  python lambda_decomp.py --in /workspace/lora-paper/e1b --tag e1b          # when E1b has finished
Outputs (in --out, default: directory of this script):
  lambda_decomp_<tag>.json, lambda_decomp_<tag>_pairs.csv, lambda_decomp_<tag>_stats.csv,
  lambda_decomp_<tag>_table.md, fig_lambda_decomp_<tag>.png
Everything produced here is EXPLORATORY (not pre-registered in E1; the E1b pre-registration lists only fixed-lambda
curves as exploratory).
"""
from __future__ import annotations

import argparse, itertools, json, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PREDICTORS = ["O_A", "tv_cosine", "sign_conflict_top20", "mean_theta_min_A_deg", "norm_ratio"]
PRED_LABEL = {"O_A": "O_A (subspace overlap)", "tv_cosine": "task-vector cosine",
              "sign_conflict_top20": "sign conflict (top 20%)", "mean_theta_min_A_deg": "mean theta_min",
              "norm_ratio": "norm ratio"}
DCOL_RE = re.compile(r"^(?:val_D_lam|TA_eval_D_lam)([0-9.]+)$")


# ----------------------------------------------------------------------------- helpers
def spearman(x, y):
    """Same conventions as e1.py: nan for n<3 or a constant input."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic)


def spearman_rows(x, Y):
    """Spearman of fixed vector x with every row of Y (average ranks, like scipy); nan for constant rows."""
    rx = rankdata(x); rx = rx - rx.mean()
    RY = rankdata(Y, axis=1); RY = RY - RY.mean(axis=1, keepdims=True)
    num = RY @ rx
    den = np.sqrt((RY ** 2).sum(axis=1) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def lam_key(v: float) -> str:
    return f"{v:g}" if v != 1 else "1.0"


def load(indir: Path):
    P = pd.read_csv(indir / "predictors.csv")
    R = pd.read_csv(indir / "pair_results.csv")
    dcols = {}
    for c in R.columns:
        m = DCOL_RE.match(c)
        if m:
            dcols[float(m.group(1))] = c
    if not dcols:
        raise SystemExit("no fixed-lambda D columns (val_D_lam* / TA_eval_D_lam*) in pair_results.csv")
    lams = sorted(dcols)
    schema = "E1" if any(c.startswith("val_D_lam") for c in dcols.values()) else "E1b"
    keep = ["pair", "D", "lam_selected"] + [dcols[l] for l in lams]
    df = P.merge(R[keep], on="pair", how="inner")
    if len(df) != len(R) or len(df) != len(P):
        print(f"WARNING: {len(P)} predictor rows, {len(R)} result rows, {len(df)} merged", file=sys.stderr)
    tasks = None
    s0p = indir / "stage0.json"
    if s0p.exists():
        s0 = json.loads(s0p.read_text())
        tasks = s0.get("valid_tasks") or s0.get("final_tasks")
    if not tasks:
        tasks = list(dict.fromkeys(list(itertools.chain.from_iterable(zip(df.t1, df.t2)))))
    present = set(df.t1) | set(df.t2)
    tasks = [t for t in tasks if t in present]
    return df, lams, dcols, tasks, schema


# ----------------------------------------------------------------------------- main analysis
def run(indir: Path, out: Path, tag: str, nboot: int, nperm: int, seed: int, e1_analysis: Path | None):
    df, lams, dcols, tasks, schema = load(indir)
    K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    Dl = np.stack([df[dcols[l]].values.astype(float) for l in lams], axis=1)       # pairs x lams
    # oracle: argmin over the grid (ties -> smaller lambda, the same tie rule as selection)
    io = np.argmin(Dl, axis=1)
    df["lam_oracle"] = [lams[i] for i in io]
    df["D_oracle"] = Dl[np.arange(len(df)), io]
    df["D_selected"] = df["D"].astype(float)
    # consistency: D_selected must equal D at lam_selected
    dsel_chk = np.array([df.loc[k, dcols[float(df.loc[k, "lam_selected"])]] for k in df.index], float)
    max_sel_diff = float(np.max(np.abs(dsel_chk - df.D_selected.values)))
    df["overshoot"] = df["D_selected"] - df["D_oracle"]
    df["overshoot_lam1"] = df[dcols[max(lams)]] - df["D_oracle"] if 1.0 in dcols else np.nan
    targets = ["D_selected", "D_oracle", "overshoot"] + [f"D_lam{lam_key(l)}" for l in lams]
    for l in lams:
        df[f"D_lam{lam_key(l)}"] = df[dcols[l]].astype(float)
    extra_targets = ["overshoot_lam1"] if 1.0 in dcols else []

    I = np.array([tix[t] for t in df.t1]); J = np.array([tix[t] for t in df.t2])
    PI = np.full((K, K), -1, dtype=np.int64); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))

    rng = np.random.default_rng(seed)
    boots, skipped = [], 0
    for _ in range(nboot):                                     # identical draw sequence to e1.py boot()
        s = rng.integers(0, K, K)
        ii = [(s[a], s[b]) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
        if len(set(tuple(sorted(p)) for p in ii)) < 4:
            skipped += 1; continue
        boots.append(np.array([PI[i, j] for i, j in ii], dtype=np.int64))
    perms = np.stack([rng.permutation(K) for _ in range(nperm)])   # identical to e1.py's first predictor stream
    pmaps = PI[perms[:, I], perms[:, J]]                        # nperm x pairs: row index of the relabeled pair
    exact_maps = None
    if K <= 8:
        allp = np.array(list(itertools.permutations(range(K))))
        exact_maps = PI[allp[:, I], allp[:, J]]

    def cell(xc, yc):
        x = df[xc].values.astype(float); y = df[yc].values.astype(float)
        obs = spearman(x, y)
        rp = spearman_rows(x, y[pmaps])
        ok = np.isfinite(rp)
        p2 = (1 + np.sum(np.abs(rp[ok]) >= abs(obs) - 1e-12)) / (1 + nperm) if np.isfinite(obs) else np.nan
        p1 = (1 + np.sum(rp[ok] >= obs - 1e-12)) / (1 + nperm) if np.isfinite(obs) else np.nan
        ex2 = ex1 = np.nan
        if exact_maps is not None and np.isfinite(obs):
            re_ = spearman_rows(x, y[exact_maps]); ok2 = np.isfinite(re_)
            ex2 = float(np.mean(np.abs(re_[ok2]) >= abs(obs) - 1e-12)); ex1 = float(np.mean(re_[ok2] >= obs - 1e-12))
        bs = np.array([spearman(x[b], y[b]) for b in boots])
        lo, hi = (np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)) if np.isfinite(bs).any() else (np.nan, np.nan)
        loto = {t: spearman(x[(I != tix[t]) & (J != tix[t])], y[(I != tix[t]) & (J != tix[t])]) for t in tasks}
        return {"predictor": xc, "target": yc, "rho": obs, "ci_lo": float(lo), "ci_hi": float(hi),
                "boot_nan_frac": float(np.mean(~np.isfinite(bs))),
                "perm_p_two_sided": float(p2), "perm_p_one_sided_ge": float(p1),
                "exact_p_two_sided": ex2, "exact_p_one_sided_ge": ex1,
                "loto_min": float(np.nanmin(list(loto.values()))), "loto_max": float(np.nanmax(list(loto.values())))}

    rows = [cell(p, t) for t in targets + extra_targets for p in PREDICTORS if p in df.columns]
    S = pd.DataFrame(rows)
    # Holm over the predictors within each target (exploratory bookkeeping only)
    S["holm_p_within_target"] = np.nan
    for t, g in S.groupby("target"):
        ps = g["perm_p_two_sided"].values; order = np.argsort(ps); m = len(ps); run_ = 0.0; adj = np.empty(m)
        for rnk, k in enumerate(order):
            run_ = max(run_, min(1.0, (m - rnk) * ps[k])); adj[k] = run_
        S.loc[g.index, "holm_p_within_target"] = adj

    # descriptive: means with task-block bootstrap CIs
    def mean_ci(col):
        v = df[col].values.astype(float)
        bm = [float(np.mean(v[b])) for b in boots]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                "task_block_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
    desc = {c: mean_ci(c) for c in ["D_selected", "D_oracle", "overshoot"] + extra_targets + [f"D_lam{lam_key(l)}" for l in lams]}
    share = df.overshoot.sum() / df.D_selected.sum() if df.D_selected.sum() != 0 else np.nan
    agree = int(np.sum(np.isclose(df.lam_selected.astype(float), df.lam_oracle.astype(float))))
    xt = pd.crosstab(df.lam_selected.astype(float), df.lam_oracle.astype(float))

    # reproduction check against the pilot's analysis.json (only meaningful for E1 input with default settings)
    repro = None
    if e1_analysis is not None and e1_analysis.exists():
        A = json.loads(e1_analysis.read_text())
        pr = A.get("primary", {})
        mine = S[(S.predictor == "O_A") & (S.target == "D_selected")].iloc[0]
        repro = {"rho_e1": pr.get("rho"), "rho_here": mine.rho,
                 "ci_e1": pr.get("task_block_bootstrap_95ci"), "ci_here": [mine.ci_lo, mine.ci_hi],
                 "perm_p2_e1": pr.get("perm_p_two_sided_10000"), "perm_p2_here": mine.perm_p_two_sided}
        sec = A.get("secondary_holm", {}) | A.get("extra_exploratory_uncorrected", {})
        repro["other_predictors_rho_ci_e1_vs_here"] = {
            p: {"e1": [sec[p]["rho"], sec[p]["task_block_bootstrap_95ci"]],
                "here": [float(S[(S.predictor == p) & (S.target == "D_selected")].rho.iloc[0]),
                         [float(S[(S.predictor == p) & (S.target == "D_selected")].ci_lo.iloc[0]),
                          float(S[(S.predictor == p) & (S.target == "D_selected")].ci_hi.iloc[0])]]}
            for p in PREDICTORS if p in sec}

    out.mkdir(parents=True, exist_ok=True)
    pcols = ["pair", "t1", "t2", "lam_selected", "lam_oracle", "D_selected", "D_oracle", "overshoot"] + extra_targets + \
            [f"D_lam{lam_key(l)}" for l in lams] + [p for p in PREDICTORS if p in df.columns]
    df[pcols].to_csv(out / f"lambda_decomp_{tag}_pairs.csv", index=False, float_format="%.6g")
    S.to_csv(out / f"lambda_decomp_{tag}_stats.csv", index=False, float_format="%.6g")
    J_ = {"label": "EXPLORATORY (not pre-registered)", "input_dir": str(indir), "schema": schema, "tasks": tasks, "K": K,
          "n_pairs": int(len(df)), "lambda_grid": lams, "nboot": nboot, "bootstrap_used": len(boots), "bootstrap_skipped": skipped,
          "nperm": nperm, "seed": seed, "exact_enumeration": exact_maps is not None,
          "max_abs_diff_D_vs_D_at_lam_selected": max_sel_diff,
          "descriptive": desc, "overshoot_share_of_total_D": float(share),
          "lam_selected_equals_lam_oracle": agree, "crosstab_selected_rows_oracle_cols": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in xt.iterrows()},
          "stats": rows_to_json(S), "reproduction_check": repro}
    (out / f"lambda_decomp_{tag}.json").write_text(json.dumps(J_, indent=1, default=float))
    write_table(S, desc, J_, out / f"lambda_decomp_{tag}_table.md", targets, extra_targets)
    figure(df, S, lams, out / f"fig_lambda_decomp_{tag}.png", tag)
    return J_, S, df


def rows_to_json(S):
    return [{k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()} for r in S.to_dict("records")]


def fmt(v, d=2):
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def write_table(S, desc, J_, path, targets, extra):
    L = [f"<!-- generated by lambda_decomp.py; input {J_['input_dir']} ({J_['schema']} schema); EXPLORATORY -->",
         f"K = {J_['K']} tasks, {J_['n_pairs']} pairs; lambda grid {J_['lambda_grid']}; task-block bootstrap {J_['bootstrap_used']} used "
         f"({J_['bootstrap_skipped']} skipped); {J_['nperm']} task-label permutations (seed {J_['seed']}).", "",
         "| quantity | mean | median | range | task-block 95% CI of mean |", "|---|---:|---:|---|---|"]
    for k, v in desc.items():
        L.append(f"| {k} | {v['mean']:.4f} | {v['median']:.4f} | [{v['min']:.4f}, {v['max']:.4f}] | [{v['task_block_ci95'][0]:.4f}, {v['task_block_ci95'][1]:.4f}] |")
    L += ["", f"lambda_selected == lambda_oracle in {J_['lam_selected_equals_lam_oracle']}/{J_['n_pairs']} pairs; "
          f"overshoot share of summed D = {J_['overshoot_share_of_total_D']:.3f}.", "",
          "| predictor | target | Spearman rho | task-block 95% CI | perm p (2-sided) | exact p (2-sided) | Holm p (within target) |",
          "|---|---|---:|---|---:|---:|---:|"]
    for t in targets + extra:
        for _, r in S[S.target == t].iterrows():
            L.append(f"| {PRED_LABEL.get(r.predictor, r.predictor)} | {t} | {fmt(r.rho, 3)} | [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | "
                     f"{fmt(r.perm_p_two_sided, 3)} | {fmt(r.exact_p_two_sided, 3)} | {fmt(r.holm_p_within_target, 3)} |")
    path.write_text("\n".join(L) + "\n")


def figure(df, S, lams, path, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    Dl = np.stack([df[f"D_lam{lam_key(l)}"].values for l in lams], axis=1)
    for k in range(len(df)):
        a0.plot(lams, Dl[k], color="0.7", lw=0.8, zorder=1)
    a0.scatter(df.lam_oracle, df.D_oracle, marker="v", s=40, color="tab:green", label="oracle lambda (eval split)", zorder=3)
    jit = (np.random.default_rng(1).random(len(df)) - 0.5) * 0.03
    a0.scatter(df.lam_selected.astype(float) + jit, df.D_selected, marker="o", s=22, facecolor="none", edgecolor="tab:red",
               label="lambda selected on held-out train", zorder=4)
    a0.axhline(0, color="k", lw=0.5)
    a0.set_xticks(lams); a0.set_xlabel("merge coefficient lambda"); a0.set_ylabel("pair loss D_lambda (eval split)")
    a0.set_title(f"(a) D as a function of lambda, {len(df)} pairs ({tag})", fontsize=10); a0.legend(fontsize=8)
    tg = ["D_selected", "D_oracle", "overshoot"]; col = {"D_selected": "k", "D_oracle": "tab:green", "overshoot": "tab:red"}
    preds = [p for p in PREDICTORS if p in set(S.predictor)]
    y = 0; yt = []; yl = []
    for p in preds:
        for j, t in enumerate(tg):
            r = S[(S.predictor == p) & (S.target == t)].iloc[0]
            yy = y + j * 0.25
            if np.isfinite(r.rho):
                a1.errorbar(r.rho, yy, xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]], fmt="o", color=col[t], ms=4, capsize=2,
                            label=t if p == preds[0] else None)
        yt.append(y + 0.25); yl.append(PRED_LABEL[p]); y += 1
    a1.axvline(0, color="k", lw=0.5); a1.set_yticks(yt); a1.set_yticklabels(yl); a1.invert_yaxis(); a1.set_xlim(-1, 1)
    a1.set_xlabel("Spearman rho (task-block bootstrap 95% CI)")
    a1.set_title("(b) predictor vs. D_selected / D_oracle / overshoot (exploratory)", fontsize=10)
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, help="directory with predictors.csv, pair_results.csv (+ stage0.json)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--tag", default=None, help="suffix for output files (default: name of input dir)")
    ap.add_argument("--nboot", type=int, default=2000); ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e1-analysis", default=None, help="analysis.json of the pilot for a reproduction check (default: <in>/analysis.json if E1 schema)")
    a = ap.parse_args()
    indir = Path(a.indir).resolve(); tag = a.tag or indir.name
    e1a = Path(a.e1_analysis) if a.e1_analysis else indir / "analysis.json"
    J_, S, df = run(indir, Path(a.out).resolve(), tag, a.nboot, a.nperm, a.seed, e1a if e1a.exists() else None)
    print(json.dumps({k: J_[k] for k in ("schema", "K", "n_pairs", "descriptive", "overshoot_share_of_total_D",
                                          "lam_selected_equals_lam_oracle", "max_abs_diff_D_vs_D_at_lam_selected", "reproduction_check")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
