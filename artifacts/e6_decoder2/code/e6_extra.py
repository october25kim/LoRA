#!/usr/bin/env python3
"""E6 pre-specified additional analyses (PREREG_E6.md sec. 6b; secondary / exploratory, no decision rule). Read-only on E4b artifacts.
  (1) O_A location breakdown: where the sub-30 deg principal angles (theta_min < 30 deg pair-layers) sit, by module type, layer index and
      relative depth, per population; share in layer 0 k_proj; the same table recomputed from the E4b (Qwen2.5-0.5B) layer file.
  (2) Gate (theta* = 30 deg) firing and benefit: gate-active pairs; GATE - TA (at lam*_TA and own-lam) on gate-active pairs: mean, median,
      task-block bootstrap CI (the pop_stats task-block resamples), Wilcoxon p (two-sided), W/T/L.
  (3) Replication contrast vs E4b and pooled decoder estimate (EXPLORATORY): per population and predictor (O_A, tv_cosine),
      rho_E4b, rho_E6, their mean (pooled decoder rho) and difference; a JOINT task-block bootstrap (the same resampled task multiset applied
      to both backbones; 2,000 reps, default_rng(0), same skip rule as pop_stats) gives CIs; a JOINT permutation (the same task relabeling
      applied to both backbones; 10,000, default_rng(0)) gives a one-sided p for the pooled mean rho. Common valid tasks only.
Output: e6_extra.json. Run via python e6.py --stage stage3 (after e6_analysis) or directly."""
import json, sys, itertools
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"; E4B = HERE.parent / "e4b_decoder"
sys.path.insert(0, str(E1B)); sys.path.insert(0, str(HERE))
import e1b as bm  # noqa: E402
bm.LOGF = HERE / "run_e6.log"; bm.OUT = HERE
spearman = bm.spearman
PRIM = {"H1": "O_A", "H2": "tv_cosine"}


def upair(a, b): return tuple(sorted((a, b)))


def location(LR, pairs, n_layers_model):
    L = LR[LR.pair.isin(pairs)].copy()
    L["module"] = L.layer.str.replace(r"^.*layers\.\d+\.", "", regex=True)
    L["layer_idx"] = L.layer.str.extract(r"layers\.(\d+)\.")[0].astype(int)
    sub = L[L.theta_min_A_deg < 30.0]
    n = int(len(sub))
    out = {"n_pair_layers": int(len(L)), "n_pairs": int(L.pair.nunique()), "n_lt30": n,
           "n_pairs_with_lt30": int(sub.pair.nunique()),
           "by_module": {k: int(v) for k, v in sub.module.value_counts().items()},
           "by_layer_idx": {int(k): int(v) for k, v in sub.layer_idx.value_counts().sort_index().items()},
           "top_locations": {k: int(v) for k, v in sub.layer.value_counts().head(12).items()},
           "share_layer0_k_proj": float((sub.layer.str.endswith("layers.0.self_attn.k_proj")).mean()) if n else float("nan"),
           "share_layer0_any": float((sub.layer_idx == 0).mean()) if n else float("nan"),
           "share_last_layer": float((sub.layer_idx == n_layers_model - 1).mean()) if n else float("nan"),
           "share_k_proj": float((sub.module == "self_attn.k_proj").mean()) if n else float("nan"),
           "share_v_proj": float((sub.module == "self_attn.v_proj").mean()) if n else float("nan"),
           "rel_depth_quartiles_counts": {q: int(v) for q, v in
                                          (sub.layer_idx / (n_layers_model - 1)).pipe(lambda x: np.minimum((x * 4).astype(int), 3)).value_counts().sort_index().items()},
           "min_theta_by_module": {k: float(g.theta_min_A_deg.min()) for k, g in L.groupby("module")},
           "median_theta_by_module": {k: float(g.theta_min_A_deg.median()) for k, g in L.groupby("module")},
           "frac_pairs_lt30_only_in_layer0_k_proj": float(np.mean([set(g.layer) <= {"model.layers.0.self_attn.k_proj"} for _, g in sub.groupby("pair")])) if n else float("nan")}
    return out


def task_structure(df, tasks):
    K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    I = np.array([tix[x] for x in df.t1_task]); J = np.array([tix[x] for x in df.t2_task])
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))
    assert (PI[~np.eye(K, dtype=bool)] >= 0).all(), "incomplete design"
    return I, J, PI


def joint(dfA, dfB, tasks, col):
    """Joint task-block bootstrap / permutation over two backbones sharing the same task set (pop_stats resampling scheme)."""
    K = len(tasks)
    IA, JA, PA_ = task_structure(dfA, tasks); IB, JB, PB_ = task_structure(dfB, tasks)
    xa, ya = dfA[col].values.astype(float), dfA.D.values.astype(float); xb, yb = dfB[col].values.astype(float), dfB.D.values.astype(float)
    ra, rb = spearman(xa, ya), spearman(xb, yb)
    brng = np.random.default_rng(0); bm_, bd, ba, bb = [], [], [], []; skipped = 0
    for _ in range(2000):
        s = brng.integers(0, K, K); ii = [(a_, b_) for a_ in range(K) for b_ in range(a_ + 1, K) if s[a_] != s[b_]]
        ia = np.array([PA_[s[a_], s[b_]] for a_, b_ in ii], dtype=np.int64); ib = np.array([PB_[s[a_], s[b_]] for a_, b_ in ii], dtype=np.int64)
        if len(np.unique(ia)) < 4: skipped += 1; continue
        va, vb = spearman(xa[ia], ya[ia]), spearman(xb[ib], yb[ib])
        ba.append(va); bb.append(vb); bm_.append(0.5 * (va + vb)); bd.append(vb - va)
    perms = np.argsort(np.random.default_rng(0).random((10000, K)), axis=1)
    mA = PA_[perms[:, IA], perms[:, JA]]; mB = PB_[perms[:, IB], perms[:, JB]]
    # relabeling permutes D across pairs within each backbone with the SAME task permutation
    pm = np.array([0.5 * (spearman(xa, ya[m1]) + spearman(xb, yb[m2])) for m1, m2 in zip(mA, mB)])
    obs = 0.5 * (ra + rb)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"rho_E4b": ra, "rho_E6": rb, "pooled_mean_rho": obs, "diff_E6_minus_E4b": rb - ra,
            "joint_boot_ci95_pooled": ci(bm_), "joint_boot_ci95_diff": ci(bd), "boot_ci95_E4b": ci(ba), "boot_ci95_E6": ci(bb),
            "joint_perm_p_one_sided_pooled": float((1 + np.sum(pm >= obs - 1e-12)) / (1 + len(pm))), "boot_reps": len(bm_), "boot_skipped": skipped}


def load_pop(d, tag, pop):
    import pandas as pd
    PR = pd.read_csv(d / f"predictors_{tag}.csv"); jl = d / f"pair_results_{pop}.jsonl"
    if not jl.exists(): return None
    R = pd.DataFrame([json.loads(l) for l in jl.read_text().splitlines() if l.strip()]).drop_duplicates("pair")
    return PR.merge(R.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")


def gate_benefit(df, tasks):
    from scipy.stats import wilcoxon
    g = df[df.gate_active.astype(bool)].reset_index(drop=True)
    out = {"n_pairs": int(len(df)), "n_gate_active": int(len(g)), "n_fail_layers_total": int(df.gate_n_fail_layers.sum()),
           "fail_layers_per_active_pair": {str(k): int(v) for k, v in g.gate_n_fail_layers.value_counts().sort_index().items()}}
    if len(g) < 5: return out
    _, _, PI = task_structure(df, tasks); K = len(tasks)
    act = set(np.flatnonzero(df.gate_active.astype(bool).values))
    brng = np.random.default_rng(0); boots = []
    for _ in range(2000):
        s = brng.integers(0, K, K)
        bi = [PI[s[a_], s[b_]] for a_ in range(K) for b_ in range(a_ + 1, K) if s[a_] != s[b_] and PI[s[a_], s[b_]] in act]
        if len(np.unique(bi)) >= 4: boots.append(np.array(bi))
    for tag, col in (("atTA", "GATE_atTA_score"), ("own", "GATE_own_score")):
        dall = (df[col] - df.TA_score_sel).values; d = dall[df.gate_active.astype(bool).values]
        nz = np.abs(d) > 1e-12
        try: wp = float(wilcoxon(d[nz]).pvalue) if nz.sum() >= 5 else float("nan")
        except Exception: wp = float("nan")
        tb = [float(np.mean(dall[bi])) for bi in boots]
        out[tag] = {"mean_pp": float(d.mean() * 100), "median_pp": float(np.median(d) * 100),
                    "task_block_ci95_pp": [float(np.percentile(tb, 2.5) * 100), float(np.percentile(tb, 97.5) * 100)] if tb else None,
                    "wilcoxon_p_two_sided": wp, "win": int(np.sum(d > 1e-12)), "tie": int(np.sum(~nz)), "loss": int(np.sum(d < -1e-12))}
    return out


def main():
    import pandas as pd
    PO = json.loads((HERE / "populations_e6.json").read_text()); tasks6 = PO["valid_tasks"]
    PA = json.loads((HERE / "prereg_e6.json").read_text())
    X = {"note": "secondary/exploratory (PREREG_E6.md sec. 6b); no decision rule"}
    # (1) location
    LR6 = pd.read_csv(HERE / "predictors_layers_e6.csv")
    loc = {"e6": {p: location(LR6, PO["populations"][p], PA["recipe"]["n_decoder_layers"]) for p in PO["populations"]}}
    try:
        POb = json.loads((E4B / "populations_e4b.json").read_text()); LRb = pd.read_csv(E4B / "predictors_layers_e4b.csv")
        loc["e4b"] = {p: location(LRb, POb["populations"][p], 24) for p in POb["populations"]}
    except Exception as ex:
        loc["e4b_error"] = repr(ex)
    X["location_lt30"] = loc
    # (2) gate benefit
    X["gate"] = {}
    for p in ("P", "R0", "S1"):
        df = load_pop(HERE, "e6", p)
        if df is not None and len(df) == PO["n"][p]:
            X["gate"][p] = gate_benefit(df, tasks6)
    # (3) replication contrast + pooled decoder estimate (exploratory)
    X["pooled_decoder"] = {}
    try:
        POb = json.loads((E4B / "populations_e4b.json").read_text())
        common = [t for t in tasks6 if t in POb["valid_tasks"]]
        X["pooled_decoder"]["common_tasks"] = common
        for p in ("P", "R0", "S1"):
            A6 = load_pop(HERE, "e6", p); A4 = load_pop(E4B, "e4b", p)
            if A6 is None or A4 is None or len(A6) != PO["n"][p]: continue
            keep = lambda d: d[d.t1_task.isin(common) & d.t2_task.isin(common)].copy()
            A6, A4 = keep(A6), keep(A4)
            # align E6 rows to the E4b row order by unordered task pair
            A4["u"] = [upair(x, y) for x, y in zip(A4.t1_task, A4.t2_task)]; A6["u"] = [upair(x, y) for x, y in zip(A6.t1_task, A6.t2_task)]
            A6 = A6.set_index("u").loc[A4.u].reset_index(); A4 = A4.reset_index(drop=True)
            X["pooled_decoder"][p] = {h: joint(A4, A6, common, c) for h, c in PRIM.items()}
            X["pooled_decoder"][p]["n_pairs_each"] = int(len(A4))
    except Exception as ex:
        X["pooled_decoder"]["error"] = repr(ex)
    bm.jdump(X, HERE / "e6_extra.json")
    bm.log("e6 extra analyses written (e6_extra.json)")


if __name__ == "__main__":
    main()
