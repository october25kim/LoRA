#!/usr/bin/env python3
"""E1c stage3: pre-registered analysis (PREREG_E1C.md secs. 6-7). Statistics code copied verbatim from e1b.stage3 (perms, task-block bootstrap,
LOTO, Holm, PASS/FAIL/INCONCLUSIVE) and parameterized by population; E3 sign-flip copied verbatim from e3.py.
Run via:  python e1c.py --stage stage3"""
import json, math, sys, time, itertools
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(E1B))
import e1b as bm  # noqa: E402
bm.LOGF = HERE / "run_e1c.log"; bm.OUT = HERE
spearman = bm.spearman
PRIM = {"H1c": "O_A", "H2c": "tv_cosine"}
SECONDARY = ["O_B", "mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A", "sign_conflict_top20",
             "norm_ratio", "null_z_O_A", "sign_conflict_all"]
LAMS = bm.LAMS


def task_of(k): return k.split("@")[0]


def upair(a, b): return tuple(sorted((a, b)))


# ---------------------------------------------------------------- verbatim e1b.stage3 statistics, parameterized
def pop_stats(df, tasks, cols, Dcol="D"):
    K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    I = np.array([tix[x] for x in df.t1_task]); J = np.array([tix[x] for x in df.t2_task])
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))
    assert (PI[~np.eye(K, dtype=bool)] >= 0).all(), "incomplete design"
    D = df[Dcol].values
    perms = np.argsort(np.random.default_rng(0).random((10000, K)), axis=1)
    maps = PI[perms[:, I], perms[:, J]]
    brng = np.random.default_rng(0); boots = []; boot_skipped = 0
    for _ in range(2000):
        s = brng.integers(0, K, K); ii = [(a_, b_) for a_ in range(K) for b_ in range(a_ + 1, K) if s[a_] != s[b_]]
        bi = np.array([PI[s[a_], s[b_]] for a_, b_ in ii], dtype=np.int64)
        if len(np.unique(bi)) < 4: boot_skipped += 1; continue
        boots.append(bi)

    def stats_for(x, y):
        obs = spearman(x, y)
        if not np.isfinite(obs):
            return {"rho": float("nan"), "perm_p_one_sided": float("nan"), "perm_p_two_sided": float("nan"),
                    "boot_ci95": [float("nan"), float("nan")], "loto": {t: float("nan") for t in tasks}, "loto_frac_gt_0.2": float("nan"),
                    "note": f"predictor constant across pairs (value {x[0]}); rho undefined"}
        rp = np.array([spearman(x, y[m]) for m in maps])
        p1 = (1 + np.sum(rp >= obs - 1e-12)) / (1 + len(maps)); p2 = (1 + np.sum(np.abs(rp) >= abs(obs) - 1e-12)) / (1 + len(maps))
        bs = np.array([spearman(x[bi], y[bi]) for bi in boots])
        loto = {t: spearman(x[(I != tix[t]) & (J != tix[t])], y[(I != tix[t]) & (J != tix[t])]) for t in tasks}
        return {"rho": obs, "perm_p_one_sided": float(p1), "perm_p_two_sided": float(p2),
                "boot_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "loto": loto,
                "loto_frac_gt_0.2": float(np.mean([v > 0.2 for v in loto.values()]))}
    ST = {c: stats_for(df[c].values.astype(float), D) for c in cols}
    adj = bm.holm({h: ST[c]["perm_p_one_sided"] for h, c in PRIM.items()})
    verdicts = {}
    for h, c in PRIM.items():
        s = ST[c]; crit = {"rho_ge_0.4": s["rho"] >= 0.4, "holm_p_lt_0.05": adj[h] < 0.05, "loto_ge_75pct_gt_0.2": s["loto_frac_gt_0.2"] >= 0.75}
        v = "PASS" if all(crit.values()) else ("FAIL" if s["boot_ci95"][1] < 0.3 else "INCONCLUSIVE")
        verdicts[h] = {"predictor": c, "verdict": v, "rho": s["rho"], "boot_ci95": s["boot_ci95"], "perm_p_one_sided": s["perm_p_one_sided"],
                       "holm_p": adj[h], "loto_frac_gt_0.2": s["loto_frac_gt_0.2"], "loto": s["loto"], "criteria": crit}
    return {"verdicts": verdicts, "all_predictor_stats": ST, "boots": boots, "boot_skipped": boot_skipped, "K": K, "n_pairs": len(df)}


def method_comp(df, boots):
    from scipy.stats import wilcoxon
    meth = {"TA@lam*": df.TA_score_sel.values, "TIES-lite@lam*TA": df.TIES_atTA_score.values, "TIES-lite@own-lam": df.TIES_own_score.values,
            "gate30@lam*TA": df.GATE_atTA_score.values, "gate30@own-lam": df.GATE_own_score.values}
    prng = np.random.default_rng(0); MC = {}
    for k, v in meth.items():
        e = {"mean_norm_score": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v))}
        if k != "TA@lam*":
            dlt = v - meth["TA@lam*"]
            tb = [float(np.mean(dlt[bi])) for bi in boots] if boots is not None else [float(np.mean(dlt[prng.integers(0, len(dlt), len(dlt))])) for _ in range(2000)]
            pb = [float(np.mean(dlt[prng.integers(0, len(dlt), len(dlt))])) for _ in range(2000)]
            nz = np.abs(dlt) > 1e-12
            try: wp = float(wilcoxon(dlt[nz]).pvalue) if nz.sum() >= 5 else float("nan")
            except Exception: wp = float("nan")
            e.update({"mean_diff_vs_TA": float(np.mean(dlt)), "task_block_ci95": [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                      "pair_boot_ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))],
                      "win": int(np.sum(dlt > 1e-12)), "tie": int(np.sum(~nz)), "loss": int(np.sum(dlt < -1e-12)), "wilcoxon_p": wp})
        MC[k] = e
    return MC


def signflip(g, nrep=10000, seed=0):                      # verbatim e3.signflip
    rng = np.random.default_rng(seed); g = np.asarray(g, float); obs = g.mean()
    fl = rng.choice([-1.0, 1.0], size=(nrep, len(g)))
    perm = (fl * g).mean(1)
    p1 = (1 + np.sum(perm >= obs - 1e-12)) / (nrep + 1)
    p2 = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (nrep + 1)
    return float(p1), float(p2)


def contrast(d, nrep=2000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    bs = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(nrep)]
    p1, p2 = signflip(d)
    return {"mean": float(d.mean()), "median": float(np.median(d)), "ci95_boot_over_tasks": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "signflip_p_two_sided": p2, "win": int(np.sum(d > 1e-12)), "tie": int(np.sum(np.abs(d) <= 1e-12)), "loss": int(np.sum(d < -1e-12)), "n": int(len(d))}


def desc(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)), "min": float(x.min()), "max": float(x.max())}


def load_pop(pop, PR):
    import pandas as pd
    jl = HERE / f"pair_results_{pop}.jsonl"
    if not jl.exists(): return None
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    R = pd.DataFrame(recs)
    df = PR.merge(R.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")
    return df


def main():
    import pandas as pd
    t0 = time.time()
    PO = json.loads((HERE / "populations_e1c.json").read_text())
    tasks = PO["valid_tasks"]
    hash_ok = all(bm.sha256(HERE / fn) == h for h, fn in (l.split() for l in (HERE / "predictors_e1c.sha256").read_text().strip().splitlines()))
    PR = pd.read_csv(HERE / "predictors_e1c.csv")
    LR = pd.read_csv(HERE / "predictors_layers_e1c.csv")
    NUL = json.loads((HERE / "predictors_null_e1c.json").read_text())
    A = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "valid_tasks": tasks, "K": len(tasks), "excluded_seed1": PO["excluded_seed1"],
         "predictors_hash_ok": hash_ok, "populations_n_expected": PO["n"], "seed0_recheck_ok": PO["seed0_recheck_ok"]}
    cols = list(PRIM.values()) + SECONDARY
    # ---- analysis-code equivalence on E1b R0 (must reproduce E1b analysis.json)
    Pr0 = pd.read_csv(E1B / "predictors.csv"); Rr0 = pd.read_csv(E1B / "pair_results.csv")
    r0 = Pr0.merge(Rr0.drop(columns=["t1", "t2"]), on="pair"); r0["t1_task"] = r0.t1; r0["t2_task"] = r0.t2
    e1b_tasks = json.loads((E1B / "stage0.json").read_text())["valid_tasks"]
    S_r0 = pop_stats(r0, e1b_tasks, cols)
    AE = json.loads((E1B / "analysis.json").read_text())
    eq = {}
    for h, h0 in (("H1c", "H1"), ("H2c", "H2")):
        a, b = S_r0["verdicts"][h], AE["verdicts"][h0]
        eq[h0] = {"rho_diff": abs(a["rho"] - b["rho"]), "ci_diff": max(abs(a["boot_ci95"][i] - b["boot_ci95"][i]) for i in (0, 1)),
                  "p_diff": abs(a["perm_p_one_sided"] - b["perm_p_one_sided"]), "holm_diff": abs(a["holm_p"] - b["holm_p"]), "verdict_equal": a["verdict"] == b["verdict"]}
    A["analysis_equivalence_on_E1b"] = {"per_hypothesis": eq, "ok": all(v["rho_diff"] < 1e-12 and v["ci_diff"] < 1e-12 and v["p_diff"] < 1e-12 and v["verdict_equal"] for v in eq.values())}
    bm.log(f"e1c stage3: analysis equivalence on E1b R0: {A['analysis_equivalence_on_E1b']}")
    # ---- populations
    dfs = {p: load_pop(p, PR) for p in ("P", "S1", "S2")}
    res = {}
    for p in ("P", "S1"):
        df = dfs[p]
        if df is None or len(df) != PO["n"][p]:
            res[p] = {"status": f"not complete ({0 if df is None else len(df)}/{PO['n'][p]} pairs)"}; continue
        S = pop_stats(df, tasks, cols)
        fixed = {c: {f"lam{l}": spearman(df[c], df[f"TA_eval_D_lam{l}"]) for l in LAMS} for c in PRIM.values()}
        res[p] = {"status": "complete", "verdicts": S["verdicts"], "all_predictor_stats": S["all_predictor_stats"], "fixed_lambda_rho": fixed,
                  "method_comparison": method_comp(df, S["boots"]), "gate_active_pairs": int(df.gate_active.sum()),
                  "bootstrap_reps_used": len(S["boots"]), "bootstrap_skipped_degenerate": S["boot_skipped"],
                  "D_summary": desc(df.D), "lam_selected_counts": {str(k): int(v) for k, v in df.lam_selected.value_counts().sort_index().items()},
                  "rho_OA_tvcos": spearman(df.O_A, df.tv_cosine), "min_theta_min_overall_deg": float(df.min_theta_min_A_deg.min())}
        bm.log(f"e1c stage3 {p}: " + ", ".join(f"{h} {v['verdict']} rho={v['rho']:.4f} ci={v['boot_ci95']} holm={v['holm_p']:.4f} loto={v['loto_frac_gt_0.2']:.2f}" for h, v in S["verdicts"].items()))
    A["P"] = res["P"]; A["S1"] = res["S1"]
    if A["P"].get("status") == "complete":
        A["verdicts_primary"] = {h: {k: v[k] for k in ("predictor", "verdict", "rho", "boot_ci95", "perm_p_one_sided", "holm_p", "loto_frac_gt_0.2", "criteria")} for h, v in A["P"]["verdicts"].items()}
    A["R0_recomputed"] = {h: {k: v[k] for k in ("verdict", "rho", "boot_ci95", "holm_p", "loto_frac_gt_0.2")} for h, v in S_r0["verdicts"].items()}
    # ---- reliability across seed configurations (same unordered task pair)
    rel = {}
    tabs = {"R0": r0.assign(u=[upair(x, y) for x, y in zip(r0.t1_task, r0.t2_task)]).set_index("u")}
    for p in ("P", "S1"):
        if dfs[p] is not None: tabs[p] = dfs[p].assign(u=[upair(x, y) for x, y in zip(dfs[p].t1_task, dfs[p].t2_task)]).set_index("u")
    for c in ("D", "O_A", "tv_cosine", "lam_selected"):
        for x, y in itertools.combinations(tabs, 2):
            common = tabs[x].index.intersection(tabs[y].index)
            rel[f"{c}:{x}~{y}"] = {"rho": spearman(tabs[x].loc[common, c], tabs[y].loc[common, c]), "n": int(len(common))}
    A["reliability"] = rel
    # ---- O_A shared vs mixed
    from scipy.stats import wilcoxon
    oa = {"null_O_A_mean": NUL["null_O_A_mean"], "null_O_A_sd": NUL["null_O_A_sd"], "null_O_A_p95": NUL["null_O_A_p95"]}
    popsets = {"R0 (shared seed 0, E1b)": r0, "S1 (shared seed 1)": PR[PR.pair.isin(PO["populations"]["S1"])],
               "P (mixed seed)": PR[PR.pair.isin(PO["populations"]["P"])], "S2 (same task, mixed seed)": PR[PR.pair.isin(PO["populations"]["S2"])]}
    for name, d in popsets.items():
        oa[name] = {c: desc(d[c]) for c in ("O_A", "null_z_O_A", "tv_cosine", "min_theta_min_A_deg", "mean_theta_min_A_deg", "O_B")}
        oa[name]["frac_O_A_above_null_p95"] = float(np.mean(d.O_A.values > NUL["null_O_A_p95"]))
        oa[name]["n_layers_lt30_total"] = int(d.n_layers_theta_min_lt30_A.sum())
    pu = {upair(x, y): v for x, y, v in zip(popsets["P (mixed seed)"].t1_task, popsets["P (mixed seed)"].t2_task, popsets["P (mixed seed)"].O_A)}
    for ref, d in (("R0", r0), ("S1", popsets["S1 (shared seed 1)"])):
        ru = {upair(x, y): v for x, y, v in zip(d.t1_task if "t1_task" in d else d.t1, d.t2_task if "t2_task" in d else d.t2, d.O_A)}
        ks = sorted(set(pu) & set(ru)); diff = np.array([pu[k] - ru[k] for k in ks])
        oa[f"paired_P_minus_{ref}"] = {"n": len(ks), "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)), "frac_P_lower": float(np.mean(diff < 0)),
                                        "wilcoxon_p": float(wilcoxon(diff).pvalue), "ratio_mean": float(np.mean([pu[k] for k in ks]) / np.mean([ru[k] for k in ks]))}
    A["O_A_shared_vs_mixed"] = oa
    # ---- S2 same-task seed pairs
    s2 = {}
    d2 = dfs["S2"]
    L2 = LR[LR.pair.isin(PO["populations"]["S2"])]
    per = []
    for pr in PO["populations"]["S2"]:
        lp = L2[L2.pair == pr]; row = PR[PR.pair == pr].iloc[0]
        e = {"pair": pr, "task": task_of(pr.split("__")[0]), "min_theta_min_deg": float(lp.theta_min_A_deg.min()),
             "argmin_layer": lp.loc[lp.theta_min_A_deg.idxmin(), "layer"], "n_layers_lt30": int((lp.theta_min_A_deg < 30).sum()),
             "layers_lt30": {r.layer: round(float(r.theta_min_A_deg), 2) for r in lp[lp.theta_min_A_deg < 30].itertuples()},
             "n_layers_lt45": int((lp.theta_min_A_deg < 45).sum()), "mean_theta_min_deg": float(lp.theta_min_A_deg.mean()),
             "O_A": float(row.O_A), "null_z_O_A": float(row.null_z_O_A), "tv_cosine": float(row.tv_cosine)}
        if d2 is not None and pr in set(d2.pair):
            r = d2[d2.pair == pr].iloc[0]
            e.update({"D": float(r.D), "lam_selected": float(r.lam_selected), "TA_score_sel": float(r.TA_score_sel), "gate_active": bool(r.gate_active),
                      "gate_n_fail_layers": int(r.gate_n_fail_layers), "GATE_atTA_score": float(r.GATE_atTA_score), "GATE_own_score": float(r.GATE_own_score),
                      "lam_GATE_own": float(r.lam_GATE_own), "TIES_atTA_score": float(r.TIES_atTA_score), "TIES_own_score": float(r.TIES_own_score)})
        per.append(e)
    s2["per_pair"] = per
    s2["n_pairs_gate_fires"] = int(sum(e["n_layers_lt30"] > 0 for e in per))
    s2["theta_min_over_all_S2"] = float(min(e["min_theta_min_deg"] for e in per))
    if d2 is not None and len(d2) == PO["n"]["S2"]:
        s2["e1b_stage2_gate_vs_TA"] = {"gate@lam*TA": contrast(d2.GATE_atTA_score - d2.TA_score_sel), "gate@own": contrast(d2.GATE_own_score - d2.TA_score_sel)}
        s2["D_summary"] = desc(d2.D)
    e3p = HERE / "e3same" / "results.jsonl"
    if e3p.exists():
        E = [json.loads(l) for l in e3p.read_text().splitlines() if l.strip()]
        rows = []
        for r in E:
            row = {"pair": r["pair"].replace("-", "__", 1) if "__" not in r["pair"] else r["pair"]}
            for m, v in r["methods"].items():
                row[m] = v["score"]; row[m + "_sel"] = json.dumps(v["sel_hp"]); row[m + "_boundary"] = v["at_boundary"]
            row["GATE_n_fail_layers"] = r["methods"]["GATE"]["info"].get("gate_n_fail_layers")
            row["TA4_score"] = r["extra"].get("TA4_score"); row["PICO_TA_c1_score"] = r["extra"].get("PICO_TA_c1_score")
            # sanity: E3 TA at the E1b lambdas == e1c stage2 S2 TA_eval values
            if d2 is not None and row["pair"] in set(d2.pair):
                q = d2[d2.pair == row["pair"]].iloc[0]; t1, t2 = r["t1"], r["t2"]; mx = 0.0
                for lam in LAMS:
                    ev = r["extra"][f"TA_eval_lam{lam}"]
                    mx = max(mx, abs(ev[t1] - q[f"TA_eval_t1_lam{lam}"]), abs(ev[t2] - q[f"TA_eval_t2_lam{lam}"]))
                row["sanity_TA_maxdiff_vs_e1c_stage2"] = mx
            rows.append(row)
        T = pd.DataFrame(rows)
        T.to_csv(HERE / "e3same" / "method_by_pair_S2.csv", index=False, float_format="%.10g")
        con = {}
        for x, y in (("GATE", "TA"), ("FORCEGATE", "TA"), ("PICO_TA", "TA"), ("GATE", "PICO_TA"), ("FORCEGATE", "PICO_TA"), ("PICO_TA_c1_score", "TA"), ("TA4_score", "TA")):
            con[f"{x}-{y}"] = contrast(T[x].values - T[y].values)
        fire = T[T.GATE_n_fail_layers.fillna(0) > 0]
        g = (fire.GATE - fire.TA).values
        s2["e3_subset"] = {"n_pairs": int(len(T)), "mean_scores": {m: float(T[m].mean()) for m in ("TA", "PICO_TA", "GATE", "FORCEGATE", "TA4_score", "PICO_TA_c1_score")},
                           "contrasts": con, "boundary_rate": {m: float(T[m + "_boundary"].mean()) for m in ("TA", "PICO_TA", "GATE", "FORCEGATE")},
                           "gate_vs_TA_on_fail_pairs": {"n": int(len(fire)), "per_pair": dict(zip(fire.pair, map(float, g))),
                                                         **({"mean": float(g.mean()), "signflip_p_two_sided": signflip(g)[1]} if len(fire) >= 5 else {})},
                           "sanity_TA_reproduces_e1c_stage2": float(T.get("sanity_TA_maxdiff_vs_e1c_stage2", pd.Series([np.nan])).max()),
                           "per_pair": T.to_dict(orient="records")}
    A["S2"] = s2
    tm = {}
    for f in HERE.glob("timing_*.json"): tm[f.stem] = json.loads(f.read_text())
    A["timing"] = tm
    bm.jdump(A, HERE / "analysis_e1c.json")
    figures(A, dfs, r0, PR, LR, NUL, PO)
    bm.log(f"e1c stage3 done in {time.time() - t0:.0f}s")


def figures(A, dfs, r0, PR, LR, NUL, PO):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, re
    # F1: scatter P (primary) and S1
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row, p in enumerate(("P", "S1")):
        df = dfs[p]
        if df is None or A[p].get("status") != "complete": continue
        for col, (h, c, lab) in enumerate((("H1c", "O_A", "O_A (layer-mean mean cos² principal angles, orth B)"), ("H2c", "tv_cosine", "task-vector cosine"))):
            ax = axes[row, col]; same = df.same_type.values
            ax.scatter(df[c][~same], df.D[~same], s=22, c="tab:blue", label="cross type"); ax.scatter(df[c][same], df.D[same], s=30, c="tab:red", label="same type group")
            for r in df.itertuples():
                if r.D > np.percentile(df.D, 90): ax.annotate(f"{r.t1_task}-{r.t2_task}", (getattr(r, c), r.D), fontsize=6, xytext=(2, 2), textcoords="offset points")
            v = A[p]["verdicts"][h]; tag = "PRIMARY (mixed seed)" if p == "P" else "secondary (seed-1 only; rule descriptive)"
            ax.set_title(f"{p} {tag}\n{h if p == 'P' else h.replace('c', '')}: ρ={v['rho']:.3f}, CI [{v['boot_ci95'][0]:.2f}, {v['boot_ci95'][1]:.2f}], Holm p={v['holm_p']:.3g} → {v['verdict']}", fontsize=9)
            ax.set_xlabel(lab); ax.set_ylabel("D (TA at held-out-selected λ)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "fig_e1c_scatter.png", dpi=140); plt.close(fig)
    # F2: O_A shared vs mixed
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    sets = [("R0 shared s0", r0.O_A.values), ("S1 shared s1", PR[PR.pair.isin(PO["populations"]["S1"])].O_A.values),
            ("P mixed", PR[PR.pair.isin(PO["populations"]["P"])].O_A.values), ("S2 same-task mixed", PR[PR.pair.isin(PO["populations"]["S2"])].O_A.values)]
    a1.boxplot([s[1] for s in sets], showfliers=True); a1.set_xticks(range(1, 5)); a1.set_xticklabels([s[0] for s in sets], fontsize=9)
    for i, s in enumerate(sets): a1.scatter(np.full(len(s[1]), i + 1) + np.random.default_rng(0).uniform(-.15, .15, len(s[1])), s[1], s=8, alpha=.5)
    a1.axhspan(NUL["null_O_A_p5"], NUL["null_O_A_p95"], color="gray", alpha=.3, label="random rank-8 null 5–95%"); a1.axhline(NUL["null_O_A_mean"], c="k", lw=.7)
    a1.set_yscale("log"); a1.set_ylabel("O_A (log)"); a1.legend(fontsize=8); a1.set_title("O_A: shared-seed vs mixed-seed pairs")
    pu = {upair(x, y): v for x, y, v in zip(dfs["P"].t1_task, dfs["P"].t2_task, dfs["P"].O_A)} if dfs["P"] is not None else {}
    ru = {upair(x, y): v for x, y, v in zip(r0.t1_task, r0.t2_task, r0.O_A)}
    ks = sorted(set(pu) & set(ru))
    if ks:
        a2.scatter([ru[k] for k in ks], [pu[k] for k in ks], s=14); lim = [min(min(ru.values()), min(pu.values())) * 0.8, max(ru.values()) * 1.2]
        a2.plot(lim, lim, "k:", lw=.8); a2.set_xscale("log"); a2.set_yscale("log"); a2.axhline(NUL["null_O_A_p95"], c="gray", ls="--", lw=.8, label="null p95")
        a2.set_xlabel("O_A, E1b shared seed 0 (R0)"); a2.set_ylabel("O_A, mixed seed (P)"); a2.set_title("Same task pair: shared vs mixed seed"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "fig_e1c_OA_shared_vs_mixed.png", dpi=140); plt.close(fig)
    # F3: S2 per-layer theta_min heatmap
    L2 = LR[LR.pair.isin(PO["populations"]["S2"])]
    sub = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
    def lkey(n):
        m = re.search(r"layer\.(\d+)\.", n)
        if m: rest = n.split(f"layer.{m.group(1)}.")[1]; return (int(m.group(1)), sub.index(rest) if rest in sub else 9)
        return (99, 0)
    layers = sorted(L2.layer.unique(), key=lkey); prs = PO["populations"]["S2"]
    H = L2.pivot(index="layer", columns="pair", values="theta_min_A_deg").loc[layers, prs].values
    fig, ax = plt.subplots(figsize=(8, 14))
    im = ax.imshow(H, aspect="auto", cmap="viridis_r", vmin=0, vmax=90); yy, xx = np.where(H < 30)
    ax.scatter(xx, yy, marker="s", s=18, facecolors="none", edgecolors="red", linewidths=0.8)
    ax.set_xticks(range(len(prs))); ax.set_xticklabels([task_of(p.split("__")[0]) for p in prs], rotation=90, fontsize=8)
    short = [n.replace("encoder.layer.", "L").replace("attention.self.", "").replace("attention.output.dense", "attn_out").replace("intermediate.dense", "inter").replace("output.dense", "out") for n in layers]
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(short, fontsize=5); plt.colorbar(im, ax=ax, fraction=0.04, label="θ_min (deg)")
    ax.set_title("S2 same-task seed pairs (s0 × s1): θ_min per layer (red: < 30°)", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_e1c_S2_theta.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
