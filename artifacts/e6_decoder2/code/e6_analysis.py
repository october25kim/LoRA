#!/usr/bin/env python3
"""E6 stage3: pre-registered analysis (PREREG_E6.md sec. 6) = E4b e4b_analysis.py with TAG e6 (statistics verbatim from E1c pop_stats:
perms / task-block bootstrap / LOTO / Holm / rule; method_comp; signflip; contrast; desc), parameterized by population.
Populations: P (primary), R0, S1 (secondary, rule descriptive), pooled (exploratory), S2. No E3 subset in E6.
Cross-backbone comparison with E1b/E1c (BERT), E4a (RoBERTa) and, new in E6, E4b (Qwen2.5-0.5B) (read-only). Run via:  python e6.py --stage stage3"""
import json, math, sys, time, itertools
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TAG = "e6"
E1B = HERE.parent / "e1b_confirmatory"
E1C = HERE.parent / "e1c_seed_diverse"
E4A = HERE.parent / "e4a_roberta"
E4B = HERE.parent / "e4b_decoder"                  # E6: Qwen2.5-0.5B reference (read-only)
sys.path.insert(0, str(E1B))
import e1b as bm  # noqa: E402
bm.LOGF = HERE / f"run_{TAG}.log"; bm.OUT = HERE
spearman = bm.spearman
PRIM = {"H1": "O_A", "H2": "tv_cosine"}
SECONDARY = ["mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A", "sign_conflict_top20",
             "norm_ratio", "null_z_O_A", "sign_conflict_all"]
LAMS = [0.3, 0.5, 0.7, 1.0]


def task_of(k): return k.split("@")[0]
def upair(a, b): return tuple(sorted((a, b)))


# ---------------------------------------------------------------- verbatim E1c pop_stats (= e1b.stage3 statistics), parameterized
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
            tb = [float(np.mean(dlt[bi])) for bi in boots] if boots else [float(np.mean(dlt[prng.integers(0, len(dlt), len(dlt))])) for _ in range(2000)]
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
    recs, seen = [], set()
    for l in jl.read_text().splitlines():                  # de-duplicate (two processes may race on the last S1 pair); keep first
        if not l.strip(): continue
        r = json.loads(l)
        if r["pair"] in seen: continue
        seen.add(r["pair"]); recs.append(r)
    R = pd.DataFrame(recs)
    order = {p: i for i, p in enumerate(PR.pair)}
    df = PR.merge(R.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")
    return df.assign(_o=[order[p] for p in df.pair]).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def main():
    import pandas as pd
    t0 = time.time()
    PO = json.loads((HERE / f"populations_{TAG}.json").read_text())
    tasks = PO["valid_tasks"]
    hash_ok = all(bm.sha256(HERE / fn) == h for h, fn in (l.split() for l in (HERE / f"predictors_{TAG}.sha256").read_text().strip().splitlines()))
    PR = pd.read_csv(HERE / f"predictors_{TAG}.csv")
    LR = pd.read_csv(HERE / f"predictors_layers_{TAG}.csv")
    NUL = json.loads((HERE / f"predictors_null_{TAG}.json").read_text())
    A = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "valid_tasks": tasks, "K": len(tasks), "excluded": PO["excluded"],
         "predictors_hash_ok": hash_ok, "populations_n_expected": PO["n"]}
    cols = list(PRIM.values()) + SECONDARY
    # ---- analysis-code equivalence on E1b R0 (must reproduce E1b analysis.json H1/H2)
    try:
        Pr0 = pd.read_csv(E1B / "predictors.csv"); Rr0 = pd.read_csv(E1B / "pair_results.csv")
        r0b = Pr0.merge(Rr0.drop(columns=["t1", "t2"]), on="pair"); r0b["t1_task"] = r0b.t1; r0b["t2_task"] = r0b.t2
        S_r0 = pop_stats(r0b, json.loads((E1B / "stage0.json").read_text())["valid_tasks"], ["O_A", "tv_cosine"])
        AE = json.loads((E1B / "analysis.json").read_text()); eq = {}
        for h in ("H1", "H2"):
            a, b = S_r0["verdicts"][h], AE["verdicts"][h]
            eq[h] = {"rho_diff": abs(a["rho"] - b["rho"]), "ci_diff": max(abs(a["boot_ci95"][i] - b["boot_ci95"][i]) for i in (0, 1)),
                     "p_diff": abs(a["perm_p_one_sided"] - b["perm_p_one_sided"]), "verdict_equal": a["verdict"] == b["verdict"]}
        A["analysis_equivalence_on_E1b"] = {"per_hypothesis": eq, "ok": all(v["rho_diff"] < 1e-12 and v["ci_diff"] < 1e-12 and v["p_diff"] < 1e-12 and v["verdict_equal"] for v in eq.values())}
    except Exception as ex:
        r0b = None; A["analysis_equivalence_on_E1b"] = {"ok": False, "error": repr(ex)}
    bm.log(f"{TAG} stage3: analysis equivalence on E1b R0: {A['analysis_equivalence_on_E1b']}")
    # ---- populations
    dfs = {p: load_pop(p, PR) for p in ("P", "R0", "S1", "S2")}
    res = {}
    for p in ("P", "R0", "S1"):
        df = dfs[p]
        if p not in PO["n"]:
            res[p] = {"status": "not in design"}; continue
        if df is None or len(df) != PO["n"][p]:
            res[p] = {"status": f"not complete ({0 if df is None else len(df)}/{PO['n'][p]} pairs) -> not run / not reported as evidence"}; continue
        S = pop_stats(df, tasks, cols)
        fixed = {c: {f"lam{l}": spearman(df[c], df[f"TA_eval_D_lam{l}"]) for l in LAMS} for c in PRIM.values()}
        res[p] = {"status": "complete", "verdicts": S["verdicts"], "all_predictor_stats": S["all_predictor_stats"], "fixed_lambda_rho": fixed,
                  "method_comparison": method_comp(df, S["boots"]), "gate_active_pairs": int(df.gate_active.sum()),
                  "bootstrap_reps_used": len(S["boots"]), "bootstrap_skipped_degenerate": S["boot_skipped"],
                  "D_summary": desc(df.D), "lam_selected_counts": {str(k): int(v) for k, v in df.lam_selected.value_counts().sort_index().items()},
                  "rho_OA_tvcos": spearman(df.O_A, df.tv_cosine), "min_theta_min_overall_deg": float(df.min_theta_min_A_deg.min())}
        bm.log(f"{TAG} stage3 {p}: " + ", ".join(f"{h} {v['verdict']} rho={v['rho']:.4f} ci={v['boot_ci95']} holm={v['holm_p']:.4f} loto={v['loto_frac_gt_0.2']:.2f}" for h, v in S["verdicts"].items()))
    for p in ("P", "R0", "S1"): A[p] = res[p]
    PRIM_POP = PO.get("primary_population", "P"); A["primary_population"] = PRIM_POP
    if A[PRIM_POP].get("status") == "complete":
        A["verdicts_primary"] = {h: {k: v[k] for k in ("predictor", "verdict", "rho", "boot_ci95", "perm_p_one_sided", "holm_p", "loto_frac_gt_0.2", "criteria")} for h, v in A[PRIM_POP]["verdicts"].items()}
    # ---- pooled 3-population exploratory summary (mean over R0, S1, P per unordered task pair)
    tabs = {}
    for p in ("R0", "S1", "P"):
        if dfs[p] is not None and res[p].get("status") == "complete":
            tabs[p] = dfs[p].assign(u=[upair(x, y) for x, y in zip(dfs[p].t1_task, dfs[p].t2_task)]).set_index("u")
    if len(tabs) == 3:
        common = sorted(set(tabs["R0"].index) & set(tabs["S1"].index) & set(tabs["P"].index))
        pooled = pd.DataFrame({"t1_task": [u[0] for u in common], "t2_task": [u[1] for u in common]})
        for c in ["D"] + cols:
            pooled[c] = np.mean([tabs[p].loc[common, c].values.astype(float) for p in tabs], axis=0)
        # task order as in the valid list
        S = pop_stats(pooled, tasks, cols)
        A["pooled_3pop"] = {"status": "complete (exploratory)", "n_pairs": len(pooled), "verdicts_descriptive": S["verdicts"], "all_predictor_stats": S["all_predictor_stats"],
                            "D_summary": desc(pooled.D)}
        bm.log(f"{TAG} stage3 pooled: " + ", ".join(f"{h} {v['verdict']} rho={v['rho']:.4f} ci={v['boot_ci95']}" for h, v in S["verdicts"].items()))
    else:
        A["pooled_3pop"] = {"status": f"not computed (complete populations: {sorted(tabs)})"}
    # ---- reliability across seed configurations
    rel = {}
    for c in ("D", "O_A", "tv_cosine", "lam_selected"):
        for x, y in itertools.combinations(sorted(tabs), 2):
            cm = tabs[x].index.intersection(tabs[y].index)
            rel[f"{c}:{x}~{y}"] = {"rho": spearman(tabs[x].loc[cm, c], tabs[y].loc[cm, c]), "n": int(len(cm))}
    A["reliability"] = rel
    # ---- O_A shared vs mixed; theta_min distribution
    from scipy.stats import wilcoxon
    oa = {"null_O_A_mean": NUL["null_O_A_mean"], "null_O_A_sd": NUL["null_O_A_sd"], "null_O_A_p95": NUL["null_O_A_p95"]}
    popsets = {k: PR[PR.pair.isin(PO["populations"][k])] for k in ("R0", "S1", "P", "S2") if k in PO["populations"]}
    labels = {"R0": "R0 (shared seed 0)", "S1": "S1 (shared seed 1)", "P": "P (mixed seed)", "S2": "S2 (same task, mixed seed)"}
    for k, d in popsets.items():
        e = {c: desc(d[c]) for c in ("O_A", "null_z_O_A", "tv_cosine", "min_theta_min_A_deg", "mean_theta_min_A_deg")}
        e["frac_O_A_above_null_p95"] = float(np.mean(d.O_A.values > NUL["null_O_A_p95"]))
        e["n_layers_lt30_total"] = int(d.n_layers_theta_min_lt30_A.sum())
        e["n_pairs_gate_would_fire"] = int((d.n_layers_theta_min_lt30_A > 0).sum())
        Ld = LR[LR.pair.isin(PO["populations"][k])]
        th = Ld.theta_min_A_deg.values
        e["theta_min_all_layers"] = {"n": int(len(th)), "min": float(th.min()), "p1": float(np.percentile(th, 1)), "p5": float(np.percentile(th, 5)),
                                     "median": float(np.median(th)), "n_lt30": int((th < 30).sum()), "n_lt45": int((th < 45).sum()), "n_lt60": int((th < 60).sum())}
        oa[labels[k]] = e
    pu = {upair(x, y): v for x, y, v in zip(popsets["P"].t1_task, popsets["P"].t2_task, popsets["P"].O_A)} if "P" in popsets else {}
    for ref in [r for r in ("R0", "S1") if r in popsets]:
        d = popsets[ref]; ru = {upair(x, y): v for x, y, v in zip(d.t1_task, d.t2_task, d.O_A)}
        ks = sorted(set(pu) & set(ru)); diff = np.array([pu[k] - ru[k] for k in ks])
        if len(ks) >= 5:
            oa[f"paired_P_minus_{ref}"] = {"n": len(ks), "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)), "frac_P_lower": float(np.mean(diff < 0)),
                                            "wilcoxon_p": float(wilcoxon(diff).pvalue), "ratio_mean": float(np.mean([pu[k] for k in ks]) / np.mean([ru[k] for k in ks]))}
    # per-layer-type theta_min (which module types come closest)
    LR2 = LR.copy(); LR2["ltype"] = LR2.layer.str.replace(r"^.*layers?\.\d+\.", "", regex=True)
    oa["theta_min_by_layer_type"] = {lt: desc(g.theta_min_A_deg) for lt, g in LR2.groupby("ltype")}
    A["O_A_and_theta"] = oa
    # ---- S2 same-task seed pairs
    s2 = {}; d2 = dfs["S2"]; L2 = LR[LR.pair.isin(PO["populations"].get("S2", []))]; per = []
    for pr in PO["populations"].get("S2", []):
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
    s2["theta_min_over_all_S2"] = float(min(e["min_theta_min_deg"] for e in per)) if per else float("nan")
    if d2 is not None and len(d2) == PO["n"].get("S2", -1):
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
            if d2 is not None and row["pair"] in set(d2.pair):
                q = d2[d2.pair == row["pair"]].iloc[0]; t1, t2 = r["t1"], r["t2"]; mx = 0.0
                for lam in LAMS:
                    ev = r["extra"][f"TA_eval_lam{lam}"]
                    mx = max(mx, abs(ev[t1] - q[f"TA_eval_t1_lam{lam}"]), abs(ev[t2] - q[f"TA_eval_t2_lam{lam}"]))
                row["sanity_TA_maxdiff_vs_stage2"] = mx
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
                           "sanity_TA_reproduces_stage2": float(T["sanity_TA_maxdiff_vs_stage2"].max()) if "sanity_TA_maxdiff_vs_stage2" in T else float("nan"),
                           "per_pair": T.to_dict(orient="records")}
    A["S2"] = s2
    # ---- cross-backbone (exploratory): RoBERTa vs BERT on common valid tasks
    cb = {}
    try:
        if r0b is not None and "R0" in tabs:
            b = r0b.assign(u=[upair(x, y) for x, y in zip(r0b.t1_task, r0b.t2_task)]).set_index("u")
            cm = b.index.intersection(tabs["R0"].index)
            for c in ("D", "O_A", "tv_cosine"):
                cb[f"{c}: E1b R0 (BERT) ~ {TAG} R0"] = {"rho": spearman(b.loc[cm, c], tabs["R0"].loc[cm, c]), "n": int(len(cm))}
        pp = E1C / "predictors_e1c.csv"; jp = E1C / "pair_results_P.jsonl"
        if pp.exists() and jp.exists() and "P" in tabs:
            pc = pd.read_csv(pp); rc = pd.DataFrame([json.loads(l) for l in jp.read_text().splitlines() if l.strip()])
            c1 = pc.merge(rc.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")
            c1 = c1.assign(u=[upair(x, y) for x, y in zip(c1.t1_task, c1.t2_task)]).set_index("u")
            cm = c1.index.intersection(tabs["P"].index)
            for c in ("D", "O_A", "tv_cosine"):
                cb[f"{c}: E1c P (BERT) ~ {TAG} P"] = {"rho": spearman(c1.loc[cm, c], tabs["P"].loc[cm, c]), "n": int(len(cm))}
    except Exception as ex:
        cb["error"] = repr(ex)
    try:                                                          # E4a (RoBERTa) populations, if complete
        pa = E4A / "predictors_e4a.csv"
        if pa.exists():
            pra = pd.read_csv(pa)
            for p in ("R0", "P"):
                ja = E4A / f"pair_results_{p}.jsonl"
                if ja.exists() and p in tabs:
                    ra = pd.DataFrame([json.loads(l) for l in ja.read_text().splitlines() if l.strip()]).drop_duplicates("pair")
                    c4 = pra.merge(ra.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")
                    c4 = c4.assign(u=[upair(x, y) for x, y in zip(c4.t1_task, c4.t2_task)]).set_index("u")
                    cm = c4.index.intersection(tabs[p].index)
                    for c in ("D", "O_A", "tv_cosine"):
                        cb[f"{c}: E4a {p} (RoBERTa) ~ {TAG} {p}"] = {"rho": spearman(c4.loc[cm, c], tabs[p].loc[cm, c]), "n": int(len(cm))}
    except Exception as ex:
        cb["error_e4a"] = repr(ex)
    try:                                                          # E6: E4b (Qwen2.5-0.5B) populations, same task pairs and seed roles
        pb = E4B / "predictors_e4b.csv"
        if pb.exists():
            prb = pd.read_csv(pb)
            for p in ("P", "R0", "S1"):
                jb = E4B / f"pair_results_{p}.jsonl"
                if jb.exists() and p in tabs:
                    rb = pd.DataFrame([json.loads(l) for l in jb.read_text().splitlines() if l.strip()]).drop_duplicates("pair")
                    c4 = prb.merge(rb.drop(columns=["t1", "t2", "t1_task", "t2_task"]), on="pair")
                    c4 = c4.assign(u=[upair(x, y) for x, y in zip(c4.t1_task, c4.t2_task)]).set_index("u")
                    cm = c4.index.intersection(tabs[p].index)
                    for c in ("D", "O_A", "tv_cosine", "lam_selected"):
                        cb[f"{c}: E4b {p} (Qwen2.5-0.5B) ~ {TAG} {p}"] = {"rho": spearman(c4.loc[cm, c], tabs[p].loc[cm, c]), "n": int(len(cm))}
    except Exception as ex:
        cb["error_e4b"] = repr(ex)
    A["cross_backbone"] = cb
    tm = {}
    for f in HERE.glob("timing_*.json"): tm[f.stem] = json.loads(f.read_text())
    A["timing"] = tm
    bm.jdump(A, HERE / f"analysis_{TAG}.json")
    try:
        figures(A, dfs, PR, LR, NUL, PO)
    except Exception as ex:
        bm.log(f"{TAG} figures failed: {ex!r}")
    bm.log(f"{TAG} stage3 done in {time.time() - t0:.0f}s")


def figures(A, dfs, PR, LR, NUL, PO):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pops = [p for p in ("P", "R0", "S1") if A[p].get("status") == "complete"]
    if pops:
        fig, axes = plt.subplots(len(pops), 2, figsize=(14, 5.2 * len(pops)), squeeze=False)
        for row, p in enumerate(pops):
            df = dfs[p]
            for col, (h, c, lab) in enumerate((("H1", "O_A", "O_A (layer-mean mean cos² principal angles, orth B)"), ("H2", "tv_cosine", "task-vector cosine"))):
                ax = axes[row, col]; same = df.same_type.values
                ax.scatter(df[c][~same], df.D[~same], s=22, c="tab:blue", label="cross type"); ax.scatter(df[c][same], df.D[same], s=30, c="tab:red", label="same type group")
                for r in df.itertuples():
                    if r.D > np.percentile(df.D, 90): ax.annotate(f"{r.t1_task}-{r.t2_task}", (getattr(r, c), r.D), fontsize=6, xytext=(2, 2), textcoords="offset points")
                v = A[p]["verdicts"][h]; tagp = "PRIMARY (mixed seed)" if p == "P" else "secondary (rule descriptive)"
                ax.set_title(f"{TAG} {p} {tagp}\n{h}: ρ={v['rho']:.3f}, CI [{v['boot_ci95'][0]:.2f}, {v['boot_ci95'][1]:.2f}], Holm p={v['holm_p']:.3g} → {v['verdict']}", fontsize=9)
                ax.set_xlabel(lab); ax.set_ylabel("D (TA at held-out-selected λ)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(HERE / f"fig_{TAG}_scatter.png", dpi=130); plt.close(fig)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    kk = [k for k in ("R0", "S1", "P", "S2") if k in PO["populations"]]
    sets = [(k, PR[PR.pair.isin(PO["populations"][k])].O_A.values) for k in kk]
    a1.boxplot([s[1] for s in sets]); a1.set_xticks(range(1, len(sets) + 1)); a1.set_xticklabels([s[0] for s in sets])
    a1.axhspan(NUL["null_O_A_p5"], NUL["null_O_A_p95"], color="gray", alpha=.3, label="random rank-8 null 5–95%"); a1.set_yscale("log"); a1.legend(fontsize=8)
    a1.set_title(f"{TAG}: O_A by population"); a1.set_ylabel("O_A (log)")
    th = [LR[LR.pair.isin(PO["populations"][k])].theta_min_A_deg.values for k in kk]
    a2.hist(th, bins=60, range=(0, 90), label=kk, histtype="step", density=True); a2.axvline(30, c="k", ls=":")
    a2.set_xlabel("θ_min per layer (deg)"); a2.set_title(f"{TAG}: θ_min distribution (all layers × pairs)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / f"fig_{TAG}_OA_theta.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
