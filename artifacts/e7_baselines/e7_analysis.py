#!/usr/bin/env python3
"""E7 analysis (EXPLORATORY / POST HOC). CPU only. Reads E6 cached results (read-only) + E7 jsonl files; writes summary_e7.json,
per-pair CSVs and RESULTS_E7.md into artifacts/e7_baselines/. Bootstrap = lam_confirm_e6.Boot (task-block, B=2000) verbatim for pairs;
an equivalent task-block bootstrap (weights = product of task counts) for 3/4-task tuples; pair-level iid bootstrap as secondary."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; E6D = HERE.parent / "e6_decoder2"; sys.path.insert(0, str(E6D))
import lam_confirm_e6 as LC  # noqa
G4, G7 = LC.G4, LC.G7; B = 2000


def jl(p):
    p = HERE / p
    if not p.exists(): return []
    out, seen = [], set()
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    return out


def first_argmax(M, grid): return np.array(grid)[np.argmax(np.round(M, 12), axis=1)]


def iid_ci(x, seed=7):
    r = np.random.default_rng(seed); n = len(x)
    bs = np.array([x[r.integers(0, n, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(c, d=2): return f"{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]"


# ================================================================== A: pairs
def part_a(S, bbtag="qwen15"):
    if bbtag == "qwen15":
        recs = {p: LC.jl(E6D / f"pair_results_{p}.jsonl") for p in ("R0", "S1", "P")}
        df = LC.build(recs, {}, [E6D / "preds"], E6D / "preds", "qwen15"); A1 = {r["id"]: r for r in jl("a1_entropy.jsonl")}; pre = "A"
    else:   # Qwen2.5-0.5B: E4b results + E5b extended lambdas (as lam_confirm_e6 --selftest-e4b, 4070 paths)
        if not jl("a1_entropy_q05.jsonl"): return None
        E4D = HERE.parent / "e4b_decoder"; E5B = HERE.parent / "e5_accept" / "e5b"
        recs = {p: LC.jl(E4D / f"pair_results_{p}.jsonl") for p in ("R0", "S1", "P")}; ext = {p: LC.jl(E5B / f"pair_ext_{p}.jsonl") for p in ("R0", "S1", "P")}
        df = LC.build(recs, ext, [E4D / "preds", E5B / "preds"], E4D / "preds", "qwen"); A1 = {r["id"]: r for r in jl("a1_entropy_q05.jsonl")}; pre = "A_q05"
    return _part_a(S, df, A1, pre, bbtag)


def _part_a(S, df, A1, pre, bbtag):
    have = df.pair.isin(A1.keys()).values
    S[f"{pre}_n_pairs_with_entropy"] = int(have.sum()); S[f"{pre}_n_pairs_total"] = len(df)
    df = df[have].reset_index(drop=True); N = len(df)
    if N == 0: return df
    BT = LC.Boot(df)
    has_stsb = ((df.task1 == "stsb") | (df.task2 == "stsb")).values
    def ent_mat(grid, norm=False):
        M = np.zeros((N, len(grid)))
        for i, p in enumerate(df.pair):
            r = A1[p]
            for c, l in enumerate(grid):
                v = [r[f"{'entn' if norm else 'ent'}{j}_{l}"] for j in (1, 2)]; v = [x for x in v if x is not None]
                M[i, c] = np.mean(v)
        return M
    # checks
    S[f"{pre}_checks"] = {"merged_cal_argmax_match_min_classif": float(min(A1[p][f"match{j}_{l}"] for p in df.pair for j in (1, 2) for l in G7
                                                                  if not (df.set_index("pair").loc[p, f"task{j}"] == "stsb"))),
                     "merged_cal_argmax_match_mean_classif": float(np.mean([A1[p][f"match{j}_{l}"] for p in df.pair for j in (1, 2) for l in G7
                                                                            if not (df.set_index("pair").loc[p, f"task{j}"] == "stsb")]))}
    agreeR = np.array([[0.5 * (A1[p][f"agree1_{l}"] + A1[p][f"agree2_{l}"]) for l in G4] for p in df.pair])
    S[f"{pre}_checks"]["U1_recomputed_equals_E6_cached_frac"] = float(np.mean(first_argmax(agreeR, G4) == first_argmax(df[[f"agree_{l}" for l in G4]].values, G4)))
    S[f"{pre}_checks"]["agree_recomputed_max_absdiff"] = float(np.abs(agreeR - df[[f"agree_{l}" for l in G4]].values).max())
    Dt = lambda lv: np.array([df.iloc[i][f"Dtest_{l}"] for i, l in enumerate(lv)])
    scopes = [("all", np.ones(N, bool))] + [(p, (df["pop"] == p).values) for p in ("P", "R0", "S1")] + [("all_no_stsb", ~has_stsb)]
    out = {}; perpair = df[["pop", "pair", "task1", "task2", "lam_sel"]].copy()
    for gname, grid in (("G4", G4), ("G7", G7)):
        lsel = first_argmax(df[[f"hold_{l}" for l in grid]].values, grid) if gname == "G7" else df.lam_sel.values
        Dsel = Dt(lsel)
        L = {"U1_agree": first_argmax(df[[f"agree_{l}" for l in grid]].values, grid),
             "U1_recomputed_E7_logits": first_argmax(np.array([[0.5 * (A1[p][f"agree1_{l}"] + A1[p][f"agree2_{l}"]) for l in grid] for p in df.pair]), grid),
             "ENT_min_entropy": first_argmax(-ent_mat(grid), grid), "ENTn_min_norm_entropy": first_argmax(-ent_mat(grid, True), grid),
             "fixed_lam1.0": np.full(N, 1.0), "fixed_lam0.7": np.full(N, 0.7),
             "Lcal_labeled_ref": first_argmax(df[[f"calnorm_{l}" for l in grid]].values, grid),
             "oracle_test": np.array(grid)[np.argmin(df[[f"Dtest_{l}" for l in grid]].values, axis=1)]}
        REG = {k: 100 * (Dt(v) - Dsel) for k, v in L.items()}
        REG["RAND_grid_expectation"] = 100 * (df[[f"Dtest_{l}" for l in grid]].values.mean(1) - Dsel)
        for k, v in L.items(): perpair[f"{gname}_lam_{k}"] = v
        for k, v in REG.items(): perpair[f"{gname}_regret_{k}"] = v
        g = out.setdefault(gname, {})
        for sc, mask in scopes:
            if mask.sum() < 2: continue
            key = f"{bbtag}/{sc}"; e = g.setdefault(sc, {"n": int(mask.sum())})
            e["regret_pp_taskblock"] = {k: BT.wmean_ci(v, key, mask) for k, v in REG.items()}
            e["U1_minus_pp_taskblock"] = {k: BT.wmean_ci(REG["U1_agree"] - v, key, mask) for k, v in REG.items() if k != "U1_agree"}
            e["U1_minus_pp_iid"] = {k: iid_ci((REG["U1_agree"] - v)[mask]) for k, v in REG.items() if k != "U1_agree"}
            e["lambda_counts"] = {k: {str(l): int((L[k][mask] == l).sum()) for l in grid} for k in ("U1_agree", "ENT_min_entropy", "ENTn_min_norm_entropy", "oracle_test")}
            e["U1_better_frac_vs_ENT"] = float((REG["U1_agree"] < REG["ENT_min_entropy"])[mask].mean())
            e["U1_worse_frac_vs_ENT"] = float((REG["U1_agree"] > REG["ENT_min_entropy"])[mask].mean())
    S[f"{pre}_grid"] = out
    # A2 AdaMerging-style gradient
    A2 = {r["id"]: r for r in jl("a2_adamerge.jsonl")} if bbtag == "qwen15" else {}
    if A2:
        m2 = df.pair.isin(A2.keys()).values
        Dada = np.full(N, np.nan)
        for i, p in enumerate(df.pair):
            if p in A2:
                r = A2[p]; Dada[i] = 1 - 0.5 * (r["test1"] / df.iloc[i]["single_test1"] + r["test2"] / df.iloc[i]["single_test2"])
        Dsel = Dt(df.lam_sel.values); regA = 100 * (Dada - Dsel); regA[~m2] = 0.0
        reg = {k: perpair[f"G4_regret_{k}"].values for k in ("U1_agree", "ENT_min_entropy", "fixed_lam1.0", "fixed_lam0.7")}
        perpair["A2_regret_ADA"] = np.where(m2, regA, np.nan); perpair["A2_lam1"] = [A2[p]["lam"][0] if p in A2 else np.nan for p in df.pair]
        perpair["A2_lam2"] = [A2[p]["lam"][1] if p in A2 else np.nan for p in df.pair]
        e = {"n": int(m2.sum()), "pops": sorted(set(df["pop"][m2])), "key": "a2subset"}
        e["regret_pp_taskblock"] = {"ADA_grad_entropy": BT.wmean_ci(regA, "a2subset", m2), **{k: BT.wmean_ci(v, "a2subset", m2) for k, v in reg.items()}}
        e["U1_minus_ADA_pp_taskblock"] = BT.wmean_ci(reg["U1_agree"] - regA, "a2subset", m2)
        e["U1_minus_ADA_pp_iid"] = iid_ci((reg["U1_agree"] - regA)[m2])
        e["ENT_minus_ADA_pp_taskblock"] = BT.wmean_ci(reg["ENT_min_entropy"] - regA, "a2subset", m2)
        lam = np.array([A2[p]["lam"] for p in df.pair[m2]])
        e["ada_lambda_summary"] = {"mean": lam.mean(0).tolist(), "min": lam.min(0).tolist(), "max": lam.max(0).tolist(), "frac_at_upper_clamp": float((lam >= 2.5 - 1e-6).mean())}
        e["entropy_first_last_mean"] = [float(np.mean([A2[p]["hist"][0][0] for p in df.pair[m2]])), float(np.mean([A2[p]["hist"][-1][0] for p in df.pair[m2]]))]
        S["A2_adamerging"] = e
    perpair.to_csv(HERE / f"e7_{pre}_per_pair.csv", index=False)
    return df


# ================================================================== B: cost
def part_b(S):
    HT = jl("holdtime.jsonl"); A2 = jl("a2_adamerge.jsonl"); rows = {}
    def summ(v): v = np.array(v, float); return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    for fn, suf in (("a1_entropy.jsonl", ""), ("a1_entropy_q05.jsonl", "_Qwen0.5B")):
      A1 = jl(fn)
      if A1:
          for gname, grid in (("G4", G4), ("G7", G7)):
              mer = [sum(r["timing"][f"merged_cal_lam{l}_s"] for l in grid) for r in A1]; sing = [r["timing"]["singles_cal_s"] for r in A1]
              ne = [r["n_fwd"]["merged_cal_examples_per_lam"] for r in A1]; ns = [r["n_fwd"]["singles_cal_examples"] for r in A1]
              pk_m = [max(r["peak_gb"][f"merged_cal_lam{l}"] for l in grid) for r in A1]; pk_s = [r["peak_gb"]["singles_cal"] for r in A1]
              rows[f"U1_{gname}{suf}"] = {"wall_s": summ(np.add(mer, sing)), "fwd_examples": summ(np.add(np.multiply(ne, len(grid)), ns)),
                                     "weight_writes": len(grid) + 2, "peak_gb_max": float(max(max(pk_m), max(pk_s))), "labels_needed": 0}
              rows[f"M3_{gname}{suf}"] = dict(rows[f"U1_{gname}{suf}"], note="Dhat_U = 1 - max agreement: identical computation to U1 (shared; marginal cost ~0 if U1 is run)")
              rows[f"ENT_{gname}{suf}"] = {"wall_s": summ(mer), "fwd_examples": summ(np.multiply(ne, len(grid))), "weight_writes": len(grid),
                                      "peak_gb_max": float(max(pk_m)), "labels_needed": 0}
    if HT:
        w = [sum(r["timing"].values()) for r in HT]
        rows["heldout_tuning_G4"] = {"wall_s": summ(w), "fwd_examples": summ([r["n_hold_examples_per_lam"] * 4 for r in HT]), "weight_writes": 4,
                                     "peak_gb_max": float(max(max(r["peak_gb"].values()) for r in HT)), "labels_needed": "2x1000 labeled held-out",
                                     "lam_sel_reproduced_frac": float(np.mean([r["lam_sel"] == r["lam_sel_e6"] for r in HT])),
                                     "hold_norm_max_absdiff_vs_E6": float(max(r[f"hold_norm_absdiff_vs_e6_{l}"] for r in HT for l in G4))}
    if A2:
        rows["ADA_grad_entropy"] = {"wall_s": summ([r["opt_s"] for r in A2]), "fwd_examples": summ([r["n_fwd_bwd_examples"] for r in A2]),
                                    "note": "forward+backward examples (60 steps x 16 per classification task), gradient checkpointing; excludes final eval",
                                    "weight_writes": 1, "peak_gb_max": float(max(r["opt_peak_gb"] for r in A2)), "labels_needed": 0}
    S["B_cost"] = rows


# ================================================================== C: multi-task
def part_c(S):
    C = jl("c_multitask.jsonl")
    if not C: return
    rows = []
    for r in C:
        m = r["m"]; row = {"id": r["id"], "m": m, "tasks": r["tasks"]}
        for l in G7:
            row[f"Dtest_{l}"] = 1 - np.mean([r[f"test{j}_{l}"] / r[f"single_test{j}"] for j in range(1, m + 1)])
            row[f"hold_{l}"] = r[f"hold_norm_{l}"]; row[f"agree_{l}"] = np.mean([r[f"agree{j}_{l}"] for j in range(1, m + 1)])
            row[f"calnorm_{l}"] = np.mean([r[f"calacc{j}_{l}"] / r[f"single_cal{j}"] for j in range(1, m + 1)])
            e = [r[f"ent{j}_{l}"] for j in range(1, m + 1) if r[f"ent{j}_{l}"] is not None]; row[f"ent_{l}"] = np.mean(e)
        rows.append(row)
    df = pd.DataFrame(rows); N = len(df)
    TN = sorted({t for ts in df.tasks for t in ts}); TI = {t: i for i, t in enumerate(TN)}
    tidx = [np.array([TI[t] for t in ts]) for ts in df.tasks]
    def tb(x, mask, seed):
        idx = np.where(mask)[0]; present = sorted({i for k in idx for i in tidx[k]}); r = np.random.default_rng(seed); bs = []
        W = np.empty((B, len(idx)))
        for b in range(B):
            c = np.bincount(r.choice(present, size=len(present), replace=True), minlength=len(TN)); W[b] = [np.prod(c[tidx[k]]) for k in idx]
        return idx, W
    Wc = {}
    def wci(x, key, mask):
        if key not in Wc: Wc[key] = tb(x, mask, 20260929 + sum(map(ord, key)))
        idx, W = Wc[key]; xv = x[idx]
        with np.errstate(invalid="ignore", divide="ignore"): bs = (W @ xv) / W.sum(1)
        return float(xv.mean()), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    def aci(s, y, key, mask):
        idx, W = Wc[key]; sv, yv = s[idx], y[idx]
        est = LC.wauc(sv, yv, np.ones(len(idx))); bs = np.array([LC.wauc(sv, yv, W[b]) for b in range(B)])
        return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    Dt = lambda lv: np.array([df.iloc[i][f"Dtest_{l}"] for i, l in enumerate(lv)])
    out = {}; pp = df[["id", "m"]].copy()
    for gname, grid in (("G4", G4), ("G7", G7)):
        lsel = first_argmax(df[[f"hold_{l}" for l in grid]].values, grid); Dsel = Dt(lsel)
        L = {"U1_agree": first_argmax(df[[f"agree_{l}" for l in grid]].values, grid), "ENT_min_entropy": first_argmax(-df[[f"ent_{l}" for l in grid]].values, grid),
             "fixed_lam1.0": np.full(N, 1.0), "fixed_lam0.7": np.full(N, 0.7),
             "Lcal_labeled_ref": first_argmax(df[[f"calnorm_{l}" for l in grid]].values, grid),
             "oracle_test": np.array(grid)[np.argmin(df[[f"Dtest_{l}" for l in grid]].values, axis=1)]}
        REG = {k: 100 * (Dt(v) - Dsel) for k, v in L.items()}; REG["RAND_grid_expectation"] = 100 * (df[[f"Dtest_{l}" for l in grid]].values.mean(1) - Dsel)
        pp[f"{gname}_lam_sel"] = lsel; pp[f"{gname}_Dtest_sel"] = Dsel
        for k, v in L.items(): pp[f"{gname}_lam_{k}"] = v
        for k, v in REG.items(): pp[f"{gname}_regret_{k}"] = v
        for sc, mask in (("m3", (df.m == 3).values), ("m4", (df.m == 4).values), ("m3+m4", np.ones(N, bool))):
            if mask.sum() == 0: continue
            e = out.setdefault(gname, {}).setdefault(sc, {"n": int(mask.sum())})
            e["regret_pp_taskblock"] = {k: wci(v, sc, mask) for k, v in REG.items()}
            e["U1_minus_pp_taskblock"] = {k: wci(REG["U1_agree"] - v, sc, mask) for k, v in REG.items() if k != "U1_agree"}
            e["U1_minus_pp_iid"] = {k: iid_ci((REG["U1_agree"] - v)[mask]) for k, v in REG.items() if k != "U1_agree"}
            e["lambda_counts"] = {k: {str(l): int((v[mask] == l).sum()) for l in grid} for k, v in list(L.items()) + [("lam_sel", lsel)] if k not in ("fixed_lam1.0", "fixed_lam0.7")}
            e["Dtest_sel_mean_pp"] = float(100 * Dsel[mask].mean())
            if gname == "G4":
                Dhat = 1 - df[[f"agree_{l}" for l in G4]].values.max(1); e["M3"] = {}
                from scipy.stats import spearmanr
                e["M3_spearman_Dhat_vs_Dtest_sel"] = float(spearmanr(Dhat[mask], Dsel[mask]).correlation)
                for tau in (0.02, 0.05):
                    y = (Dsel > tau).astype(int); pred = (Dhat >= LC.FROZEN[f"M3_threshold_tau{tau:g}"]).astype(int)
                    d = {"prevalence": float(y[mask].mean()), "n_pos": int(y[mask].sum())}
                    if 0 < y[mask].sum() < mask.sum():
                        d["auroc_taskblock"] = aci(Dhat, y, sc, mask)
                    else: d["auroc_taskblock"] = "not defined (one class only)"
                    d["acc_frozen_thr"] = wci((pred == y).astype(float), sc, mask); d["acc_always_merge"] = wci((y == 0).astype(float), sc, mask)
                    d["acc_diff_vs_always_merge"] = wci((pred == y).astype(float) - (y == 0).astype(float), sc, mask)
                    e["M3"][str(tau)] = d
    pp.to_csv(HERE / "e7_C_per_tuple.csv", index=False)
    S["C_multitask"] = out


def md_report(S):
    L = ["# RESULTS_E7 — EXPLORATORY / POST HOC (not preregistered)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')} by e7_analysis.py. Plan: PLAN_E7.md (stamp PLAN_E7.stamp); deviations: DEVIATIONS_E7.md.",
         "Regret = 100*(Dtest(rule) - Dtest(held-out-tuned lambda_sel)) in pp (lower is better; negative = better than held-out tuning).",
         "CIs: 95% task-block bootstrap (B=2000; lam_confirm_e6.Boot). 'U1 - X' < 0 means U1 has lower regret than X.", ""]
    for pre, title in (("A", "Qwen2.5-1.5B (E6)"), ("A_q05", "Qwen2.5-0.5B (E4b/E5b)")):
      if f"{pre}_grid" not in S: continue
      L += [f"## {pre}. Label-free lambda selection, {title} pairs (n = {S[pre + '_n_pairs_with_entropy']}/{S[pre + '_n_pairs_total']})", "",
            f"Checks: {json.dumps(S[pre + '_checks'])}", ""]
      for g in ("G4", "G7"):
            for sc, e in S[f"{pre}_grid"][g].items():
                L += [f"### {g} / {sc} (n={e['n']})", "", "| rule | mean regret pp [95% CI] | U1 - rule pp [task-block CI] | U1 - rule [iid CI] |", "|---|---|---|---|"]
                for k, v in e["regret_pp_taskblock"].items():
                    L.append(f"| {k} | {fmt(v)} | {fmt(e['U1_minus_pp_taskblock'][k]) if k in e['U1_minus_pp_taskblock'] else '—'} | {fmt(e['U1_minus_pp_iid'][k]) if k in e['U1_minus_pp_iid'] else '—'} |")
                L += ["", f"lambda counts: {json.dumps(e['lambda_counts'])}; U1 better/worse than ENT in {e['U1_better_frac_vs_ENT']:.3f}/{e['U1_worse_frac_vs_ENT']:.3f} of pairs", ""]
    if "A2_adamerging" in S:
        e = S["A2_adamerging"]
        L += [f"## A2. AdaMerging-style gradient entropy optimisation (task-wise lam1, lam2), n={e['n']} pairs ({e['pops']})", "", "| rule | mean regret pp [CI] |", "|---|---|"]
        L += [f"| {k} | {fmt(v)} |" for k, v in e["regret_pp_taskblock"].items()]
        L += ["", f"U1 - ADA: {fmt(e['U1_minus_ADA_pp_taskblock'])} (task-block), {fmt(e['U1_minus_ADA_pp_iid'])} (iid); ENT(grid) - ADA: {fmt(e['ENT_minus_ADA_pp_taskblock'])}",
              f"ADA lambdas: {json.dumps(e['ada_lambda_summary'])}; mean entropy first->last step: {e['entropy_first_last_mean']}", ""]
    if "B_cost" in S:
        L += ["## B. Compute cost per pair (RTX 4070 Ti SUPER 16GB, bf16, Qwen2.5-1.5B; model load and adapter loading excluded)", "",
              "| rule | wall s mean (median) [min–max] | examples forwarded mean | weight writes | peak GPU GB | labels |", "|---|---|---|---|---|---|"]
        for k, v in S["B_cost"].items():
            w = v["wall_s"]; f = v["fwd_examples"]
            L.append(f"| {k} (n={w['n']}) | {w['mean']:.2f} ({w['median']:.2f}) [{w['min']:.2f}–{w['max']:.2f}] | {f['mean']:.0f} | {v['weight_writes']} | {v['peak_gb_max']:.2f} | {v['labels_needed']} |")
        L += ["", "Notes: " + "; ".join(f"{k}: {v['note']}" for k, v in S["B_cost"].items() if "note" in v)]
        if "heldout_tuning_G4" in S["B_cost"]:
            h = S["B_cost"]["heldout_tuning_G4"]; L.append(f"Held-out tuning check: lam_selected reproduced in {h['lam_sel_reproduced_frac']:.3f} of timed pairs; max |hold_norm - E6| = {h['hold_norm_max_absdiff_vs_E6']:.2e}")
        L.append("")
    if "C_multitask" in S:
        L += ["## C. Multi-task (3- and 4-adapter) task-arithmetic merges, Qwen2.5-1.5B", ""]
        for g, gg in S["C_multitask"].items():
            for sc, e in gg.items():
                L += [f"### {g} / {sc} (n={e['n']}; mean Dtest at lambda_sel = {e['Dtest_sel_mean_pp']:.2f} pp)", "", "| rule | mean regret pp [CI] | U1 - rule [task-block CI] | U1 - rule [iid CI] |", "|---|---|---|---|"]
                for k, v in e["regret_pp_taskblock"].items():
                    L.append(f"| {k} | {fmt(v)} | {fmt(e['U1_minus_pp_taskblock'][k]) if k in e['U1_minus_pp_taskblock'] else '—'} | {fmt(e['U1_minus_pp_iid'][k]) if k in e['U1_minus_pp_iid'] else '—'} |")
                L += ["", f"lambda counts: {json.dumps(e['lambda_counts'])}"]
                if "M3" in e:
                    L.append(f"M3: Spearman(Dhat_U, Dtest_sel) = {e['M3_spearman_Dhat_vs_Dtest_sel']:.3f}")
                    for tau, d in e["M3"].items():
                        au = d["auroc_taskblock"]; au = fmt(au, 3) if isinstance(au, tuple) or isinstance(au, list) else au
                        L.append(f"- tau={tau}: prevalence {d['prevalence']:.3f} (n_pos {d['n_pos']}), AUROC {au}; acc(frozen thr) {fmt(d['acc_frozen_thr'],3)} vs always-merge {fmt(d['acc_always_merge'],3)}; diff {fmt(d['acc_diff_vs_always_merge'],3)}")
                L.append("")
    (HERE / "RESULTS_E7.md").write_text("\n".join(L) + "\n")


def main():
    S = {"label": "E7 EXPLORATORY / POST HOC", "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    part_a(S, "qwen15"); part_a(S, "qwen05"); part_b(S); part_c(S)
    (HERE / "summary_e7.json").write_text(json.dumps(S, indent=1, default=float)); md_report(S); print("ok", list(S))


if __name__ == "__main__":
    main()
