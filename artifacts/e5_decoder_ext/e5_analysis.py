#!/usr/bin/env python3
"""E5 pre-registered analysis (PREREG_E5.md sec. 5). Runs on ubuntu-4070 (conda env torch) in artifacts/e5_accept/.
Imports UNMODIFIED: e4b_analysis.pop_stats (verbatim E1c/E1b statistics) and e3.holm / e3.task_block_boot / e3.signflip.
Usage: python e5_analysis.py [e5a] [e5b] [e5d]   -> e5a/analysis_e5a.json, e5b/analysis_e5b.json, e5d/analysis_e5d.json"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ART = HERE.parent; E4B = ART / "e4b_decoder"; E3C = ART / "e3_baselines" / "code"
PR5 = json.loads((HERE / "prereg_e5.json").read_text())
sys.path.insert(0, str(E4B)); import e4b_analysis as EA   # noqa: E402
EA.bm.LOGF = HERE / "run_e5.log"
sys.path.insert(0, str(E3C)); import e3                   # noqa: E402
from scipy.stats import wilcoxon                           # noqa: E402
spearman = EA.spearman
TASKS = json.loads((E4B / "populations_e4b.json").read_text())["valid_tasks"]


def jdump(o, p): Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
def tk(k): return k.split("@")[0]


def wil(g, alt):
    g = np.asarray(g, float)
    if np.all(np.abs(g) <= 1e-12): return 1.0
    return float(wilcoxon(g, zero_method="wilcox", alternative=alt).pvalue)


def contrast(d, pk):
    d = np.asarray(d, float); ci, sk = e3.task_block_boot(dict(zip(pk, d)), TASKS) if len(d) >= 10 else ([float("nan")] * 2, None)
    p1, p2 = e3.signflip(d)
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "task_block_95ci": ci, "wilcoxon_p_two_sided": wil(d, "two-sided"),
            "signflip_p_two_sided": p2, "win": int((d > 1e-12).sum()), "tie": int((np.abs(d) <= 1e-12).sum()), "loss": int((d < -1e-12).sum())}


def load_jsonl(p):
    out, seen = [], set()
    for l in Path(p).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["pair"] not in seen: seen.add(r["pair"]); out.append(r)
    return out


# ------------------------------------------------------------------ E5a
def e5a():
    recs = load_jsonl(HERE / "e5a" / "results.jsonl")
    fam = PR5["e5a"]["family"]; meth = ["TA"] + fam
    rows = []
    for r in recs:
        x, y = r["t1"], r["t2"]
        row = {"pair": f"{x}__{y}", "t1": x, "t2": y, "t1_task": tk(x), "t2_task": tk(y)}
        for m in meth:
            row[m] = r["methods"][m]["score"]; row[f"{m}__sel"] = json.dumps(r["methods"][m]["sel_hp"]); row[f"{m}__bnd"] = r["methods"][m]["at_boundary"]
        row["TA4_E1bgrid"] = r["extra"]["TA4_score"]; row["PICO_TA_c1"] = r["extra"]["PICO_TA_c1_score"]
        row["gate_n_fail_layers"] = r["methods"]["GATE"]["info"].get("gate_n_fail_layers")
        row["secs"] = r["secs"]
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(HERE / "e5a" / "method_by_pair_e5a.csv", index=False, float_format="%.6f")
    pk = list(zip(df.t1_task, df.t2_task))
    A = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_pairs": len(df), "n_planned": 91, "family": fam,
         "complete": len(df) == 91, "TA_mean_score": float(df.TA.mean())}
    per = {}
    for m in fam:
        g = (df[m] - df.TA).values
        ci, sk = e3.task_block_boot(dict(zip(pk, g)), TASKS); p1, p2 = e3.signflip(g)
        per[m] = {"mean_score": float(df[m].mean()), "mean_gain_vs_TA": float(g.mean()), "median_gain": float(np.median(g)),
                  "task_block_95ci": ci, "wilcoxon_p_one_sided_greater": wil(g, "greater"), "wilcoxon_p_two_sided": wil(g, "two-sided"),
                  "wilcoxon_p_one_sided_less": wil(g, "less"), "signflip_p_one_sided": p1, "signflip_p_two_sided": p2,
                  "win": int((g > 1e-12).sum()), "tie": int((np.abs(g) <= 1e-12).sum()), "loss": int((g < -1e-12).sum()),
                  "boundary_rate": float(df[f"{m}__bnd"].mean()), "selected": df[f"{m}__sel"].value_counts().to_dict()}
    hw = e3.holm({m: per[m]["wilcoxon_p_one_sided_greater"] for m in fam}); hs = e3.holm({m: per[m]["signflip_p_one_sided"] for m in fam})
    for m in fam:
        per[m]["holm_p_wilcoxon"] = hw[m]; per[m]["holm_p_signflip"] = hs[m]
        per[m]["beats_TA"] = bool(per[m]["mean_gain_vs_TA"] >= PR5["decision"]["min_gain"] and hw[m] < PR5["decision"]["alpha"])
        per[m]["beats_TA_under_E3_signflip_test"] = bool(per[m]["mean_gain_vs_TA"] >= PR5["decision"]["min_gain"] and hs[m] < PR5["decision"]["alpha"])
        per[m]["ci_upper_lt_0.5pp"] = bool(per[m]["task_block_95ci"][1] < 0.005)
    A["per_method"] = per
    A["TA"] = {"boundary_rate": float(df["TA__bnd"].mean()), "selected": df["TA__sel"].value_counts().to_dict()}
    A["any_method_beats_TA"] = any(per[m]["beats_TA"] for m in fam)
    sec = {}
    ga = df.gate_n_fail_layers.fillna(0) > 0
    sec["n_gate_active"] = int(ga.sum())
    sec["GATE_minus_TA_on_gate_active"] = contrast((df.GATE - df.TA)[ga].values, [p for p, a_ in zip(pk, ga) if a_])
    for a_, b_ in (("GATE", "PICO_TA"), ("FORCEGATE", "PICO_TA"), ("PICO_GATE", "PICO_TA"), ("PICO_GATE", "GATE")):
        sec[f"{a_}_minus_{b_}"] = contrast((df[a_] - df[b_]).values, pk)
    sec["PICO_GATE_minus_PICO_TA_on_gate_active"] = contrast((df.PICO_GATE - df.PICO_TA)[ga].values, [p for p, a_ in zip(pk, ga) if a_])
    sec["TA8_minus_TA4_E1bgrid"] = contrast((df.TA - df.TA4_E1bgrid).values, pk)
    sec["PICO_c1_untuned_minus_TA"] = contrast((df.PICO_TA_c1 - df.TA).values, pk)
    A["secondary"] = sec
    # sanity: e3 TA at the E1b lambdas == E4b stage-2 values
    ref = {r["pair"]: r for r in load_jsonl(E4B / "pair_results_P.jsonl")}
    mx, lam_mis = 0.0, 0
    for r in recs:
        k = f"{r['t1']}__{r['t2']}"; R = ref[k]
        for l in (0.3, 0.5, 0.7, 1.0):
            for j, t in ((1, r["t1"]), (2, r["t2"])):
                mx = max(mx, abs(r["extra"][f"TA_eval_lam{l}"][t] - R[f"TA_eval_t{j}_lam{l}"]))
        lam_mis += int(r["extra"]["TA4_lam_selected"] != R["lam_selected"])
    A["sanity_TA_vs_E4b_stage2"] = {"max_abs_diff": mx, "lam_selected_mismatches": lam_mis, "exact": mx <= 1e-9 and lam_mis == 0}
    try:
        PR = pd.read_csv(E4B / "predictors_e4b.csv").set_index("pair")
        A["exploratory_predictor_vs_gain_spearman"] = {m: {c: spearman(PR.loc[df.pair, c].values, (df[m] - df.TA).values) for c in ("O_A", "tv_cosine")} for m in fam}
    except Exception as ex:
        A["exploratory_predictor_vs_gain_spearman"] = repr(ex)
    tim = {m: {k: float(sum(r["timing"][m][k] for r in recs)) for k in ("merge_s", "hold_s", "eval_s", "n_hold_evals", "n_eval_evals")} for m in meth}
    A["timing"] = tim; A["total_wall_s"] = float(df.secs.sum())
    jdump(A, HERE / "e5a" / "analysis_e5a.json")
    print(json.dumps({m: (round(100 * per[m]["mean_gain_vs_TA"], 3), round(per[m]["holm_p_wilcoxon"], 4), per[m]["beats_TA"]) for m in fam}))
    return A


# ------------------------------------------------------------------ E5b
def e5b():
    PR = pd.read_csv(E4B / "predictors_e4b.csv")
    LX = PR5["e5b"]["lambda_ext"]; LALL = [0.3, 0.5, 0.7, 1.0] + LX
    A = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "lambda_grid_extended": LALL, "label": "pre-registered SENSITIVITY (does not replace the E4b primary verdict)"}
    for pop in PR5["e5b"]["populations_order"]:
        p = HERE / "e5b" / f"pair_ext_{pop}.jsonl"
        if not p.exists(): A[pop] = {"status": "not run"}; continue
        ext = {r["pair"]: r for r in load_jsonl(p)}
        ref = {r["pair"]: r for r in load_jsonl(E4B / f"pair_results_{pop}.jsonl")}
        if len(ext) < len(ref): A[pop] = {"status": f"incomplete ({len(ext)}/{len(ref)})"}; continue
        rows = []
        for k, R in ref.items():
            X = ext[k]; hold = {l: (R if l <= 1.0 else X)[f"TA_hold_norm_lam{l}"] for l in LALL}
            Dl = {l: (R if l <= 1.0 else X)[f"TA_eval_D_lam{l}"] for l in LALL}
            lstar = LALL[int(np.argmax([hold[l] for l in LALL]))]
            row = {"pair": k, "D": R["D"], "lam_selected": R["lam_selected"], "D_ext": Dl[lstar], "lam_selected_ext": lstar}
            for l in LX: row[f"TA_eval_D_lam{l}"] = Dl[l]
            if "sanity_lam1.0_maxabs" in X: row["sanity"] = X["sanity_lam1.0_maxabs"]
            rows.append(row)
        df = PR.merge(pd.DataFrame(rows), on="pair")
        assert len(df) == len(ref)
        S0 = EA.pop_stats(df, TASKS, ["O_A", "tv_cosine"], Dcol="D")
        S1 = EA.pop_stats(df, TASKS, ["O_A", "tv_cosine"], Dcol="D_ext")
        strip = lambda S: {h: {k_: v[k_] for k_ in ("predictor", "verdict", "rho", "boot_ci95", "perm_p_one_sided", "holm_p", "loto_frac_gt_0.2", "criteria")} for h, v in S["verdicts"].items()}
        A[pop] = {"status": "complete", "n": len(df), "original_grid": strip(S0), "extended_grid": strip(S1),
                  "lam_selected_counts_original": {str(k_): int(v) for k_, v in df.lam_selected.value_counts().sort_index().items()},
                  "lam_selected_counts_extended": {str(k_): int(v) for k_, v in df.lam_selected_ext.value_counts().sort_index().items()},
                  "frac_at_new_top_2.0": float((df.lam_selected_ext == max(LALL)).mean()),
                  "n_lam_changed": int((df.lam_selected_ext != df.lam_selected).sum()),
                  "D_summary_original": EA.desc(df.D), "D_summary_extended": EA.desc(df.D_ext),
                  "mean_D_minus_Dext": float((df.D - df.D_ext).mean()), "spearman_D_Dext": spearman(df.D, df.D_ext),
                  "fixed_lambda_rho_ext": {c: {f"lam{l}": spearman(df[c], df[f"TA_eval_D_lam{l}"]) for l in LX} for c in ("O_A", "tv_cosine")},
                  "sanity_lam1.0_maxabs": float(df["sanity"].max()) if "sanity" in df and df["sanity"].notna().any() else None}
        df.to_csv(HERE / "e5b" / f"D_ext_{pop}.csv", index=False, float_format="%.8g")
    # cross-check E5a TA held-out at 1.3/1.5 == E5b (P)
    try:
        e5 = {f"{r['t1']}__{r['t2']}": r for r in load_jsonl(HERE / "e5a" / "results.jsonl")}
        ext = {r["pair"]: r for r in load_jsonl(HERE / "e5b" / "pair_ext_P.jsonl")}
        mx, n = 0.0, 0
        for k, r in e5.items():
            if k not in ext: continue
            for g in r["methods"]["TA"]["grid"]:
                l = g["hp"]["lam"]
                if l in (1.3, 1.5):
                    mx = max(mx, abs(g["hold"][r["t1"]] - ext[k][f"TA_hold_t1_lam{l}"]), abs(g["hold"][r["t2"]] - ext[k][f"TA_hold_t2_lam{l}"])); n += 1
        A["crosscheck_e5a_vs_e5b_hold"] = {"n_compared": n, "max_abs_diff": mx}
    except Exception as ex:
        A["crosscheck_e5a_vs_e5b_hold"] = repr(ex)
    jdump(A, HERE / "e5b" / "analysis_e5b.json")
    print(json.dumps({p: A[p].get("extended_grid", A[p].get("status")) for p in PR5["e5b"]["populations_order"]}, default=str)[:3000])
    return A


# ------------------------------------------------------------------ E5d
def e5d():
    L = pd.read_csv(HERE / "e5d" / "lemma_layers_e5d.csv")
    L["ltype"] = [n.split(".")[-1] for n in L.layer]
    def summ(d):
        q = lambda c: [float(np.nanpercentile(d[c], 25)), float(np.nanmedian(d[c])), float(np.nanpercentile(d[c], 75))]
        return {"n_layers": int(len(d)), "n_pairs": int(d.pair.nunique()), "abs_cos_q25_med_q75": q("abs_cos"), "c_q25_med_q75": q("c"),
                "frac_not_coef_change_q25_med_q75": q("frac_edit_energy_not_coef_change"), "eps_q25_med_q75": q("eps_rel_residual"),
                "mu_q25_med_q75": q("mu_equiv"), "edit_size_rel_merge_med": float(np.nanmedian(d.edit_size_rel_to_merge)),
                "mean_frac_not_coef_change": float(np.nanmean(d.frac_edit_energy_not_coef_change)),
                "frac_layers_c_gt_0": float((d.c > 0).mean()), "frac_layers_abs_cos_ge_0.9": float((d.abs_cos >= 0.9).mean()),
                "frac_layers_abs_cos_ge_0.5": float((d.abs_cos >= 0.5).mean()),
                "frac_entries_sign_flipped_med": float(np.nanmedian(d.frac_entries_sign_flipped_by_gate)),
                "identity_residual_max": float(d.identity_residual_max.max()), "regimes": d.regime.value_counts().to_dict()}
    A = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "label": "derived quantities (Lemma 1 / Prop. 2-3 of paper/LEMMA_PROJECTION.md)"}
    g = L[L["mode"] == "gate30"]
    A["gate30_all"] = summ(g)
    A["gate30_by_population"] = {p: summ(d) for p, d in g.groupby("population")}
    A["gate30_by_layer_type"] = {p: summ(d) for p, d in g.groupby("ltype")}
    f = L[L["mode"] == "forced_k1"]
    if len(f):
        A["forced_k1_P_all"] = summ(f); A["forced_k1_P_by_layer_type"] = {p: summ(d) for p, d in f.groupby("ltype")}
    A["reference_BERT_seed_pair"] = "paper/verify_lemma_seed_pair.json: 2 FAIL layers, cos 0.107 / 0.184, eps 0.857 / 0.850, c 0.102 / 0.190"
    jdump(A, HERE / "e5d" / "analysis_e5d.json")
    print(json.dumps(A["gate30_all"], default=str))
    return A


if __name__ == "__main__":
    which = sys.argv[1:] or ["e5a", "e5b", "e5d"]
    for w in which: globals()[w]()
