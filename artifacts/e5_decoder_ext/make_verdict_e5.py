#!/usr/bin/env python3
"""Formatting only: writes VERDICT_E5.md from analysis_e5a.json, analysis_e5b.json, analysis_e5d.json, e5c_reliability.json, timing_e5.json.
No statistics are computed here except unit conversions (pp) and sums of logged seconds (GPU time)."""
import json, sys, time
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
J = lambda p: json.loads((R / p).read_text()) if (R / p).exists() else None
A, B, D, C, T = J("e5a/analysis_e5a.json"), J("e5b/analysis_e5b.json"), J("e5d/analysis_e5d.json"), J("e5c/e5c_reliability.json"), J("timing_e5.json")
PR = J("prereg_e5.json")
pp = lambda x: f"{100 * x:+.2f}"
L = []
L.append("# E5 VERDICT — acceptance-odds experiments on the E4b Qwen2.5-0.5B adapters\n")
L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `make_verdict_e5.py` (formatting only). Pre-registration: `PREREG_E5.md` + `prereg_e5.json` "
         "(sha256 in `prereg_e5.sha256`; record `PREREG_RECORD_E5.txt`; local git commit `d0975c4` on branch `e5-prereg`, not pushed). "
         "**Disclosure: the pre-registration was written after the E4b verdict was known** (see PREREG_E5.md §0). Deviations: `DEVIATIONS_E5.md`.\n")
if A:
    fam = A["family"]
    L.append(f"## E5a — method comparison vs tuned TA on Qwen P ({A['n_pairs']}/{A['n_planned']} pairs; equal budget of 8 configurations per method)\n")
    L.append(f"Decision rule (pre-registered): a method beats TA iff mean gain ≥ +0.5 pp AND Holm-adjusted one-sided Wilcoxon p < 0.05 (Holm over {len(fam)} methods). "
             f"TA mean normalized score {A['TA_mean_score']:.4f}; TA boundary rate {A['TA']['boundary_rate']:.2f}; TA selected λ: {A['TA']['selected']}.\n")
    L.append("| method | mean score | mean Δ vs TA (pp) | median Δ (pp) | task-block 95% CI (pp) | W/T/L | Wilcoxon p (1-sided >) | **Holm p** | sign-flip Holm p (E3 test) | Wilcoxon p (2-sided) | boundary | **beats TA?** |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
    for m in fam:
        s = A["per_method"][m]
        L.append(f"| {m} | {s['mean_score']:.4f} | {pp(s['mean_gain_vs_TA'])} | {pp(s['median_gain'])} | [{pp(s['task_block_95ci'][0])}, {pp(s['task_block_95ci'][1])}] | "
                 f"{s['win']}/{s['tie']}/{s['loss']} | {s['wilcoxon_p_one_sided_greater']:.4g} | {s['holm_p_wilcoxon']:.4g} | {s['holm_p_signflip']:.4g} | "
                 f"{s['wilcoxon_p_two_sided']:.3g} | {s['boundary_rate']:.2f} | **{'YES' if s['beats_TA'] else 'no'}** |")
    L.append(f"\n**Any method beats TA: {'YES' if A['any_method_beats_TA'] else 'NO'}.** "
             + " ".join(f"{m}: CI upper < +0.5 pp → a ≥ 0.5 pp gain is not supported." for m in fam if A['per_method'][m]['ci_upper_lt_0.5pp']))
    dis = [m for m in fam if A["per_method"][m]["beats_TA"] != A["per_method"][m]["beats_TA_under_E3_signflip_test"]]
    L.append(f"\nWilcoxon vs E3 sign-flip decision disagreement: {dis if dis else 'none'}.\n")
    L.append("Selected configurations: " + "; ".join(f"{m}: {A['per_method'][m]['selected']}" for m in fam) + "\n")
    L.append("### Secondary (no multiplicity correction)\n")
    L.append("| contrast | n | mean (pp) | median (pp) | task-block 95% CI (pp) | Wilcoxon p (2-sided) | sign-flip p (2-sided) | W/T/L |")
    L.append("|---|---:|---:|---:|---|---:|---:|---|")
    for k, s in A["secondary"].items():
        if not isinstance(s, dict): continue
        ci = s["task_block_95ci"]
        L.append(f"| {k} | {s['n']} | {pp(s['mean'])} | {pp(s['median'])} | [{pp(ci[0])}, {pp(ci[1])}] | {s['wilcoxon_p_two_sided']:.3g} | {s['signflip_p_two_sided']:.3g} | {s['win']}/{s['tie']}/{s['loss']} |")
    L.append(f"\nGate-active pairs (θ★ = 30°): {A['secondary']['n_gate_active']}. Sanity (E3 TA at the E4b λs vs E4b stage 2): {A['sanity_TA_vs_E4b_stage2']}.\n")
    L.append(f"Exploratory Spearman(predictor, gain): {json.dumps(A['exploratory_predictor_vs_gain_spearman'])}\n")
if B:
    L.append("## E5b — λ grid extended to {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0} (pre-registered SENSITIVITY; the E4b primary verdict is unchanged)\n")
    L.append("| population | H | predictor | ρ original grid | ρ extended grid | task-block CI (ext) | Holm p (ext) | LOTO frac (ext) | rule (ext, descriptive) | rule (orig) |")
    L.append("|---|---|---|---:|---:|---|---:|---:|---|---|")
    for p in PR["e5b"]["populations_order"]:
        b = B.get(p, {})
        if b.get("status") != "complete": L.append(f"| {p} | — | — | {b.get('status')} | | | | | | |"); continue
        for h in ("H1", "H2"):
            o, e = b["original_grid"][h], b["extended_grid"][h]
            L.append(f"| {p} | {h} | {e['predictor']} | {o['rho']:.4f} | {e['rho']:.4f} | [{e['boot_ci95'][0]:.3f}, {e['boot_ci95'][1]:.3f}] | {e['holm_p']:.4f} | {e['loto_frac_gt_0.2']:.2f} | {e['verdict']} | {o['verdict']} |")
    L.append("")
    for p in PR["e5b"]["populations_order"]:
        b = B.get(p, {})
        if b.get("status") != "complete": continue
        L.append(f"- **{p}**: λ\\* counts original {b['lam_selected_counts_original']} → extended {b['lam_selected_counts_extended']}; λ changed in {b['n_lam_changed']}/{b['n']} pairs; "
                 f"share at the new top λ = 2.0: {b['frac_at_new_top_2.0']:.2f}; mean D {b['D_summary_original']['mean']:.4f} → {b['D_summary_extended']['mean']:.4f} "
                 f"(mean D − D₊ = {b['mean_D_minus_Dext']:.4f}); Spearman(D, D₊) = {b['spearman_D_Dext']:.3f}; fixed-λ ρ at the new λs: {json.dumps(b['fixed_lambda_rho_ext'])}; "
                 f"sanity λ = 1.0 max |Δ| = {b['sanity_lam1.0_maxabs']}.")
    L.append(f"\nCross-check E5a TA held-out at λ = 1.3/1.5 vs E5b: {B.get('crosscheck_e5a_vs_e5b_hold')}.\n")
if C:
    L.append("## E5c — evaluation-sampling reliability of D (CPU; saved per-example predictions at the original λ\\*)\n")
    L.append("| exp | pop | n | SD(D) between pairs | mean / median / p95 SE(D) | 1 − mean SE²/Var D | indep.-replicate r → SB (E1b method) | true split-half r [95%] → SB | √SB | ρ(O_A,D): obs / boot SD / [2.5, 97.5] | ρ(cos,D): obs / boot SD / [2.5, 97.5] |")
    L.append("|---|---|---:|---:|---|---:|---|---|---:|---|---|")
    for r in C["results"]:
        o, c = r["rho_under_eval_resampling"]["O_A"], r["rho_under_eval_resampling"]["tv_cosine"]
        L.append(f"| {r['experiment']} | {r['population']} | {r['n_pairs']} | {r['D_sd_between_pairs']:.4f} | {r['mean_pair_se']:.4f} / {r['median_pair_se']:.4f} / {r['p95_pair_se']:.4f} | "
                 f"{r['sampling_reliability_1_minus_meanSE2_over_varD']:.3f} | {r['indep_replicate_r_mean']:.3f} → {r['indep_replicate_SB']:.3f} | "
                 f"{r['split_half_r_mean']:.3f} [{r['split_half_r_p2.5']:.3f}, {r['split_half_r_p97.5']:.3f}] → {r['split_half_SB']:.3f} | {r['sampling_attenuation_ceiling_sqrt_SB']:.3f} | "
                 f"{o['rho_observed']:.3f} / {o['boot_sd']:.3f} / [{o['boot_p2.5']:.3f}, {o['boot_p97.5']:.3f}] | {c['rho_observed']:.3f} / {c['boot_sd']:.3f} / [{c['boot_p2.5']:.3f}, {c['boot_p97.5']:.3f}] |")
    L.append(f"\nD recomputed from saved predictions equals the reported D in every population (max |Δ| = {max(r['D_recompute_max_abs_diff'] for r in C['results']):.1e}). R = {C['R']}, split-halves = {C['S_split']}, seeds {C['seeds']}.\n")
if D:
    L.append("## E5d — lemma quantities on Qwen gate layers (derived; Prop. 2–3 of `paper/LEMMA_PROJECTION.md`)\n")
    def row(name, s):
        return (f"| {name} | {s['n_layers']} | {s['n_pairs']} | {s['abs_cos_q25_med_q75'][1]:.3f} [{s['abs_cos_q25_med_q75'][0]:.3f}, {s['abs_cos_q25_med_q75'][2]:.3f}] | "
                f"{s['frac_not_coef_change_q25_med_q75'][1]:.3f} [{s['frac_not_coef_change_q25_med_q75'][0]:.3f}, {s['frac_not_coef_change_q25_med_q75'][2]:.3f}] | {s['mean_frac_not_coef_change']:.3f} | "
                f"{s['c_q25_med_q75'][1]:.3f} | {s['frac_layers_c_gt_0']:.2f} | {s['frac_layers_abs_cos_ge_0.5']:.2f} / {s['frac_layers_abs_cos_ge_0.9']:.2f} | {s['eps_q25_med_q75'][1]:.2f} | {s['edit_size_rel_merge_med']:.3f} | {s['identity_residual_max']:.1e} |")
    L.append("| set | layers | pairs | \\|cos φ\\| median [IQR] (collinearity of X1, X2) | fraction of gate edit not a coefficient change (1 − cos²φ) median [IQR] | mean | median c | share c > 0 | share \\|cos\\| ≥ 0.5 / ≥ 0.9 | median ε | median edit/‖M_sum‖ | max identity residual |")
    L.append("|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|")
    L.append(row("θ★=30° FAIL layers, all pops", D["gate30_all"]))
    for p, s in D["gate30_by_population"].items(): L.append(row(f"  gate30 {p}", s))
    for p, s in D["gate30_by_layer_type"].items(): L.append(row(f"  gate30 {p}", s))
    if "forced_k1_P_all" in D:
        L.append(row("forced k=1, all 168 layers of P", D["forced_k1_P_all"]))
        for p, s in D["forced_k1_P_by_layer_type"].items(): L.append(row(f"  forced {p}", s))
    L.append(f"\nReference (BERT seed pair): {D['reference_BERT_seed_pair']}.\n")
if T:
    tot = sum(v for k, v in T.items() if k.startswith("e5a") or k.startswith("e5b"))
    L.append(f"## Compute\n\nLogged GPU process seconds: {json.dumps({k: round(v) for k, v in T.items()})}; GPU jobs (E5a + E5b) total {tot / 3600:.2f} h (serial). E5d ran on CPU; E5c on the box CPU.\n")
(R / "VERDICT_E5.md").write_text("\n".join(L) + "\n")
print("wrote", R / "VERDICT_E5.md")
