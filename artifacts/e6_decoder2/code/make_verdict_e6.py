#!/usr/bin/env python3
"""Writes VERDICT_E6_DECODER2.md from analysis_e6.json, e6_extra.json, stage0 files, pilot decision and timing (reporting only; no statistics
computed here). Copy of E4b make_verdict_e4b.py for the Qwen2.5-1.5B study, plus the E4b comparison / pooled-decoder / location / gate sections."""
import json, sys
from pathlib import Path
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
TAG, NAME = "e6", "E6"
E4B = HERE.parent / "e4b_decoder"
A = json.loads((HERE / f"analysis_{TAG}.json").read_text())
f = lambda x, d=4: "nan" if x is None or x != x else f"{x:.{d}f}"
L = []
vp = A.get("verdicts_primary", {})
PP = A.get("primary_population", "P")
L.append(f"# {NAME} VERDICT (second decoder: Qwen2.5-1.5B; replication of E4b) — H1 (O_A): **{vp.get('H1', {}).get('verdict', 'n/a')}**; H2 (task-vector cosine): **{vp.get('H2', {}).get('verdict', 'n/a')}**  (primary population {PP}{', mixed seed' if PP == 'P' else ', seed 0'})\n")
L.append(f"Generated {A['generated']}. Pre-registration: `PREREG_{NAME}.md` / `prereg_{TAG}.json` (sha256 in `prereg_{TAG}.sha256`); deviations/notes: `DEVIATIONS_{NAME}.md`. "
         "**Disclosure: the E6 pre-registration was written after the E4b and E5 results were known** (PREREG_E6.md sec. 0); every analysis choice is frozen from E4b.\n")
L.append(f"Valid tasks K = {A['K']}: {', '.join(A['valid_tasks'])}. Exclusions: {A['excluded'] or 'none'}. Predictor hash ok: {A['predictors_hash_ok']}. "
         f"Analysis-code equivalence on E1b (reproduces E1b analysis.json): {A['analysis_equivalence_on_E1b'].get('ok')}.\n")
pd_ = HERE / "pilot" / "pilot_decision.json"
if pd_.exists():
    D = json.loads(pd_.read_text())
    pj = D.get("projection", {})
    L.append(f"Pilot projection (recorded only; seeds fixed by PREREG_E6): train {f(pj.get('train_h_per_seed'), 2)} h/seed, core (2 seeds) {f(pj.get('core_h_two_seeds'), 2)} h -> seeds {D.get('seeds')}, primary {D.get('primary_population')}.\n")
    L.append(f"Pilot (pre-registered, held-out only): lr_main = {D.get('lr_main')} (non-diverged {D.get('non_diverged')}; mean held-out {D.get('mean_holdout_acc')}). " +
             "; ".join(f"lr {lr}: " + ", ".join(f"{t} hold {v['holdout_acc']:.4f} (maj {v['holdout_majority_baseline']:.3f}, tail loss {f(v['tail_mean_loss'], 3)}, diverged {v['diverged']})" for t, v in r.items()) for lr, r in D["results"].items()) + "\n")
def vt(res, tag):
    out = ["| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |", "|---|---|---:|---|---:|---:|---:|---|"]
    for h, v in res["verdicts"].items():
        out.append(f"| {h} ({tag}) | {v['predictor']} | {f(v['rho'])} | [{f(v['boot_ci95'][0], 3)}, {f(v['boot_ci95'][1], 3)}] | {f(v['perm_p_one_sided'])} | {f(v['holm_p'])} | {f(v['loto_frac_gt_0.2'], 2)} | **{v['verdict']}** |")
    return out
def popsum(p, X):
    return (f"{p}: {X['D_summary']['n']} pairs; D mean {f(X['D_summary']['mean'])}, median {f(X['D_summary']['median'])}, range [{f(X['D_summary']['min'])}, {f(X['D_summary']['max'])}]; "
            f"λ* counts {X['lam_selected_counts']}; ρ(O_A, tv_cos) = {f(X['rho_OA_tvcos'], 3)}; min θ_min over pairs/layers = {f(X['min_theta_min_overall_deg'], 1)}°; gate active in {X['gate_active_pairs']} pairs.\n")
L.append("## Primary: " + ("mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)\n" if PP == "P" else "seed-0 cross-task pairs R0 (single-seed design)\n"))
P = A[PP]
if P.get("status") == "complete":
    L += vt(P, PP)
    L.append("\nRule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).")
    for h, v in P["verdicts"].items():
        L.append(f"\n{h} criteria: {v['criteria']}. LOTO ρ: " + ", ".join(f"{t} {x:.2f}" for t, x in v["loto"].items()))
    L.append("\n" + popsum(PP, P))
else:
    L.append(f"P status: {P.get('status')}\n")
L.append("## Secondary (exploratory; no multiplicity correction across populations)\n")
for p, lab in [x for x in (("R0", "seed-0 cross-task pairs (rule applied descriptively)"), ("S1", "seed-1 cross-task pairs (rule applied descriptively)")) if x[0] != PP]:
    L.append(f"### {p}: {lab}\n")
    X = A[p]
    if X.get("status") == "complete":
        L += vt(X, p); L.append("\n" + popsum(p, X))
    else:
        L.append(f"{p} status: {X.get('status')}\n")
L.append("### Pooled 3-population summary (mean D and predictors over R0, S1, P per task pair; descriptive)\n")
PL = A.get("pooled_3pop", {})
if "verdicts_descriptive" in PL:
    L += vt({"verdicts": PL["verdicts_descriptive"]}, "pooled")
    L.append(f"\nPooled D mean {f(PL['D_summary']['mean'])}, n = {PL['n_pairs']} task pairs.\n")
else:
    L.append(f"{PL.get('status')}\n")
L.append("### Secondary predictors\n")
L.append("| predictor | ρ P | CI P | ρ R0 | CI R0 | ρ S1 | CI S1 |"); L.append("|---|---:|---|---:|---|---:|---|")
nanst = {"rho": float("nan"), "boot_ci95": [float("nan")] * 2}
if P.get("status") == "complete":
    for c, s in P["all_predictor_stats"].items():
        r0 = A["R0"].get("all_predictor_stats", {}).get(c, nanst); s1 = A["S1"].get("all_predictor_stats", {}).get(c, nanst)
        L.append(f"| {c} | {f(s['rho'], 3)} | [{f(s['boot_ci95'][0], 2)}, {f(s['boot_ci95'][1], 2)}] | {f(r0['rho'], 3)} | [{f(r0['boot_ci95'][0], 2)}, {f(r0['boot_ci95'][1], 2)}] | {f(s1['rho'], 3)} | [{f(s1['boot_ci95'][0], 2)}, {f(s1['boot_ci95'][1], 2)}] |")
    for p in ("P", "R0", "S1"):
        if A[p].get("status") == "complete":
            L.append(f"\nFixed-λ ρ ({p}): {json.dumps({k: {l: round(x, 3) for l, x in v.items()} for k, v in A[p]['fixed_lambda_rho'].items()})}")
L.append("\n### D stability / reliability across seed configurations (Spearman over the same unordered task pairs)\n")
L.append("| quantity | P~R0 | P~S1 | R0~S1 |"); L.append("|---|---:|---:|---:|")
for q in ("D", "O_A", "tv_cosine", "lam_selected"):
    g = lambda k: A["reliability"].get(f"{q}:{k}", {}).get("rho", float("nan"))
    L.append(f"| {q} | {f(g('P~R0'), 3)} | {f(g('P~S1'), 3)} | {f(g('R0~S1'), 3)} |")
L.append("\n### O_A vs random null; θ_min distribution; gate firings\n")
OA = A["O_A_and_theta"]
L.append(f"Random rank-8 null: O_A mean {OA['null_O_A_mean']:.5f} (sd {OA['null_O_A_sd']:.5f}, p95 {OA['null_O_A_p95']:.5f}).\n")
L.append("| population | n | O_A mean | sd | median | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layer-θ_min p1 / p5 / median (°) | layers <30° / <45° | pairs where gate fires |")
L.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|")
for k, v in OA.items():
    if not isinstance(v, dict) or "O_A" not in v: continue
    o = v["O_A"]; z = v["null_z_O_A"]; th = v["theta_min_all_layers"]
    L.append(f"| {k} | {o['n']} | {o['mean']:.5f} | {o['sd']:.5f} | {o['median']:.5f} | {o['min']:.5f} | {o['max']:.5f} | [{z['min']:.1f}, {z['max']:.1f}] | {v['frac_O_A_above_null_p95']:.2f} | "
             f"{v['tv_cosine']['mean']:.4f} | {v['min_theta_min_A_deg']['min']:.1f} | {th['p1']:.1f} / {th['p5']:.1f} / {th['median']:.1f} | {th['n_lt30']} / {th['n_lt45']} of {th['n']} | {v['n_pairs_gate_would_fire']} |")
for k in ("paired_P_minus_R0", "paired_P_minus_S1"):
    if k in OA:
        v = OA[k]; L.append(f"\n{k}: n={v['n']}, mean diff {v['mean_diff']:.5f}, P lower in {v['frac_P_lower']:.0%} of task pairs, ratio of means {v['ratio_mean']:.3f}, Wilcoxon p = {v['wilcoxon_p']:.2e}")
L.append("\nθ_min by layer type (all populations): " + "; ".join(f"{lt}: min {d['min']:.1f}°, median {d['median']:.1f}°" for lt, d in OA["theta_min_by_layer_type"].items()))
L.append("\n### S2: same-task seed pairs (t@s0 × t@s1)\n")
S2 = A["S2"]
L.append(f"Gate (θ★ = 30°) fires in **{S2['n_pairs_gate_fires']}/{len(S2['per_pair'])}** same-task pairs; min θ_min over all S2 pairs/layers = {S2['theta_min_over_all_S2']:.2f}°.\n")
L.append("| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |")
L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for e in S2["per_pair"]:
    L.append(f"| {e['task']} | {e['min_theta_min_deg']:.2f} | {e['argmin_layer']} | {e['n_layers_lt30']} | {e['n_layers_lt45']} | "
             f"{e['O_A']:.5f} | {e['null_z_O_A']:.1f} | {e['tv_cosine']:.4f} | {e.get('lam_selected', float('nan'))} | {f(e.get('D'))} | {f(e.get('TA_score_sel'))} | {f(e.get('GATE_atTA_score'))} | {f(e.get('GATE_own_score'))} |")
if "e1b_stage2_gate_vs_TA" in S2:
    for k, c in S2["e1b_stage2_gate_vs_TA"].items():
        L.append(f"\nE1b-grid {k} − TA: mean {c['mean']*100:+.3f} pp, CI [{c['ci95_boot_over_tasks'][0]*100:+.3f}, {c['ci95_boot_over_tasks'][1]*100:+.3f}] pp, sign-flip p = {c['signflip_p_two_sided']:.3f}, W/T/L {c['win']}/{c['tie']}/{c['loss']}")
if "e3_subset" in S2:
    E = S2["e3_subset"]
    L.append(f"\n**E3 code (amendment-A2 grids) on S2, {E['n_pairs']} pairs.** Mean normalized score: " + ", ".join(f"{k} {v:.4f}" for k, v in E["mean_scores"].items()) + ".\n")
    L.append("| contrast | mean (pp) | 95% CI over tasks (pp) | sign-flip p (2-sided) | W/T/L |"); L.append("|---|---:|---|---:|---|")
    for k, c in E["contrasts"].items():
        L.append(f"| {k} | {c['mean']*100:+.3f} | [{c['ci95_boot_over_tasks'][0]*100:+.3f}, {c['ci95_boot_over_tasks'][1]*100:+.3f}] | {c['signflip_p_two_sided']:.4f} | {c['win']}/{c['tie']}/{c['loss']} |")
    g = E["gate_vs_TA_on_fail_pairs"]
    L.append(f"\nGate vs TA on S2 pairs with ≥ 1 FAIL layer: n = {g['n']}; per pair (pp): " + (", ".join(f"{k.split('@')[0]} {v*100:+.3f}" for k, v in g["per_pair"].items()) or "—") +
             (f"; mean {g['mean']*100:+.3f} pp, sign-flip p = {g['signflip_p_two_sided']:.3f}" if "mean" in g else "") + ".")
    L.append(f"\nSelected-config boundary rate: {E['boundary_rate']}. Sanity (E3 TA at E1b λs = stage-2 values): max |Δ| = {E['sanity_TA_reproduces_stage2']:.2e}.\n")
L.append("\n### Method comparison (E1b stage-2 exploratory methods)\n")
for p in ("P", "R0", "S1"):
    if A[p].get("status") != "complete": continue
    L.append(f"**{p}**\n"); L.append("| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |"); L.append("|---|---:|---:|---|---|---:|")
    for k, e in A[p]["method_comparison"].items():
        if "mean_diff_vs_TA" in e: L.append(f"| {k} | {e['mean_norm_score']:.4f} | {e['mean_diff_vs_TA']:+.4f} | [{e['task_block_ci95'][0]:+.4f}, {e['task_block_ci95'][1]:+.4f}] | {e['win']}/{e['tie']}/{e['loss']} | {e['wilcoxon_p']:.3g} |")
        else: L.append(f"| {k} | {e['mean_norm_score']:.4f} | — | — | — | — |")
    L.append("")
L.append("### Cross-backbone (exploratory): Spearman over common task pairs\n")
for k, v in A.get("cross_backbone", {}).items():
    L.append(f"- {k}: " + (f"ρ = {f(v['rho'], 3)} (n = {v['n']})" if isinstance(v, dict) else str(v)))
# ---------------- E6: comparison with E4b (Qwen2.5-0.5B), pooled decoder estimate, location of sub-30° overlap, gate
X = json.loads((HERE / "e6_extra.json").read_text()) if (HERE / "e6_extra.json").exists() else {}
A4 = json.loads((E4B / "analysis_e4b.json").read_text()) if (E4B / "analysis_e4b.json").exists() else {}
L.append("\n## Replication vs E4b (Qwen2.5-0.5B) and pooled decoder estimate\n")
L.append("| population | predictor | ρ E4b (0.5B) | verdict E4b | ρ E6 (1.5B) | verdict E6 | ρ E6 − ρ E4b [joint task-block 95% CI] | pooled decoder ρ (mean) [joint CI] | joint perm p (1-sided) |")
L.append("|---|---|---:|---|---:|---|---|---|---:|")
PD = X.get("pooled_decoder", {})
for p in ("P", "R0", "S1"):
    if p not in PD: continue
    for h, c in (("H1", "O_A"), ("H2", "tv_cosine")):
        j = PD[p][h]; v4 = A4.get(p, {}).get("verdicts", {}).get(h, {}).get("verdict", "n/a"); v6 = A.get(p, {}).get("verdicts", {}).get(h, {}).get("verdict", "n/a")
        L.append(f"| {p}{' (primary)' if p == 'P' else ''} | {c} | {f(j['rho_E4b'], 3)} | {v4} | {f(j['rho_E6'], 3)} | {v6} | {j['diff_E6_minus_E4b']:+.3f} [{j['joint_boot_ci95_diff'][0]:+.3f}, {j['joint_boot_ci95_diff'][1]:+.3f}] | "
                 f"{f(j['pooled_mean_rho'], 3)} [{j['joint_boot_ci95_pooled'][0]:.3f}, {j['joint_boot_ci95_pooled'][1]:.3f}] | {f(j['joint_perm_p_one_sided_pooled'])} |")
if PD: L.append(f"\nCommon tasks: {len(PD.get('common_tasks', []))}. The pooled decoder estimate is **exploratory** (not pre-registered as a test in E4b; pre-specified in PREREG_E6 sec. 6b; no decision rule).")
if "error" in PD: L.append(f"\nPooled-decoder error: {PD['error']}")
LOC = X.get("location_lt30", {})
if LOC:
    L.append("\n## Where the sub-30° overlap sits (θ_min < 30° pair-layers)\n")
    L.append("| model | population | pair-layers | # < 30° | pairs with ≥1 | share layer-0 k_proj | share layer 0 | share last layer | share k_proj | share v_proj | pairs whose only <30° layer is L0 k_proj | top locations |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in ("e6", "e4b"):
        for p, v in LOC.get(m, {}).items():
            top = ", ".join(f"{k.replace('model.layers.', 'L')}:{c}" for k, c in list(v["top_locations"].items())[:5])
            L.append(f"| {'Qwen2.5-1.5B (E6)' if m == 'e6' else 'Qwen2.5-0.5B (E4b)'} | {p} | {v['n_pair_layers']} | {v['n_lt30']} | {v['n_pairs_with_lt30']}/{v['n_pairs']} | {f(v['share_layer0_k_proj'], 2)} | {f(v['share_layer0_any'], 2)} | {f(v['share_last_layer'], 2)} | {f(v['share_k_proj'], 2)} | {f(v['share_v_proj'], 2)} | {f(v['frac_pairs_lt30_only_in_layer0_k_proj'], 2)} | {top} |")
    for p, v in LOC.get("e6", {}).items():
        L.append(f"\nE6 {p}: <30° by module {v['by_module']}; by layer index {v['by_layer_idx']}; relative-depth quartile counts {v['rel_depth_quartiles_counts']}; min θ_min by module " + ", ".join(f"{k} {x:.1f}°" for k, x in v["min_theta_by_module"].items()))
G = X.get("gate", {})
if G:
    L.append("\n## Gate (θ★ = 30°): firing and benefit on gate-active pairs\n")
    L.append("| population | gate-active pairs | fail layers total | GATE@λ*TA − TA mean / median (pp) [task-block CI] | Wilcoxon p | W/T/L | GATE@own − TA mean (pp) [CI] | Wilcoxon p | W/T/L |")
    L.append("|---|---:|---:|---|---:|---|---|---:|---|")
    for p, g in G.items():
        a_, o_ = g.get("atTA"), g.get("own")
        if a_ is None:
            L.append(f"| {p} | {g['n_gate_active']}/{g['n_pairs']} | {g['n_fail_layers_total']} | — | — | — | — | — | — |"); continue
        ci = lambda e: f"[{e['task_block_ci95_pp'][0]:+.3f}, {e['task_block_ci95_pp'][1]:+.3f}]" if e.get("task_block_ci95_pp") else "—"
        L.append(f"| {p} | {g['n_gate_active']}/{g['n_pairs']} | {g['n_fail_layers_total']} | {a_['mean_pp']:+.3f} / {a_['median_pp']:+.3f} {ci(a_)} | {f(a_['wilcoxon_p_two_sided'], 3)} | {a_['win']}/{a_['tie']}/{a_['loss']} | {o_['mean_pp']:+.3f} {ci(o_)} | {f(o_['wilcoxon_p_two_sided'], 3)} | {o_['win']}/{o_['tie']}/{o_['loss']} |")
LC = HERE / "lam_confirm_e6.json"
if LC.exists():
    C = json.loads(LC.read_text())
    ci3 = lambda t, d=3: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
    L.append("\n## Confirmatory secondary hypotheses: λ rule U1 and merge decision M3 (frozen from e6_lambda; PREREG_E6.md sec. 6c)\n")
    L.append(f"Primary scope {C['primary_scope']}; populations used {C.get('populations_used')}; {C['n_pairs']} pairs; recompute check max |diff| = {C['recompute_maxdiff']:.1e}. "
             "Regret = 100·(D_test(λ_rule) − D_test(λ_sel)) pp on evaluation examples outside the unlabeled calibration sets; task-block bootstrap 95% CIs (2,000).\n")
    L.append("| hypothesis | rule | outcome |"); L.append("|---|---|---|")
    for h, v in C["decisions_primary_scope"].items(): L.append(f"| {h} | {C['hypothesis_rules'][h]} | **{v}** |")
    for sc, e in C["scopes"].items():
        for gname in ("G4", "G7"):
            g = e.get(gname)
            if not g: continue
            r = g["regret_pp"]
            L.append(f"\n{sc} {gname}: regret (pp) U1 {ci3(r['U1_agree'])}; λ=1 {ci3(r['W1_lam1.0'])}; λ=0.7 {ci3(r['fixed_lam0.7'])}; λ=0.5 {ci3(r['W2_lam0.5'])}; "
                     f"L-cal (labeled ref.) {ci3(r['Lcal_labeled_ref'])}; eval-optimal floor {ci3(r['evalopt_test'])}. U1−λ0.7 {ci3(g['U1_minus_fixed0.7_pp'])}; U1−λ1 {ci3(g['U1_minus_lam1.0_pp'])}; "
                     f"U1 removes {ci3(g['U1_removed_vs_lam1.0'])} of λ=1 loss, {ci3(g['U1_removed_vs_fixed0.7'])} of λ=0.7 loss; U1 λ counts {g['U1_lambda_counts']} (λ_sel {g['lam_sel_counts']}); U1 = λ_sel in {g['U1_match_sel']:.2f}.")
        for tau, m in e.get("M3", {}).items():
            L.append(f"\n{sc} M3 τ={tau}: prevalence {m['prevalence']:.3f}; AUROC {ci3(m['auroc'])}; frozen-threshold accuracy {ci3(m['acc_frozen_thr'])} vs always-merge {ci3(m['acc_always_merge'])} "
                     f"(diff {ci3(m['acc_diff_vs_always_merge'])}); unfitted D̂_U>τ accuracy {ci3(m['acc_unfitted_Dhat_gt_tau'])}; recall {m['recall_bad']:.2f}, precision {m['precision_bad']:.2f}.")
L.append("\n## Integrity (stage0, per seed; rule (b) not applicable)\n")
L.append("| task | seed | lr | steps | micro-bs | train loss (mean) | last logged loss | prior entropy | eval | (a) thr | (a) | (c) max diff / tol | (c2) hold bf16 / fp32 | valid |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
for s in (0, 1):
    p = HERE / f"s{s}" / "stage0.json"
    if not p.exists(): continue
    s0 = json.loads(p.read_text())
    for t, r in s0["per_task"].items():
        L.append(f"| {t} | {s} | {r['lr']:g} | {r['train_steps']} | {r['micro_batch']} | {r['train_loss']:.4f} | {f(r['loss_last_logged'])} | {f(r['label_prior_entropy'])} | {r['eval']['main']:.4f} | {r['a_threshold']:.4f} | {r['a_ok']} | {r['c_max_abs_logit_diff']:.1e} / {r['c_tol']:.1e} {r['c_ok']} | {r['c2_hold_bf16']:.4f} / {r['c2_hold_fp32']:.4f} {r['c2_ok']} | **{r['valid']}** |")
L.append("\n## Deviations and notes\n")
dv = HERE / f"DEVIATIONS_{NAME}.md"
L.append(dv.read_text() if dv.exists() else "None.")
L.append("\n## Compute\n")
L.append("Timing files (s): " + json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in A.get("timing", {}).items()}))
tl = HERE / "gpu_time_summary.json"
if tl.exists(): L.append("\nGPU wall-clock summary: " + tl.read_text())
(HERE / "VERDICT_E6_DECODER2.md").write_text("\n".join(L) + "\n")
print("wrote", HERE / "VERDICT_E6_DECODER2.md")
