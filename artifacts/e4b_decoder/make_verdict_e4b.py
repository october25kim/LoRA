#!/usr/bin/env python3
"""Writes VERDICT_E4B.md from analysis_e4b.json, stage0 files, pilot decision and timing (reporting only; no statistics computed here).
Copy of E4a make_verdict_e4a.py (adapted from E1c make_verdict_e1c.py) for the Qwen2.5-0.5B study."""
import json, sys
from pathlib import Path
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
TAG, NAME = "e4b", "E4B"
A = json.loads((HERE / f"analysis_{TAG}.json").read_text())
f = lambda x, d=4: "nan" if x is None or x != x else f"{x:.{d}f}"
L = []
vp = A.get("verdicts_primary", {})
PP = A.get("primary_population", "P")
L.append(f"# {NAME} VERDICT (Qwen2.5-0.5B decoder) — H1 (O_A): **{vp.get('H1', {}).get('verdict', 'n/a')}**; H2 (task-vector cosine): **{vp.get('H2', {}).get('verdict', 'n/a')}**  (primary population {PP}{', mixed seed' if PP == 'P' else ', seed 0'})\n")
L.append(f"Generated {A['generated']}. Pre-registration: `PREREG_{NAME}.md` / `prereg_{TAG}.json` (sha256 in `prereg_{TAG}.sha256`); deviations/notes: `DEVIATIONS_{NAME}.md`.\n")
L.append(f"Valid tasks K = {A['K']}: {', '.join(A['valid_tasks'])}. Exclusions: {A['excluded'] or 'none'}. Predictor hash ok: {A['predictors_hash_ok']}. "
         f"Analysis-code equivalence on E1b (reproduces E1b analysis.json): {A['analysis_equivalence_on_E1b'].get('ok')}.\n")
pd_ = HERE / "pilot" / "pilot_decision.json"
if pd_.exists():
    D = json.loads(pd_.read_text())
    pj = D.get("projection", {})
    L.append(f"Pilot projection: train {f(pj.get('train_h_per_seed'), 2)} h/seed, core (2 seeds) {f(pj.get('core_h_two_seeds'), 2)} h vs threshold {pj.get('threshold_h')} h -> seeds {D.get('seeds')}, primary {D.get('primary_population')}.\n")
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
for p, lab in [x for x in (("R0", "seed-0 cross-task pairs (E1b confirmatory design on Qwen; rule applied descriptively)"), ("S1", "seed-1 cross-task pairs (rule applied descriptively)")) if x[0] != PP]:
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
(HERE / f"VERDICT_{NAME}.md").write_text("\n".join(L) + "\n")
print("wrote", HERE / f"VERDICT_{NAME}.md")
