#!/usr/bin/env python3
"""Writes VERDICT_E1C.md from analysis_e1c.json and the stage0 files (reporting only; no statistics computed here)."""
import json, sys
from pathlib import Path
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
E1B = Path(__file__).resolve().parent.parent / "e1b_confirmatory"
A = json.loads((HERE / "analysis_e1c.json").read_text())
f = lambda x, d=4: "nan" if x is None or x != x else f"{x:.{d}f}"
L = []
vp = A.get("verdicts_primary", {})
L.append(f"# E1c VERDICT — H1c (O_A): **{vp.get('H1c', {}).get('verdict', 'n/a')}**; H2c (task-vector cosine): **{vp.get('H2c', {}).get('verdict', 'n/a')}**\n")
L.append(f"Generated {A['generated']}. Pre-registration: `PREREG_E1C.md` / `prereg_e1c.json` (sha256 in `prereg_e1c.sha256`); deviations/notes: `DEVIATIONS_E1C.md`.\n")
L.append(f"Valid tasks K = {A['K']}: {', '.join(A['valid_tasks'])}. Seed-1 exclusions: {A['excluded_seed1'] or 'none'}. Predictor hash ok: {A['predictors_hash_ok']}. "
         f"Seed-0 re-check ok: {A['seed0_recheck_ok']}. Analysis-code equivalence on E1b (reproduces E1b analysis.json exactly): {A['analysis_equivalence_on_E1b']['ok']}.\n")
def vt(res, tag):
    out = ["| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |", "|---|---|---:|---|---:|---:|---:|---|"]
    for h, v in res["verdicts"].items():
        out.append(f"| {h if tag == 'P' else h.replace('c', '') + ' (S1)'} | {v['predictor']} | {f(v['rho'])} | [{f(v['boot_ci95'][0], 3)}, {f(v['boot_ci95'][1], 3)}] | {f(v['perm_p_one_sided'])} | {f(v['holm_p'])} | {f(v['loto_frac_gt_0.2'], 2)} | **{v['verdict']}** |")
    return out
L.append("## Primary: mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)\n")
P = A["P"]
if P.get("status") == "complete":
    L += vt(P, "P")
    L.append("\nRule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).")
    for h, v in P["verdicts"].items():
        L.append(f"\n{h} criteria: {v['criteria']}. LOTO ρ: " + ", ".join(f"{t} {x:.2f}" for t, x in v["loto"].items()))
    L.append(f"\nP: {P['D_summary']['n']} pairs; D mean {f(P['D_summary']['mean'])}, median {f(P['D_summary']['median'])}, range [{f(P['D_summary']['min'])}, {f(P['D_summary']['max'])}]; "
             f"λ* counts {P['lam_selected_counts']}; ρ(O_A, tv_cos) = {f(P['rho_OA_tvcos'], 3)}; min θ_min over pairs/layers = {f(P['min_theta_min_overall_deg'], 1)}°; gate active in {P['gate_active_pairs']} pairs.\n")
    L.append(f"E1b (R0, shared seed 0) for comparison, recomputed with this code: " + "; ".join(f"{h.replace('c', '')} {v['verdict']} ρ={f(v['rho'])} CI [{f(v['boot_ci95'][0], 3)}, {f(v['boot_ci95'][1], 3)}]" for h, v in A["R0_recomputed"].items()) + "\n")
else:
    L.append(f"P status: {P.get('status')}\n")
L.append("## Secondary (exploratory; no multiplicity correction)\n")
L.append("### S1: seed-1-only cross-task pairs (shared seed 1; E1b rule applied descriptively)\n")
S1 = A["S1"]
if S1.get("status") == "complete":
    L += vt(S1, "S1")
    L.append(f"\nS1 D mean {f(S1['D_summary']['mean'])}, range [{f(S1['D_summary']['min'])}, {f(S1['D_summary']['max'])}]; λ* counts {S1['lam_selected_counts']}; min θ_min {f(S1['min_theta_min_overall_deg'], 1)}°; gate active in {S1['gate_active_pairs']} pairs.\n")
else:
    L.append(f"S1 status: {S1.get('status')}\n")
L.append("### Secondary predictors (P and S1)\n")
L.append("| predictor | ρ P | CI P | ρ S1 | CI S1 |"); L.append("|---|---:|---|---:|---|")
if P.get("status") == "complete":
    for c, s in P["all_predictor_stats"].items():
        s1 = S1.get("all_predictor_stats", {}).get(c, {"rho": float("nan"), "boot_ci95": [float("nan")] * 2})
        L.append(f"| {c} | {f(s['rho'], 3)} | [{f(s['boot_ci95'][0], 2)}, {f(s['boot_ci95'][1], 2)}] | {f(s1['rho'], 3)} | [{f(s1['boot_ci95'][0], 2)}, {f(s1['boot_ci95'][1], 2)}] |")
    L.append(f"\nFixed-λ ρ (P): {json.dumps({k: {l: round(x, 3) for l, x in v.items()} for k, v in P['fixed_lambda_rho'].items()})}\n")
L.append("### Reliability across seed configurations (Spearman over the same unordered task pairs)\n")
L.append("| quantity | R0~P | R0~S1 | P~S1 |"); L.append("|---|---:|---:|---:|")
for q in ("D", "O_A", "tv_cosine", "lam_selected"):
    g = lambda k: A["reliability"].get(f"{q}:{k}", {}).get("rho", float("nan"))
    L.append(f"| {q} | {f(g('R0~P'), 3)} | {f(g('R0~S1'), 3)} | {f(g('P~S1'), 3)} |")
L.append("\n### O_A: shared-seed vs mixed-seed\n")
OA = A["O_A_shared_vs_mixed"]
L.append(f"Random rank-8 null: O_A mean {OA['null_O_A_mean']:.5f} (sd {OA['null_O_A_sd']:.5f}, p95 {OA['null_O_A_p95']:.5f}).\n")
L.append("| population | n | O_A mean | sd | median | IQR | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layers θ_min<30° (total) |")
L.append("|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|")
for k, v in OA.items():
    if not isinstance(v, dict) or "O_A" not in v: continue
    o = v["O_A"]; z = v["null_z_O_A"]
    L.append(f"| {k} | {o['n']} | {o['mean']:.5f} | {o['sd']:.5f} | {o['median']:.5f} | [{o['q25']:.5f}, {o['q75']:.5f}] | {o['min']:.5f} | {o['max']:.5f} | "
             f"[{z['min']:.1f}, {z['max']:.1f}] | {v['frac_O_A_above_null_p95']:.2f} | {v['tv_cosine']['mean']:.4f} | {v['min_theta_min_A_deg']['min']:.1f} | {v['n_layers_lt30_total']} |")
for k in ("paired_P_minus_R0", "paired_P_minus_S1"):
    v = OA[k]; L.append(f"\n{k}: n={v['n']}, mean diff {v['mean_diff']:.5f}, median {v['median_diff']:.5f}, P lower in {v['frac_P_lower']:.0%} of task pairs, ratio of means {v['ratio_mean']:.3f}, Wilcoxon p = {v['wilcoxon_p']:.2e}")
L.append("\n### S2: same-task seed pairs (t@s0 × t@s1)\n")
S2 = A["S2"]
L.append(f"Gate (θ★ = 30°) fires in **{S2['n_pairs_gate_fires']}/{len(S2['per_pair'])}** same-task pairs; min θ_min over all S2 pairs/layers = {S2['theta_min_over_all_S2']:.2f}°.\n")
L.append("| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | layers < 30° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |")
L.append("|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for e in S2["per_pair"]:
    L.append(f"| {e['task']} | {e['min_theta_min_deg']:.2f} | {e['argmin_layer']} | {e['n_layers_lt30']} | {e['n_layers_lt45']} | {', '.join(f'{k} {v}' for k, v in e['layers_lt30'].items()) or '—'} | "
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
    L.append(f"\nSelected-config boundary rate: {E['boundary_rate']}. Sanity (E3 TA at E1b λs = e1c stage2 values): max |Δ| = {E['sanity_TA_reproduces_e1c_stage2']:.2e}.\n")
    L.append("| task | TA | PICO_TA | GATE | FORCEGATE | GATE − TA (pp) | PICO − TA (pp) | FORCEGATE − TA (pp) | GATE fail layers | TA sel | PICO sel |"); L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in E["per_pair"]:
        L.append(f"| {r['pair'].split('@')[0]} | {r['TA']:.4f} | {r['PICO_TA']:.4f} | {r['GATE']:.4f} | {r['FORCEGATE']:.4f} | {(r['GATE']-r['TA'])*100:+.3f} | {(r['PICO_TA']-r['TA'])*100:+.3f} | {(r['FORCEGATE']-r['TA'])*100:+.3f} | {r['GATE_n_fail_layers']} | {r['TA_sel']} | {r['PICO_TA_sel']} |")
L.append("\n### Method comparison (E1b stage-2 exploratory methods)\n")
for p in ("P", "S1"):
    if A[p].get("status") != "complete": continue
    L.append(f"**{p}**\n"); L.append("| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |"); L.append("|---|---:|---:|---|---|---:|")
    for k, e in A[p]["method_comparison"].items():
        if "mean_diff_vs_TA" in e: L.append(f"| {k} | {e['mean_norm_score']:.4f} | {e['mean_diff_vs_TA']:+.4f} | [{e['task_block_ci95'][0]:+.4f}, {e['task_block_ci95'][1]:+.4f}] | {e['win']}/{e['tie']}/{e['loss']} | {e['wilcoxon_p']:.3g} |")
        else: L.append(f"| {k} | {e['mean_norm_score']:.4f} | — | — | — | — |")
    L.append("")
# integrity
L.append("## Integrity (seed-1 adapters; E1b stage0 unmodified)\n")
s1 = json.loads((HERE / "s1" / "stage0.json").read_text()) if (HERE / "s1" / "stage0.json").exists() else None
s0 = json.loads((E1B / "stage0.json").read_text())
if s1:
    L.append("| task | steps | train loss | eval (s1) | eval (s0, E1b) | (a) thr | (a) | (b) ref−5pp | (b) | (c) max diff | valid |"); L.append("|---|---:|---:|---:|---:|---:|---|---|---|---:|---|")
    for t, r in s1["per_task"].items():
        b = f"{r['b_metric']} {r['b_ours']:.4f} ≥ {r['b_threshold']:.4f}" if r.get("b_ok") is not None else "n/a"
        L.append(f"| {t} | {r['train_steps']} | {r['train_loss']:.4f} | {r['eval']['main']:.4f} | {s0['per_task'][t]['eval']['main']:.4f} | {r['a_threshold']:.4f} | {r['a_ok']} | {b} | {r['b_ok']} | {r['c_max_abs_logit_diff']:.1e} | **{r['valid']}** |")
L.append("\n## Deviations and notes\n")
if (HERE / "DEVIATIONS_E1C.md").exists(): L.append((HERE / "DEVIATIONS_E1C.md").read_text())
L.append("\n## Compute\n")
L.append("Timing files (s): " + json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in A.get("timing", {}).items()}))
tl = HERE / "gpu_time_summary.json"
if tl.exists(): L.append("\nGPU wall-clock summary: " + tl.read_text())
(HERE / "VERDICT_E1C.md").write_text("\n".join(L) + "\n")
print("wrote", HERE / "VERDICT_E1C.md")
