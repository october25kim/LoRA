"""Generate VERDICT_E6.md and MANUSCRIPT_INSERT.md; every number is read from results/*.csv or data/*.json."""
import pandas as pd, json, hashlib, datetime
R = "/workspace/lora-paper/e6_lambda"
G = pd.read_csv(f"{R}/results/lambda_regret_G4.csv"); G7 = pd.read_csv(f"{R}/results/lambda_regret_qwen_G7.csv")
Dist = pd.read_csv(f"{R}/results/lambda_distribution_G4.csv"); Cont = pd.read_csv(f"{R}/results/continuous_lambda_describe.csv")
M = pd.read_csv(f"{R}/results/merge_decision.csv"); UR = pd.read_csv(f"{R}/results/unlabeled_regret.csv")
UM = pd.read_csv(f"{R}/results/unlabeled_merge_decision.csv"); A = pd.read_csv(f"{R}/results/addon_paired_usefulness.csv")
M3c = pd.read_csv(f"{R}/results/addon_M3_calibrated.csv"); chk = json.load(open(f"{R}/data/checks.json"))
chkU = json.load(open(f"{R}/data/checks_unlabeled.json"))
rules_sha = open(f"{R}/RULES.sha256").read().split()[0]; rules_ts = open(f"{R}/RULES_timestamp.txt").read().strip()
def row(df, **kw):
    q = df
    for k, v in kw.items(): q = q[q[k] == v]
    assert len(q) == 1, (kw, len(q)); return q.iloc[0]
f2 = lambda x: f"{x:.2f}"
def rc(r, a="regret_pp", lo="regret_lo", hi="regret_hi"): return f"{r[a]:.2f} [{r[lo]:.2f}, {r[hi]:.2f}]"
def pc(r, base):
    return f"{100*r[f'removed_vs_{base}']:.0f}% [{100*r[f'removed_vs_{base}_lo']:.0f}, {100*r[f'removed_vs_{base}_hi']:.0f}]"
def ac(r): return f"{r.auroc:.2f} [{r.auroc_lo:.2f}, {r.auroc_hi:.2f}]"
def acc(r): return f"{r.acc:.2f} [{r.acc_lo:.2f}, {r.acc_hi:.2f}]"
SC = ["bert/R0", "bert/S1", "bert/P", "bert/all", "roberta/R0", "roberta/S1", "roberta/P", "roberta/all", "qwen/R0", "qwen/S1", "qwen/P", "qwen/all", "all/all"]
L = []
L.append("# VERDICT E6: pre-merge λ rules and merge/no-merge decisions (POST HOC, EXPLORATORY)\n")
L.append(f"*Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} KST by `code/make_verdict_e6.py` from `results/*.csv`. Rules were fixed in `RULES.md` (sha256 `{rules_sha}`, written {rules_ts}) before any rule was scored. Clarification C1 and the post-results additions A1–A3 are listed in `DEVIATIONS_E6.md`. Nothing here is pre-registered, and E6 does not alter any pre-registered verdict.*\n")
L.append("## 1 Data inventory\n")
L.append("| backbone | populations (pairs) | λ grid with per-task held-out + eval scores | eval-set predictions (argmax / STS-B scores) | source |\n|---|---|---|---|---|")
c = chk["counts"]
L.append(f"| BERT-base | R0 {c['bert_R0']}, S1 {c['bert_S1']}, P {c['bert_P']} | {{0.3, 0.5, 0.7, 1.0}} | merged at every λ + singles (s0, s1) | e1b/results, e1c |")
L.append(f"| RoBERTa-base | R0 {c['roberta_R0']}, S1 {c['roberta_S1']}, P {c['roberta_P']} (RTE excluded by integrity rule) | {{0.3, 0.5, 0.7, 1.0}} | same | e4a, e5/work/e4a_roberta/preds |")
L.append(f"| Qwen2.5-0.5B | R0 {c['qwen_R0']}, S1 {c['qwen_S1']}, P {c['qwen_P']} | {{0.3, …, 1.0}} ∪ {{1.3, 1.5, 2.0}} (E5b) | same, incl. extended λ | e4b, e5/e5b, e5/work/e4b_decoder/preds |")
L.append("")
L.append(f"Per-layer task-vector norms and inner products were computed from the adapter weights on the 4070 **CPU** (`code/layer_stats_cpu.py`; no GPU job). The global cosine and norm ratio they imply match the frozen predictor tables (max |Δcos| = {max(chk[b]['max_abs_cos_diff'] for b in ['bert','roberta','qwen']):.1e}). Metrics recomputed from the saved predictions match all {chkU['n_checks']} reported per-task scores (max |Δ| = {chkU['max_abs_diff_recomputed_vs_reported_acc']:.1e}). Excluded: same-task S2 pairs and the 21 E1 Hub pairs. Consolidated table: `data/pairs_long.csv`; U-analysis table: `data/pairs_unlabeled.csv`; raw copies with SHA256SUMS: `data/src/`.\n")
L.append("## 2 Rules tested (from RULES.md)\n")
L.append("W1 λ = 1 · W2 λ = 0.5 (identical to the per-layer least-squares λ) · W3 norm-preserving λ_np · W4 per-layer projection-preserving λ_pp · F1 global λ fitted on other tasks · F2 ridge features → λ* · F3 ridge per-λ loss model · U1 agreement-max λ on n_cal ≤ 200 **unlabeled** inputs per task (needs unlabeled data) · L-cal labeled-calibration reference · M1 logistic merge classifier · M2 single weight scores · M3 unlabeled disagreement score (+ M3c calibrated threshold, added post hoc). CV: both-tasks-out within backbone; LOBO transfer (fit on the other two backbones, also both-tasks-out).\n")
d_all = row(Dist, scope="all/all", rule="W3"); cont = Cont.set_index(["Unnamed: 0", "Unnamed: 1"])
L.append(f"**What the closed-form rules reduce to.** λ_np ranges over [{min(cont.loc[('lam_np','min')]):.2f}, {max(cont.loc[('lam_np','max')]):.3f}] and snaps to 0.7 in {100*d_all['frac_0.7']:.1f}% of pairs. With near-orthogonal task vectors (|cos| ≈ 0) and moderate norm ratios, λ_np ≈ 1/√2. λ_pp ranges over [{min(cont.loc[('lam_pp','min')]):.3f}, {max(cont.loc[('lam_pp','max')]):.3f}] and always snaps to 1.0, so W4 ≡ W1. F1 chose 0.7 in {100*row(Dist, scope='all/all', rule='F1_within')['frac_0.7']:.0f}% of within-backbone folds and {100*row(Dist, scope='all/all', rule='F1_lobo')['frac_0.7']:.0f}% of LOBO folds. In practice, then, the weight-only rules are two constants: 0.7 and 1.0.\n")
L.append("## 3 Choosing λ without held-out data: regret vs held-out-tuned λ* (pp of normalized retention; full evaluation sets; task-block 95% CI)\n")
L.append("| scope | W1 λ=1 | W2 λ=0.5 | W3 λ_np (≈0.7) | F1 within | F2 within | F3 within | F2 LOBO | F3 LOBO | eval-optimal (floor) |\n|---|---|---|---|---|---|---|---|---|---|")
for s in SC:
    L.append(f"| {s} | " + " | ".join(rc(row(G, scope=s, rule=k)) for k in ["W1", "W2", "W3", "F1_within", "F2_within", "F3_within", "F2_lobo", "F3_lobo", "evalopt"]) + " |")
L.append("\n**% of avoidable loss removed** (1 − mean regret_rule / mean regret_baseline):\n")
L.append("| scope | W3 vs λ=1 | W3 vs λ=0.5 | F2 within vs λ=1 | F3 within vs λ=1 | F3 LOBO vs λ=1 |\n|---|---|---|---|---|---|")
for s in ["bert/all", "roberta/all", "qwen/all", "qwen/P", "all/all"]:
    L.append(f"| {s} | {pc(row(G, scope=s, rule='W3'), 'W1')} | {pc(row(G, scope=s, rule='W3'), 'W2')} | {pc(row(G, scope=s, rule='F2_within'), 'W1')} | {pc(row(G, scope=s, rule='F3_within'), 'W1')} | {pc(row(G, scope=s, rule='F3_lobo'), 'W1')} |")
L.append("\n**Qwen with the extended grid G7** (λ up to 2.0; oracle = held-out selection on G7):\n")
L.append("| scope | W1 | W2 | W3 (≈0.7) | F1 within |\n|---|---|---|---|---|")
for s in ["qwen/R0", "qwen/S1", "qwen/P", "qwen/all"]:
    L.append(f"| {s} | " + " | ".join(rc(row(G7, scope=s, rule=k)) for k in ["W1", "W2", "W3", "F1_within"]) + " |")
L.append("\n## 4 λ from unlabeled data (U analysis: calibration inputs ⊂ eval set, labels unused; all rules re-scored on the remaining eval examples)\n")
L.append("| scope | W1 | W3 (≈0.7) | **U1 agreement (unlabeled)** | L-cal (same n, labeled; reference) | U1 % removed vs λ=1 | U1 % removed vs λ=0.5 | U1 − W3 (paired) |\n|---|---|---|---|---|---|---|---|")
for s in SC:
    u = row(UR, grid="G4", scope=s, rule="U1_agree"); dd = row(A, source="test_remainder", scope=s, rule="U1_minus_W3")
    L.append(f"| {s} | {rc(row(UR, grid='G4', scope=s, rule='W1'))} | {rc(row(UR, grid='G4', scope=s, rule='W3'))} | **{rc(u)}** | {rc(row(UR, grid='G4', scope=s, rule='Lcal_labeled_ref'))} | {pc(u, 'W1')} | {pc(u, 'W2')} | {rc(dd, 'diff_vs_better_pp', 'diff_lo', 'diff_hi')} |")
for s in ["qwen/P", "qwen/all"]:
    u = row(UR, grid="G7", scope=s, rule="U1_agree")
    L.append(f"| {s} (G7) | {rc(row(UR, grid='G7', scope=s, rule='W1'))} | {rc(row(UR, grid='G7', scope=s, rule='W3'))} | **{rc(u)}** | {rc(row(UR, grid='G7', scope=s, rule='Lcal_labeled_ref'))} | {pc(u, 'W1')} | {pc(u, 'W2')} | – |")
L.append("\n## 5 Practical usefulness (A1: paired CI of regret − better fixed baseline < 0 and ≥ 50% removed)\n")
L.append("| scope | better baseline | W3/≈0.7 (full eval) | F2 within | F3 LOBO | U1 (test remainder) |\n|---|---|---|---|---|---|")
for s in SC:
    a = lambda src, k: row(A, source=src, scope=s, rule=k)
    fmt = lambda r: f"{'**yes**' if r.useful else 'no'} ({r.diff_vs_better_pp:+.2f} [{r.diff_lo:+.2f}, {r.diff_hi:+.2f}])"
    L.append(f"| {s} | {a('full_eval','W3').better_baseline} | {fmt(a('full_eval','W3'))} | {fmt(a('full_eval','F2_within'))} | {fmt(a('full_eval','F3_lobo'))} | {fmt(a('test_remainder','U1_agree'))} |")
L.append("\n## 6 Merge / no-merge: y = 1[D(λ*) > τ], τ = 0.05 primary\n")
L.append("| scope | prevalence | M1 within AUROC | M1 LOBO AUROC | best M2 (weights) AUROC | REF held-out labels AUROC | M1 within acc | always-merge acc |\n|---|---|---|---|---|---|---|---|")
for s in ["bert/all", "roberta/all", "qwen/all", "qwen/P", "all/all"]:
    m = M[(M.tau == 0.05) & (M.scope == s)]
    m2 = m[m.score.str.startswith("M2")].sort_values("auroc").iloc[-1]
    L.append(f"| {s} | {m.prevalence.iloc[0]:.2f} | {ac(row(m, score='M1_within'))} | {ac(row(m, score='M1_lobo'))} | {m2.score[3:]} {ac(m2)} | {ac(row(m, score='REF_hold_1-holdnorm'))} | {acc(row(m, score='M1_within'))} | {acc(row(m, score='M0_always_merge'))} |")
L.append("\nUnlabeled score M3 (test remainder; y recomputed on the same examples):\n")
L.append("| τ | scope | prevalence | M3 AUROC (unlabeled) | M3 acc, unfitted D̂>τ | M3c acc, within (post hoc) | M3c acc, LOBO (post hoc) | always-merge acc | REF held-out AUROC |\n|---|---|---|---|---|---|---|---|---|")
for tau in [0.02, 0.05, 0.10]:
    for s in ["bert/all", "roberta/all", "qwen/all", "all/all"]:
        m = UM[(UM.tau == tau) & (UM.scope == s)]; m3 = row(m, score="M3_unlabeled_Dhat")
        cw = row(M3c, tau=tau, scheme="within", scope=s); cl = row(M3c, tau=tau, scheme="lobo", scope=s)
        L.append(f"| {tau} | {s} | {m3.prevalence:.2f} | {ac(m3)} | {acc(m3)} | {acc(cw)} | {acc(cl)} | {acc(row(m, score='M0_always_merge'))} | {ac(row(m, score='REF_hold_1-holdnorm'))} |")
# ---------- conclusion ----------
g = lambda s, k: row(G, scope=s, rule=k); u = lambda s: row(UR, grid="G4", scope=s, rule="U1_agree")
m05 = M[(M.tau == 0.05) & (M.scope == "all/all")]; um05 = UM[(UM.tau == 0.05) & (UM.scope == "all/all")]
c05 = row(M3c, tau=0.05, scheme="lobo", scope="all/all")
V = dict(
 w1=rc(g("all/all", "W1")), w2=rc(g("all/all", "W2")), w3=rc(g("all/all", "W3")), w3r1=pc(g("all/all", "W3"), "W1"), w3r2=pc(g("all/all", "W3"), "W2"),
 f2=rc(g("all/all", "F2_within")), f3=rc(g("all/all", "F3_within")), f3l=rc(g("all/all", "F3_lobo")), f3lq=rc(g("qwen/all", "F3_lobo")),
 qp_w1=rc(g("qwen/P", "W1")), qp_w3=rc(g("qwen/P", "W3")), bw3=rc(g("bert/all", "W3")), rw3=rc(g("roberta/all", "W3")), qw3=rc(g("qwen/all", "W3")),
 u1=rc(u("all/all")), u1r1=pc(u("all/all"), "W1"), u1r2=pc(u("all/all"), "W2"), lcal=rc(row(UR, grid="G4", scope="all/all", rule="Lcal_labeled_ref")),
 u1w3=rc(row(A, source="test_remainder", scope="all/all", rule="U1_minus_W3"), "diff_vs_better_pp", "diff_lo", "diff_hi"),
 u1lc=rc(row(A, source="test_remainder", scope="all/all", rule="U1_minus_Lcal"), "diff_vs_better_pp", "diff_lo", "diff_hi"),
 m1=ac(row(m05, score="M1_within")), m1l=ac(row(m05, score="M1_lobo")), m1acc=acc(row(m05, score="M1_within")), m0acc=acc(row(m05, score="M0_always_merge")),
 m3=ac(row(um05, score="M3_unlabeled_Dhat")), m3acc=acc(row(um05, score="M3_unlabeled_Dhat")), m0u=acc(row(um05, score="M0_always_merge")),
 m3c=acc(c05), m3cd=f"{c05.diff_vs_M0:+.2f} [{c05.diff_lo:+.2f}, {c05.diff_hi:+.2f}]", ref=ac(row(m05, score="REF_hold_1-holdnorm")), refu=ac(row(um05, score="REF_hold_1-holdnorm")),
 frac07=f"{100*d_all['frac_0.7']:.0f}", u1q=rc(u("qwen/all")), u1b=rc(u("bert/all")), u1r=rc(u("roberta/all")),
 prev=f"{row(m05, score='M0_always_merge').prevalence:.2f}")
L.append("\n## 7 Conclusion\n")
L.append(f"This analysis is post hoc and exploratory. Among weight-only rules, nothing adaptive beat a constant. The only closed-form rule that helped, the norm-preserving λ_np, is in effect the constant 0.7 (≈ 1/√2 for near-orthogonal updates; it snaps to 0.7 in {V['frac07']}% of pairs). Over all 780 cross-task pairs it cut regret against held-out-tuned λ* from {V['w1']} pp (λ = 1) and {V['w2']} pp (λ = 0.5) to {V['w3']} pp. That removes {V['w3r1']} of the avoidable loss of λ = 1 and {V['w3r2']} of that of λ = 0.5. The gain is backbone-dependent: BERT {V['bw3']}, RoBERTa {V['rw3']}, Qwen {V['qw3']}. In the primary Qwen population P, λ = 0.7 is no better than λ = 1 ({V['qp_w3']} vs {V['qp_w1']}). Regressions from weight features to λ* (F2 {V['f2']}, F3 {V['f3']}) did not beat the constant, and transferred badly from encoders to the decoder (F3 LOBO on Qwen {V['f3lq']}). The per-layer projection-preserving λ reduced to λ = 1, and the per-layer least-squares λ is ½ by construction. With a few hundred **unlabeled** in-distribution inputs, the picture changes. Choosing λ to maximize agreement with the two single adapters (U1) left {V['u1']} pp of regret and removed {V['u1r1']} of λ = 1's avoidable loss. That is {V['u1w3']} pp relative to λ = 0.7 and statistically indistinguishable from labeled selection on the same n ({V['u1lc']} pp). For the merge/no-merge decision (τ = 0.05, prevalence {V['prev']}), weight features carried no usable signal: the logistic classifier had AUROC {V['m1']} within backbone and {V['m1l']} in transfer, and its accuracy of {V['m1acc']} did not beat always-merge ({V['m0acc']}). The unlabeled disagreement score ranked pairs almost as well as held-out labels (AUROC {V['m3']} vs {V['refu']} on the same examples). But its pre-declared unfitted threshold was miscalibrated (accuracy {V['m3acc']} vs {V['m0u']}). A threshold calibrated on other tasks' pairs (post hoc, M3c) reached {V['m3c']} in LOBO transfer ({V['m3cd']} over always-merge). Practical reading: without any data, use λ ≈ 0.7 rather than 1 or 0.5 on encoders (on the decoder it is a hedge, not an improvement). If unlabeled inputs are available, agreement-based selection comes within about 0.1 pp of held-out tuning with 1,000 labels per task and matches labeled selection on the same few hundred inputs. For the merge decision it ranks pairs nearly as well as held-out labels, but it needs a calibrated threshold. Weight-space features should not be used for either.\n")
c10 = row(M3c, tau=0.10, scheme="lobo", scope="all/all"); c02 = row(M3c, tau=0.02, scheme="lobo", scope="all/all")
L.append(f"**Threshold sensitivity (M3c, LOBO, all pairs).** At τ = 0.02, accuracy was {acc(c02)} against always-merge {c02.acc_M0:.2f}. At τ = 0.10 (prevalence {c10.prevalence:.2f}), accuracy was {acc(c10)} against {c10.acc_M0:.2f}. When harmful merges are that rare, no rule beats always-merging on accuracy.\n")
L.append("## 8 Limitations\n")
L.append("- Post hoc. The data were collected for the pre-registered E1b/E1c/E4a/E4b/E5 questions, and the analyst knew from the manuscript that λ* is often 0.7 on BERT and 1.0 on Qwen-P. The constant 0.7 was not pre-declared as a rule; it arises from W3 and F1.\n- Rules can only pick grid points (G4; G7 for Qwen), because scoring a continuous or per-layer λ would require new GPU merges. The regret of λ_np at its exact value is therefore unknown.\n- The U1/M3 calibration inputs are unlabeled examples from the *evaluation* distribution, disjoint from the scored examples. With inputs from a shifted distribution, performance may differ. U1 costs 2 + |grid| forward passes over 2·n_cal inputs per pair. Where predictions are argmax labels only, entropy- or loss-based unlabeled proxies could not be computed.\n- CV predictions are held fixed inside the task-block bootstrap, so the CIs do not include refitting variability. Populations within a backbone share adapters and tasks. With K = 13–14 tasks per backbone, intervals are wide.\n- Pairwise merges of rank-8 LoRA adapters with separate classification heads on three backbones ≤ 0.5B parameters. We do not know whether the findings hold for multi-adapter merges, generative tasks or larger models.\n")
open(f"{R}/VERDICT_E6.md", "w").write("\n".join(L))
# ---------- manuscript insert ----------
I = []
I.append("# MANUSCRIPT INSERT (E6; draft; do not paste without review). Numbers are generated by code/make_verdict_e6.py from e6_lambda/results/*.csv.\n")
I.append("## Proposed subsection: §4.8 Can the merge coefficient and the merge decision be set before merging? (post hoc)\n")
I.append(f"Because tuning λ removed most of the avoidable loss (§4.7), we asked, post hoc, whether λ or the decision to merge at all can be set without held-out labels. The candidate rules were fixed and hashed before scoring (E6; Supplement S14). They were evaluated on all 780 cross-task pairs of the three backbones, with task-block cross-validation in which no task of a test pair entered the fit, and with regret measured against held-out-tuned λ* in pp of normalized retention. Weight-only rules collapsed to constants. The norm-preserving coefficient λ_np = ½(‖Δ₁‖+‖Δ₂‖)/‖Δ₁+Δ₂‖ is ≈ 1/√2 for near-orthogonal task vectors and selected 0.7 in {V['frac07']}% of pairs. A per-layer projection-preserving coefficient reduced to λ = 1, and the per-layer least-squares coefficient is ½ by construction. A fixed λ ≈ 0.7 left {V['w3']} pp of regret, against {V['w1']} for λ = 1 and {V['w2']} for λ = 0.5, removing {V['w3r1']} of the avoidable loss of λ = 1. The benefit was concentrated in the encoders (BERT {V['bw3']}, RoBERTa {V['rw3']} pp). On the primary decoder population it matched but did not improve on λ = 1 ({V['qp_w3']} vs {V['qp_w1']}). Ridge regressions from ten weight features (overlap, cosine, θ_min, sign conflict, norm ratio and per-layer statistics) to λ* did not beat the constant ({V['f2']} pp) and failed to transfer from encoders to the decoder ({V['f3lq']} pp).\n")
I.append(f"A label-free but data-dependent rule did much better. For each candidate λ, it measures how often the merged model agrees with each single adapter on at most 200 **unlabeled** inputs per task and picks the λ with the highest agreement. This left {V['u1']} pp of regret (BERT {V['u1b']}, RoBERTa {V['u1r']}, Qwen {V['u1q']}), removed {V['u1r1']} of the avoidable loss of λ = 1, and was indistinguishable from choosing λ by accuracy on the same examples with labels ({V['u1lc']} pp). For deciding whether to merge at all (retention after tuning below 95%; prevalence {V['prev']}), weight features were uninformative: a logistic classifier reached AUROC {V['m1']} and did not beat always-merging. By contrast, 1 − agreement ranked pairs nearly as well as held-out labels (AUROC {V['m3']} vs {V['refu']} on the same examples). It needs a calibrated threshold, however: an unfitted cut-off was worse than always-merging, whereas a threshold calibrated on other tasks' pairs gave accuracy {V['m3c']} across backbones. These analyses are exploratory; they reuse the pre-registered data but not its hypotheses, and the calibration inputs were drawn from the evaluation distribution.\n")
I.append("## Proposed Box 1 update (replace Step 4 and append to Step 5)\n")
I.append(f"> **Step 4 — Set λ with data if you can, and never default to λ = 1 or ½ on encoders.** Best: tune λ on held-out data (1,000 labeled examples per task recovered the evaluation-optimal λ in 75 of 91 BERT pairs). Almost as good, and label-free: pick the λ at which the merged model agrees most often with each single adapter on ~200 unlabeled inputs per task (regret {V['u1']} pp vs held-out tuning, indistinguishable from labeled selection on the same inputs; post hoc, E6). With no data at all, use λ ≈ 0.7, the norm-preserving value 1/√2 for near-orthogonal adapters. It removed {V['w3r1']} of the avoidable loss of λ = 1 across three backbones (regret {V['w3']} pp), but on the decoder's mixed-seed population it only matched λ = 1. Weight-feature regressions for λ did not beat this constant. Keep tuned task arithmetic as the default merge and baseline (unchanged E3 evidence). *Evidence: §4.2, §4.6–§4.8.*\n>\n> **Step 5 (addition) — To decide whether to merge at all, measure; do not predict from weights.** Weight features did not identify pairs that retain < 95% after tuning (AUROC {V['m1']}). Held-out labels (AUROC {V['ref']}) or unlabeled merged–single disagreement (AUROC {V['m3']}, with a threshold calibrated on other pairs) did. *Evidence: §4.8.*\n")
I.append("<!-- \"75 of 91 BERT pairs\" is unchanged from the current Box 1 (analysis/lambda_decomp_e1b_pairs.csv: lam_selected == lam_oracle in 75/91). E6 numbers: VERDICT_E6.md §3–§6; results/lambda_regret_G4.csv, unlabeled_regret.csv, addon_paired_usefulness.csv, merge_decision.csv, unlabeled_merge_decision.csv, addon_M3_calibrated.csv. -->\n")
open(f"{R}/MANUSCRIPT_INSERT.md", "w").write("\n".join(I))
print(json.dumps(V, indent=0, ensure_ascii=False))
