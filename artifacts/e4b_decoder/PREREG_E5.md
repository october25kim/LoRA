# E5 — "Raise acceptance odds" add-on experiments on the E4b Qwen2.5-0.5B adapters — PRE-REGISTRATION (addendum)

Written 2026-09-27 (KST). This file and `prereg_e5.json` (machine-readable, authoritative for parameters) are hashed in `prereg_e5.sha256`,
committed to the local git repository on ubuntu-4070 (no push), and **never edited**. Later changes go to `DEVIATIONS_E5.md` (append-only,
timestamped KST, with a reason). Directory: `artifacts/e5_accept/` on ubuntu-4070; mirror `/workspace/lora-paper/e5/` (box).
The E4b folders (`artifacts/e4b_decoder/`, `artifacts/e3_baselines/`, `artifacts/e4a_roberta/`) are read-only for E5.

## 0. Disclosure (important)
**This addendum is written AFTER the E4b verdict was seen.** Known when writing:
- E4b (Qwen2.5-0.5B, `VERDICT_E4B.md`): H1 (O_A) and H2 (tv_cosine) **INCONCLUSIVE** on P (ρ = 0.345 / 0.368, Holm p = 0.030 / 0.024,
  task-block CIs include 0); R0/S1/pooled INCONCLUSIVE. P: TA λ\* counts on {0.3, 0.5, 0.7, 1.0} = {1, 1, 41, 48} (λ\* = 1.0, the grid top,
  in 48/91 = 53%). Gate (θ★ = 30°) active in 66/91 P pairs. E4b stage-2 exploratory method comparison on P: TIES-lite@λ\*TA −3.0 pp,
  TIES-lite@own-λ (grid ≤ 1.0) −2.1 pp, gate30 ≈ 0.0 pp vs TA. E3 code on S2 (14 same-task pairs): GATE +0.01, FORCEGATE +0.08, PICO_TA −0.22 pp vs TA.
- E3 on E1b (BERT, 91 pairs): no method beat TA; PICO_TA +0.22 pp (Holm sign-flip p = 0.033, below the 0.5 pp threshold); TIES −1.51 pp.
- E4a (RoBERTa): INCONCLUSIVE/FAIL-type results as in `VERDICT_E4A.md`; E1b/E1c noise-ceiling analysis (`analysis/NOISE_CEILING.md`).
- **Not known:** any E5 number. No E5a/E5b merge, no E5c bootstrap, no E5d lemma quantity on Qwen had been computed. The only E5 code run
  before hashing is a pipeline smoke test on the THROWAWAY E4b pilot adapters (`pilot/lr0.0001/sst2` × `pilot/lr0.001/rte`, not study adapters,
  singles set to 1.0), which produces no study numbers.
- Because the E4b results were known, E5 is **confirmatory only in the limited sense of a fixed, hashed analysis plan**; the choice to run it
  was motivated by E4b (λ\* at the grid top; reviewers' request for strong baselines on the decoder). Results are reported whatever they are.

## 1. Hypotheses
- **E5a (primary of E5).** Q: under an equal held-out tuning budget (8 configurations each), does any of PICO_TA (Pico-style soft reweighting),
  GATE (hard θ★ = 30° gate), FORCEGATE (threshold-free k = 1 gate), TIES (k = 20 trim = E1b "TIES-lite" trim, official E3 implementation) or
  PICO_GATE (gate, then Pico) beat tuned TA on the Qwen primary population P? Expectation stated in advance: none beats TA by ≥ 0.5 pp
  (null-leaning, based on E3/E4b). Either outcome is reported.
- **E5b (pre-registered sensitivity).** Extending the E4b TA λ grid above 1.0 (λ ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0}) changes D and may change
  ρ(O_A, D) and ρ(tv_cosine, D). This is a sensitivity analysis; **the E4b primary verdict stays as registered**.
- **E5c (reliability, descriptive).** How much of the between-pair variance of D (E4a and E4b) is evaluation-sampling noise?
- **E5d (mechanism, descriptive).** On Qwen FAIL layers, are the shared-direction components X1 = UᵀΔW1, X2 = UᵀΔW2 collinear (gate ≈ a
  coefficient change, Prop. 2 of `paper/LEMMA_PROJECTION.md`) or not (gate = coefficient change + deletion of R, Prop. 3)?

## 2. Populations
- E5a: **P** = the 91 mixed-seed cross-task pairs of E4b (`populations_e4b.json`, t1 = alphabetically first task @ seed 0, t2 = other @ seed 1;
  the gates project t2). **All 91 pairs are planned** (projected ≈ 3–4.5 GPU-h). Pairs run in the order `default_rng(20260927).permutation(91)`;
  a deadline guard stops starting new pairs at launch + 7.0 h. If it triggers, the completed prefix of this seeded order is the
  **pre-registered random subset** and is analysed with the same rule (n reported).
- E5b: P (sensitivity of the primary population), then R0 and S1 (secondary, descriptive), in that order, deadline launch + 11.5 h.
- E5c: E4a P/R0/S1 (78 pairs each) and E4b P/R0/S1 (91 pairs each).
- E5d: every layer with θ_min < 30° (definition A) in P, R0, S1, S2 (from the frozen `predictors_layers_e4b.csv`), plus the forced k = 1
  direction on all 168 layers of the 91 P pairs.

## 3. Methods (E5a)
Implementation: `e3.run()` and `merges.py` of E3 **unmodified** (sha256 recorded in `e5a/run_config.json`), the Qwen encoder `e4b.Enc`
(bf16 merged weights = bf16(W0 + Δ), fp32 head, eval batch 128), E4b singles (stage0) for normalization, E4b held-out sets (1,000 TRAIN
examples per task) for selection and E4b eval sets for the final score. Normalized pair score = ½(merged_t1/single_t1 + merged_t2/single_t2).
Grids (8 configurations each = equal tuning budget; order = tie-break order):

| method | definition | grid |
|---|---|---|
| TA | λ(ΔW1 + ΔW2) | λ ∈ {0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5} (= the registered E3 v1 TA grid; contains the E4b grid) |
| PICO_TA | Pico calibration of [B1, B2], γ-rescale, × c (E3 definition) | c ∈ {0.42, 0.71, 0.99, 1.2, 1.41, 1.63, 1.84, 2.12} = √2 × the TA grid (for near-orthogonal equal-norm pairs c·γ ≈ c/√2, so this matches the TA λ range) |
| GATE | hard θ★ = 30° project-off of t2 on FAIL layers, then λ(ΔW1 + ΔW2g) (E1b/E3 definition; ≡ TA on pairs without a FAIL layer) | TA grid |
| FORCEGATE | project t2 off the k = 1 most-aligned principal vector of orth(B1) on every layer (E3 A2) | TA grid |
| TIES | top-20% magnitude trim per task (global over the task's 168 LoRA matrices), sign election, disjoint mean, × λ (E3 `ties`; same trim as E1b TIES-lite) | k = 20, λ ∈ {0.7, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0} (E4b TIES-lite was grid-limited at λ = 1.0) |
| PICO_GATE | hard θ★ = 30° gate on t2 (B2 ← (I − SSᵀ)B2, ΔW2 ← (I − SSᵀ)ΔW2 on FAIL layers), then PICO_TA on the gated pair (γ uses the gated updates); ≡ PICO_TA on pairs without a FAIL layer. Wrapper around `merges.merge_model` in `e5.py` | PICO_TA grid |

Also recorded (not in the family): TA restricted to the E4b grid {0.3, 0.5, 0.7, 1.0} (`TA4`), untuned Pico c = 1.
Sanity (checked before interpretation): E3 TA at λ ∈ {0.3, 0.5, 0.7, 1.0} must reproduce the E4b stage-2 eval values exactly (|Δ| ≤ 1e-9)
and the E4b λ\*.

## 4. Other experiments
- **4b. E5b.** For each pair, TA at λ ∈ {1.3, 1.5, 2.0}: held-out and eval (E4b stage-2 arithmetic `W0 + λ(ΔW1 + ΔW2)`, same Enc/batch).
  λ₊\* = first max of the held-out normalized score over {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0} (E4b values reused for λ ≤ 1.0).
  D₊ = 1 − normalized eval score at λ₊\*. Sanity: λ = 1.0 recomputed for the first 2 P pairs must equal E4b stage 2; E5a TA held-out at
  λ = 1.3/1.5 must equal E5b's. Analysis: `e4b_analysis.pop_stats` (unmodified; 10,000-relabeling permutation p, Holm over {H1, H2},
  task-block bootstrap 2,000, LOTO) with D₊; the E4b rule (PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; FAIL iff bootstrap upper
  < 0.3; else INCONCLUSIVE) is applied **descriptively**. Also: λ₊\* counts, share at the new top (2.0), Spearman(D, D₊), fixed-λ ρ at 1.3/1.5/2.0.
- **4c. E5c** (`e5c_reliability.py`, CPU, box). Saved per-example eval predictions at the original λ\* (never reselected). (i) Bootstrap
  R = 1,000 (seed 20260927; indices shared by single and merged predictions and across pairs with that task): per-pair SE of D;
  independent-replicate rank correlation and Spearman–Brown (= the E1b/E1c `NOISE_CEILING` method); (ii) true split-half: 200 random
  disjoint halves per task (seed 20260928), Spearman over pairs of D(half 1) vs D(half 2), Spearman–Brown; (iii) distribution of ρ(O_A, D\*)
  and ρ(tv_cosine, D\*) over the bootstrap replicates (SD, 2.5/97.5 percentiles, P(ρ ≥ 0.4)); (iv) 1 − mean(SE²)/Var(D) and √SB.
  Check: D recomputed from the saved predictions equals the reported D.
- **4d. E5d** (`e5.py --stage e5d`, CPU). For each FAIL layer: U = qr(orth(B1) Π_J), J = {θ_j < 30°}; `lemma_stats` from `paper/verify_lemma.py`
  (copied unmodified, hash recorded): c, cos φ = ⟨X1, X2⟩/(‖X1‖‖X2‖) (collinearity), μ = 1/(1 + c), ε = ‖R‖/‖(1 + c)X1‖,
  **fraction of the gate edit that is not a coefficient change** = ‖R‖²/‖X2‖² = 1 − cos²φ, edit size relative to ‖M_sum‖, entrywise sign flips.
  Same for the forced k = 1 direction on all P layers. Summaries: quartiles, share with c > 0, share with |cos φ| ≥ 0.9 / ≥ 0.5, by layer type.

## 5. Analysis code and decision rule (E5a)
- Code: `e5_analysis.py` (imports `e4b_analysis.pop_stats`, `e3.holm`, `e3.task_block_boot`, `e3.signflip` unmodified). Per-pair gain
  g_p = score_m(p) − score_TA(p), each method at its own held-out-selected configuration.
- **Test:** one-sided Wilcoxon signed-rank, `scipy.stats.wilcoxon(g, zero_method="wilcox", alternative="greater")` (p = 1 if all gains are 0),
  **Holm over the 5 methods**. **Decision (same thresholds as E3): "m beats TA" iff mean gain ≥ +0.5 pp (0.005) AND Holm-adjusted p < 0.05.**
  Note: E3 used a pair-level sign-flip permutation test with the same thresholds; as instructed by the study owner, E5 uses the Wilcoxon test
  as the decision test. The E3 sign-flip test (with Holm) is reported alongside; any disagreement between the two is reported explicitly.
- Reported per method: mean/median gain, task-block bootstrap 95% CI (2,000, `default_rng(0)`), W/T/L, two-sided Wilcoxon p, boundary rate,
  selected configurations. Descriptive: "a ≥ 0.5 pp gain is not supported by the CI" when the CI upper bound < 0.5 pp.
- Secondary (no multiplicity correction): GATE − TA on gate-active pairs; GATE − PICO_TA, FORCEGATE − PICO_TA, PICO_GATE − PICO_TA (all and
  gate-active), PICO_GATE − GATE; TA(8) − TA4; untuned Pico c = 1 − TA. Exploratory: Spearman of O_A / tv_cosine with each method's gain.
- Caveat stated in advance: pairs share tasks; the Wilcoxon and sign-flip tests treat pairs as independent. The task-block CI is the
  dependence-robust quantity.

## 6. Compute and operations
- GPU jobs run **serially**, one process at a time (a parallel stage-2 run previously ran out of memory at ≈ 13 GB): E5a (mem-frac 0.9),
  then E5b (P → R0 → S1). E5d and E5c are CPU-only and may run concurrently. Target ≤ 12 GPU-hours; priority E5a > E5b(P) > E5c > E5d > E5b(R0, S1).
- `e5.py` refuses to run unless `prereg_e5.sha256` and `code_e5.sha256` match. Code files: `e5.py`, `e5_analysis.py`, `e5c_reliability.py`,
  `verify_lemma.py`, `run_e5.sh`, `make_verdict_e5.py` (verdict writer; may be added later, formatting only — logged in DEVIATIONS if so).
- Outputs: `VERDICT_E5.md`, `DEVIATIONS_E5.md`, compact results (no adapters/caches) mirrored to box `/workspace/lora-paper/e5/`. No push, no upload.
