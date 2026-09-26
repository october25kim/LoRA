# E3 PRE-REGISTRATION: strong LoRA-merge baselines vs task arithmetic; hard θ★-gate vs Pico soft B-space reweighting

Written 2026-09-25 (KST), **before any E3 computation on the E1b adapters**. At writing time E1b training and evaluation
were still in progress. No E1b adapter had been loaded, merged, or scored by any E3 code. This file is hashed in
`PREREG_E3.sha256` and is never edited. Later changes go to `PREREG_E3_AMENDMENTS.md` (dated, with a reason, and written
before the E1b run is started). `e3.py --setting e1b` refuses to run if this file does not match its hash.

Order of work: grids and rules were fixed here first. After that, the exploratory pilot on the 21 E1 Hub pairs was run
(`pilot_e1/`, clearly labelled PILOT). Pilot results do not change anything in this file.

## 0. Framing
The project is now a study of whether weight-space predictors forecast LoRA merge loss (E1/E1b), and of hard gating vs
soft reweighting of shared directions. E3 asks:
- **Q1 (primary).** Does any strong merge baseline beat task arithmetic (TA) on normalized pair score when every method
  gets the same held-out tuning budget?
- **Q2 (secondary).** How does the hard θ★ = 30° project-off gate compare with Pico-style soft shrinkage of shared B-space
  directions?
- **Q3 (exploratory).** Do the E1b weight-space predictors (O_A, O_B, task-vector cosine) forecast which pairs a method helps?

## 1. Units, adapters, data
- Adapters: the 16 E1b adapters. Details are in `~/Desktop/Workspace/LoRA/artifacts/e1b_confirmatory/PREREG.md` (read-only),
  with deviation D1 (full-data MNLI and QQP) in `deviations.md` there. Setup: BERT-base-uncased @86b5e09, LoRA r = 8,
  α = 16 (ΔW = 2BA), 73 LoRA layers (12 × {q, k, v, attn.out, intermediate, output} + pooler), per-task classifier heads,
  standard BERT token_type_ids (the E1b convention).
- Units: **all cross-task pairs of the valid E1b tasks**, i.e. those in E1b `stage0.json:valid_tasks` (up to 120 pairs). E3 is
  not run if E1b stage 0 failed (fewer than 12 valid tasks). Pair order and "t2" (the second task, which the gate projects)
  follow E1b `pairs_of`.
- **Tuning uses only the E1b 1,000 held-out TRAIN examples per task** (seed-0 hold-out, never trained on; `hold` split of the
  E1b cache; the index sha256 is checked against each adapter's `train_meta.json`).
- Final scores use the E1b eval sets (seeded ≤ 10k subsets, as in E1b PREREG §1). Each task is scored with its own head.
  Precision: fp32 with TF32 off; batch size 256 (as in E1b).
- Metric: accuracy (STS-B: Spearman), following E1b. Single-adapter reference values come from E1b `stage0.json`
  (hold and eval `main`).
- **Normalized pair score** = ½ (merged_t1/single_t1 + merged_t2/single_t2), on the eval set, for each method at its selected config.

## 2. Methods (implementation: `merges.py`; unit tests: `test_merges.py`)
All methods operate on the 73 per-layer updates ΔW_t = 2 B_t A_t. The merged model is W0 + merged ΔW. Every method has a
global scale (λ, or c for Pico). The tuning grid is **8 configurations for TA and every primary method**.

| method | definition (summary) | grid (order = tie-break order) |
|---|---|---|
| TA | λ(ΔW1+ΔW2) | λ ∈ {0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5} (8; ⊃ E1b grid) |
| TIES | official TIES: keep top-k% by magnitude per task (threshold global over the task's 73 LoRA matrices; official `kthvalue` rule); elect sign = sign of summed trimmed values; disjoint mean; × λ | k=10: λ ∈ {1.5, 2.0, 2.5}; k=20: λ ∈ {1.0, 1.5, 2.0}; k=30: λ ∈ {1.0, 1.5} (8) |
| DARE_TA | DARE: drop each entry w.p. p, rescale by 1/(1−p) (torch.Generator seed 0); then TA | p ∈ {0.5, 0.9} × λ ∈ {0.5, 0.7, 1.0, 1.3} (8) |
| DARE_TIES | DARE (as above), then full TIES with k = 20 (MergeLM `mask_apply_method=ties_merging`, default trim rate 0.8) | p ∈ {0.5, 0.9} × λ ∈ {0.7, 1.0, 1.5, 2.0} (8) |
| TSVM | TSV-M (official `compute_and_sum_svd_mem_reduction`): per-task SVD, keep k components per task, concatenate, orthogonalize U and V by polar factor, U⊥ diag(S) V⊥ᵀ, × λ. kmode `lora`: k = r = 8 (non-null singular vectors only); `official`: k = int(min(d_out, d_in)/2) | kmode ∈ {lora, official} × λ ∈ {0.5, 0.7, 1.0, 1.3} (8) |
| KNOTS | KnOTS-TIES (official SVDMerger): SVD of [ΔW1 \| ΔW2] (fp64, keep s > 1e-5), task reps diag(s)V_t; TIES mask on reps (top-K% per task, global over layers; sign by mass); λ·disjoint mean; map back with U | topK ∈ {20, 100} × λ ∈ {1.0, 1.4, 1.8, 2.2} (8) |
| PICO_TA | Pico (Tang & Yang, arXiv 2604.16826, §4): per layer, SVD of [B1, B2] (scaling absorbed); s_j = σ_j²/Σσ²; α_j = 1/(1+(T−1)s_j); B̃_t = (I + U diag(α−1) Uᵀ)B_t; ΔW_calib = ΣB̃_tA_t; γ = mean_t‖ΔW_t‖_F / ‖ΔW_calib‖_F; merged = c·γ·ΔW_calib (c = 1 is the paper's method; λ of the TA base merger cancels) | c ∈ {0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6} (8) |
| GATE | hard θ★ = 30° gate (E1b definition): on layers with θ_min(orth(B1), orth(B2)) < 30°, ΔW2 ← (I − SSᵀ)ΔW2, where S = principal vectors of orth(B1) at angle < 30°; then λ(ΔW1 + ΔW2). **Identical to TA on pairs without a FAIL layer.** | λ = TA grid (8) |
| PICO_TIES (exploratory) | Pico calibration, TIES k = 20 on the calibrated updates, γ-rescale, × c | c ∈ {0.75, 1.0, 1.25, 1.5} (4) |
| SOFTGATE (exploratory) | as GATE, but ΔW2 ← ΔW2 − β SSᵀΔW2 | β ∈ {0.25, 0.5, 0.75} × λ ∈ {0.7, 1.0} (6) |

Fidelity notes and every ambiguity (Pico in particular) are listed in `IMPLEMENTATION_NOTES.md`. They are part of this
registration by reference and were written before the pilot. The code actually run is identified by the sha256 values of
`merges.py` and `e3.py`, which `e3.py` records in `run_config.json`.

**Selection.** For each pair and method, choose the config that maximizes the held-out mean normalized score (merged/single
on the 1,000 held-out examples of each task, averaged over the two tasks). Ties go to the first config in grid order. Then
evaluate that single config once on the eval sets. No eval-set information is used for selection.

## 3. Primary analysis (Q1)
- For each primary method m ∈ {TIES, DARE_TA, DARE_TIES, TSVM, KNOTS, PICO_TA, GATE}, the per-pair gain is
  g_p = score_m(p) − score_TA(p), where both methods use their own tuned configs. The statistic is the mean gain over all pairs.
- **95% CI:** paired task-block bootstrap, 2,000 reps, `numpy.random.default_rng(0)`. Resample the K valid tasks with
  replacement, keep every pair of distinct resampled indices (with multiplicity), take the mean gain, and report the
  percentile interval. Replicates with fewer than 4 distinct pairs are skipped.
- **Test:** pair-level sign-flip permutation test, 10,000 random sign vectors (`default_rng(0)`). One-sided
  p = (1 + #{mean_perm ≥ mean_obs}) / 10,001. **Holm** correction over the 7 methods.
- **Decision rule.** "m beats TA" is **confirmed** iff mean gain ≥ **+0.5 pp** (0.005 in normalized score) **AND**
  Holm-adjusted p < 0.05. Otherwise it is not confirmed. As a descriptive addition, when the CI upper bound is < +0.5 pp we
  report "a ≥ 0.5 pp gain is not supported by the CI".
- Also reported per method: win/tie/loss counts, the median gain, the eval-set mean score, and how often the selected
  config lies on the grid boundary.
- Caveat stated in advance: pairs share tasks, so pair-level sign flips ignore task-level dependence. The task-block CI is
  the dependence-robust quantity and must be read alongside the test. GATE's gain is exactly 0 on every pair without a
  FAIL layer.

## 4. Secondary analyses (Q2), no multiplicity correction, reported descriptively
- **S1. Hard gate vs Pico-soft, head to head:** d_p = score_GATE − score_PICO_TA over all pairs. Report the mean, task-block
  bootstrap CI, two-sided sign-flip p, and win/tie/loss.
- **S2. Hard gate vs TA on pairs with ≥ 1 FAIL layer** (θ_min < 30° by orth(B) principal angles, float64). Report the
  **number of such pairs** and the per-pair differences. If n ≥ 5, also report the mean and the two-sided sign-flip p. If n = 0,
  S2 is "not estimable".

## 5. Exploratory (labelled as such)
- PICO_TIES; SOFTGATE (projection-strength sweep); TA restricted to the E1b 4-λ grid; Pico at c = 1 (paper default, untuned).
- Q3: Spearman correlations between the E1b predictors (O_A, O_B, tv_cosine from E1b `predictors.csv`) and each method's gain,
  and with TA loss (1 − score).
- Compute time per method: merge-construction seconds, held-out evaluation seconds, and eval-set seconds.

## 6. Sanity checks (checked before interpreting; they do not change the rules)
- TA evaluated at λ ∈ {0.3, 0.5, 0.7, 1.0} on the eval sets must reproduce E1b `pair_results.csv` `TA_eval_<task>_lam<λ>`
  exactly (|Δ| ≤ 1e-9). Any mismatch is reported and investigated.
- `test_merges.py` must pass on the GPU machine before launch (`run_e3_on_e1b.sh` runs it).

## 7. Compute and operations
- Launch: `run_e3_on_e1b.sh` on ubuntu-4070, only after E1b is finished (its `VERDICT.md` exists and no E1b process is running).
  Uses `nice -n 10` and a CUDA memory cap (`--mem-frac 0.22`, about 3.5 GB). Resumable per pair (`results.jsonl`).
- Expected cost: about 62 held-out evaluations of 2 × 1,000 examples plus about 12 eval-set evaluations per pair. That is
  roughly 2 minutes per pair on the RTX 4070 Ti SUPER, or about 4 GPU-hours for 120 pairs.
- Outputs: `artifacts/e3_baselines/e1b_run/` (results.jsonl, method_by_pair.csv, analysis.json, E3_E1B_REPORT.md), mirrored to
  box `/workspace/lora-paper/e3/`. E3 never writes to `artifacts/e1b_confirmatory/`.
