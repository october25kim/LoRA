# E1c — seed-diverse replication of the E1b confirmatory study — PRE-REGISTRATION

Written 2026-09-26 (KST), **before any E1c training, integrity check, predictor computation or merge**. No seed-1 adapter exists
when this file is written. This file and `prereg_e1c.json` (machine-readable; authoritative for parameters) are hashed in
`prereg_e1c.sha256` and are **never edited**. Any later change goes to `DEVIATIONS_E1C.md` (append-only, timestamped KST, with a reason).
Directory: `~/Desktop/Workspace/LoRA/artifacts/e1c_seed_diverse/` on ubuntu-4070; mirror: box `/workspace/lora-paper/e1c/`.

## 0. Disclosure: written after E1b and E3 results were known, and why
- **Known when this was written.** E1b (`artifacts/e1b_confirmatory/VERDICT.md`, 2026-09-26 00:22 KST): 14 valid tasks, 91 pairs.
  H1 (O_A) **FAIL**, ρ = −0.194, task-block 95% CI [−0.521, 0.203]. H2 (task-vector cosine) **INCONCLUSIVE**, ρ = −0.044, CI [−0.425, 0.316].
  D mean 0.048 (range −0.042 to 0.189). TA λ* counts {0.5: 22, 0.7: 60, 1.0: 9}. O_A null z = 16–95. Minimum θ_min over all pairs and layers = 45.1°,
  so the 30° gate never fired. qqp (full-data D1 adapter collapsed) and boolq (never learned) were excluded.
  E3 (`artifacts/e3_baselines/e1b_run/E3_E1B_REPORT.md`, 05:20 KST): no method met the "beats TA" rule. PICO_TA +0.22 pp
  (CI [−0.04, +0.52], Holm p 0.033, below the +0.5 pp threshold). FORCEGATE −0.03 pp. GATE ≡ TA.
  The earlier natural hubish pair (MNLI seed 7 × MNLI seed 42, different recipe) had 2 layers below 30° (manuscript Table 9).
- **Why E1c comes after the fact.** Every E1b adapter used seed 0, so all adapters shared the same LoRA-A initialization and the
  same data-order seed. E1b's post-run note 7 and the null-z values point to this as a confound: shared init probably inflates
  O_A uniformly and may restrict its range, and reviewers will raise it. E1c is a targeted robustness replication that removes
  shared initialization from every analysed pair. It could not have been planned before E1b showed the inflated overlaps.
- **What this means for reading E1c.** The recipe, integrity rules, predictors, λ grid, selection rule, D, statistics, and
  PASS/FAIL/INCONCLUSIVE thresholds are copied **verbatim** from E1b (`PREREG.md` sha256 a04f1485…, `prereg.json` f9c43e51…).
  The only new choices are listed in §1–§3: seed value 1 (the next integer, not selected), the alphabetical side rule, the
  populations, the integrity STOP threshold (10, set by the study specification), and the secondary analyses. All are fixed here,
  before any seed-1 result exists. E1c is a confirmatory test of H1c/H2c *conditional on these disclosed facts*. It is not a blind
  replication by an independent team.

## 1. Adapters: the 14 valid E1b tasks again, with seed 1
- Tasks (E1b `stage0.json: valid_tasks`): cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity.
  qqp and boolq are not trained (they were invalid in E1b).
- Recipe = E1b §1, unchanged: bert-base-uncased @ 86b5e0934494bd15c9632b12f734a8a67f723594; PEFT LoRA r = 8, α = 16, dropout 0.1,
  target ["query","key","value","dense"] (73 layers incl. pooler), modules_to_save ["classifier"]; AdamW lr 1e-3, batch 32,
  wd 0.01, linear schedule, warmup = min(500, round(0.1·max_steps)), fp16 AMP, max_len 128, dynamic padding; standard BERT
  token_type_ids in training and evaluation.
- **Only change: seed 1** instead of 0: `set_seed(1)` before model construction, `TrainingArguments(seed=1, data_seed=1)`. This changes
  the LoRA-A initialization (kaiming-uniform), the classifier-head initialization, the dropout masks, and the data order.
  LoRA-B is zero-initialized under both seeds.
- **Unchanged (fixed seed 0 inside `tasks.split_indices`, not the training seed):** the 1,000 held-out TRAIN examples used for λ
  selection (identical to E1b), the training subset (the same seeded 60k subsample for capped tasks, or the whole pool), and the eval subsets.
  The held-out and training index sha256 values must equal those in the corresponding E1b adapter's `train_meta.json` (asserted).
- **Budgets exactly as E1b used them for each task**, including deviation D1 for MNLI (full pool, 391,702 examples, 3 epochs,
  36,723 steps, warmup 500). All other tasks use the pre-registered E1b budgets (table). Total 77,121 optimizer steps.

| task | used | epochs | max_steps | warmup | eval split | n_eval | metric | E1b budget source |
|---|---:|---:|---:|---:|---|---:|---|---|
| cola | 7,551 | 10 | 2,360 | 236 | validation | 1,043 | acc | prereg |
| sst2 | 60,000 | 3 | 5,625 | 500 | validation | 872 | acc | prereg |
| mrpc | 2,668 | 10 | 840 | 84 | validation | 408 | acc | prereg |
| stsb | 4,749 | 10 | 1,490 | 149 | validation | 1,500 | Spearman | prereg |
| mnli | 391,702 | 3 | 36,723 | 500 | validation_matched | 9,815 | acc | **D1 (full data)** |
| qnli | 60,000 | 3 | 5,625 | 500 | validation | 5,463 | acc | prereg |
| rte | 1,490 | 10 | 470 | 47 | validation | 277 | acc | prereg |
| wic | 4,428 | 10 | 1,390 | 139 | validation | 638 | acc | prereg |
| snli | 60,000 | 3 | 5,625 | 500 | validation | 9,842 | acc | prereg |
| scitail | 22,097 | 3 | 2,073 | 207 | validation | 1,304 | acc | prereg |
| ag_news | 60,000 | 3 | 5,625 | 500 | test | 7,600 | acc | prereg |
| imdb | 24,000 | 3 | 2,250 | 225 | test | 10,000 | acc | prereg |
| trec | 4,452 | 10 | 1,400 | 140 | test | 500 | acc | prereg |
| yelp_polarity | 60,000 | 3 | 5,625 | 500 | test | 10,000 | acc | prereg |

- Code: `train_e1c.py`, a copy of E1b `train_e1b.py`/`train_e1b_d1.py`. It imports E1b `tasks.py` unmodified, reads budgets and recipe from E1b
  `prereg.json`/`deviations_d1.json`, and overrides only `seed` → 1 and the output directory (`adapters_s1/<task>`).
- Scheduling (not part of the recipe): MNLI trains in one process while the other 13 tasks train sequentially in a second process on the
  same GPU. No eval-set or held-out scoring happens during training.
- Seed-0 adapters: the E1b adapters (`artifacts/e1b_confirmatory/adapters/<task>`), read-only, sha256 checked against E1b `stage0.json`.

## 2. Integrity (applied to seed-1 adapters BEFORE predictors and merges)
- E1b §2 verbatim: (a) metric ≥ majority baseline + 10 pp (STS-B Spearman ≥ 0.70); (b) the 7 GLUE tasks here (cola MCC, sst2, mrpc, stsb, mnli, qnli, rte) ≥
  HF bert-base-cased reference − 5 pp; (c) manual-merge logits = PeftModel logits within 1e-4; structural checks (73 layers, head shape, r/α,
  train/held-out index sha256 = train_meta).
  Run by E1b `e1b.py` `stage0` imported **unmodified** (outputs redirected to `e1c/s1/`).
- **Exclusion rule:** if a task's seed-1 adapter fails, that task is removed, with all its pairs, from **every** E1c population (P, S1, S2).
  This keeps a complete K-task design, which the task-block bootstrap, the permutation, and LOTO require.
  Seed-0 validity is taken from E1b `stage0.json` (all 14 valid).
- **STOP** if fewer than **10** valid tasks remain (E1b used 12; 10 is set by the E1c specification). Report and do not merge.
- Reproducibility check (does not change validity): E1b `stage0` is also re-run on the 14 seed-0 adapters (outputs in `e1c/s0_recheck/`).
  Single-adapter eval and held-out scores must match E1b `stage0.json` within 1e-6. **The E1b `stage0.json` values are the ones used** as seed-0
  single references in D. A mismatch is reported.

## 3. Populations
Let V be the valid task set (K = |V|; 14 if nothing is excluded).
- **P (PRIMARY) — mixed-seed cross-task pairs.** One pair for each unordered task pair {i, j} ⊂ V, i ≠ j. **Side rule:** the task that is first in
  Python string sort order (alphabetical) uses its **seed-0** adapter and is t1; the other task uses its **seed-1** adapter and is t2.
  The gate projects ΔW of t2. K(K−1)/2 pairs (91). No pair shares an initialization.
  Alphabetical order: ag_news, cola, imdb, mnli, mrpc, qnli, rte, scitail, snli, sst2, stsb, trec, wic, yelp_polarity. So ag_news is always seed 0
  and yelp_polarity always seed 1. Task and side are therefore confounded by construction. The mirror assignment is **not** run (both orders are not needed).
- **S1 (secondary) — seed-1-only cross-task pairs**: i@s1 × j@s1 in E1b task-list order (t1 precedes t2 in the E1b list), 91 pairs. These pairs share seed 1
  (shared init, like E1b).
- **S2 (secondary) — same-task seed pairs**: t@s0 (t1) × t@s1 (t2) for t ∈ V, 14 pairs. Each side is scored with its own head and normalized by its own single score.
- **R0 (reference, not recomputed)**: the 91 E1b seed-0 cross-task pairs (`e1b_confirmatory/predictors.csv`, `pair_results.csv`).

## 4. Predictors (frozen by sha256 before any merge)
Same definitions and numerics as E1b §3 / `e1b.stage1` (float64): O_A (mean over 73 layers of mean_k cos²θ_k between orth(B1) and orth(B2)),
tv_cosine (cosine of the flattened concatenated ΔW = 2BA), O_B, mean/min θ_min, number of layers with θ_min < 30°, TIES sign conflict (top-20%), norm ratio,
null z of O_A (1000 random rank-8 pairs, seed 0), per-layer table. Computed for all pairs in P, S1, S2 by `e1c.py stage1`. This is a generalization of
`e1b.stage1` to explicit (adapter, adapter) pair lists; the arithmetic is unchanged.
**Equivalence gate (run before training, on E1b adapters only):** `e1c.py` stage1 applied to the 91 R0 pairs must reproduce E1b `predictors.csv`
(max relative difference ≤ 1e-9 on every numeric column). If it does not, the code is fixed before training and the fix is logged in `DEVIATIONS_E1C.md`.
Frozen files: `predictors_e1c.csv`, `predictors_layers_e1c.csv`, `predictors_null_e1c.json`, `predictors_aux_e1c.json` → `predictors_e1c.sha256`.

## 5. Merges
E1b §4 verbatim for every pair in P, S1, S2: TA W0 + λ(ΔW1 + ΔW2), each side with its own head; λ ∈ {0.3, 0.5, 0.7, 1.0} selected per pair on the two
1,000-example held-out TRAIN sets (mean normalized score; ties → smaller λ); **D = 1 − mean of the two normalized eval-set scores at λ\***;
eval sets are also scored at every λ. Exploratory TIES-lite and θ★ = 30° hard gate as in E1b. Single references: seed 0 from E1b `stage0.json`,
seed 1 from E1c `s1/stage0.json`. Implementation: `e1c.py stage2`, generalizing `e1b.stage2` with the same arithmetic. It imports `e1b.Enc`, `score`,
`compact`, `delta`, `load_adapter`, `get_data` unmodified.
**Equivalence gate (before training):** the generalized stage2 on 2 R0 pairs (cola-sst2, rte-wic) must reproduce E1b `pair_results.csv` (D, λ\*,
all TA_eval/TA_hold values) within 1e-9.

## 6. Primary analysis (population P only)
- **H1c:** Spearman ρ(O_A, D) > 0. **H2c:** ρ(tv_cosine, D) > 0. **Holm** over {H1c, H2c} on the one-sided permutation p.
- E1b §5 verbatim: permutation = 10,000 random relabelings of the K tasks applied to the D matrix with the predictor fixed (`default_rng(0)`),
  one-sided p = (1 + #{ρ_perm ≥ ρ_obs}) / 10,001. Task-block bootstrap = 2,000 reps (separate `default_rng(0)`): resample K tasks, keep all
  pairs of distinct resampled indices, skip replicates with < 4 distinct pairs, percentile 95% CI. LOTO: ρ with each task removed.
  The "task" of a P pair side is its task name (seed does not matter). The task index order is the E1b task-list order restricted to V (as in E1b).
- **PASS** iff ρ ≥ 0.4 AND Holm p < 0.05 AND ≥ 75% of LOTO ρ > 0.2. **FAIL** iff not PASS and bootstrap upper bound < 0.3. Otherwise **INCONCLUSIVE**.
  Each of H1c and H2c gets its own verdict.
- Reading fixed in advance: H1c FAIL (or H2c FAIL/INCONCLUSIVE with |ρ| small) → the E1b null survives removal of shared LoRA-A initialization.
  H1c PASS → the E1b null may have been produced by shared initialization; E1b and E1c would then conflict, and both are reported.
  INCONCLUSIVE → E1c cannot decide.
- Statistical note (from the E1b power simulation, 16 tasks): even under ρ = 0, FAIL is reached in only 35–49% of studies. With 14 tasks,
  INCONCLUSIVE under a true null is at least as likely.

## 7. Secondary analyses (exploratory; no multiplicity correction; no verdict-level claims)
1. **S1 seed-1-only:** same statistics as §6 (ρ, CI, perm p, Holm over the two, LOTO). The E1b rule is applied descriptively ("rule outcome"),
   as a same-recipe replication of E1b with another seed.
2. **Reliability of D across seed configurations:** Spearman of per-task-pair D between R0, S1, and P (same unordered task pair), and the same for O_A and tv_cosine.
3. **S2 same-task seed pairs (14):** per pair, θ_min per layer (orth(B), float64), min over layers, number of layers with θ_min < 30°, whether the
   θ★ = 30° gate fires (≥ 1 layer), D (TA, E1b grid), and E1b-stage2 gate@λ\*TA and gate@own-λ vs TA.
   **Gate / TA / Pico comparison with E3 code:** E3 `e3.py run()` and `merges.py` imported **unmodified**, with the E3 amendment-A2 grids and selection rule, for methods
   TA (λ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3}), PICO_TA (c ∈ {0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5}), GATE (θ★ = 30°, λ = TA grid),
   and FORCEGATE (k = 1, λ = TA grid). E3 also produces Pico c = 1 and TA on the 4-λ grid as extras. Tuning uses the 1,000 held-out examples of the task, per side; the final score uses the eval set.
   Report per pair and as means: GATE − TA, FORCEGATE − TA, PICO_TA − TA, GATE − PICO_TA, FORCEGATE − PICO_TA. For each, report the mean, a 95% bootstrap CI over the 14 tasks (2,000 reps, `default_rng(0)`;
   here task-block = pair resampling, because each S2 pair is one task), the two-sided sign-flip p (10,000, `default_rng(0)`), and W/T/L.
   Also the E3-S2 analogue: gate vs TA restricted to S2 pairs with ≥ 1 FAIL layer (n; per-pair differences; mean and sign-flip p if n ≥ 5).
4. **O_A shared-seed vs mixed-seed:** distributions of O_A (also null z, tv_cosine, min θ_min) in R0 (shared seed 0), S1 (shared seed 1) and P (mixed).
   Report mean, sd, median, IQR, min, and max. Report paired differences P − R0 and P − S1 over the same unordered task pairs, with a Wilcoxon signed-rank test.
   Report the fraction of P pairs whose O_A exceeds the random-subspace null 95th percentile. Also report S2 O_A (same task, mixed seed).
5. Fixed-λ ρ curves, the other secondary predictors, and the TIES-lite / gate method comparison for P and S1, as in E1b (exploratory).

## 8. Compute budget and priorities
- **GPU budget cap: ≤ 8 h wall-clock** from the first E1c training launch to the end of the last E1c GPU job. At most 2 concurrent GPU processes.
  Projection: training ≈ 1 h (E1b took 0.97 h sequentially for these budgets), stage0 ×2 ≈ 5 min, predictors ≈ 2 min,
  merges 196 pairs × ≈ 70 s ≈ 3.8 GPU-h (partly in 2 concurrent processes), E3 subset 14 pairs ≈ 15 min. Total ≈ 5 h.
- Priority if the cap is at risk: training → integrity → predictors (frozen) → **P merges** → S2 merges + E3 subset → S1 merges.
  Work not finished within the cap is reported as not run. It is never reported as partial evidence for a verdict.

## 9. Code, provenance, files
- New code, all in `e1c_seed_diverse/`: `train_e1c.py`, `run_train_e1c.sh`, `e1c.py` (stages: equiv, stage0, stage1, stage2, e3same, stage3), `run_pipeline_e1c.sh`.
  E1b `e1b.py`/`tasks.py` and E3 `e3.py`/`merges.py` are imported read-only (PYTHONDONTWRITEBYTECODE=1). Their sha256 values are recorded
  (`code_e1c.sha256` at launch) and must equal the final E1b code (`code_stage3_rerun.sha256`: e1b.py 4610d045…; `code_step1.sha256`: tasks.py 34dd6bd2…)
  and the E3 launch record (`e1b_run/code_at_launch.sha256`: e3.py a25e393e…, merges.py 7ec5774b…).
- Nothing is written to `artifacts/e1b_confirmatory/` or `artifacts/e3_baselines/`. The data cache is copied from E1b `cache/` into `e1c_seed_diverse/cache/`, and stage0 verifies it
  against each adapter's `train_meta.json` index hashes.
- Outputs: `adapters_s1/`, `s1/stage0.json`, `s0_recheck/stage0.json`, `predictors_e1c*.{csv,json}` + `predictors_e1c.sha256`, `pair_results_e1c.{jsonl,csv}`,
  `e3same/` (E3 subset), `analysis_e1c.json`, `VERDICT_E1C.md`, figures `fig_e1c_*.png`, `timing_e1c.json`, `run_e1c.log`.
