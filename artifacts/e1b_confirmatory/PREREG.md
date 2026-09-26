# E1b confirmatory study — PRE-REGISTRATION

Written 2026-09-25 (KST) **before any E1b training, predictor computation, or merge**. Only a dataset-loading probe
(`probe_data.py` → `data_probe.json`: split sizes and label counts, no model) and a synthetic power simulation (`power_sim.py` →
`power_sim.json`) were run beforehand. This file and `prereg.json` (machine-readable, authoritative for budgets and
parameters) are hashed in `prereg.sha256` and are **never edited**. Any later change goes to `deviations.md`, with a timestamp and a reason.

## 0. Question and motivation
Pilot E1 used 7 public prateeky2806 BERT-base GLUE LoRAs (21 pairs). It found ρ(O_A, D) = 0.168, task-block 95% CI [−0.49, 0.94], and permutation p = 0.56.
It was underpowered because only 7 tasks were independent. E1b trains 16 adapters under one controlled recipe (120 pairs) and tests
whether weights-only overlap predictors predict task-arithmetic merge loss.

## 1. Adapters (training recipe)
- Backbone `bert-base-uncased` @ `86b5e0934494bd15c9632b12f734a8a67f723594`. PEFT LoRA r=8, alpha=16 (scaling 2), **dropout 0.1**,
  target_modules = ["query","key","value","dense"]. These are the same as the local hubish adapters (`adapters/mnli_s7_hubish/adapter_config.json`, `scripts/train_lora_glue.py`).
  "dense" matches attention.output.dense, intermediate.dense, output.dense, **and pooler.dense**, which gives 73 LoRA layers.
  modules_to_save = ["classifier"], task_type SEQ_CLS.
- Hubish recipe kept: AdamW lr 1e-3, batch 32, weight decay 0.01, linear schedule, fp16 AMP, max_seq_len 128, dynamic padding.
  Differences from hubish: dropout 0.1 (hubish 0.05), set by the study specification. Seed 0 (hubish 7/42). Warmup = min(500, round(0.1·max_steps)).
  The hubish fixed 500-step warmup exceeds the total steps of small tasks: hubish RTE never reached peak lr.
- **Segment IDs: standard BERT token_type_ids** from the tokenizer (segment A incl. [CLS] and first [SEP] = 0, segment B = 1),
  in both training AND evaluation. Recorded in each adapter's `train_meta.json`.
- Budget rule: hold out 1,000 train examples first (seed 0). Take the remaining pool, then a seeded subsample capped at 60,000. Pool ≤ 10k → 10 epochs,
  otherwise 3 epochs. Step cap = 5,625 (= 60,000/32 × 3, the large-task budget), which never binds for ≤10k pools (max 3,125).
  No evaluation-set scoring happens during training.
- WiC input: segment A = "{word}: {sentence1}", segment B = sentence2. SciTail = `tsv_format` (entails→1, neutral→0).
  SNLI: label −1 removed. TREC: coarse_label (6-way). BoolQ/WiC from `aps/super_glue`.
- TREC: `CogComp/trec` main branch ships a dataset script that datasets 5.0 refuses to run. The Hub-generated parquet branch of the
  **same repo** (`refs/convert/parquet`, commit 65752bf5…) loads with the identical 5,452/500 splits. This is not a substitution.
  All 16 tasks loaded, so **no substitute** (rotten_tomatoes, emotion, paws) is used.

| task | source | n_train | pool (−1000) | used | epochs | max_steps | warmup | eval split | n_eval used | metric |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| cola | glue | 8,551 | 7,551 | 7,551 | 10 | 2,360 | 236 | validation | 1,043 | acc |
| sst2 | glue | 67,349 | 66,349 | 60,000 | 3 | 5,625 | 500 | validation | 872 | acc |
| mrpc | glue | 3,668 | 2,668 | 2,668 | 10 | 840 | 84 | validation | 408 | acc |
| qqp | glue | 363,846 | 362,846 | 60,000 | 3 | 5,625 | 500 | validation | 10,000 of 40,430 | acc |
| stsb | glue | 5,749 | 4,749 | 4,749 | 10 | 1,490 | 149 | validation | 1,500 | Spearman |
| mnli | glue | 392,702 | 391,702 | 60,000 | 3 | 5,625 | 500 | validation_matched | 9,815 | acc |
| qnli | glue | 104,743 | 103,743 | 60,000 | 3 | 5,625 | 500 | validation | 5,463 | acc |
| rte | glue | 2,490 | 1,490 | 1,490 | 10 | 470 | 47 | validation | 277 | acc |
| boolq | super_glue | 9,427 | 8,427 | 8,427 | 10 | 2,640 | 264 | validation | 3,270 | acc |
| wic | super_glue | 5,428 | 4,428 | 4,428 | 10 | 1,390 | 139 | validation | 638 | acc |
| snli | stanfordnlp/snli | 549,367 | 548,367 | 60,000 | 3 | 5,625 | 500 | validation | 9,842 | acc |
| scitail | allenai/scitail tsv | 23,097 | 22,097 | 22,097 | 3 | 2,073 | 207 | validation | 1,304 | acc |
| ag_news | fancyzhx/ag_news | 120,000 | 119,000 | 60,000 | 3 | 5,625 | 500 | test | 7,600 | acc |
| imdb | stanfordnlp/imdb | 25,000 | 24,000 | 24,000 | 3 | 2,250 | 225 | test | 10,000 of 25,000 | acc |
| trec | CogComp/trec (parquet) | 5,452 | 4,452 | 4,452 | 10 | 1,400 | 140 | test | 500 | acc |
| yelp_polarity | fancyzhx/yelp_polarity | 560,000 | 559,000 | 60,000 | 3 | 5,625 | 500 | test | 10,000 of 38,000 | acc |

Total 54,288 optimizer steps. Projected training GPU time is about 1.5 h, well under the 12 h cap.
Splits (`tasks.py:split_indices`): `perm = default_rng(0).permutation(n_train)`, hold = perm[:1000], train = perm[1000:][:60000].
Eval subsets > 10,000 use `default_rng(1).choice(n_eval, 10000, replace=False)`. All index sets are sorted.

## 2. Integrity rule (STEP 2 → stage0.json; applied BEFORE predictors/merges)
A task's single adapter is **valid** if all applicable criteria hold:
- (a) eval metric ≥ majority-class baseline + 10pp. The majority label is the most frequent label in the task's training set.
  The baseline is that label's accuracy on the eval set used; on a tie, take the higher. For STS-B: Spearman ≥ 0.70.
- (b) 8 GLUE tasks only: ours ≥ reference − 5pp. **Reference:** HuggingFace transformers
  `examples/pytorch/text-classification/README.md`, run_glue.py dev results for **bert-base-cased** (FP32 column; seq 128, bs 32,
  lr 2e-5, 3 epochs, MRPC 5): CoLA MCC 56.53, SST-2 acc 92.32, MRPC acc 84.07, STS-B Spearman 88.48, QQP acc 90.71,
  MNLI-m acc 83.91, QNLI acc 90.66, RTE acc 65.70. Two caveats: cased (not uncased) full-fine-tuning numbers, and full training
  data. CoLA uses **MCC** for (b) because no accuracy is published. Criterion (b) is not omitted for any GLUE task.
- (c) manual-merge logits (W0 + ΔW, own head) equal PeftModel logits within 1e-4 (max abs, first 64 eval examples, fp32, TF32 off).
- Invalid tasks are removed together with all their pairs. **If fewer than 12 valid tasks remain: STOP and report.**
- **Risk noted in advance (no rule change):** the 60k cap means MNLI and QQP train on 15–16% of their data. At that size,
  BERT-base typically lands near or below reference − 5pp (MNLI ≥ 78.91, QQP ≥ 85.71), so these two may fail (b).
  BoolQ at max_len 128 (passages truncated) may miss (a) (needs ≥ 72.2%). RTE (needs ≥ 62.7% for (a), 60.7% for (b)) is borderline.
  These risks are written down here only so the outcome can be read in light of them. Thresholds are not changed.

## 3. Predictors (STEP 3, weights only; frozen by sha256 before any merge)
Over all 73 LoRA layers:
- **O_A** = mean over layers of mean_k cos²θ_k, where θ are the principal angles between orth(B1) and orth(B2) (float64 QR/SVD).
- **Task-vector cosine** = cosine of the flattened, concatenated ΔW = 2·B@A.
- Secondary (exploratory):
  - O_B
  - mean θ_min and min θ_min
  - number of layers with θ_min < 30°
  - TIES sign conflict (top-20% |ΔW| per task, global)
  - norm ratio
  - null z-score of O_A vs 1000 random rank-8 subspace pairs (seed 0)

## 4. Merges (STEP 4)
- Task arithmetic: W0 + λ(ΔW1 + ΔW2) on all LoRA layers. Each task uses its own classifier head.
- λ ∈ {0.3, 0.5, 0.7, 1.0} is chosen per pair on the two 1,000-example held-out TRAIN sets (never trained on). The choice maximizes the mean
  normalized score; ties go to the smaller λ.
- Normalized score = merged / single metric on the same set (accuracy; STS-B Spearman).
- **D = 1 − mean of the two normalized eval-set scores at the selected λ.** Eval sets are also scored at every λ (exploratory fixed-λ curves).

## 5. Pre-registered analysis (STEP 5)
- Two primary hypotheses: **H1**: ρ_Spearman(O_A, D) > 0; **H2**: ρ(task-vector cosine, D) > 0. Holm correction is applied over the two.
- Permutation: 10,000 random relabelings of the tasks, applied to the D matrix with the predictor held fixed (`default_rng(0)`).
  One-sided p = (1 + #{ρ_perm ≥ ρ_obs}) / 10,001. Holm adjustment uses these p.
- Task-block bootstrap: 2,000 reps (separate `default_rng(0)`). Resample K tasks with replacement, keep every pair of distinct resampled
  indices, and take the percentile 95% CI.
- Leave-one-task-out (LOTO): ρ recomputed with each valid task removed.
- **PASS** if ρ ≥ 0.4 AND Holm-adjusted one-sided permutation p < 0.05 AND ≥ 75% of LOTO ρ > 0.2.
  **FAIL** (meaningful effect ruled out) if not PASS and the task-block bootstrap 95% upper bound < 0.3. Otherwise **INCONCLUSIVE**.
  Each hypothesis gets its own verdict.
- Exploratory only (not used for verdicts):
  - secondary predictors
  - fixed-λ ρ curves
  - **method comparison**: per pair, TA at the selected λ is compared with (i) TIES-lite and (ii) the θ★ = 30° hard gate, each evaluated at the TA-selected λ and at its own held-out-selected λ.
    - TIES-lite: global top-20% trim per task, sign election by sum, disjoint mean, W0 + λτ.
    - Hard gate: per layer, S = principal vectors of orth(B1) at angle < 30° to orth(B2). On those "FAIL" layers ΔW2 ← (I − SSᵀ)ΔW2, where "second" = t2 in task-list order. If no layer is below 30°, the gate equals TA.
    - Reported: mean normalized score, win/tie/loss vs TA, paired Wilcoxon.

## 6. Power simulation (run before this file was written; `power_sim.py`, `power_sim.json`)
Synthetic pair predictor X and loss D for K = 16 tasks (120 pairs), built with crossed task random effects:
X_ij = √w (u_i + u_j)/√2 + √(1−w) e_ij, and D the same with (v, n). corr(u,v) = corr(e,n) = 2 sin(πρ/6), so the population Spearman is ρ.
The task-block variance share is w ∈ {0.25, 0.5, 0.75}. The decision rule and resampling are exactly as above (2,000 bootstrap, 10,000 permutations,
LOTO), with 1,000 simulated studies per cell. The verdict is evaluated at one-sided α = 0.025 (Holm, worst case) and 0.05 (Holm, best case). **The two gave
identical results in every cell**, because the ρ ≥ 0.4 and LOTO requirements bind before the p-value does.

| true ρ | w | mean ρ̂ (sd) | mean boot upper | P(PASS) | P(FAIL) | P(INCONCLUSIVE) |
|---|---|---|---:|---:|---:|---:|
| 0.0 | 0.25 | −0.006 (0.107) | 0.303 | 0.000 | 0.486 | 0.514 |
| 0.0 | 0.50 | −0.003 (0.142) | 0.321 | 0.001 | 0.417 | 0.582 |
| 0.0 | 0.75 | −0.004 (0.189) | 0.359 | 0.017 | 0.346 | 0.637 |
| 0.3 | 0.25 | 0.294 (0.100) | 0.554 | 0.150 | 0.003 | 0.847 |
| 0.3 | 0.50 | 0.302 (0.124) | 0.576 | 0.224 | 0.007 | 0.769 |
| 0.3 | 0.75 | 0.291 (0.170) | 0.598 | 0.285 | 0.019 | 0.696 |
| 0.5 | 0.25 | 0.494 (0.083) | 0.702 | 0.863 | 0.000 | 0.137 |
| 0.5 | 0.50 | 0.499 (0.107) | 0.716 | 0.833 | 0.000 | 0.167 |
| 0.5 | 0.75 | 0.497 (0.147) | 0.736 | 0.746 | 0.000 | 0.254 |

What the simulation shows:
- The permutation test is calibrated at ρ = 0 (P(p < 0.05) = 0.047–0.051).
- **FAIL is reachable under ρ = 0 with 16 tasks, but it is not the typical outcome:** P(FAIL | ρ = 0) is 0.35–0.49. That means that
  even when no effect exists, INCONCLUSIVE is slightly more likely than FAIL. The mean bootstrap CI width is about 0.6–0.7.
- True ρ = 0.3 almost never gives FAIL (≤ 2%). It passes in 15–29% of cases, because the ρ ≥ 0.4 point threshold is above 0.3.
- True ρ = 0.5 passes in 75–86% of cases.
- If K drops to 12 valid tasks, power will be lower and CIs wider.
Thresholds are **not** changed in response to the simulation. It is reported only so the verdicts can be read correctly.

## 7. Resumability, logging, and file layout (artifacts/e1b_confirmatory/)
- `tasks.py`: shared specs and splits.
- `train_e1b.py`, `run_train.sh`: STEP 1. Resumable through `adapters/<task>/DONE`. Log: `train.log`, `logs/train_<task>.log`.
- Later steps: `stage0.json`; `predictors.csv` + `predictors.sha256`; `pair_results.csv`; `analysis.json`; `VERDICT.md`;
  `fig_f1_scatter.png`; `fig_f2_heatmap.png`; method comparison table.
- Other repo files are never modified, and no git operations are run.
- Final outputs are mirrored to the box at `/workspace/lora-paper/e1b/`.
