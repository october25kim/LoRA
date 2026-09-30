# E4a — RoBERTa-base second-backbone replication of E1b/E1c — PRE-REGISTRATION

Written 2026-09-26 (KST) **before any E4a training (the pilot included), integrity check, predictor computation or merge**. No RoBERTa
adapter exists when this file is written; only the roberta-base checkpoint was downloaded. This file and `prereg_e4a.json`
(machine-readable; authoritative for parameters) are hashed in `prereg_e4a.sha256` and are **never edited**. Later changes go to
`DEVIATIONS_E4A.md` (append-only, timestamped KST, with a reason). Directory: `artifacts/e4a_roberta/` on ubuntu-4070; mirror: box `/workspace/lora-paper/e4a/`.

## 0. Disclosure and purpose
- Known when this was written: E1b (BERT-base, all adapters seed 0): H1 (O_A) **FAIL** ρ = −0.194, CI [−0.521, 0.203]; H2 (tv-cosine)
  **INCONCLUSIVE** ρ = −0.044. E1c (seed-diverse): P (mixed seed) H1c ρ = −0.025 CI [−0.51, 0.43], H2c ρ = 0.127, both **INCONCLUSIVE**; S1 both
  INCONCLUSIVE; min θ_min 39.8° (same-task), the 30° gate fired in 0/196 pairs; D reliability across seed configurations 0.67–0.81. E3: no method beat TA.
- Purpose: test whether the weights-only overlap predictors fail to predict task-arithmetic merge loss on a **second encoder backbone**
  (RoBERTa-base), under the E1b protocol, with two training seeds so that both the shared-seed (E1b-like) and mixed-seed (E1c-like) designs are available.
- Everything is copied from E1b/E1c except the backbone-specific items in §1–§2 and the choices in §3–§6, all fixed here.

## 1. Adapters
- Backbone `FacebookAI/roberta-base` @ `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. PEFT LoRA r = 8, α = 16 (scaling 2), dropout 0.1.
- **Target modules** `["query","key","value","output.dense","intermediate.dense"]`. PEFT suffix matching makes `output.dense` hit both
  `attention.output.dense` and `layer.N.output.dense`, so every layer gets q, k, v, attn-out, intermediate, output = **72 LoRA layers** (asserted at training time).
  This is E1b's set minus the pooler: `RobertaForSequenceClassification` has no pooler. Its classification head (`classifier.dense` 768×768 → tanh →
  `classifier.out_proj`) is fully trained per task (`modules_to_save=["classifier"]`), saved with the adapter, and is not LoRA-adapted
  (E1b's generic `"dense"` pattern would also hit `classifier.dense`, so it is not reused).
- Recipe = E1b §1: AdamW lr **1e-3** (replaced only by the pilot rule in §1a), batch 32, weight decay 0.01, linear schedule,
  warmup = min(500, round(0.1·max_steps)), fp16 AMP, max_len 128, dynamic padding.
- **Segment IDs: none.** RoBERTa has `type_vocab_size = 1` and its tokenizer returns no `token_type_ids`. E4a never passes `token_type_ids` to the model:
  not in training, not in integrity check (c), not in any evaluation. This is equivalent to all zeros. Pair inputs use the tokenizer default `<s> A </s></s> B </s>`
  with longest-first truncation at 128. Padding uses `<pad>` (id 1) with an attention mask. WiC/SciTail/SNLI/TREC input construction as in E1b (`tasks.py`, imported unmodified).
- **Seeds 0 and 1**: `set_seed(seed)` before model construction and `TrainingArguments(seed, data_seed)`. This changes LoRA-A init, head init, dropout and data order.
  Splits do not depend on the seed: the held-out 1,000 TRAIN examples, the training subset and the eval subsets come from E1b `tasks.split_indices` (fixed seed 0),
  and their sha256 must equal the E1b adapter's `train_meta.json` (asserted).
- **Tasks**: the 14 valid E1b tasks cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity (qqp, boolq not trained).
  **Budgets exactly as E1b** (table in `prereg_e4a.json`), MNLI under E1b deviation D1 (full pool 391,702, 3 epochs, 36,723 steps, warmup 500). 77,121 steps per seed.
- Collapse monitoring: every adapter logs its loss every 50 steps (`train_meta.json: loss_history`) with the entropy of the training-label prior. A collapsed adapter
  is **not retrained**; it is handled by the integrity rule (§2). Loss curves are reported.

### 1a. Pre-registered pilot (sst2, rte; the only allowed recipe change)
- Train sst2 (5,625 steps) and rte (470 steps), seed 0, with the main recipe at lr 1e-3, into `pilot/lr0.001/`. Evaluate **only on the 1,000 held-out TRAIN examples**
  (manual merge, fp32). No eval-set scoring.
- A pilot task has **diverged** if any of these holds:
  (i) a logged loss is NaN/inf;
  (ii) the mean logged loss over the last 20% of logged steps is ≥ 0.9 × the entropy of the training-label prior (collapse to the prior);
  (iii) held-out accuracy is below the held-out majority baseline + 10 pp.
- lr ladder 1e-3 → 5e-4 → 2e-4. The next rung is piloted only if a task diverged at the previous one. `lr_main` = the first rung where neither task diverged.
  If all three rungs diverge, use the rung with the best mean held-out accuracy, document it, and proceed; the integrity rule decides.
  The decision is written to `pilot/pilot_decision.json` before main training, which reads it. The lr applies to all tasks and both seeds.
  Pilot adapters are not used in the study.

## 2. Integrity (both seeds, BEFORE predictors and merges)
- (a) E1b verbatim: metric ≥ majority baseline + 10 pp; STS-B Spearman ≥ 0.70.
- (b) **RoBERTa-specific citable reference**: fairseq `examples/roberta/README.md` @ commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`, table
  "GLUE (dev set, single model, single-task finetuning)", row `roberta.base`: MNLI 87.6, QNLI 92.8, RTE 78.7, SST-2 94.8, MRPC 90.2, CoLA 63.6, STS-B 91.2.
  Rule as E1b: ours ≥ reference − 5 pp. CoLA uses MCC (63.6 is MCC by GLUE convention). MRPC, RTE, SST-2, QNLI and MNLI-m use accuracy. STS-B uses our Spearman (the README does not name the metric).
  Thresholds: CoLA MCC 0.586, SST-2 0.898, MRPC 0.852, STS-B 0.862, MNLI 0.826, QNLI 0.878, RTE 0.737.
  Caveats: the references are full fine-tuning on full data with best-checkpoint-on-dev over a hyperparameter search, which is stricter than E1b's BERT reference.
  **Risk noted in advance (no rule change):** RTE (1,490 training examples, needs ≥ 73.7%), CoLA (MCC ≥ 0.586) and MRPC (2,668 examples, ≥ 85.2%) may fail (b), and
  MNLI (36.7k steps at lr 1e-3) may collapse like E1b's QQP. The rule was chosen over the alternatives (BERT references − margin, or dropping (b)) because the task
  specification requires a citable RoBERTa reference when one exists.
- (c) E1b verbatim, RoBERTa model: manual merge (W0 + ΔW + own head) vs PeftModel logits within 1e-4, first 64 eval examples, fp32, TF32 off, no token_type_ids.
- Structural: 72 LoRA layers; head shapes; r/α; train/held-out index sha256 = train_meta; main-run adapter of that seed.
- **Exclusion:** a task that fails in **either** seed is removed, with all its pairs, from **every** population. **STOP** if fewer than **10** valid tasks remain.
- Singles (eval and held-out) are those computed in this stage0, per seed.

## 3. Populations (V = valid tasks, K = |V|)
- **P — PRIMARY — mixed-seed cross-task pairs** (E1c rule): for every unordered {i, j} ⊂ V, the alphabetically first task uses its seed-0 adapter (t1) and the
  other its seed-1 adapter (t2). K(K−1)/2 pairs. No pair shares an initialization. The gate projects t2.
- **R0 — secondary — seed-0 cross-task pairs** in E1b task-list order: the E1b confirmatory design on the new backbone.
- **S1 — secondary — seed-1 cross-task pairs** (E1b order).
- **S2 — secondary — same-task seed pairs** t@s0 × t@s1 (K pairs).
- Why P is primary: E1c showed that the shared seed inflates O_A and cosine uniformly. Mixed-seed pairs remove that confound.
  R0 is the like-for-like E1b design and is reported with the same rule as a secondary "rule outcome".

## 4. Predictors (weights only; frozen by sha256 before any merge)
E1b §3 / E1c `compute_predictors` arithmetic (float64) over the 72 layers, for all pairs of P, R0, S1, S2: **O_A** (mean over layers of mean_k cos²θ_k between
orth(B1) and orth(B2)), **tv_cosine** (flattened concatenated ΔW = 2BA), mean/min θ_min, number of layers with θ_min < 30°, TIES sign conflict (top 20%, and all), norm ratio,
null z of O_A (1,000 random rank-8 pairs per layer, seed 0), per-layer table. **O_B and mean_theta_min_B are dropped**: they are identical to the A-side quantities (E1c N3.1).
Frozen files → `predictors_e4a.sha256`, written before any merge. stage2 refuses to run on a hash mismatch.

## 5. Merges
E1b §4 verbatim for every pair of every population: TA W0 + λ(ΔW1 + ΔW2) on the 72 layers, each side with its own head, λ ∈ {0.3, 0.5, 0.7, 1.0}, chosen per pair on the two 1,000-example
held-out TRAIN sets (mean normalized score; ties → smaller λ). **D = 1 − mean of the two normalized eval-set scores at λ\***. All λ are also scored on eval.
Exploratory TIES-lite and the θ★ = 30° hard gate are exactly as in E1b. Precision fp32, TF32 off, batch 256. Singles come from each seed's stage0.

## 6. Analysis
- **Primary (P only): H1** ρ_Spearman(O_A, D) > 0; **H2** ρ(tv_cosine, D) > 0; **Holm over {H1, H2}** on the one-sided permutation p.
- Statistics verbatim from E1c `pop_stats` (= E1b stage3): 10,000 task relabelings (`default_rng(0)`), p = (1 + #{ρ_perm ≥ ρ_obs})/10,001; task-block bootstrap
  2,000 (`default_rng(0)`, skip resamples with < 4 distinct pairs, percentile 95% CI); LOTO.
- **PASS** iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% of LOTO ρ > 0.2. **FAIL** iff not PASS ∧ bootstrap upper < 0.3. Otherwise **INCONCLUSIVE**. One verdict per hypothesis.
- **Secondary** (exploratory, uncorrected):
  1. R0 and S1: the same statistics and rule (Holm within each population), reported as "rule outcome".
  2. **Pooled 3-population summary**: per unordered task pair, the mean of D and of each predictor over R0, S1 and P; the same statistics, descriptive only.
  3. θ_min distribution per population (min, quantiles, layers < 30° / < 45°), gate firings, O_A vs the random null (z range, fraction above the null p95),
     shared-vs-mixed O_A (paired P − R0, P − S1, Wilcoxon).
  4. Reliability of D, O_A, tv_cosine and λ\* across seed configurations: Spearman over the same unordered task pairs, R0~S1, R0~P, S1~P.
  5. S2 same-task pairs: θ_min, gate firing, D, and E1b-stage2 gate vs TA. **E3 `e3.py run()` and `merges.py` imported unmodified** (the encoder class is swapped for the
     RoBERTa one; same interface), amendment-A2 grids: TA, PICO_TA, GATE (θ★ 30°), FORCEGATE (k = 1). Contrasts GATE−TA, FORCEGATE−TA, PICO_TA−TA, GATE−PICO_TA,
     FORCEGATE−PICO_TA: mean, 2,000-rep bootstrap CI over tasks, two-sided sign-flip p (10,000), W/T/L (E1c §7.3 code).
  6. Fixed-λ ρ curves, secondary predictors, TIES-lite and gate vs TA for P, R0 and S1.
  7. Cross-backbone: Spearman of per-task-pair D and O_A, RoBERTa vs BERT (E4a R0 vs E1b R0; E4a P vs E1c P), on common valid tasks.
- **Reading fixed in advance.** H1 FAIL (or INCONCLUSIVE with ρ ≤ 0) on P → the BERT null generalizes to RoBERTa. H1 PASS → the overlap predictor works on RoBERTa;
  that conflicts with BERT and both are reported. INCONCLUSIVE → E4a cannot decide. The E1b power simulation (16 tasks) gives P(FAIL | ρ = 0) of only 0.35–0.49.

## 7. Compute budget, priorities, code
- **Cap: 10 GPU-hours**, measured as wall-clock from the first E4a GPU job (pilot) to the end of the last one. Summed process-hours are also reported. At most 3 concurrent GPU processes.
  Projection: pilot 0.3 h, training 2 seeds ≈ 1.8 h (3 processes), stage0 0.2 h, merges 287 pairs ≈ 3.8 h (2 processes), E3 subset 0.4 h; total ≈ 6.5 h.
- If the cap is at risk, priority is: training → integrity → predictors → **P** → R0 → S2 + E3 → S1. Unfinished work is reported as not run, never as partial evidence.
- A pipeline smoke test with throwaway 20-step adapters runs in a scratch copy outside `artifacts/`. It produces no eval numbers for real adapters.
- Code in `artifacts/e4a_roberta/`: `tasks_e4.py`, `train_e4a.py`, `run_train_e4a.sh`, `e4a.py`, `run_pipeline_e4a.sh`, `e4a_analysis.py`, `make_verdict_e4a.py`.
  E1b `e1b.py`/`tasks.py` and E3 `e3.py`/`merges.py` are imported read-only (`PYTHONDONTWRITEBYTECODE=1`) and their sha256 values are checked against the values in `prereg_e4a.json`.
  Code hashes are recorded at launch (`code_e4a.sha256`). Analysis/report code may be written during training, before any E4a eval-set number exists, and is hashed before stage0.
- Nothing is written to other `artifacts/` folders. No git operations.
