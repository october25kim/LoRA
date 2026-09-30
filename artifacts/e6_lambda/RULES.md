# E6: pre-merge rules for choosing λ and deciding whether to merge (POST HOC / EXPLORATORY)

**Status.** Post hoc and exploratory. E6 was **not** part of any pre-registration (E1b/E1c/E4a/E4b/E5). All data were collected under earlier protocols. The author of these rules had already seen summary facts about the data before writing them: the manuscript reports that held-out λ* is often 0.7 on BERT, that λ* = 1.0 in 48/91 Qwen-P pairs, and that λ = 1 adds 1.0–5.7 pp of loss over the oracle. The column headers of the per-pair files were inspected, and weight features were checked against the frozen predictor tables (they match to about 1e-10). No rule below had been scored on any pair before this file was hashed. The list is fixed here; anything added later will be labelled "added after RULES.md".

## Data (780 cross-task pairs, 9 populations)
BERT-base R0/S1/P (91 each; E1b + E1c), RoBERTa-base R0/S1/P (78 each; E4a), Qwen2.5-0.5B R0/S1/P (91 each; E4b; λ grid extended by E5b). Same-task S2 pairs and the 21 E1 Hub pairs are excluded. Task identity = task name regardless of seed.

## Quantities
- Primary grid G4 = {0.3, 0.5, 0.7, 1.0} (common to all backbones). Secondary grid for Qwen only: G7 = G4 ∪ {1.3, 1.5, 2.0}.
- D(λ) = 1 − ½(acc_m,t1/acc_s,t1 + acc_m,t2/acc_s,t2) on the evaluation sets (the paper's pair loss).
- **Reference ("oracle") λ_sel** = held-out-selected λ (1,000 labeled held-out training examples per task, ties to smaller λ), i.e. the paper's tuned TA. Also reported: the evaluation-optimal λ (unattainable floor).
- **Regret** of rule R on a pair = 100 × (D(λ_R) − D(λ_sel)), in pp of mean normalized retention (it can be negative).
- **% avoidable loss removed** relative to baseline B ∈ {λ=1, λ=0.5} = 1 − mean regret_R / mean regret_B (ratio of means over pairs).
- Continuous λ outputs are snapped to the nearest grid point (linear distance; ties to the smaller λ).

## λ rules (no held-out labels for the pair's own tasks)
Weight-only, nothing fitted:
- **W1** λ = 1.0 (sum; baseline).
- **W2** λ = 0.5 (average; baseline). Note: the per-layer least-squares coefficient that minimizes Σ_t‖λ_ℓ(Δ1ℓ+Δ2ℓ) − Δtℓ‖² is exactly ½ in every layer, so the "least-squares per-layer analytic λ" coincides with W2 and is not a separate rule.
- **W3** norm-preserving λ_np = ½(‖Δ1‖+‖Δ2‖)/‖Δ1+Δ2‖ (global task-vector norms and inner product), so the merged update keeps the mean single-update norm.
- **W4** per-layer projection-preserving λ_pp: λ_ℓ = ½ Σ_t ‖Δtℓ‖²/(‖Δtℓ‖² + ⟨Δ1ℓ,Δ2ℓ⟩), clipped to [0.25, 4], aggregated by an energy-weighted mean (weights ‖Δ1ℓ‖²+‖Δ2ℓ‖²). This λ makes each task's own component of the merged update, projected onto that task's update, equal to the update itself. Uniform λ only: per-layer λ vectors cannot be scored without new GPU merges.

Fitted on other pairs (weights plus labeled results of *other* pairs; never a task of the test pair):
- **F1** global constant λ = the grid point minimizing mean D over the fit pairs.
- **F2** ridge regression (α = 1, standardized features) from features X to λ_sel of the fit pairs; the prediction is snapped.
- **F3** per-λ ridge "loss model": for each grid λ, ridge (α = 1) from X to D(λ) on the fit pairs; choose the argmin of the predicted D.

Features X (scale-free, 10): log norm ratio, task-vector cosine, log O_A, mean θ_min (A-side, degrees), fraction of layers with θ_min < 30°, sign conflict (top 20%), λ_np, λ_pp, max_ℓ|cos_ℓ| (per-layer task-vector cosine), and energy-weighted mean |½ log(‖Δ1ℓ‖²/‖Δ2ℓ‖²)| (layer dominance). The rank is constant (8), so it is not a feature.

Needs unlabeled data (clearly separated):
- **U1** agreement-max λ: choose λ ∈ grid to maximize the mean over the pair's two tasks of the agreement rate between the merged model's and that task's single adapter's argmax predictions on unlabeled calibration inputs. Calibration = n_cal = min(200, ⌊n_eval/2⌋) examples per task, sampled once per task (seed = 20260927 + task index) from the evaluation set. Labels are never used. Cost: two single-adapter passes plus |grid| merged passes over 2·n_cal inputs (CPU-feasible).
- Reference, **not** a label-free rule: **L-cal**, the λ that maximizes normalized accuracy on the same n_cal examples *with* labels.
- In the U analysis, every rule (W, F, U, L-cal, λ_sel) is re-scored on the remaining evaluation examples only (eval minus calibration), so the comparison is fair.

## Merge / no-merge decision
Ground truth y = 1[D(λ_sel) > τ], with τ = 0.05 primary (merged model retains < 95% of single-adapter performance on average even after held-out tuning) and τ ∈ {0.02, 0.10} as sensitivity.
- **M1** L2 logistic regression (penalty ½‖w‖², standardized X), fitted under the same CV schemes; decision at p ≥ 0.5.
- **M2** unfitted single scores (AUROC only; sign fixed a priori so that higher means worse): tv_cosine, O_A, sign_conflict_top20, max_ℓ|cos_ℓ|, |log norm ratio|, −mean θ_min.
- **M3** (unlabeled) D̂_U = 1 − (max over λ of the U1 agreement). AUROC; decision D̂_U > τ, unfitted.
- **M0** baseline: always merge (accuracy = 1 − prevalence). Reference (uses held-out labels): 1 − hold_norm(λ_sel).

## Cross-validation and inference
- **Within-backbone, both-tasks-out:** for each test pair (a, b), the fit set is every pair of the same backbone (all three populations pooled) that contains neither task a nor task b. So no task appears in both fit and test.
- **Transfer, leave-one-backbone-out (LOBO):** the fit set is the pairs of the other two backbones, again excluding pairs that contain task a or b (encoders → Qwen, BERT+Qwen → RoBERTa, RoBERTa+Qwen → BERT).
- CIs: task-block bootstrap, 2,000 replicates, seed 20260927. Tasks are resampled with replacement and pair (a, b) gets weight c_a·c_b. For a backbone summary, the draw applies jointly to its three populations; for the overall summary it applies jointly over the union of task names. CV predictions are held fixed inside the bootstrap (no refitting), so the CIs do not include fitting variability.
- Qwen G7: W1–W4, F1 (within-Qwen) and U1 are re-evaluated on G7 against λ_sel on G7.

## Pre-declared reading
A rule is called "practically useful" only if its mean regret CI lies entirely below the better of the two fixed baselines (W1, W2) in the target population *and* it removes ≥ 50% of the avoidable loss relative to that better baseline. A merge classifier is useful only if AUROC's lower CI > 0.5 **and** its accuracy exceeds M0's. Everything is descriptive; no p-values are used for decisions.
