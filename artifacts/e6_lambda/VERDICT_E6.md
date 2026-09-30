# VERDICT E6: pre-merge λ rules and merge/no-merge decisions (POST HOC, EXPLORATORY)

*Generated 2026-09-27 22:40 KST by `code/make_verdict_e6.py` from `results/*.csv`. Rules were fixed in `RULES.md` (sha256 `199a9cb7c08d9224c1913b7e12f0059f7f6fcec26ac78bfb54d6724ef0709f0c`, written 2026-09-27 22:32:02 KST) before any rule was scored. Clarification C1 and the post-results additions A1–A3 are listed in `DEVIATIONS_E6.md`. Nothing here is pre-registered, and E6 does not alter any pre-registered verdict.*

## 1 Data inventory

| backbone | populations (pairs) | λ grid with per-task held-out + eval scores | eval-set predictions (argmax / STS-B scores) | source |
|---|---|---|---|---|
| BERT-base | R0 91, S1 91, P 91 | {0.3, 0.5, 0.7, 1.0} | merged at every λ + singles (s0, s1) | e1b/results, e1c |
| RoBERTa-base | R0 78, S1 78, P 78 (RTE excluded by integrity rule) | {0.3, 0.5, 0.7, 1.0} | same | e4a, e5/work/e4a_roberta/preds |
| Qwen2.5-0.5B | R0 91, S1 91, P 91 | {0.3, …, 1.0} ∪ {1.3, 1.5, 2.0} (E5b) | same, incl. extended λ | e4b, e5/e5b, e5/work/e4b_decoder/preds |

Per-layer task-vector norms and inner products were computed from the adapter weights on the 4070 **CPU** (`code/layer_stats_cpu.py`; no GPU job). The global cosine and norm ratio they imply match the frozen predictor tables (max |Δcos| = 2.7e-11). Metrics recomputed from the saved predictions match all 7878 reported per-task scores (max |Δ| = 1.1e-16). Excluded: same-task S2 pairs and the 21 E1 Hub pairs. Consolidated table: `data/pairs_long.csv`; U-analysis table: `data/pairs_unlabeled.csv`; raw copies with SHA256SUMS: `data/src/`.

## 2 Rules tested (from RULES.md)

W1 λ = 1 · W2 λ = 0.5 (identical to the per-layer least-squares λ) · W3 norm-preserving λ_np · W4 per-layer projection-preserving λ_pp · F1 global λ fitted on other tasks · F2 ridge features → λ* · F3 ridge per-λ loss model · U1 agreement-max λ on n_cal ≤ 200 **unlabeled** inputs per task (needs unlabeled data) · L-cal labeled-calibration reference · M1 logistic merge classifier · M2 single weight scores · M3 unlabeled disagreement score (+ M3c calibrated threshold, added post hoc). CV: both-tasks-out within backbone; LOBO transfer (fit on the other two backbones, also both-tasks-out).

**What the closed-form rules reduce to.** λ_np ranges over [0.57, 0.707] and snaps to 0.7 in 98.3% of pairs. With near-orthogonal task vectors (|cos| ≈ 0) and moderate norm ratios, λ_np ≈ 1/√2. λ_pp ranges over [0.923, 1.015] and always snaps to 1.0, so W4 ≡ W1. F1 chose 0.7 in 100% of within-backbone folds and 100% of LOBO folds. In practice, then, the weight-only rules are two constants: 0.7 and 1.0.

## 3 Choosing λ without held-out data: regret vs held-out-tuned λ* (pp of normalized retention; full evaluation sets; task-block 95% CI)

| scope | W1 λ=1 | W2 λ=0.5 | W3 λ_np (≈0.7) | F1 within | F2 within | F3 within | F2 LOBO | F3 LOBO | eval-optimal (floor) |
|---|---|---|---|---|---|---|---|---|---|
| bert/R0 | 5.59 [2.65, 9.18] | 2.37 [1.11, 3.96] | 0.79 [0.11, 1.80] | 0.76 [0.09, 1.78] | 1.45 [0.22, 3.50] | 1.08 [0.23, 2.20] | 1.35 [0.25, 3.01] | 1.80 [0.44, 3.79] | -0.13 [-0.34, -0.03] |
| bert/S1 | 5.76 [2.54, 10.03] | 2.37 [1.35, 3.69] | 0.76 [0.08, 2.11] | 0.75 [0.08, 2.09] | 1.30 [0.19, 3.05] | 0.96 [0.21, 2.07] | 1.92 [0.33, 4.34] | 1.77 [0.60, 3.73] | -0.14 [-0.30, -0.02] |
| bert/P | 4.85 [2.20, 8.62] | 2.63 [1.61, 3.96] | 0.63 [0.05, 1.84] | 0.70 [0.08, 1.98] | 1.01 [0.08, 2.42] | 0.96 [0.23, 2.05] | 1.72 [0.32, 3.75] | 1.98 [0.76, 3.76] | -0.10 [-0.30, -0.00] |
| bert/all | 5.40 [2.63, 9.38] | 2.46 [1.34, 3.75] | 0.72 [0.14, 1.81] | 0.74 [0.14, 1.82] | 1.25 [0.21, 2.96] | 1.00 [0.30, 1.91] | 1.66 [0.35, 3.52] | 1.85 [0.73, 3.65] | -0.13 [-0.26, -0.04] |
| roberta/R0 | 1.09 [0.39, 2.22] | 1.00 [0.39, 1.84] | 0.23 [-0.05, 0.78] | 0.22 [-0.06, 0.79] | 0.28 [-0.04, 0.91] | 0.36 [0.00, 0.95] | 0.28 [-0.06, 0.87] | 0.31 [-0.08, 1.08] | -0.12 [-0.21, -0.05] |
| roberta/S1 | 0.97 [0.30, 2.16] | 1.00 [0.46, 1.77] | 0.16 [-0.14, 0.62] | 0.13 [-0.17, 0.60] | 0.28 [-0.04, 0.77] | 0.30 [-0.03, 0.82] | 0.30 [-0.10, 0.94] | 0.28 [-0.12, 0.95] | -0.20 [-0.41, -0.06] |
| roberta/P | 0.95 [0.28, 1.98] | 1.02 [0.53, 1.59] | 0.10 [-0.11, 0.39] | 0.09 [-0.12, 0.38] | 0.13 [-0.07, 0.40] | 0.18 [-0.06, 0.52] | 0.07 [-0.16, 0.37] | 0.07 [-0.16, 0.37] | -0.17 [-0.46, -0.03] |
| roberta/all | 1.01 [0.35, 2.05] | 1.01 [0.49, 1.72] | 0.17 [-0.07, 0.56] | 0.15 [-0.08, 0.55] | 0.23 [-0.01, 0.61] | 0.28 [0.01, 0.72] | 0.22 [-0.05, 0.66] | 0.22 [-0.06, 0.74] | -0.16 [-0.29, -0.07] |
| qwen/R0 | 3.23 [1.68, 5.22] | 1.11 [0.41, 1.96] | 1.08 [0.31, 2.47] | 1.08 [0.31, 2.47] | 1.23 [0.39, 2.61] | 1.44 [0.53, 2.79] | 0.99 [0.41, 1.73] | 9.06 [5.48, 12.87] | -0.13 [-0.30, -0.03] |
| qwen/S1 | 2.01 [0.80, 3.68] | 2.51 [1.24, 4.35] | 0.78 [0.11, 1.65] | 0.78 [0.11, 1.65] | 0.95 [0.28, 1.83] | 0.98 [0.32, 1.87] | 2.96 [0.82, 6.26] | 9.55 [5.52, 14.30] | -0.14 [-0.31, -0.03] |
| qwen/P | 0.90 [0.12, 2.46] | 3.98 [2.31, 6.19] | 0.88 [0.27, 1.74] | 0.88 [0.27, 1.74] | 0.88 [0.09, 2.45] | 1.08 [0.15, 2.60] | 5.52 [1.81, 10.95] | 10.82 [6.94, 15.44] | -0.13 [-0.29, -0.02] |
| qwen/all | 2.04 [1.01, 3.50] | 2.53 [1.52, 3.94] | 0.92 [0.34, 1.78] | 0.92 [0.34, 1.78] | 1.02 [0.34, 1.99] | 1.17 [0.45, 2.06] | 3.16 [1.35, 5.89] | 9.81 [6.69, 13.59] | -0.13 [-0.24, -0.06] |
| all/all | 2.91 [1.53, 4.59] | 2.05 [1.45, 2.87] | 0.62 [0.26, 1.14] | 0.62 [0.25, 1.14] | 0.86 [0.31, 1.58] | 0.84 [0.40, 1.35] | 1.75 [0.73, 3.03] | 4.15 [2.70, 5.80] | -0.14 [-0.23, -0.07] |

**% of avoidable loss removed** (1 − mean regret_rule / mean regret_baseline):

| scope | W3 vs λ=1 | W3 vs λ=0.5 | F2 within vs λ=1 | F3 within vs λ=1 | F3 LOBO vs λ=1 |
|---|---|---|---|---|---|
| bert/all | 87% [75, 97] | 71% [1, 95] | 77% [32, 95] | 81% [61, 93] | 66% [45, 84] |
| roberta/all | 84% [40, 109] | 84% [53, 110] | 77% [29, 101] | 72% [21, 98] | 78% [13, 108] |
| qwen/all | 55% [31, 77] | 64% [18, 86] | 50% [29, 72] | 43% [2, 69] | -380% [-1086, -121] |
| qwen/P | 1% [-676, 66] | 78% [62, 90] | 2% [-137, 72] | -20% [-340, 63] | -1108% [-10819, -277] |
| all/all | 79% [69, 87] | 70% [37, 87] | 70% [36, 86] | 71% [49, 83] | -43% [-172, 22] |

**Qwen with the extended grid G7** (λ up to 2.0; oracle = held-out selection on G7):

| scope | W1 | W2 | W3 (≈0.7) | F1 within |
|---|---|---|---|---|
| qwen/R0 | 3.23 [1.68, 5.22] | 1.11 [0.41, 1.96] | 1.08 [0.31, 2.47] | 1.08 [0.31, 2.47] |
| qwen/S1 | 2.10 [0.86, 3.75] | 2.61 [1.25, 4.39] | 0.87 [0.12, 1.83] | 0.87 [0.12, 1.83] |
| qwen/P | 0.93 [0.13, 2.50] | 4.01 [2.32, 6.22] | 0.91 [0.28, 1.81] | 0.91 [0.28, 1.81] |
| qwen/all | 2.09 [1.03, 3.56] | 2.58 [1.54, 3.97] | 0.96 [0.35, 1.84] | 0.96 [0.35, 1.84] |

## 4 λ from unlabeled data (U analysis: calibration inputs ⊂ eval set, labels unused; all rules re-scored on the remaining eval examples)

| scope | W1 | W3 (≈0.7) | **U1 agreement (unlabeled)** | L-cal (same n, labeled; reference) | U1 % removed vs λ=1 | U1 % removed vs λ=0.5 | U1 − W3 (paired) |
|---|---|---|---|---|---|---|---|
| bert/R0 | 5.51 [2.62, 9.02] | 0.68 [-0.06, 1.73] | **0.27 [-0.18, 0.71]** | 0.50 [-0.05, 1.06] | 95% [85, 104] | 89% [65, 110] | -0.42 [-1.35, 0.28] |
| bert/S1 | 5.50 [2.28, 9.78] | 0.72 [0.01, 2.02] | **0.27 [-0.08, 0.93]** | 0.17 [-0.11, 0.51] | 95% [83, 102] | 88% [50, 103] | -0.45 [-1.60, 0.21] |
| bert/P | 4.82 [2.20, 8.56] | 0.62 [0.05, 1.84] | **0.19 [0.03, 0.48]** | 0.21 [-0.04, 0.64] | 96% [90, 99] | 93% [80, 99] | -0.43 [-1.72, 0.20] |
| bert/all | 5.28 [2.61, 9.20] | 0.67 [0.10, 1.74] | **0.24 [0.02, 0.59]** | 0.29 [0.10, 0.58] | 95% [89, 100] | 90% [70, 99] | -0.43 [-1.42, 0.15] |
| roberta/R0 | 0.98 [0.34, 2.03] | 0.22 [-0.07, 0.80] | **0.10 [-0.05, 0.30]** | 0.02 [-0.09, 0.16] | 90% [55, 106] | 90% [76, 109] | -0.12 [-0.64, 0.14] |
| roberta/S1 | 0.97 [0.27, 2.20] | 0.19 [-0.15, 0.63] | **0.14 [-0.15, 0.45]** | 0.15 [-0.15, 0.45] | 85% [22, 124] | 86% [61, 123] | -0.04 [-0.42, 0.30] |
| roberta/P | 0.90 [0.27, 1.87] | 0.06 [-0.11, 0.29] | **-0.06 [-0.32, 0.05]** | -0.01 [-0.27, 0.17] | 107% [90, 151] | 107% [93, 131] | -0.12 [-0.47, 0.08] |
| roberta/all | 0.95 [0.32, 1.98] | 0.16 [-0.07, 0.53] | **0.06 [-0.08, 0.19]** | 0.05 [-0.12, 0.20] | 94% [72, 110] | 94% [80, 109] | -0.10 [-0.49, 0.14] |
| qwen/R0 | 3.05 [1.52, 5.08] | 1.11 [0.34, 2.43] | **0.05 [-0.18, 0.24]** | 0.13 [-0.09, 0.38] | 98% [90, 106] | 96% [70, 126] | -1.06 [-2.49, -0.24] |
| qwen/S1 | 2.08 [0.75, 3.86] | 0.76 [0.05, 1.71] | **-0.00 [-0.22, 0.28]** | 0.07 [-0.26, 0.58] | 100% [85, 117] | 100% [84, 110] | -0.77 [-1.71, -0.02] |
| qwen/P | 0.99 [0.18, 2.54] | 0.80 [0.17, 1.71] | **0.05 [-0.19, 0.33]** | 0.10 [-0.16, 0.41] | 94% [42, 130] | 99% [92, 107] | -0.75 [-1.71, -0.11] |
| qwen/all | 2.04 [0.95, 3.54] | 0.89 [0.30, 1.75] | **0.03 [-0.09, 0.20]** | 0.10 [-0.06, 0.31] | 98% [88, 105] | 99% [92, 105] | -0.86 [-1.75, -0.25] |
| all/all | 2.85 [1.47, 4.53] | 0.60 [0.23, 1.11] | **0.11 [0.01, 0.25]** | 0.15 [0.05, 0.29] | 96% [90, 99] | 94% [87, 99] | -0.48 [-0.95, -0.11] |
| qwen/P (G7) | 1.03 [0.21, 2.57] | 0.84 [0.18, 1.78] | **0.09 [-0.19, 0.39]** | 0.13 [-0.10, 0.44] | 91% [34, 130] | 98% [91, 107] | – |
| qwen/all (G7) | 2.08 [0.98, 3.60] | 0.93 [0.31, 1.84] | **0.04 [-0.11, 0.22]** | 0.13 [-0.04, 0.37] | 98% [86, 106] | 98% [91, 106] | – |

## 5 Practical usefulness (A1: paired CI of regret − better fixed baseline < 0 and ≥ 50% removed)

| scope | better baseline | W3/≈0.7 (full eval) | F2 within | F3 LOBO | U1 (test remainder) |
|---|---|---|---|---|---|
| bert/R0 | W2 | no (-1.58 [-3.21, +0.22]) | no (-0.91 [-2.40, +1.14]) | no (-0.57 [-2.72, +2.24]) | **yes** (-2.08 [-3.52, -0.93]) |
| bert/S1 | W2 | no (-1.62 [-3.06, +0.29]) | no (-1.08 [-2.46, +0.95]) | no (-0.61 [-2.60, +1.95]) | **yes** (-2.01 [-3.44, -0.71]) |
| bert/P | W2 | **yes** (-2.01 [-3.37, -0.44]) | **yes** (-1.62 [-2.81, -0.31]) | no (-0.66 [-2.59, +1.77]) | **yes** (-2.40 [-3.76, -1.37]) |
| bert/all | W2 | **yes** (-1.74 [-3.13, -0.01]) | no (-1.20 [-2.47, +0.57]) | no (-0.61 [-2.50, +1.86]) | **yes** (-2.16 [-3.47, -1.04]) |
| roberta/R0 | W2 | **yes** (-0.78 [-1.32, -0.22]) | **yes** (-0.72 [-1.27, -0.17]) | **yes** (-0.69 [-1.10, -0.19]) | **yes** (-0.87 [-1.53, -0.36]) |
| roberta/S1 | W1 | **yes** (-0.81 [-1.79, -0.03]) | no (-0.70 [-1.67, +0.05]) | no (-0.70 [-1.86, +0.32]) | **yes** (-0.83 [-2.03, -0.08]) |
| roberta/P | W1 | **yes** (-0.85 [-1.81, -0.20]) | **yes** (-0.82 [-1.78, -0.19]) | **yes** (-0.88 [-1.86, -0.22]) | **yes** (-0.96 [-2.01, -0.29]) |
| roberta/all | W1 | **yes** (-0.84 [-1.77, -0.18]) | **yes** (-0.78 [-1.70, -0.12]) | **yes** (-0.79 [-1.76, -0.06]) | **yes** (-0.89 [-1.93, -0.26]) |
| qwen/R0 | W2 | no (-0.03 [-1.42, +1.62]) | no (+0.11 [-1.22, +1.71]) | no (+7.94 [+4.92, +11.16]) | **yes** (-1.03 [-1.89, -0.34]) |
| qwen/S1 | W1 | **yes** (-1.23 [-2.23, -0.37]) | **yes** (-1.06 [-2.09, -0.09]) | no (+7.54 [+3.18, +12.93]) | **yes** (-2.08 [-3.83, -0.80]) |
| qwen/P | W1 | no (-0.01 [-1.16, +1.07]) | no (-0.02 [-0.40, +0.40]) | no (+9.93 [+5.76, +14.98]) | **yes** (-0.94 [-2.55, -0.12]) |
| qwen/all | W1 | **yes** (-1.13 [-1.92, -0.39]) | **yes** (-1.03 [-1.65, -0.38]) | no (+7.76 [+3.91, +12.20]) | **yes** (-2.01 [-3.50, -0.90]) |
| all/all | W2 | **yes** (-1.43 [-2.34, -0.61]) | **yes** (-1.19 [-1.85, -0.44]) | no (+2.10 [+0.80, +3.58]) | **yes** (-1.89 [-2.69, -1.30]) |

## 6 Merge / no-merge: y = 1[D(λ*) > τ], τ = 0.05 primary

| scope | prevalence | M1 within AUROC | M1 LOBO AUROC | best M2 (weights) AUROC | REF held-out labels AUROC | M1 within acc | always-merge acc |
|---|---|---|---|---|---|---|---|
| bert/all | 0.38 | 0.40 [0.27, 0.61] | 0.53 [0.40, 0.68] | abs_log_norm_ratio 0.63 [0.48, 0.75] | 0.95 [0.89, 0.99] | 0.55 [0.40, 0.70] | 0.62 [0.40, 0.82] |
| roberta/all | 0.07 | 0.34 [0.10, 0.80] | 0.55 [0.28, 0.93] | O_A 0.68 [0.06, 0.96] | 0.99 [0.94, 1.00] | 0.92 [0.77, 1.00] | 0.93 [0.78, 1.00] |
| qwen/all | 0.29 | 0.38 [0.24, 0.53] | 0.46 [0.30, 0.63] | O_A 0.60 [0.43, 0.74] | 0.97 [0.93, 1.00] | 0.63 [0.46, 0.78] | 0.71 [0.51, 0.88] |
| qwen/P | 0.26 | 0.40 [0.12, 0.68] | 0.64 [0.26, 0.92] | tv_cosine 0.65 [0.42, 0.83] | 0.99 [0.95, 1.00] | 0.69 [0.49, 0.88] | 0.74 [0.52, 0.93] |
| all/all | 0.26 | 0.56 [0.49, 0.65] | 0.41 [0.30, 0.54] | abs_log_norm_ratio 0.58 [0.46, 0.68] | 0.97 [0.94, 0.99] | 0.69 [0.55, 0.80] | 0.74 [0.58, 0.89] |

Unlabeled score M3 (test remainder; y recomputed on the same examples):

| τ | scope | prevalence | M3 AUROC (unlabeled) | M3 acc, unfitted D̂>τ | M3c acc, within (post hoc) | M3c acc, LOBO (post hoc) | always-merge acc | REF held-out AUROC |
|---|---|---|---|---|---|---|---|---|
| 0.02 | bert/all | 0.79 | 0.79 [0.56, 0.97] | 0.79 [0.61, 0.93] | 0.85 [0.73, 0.95] | 0.83 [0.70, 0.94] | 0.21 [0.07, 0.39] | 0.88 [0.77, 0.97] |
| 0.02 | roberta/all | 0.50 | 0.86 [0.72, 0.95] | 0.51 [0.26, 0.72] | 0.72 [0.57, 0.83] | 0.78 [0.64, 0.90] | 0.50 [0.28, 0.76] | 0.90 [0.81, 0.96] |
| 0.02 | qwen/all | 0.59 | 0.87 [0.74, 0.96] | 0.63 [0.41, 0.82] | 0.74 [0.57, 0.86] | 0.79 [0.63, 0.90] | 0.41 [0.20, 0.65] | 0.91 [0.83, 0.97] |
| 0.02 | all/all | 0.63 | 0.86 [0.74, 0.94] | 0.65 [0.48, 0.78] | 0.77 [0.70, 0.83] | 0.80 [0.73, 0.86] | 0.37 [0.22, 0.56] | 0.91 [0.84, 0.95] |
| 0.05 | bert/all | 0.36 | 0.90 [0.79, 0.98] | 0.45 [0.30, 0.62] | 0.79 [0.67, 0.89] | 0.82 [0.71, 0.92] | 0.64 [0.43, 0.84] | 0.93 [0.85, 0.98] |
| 0.05 | roberta/all | 0.08 | 0.96 [0.88, 1.00] | 0.52 [0.31, 0.78] | 0.94 [0.84, 1.00] | 0.95 [0.85, 1.00] | 0.92 [0.78, 1.00] | 0.97 [0.92, 1.00] |
| 0.05 | qwen/all | 0.28 | 0.93 [0.84, 0.99] | 0.55 [0.38, 0.72] | 0.85 [0.72, 0.95] | 0.85 [0.73, 0.94] | 0.72 [0.52, 0.88] | 0.97 [0.91, 1.00] |
| 0.05 | all/all | 0.25 | 0.93 [0.87, 0.98] | 0.51 [0.40, 0.63] | 0.86 [0.76, 0.93] | 0.87 [0.78, 0.93] | 0.75 [0.61, 0.89] | 0.96 [0.91, 0.98] |
| 0.1 | bert/all | 0.05 | 0.97 [0.89, 1.00] | 0.50 [0.28, 0.74] | 0.95 [0.87, 1.00] | 0.92 [0.81, 0.99] | 0.95 [0.85, 1.00] | 1.00 [0.98, 1.00] |
| 0.1 | roberta/all | 0.01 | 0.98 [0.91, 1.00] | 0.90 [0.71, 1.00] | 0.98 [0.94, 1.00] | 0.99 [0.94, 1.00] | 0.99 [0.94, 1.00] | 0.99 [0.96, 1.00] |
| 0.1 | qwen/all | 0.09 | 0.96 [0.90, 1.00] | 0.71 [0.50, 0.89] | 0.91 [0.80, 0.98] | 0.94 [0.84, 0.99] | 0.91 [0.80, 0.98] | 0.98 [0.93, 1.00] |
| 0.1 | all/all | 0.05 | 0.96 [0.92, 0.99] | 0.69 [0.51, 0.87] | 0.95 [0.88, 0.99] | 0.95 [0.89, 0.99] | 0.95 [0.90, 0.99] | 0.98 [0.96, 1.00] |

## 7 Conclusion

This analysis is post hoc and exploratory. Among weight-only rules, nothing adaptive beat a constant. The only closed-form rule that helped, the norm-preserving λ_np, is in effect the constant 0.7 (≈ 1/√2 for near-orthogonal updates; it snaps to 0.7 in 98% of pairs). Over all 780 cross-task pairs it cut regret against held-out-tuned λ* from 2.91 [1.53, 4.59] pp (λ = 1) and 2.05 [1.45, 2.87] pp (λ = 0.5) to 0.62 [0.26, 1.14] pp. That removes 79% [69, 87] of the avoidable loss of λ = 1 and 70% [37, 87] of that of λ = 0.5. The gain is backbone-dependent: BERT 0.72 [0.14, 1.81], RoBERTa 0.17 [-0.07, 0.56], Qwen 0.92 [0.34, 1.78]. In the primary Qwen population P, λ = 0.7 is no better than λ = 1 (0.88 [0.27, 1.74] vs 0.90 [0.12, 2.46]). Regressions from weight features to λ* (F2 0.86 [0.31, 1.58], F3 0.84 [0.40, 1.35]) did not beat the constant, and transferred badly from encoders to the decoder (F3 LOBO on Qwen 9.81 [6.69, 13.59]). The per-layer projection-preserving λ reduced to λ = 1, and the per-layer least-squares λ is ½ by construction. With a few hundred **unlabeled** in-distribution inputs, the picture changes. Choosing λ to maximize agreement with the two single adapters (U1) left 0.11 [0.01, 0.25] pp of regret and removed 96% [90, 99] of λ = 1's avoidable loss. That is -0.48 [-0.95, -0.11] pp relative to λ = 0.7 and statistically indistinguishable from labeled selection on the same n (-0.04 [-0.12, 0.05] pp). For the merge/no-merge decision (τ = 0.05, prevalence 0.26), weight features carried no usable signal: the logistic classifier had AUROC 0.56 [0.49, 0.65] within backbone and 0.41 [0.30, 0.54] in transfer, and its accuracy of 0.69 [0.55, 0.80] did not beat always-merge (0.74 [0.58, 0.89]). The unlabeled disagreement score ranked pairs almost as well as held-out labels (AUROC 0.93 [0.87, 0.98] vs 0.96 [0.91, 0.98] on the same examples). But its pre-declared unfitted threshold was miscalibrated (accuracy 0.51 [0.40, 0.63] vs 0.75 [0.61, 0.89]). A threshold calibrated on other tasks' pairs (post hoc, M3c) reached 0.87 [0.78, 0.93] in LOBO transfer (+0.12 [+0.02, +0.23] over always-merge). Practical reading: without any data, use λ ≈ 0.7 rather than 1 or 0.5 on encoders (on the decoder it is a hedge, not an improvement). If unlabeled inputs are available, agreement-based selection comes within about 0.1 pp of held-out tuning with 1,000 labels per task and matches labeled selection on the same few hundred inputs. For the merge decision it ranks pairs nearly as well as held-out labels, but it needs a calibrated threshold. Weight-space features should not be used for either.

**Threshold sensitivity (M3c, LOBO, all pairs).** At τ = 0.02, accuracy was 0.80 [0.73, 0.86] against always-merge 0.37. At τ = 0.10 (prevalence 0.05), accuracy was 0.95 [0.89, 0.99] against 0.95. When harmful merges are that rare, no rule beats always-merging on accuracy.

## 8 Limitations

- Post hoc. The data were collected for the pre-registered E1b/E1c/E4a/E4b/E5 questions, and the analyst knew from the manuscript that λ* is often 0.7 on BERT and 1.0 on Qwen-P. The constant 0.7 was not pre-declared as a rule; it arises from W3 and F1.
- Rules can only pick grid points (G4; G7 for Qwen), because scoring a continuous or per-layer λ would require new GPU merges. The regret of λ_np at its exact value is therefore unknown.
- The U1/M3 calibration inputs are unlabeled examples from the *evaluation* distribution, disjoint from the scored examples. With inputs from a shifted distribution, performance may differ. U1 costs 2 + |grid| forward passes over 2·n_cal inputs per pair. Where predictions are argmax labels only, entropy- or loss-based unlabeled proxies could not be computed.
- CV predictions are held fixed inside the task-block bootstrap, so the CIs do not include refitting variability. Populations within a backbone share adapters and tasks. With K = 13–14 tasks per backbone, intervals are wide.
- Pairwise merges of rank-8 LoRA adapters with separate classification heads on three backbones ≤ 0.5B parameters. We do not know whether the findings hold for multi-adapter merges, generative tasks or larger models.
