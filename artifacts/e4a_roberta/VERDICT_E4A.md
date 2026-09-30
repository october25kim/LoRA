# E4A VERDICT (RoBERTa-base) — H1 (O_A): **INCONCLUSIVE**; H2 (task-vector cosine): **INCONCLUSIVE**  (primary population P, mixed seed)

Generated 2026-09-26 15:01:15 KST. Pre-registration: `PREREG_E4A.md` / `prereg_e4a.json` (sha256 in `prereg_e4a.sha256`); deviations/notes: `DEVIATIONS_E4A.md`.

Valid tasks K = 13: cola, sst2, mrpc, stsb, mnli, qnli, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity. Exclusions: {'rte': {'s0': ['(b) accuracy 0.7004 < 0.7370'], 's1': ['(b) accuracy 0.7040 < 0.7370']}}. Predictor hash ok: True. Analysis-code equivalence on E1b (reproduces E1b analysis.json): True.

Pilot (pre-registered, held-out only): lr_main = 0.0005 (decided). lr 0.001: sst2 hold 0.9530 (maj 0.560, tail loss 0.142, diverged False), rte hold 0.5120 (maj 0.488, tail loss 0.695, diverged True); lr 0.0005: sst2 hold 0.9580 (maj 0.560, tail loss 0.116, diverged False), rte hold 0.7420 (maj 0.488, tail loss 0.096, diverged False)

## Primary: mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (P) | O_A | 0.1606 | [-0.342, 0.589] | 0.2095 | 0.4190 | 0.31 | **INCONCLUSIVE** |
| H2 (P) | tv_cosine | 0.1123 | [-0.423, 0.554] | 0.2379 | 0.4190 | 0.23 | **INCONCLUSIVE** |

Rule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).

H1 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.18, sst2 0.20, mrpc 0.09, stsb 0.18, mnli 0.30, qnli 0.04, wic 0.10, snli 0.09, scitail 0.16, ag_news 0.16, imdb 0.24, trec 0.09, yelp_polarity 0.25

H2 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.11, sst2 0.23, mrpc 0.06, stsb 0.08, mnli 0.20, qnli 0.00, wic 0.07, snli 0.04, scitail 0.06, ag_news 0.17, imdb 0.23, trec 0.04, yelp_polarity 0.15

P: 78 pairs; D mean 0.0255, median 0.0207, range [0.0030, 0.1082]; λ* counts {'0.5': 15, '0.7': 42, '1.0': 21}; ρ(O_A, tv_cos) = 0.721; min θ_min over pairs/layers = 37.2°; gate active in 0 pairs.

## Secondary (exploratory; no multiplicity correction across populations)

### R0: seed-0 cross-task pairs (E1b confirmatory design on RoBERTa; rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (R0) | O_A | 0.1459 | [-0.368, 0.601] | 0.2195 | 0.4390 | 0.23 | **INCONCLUSIVE** |
| H2 (R0) | tv_cosine | 0.1296 | [-0.323, 0.527] | 0.2387 | 0.4390 | 0.15 | **INCONCLUSIVE** |

R0: 78 pairs; D mean 0.0266, median 0.0207, range [0.0017, 0.1073]; λ* counts {'0.5': 13, '0.7': 40, '1.0': 25}; ρ(O_A, tv_cos) = 0.873; min θ_min over pairs/layers = 35.9°; gate active in 0 pairs.

### S1: seed-1 cross-task pairs (rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (S1) | O_A | 0.0373 | [-0.422, 0.500] | 0.4383 | 0.8705 | 0.08 | **INCONCLUSIVE** |
| H2 (S1) | tv_cosine | 0.0375 | [-0.424, 0.446] | 0.4353 | 0.8705 | 0.00 | **INCONCLUSIVE** |

S1: 78 pairs; D mean 0.0235, median 0.0198, range [0.0020, 0.0874]; λ* counts {'0.5': 14, '0.7': 42, '1.0': 22}; ρ(O_A, tv_cos) = 0.893; min θ_min over pairs/layers = 37.2°; gate active in 0 pairs.

### Pooled 3-population summary (mean D and predictors over R0, S1, P per task pair; descriptive)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (pooled) | O_A | 0.1464 | [-0.365, 0.587] | 0.2420 | 0.4840 | 0.31 | **INCONCLUSIVE** |
| H2 (pooled) | tv_cosine | 0.1375 | [-0.353, 0.599] | 0.2481 | 0.4840 | 0.31 | **INCONCLUSIVE** |

Pooled D mean 0.0252, n = 78 task pairs.

### Secondary predictors

| predictor | ρ P | CI P | ρ R0 | CI R0 | ρ S1 | CI S1 |
|---|---:|---|---:|---|---:|---|
| O_A | 0.161 | [-0.34, 0.59] | 0.146 | [-0.37, 0.60] | 0.037 | [-0.42, 0.50] |
| tv_cosine | 0.112 | [-0.42, 0.55] | 0.130 | [-0.32, 0.53] | 0.038 | [-0.42, 0.45] |
| mean_theta_min_A_deg | -0.155 | [-0.58, 0.33] | -0.117 | [-0.57, 0.38] | -0.041 | [-0.50, 0.41] |
| min_theta_min_A_deg | -0.299 | [-0.59, 0.11] | -0.242 | [-0.54, 0.16] | -0.184 | [-0.61, 0.30] |
| n_layers_theta_min_lt30_A | nan | [nan, nan] | nan | [nan, nan] | nan | [nan, nan] |
| sign_conflict_top20 | -0.141 | [-0.57, 0.37] | -0.154 | [-0.55, 0.32] | -0.057 | [-0.47, 0.40] |
| norm_ratio | 0.231 | [-0.17, 0.55] | 0.249 | [-0.08, 0.53] | 0.395 | [-0.02, 0.70] |
| null_z_O_A | 0.161 | [-0.34, 0.59] | 0.146 | [-0.37, 0.60] | 0.037 | [-0.42, 0.50] |
| sign_conflict_all | -0.088 | [-0.53, 0.42] | -0.047 | [-0.47, 0.40] | 0.033 | [-0.36, 0.47] |

Fixed-λ ρ (P): {"O_A": {"lam0.3": -0.012, "lam0.5": 0.035, "lam0.7": 0.216, "lam1.0": 0.3}, "tv_cosine": {"lam0.3": -0.001, "lam0.5": -0.042, "lam0.7": 0.152, "lam1.0": 0.271}}

Fixed-λ ρ (R0): {"O_A": {"lam0.3": 0.013, "lam0.5": 0.074, "lam0.7": 0.23, "lam1.0": 0.287}, "tv_cosine": {"lam0.3": 0.023, "lam0.5": 0.07, "lam0.7": 0.237, "lam1.0": 0.283}}

Fixed-λ ρ (S1): {"O_A": {"lam0.3": -0.042, "lam0.5": -0.042, "lam0.7": 0.115, "lam1.0": 0.242}, "tv_cosine": {"lam0.3": -0.018, "lam0.5": -0.09, "lam0.7": 0.093, "lam1.0": 0.275}}

### D stability / reliability across seed configurations (Spearman over the same unordered task pairs)

| quantity | P~R0 | P~S1 | R0~S1 |
|---|---:|---:|---:|
| D | 0.793 | 0.801 | 0.648 |
| O_A | 0.941 | 0.975 | 0.933 |
| tv_cosine | 0.748 | 0.781 | 0.898 |
| lam_selected | 0.544 | 0.571 | 0.509 |

### O_A vs random null; θ_min distribution; gate firings

Random rank-8 null: O_A mean 0.00911 (sd 0.00020, p95 0.00943).

| population | n | O_A mean | sd | median | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layer-θ_min p1 / p5 / median (°) | layers <30° / <45° | pairs where gate fires |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| R0 (shared seed 0) | 78 | 0.01906 | 0.00395 | 0.01819 | 0.01277 | 0.03225 | [18.7, 118.2] | 1.00 | 0.0069 | 35.9 | 62.4 / 68.7 / 76.4 | 0 / 2 of 5616 | 0 |
| S1 (shared seed 1) | 78 | 0.01981 | 0.00445 | 0.01879 | 0.01239 | 0.03396 | [16.8, 126.9] | 1.00 | 0.0071 | 37.2 | 61.1 / 68.4 / 76.2 | 0 / 8 of 5616 | 0 |
| P (mixed seed) | 78 | 0.01937 | 0.00393 | 0.01841 | 0.01242 | 0.03434 | [16.9, 128.8] | 1.00 | 0.0027 | 37.2 | 61.8 / 68.6 / 76.3 | 0 / 5 of 5616 | 0 |
| S2 (same task, mixed seed) | 13 | 0.05031 | 0.00997 | 0.04994 | 0.03688 | 0.07689 | [141.8, 346.1] | 1.00 | 0.0672 | 31.1 | 49.3 / 54.4 / 66.5 | 0 / 5 of 936 | 0 |

paired_P_minus_R0: n=78, mean diff 0.00031, P lower in 40% of task pairs, ratio of means 1.016, Wilcoxon p = 5.84e-03

paired_P_minus_S1: n=78, mean diff -0.00044, P lower in 62% of task pairs, ratio of means 0.978, Wilcoxon p = 5.94e-04

θ_min by layer type (all populations): attention.output.dense: min 48.4°, median 76.6°; attention.self.key: min 31.1°, median 73.4°; attention.self.query: min 46.7°, median 73.8°; attention.self.value: min 44.5°, median 76.1°; intermediate.dense: min 57.1°, median 83.5°; output.dense: min 44.0°, median 76.7°

### S2: same-task seed pairs (t@s0 × t@s1)

Gate (θ★ = 30°) fires in **0/13** same-task pairs; min θ_min over all S2 pairs/layers = 31.11°.

| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cola | 32.87 | encoder.layer.11.attention.self.key | 0 | 1 | 0.05049 | 211.3 | 0.0808 | 0.7 | -0.0138 | 1.0138 | 1.0138 | 1.0138 |
| sst2 | 49.86 | encoder.layer.0.attention.self.value | 0 | 0 | 0.04431 | 179.8 | 0.0570 | 0.7 | -0.0037 | 1.0037 | 1.0037 | 1.0037 |
| mrpc | 42.73 | encoder.layer.11.attention.self.key | 0 | 1 | 0.05430 | 230.8 | 0.0671 | 0.5 | -0.0014 | 1.0014 | 1.0014 | 1.0014 |
| stsb | 31.11 | encoder.layer.11.attention.self.key | 0 | 2 | 0.07689 | 346.1 | 0.0973 | 0.5 | -0.0007 | 1.0007 | 1.0007 | 1.0007 |
| mnli | 44.52 | encoder.layer.0.attention.self.value | 0 | 1 | 0.05152 | 216.6 | 0.0558 | 0.5 | -0.0040 | 1.0040 | 1.0040 | 1.0040 |
| qnli | 48.75 | encoder.layer.6.attention.self.value | 0 | 0 | 0.04038 | 159.7 | 0.0440 | 0.7 | 0.0022 | 0.9978 | 0.9978 | 0.9978 |
| wic | 45.13 | encoder.layer.11.attention.self.key | 0 | 0 | 0.04442 | 180.3 | 0.0507 | 0.5 | -0.0036 | 1.0036 | 1.0036 | 1.0036 |
| snli | 49.50 | encoder.layer.11.attention.self.key | 0 | 0 | 0.04856 | 201.5 | 0.0653 | 0.5 | -0.0016 | 1.0016 | 1.0016 | 1.0016 |
| scitail | 50.31 | encoder.layer.5.attention.self.query | 0 | 0 | 0.04994 | 208.5 | 0.0693 | 0.5 | 0.0004 | 0.9996 | 0.9996 | 0.9996 |
| ag_news | 61.85 | encoder.layer.11.attention.self.key | 0 | 0 | 0.03688 | 141.8 | 0.0532 | 0.5 | 0.0006 | 0.9994 | 0.9994 | 0.9994 |
| imdb | 49.66 | encoder.layer.11.attention.self.key | 0 | 0 | 0.05629 | 241.0 | 0.0810 | 0.5 | -0.0021 | 1.0021 | 1.0021 | 1.0021 |
| trec | 51.44 | encoder.layer.11.attention.self.key | 0 | 0 | 0.05641 | 241.6 | 0.0992 | 0.5 | 0.0010 | 0.9990 | 0.9990 | 0.9990 |
| yelp_polarity | 54.34 | encoder.layer.11.attention.output.dense | 0 | 0 | 0.04369 | 176.6 | 0.0533 | 0.7 | 0.0006 | 0.9994 | 0.9994 | 0.9994 |

E1b-grid gate@lam*TA − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/13/0

E1b-grid gate@own − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/13/0

**E3 code (amendment-A2 grids) on S2, 13 pairs.** Mean normalized score: TA 1.0025, PICO_TA 1.0024, GATE 1.0025, FORCEGATE 1.0005, TA4_score 1.0020, PICO_TA_c1_score 0.9987.

| contrast | mean (pp) | 95% CI over tasks (pp) | sign-flip p (2-sided) | W/T/L |
|---|---:|---|---:|---|
| GATE-TA | +0.000 | [+0.000, +0.000] | 1.0000 | 0/13/0 |
| FORCEGATE-TA | -0.202 | [-0.499, +0.026] | 0.1715 | 5/0/8 |
| PICO_TA-TA | -0.007 | [-0.123, +0.104] | 0.9260 | 6/1/6 |
| GATE-PICO_TA | +0.007 | [-0.104, +0.123] | 0.9260 | 6/1/6 |
| FORCEGATE-PICO_TA | -0.195 | [-0.440, -0.002] | 0.1034 | 5/0/8 |
| PICO_TA_c1_score-TA | -0.379 | [-0.655, -0.120] | 0.0143 | 2/0/11 |
| TA4_score-TA | -0.048 | [-0.274, +0.154] | 0.7101 | 5/5/3 |

Gate vs TA on S2 pairs with ≥ 1 FAIL layer: n = 0; per pair (pp): —.

Selected-config boundary rate: {'TA': 0.0, 'PICO_TA': 0.0, 'GATE': 0.0, 'FORCEGATE': 0.0}. Sanity (E3 TA at E1b λs = stage-2 values): max |Δ| = 0.00e+00.


### Method comparison (E1b stage-2 exploratory methods)

**P**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9745 | — | — | — | — |
| TIES-lite@lam*TA | 0.9562 | -0.0183 | [-0.0319, -0.0087] | 12/0/66 | 1.99e-12 |
| TIES-lite@own-lam | 0.9718 | -0.0027 | [-0.0055, +0.0004] | 23/0/55 | 0.000175 |
| gate30@lam*TA | 0.9745 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |
| gate30@own-lam | 0.9745 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |

**R0**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9734 | — | — | — | — |
| TIES-lite@lam*TA | 0.9562 | -0.0171 | [-0.0286, -0.0092] | 8/0/70 | 2.14e-12 |
| TIES-lite@own-lam | 0.9699 | -0.0035 | [-0.0056, -0.0012] | 19/1/58 | 4.2e-07 |
| gate30@lam*TA | 0.9734 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |
| gate30@own-lam | 0.9734 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |

**S1**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9765 | — | — | — | — |
| TIES-lite@lam*TA | 0.9581 | -0.0184 | [-0.0319, -0.0073] | 16/0/62 | 1.28e-11 |
| TIES-lite@own-lam | 0.9738 | -0.0027 | [-0.0062, +0.0005] | 29/0/49 | 0.00269 |
| gate30@lam*TA | 0.9765 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |
| gate30@own-lam | 0.9765 | +0.0000 | [+0.0000, +0.0000] | 0/78/0 | nan |

### Cross-backbone (exploratory): Spearman over common task pairs

- D: E1b R0 (BERT) ~ e4a R0: ρ = 0.531 (n = 78)
- O_A: E1b R0 (BERT) ~ e4a R0: ρ = 0.826 (n = 78)
- tv_cosine: E1b R0 (BERT) ~ e4a R0: ρ = 0.667 (n = 78)
- D: E1c P (BERT) ~ e4a P: ρ = 0.607 (n = 78)
- O_A: E1c P (BERT) ~ e4a P: ρ = 0.855 (n = 78)
- tv_cosine: E1c P (BERT) ~ e4a P: ρ = 0.359 (n = 78)

## Integrity (stage0, both seeds)

| task | seed | lr | steps | train loss (mean) | last logged loss | prior entropy | eval | (a) thr | (a) | (b) ref−5pp | (b) | (c) max diff | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|
| cola | 0 | 0.0005 | 2360 | 0.2188 | 0.0489 | 0.6046 | 0.8399 | 0.7913 | True | mcc 0.6082 ≥ 0.5860 | True | 1.8e-05 | **True** |
| sst2 | 0 | 0.0005 | 5625 | 0.1868 | 0.1101 | 0.6865 | 0.9495 | 0.6092 | True | accuracy 0.9495 ≥ 0.8980 | True | 8.1e-06 | **True** |
| mrpc | 0 | 0.0005 | 840 | 0.1998 | 0.0256 | 0.6267 | 0.8750 | 0.7838 | True | accuracy 0.8750 ≥ 0.8520 | True | 4.2e-05 | **True** |
| stsb | 0 | 0.0005 | 1490 | 0.5377 | 0.1359 | nan | 0.9082 | 0.7000 | True | spearman 0.9082 ≥ 0.8620 | True | 3.3e-06 | **True** |
| mnli | 0 | 0.0005 | 36723 | 0.4089 | 0.3081 | 1.0986 | 0.8713 | 0.4274 | True | accuracy 0.8713 ≥ 0.8260 | True | 1.5e-05 | **True** |
| qnli | 0 | 0.0005 | 5625 | 0.2778 | 0.1972 | 0.6931 | 0.9202 | 0.6054 | True | accuracy 0.9202 ≥ 0.8780 | True | 1.4e-05 | **True** |
| rte | 0 | 0.0005 | 470 | 0.3607 | 0.0601 | 0.6931 | 0.7004 | 0.5729 | True | accuracy 0.7004 ≥ 0.7370 | False | 4.5e-05 | **False** |
| wic | 0 | 0.0005 | 1390 | 0.3145 | 0.0634 | 0.6931 | 0.6724 | 0.6000 | True | n/a | None | 2.3e-05 | **True** |
| snli | 0 | 0.0005 | 5625 | 0.4083 | 0.2797 | 1.0986 | 0.8940 | 0.4331 | True | n/a | None | 1.3e-05 | **True** |
| scitail | 0 | 0.0005 | 2073 | 0.1779 | 0.0783 | 0.6571 | 0.9525 | 0.5962 | True | n/a | None | 5.7e-05 | **True** |
| ag_news | 0 | 0.0005 | 5625 | 0.2038 | 0.1407 | 1.3863 | 0.9462 | 0.3500 | True | n/a | None | 5.0e-06 | **True** |
| imdb | 0 | 0.0005 | 2250 | 0.2395 | 0.1707 | 0.6931 | 0.9148 | 0.5977 | True | n/a | None | 7.1e-06 | **True** |
| trec | 0 | 0.0005 | 1400 | 0.1682 | 0.0067 | 1.6528 | 0.9700 | 0.2880 | True | n/a | None | 3.1e-05 | **True** |
| yelp_polarity | 0 | 0.0005 | 5625 | 0.1299 | 0.0769 | 0.6931 | 0.9560 | 0.6108 | True | n/a | None | 1.1e-05 | **True** |
| cola | 1 | 0.0005 | 2360 | 0.2149 | 0.0610 | 0.6046 | 0.8341 | 0.7913 | True | mcc 0.5936 ≥ 0.5860 | True | 3.0e-05 | **True** |
| sst2 | 1 | 0.0005 | 5625 | 0.1861 | 0.1122 | 0.6865 | 0.9381 | 0.6092 | True | accuracy 0.9381 ≥ 0.8980 | True | 4.4e-06 | **True** |
| mrpc | 1 | 0.0005 | 840 | 0.2102 | 0.0217 | 0.6267 | 0.8799 | 0.7838 | True | accuracy 0.8799 ≥ 0.8520 | True | 5.4e-05 | **True** |
| stsb | 1 | 0.0005 | 1490 | 0.5302 | 0.1400 | nan | 0.9036 | 0.7000 | True | spearman 0.9036 ≥ 0.8620 | True | 2.4e-06 | **True** |
| mnli | 1 | 0.0005 | 36723 | 0.4094 | 0.3302 | 1.0986 | 0.8725 | 0.4274 | True | accuracy 0.8725 ≥ 0.8260 | True | 7.6e-06 | **True** |
| qnli | 1 | 0.0005 | 5625 | 0.2813 | 0.1968 | 0.6931 | 0.9193 | 0.6054 | True | accuracy 0.9193 ≥ 0.8780 | True | 6.4e-06 | **True** |
| rte | 1 | 0.0005 | 470 | 0.4024 | 0.1087 | 0.6931 | 0.7040 | 0.5729 | True | accuracy 0.7040 ≥ 0.7370 | False | 3.1e-05 | **False** |
| wic | 1 | 0.0005 | 1390 | 0.2956 | 0.0408 | 0.6931 | 0.6661 | 0.6000 | True | n/a | None | 3.0e-05 | **True** |
| snli | 1 | 0.0005 | 5625 | 0.4064 | 0.3038 | 1.0986 | 0.8942 | 0.4331 | True | n/a | None | 1.1e-05 | **True** |
| scitail | 1 | 0.0005 | 2073 | 0.1840 | 0.0707 | 0.6571 | 0.9548 | 0.5962 | True | n/a | None | 6.3e-06 | **True** |
| ag_news | 1 | 0.0005 | 5625 | 0.2024 | 0.1163 | 1.3863 | 0.9461 | 0.3500 | True | n/a | None | 5.7e-06 | **True** |
| imdb | 1 | 0.0005 | 2250 | 0.2416 | 0.1657 | 0.6931 | 0.9127 | 0.5977 | True | n/a | None | 6.3e-06 | **True** |
| trec | 1 | 0.0005 | 1400 | 0.1672 | 0.0074 | 1.6528 | 0.9680 | 0.2880 | True | n/a | None | 1.6e-05 | **True** |
| yelp_polarity | 1 | 0.0005 | 5625 | 0.1317 | 0.0781 | 0.6931 | 0.9562 | 0.6108 | True | n/a | None | 3.6e-06 | **True** |

## Deviations and notes

# E4a — deviations and notes (append-only; KST)

**No deviation from PREREG_E4A.md / prereg_e4a.json affected training, integrity, predictors, merges or the analysis.** Notes:

- **N1 (2026-09-26 09:54, pilot outcome; pre-registered rule, not a deviation).** The pilot ran 09:44:18–09:54:11.
  - lr 1e-3: sst2 held-out 0.953 (OK); rte **diverged** (held-out 0.512 vs majority 0.488; tail loss 0.695 ≈ ln 2 = prior entropy; the model never learned).
  - lr 5e-4: sst2 0.958, rte 0.742, neither diverged.
  - → **lr_main = 5e-4** for all tasks and both seeds (`pilot/pilot_decision.json`, sha256 in `pilot/pilot_decision.sha256`). lr 2e-4 was not piloted, as the ladder requires.
- **N2 (09:55, code; before any main training and before code hashing).** The throwaway 3-task smoke test (scratch copy outside `artifacts/`, 20-step adapters) found a crash in
  `e4a_analysis.method_comp`: with 3 tasks every task-block bootstrap resample is degenerate, so `boots == []`. The test `boots is not None` became `if boots`, so the pair
  bootstrap is used only when there are no task-block resamples. With 13 tasks, 2,000/2,000 task-block resamples were used in P, R0 and S1 (0 skipped), so the fix has no effect on the
  reported numbers. Code was hashed afterwards (`code_e4a.sha256`, `code_analysis_e4a.sha256`, 09:56). The smoke copy was deleted.
- **N3 (11:39, integrity; pre-registered rule and pre-registered risk).** **RTE failed rule (b) in both seeds**: eval accuracy 0.7004 (s0) and 0.7040 (s1), below the threshold 0.737
  (fairseq roberta.base 78.7 − 5 pp). (a) and (c) passed. As §2 requires, RTE is removed from every population. **K = 13 valid tasks** (≥ 10, no STOP);
  n = P 78, R0 78, S1 78, S2 13. CoLA (MCC), MRPC, SST-2, STS-B, MNLI and QNLI passed (b) in both seeds. No training collapse: every adapter's final logged loss was far below the prior entropy.
- **N4 (15:00, merges).** The two S1 processes (forward and reverse) met in the middle, and both computed `snli@s1__imdb@s1`. The two records are bitwise-identical
  (D = 0.015116, λ* = 0.7). The pre-written de-duplication in `load_pop` keeps the first one. S1 has 78 unique pairs.
- **N5 (compute).** Pilot start 09:44:18 → training 09:56:25–11:36:55 → stage0/1 → merges end 15:01:14 → stage3 end 15:02:09.
  **5.28 h wall-clock** (cap 10 h). Summed process time: training 4.82 h, pilot 0.15 h, stage-2 merges 6.11 h, E3 subset 0.53 h (`gpu_time_summary.json`).
- **N6 (timing of the E4b pre-registration).** `PREREG_E4B.md` / `prereg_e4b.json` were hashed at 10:02, while E4a was training and before any E4a integrity or merge result existed.


## Compute

Timing files (s): {"timing_R0": {"stage2_gpu_wall_s": 6873.5}, "timing_S1": {"stage2_gpu_wall_s": 7045.7}, "timing_P": {"stage2_gpu_wall_s": 6916.6}, "timing_main": {"stage0_s": 184.7, "stage1_s": 76.5}, "timing_S2": {"stage2_gpu_wall_s": 1158.4, "e3same_s": 1909.7}}

GPU wall-clock summary: {
 "pilot_start_kst": "2026-09-26 09:44:18",
 "training_start_kst": "2026-09-26 09:56:25",
 "training_end_kst": "2026-09-26 11:36:55",
 "merges_end_kst": "2026-09-26 15:01:14",
 "stage3_end_kst": "2026-09-26 15:02:09",
 "wall_clock_h_pilot_to_last_gpu_job": 5.282222222222222,
 "wall_clock_h_incl_stage3": 5.2975,
 "cap_h": 10,
 "training_process_h": 4.818443792329894,
 "pilot_process_h": 0.1474621464146508,
 "stage2_process_h": 6.109503570993741,
 "e3same_h": 0.5304812229341931,
 "stage0_h": 0.05131815380520291,
 "stage1_h": 0.02123980548646715,
 "train_wall_s_by_adapter": {
  "stsb@s1": 136.45516562461853,
  "trec@s1": 29.325154542922974,
  "cola@s1": 129.35895681381226,
  "rte@s1": 64.61135339736938,
  "yelp_polarity@s1": 899.1875894069672,
  "mnli@s1": 4403.571870088577,
  "ag_news@s1": 789.9792568683624,
  "mrpc@s1": 96.69516468048096,
  "wic@s1": 66.99471521377563,
  "qnli@s1": 768.5260903835297,
  "imdb@s1": 276.1829295158386,
  "scitail@s1": 76.84716129302979,
  "sst2@s1": 373.8384110927582,
  "snli@s1": 401.3454804420471,
  "stsb@s0": 150.8938181400299,
  "trec@s0": 79.92918062210083,
  "cola@s0": 81.78892588615417,
  "rte@s0": 77.09247279167175,
  "yelp_polarity@s0": 891.925493478775,
  "mnli@s0": 4404.103250026703,
  "ag_news@s0": 784.5168504714966,
  "mrpc@s0": 94.09205341339111,
  "wic@s0": 96.46154427528381,
  "qnli@s0": 756.1491410732269,
  "imdb@s0": 370.6653673648834,
  "scitail@s0": 218.35877513885498,
  "sst2@s0": 392.35151386260986,
  "snli@s0": 435.1499664783478
 },
 "pilot_wall_s": {
  "rte@lr0.001": 53.996750831604004,
  "sst2@lr0.001": 149.45166778564453,
  "rte@lr0.0005": 56.9896674156189,
  "sst2@lr0.0005": 270.4256410598755
 }
}
