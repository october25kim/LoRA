# E6 VERDICT (second decoder: Qwen2.5-1.5B; replication of E4b) — H1 (O_A): **INCONCLUSIVE**; H2 (task-vector cosine): **INCONCLUSIVE**  (primary population P, mixed seed)

Generated 2026-09-29 01:45:36 KST. Pre-registration: `PREREG_E6.md` / `prereg_e6.json` (sha256 in `prereg_e6.sha256`); deviations/notes: `DEVIATIONS_E6.md`. **Disclosure: the E6 pre-registration was written after the E4b and E5 results were known** (PREREG_E6.md sec. 0); every analysis choice is frozen from E4b.

Valid tasks K = 14: cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity. Exclusions: none. Predictor hash ok: True. Analysis-code equivalence on E1b (reproduces E1b analysis.json): True.

Pilot projection (recorded only; seeds fixed by PREREG_E6): train 5.95 h/seed, core (2 seeds) 18.53 h -> seeds [0, 1], primary P.

Pilot (pre-registered, held-out only): lr_main = 0.0003 (non-diverged [0.0001, 0.0003, 0.001]; mean held-out {'0.0001': 0.9059999999999999, '0.0003': 0.9139999999999999, '0.001': 0.8465}). lr 0.0001: sst2 hold 0.9600 (maj 0.560, tail loss 0.130, diverged False), rte hold 0.8520 (maj 0.488, tail loss 0.000, diverged False); lr 0.0003: sst2 hold 0.9640 (maj 0.560, tail loss 0.128, diverged False), rte hold 0.8640 (maj 0.488, tail loss 0.000, diverged False); lr 0.001: sst2 hold 0.8590 (maj 0.560, tail loss 0.355, diverged False), rte hold 0.8340 (maj 0.488, tail loss 0.003, diverged False)

## Primary: mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (P) | O_A | 0.1226 | [-0.294, 0.465] | 0.2024 | 0.2824 | 0.07 | **INCONCLUSIVE** |
| H2 (P) | tv_cosine | 0.1504 | [-0.279, 0.467] | 0.1412 | 0.2824 | 0.07 | **INCONCLUSIVE** |

Rule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).

H1 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.12, sst2 0.19, mrpc 0.11, stsb 0.08, mnli 0.13, qnli 0.09, rte 0.21, wic 0.19, snli 0.10, scitail 0.13, ag_news 0.07, imdb 0.14, trec -0.01, yelp_polarity 0.14

H2 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.12, sst2 0.19, mrpc 0.15, stsb 0.07, mnli 0.16, qnli 0.12, rte 0.28, wic 0.15, snli 0.17, scitail 0.17, ag_news 0.12, imdb 0.17, trec 0.03, yelp_polarity 0.17

P: 91 pairs; D mean 0.0105, median 0.0068, range [-0.0099, 0.0658]; λ* counts {'0.5': 8, '0.7': 44, '1.0': 39}; ρ(O_A, tv_cos) = 0.852; min θ_min over pairs/layers = 31.7°; gate active in 0 pairs.

## Secondary (exploratory; no multiplicity correction across populations)

### R0: seed-0 cross-task pairs (rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (R0) | O_A | 0.3305 | [-0.085, 0.574] | 0.0297 | 0.0594 | 1.00 | **INCONCLUSIVE** |
| H2 (R0) | tv_cosine | 0.2153 | [-0.279, 0.579] | 0.1288 | 0.1288 | 0.57 | **INCONCLUSIVE** |

R0: 91 pairs; D mean 0.0324, median 0.0169, range [-0.0221, 0.3342]; λ* counts {'0.3': 2, '0.5': 11, '0.7': 51, '1.0': 27}; ρ(O_A, tv_cos) = 0.869; min θ_min over pairs/layers = 23.0°; gate active in 6 pairs.

### S1: seed-1 cross-task pairs (rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (S1) | O_A | 0.4280 | [-0.010, 0.682] | 0.0078 | 0.0156 | 1.00 | **PASS** |
| H2 (S1) | tv_cosine | 0.3717 | [-0.090, 0.689] | 0.0194 | 0.0194 | 0.93 | **INCONCLUSIVE** |

S1: 91 pairs; D mean 0.0238, median 0.0143, range [-0.0109, 0.2402]; λ* counts {'0.3': 1, '0.5': 4, '0.7': 46, '1.0': 40}; ρ(O_A, tv_cos) = 0.833; min θ_min over pairs/layers = 25.0°; gate active in 4 pairs.

### Pooled 3-population summary (mean D and predictors over R0, S1, P per task pair; descriptive)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (pooled) | O_A | 0.4053 | [-0.045, 0.647] | 0.0076 | 0.0152 | 1.00 | **PASS** |
| H2 (pooled) | tv_cosine | 0.3213 | [-0.168, 0.656] | 0.0396 | 0.0396 | 0.93 | **INCONCLUSIVE** |

Pooled D mean 0.0222, n = 91 task pairs.

### Secondary predictors

| predictor | ρ P | CI P | ρ R0 | CI R0 | ρ S1 | CI S1 |
|---|---:|---|---:|---|---:|---|
| O_A | 0.123 | [-0.29, 0.47] | 0.330 | [-0.09, 0.57] | 0.428 | [-0.01, 0.68] |
| tv_cosine | 0.150 | [-0.28, 0.47] | 0.215 | [-0.28, 0.58] | 0.372 | [-0.09, 0.69] |
| mean_theta_min_A_deg | -0.121 | [-0.45, 0.31] | -0.323 | [-0.60, 0.12] | -0.426 | [-0.69, 0.01] |
| min_theta_min_A_deg | -0.325 | [-0.63, 0.04] | -0.115 | [-0.43, 0.29] | -0.154 | [-0.45, 0.28] |
| n_layers_theta_min_lt30_A | nan | [nan, nan] | 0.020 | [-0.28, 0.30] | -0.077 | [-0.30, 0.07] |
| sign_conflict_top20 | -0.169 | [-0.52, 0.28] | -0.239 | [-0.60, 0.27] | -0.392 | [-0.69, 0.07] |
| norm_ratio | -0.014 | [-0.35, 0.32] | 0.178 | [-0.27, 0.55] | 0.177 | [-0.17, 0.49] |
| null_z_O_A | 0.123 | [-0.29, 0.47] | 0.330 | [-0.09, 0.57] | 0.428 | [-0.01, 0.68] |
| sign_conflict_all | -0.146 | [-0.51, 0.31] | -0.208 | [-0.53, 0.22] | -0.291 | [-0.64, 0.17] |

Fixed-λ ρ (P): {"O_A": {"lam0.3": 0.16, "lam0.5": 0.053, "lam0.7": 0.08, "lam1.0": 0.203}, "tv_cosine": {"lam0.3": 0.105, "lam0.5": 0.044, "lam0.7": 0.066, "lam1.0": 0.272}}

Fixed-λ ρ (R0): {"O_A": {"lam0.3": 0.195, "lam0.5": 0.281, "lam0.7": 0.325, "lam1.0": 0.414}, "tv_cosine": {"lam0.3": 0.112, "lam0.5": 0.162, "lam0.7": 0.204, "lam1.0": 0.297}}

Fixed-λ ρ (S1): {"O_A": {"lam0.3": 0.298, "lam0.5": 0.317, "lam0.7": 0.398, "lam1.0": 0.493}, "tv_cosine": {"lam0.3": 0.305, "lam0.5": 0.276, "lam0.7": 0.325, "lam1.0": 0.417}}

### D stability / reliability across seed configurations (Spearman over the same unordered task pairs)

| quantity | P~R0 | P~S1 | R0~S1 |
|---|---:|---:|---:|
| D | 0.511 | 0.530 | 0.792 |
| O_A | 0.899 | 0.877 | 0.976 |
| tv_cosine | 0.859 | 0.853 | 0.967 |
| lam_selected | 0.463 | 0.261 | 0.300 |

### O_A vs random null; θ_min distribution; gate firings

Random rank-8 null: O_A mean 0.01142 (sd 0.00021, p95 0.01176).

| population | n | O_A mean | sd | median | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layer-θ_min p1 / p5 / median (°) | layers <30° / <45° | pairs where gate fires |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| R0 (shared seed 0) | 91 | 0.03574 | 0.00656 | 0.03449 | 0.02360 | 0.06599 | [59.2, 265.5] | 1.00 | 0.0078 | 23.0 | 40.9 / 50.5 / 73.8 | 10 / 387 of 17836 | 6 |
| S1 (shared seed 1) | 91 | 0.03539 | 0.00607 | 0.03480 | 0.02388 | 0.06063 | [60.6, 239.4] | 1.00 | 0.0075 | 25.0 | 41.8 / 51.5 / 73.6 | 5 / 293 of 17836 | 4 |
| P (mixed seed) | 91 | 0.02922 | 0.00394 | 0.02857 | 0.02185 | 0.04084 | [50.7, 143.1] | 1.00 | 0.0009 | 31.7 | 47.0 / 54.3 / 74.7 | 0 / 106 of 17836 | 0 |
| S2 (same task, mixed seed) | 14 | 0.05390 | 0.00693 | 0.05141 | 0.04605 | 0.07079 | [168.5, 288.9] | 1.00 | 0.0306 | 31.5 | 39.7 / 46.6 / 65.9 | 0 / 96 of 2744 | 0 |

paired_P_minus_R0: n=91, mean diff -0.00652, P lower in 100% of task pairs, ratio of means 0.818, Wilcoxon p = 1.19e-16

paired_P_minus_S1: n=91, mean diff -0.00617, P lower in 100% of task pairs, ratio of means 0.826, Wilcoxon p = 1.19e-16

θ_min by layer type (all populations): mlp.down_proj: min 23.9°, median 77.6°; mlp.gate_proj: min 57.3°, median 86.5°; mlp.up_proj: min 45.6°, median 83.4°; self_attn.k_proj: min 31.3°, median 59.0°; self_attn.o_proj: min 25.9°, median 78.0°; self_attn.q_proj: min 47.3°, median 72.2°; self_attn.v_proj: min 23.0°, median 68.9°

### S2: same-task seed pairs (t@s0 × t@s1)

Gate (θ★ = 30°) fires in **0/14** same-task pairs; min θ_min over all S2 pairs/layers = 31.51°.

| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cola | 31.51 | model.layers.17.self_attn.k_proj | 0 | 3 | 0.05126 | 193.8 | 0.0291 | 0.7 | -0.0017 | 1.0017 | 1.0017 | 1.0017 |
| sst2 | 32.82 | model.layers.25.self_attn.k_proj | 0 | 6 | 0.06329 | 252.4 | 0.0249 | 0.5 | -0.0066 | 1.0066 | 1.0066 | 1.0066 |
| mrpc | 37.11 | model.layers.26.self_attn.k_proj | 0 | 4 | 0.05156 | 195.3 | 0.0313 | 0.5 | -0.0111 | 1.0111 | 1.0111 | 1.0111 |
| stsb | 32.48 | model.layers.27.self_attn.k_proj | 0 | 14 | 0.07079 | 288.9 | 0.0563 | 0.7 | -0.0028 | 1.0028 | 1.0028 | 1.0028 |
| mnli | 37.74 | model.layers.27.mlp.down_proj | 0 | 8 | 0.05501 | 212.1 | 0.0334 | 0.7 | -0.0008 | 1.0008 | 1.0008 | 1.0008 |
| qnli | 38.46 | model.layers.27.self_attn.v_proj | 0 | 9 | 0.05491 | 211.6 | 0.0251 | 0.5 | 0.0074 | 0.9926 | 0.9926 | 0.9926 |
| rte | 32.66 | model.layers.18.self_attn.k_proj | 0 | 8 | 0.05033 | 189.3 | 0.0300 | 0.7 | -0.0240 | 1.0240 | 1.0240 | 1.0240 |
| wic | 31.88 | model.layers.26.self_attn.k_proj | 0 | 4 | 0.04801 | 178.0 | 0.0325 | 0.5 | -0.0044 | 1.0044 | 1.0044 | 1.0044 |
| snli | 34.66 | model.layers.23.self_attn.v_proj | 0 | 12 | 0.05805 | 226.9 | 0.0259 | 0.7 | 0.0005 | 0.9995 | 0.9995 | 0.9995 |
| scitail | 37.45 | model.layers.19.self_attn.v_proj | 0 | 8 | 0.05890 | 231.0 | 0.0353 | 0.5 | 0.0004 | 0.9996 | 0.9996 | 0.9996 |
| ag_news | 39.23 | model.layers.27.mlp.down_proj | 0 | 1 | 0.04605 | 168.5 | 0.0288 | 1.0 | 0.0043 | 0.9957 | 0.9957 | 0.9957 |
| imdb | 35.30 | model.layers.17.self_attn.v_proj | 0 | 8 | 0.05116 | 193.4 | 0.0256 | 0.5 | 0.0018 | 0.9982 | 0.9982 | 0.9982 |
| trec | 42.99 | model.layers.20.self_attn.k_proj | 0 | 4 | 0.04743 | 175.2 | 0.0262 | 0.7 | 0.0021 | 0.9979 | 0.9979 | 0.9979 |
| yelp_polarity | 39.29 | model.layers.19.self_attn.v_proj | 0 | 7 | 0.04785 | 177.2 | 0.0245 | 0.7 | -0.0009 | 1.0009 | 1.0009 | 1.0009 |

E1b-grid gate@lam*TA − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/14/0

E1b-grid gate@own − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/14/0

### Method comparison (E1b stage-2 exploratory methods)

**P**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9895 | — | — | — | — |
| TIES-lite@lam*TA | 0.9792 | -0.0104 | [-0.0195, -0.0040] | 14/0/77 | 2.59e-13 |
| TIES-lite@own-lam | 0.9871 | -0.0024 | [-0.0051, -0.0000] | 28/1/62 | 0.000253 |
| gate30@lam*TA | 0.9895 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |
| gate30@own-lam | 0.9895 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |

**R0**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9676 | — | — | — | — |
| TIES-lite@lam*TA | 0.9523 | -0.0153 | [-0.0347, -0.0042] | 18/0/73 | 1.23e-10 |
| TIES-lite@own-lam | 0.9616 | -0.0060 | [-0.0219, +0.0020] | 37/0/54 | 0.0925 |
| gate30@lam*TA | 0.9676 | -0.0000 | [-0.0001, +0.0000] | 1/86/4 | 0.312 |
| gate30@own-lam | 0.9676 | -0.0000 | [-0.0001, +0.0000] | 1/86/4 | 0.312 |

**S1**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9762 | — | — | — | — |
| TIES-lite@lam*TA | 0.9653 | -0.0109 | [-0.0244, -0.0028] | 19/0/72 | 2.96e-10 |
| TIES-lite@own-lam | 0.9708 | -0.0055 | [-0.0160, +0.0007] | 34/0/57 | 0.000885 |
| gate30@lam*TA | 0.9762 | -0.0000 | [-0.0000, +0.0000] | 1/89/1 | nan |
| gate30@own-lam | 0.9762 | -0.0000 | [-0.0000, +0.0000] | 1/89/1 | nan |

### Cross-backbone (exploratory): Spearman over common task pairs

- D: E1b R0 (BERT) ~ e6 R0: ρ = 0.327 (n = 91)
- O_A: E1b R0 (BERT) ~ e6 R0: ρ = 0.273 (n = 91)
- tv_cosine: E1b R0 (BERT) ~ e6 R0: ρ = 0.619 (n = 91)
- D: E1c P (BERT) ~ e6 P: ρ = 0.399 (n = 91)
- O_A: E1c P (BERT) ~ e6 P: ρ = 0.212 (n = 91)
- tv_cosine: E1c P (BERT) ~ e6 P: ρ = 0.574 (n = 91)
- D: E4a R0 (RoBERTa) ~ e6 R0: ρ = 0.537 (n = 78)
- O_A: E4a R0 (RoBERTa) ~ e6 R0: ρ = 0.436 (n = 78)
- tv_cosine: E4a R0 (RoBERTa) ~ e6 R0: ρ = 0.463 (n = 78)
- D: E4a P (RoBERTa) ~ e6 P: ρ = 0.239 (n = 78)
- O_A: E4a P (RoBERTa) ~ e6 P: ρ = 0.341 (n = 78)
- tv_cosine: E4a P (RoBERTa) ~ e6 P: ρ = 0.454 (n = 78)
- D: E4b P (Qwen2.5-0.5B) ~ e6 P: ρ = 0.503 (n = 91)
- O_A: E4b P (Qwen2.5-0.5B) ~ e6 P: ρ = 0.899 (n = 91)
- tv_cosine: E4b P (Qwen2.5-0.5B) ~ e6 P: ρ = 0.837 (n = 91)
- lam_selected: E4b P (Qwen2.5-0.5B) ~ e6 P: ρ = 0.254 (n = 91)
- D: E4b R0 (Qwen2.5-0.5B) ~ e6 R0: ρ = 0.633 (n = 91)
- O_A: E4b R0 (Qwen2.5-0.5B) ~ e6 R0: ρ = 0.866 (n = 91)
- tv_cosine: E4b R0 (Qwen2.5-0.5B) ~ e6 R0: ρ = 0.842 (n = 91)
- lam_selected: E4b R0 (Qwen2.5-0.5B) ~ e6 R0: ρ = 0.092 (n = 91)
- D: E4b S1 (Qwen2.5-0.5B) ~ e6 S1: ρ = 0.730 (n = 91)
- O_A: E4b S1 (Qwen2.5-0.5B) ~ e6 S1: ρ = 0.925 (n = 91)
- tv_cosine: E4b S1 (Qwen2.5-0.5B) ~ e6 S1: ρ = 0.956 (n = 91)
- lam_selected: E4b S1 (Qwen2.5-0.5B) ~ e6 S1: ρ = 0.452 (n = 91)

## Replication vs E4b (Qwen2.5-0.5B) and pooled decoder estimate

| population | predictor | ρ E4b (0.5B) | verdict E4b | ρ E6 (1.5B) | verdict E6 | ρ E6 − ρ E4b [joint task-block 95% CI] | pooled decoder ρ (mean) [joint CI] | joint perm p (1-sided) |
|---|---|---:|---|---:|---|---|---|---:|
| P (primary) | O_A | 0.345 | INCONCLUSIVE | 0.123 | INCONCLUSIVE | -0.223 [-0.488, +0.075] | 0.234 [-0.160, 0.517] | 0.0606 |
| P (primary) | tv_cosine | 0.368 | INCONCLUSIVE | 0.150 | INCONCLUSIVE | -0.218 [-0.556, +0.107] | 0.259 [-0.145, 0.537] | 0.0299 |
| R0 | O_A | 0.366 | INCONCLUSIVE | 0.330 | INCONCLUSIVE | -0.036 [-0.335, +0.254] | 0.348 [-0.058, 0.580] | 0.0127 |
| R0 | tv_cosine | 0.395 | INCONCLUSIVE | 0.215 | INCONCLUSIVE | -0.180 [-0.450, +0.136] | 0.305 [-0.161, 0.595] | 0.0306 |
| S1 | O_A | 0.211 | INCONCLUSIVE | 0.428 | PASS | +0.217 [-0.122, +0.459] | 0.320 [-0.103, 0.586] | 0.0343 |
| S1 | tv_cosine | 0.105 | INCONCLUSIVE | 0.372 | INCONCLUSIVE | +0.267 [+0.000, +0.467] | 0.238 [-0.211, 0.575] | 0.1002 |

Common tasks: 14. The pooled decoder estimate is **exploratory** (not pre-registered as a test in E4b; pre-specified in PREREG_E6 sec. 6b; no decision rule).

## Where the sub-30° overlap sits (θ_min < 30° pair-layers)

| model | population | pair-layers | # < 30° | pairs with ≥1 | share layer-0 k_proj | share layer 0 | share last layer | share k_proj | share v_proj | pairs whose only <30° layer is L0 k_proj | top locations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B (E6) | P | 17836 | 0 | 0/91 | nan | nan | nan | nan | nan | nan |  |
| Qwen2.5-1.5B (E6) | R0 | 17836 | 10 | 6/91 | 0.00 | 0.00 | 0.90 | 0.00 | 0.60 | 0.00 | L27.self_attn.v_proj:5, L27.self_attn.o_proj:3, L27.mlp.down_proj:1, L23.self_attn.v_proj:1 |
| Qwen2.5-1.5B (E6) | S1 | 17836 | 5 | 4/91 | 0.00 | 0.00 | 1.00 | 0.00 | 0.80 | 0.00 | L27.self_attn.v_proj:4, L27.self_attn.o_proj:1 |
| Qwen2.5-1.5B (E6) | S2 | 2744 | 0 | 0/14 | nan | nan | nan | nan | nan | nan |  |
| Qwen2.5-0.5B (E4b) | P | 15288 | 72 | 66/91 | 0.92 | 0.92 | 0.04 | 1.00 | 0.00 | 0.91 | L0.self_attn.k_proj:66, L23.self_attn.k_proj:3, L19.self_attn.k_proj:3 |
| Qwen2.5-0.5B (E4b) | R0 | 15288 | 138 | 78/91 | 0.48 | 0.48 | 0.43 | 0.52 | 0.34 | 0.35 | L0.self_attn.k_proj:66, L23.self_attn.v_proj:40, L23.mlp.down_proj:14, L23.self_attn.o_proj:3, L22.self_attn.v_proj:3 |
| Qwen2.5-0.5B (E4b) | S1 | 15288 | 126 | 78/91 | 0.52 | 0.52 | 0.25 | 0.61 | 0.29 | 0.59 | L0.self_attn.k_proj:66, L23.self_attn.v_proj:19, L22.self_attn.v_proj:7, L21.self_attn.v_proj:6, L23.self_attn.o_proj:5 |
| Qwen2.5-0.5B (E4b) | S2 | 2352 | 15 | 13/14 | 0.73 | 0.73 | 0.00 | 1.00 | 0.00 | 0.69 | L0.self_attn.k_proj:11, L19.self_attn.k_proj:2, L21.self_attn.k_proj:1, L20.self_attn.k_proj:1 |

E6 P: <30° by module {}; by layer index {}; relative-depth quartile counts {}; min θ_min by module mlp.down_proj 38.5°, mlp.gate_proj 77.3°, mlp.up_proj 63.3°, self_attn.k_proj 36.2°, self_attn.o_proj 43.9°, self_attn.q_proj 52.0°, self_attn.v_proj 31.7°

E6 R0: <30° by module {'self_attn.v_proj': 6, 'self_attn.o_proj': 3, 'mlp.down_proj': 1}; by layer index {'23': 1, '27': 9}; relative-depth quartile counts {'3': 10}; min θ_min by module mlp.down_proj 23.9°, mlp.gate_proj 57.3°, mlp.up_proj 45.6°, self_attn.k_proj 31.3°, self_attn.o_proj 25.9°, self_attn.q_proj 47.7°, self_attn.v_proj 23.0°

E6 S1: <30° by module {'self_attn.v_proj': 4, 'self_attn.o_proj': 1}; by layer index {'27': 5}; relative-depth quartile counts {'3': 5}; min θ_min by module mlp.down_proj 30.8°, mlp.gate_proj 62.9°, mlp.up_proj 47.9°, self_attn.k_proj 34.3°, self_attn.o_proj 29.5°, self_attn.q_proj 50.4°, self_attn.v_proj 25.0°

E6 S2: <30° by module {}; by layer index {}; relative-depth quartile counts {}; min θ_min by module mlp.down_proj 37.7°, mlp.gate_proj 59.9°, mlp.up_proj 53.8°, self_attn.k_proj 31.5°, self_attn.o_proj 41.1°, self_attn.q_proj 47.3°, self_attn.v_proj 34.7°

## Gate (θ★ = 30°): firing and benefit on gate-active pairs

| population | gate-active pairs | fail layers total | GATE@λ*TA − TA mean / median (pp) [task-block CI] | Wilcoxon p | W/T/L | GATE@own − TA mean (pp) [CI] | Wilcoxon p | W/T/L |
|---|---:|---:|---|---:|---|---|---:|---|
| P | 0/91 | 0 | — | — | — | — | — | — |
| R0 | 6/91 | 10 | -0.042 / -0.017 [-0.084, -0.007] | 0.312 | 1/1/4 | -0.042 [-0.084, -0.007] | 0.312 | 1/1/4 |
| S1 | 4/91 | 5 | — | — | — | — | — | — |

## Confirmatory secondary hypotheses: λ rule U1 and merge decision M3 (frozen from e6_lambda; PREREG_E6.md sec. 6c)

Primary scope qwen15/all (R0 + S1 + P cross-task pairs), grid G4, tau 0.05; populations used ['P', 'R0', 'S1']; 273 pairs; recompute check max |diff| = 0.0e+00. Regret = 100·(D_test(λ_rule) − D_test(λ_sel)) pp on evaluation examples outside the unlabeled calibration sets; task-block bootstrap 95% CIs (2,000).

| hypothesis | rule | outcome |
|---|---|---|
| H-U1 | task-block 95% CI upper < 0 for BOTH paired differences regret(U1)-regret(lam=0.7) and regret(U1)-regret(lam=1.0) | **SUPPORTED** |
| H-U1b | U1 removes >= 80% of the avoidable loss of lam=1.0: 1 - mean regret(U1)/mean regret(lam=1) >= 0.80 (point estimate; CI reported); NOT APPLICABLE if mean regret(lam=1) <= 0 (no avoidable loss to remove) | **SUPPORTED** |
| H-M3 | AUROC of Dhat_U for y = 1[D_test(lam_sel) > 0.05] >= 0.75 AND task-block CI lower bound > 0.5 | **SUPPORTED** |
| H-M3acc | accuracy of the frozen-threshold M3 decision minus always-merge accuracy: task-block CI lower bound > 0 | **NOT SUPPORTED** |

qwen15/all G4: regret (pp) U1 -0.011 [-0.142, 0.103]; λ=1 0.487 [0.104, 0.975]; λ=0.7 0.213 [0.014, 0.496]; λ=0.5 1.378 [0.770, 2.136]; L-cal (labeled ref.) 0.048 [-0.086, 0.204]; eval-optimal floor -0.223 [-0.360, -0.105]. U1−λ0.7 -0.224 [-0.478, -0.035]; U1−λ1 -0.498 [-1.017, -0.073]; U1 removes 1.022 [0.572, 1.379] of λ=1 loss, 1.050 [0.207, 2.788] of λ=0.7 loss; U1 λ counts {'0.3': 2, '0.5': 31, '0.7': 133, '1.0': 107} (λ_sel {'0.3': 3, '0.5': 23, '0.7': 141, '1.0': 106}); U1 = λ_sel in 0.60.

qwen15/all G7: regret (pp) U1 -0.024 [-0.181, 0.099]; λ=1 0.486 [0.102, 0.974]; λ=0.7 0.212 [0.014, 0.492]; λ=0.5 1.376 [0.771, 2.125]; L-cal (labeled ref.) 0.114 [-0.079, 0.328]; eval-optimal floor -0.269 [-0.426, -0.133]. U1−λ0.7 -0.236 [-0.540, -0.029]; U1−λ1 -0.510 [-1.038, -0.078]; U1 removes 1.050 [0.615, 1.504] of λ=1 loss, 1.114 [0.079, 2.778] of λ=0.7 loss; U1 λ counts {'0.3': 2, '0.5': 30, '0.7': 133, '1.0': 97, '1.3': 10, '1.5': 1, '2.0': 0} (λ_sel {'0.3': 3, '0.5': 23, '0.7': 140, '1.0': 102, '1.3': 5, '1.5': 0, '2.0': 0}); U1 = λ_sel in 0.56.

qwen15/all M3 τ=0.02: prevalence 0.311; AUROC 0.904 [0.834, 0.971]; frozen-threshold accuracy 0.788 [0.663, 0.913] vs always-merge 0.689 [0.486, 0.868] (diff 0.099 [-0.073, 0.293]); unfitted D̂_U>τ accuracy 0.513 [0.367, 0.674]; recall 0.82, precision 0.62.

qwen15/all M3 τ=0.05: prevalence 0.121; AUROC 0.933 [0.833, 0.995]; frozen-threshold accuracy 0.927 [0.839, 0.992] vs always-merge 0.879 [0.751, 0.972] (diff 0.048 [-0.017, 0.129]); unfitted D̂_U>τ accuracy 0.681 [0.486, 0.884]; recall 0.55, precision 0.78.

qwen15/all M3 τ=0.1: prevalence 0.022; AUROC 0.999 [0.992, 1.000]; frozen-threshold accuracy 0.993 [0.973, 1.000] vs always-merge 0.978 [0.930, 1.000] (diff 0.015 [0.000, 0.049]); unfitted D̂_U>τ accuracy 0.890 [0.762, 0.975]; recall 0.67, precision 1.00.

qwen15/P G4: regret (pp) U1 -0.002 [-0.184, 0.168]; λ=1 0.227 [-0.104, 0.680]; λ=0.7 0.080 [-0.138, 0.281]; λ=0.5 1.267 [0.700, 1.946]; L-cal (labeled ref.) 0.047 [-0.163, 0.274]; eval-optimal floor -0.239 [-0.416, -0.103]. U1−λ0.7 -0.082 [-0.271, 0.091]; U1−λ1 -0.229 [-0.704, 0.115]; U1 removes 1.008 [-1.962, 3.559] of λ=1 loss, 1.023 [-5.595, 7.386] of λ=0.7 loss; U1 λ counts {'0.3': 0, '0.5': 9, '0.7': 44, '1.0': 38} (λ_sel {'0.3': 0, '0.5': 8, '0.7': 44, '1.0': 39}); U1 = λ_sel in 0.59.

qwen15/P G7: regret (pp) U1 0.017 [-0.172, 0.192]; λ=1 0.229 [-0.101, 0.680]; λ=0.7 0.082 [-0.137, 0.285]; λ=0.5 1.269 [0.704, 1.946]; L-cal (labeled ref.) 0.186 [-0.063, 0.492]; eval-optimal floor -0.246 [-0.418, -0.111]. U1−λ0.7 -0.065 [-0.262, 0.109]; U1−λ1 -0.212 [-0.680, 0.118]; U1 removes 0.924 [-1.873, 3.412] of λ=1 loss, 0.789 [-5.619, 6.084] of λ=0.7 loss; U1 λ counts {'0.3': 0, '0.5': 9, '0.7': 44, '1.0': 35, '1.3': 3, '1.5': 0, '2.0': 0} (λ_sel {'0.3': 0, '0.5': 8, '0.7': 43, '1.0': 39, '1.3': 1, '1.5': 0, '2.0': 0}); U1 = λ_sel in 0.57.

qwen15/P M3 τ=0.02: prevalence 0.176; AUROC 0.868 [0.658, 0.992]; frozen-threshold accuracy 0.791 [0.583, 0.965] vs always-merge 0.824 [0.654, 0.976] (diff -0.033 [-0.259, 0.176]); unfitted D̂_U>τ accuracy 0.429 [0.229, 0.631]; recall 0.69, precision 0.44.

qwen15/P M3 τ=0.05: prevalence 0.033; AUROC 0.905 [0.732, 1.000]; frozen-threshold accuracy 0.967 [0.881, 1.000] vs always-merge 0.967 [0.881, 1.000] (diff 0.000 [0.000, 0.000]); unfitted D̂_U>τ accuracy 0.736 [0.500, 0.942]; recall 0.00, precision 0.00.

qwen15/P M3 τ=0.1: prevalence 0.000; AUROC nan [nan, nan]; frozen-threshold accuracy 1.000 [1.000, 1.000] vs always-merge 1.000 [1.000, 1.000] (diff 0.000 [0.000, 0.000]); unfitted D̂_U>τ accuracy 0.978 [0.906, 1.000]; recall 0.00, precision 0.00.

qwen15/R0 G4: regret (pp) U1 0.030 [-0.144, 0.229]; λ=1 0.647 [0.144, 1.485]; λ=0.7 0.320 [0.011, 0.834]; λ=0.5 1.353 [0.658, 2.257]; L-cal (labeled ref.) 0.072 [-0.116, 0.281]; eval-optimal floor -0.201 [-0.353, -0.082]. U1−λ0.7 -0.290 [-0.796, -0.008]; U1−λ1 -0.617 [-1.428, -0.088]; U1 removes 0.954 [0.371, 1.412] of λ=1 loss, 0.907 [-0.224, 2.442] of λ=0.7 loss; U1 λ counts {'0.3': 1, '0.5': 13, '0.7': 49, '1.0': 28} (λ_sel {'0.3': 2, '0.5': 11, '0.7': 51, '1.0': 27}); U1 = λ_sel in 0.57.

qwen15/R0 G7: regret (pp) U1 -0.006 [-0.265, 0.212]; λ=1 0.660 [0.157, 1.496]; λ=0.7 0.333 [0.012, 0.860]; λ=0.5 1.366 [0.658, 2.305]; L-cal (labeled ref.) 0.091 [-0.221, 0.381]; eval-optimal floor -0.281 [-0.560, -0.103]. U1−λ0.7 -0.339 [-1.021, -0.003]; U1−λ1 -0.666 [-1.556, -0.101]; U1 removes 1.009 [0.397, 1.694] of λ=1 loss, 1.019 [-0.289, 2.437] of λ=0.7 loss; U1 λ counts {'0.3': 1, '0.5': 13, '0.7': 49, '1.0': 23, '1.3': 4, '1.5': 1, '2.0': 0} (λ_sel {'0.3': 2, '0.5': 11, '0.7': 51, '1.0': 25, '1.3': 2, '1.5': 0, '2.0': 0}); U1 = λ_sel in 0.53.

qwen15/R0 M3 τ=0.02: prevalence 0.407; AUROC 0.900 [0.775, 0.977]; frozen-threshold accuracy 0.758 [0.593, 0.890] vs always-merge 0.593 [0.370, 0.819] (diff 0.165 [-0.086, 0.440]); unfitted D̂_U>τ accuracy 0.582 [0.423, 0.738]; recall 0.81, precision 0.67.

qwen15/R0 M3 τ=0.05: prevalence 0.176; AUROC 0.954 [0.877, 1.000]; frozen-threshold accuracy 0.923 [0.825, 1.000] vs always-merge 0.824 [0.646, 0.965] (diff 0.099 [-0.023, 0.273]); unfitted D̂_U>τ accuracy 0.659 [0.458, 0.875]; recall 0.69, precision 0.85.

qwen15/R0 M3 τ=0.1: prevalence 0.055; AUROC 0.998 [0.980, 1.000]; frozen-threshold accuracy 0.978 [0.918, 1.000] vs always-merge 0.945 [0.833, 1.000] (diff 0.033 [0.000, 0.110]); unfitted D̂_U>τ accuracy 0.857 [0.675, 0.976]; recall 0.60, precision 1.00.

qwen15/S1 G4: regret (pp) U1 -0.060 [-0.277, 0.103]; λ=1 0.588 [0.027, 1.377]; λ=0.7 0.240 [-0.051, 0.570]; λ=0.5 1.512 [0.722, 2.587]; L-cal (labeled ref.) 0.026 [-0.214, 0.256]; eval-optimal floor -0.229 [-0.452, -0.062]. U1−λ0.7 -0.300 [-0.642, -0.051]; U1−λ1 -0.648 [-1.499, -0.051]; U1 removes 1.102 [0.354, 2.087] of λ=1 loss, 1.251 [-3.106, 4.960] of λ=0.7 loss; U1 λ counts {'0.3': 1, '0.5': 9, '0.7': 40, '1.0': 41} (λ_sel {'0.3': 1, '0.5': 4, '0.7': 46, '1.0': 40}); U1 = λ_sel in 0.64.

qwen15/S1 G7: regret (pp) U1 -0.084 [-0.304, 0.084]; λ=1 0.570 [0.009, 1.343]; λ=0.7 0.222 [-0.053, 0.522]; λ=0.5 1.494 [0.717, 2.546]; L-cal (labeled ref.) 0.064 [-0.195, 0.345]; eval-optimal floor -0.279 [-0.482, -0.112]. U1−λ0.7 -0.306 [-0.647, -0.058]; U1−λ1 -0.654 [-1.500, -0.067]; U1 removes 1.147 [0.348, 2.518] of λ=1 loss, 1.376 [-2.748, 6.012] of λ=0.7 loss; U1 λ counts {'0.3': 1, '0.5': 8, '0.7': 40, '1.0': 39, '1.3': 3, '1.5': 0, '2.0': 0} (λ_sel {'0.3': 1, '0.5': 4, '0.7': 46, '1.0': 38, '1.3': 2, '1.5': 0, '2.0': 0}); U1 = λ_sel in 0.59.

qwen15/S1 M3 τ=0.02: prevalence 0.352; AUROC 0.918 [0.790, 0.996]; frozen-threshold accuracy 0.813 [0.651, 0.942] vs always-merge 0.648 [0.390, 0.864] (diff 0.165 [-0.089, 0.422]); unfitted D̂_U>τ accuracy 0.527 [0.353, 0.718]; recall 0.91, precision 0.67.

qwen15/S1 M3 τ=0.05: prevalence 0.154; AUROC 0.890 [0.673, 1.000]; frozen-threshold accuracy 0.890 [0.738, 1.000] vs always-merge 0.846 [0.682, 0.975] (diff 0.044 [-0.071, 0.159]); unfitted D̂_U>τ accuracy 0.648 [0.420, 0.859]; recall 0.50, precision 0.70.

qwen15/S1 M3 τ=0.1: prevalence 0.011; AUROC 1.000 [1.000, 1.000]; frozen-threshold accuracy 1.000 [1.000, 1.000] vs always-merge 0.989 [0.935, 1.000] (diff 0.011 [0.000, 0.065]); unfitted D̂_U>τ accuracy 0.835 [0.655, 0.964]; recall 1.00, precision 1.00.

## Integrity (stage0, per seed; rule (b) not applicable)

| task | seed | lr | steps | micro-bs | train loss (mean) | last logged loss | prior entropy | eval | (a) thr | (a) | (c) max diff / tol | (c2) hold bf16 / fp32 | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| cola | 0 | 0.0003 | 2360 | 32 | 0.1465 | 0.0007 | 0.6046 | 0.8476 | 0.7913 | True | 1.5e-04 / 1.0e-02 True | 0.8490 / 0.8510 True | **True** |
| sst2 | 0 | 0.0003 | 1563 | 32 | 0.1918 | 0.1316 | 0.6867 | 0.9576 | 0.6092 | True | 1.6e-05 / 5.3e-03 True | 0.9640 / 0.9640 True | **True** |
| mrpc | 0 | 0.0003 | 840 | 32 | 0.1585 | 0.0001 | 0.6267 | 0.8873 | 0.7838 | True | 6.1e-05 / 1.2e-02 True | 0.8760 / 0.8750 True | **True** |
| stsb | 0 | 0.0003 | 1490 | 32 | 0.3058 | 0.0202 | nan | 0.9217 | 0.7000 | True | 4.3e-06 / 5.0e-03 True | 0.9085 / 0.9091 True | **True** |
| mnli | 0 | 0.0003 | 1563 | 32 | 0.4183 | 0.3057 | 1.0986 | 0.8830 | 0.4560 | True | 1.6e-05 / 7.2e-03 True | 0.8950 / 0.8960 True | **True** |
| qnli | 0 | 0.0003 | 1563 | 32 | 0.2949 | 0.1670 | 0.6931 | 0.9464 | 0.6056 | True | 1.5e-05 / 4.4e-03 True | 0.9200 / 0.9200 True | **True** |
| rte | 0 | 0.0003 | 470 | 32 | 0.1702 | 0.0000 | 0.6931 | 0.8267 | 0.5729 | True | 1.4e-04 / 1.3e-02 True | 0.8640 / 0.8610 True | **True** |
| wic | 0 | 0.0003 | 1390 | 32 | 0.1829 | 0.0000 | 0.6931 | 0.7257 | 0.6000 | True | 1.6e-04 / 9.3e-03 True | 0.7920 / 0.7910 True | **True** |
| snli | 0 | 0.0003 | 1563 | 32 | 0.3810 | 0.2980 | 1.0986 | 0.9156 | 0.4328 | True | 1.5e-05 / 7.2e-03 True | 0.8970 / 0.8960 True | **True** |
| scitail | 0 | 0.0003 | 1382 | 32 | 0.1246 | 0.0283 | 0.6571 | 0.9778 | 0.5962 | True | 1.2e-05 / 6.9e-03 True | 0.9770 / 0.9760 True | **True** |
| ag_news | 0 | 0.0003 | 1563 | 32 | 0.2722 | 0.1577 | 1.3863 | 0.9490 | 0.3494 | True | 3.5e-05 / 1.5e-02 True | 0.9430 / 0.9440 True | **True** |
| imdb | 0 | 0.0003 | 1500 | 16 | 0.3680 | 0.1634 | 0.6931 | 0.9534 | 0.5932 | True | 2.8e-05 / 5.1e-03 True | 0.9430 / 0.9440 True | **True** |
| trec | 0 | 0.0003 | 1400 | 32 | 0.1587 | 0.0000 | 1.6528 | 0.9740 | 0.2880 | True | 4.6e-05 / 2.4e-02 True | 0.9610 / 0.9610 True | **True** |
| yelp_polarity | 0 | 0.0003 | 1563 | 16 | 0.2878 | 0.1824 | 0.6931 | 0.9770 | 0.6052 | True | 1.3e-05 / 7.1e-03 True | 0.9890 / 0.9890 True | **True** |
| cola | 1 | 0.0003 | 2360 | 32 | 0.1367 | 0.0038 | 0.6046 | 0.8476 | 0.7913 | True | 1.6e-04 / 1.1e-02 True | 0.8510 / 0.8530 True | **True** |
| sst2 | 1 | 0.0003 | 1563 | 32 | 0.1880 | 0.1242 | 0.6867 | 0.9564 | 0.6092 | True | 1.6e-05 / 7.7e-03 True | 0.9590 / 0.9580 True | **True** |
| mrpc | 1 | 0.0003 | 840 | 32 | 0.1526 | 0.0000 | 0.6267 | 0.8725 | 0.7838 | True | 9.6e-05 / 1.4e-02 True | 0.8810 / 0.8810 True | **True** |
| stsb | 1 | 0.0003 | 1490 | 32 | 0.3167 | 0.0238 | nan | 0.9154 | 0.7000 | True | 5.0e-06 / 5.0e-03 True | 0.9053 / 0.9053 True | **True** |
| mnli | 1 | 0.0003 | 1563 | 32 | 0.4322 | 0.2919 | 1.0986 | 0.8864 | 0.4560 | True | 1.7e-05 / 6.2e-03 True | 0.8900 / 0.8890 True | **True** |
| qnli | 1 | 0.0003 | 1563 | 32 | 0.2775 | 0.2080 | 0.6931 | 0.9448 | 0.6056 | True | 1.5e-05 / 4.8e-03 True | 0.9160 / 0.9160 True | **True** |
| rte | 1 | 0.0003 | 470 | 32 | 0.1582 | 0.0000 | 0.6931 | 0.8303 | 0.5729 | True | 1.2e-04 / 1.6e-02 True | 0.8530 / 0.8540 True | **True** |
| wic | 1 | 0.0003 | 1390 | 32 | 0.1789 | 0.0000 | 0.6931 | 0.7179 | 0.6000 | True | 9.4e-05 / 1.2e-02 True | 0.7900 / 0.7910 True | **True** |
| snli | 1 | 0.0003 | 1563 | 32 | 0.3846 | 0.2752 | 1.0986 | 0.9200 | 0.4328 | True | 1.1e-05 / 4.8e-03 True | 0.8830 / 0.8840 True | **True** |
| scitail | 1 | 0.0003 | 1382 | 32 | 0.1220 | 0.0465 | 0.6571 | 0.9701 | 0.5962 | True | 1.2e-05 / 1.1e-02 True | 0.9750 / 0.9750 True | **True** |
| ag_news | 1 | 0.0003 | 1563 | 32 | 0.2832 | 0.1500 | 1.3863 | 0.9480 | 0.3494 | True | 4.4e-05 / 1.5e-02 True | 0.9460 / 0.9460 True | **True** |
| imdb | 1 | 0.0003 | 1500 | 16 | 0.3674 | 0.1911 | 0.6931 | 0.9522 | 0.5932 | True | 2.4e-05 / 6.6e-03 True | 0.9450 / 0.9440 True | **True** |
| trec | 1 | 0.0003 | 1400 | 32 | 0.1328 | 0.0000 | 1.6528 | 0.9700 | 0.2880 | True | 3.1e-05 / 3.0e-02 True | 0.9700 / 0.9700 True | **True** |
| yelp_polarity | 1 | 0.0003 | 1563 | 16 | 0.2597 | 0.1096 | 0.6931 | 0.9774 | 0.6052 | True | 1.0e-05 / 6.3e-03 True | 0.9830 / 0.9830 True | **True** |

## Deviations and notes

None.

## Compute

Timing files (s): {"timing_R0": {"stage2_gpu_wall_s": 19666.5}, "timing_S1": {"stage2_gpu_wall_s": 19086.0}, "timing_P": {"stage2_gpu_wall_s": 19093.8}, "timing_main": {"stage0_s": 735.9, "stage1_s": 419.5}, "timing_S2": {"stage2_gpu_wall_s": 3030.6}}
