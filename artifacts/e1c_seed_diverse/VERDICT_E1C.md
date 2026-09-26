# E1c VERDICT — H1c (O_A): **INCONCLUSIVE**; H2c (task-vector cosine): **INCONCLUSIVE**

Generated 2026-09-26 09:14:05 KST. Pre-registration: `PREREG_E1C.md` / `prereg_e1c.json` (sha256 in `prereg_e1c.sha256`); deviations/notes: `DEVIATIONS_E1C.md`.

Valid tasks K = 14: cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity. Seed-1 exclusions: none. Predictor hash ok: True. Seed-0 re-check ok: True. Analysis-code equivalence on E1b (reproduces E1b analysis.json exactly): True.

## Primary: mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1c | O_A | -0.0249 | [-0.507, 0.426] | 0.5658 | 0.5658 | 0.07 | **INCONCLUSIVE** |
| H2c | tv_cosine | 0.1270 | [-0.354, 0.489] | 0.2067 | 0.4134 | 0.07 | **INCONCLUSIVE** |

Rule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).

H1c criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.00, sst2 -0.02, mrpc -0.14, stsb -0.02, mnli 0.20, qnli -0.15, rte 0.02, wic -0.07, snli -0.07, scitail -0.04, ag_news -0.03, imdb -0.00, trec -0.02, yelp_polarity -0.02

H2c criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola 0.20, sst2 0.19, mrpc 0.11, stsb 0.13, mnli -0.03, qnli 0.07, rte 0.09, wic 0.11, snli 0.11, scitail 0.13, ag_news 0.17, imdb 0.17, trec 0.06, yelp_polarity 0.23

P: 91 pairs; D mean 0.0426, median 0.0363, range [0.0024, 0.1778]; λ* counts {'0.5': 19, '0.7': 58, '1.0': 14}; ρ(O_A, tv_cos) = 0.152; min θ_min over pairs/layers = 44.9°; gate active in 0 pairs.

E1b (R0, shared seed 0) for comparison, recomputed with this code: H1 FAIL ρ=-0.1935 CI [-0.521, 0.203]; H2 INCONCLUSIVE ρ=-0.0441 CI [-0.425, 0.316]

## Secondary (exploratory; no multiplicity correction)

### S1: seed-1-only cross-task pairs (shared seed 1; E1b rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (S1) | O_A | -0.0295 | [-0.437, 0.391] | 0.5787 | 0.6699 | 0.00 | **INCONCLUSIVE** |
| H2 (S1) | tv_cosine | 0.0574 | [-0.365, 0.416] | 0.3350 | 0.6699 | 0.00 | **INCONCLUSIVE** |

S1 D mean 0.0459, range [-0.0175, 0.1705]; λ* counts {'0.5': 19, '0.7': 58, '1.0': 14}; min θ_min 42.9°; gate active in 0 pairs.

### Secondary predictors (P and S1)

| predictor | ρ P | CI P | ρ S1 | CI S1 |
|---|---:|---|---:|---|
| O_A | -0.025 | [-0.51, 0.43] | -0.029 | [-0.44, 0.39] |
| tv_cosine | 0.127 | [-0.35, 0.49] | 0.057 | [-0.36, 0.42] |
| O_B | -0.025 | [-0.51, 0.43] | -0.029 | [-0.44, 0.39] |
| mean_theta_min_A_deg | -0.001 | [-0.43, 0.49] | 0.011 | [-0.38, 0.42] |
| min_theta_min_A_deg | 0.059 | [-0.29, 0.44] | 0.067 | [-0.33, 0.45] |
| n_layers_theta_min_lt30_A | nan | [nan, nan] | nan | [nan, nan] |
| sign_conflict_top20 | -0.152 | [-0.50, 0.33] | -0.044 | [-0.40, 0.36] |
| norm_ratio | 0.211 | [-0.15, 0.50] | 0.189 | [-0.15, 0.50] |
| null_z_O_A | -0.025 | [-0.51, 0.43] | -0.029 | [-0.44, 0.39] |
| sign_conflict_all | -0.050 | [-0.45, 0.41] | 0.004 | [-0.41, 0.42] |

Fixed-λ ρ (P): {"O_A": {"lam0.3": -0.373, "lam0.5": -0.233, "lam0.7": -0.041, "lam1.0": 0.064}, "tv_cosine": {"lam0.3": 0.035, "lam0.5": -0.148, "lam0.7": 0.141, "lam1.0": 0.28}}

### Reliability across seed configurations (Spearman over the same unordered task pairs)

| quantity | R0~P | R0~S1 | P~S1 |
|---|---:|---:|---:|
| D | 0.669 | 0.670 | 0.808 |
| O_A | 0.956 | 0.940 | 0.937 |
| tv_cosine | 0.609 | 0.729 | 0.512 |
| lam_selected | 0.627 | 0.597 | 0.725 |

### O_A: shared-seed vs mixed-seed

Random rank-8 null: O_A mean 0.00913 (sd 0.00019, p95 0.00944).

| population | n | O_A mean | sd | median | IQR | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layers θ_min<30° (total) |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| R0 (shared seed 0, E1b) | 91 | 0.01736 | 0.00308 | 0.01671 | [0.01527, 0.01855] | 0.01225 | 0.02767 | [16.1, 95.4] | 1.00 | 0.0017 | 45.1 | 0 |
| S1 (shared seed 1) | 91 | 0.01750 | 0.00303 | 0.01684 | [0.01571, 0.01879] | 0.01225 | 0.02677 | [16.0, 90.7] | 1.00 | 0.0016 | 42.9 | 0 |
| P (mixed seed) | 91 | 0.01657 | 0.00270 | 0.01598 | [0.01473, 0.01741] | 0.01209 | 0.02368 | [15.3, 74.8] | 1.00 | 0.0008 | 44.9 | 0 |
| S2 (same task, mixed seed) | 14 | 0.03142 | 0.00482 | 0.03152 | [0.02939, 0.03321] | 0.02285 | 0.04206 | [70.6, 169.4] | 1.00 | 0.0210 | 39.8 | 0 |

paired_P_minus_R0: n=91, mean diff -0.00079, median -0.00072, P lower in 85% of task pairs, ratio of means 0.954, Wilcoxon p = 1.75e-12

paired_P_minus_S1: n=91, mean diff -0.00093, median -0.00093, P lower in 86% of task pairs, ratio of means 0.947, Wilcoxon p = 1.52e-12

### S2: same-task seed pairs (t@s0 × t@s1)

Gate (θ★ = 30°) fires in **0/14** same-task pairs; min θ_min over all S2 pairs/layers = 39.82°.

| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | layers < 30° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cola | 61.66 | encoder.layer.8.attention.self.query | 0 | 0 | — | 0.02935 | 104.0 | 0.0221 | 0.5 | 0.0016 | 0.9984 | 0.9984 | 0.9984 |
| sst2 | 61.38 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.02489 | 81.1 | 0.0156 | 0.5 | -0.0125 | 1.0125 | 1.0125 | 1.0125 |
| mrpc | 51.71 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.03687 | 142.7 | 0.0248 | 0.5 | -0.0043 | 1.0043 | 1.0043 | 1.0043 |
| stsb | 48.42 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.04206 | 169.4 | 0.0395 | 0.5 | -0.0012 | 1.0012 | 1.0012 | 1.0012 |
| mnli | 39.82 | pooler.dense | 0 | 1 | — | 0.03294 | 122.5 | 0.0203 | 0.5 | 0.0005 | 0.9995 | 0.9995 | 0.9995 |
| qnli | 53.87 | encoder.layer.11.attention.self.query | 0 | 0 | — | 0.03007 | 107.7 | 0.0145 | 0.5 | -0.0079 | 1.0079 | 1.0079 | 1.0079 |
| rte | 48.53 | encoder.layer.11.output.dense | 0 | 0 | — | 0.03010 | 107.9 | 0.0106 | 0.5 | -0.0222 | 1.0222 | 1.0222 | 1.0222 |
| wic | 49.43 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.03418 | 128.9 | 0.0220 | 0.5 | 0.0022 | 0.9978 | 0.9978 | 0.9978 |
| snli | 55.18 | pooler.dense | 0 | 0 | — | 0.02782 | 96.1 | 0.0166 | 0.5 | 0.0030 | 0.9970 | 0.9970 | 0.9970 |
| scitail | 55.58 | encoder.layer.10.attention.output.dense | 0 | 0 | — | 0.03301 | 122.9 | 0.0187 | 0.5 | -0.0098 | 1.0098 | 1.0098 | 1.0098 |
| ag_news | 56.58 | encoder.layer.11.output.dense | 0 | 0 | — | 0.02285 | 70.6 | 0.0102 | 0.7 | 0.0054 | 0.9946 | 0.9946 | 0.9946 |
| imdb | 52.60 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.03328 | 124.2 | 0.0267 | 0.5 | -0.0039 | 1.0039 | 1.0039 | 1.0039 |
| trec | 60.12 | encoder.layer.11.attention.output.dense | 0 | 0 | — | 0.03295 | 122.5 | 0.0309 | 0.5 | 0.0010 | 0.9990 | 0.9990 | 0.9990 |
| yelp_polarity | 55.00 | encoder.layer.11.attention.self.value | 0 | 0 | — | 0.02951 | 104.9 | 0.0210 | 0.7 | -0.0009 | 1.0009 | 1.0009 | 1.0009 |

E1b-grid gate@lam*TA − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/14/0

E1b-grid gate@own − TA: mean +0.000 pp, CI [+0.000, +0.000] pp, sign-flip p = 1.000, W/T/L 0/14/0

**E3 code (amendment-A2 grids) on S2, 14 pairs.** Mean normalized score: TA 1.0038, PICO_TA 1.0036, GATE 1.0038, FORCEGATE 1.0026, TA4_score 1.0035, PICO_TA_c1_score 0.9945.

| contrast | mean (pp) | 95% CI over tasks (pp) | sign-flip p (2-sided) | W/T/L |
|---|---:|---|---:|---|
| GATE-TA | +0.000 | [+0.000, +0.000] | 1.0000 | 0/14/0 |
| FORCEGATE-TA | -0.115 | [-0.359, +0.167] | 0.4457 | 7/0/7 |
| PICO_TA-TA | -0.014 | [-0.200, +0.169] | 0.8899 | 6/0/8 |
| GATE-PICO_TA | +0.014 | [-0.169, +0.200] | 0.8899 | 8/0/6 |
| FORCEGATE-PICO_TA | -0.101 | [-0.308, +0.072] | 0.3885 | 6/0/8 |
| PICO_TA_c1_score-TA | -0.922 | [-1.803, -0.107] | 0.0570 | 3/1/10 |
| TA4_score-TA | -0.027 | [-0.180, +0.081] | 0.8137 | 3/8/3 |

Gate vs TA on S2 pairs with ≥ 1 FAIL layer: n = 0; per pair (pp): —.

Selected-config boundary rate: {'TA': 0.0, 'PICO_TA': 0.0, 'GATE': 0.0, 'FORCEGATE': 0.0}. Sanity (E3 TA at E1b λs = e1c stage2 values): max |Δ| = 0.00e+00.

| task | TA | PICO_TA | GATE | FORCEGATE | GATE − TA (pp) | PICO − TA (pp) | FORCEGATE − TA (pp) | GATE fail layers | TA sel | PICO sel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| cola | 1.0064 | 1.0035 | 1.0064 | 1.0024 | +0.000 | -0.288 | -0.403 | 0 | {"lam": 0.6} | {"c": 0.9} |
| sst2 | 1.0125 | 1.0131 | 1.0125 | 1.0087 | +0.000 | +0.062 | -0.375 | 0 | {"lam": 0.5} | {"c": 0.75} |
| mrpc | 1.0043 | 1.0058 | 1.0043 | 0.9942 | +0.000 | +0.145 | -1.007 | 0 | {"lam": 0.5} | {"c": 0.75} |
| stsb | 1.0026 | 1.0025 | 1.0026 | 1.0019 | +0.000 | -0.010 | -0.077 | 0 | {"lam": 0.4} | {"c": 0.6} |
| mnli | 0.9995 | 0.9967 | 0.9995 | 0.9971 | +0.000 | -0.285 | -0.237 | 0 | {"lam": 0.5} | {"c": 0.75} |
| qnli | 1.0055 | 1.0083 | 1.0055 | 1.0060 | +0.000 | +0.276 | +0.051 | 0 | {"lam": 0.6} | {"c": 0.75} |
| rte | 1.0222 | 1.0139 | 1.0222 | 1.0141 | +0.000 | -0.824 | -0.806 | 0 | {"lam": 0.5} | {"c": 0.75} |
| wic | 0.9978 | 1.0045 | 0.9978 | 1.0101 | +0.000 | +0.674 | +1.236 | 0 | {"lam": 0.5} | {"c": 0.75} |
| snli | 0.9970 | 0.9941 | 0.9970 | 0.9923 | +0.000 | -0.292 | -0.467 | 0 | {"lam": 0.5} | {"c": 0.75} |
| scitail | 1.0098 | 1.0123 | 1.0098 | 1.0107 | +0.000 | +0.246 | +0.082 | 0 | {"lam": 0.5} | {"c": 0.75} |
| ag_news | 0.9946 | 0.9934 | 0.9946 | 0.9957 | +0.000 | -0.119 | +0.112 | 0 | {"lam": 0.85} | {"c": 1.25} |
| imdb | 1.0006 | 1.0039 | 1.0006 | 1.0033 | +0.000 | +0.337 | +0.275 | 0 | {"lam": 0.6} | {"c": 0.75} |
| trec | 0.9990 | 0.9980 | 0.9990 | 0.9990 | +0.000 | -0.102 | +0.001 | 0 | {"lam": 0.6} | {"c": 0.75} |
| yelp_polarity | 1.0009 | 1.0007 | 1.0009 | 1.0010 | +0.000 | -0.016 | +0.005 | 0 | {"lam": 0.7} | {"c": 1.0} |

### Method comparison (E1b stage-2 exploratory methods)

**P**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9574 | — | — | — | — |
| TIES-lite@lam*TA | 0.8905 | -0.0668 | [-0.1010, -0.0362] | 2/0/89 | 1.77e-16 |
| TIES-lite@own-lam | 0.9360 | -0.0214 | [-0.0339, -0.0126] | 14/0/77 | 5.38e-13 |
| gate30@lam*TA | 0.9574 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |
| gate30@own-lam | 0.9574 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |

**S1**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9541 | — | — | — | — |
| TIES-lite@lam*TA | 0.8864 | -0.0677 | [-0.1039, -0.0367] | 2/0/89 | 2.91e-16 |
| TIES-lite@own-lam | 0.9346 | -0.0195 | [-0.0303, -0.0107] | 15/0/76 | 5.71e-12 |
| gate30@lam*TA | 0.9541 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |
| gate30@own-lam | 0.9541 | +0.0000 | [+0.0000, +0.0000] | 0/91/0 | nan |

## Integrity (seed-1 adapters; E1b stage0 unmodified)

| task | steps | train loss | eval (s1) | eval (s0, E1b) | (a) thr | (a) | (b) ref−5pp | (b) | (c) max diff | valid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| cola | 2360 | 0.2183 | 0.8360 | 0.8236 | 0.7913 | True | mcc 0.5981 ≥ 0.5153 | True | 1.1e-05 | **True** |
| sst2 | 5625 | 0.1959 | 0.9174 | 0.9186 | 0.6092 | True | accuracy 0.9174 ≥ 0.8732 | True | 5.1e-06 | **True** |
| mrpc | 840 | 0.1965 | 0.8529 | 0.8505 | 0.7838 | True | accuracy 0.8529 ≥ 0.7907 | True | 7.6e-06 | **True** |
| stsb | 1490 | 0.4165 | 0.8942 | 0.8894 | 0.7000 | True | spearman 0.8942 ≥ 0.8348 | True | 2.4e-06 | **True** |
| mnli | 36723 | 0.6052 | 0.8202 | 0.8237 | 0.4274 | True | accuracy 0.8202 ≥ 0.7891 | True | 5.3e-06 | **True** |
| qnli | 5625 | 0.3338 | 0.8931 | 0.8977 | 0.6054 | True | accuracy 0.8931 ≥ 0.8566 | True | 2.6e-06 | **True** |
| rte | 470 | 0.3485 | 0.6570 | 0.6426 | 0.5729 | True | accuracy 0.6570 ≥ 0.6070 | True | 9.3e-06 | **True** |
| wic | 1390 | 0.2140 | 0.6991 | 0.6959 | 0.6000 | True | n/a | None | 2.4e-05 | **True** |
| snli | 5625 | 0.4830 | 0.8693 | 0.8702 | 0.4331 | True | n/a | None | 4.6e-06 | **True** |
| scitail | 2073 | 0.1654 | 0.9379 | 0.9333 | 0.5962 | True | n/a | None | 1.2e-05 | **True** |
| ag_news | 5625 | 0.2386 | 0.9405 | 0.9387 | 0.3500 | True | n/a | None | 8.0e-06 | **True** |
| imdb | 2250 | 0.2718 | 0.8898 | 0.8898 | 0.5977 | True | n/a | None | 5.8e-06 | **True** |
| trec | 1400 | 0.1705 | 0.9780 | 0.9660 | 0.2880 | True | n/a | None | 3.6e-06 | **True** |
| yelp_polarity | 5625 | 0.1766 | 0.9403 | 0.9407 | 0.6108 | True | n/a | None | 3.5e-06 | **True** |

## Deviations and notes

# E1c deviations and implementation notes (append-only; KST)

## N1 — 2026-09-26 05:43 KST (implementation note, not a change of method)
The stage-3 analysis code is in a separate file, `e1c_analysis.py`, which `e1c.py --stage stage3` calls. It was written while seed-1 training
was running, before any seed-1 adapter had been evaluated, and before integrity, predictors, or merges existed for E1c. Its sha256 is recorded in
`code_analysis_e1c.sha256` at deployment. The statistics are copied verbatim from `e1b.stage3` (perms, bootstrap, LOTO, Holm, rule) and from
`e3.signflip`. It also reruns the same statistics on the E1b R0 data and checks that they reproduce E1b `analysis.json` exactly (analysis-code equivalence).

## N2 — equivalence gates (PREREG_E1C §4–5), run 05:39:47–05:41:09 KST, before training (05:41:26)
stage1 on the 91 R0 pairs reproduced E1b `predictors.csv` exactly: max relative difference 0 on all 13 numeric columns.
stage2 on cola-sst2 and rte-wic reproduced E1b `pair_results.csv`: max |Δ| 4.9e-11 and 4.3e-11, which is CSV %.10g rounding, with λ\* equal.
Both gates passed (`equiv/equivalence.json`).

## N3 — post-run notes, 2026-09-26 ~09:20 KST (written AFTER all E1c results; they change no predictor, selection, exclusion, or verdict)
1. **O_B is identical to O_A.** This holds in E1b as well as E1c (max |O_A − O_B| = 0 in both `predictors.csv` files). In `e1b.stage1`,
   QB = Qb·U, where U comes from the SVD of Rb·A (8 × 8, orthogonal), so QB spans col(B), the same subspace as orth(B). "O_B" and
   `mean_theta_min_B_deg` therefore duplicate O_A and `mean_theta_min_A_deg`. They are not an independent A-side predictor, and this also
   applies to the E1b secondary table. It does not affect H1/H2/H1c/H2c.
2. **Shared initialization explains little of the inflated O_A.** Mixed-seed O_A is only about 5% lower than shared-seed O_A for the same task pair
   (P/R0 ratio of means 0.954; P lower in 85% of pairs; Wilcoxon p ≈ 2e-12). The null z is still 15–75, and 100% of P pairs lie above the null p95.
   Task-vector cosine roughly halves (mean 0.0008 vs 0.0017). The O_A range is not wider under mixed seeds (IQR [0.0147, 0.0174] vs [0.0153, 0.0186]).
3. **E1b's H1 FAIL does not replicate as FAIL.** P (mixed seed): ρ = −0.025, CI [−0.51, 0.43] → INCONCLUSIVE. S1 (E1b recipe, seed 1, shared init):
   ρ = −0.030, CI [−0.44, 0.39] → INCONCLUSIVE (rule applied descriptively). All three point estimates are ≤ 0 (−0.19, −0.03, −0.02), and none comes close to PASS.
   What changes is the width of the task-block CI. So the FAIL-vs-INCONCLUSIVE boundary for H1 is seed-sensitive at K = 14.
   The per-pair D reliability across seed configurations is Spearman 0.67 (R0~P), 0.67 (R0~S1), and 0.81 (P~S1).
4. **Same-task seed pairs never reach 30°.** Minimum θ_min = 39.8° (MNLI, pooler.dense). In the hubish MNLI s7 × s42 pair it was 20.6°, under a different recipe
   (dropout 0.05, different seeds, 500-step warmup). The θ★ = 30° gate therefore never fires in E1c (0/14 S2, 0/91 P, 0/91 S1), and the E3-S2 analogue is not estimable (n = 0).
5. `make_verdict_e1c.py` (a report writer that reads `analysis_e1c.json`), `pair_results_e1c.csv` (the three jsonl files concatenated), and
   `gpu_time_summary.json` were created after the results. They are for reporting only.


## Compute

Timing files (s): {"timing_S1": {"stage2_gpu_wall_s": 6165.8}, "timing_P": {"stage2_gpu_wall_s": 7794.1}, "timing_main": {"stage0_s": 112.1, "stage1_s": 87.2}, "timing_S2": {"stage2_gpu_wall_s": 1222.0, "e3same_s": 2067.5}, "timing_equiv": {"stage1_s": 62.0, "stage2_gpu_wall_s": 17.2, "stage2_s": 19.6}}

GPU wall-clock summary: {"equivalence_gates": "05:39:47-05:41:09 KST (82 s, before training)",
 "first_training_launch": "2026-09-26 05:41:26 KST", "training_end": "2026-09-26 06:32:25 KST (mnli 06:30:19, queue B 06:32:25)",
 "stage0": "06:32:29-06:34:22", "stage1_predictors_frozen": "06:35:50",
 "merges": "06:35:50-09:13:38 (2 concurrent processes: P 06:35-08:45:49; S2 -> e3same 06:56-07:30:48 -> S1 07:30:48-09:13:38)",
 "last_gpu_job_end": "2026-09-26 09:13:38 KST",
 "wall_clock_first_training_to_last_gpu_job_h": 3.537, "cap_h": 8, "within_cap": true}

