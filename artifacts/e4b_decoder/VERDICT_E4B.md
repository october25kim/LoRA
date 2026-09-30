# E4B VERDICT (Qwen2.5-0.5B decoder) — H1 (O_A): **INCONCLUSIVE**; H2 (task-vector cosine): **INCONCLUSIVE**  (primary population P, mixed seed)

Generated 2026-09-26 23:07:16 KST. Pre-registration: `PREREG_E4B.md` / `prereg_e4b.json` (sha256 in `prereg_e4b.sha256`); deviations/notes: `DEVIATIONS_E4B.md`.

Valid tasks K = 14: cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity. Exclusions: none. Predictor hash ok: True. Analysis-code equivalence on E1b (reproduces E1b analysis.json): True.

Pilot projection: train 1.35 h/seed, core (2 seeds) 5.52 h vs threshold 11.0 h -> seeds [0, 1], primary P.

Pilot (pre-registered, held-out only): lr_main = 0.0003 (non-diverged [0.0001, 0.0003, 0.001]; mean held-out {'0.0001': 0.8385, '0.0003': 0.8694999999999999, '0.001': 0.833}). lr 0.0001: sst2 hold 0.9420 (maj 0.560, tail loss 0.158, diverged False), rte hold 0.7350 (maj 0.488, tail loss 0.002, diverged False); lr 0.0003: sst2 hold 0.9560 (maj 0.560, tail loss 0.147, diverged False), rte hold 0.7830 (maj 0.488, tail loss 0.004, diverged False); lr 0.001: sst2 hold 0.9200 (maj 0.560, tail loss 0.226, diverged False), rte hold 0.7460 (maj 0.488, tail loss 0.050, diverged False)

## Primary: mixed-seed cross-task pairs P (seed-0 adapter for the alphabetically first task × seed-1 adapter for the other)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (P) | O_A | 0.3452 | [-0.096, 0.609] | 0.0301 | 0.0301 | 1.00 | **INCONCLUSIVE** |
| H2 (P) | tv_cosine | 0.3684 | [-0.079, 0.652] | 0.0122 | 0.0244 | 1.00 | **INCONCLUSIVE** |

Rule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥ 75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE (E1b rule, verbatim).

H1 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': True, 'loto_ge_75pct_gt_0.2': True}. LOTO ρ: cola 0.34, sst2 0.43, mrpc 0.37, stsb 0.30, mnli 0.33, qnli 0.35, rte 0.37, wic 0.38, snli 0.30, scitail 0.33, ag_news 0.34, imdb 0.37, trec 0.20, yelp_polarity 0.39

H2 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': True, 'loto_ge_75pct_gt_0.2': True}. LOTO ρ: cola 0.36, sst2 0.44, mrpc 0.37, stsb 0.29, mnli 0.30, qnli 0.37, rte 0.45, wic 0.37, snli 0.33, scitail 0.32, ag_news 0.37, imdb 0.42, trec 0.27, yelp_polarity 0.46

P: 91 pairs; D mean 0.0406, median 0.0269, range [-0.0021, 0.2028]; λ* counts {'0.3': 1, '0.5': 1, '0.7': 41, '1.0': 48}; ρ(O_A, tv_cos) = 0.747; min θ_min over pairs/layers = 21.1°; gate active in 66 pairs.

## Secondary (exploratory; no multiplicity correction across populations)

### R0: seed-0 cross-task pairs (E1b confirmatory design on Qwen; rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (R0) | O_A | 0.3665 | [-0.078, 0.635] | 0.0219 | 0.0220 | 1.00 | **INCONCLUSIVE** |
| H2 (R0) | tv_cosine | 0.3948 | [-0.076, 0.665] | 0.0110 | 0.0220 | 1.00 | **INCONCLUSIVE** |

R0: 91 pairs; D mean 0.0413, median 0.0353, range [-0.0110, 0.1843]; λ* counts {'0.3': 3, '0.5': 34, '0.7': 43, '1.0': 11}; ρ(O_A, tv_cos) = 0.883; min θ_min over pairs/layers = 13.9°; gate active in 78 pairs.

### S1: seed-1 cross-task pairs (rule applied descriptively)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (S1) | O_A | 0.2112 | [-0.242, 0.551] | 0.1403 | 0.2806 | 0.71 | **INCONCLUSIVE** |
| H2 (S1) | tv_cosine | 0.1046 | [-0.372, 0.512] | 0.2991 | 0.2991 | 0.07 | **INCONCLUSIVE** |

S1: 91 pairs; D mean 0.0453, median 0.0314, range [0.0017, 0.2789]; λ* counts {'0.3': 3, '0.5': 18, '0.7': 45, '1.0': 25}; ρ(O_A, tv_cos) = 0.879; min θ_min over pairs/layers = 16.5°; gate active in 78 pairs.

### Pooled 3-population summary (mean D and predictors over R0, S1, P per task pair; descriptive)

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | rule outcome |
|---|---|---:|---|---:|---:|---:|---|
| H1 (pooled) | O_A | 0.3665 | [-0.060, 0.649] | 0.0378 | 0.0756 | 1.00 | **INCONCLUSIVE** |
| H2 (pooled) | tv_cosine | 0.3358 | [-0.130, 0.658] | 0.0492 | 0.0756 | 0.93 | **INCONCLUSIVE** |

Pooled D mean 0.0424, n = 91 task pairs.

### Secondary predictors

| predictor | ρ P | CI P | ρ R0 | CI R0 | ρ S1 | CI S1 |
|---|---:|---|---:|---|---:|---|
| O_A | 0.345 | [-0.10, 0.61] | 0.366 | [-0.08, 0.63] | 0.211 | [-0.24, 0.55] |
| tv_cosine | 0.368 | [-0.08, 0.65] | 0.395 | [-0.08, 0.67] | 0.105 | [-0.37, 0.51] |
| mean_theta_min_A_deg | -0.333 | [-0.63, 0.14] | -0.369 | [-0.65, 0.07] | -0.184 | [-0.56, 0.29] |
| min_theta_min_A_deg | 0.052 | [-0.32, 0.42] | 0.074 | [-0.30, 0.43] | -0.014 | [-0.42, 0.45] |
| n_layers_theta_min_lt30_A | 0.018 | [-0.31, 0.34] | -0.097 | [-0.46, 0.30] | -0.038 | [-0.35, 0.30] |
| sign_conflict_top20 | -0.387 | [-0.67, 0.12] | -0.450 | [-0.69, 0.02] | -0.133 | [-0.53, 0.34] |
| norm_ratio | 0.234 | [-0.18, 0.57] | 0.271 | [-0.23, 0.67] | 0.209 | [-0.25, 0.54] |
| null_z_O_A | 0.345 | [-0.10, 0.61] | 0.366 | [-0.08, 0.63] | 0.211 | [-0.24, 0.55] |
| sign_conflict_all | -0.289 | [-0.60, 0.18] | -0.253 | [-0.60, 0.21] | -0.061 | [-0.47, 0.40] |

Fixed-λ ρ (P): {"O_A": {"lam0.3": 0.64, "lam0.5": 0.515, "lam0.7": 0.376, "lam1.0": 0.255}, "tv_cosine": {"lam0.3": 0.508, "lam0.5": 0.512, "lam0.7": 0.426, "lam1.0": 0.332}}

Fixed-λ ρ (R0): {"O_A": {"lam0.3": 0.21, "lam0.5": 0.287, "lam0.7": 0.364, "lam1.0": 0.388}, "tv_cosine": {"lam0.3": 0.124, "lam0.5": 0.281, "lam0.7": 0.4, "lam1.0": 0.405}}

Fixed-λ ρ (S1): {"O_A": {"lam0.3": 0.316, "lam0.5": 0.219, "lam0.7": 0.225, "lam1.0": 0.217}, "tv_cosine": {"lam0.3": 0.216, "lam0.5": 0.095, "lam0.7": 0.137, "lam1.0": 0.078}}

### D stability / reliability across seed configurations (Spearman over the same unordered task pairs)

| quantity | P~R0 | P~S1 | R0~S1 |
|---|---:|---:|---:|
| D | 0.690 | 0.653 | 0.760 |
| O_A | 0.784 | 0.850 | 0.852 |
| tv_cosine | 0.704 | 0.712 | 0.863 |
| lam_selected | 0.119 | 0.333 | 0.503 |

### O_A vs random null; θ_min distribution; gate firings

Random rank-8 null: O_A mean 0.02214 (sd 0.00043, p95 0.02285).

| population | n | O_A mean | sd | median | min | max | null z range | frac O_A > null p95 | tv_cos mean | min θ_min (°) | layer-θ_min p1 / p5 / median (°) | layers <30° / <45° | pairs where gate fires |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| R0 (shared seed 0) | 91 | 0.05742 | 0.00769 | 0.05585 | 0.04457 | 0.09433 | [51.6, 166.0] | 1.00 | 0.0136 | 13.9 | 31.2 / 42.3 / 69.5 | 138 / 1116 of 15288 | 78 |
| S1 (shared seed 1) | 91 | 0.06113 | 0.00960 | 0.06095 | 0.04245 | 0.10403 | [46.7, 188.3] | 1.00 | 0.0147 | 16.5 | 30.6 / 40.6 / 68.4 | 126 / 1386 of 15288 | 78 |
| P (mixed seed) | 91 | 0.04967 | 0.00458 | 0.04921 | 0.04085 | 0.06447 | [43.0, 97.3] | 1.00 | 0.0009 | 21.1 | 35.8 / 45.0 / 71.0 | 72 / 771 of 15288 | 66 |
| S2 (same task, mixed seed) | 14 | 0.07150 | 0.00485 | 0.07161 | 0.06345 | 0.07863 | [95.0, 129.9] | 1.00 | 0.0178 | 22.0 | 32.2 / 39.3 / 64.0 | 15 / 263 of 2352 | 13 |

paired_P_minus_R0: n=91, mean diff -0.00775, P lower in 100% of task pairs, ratio of means 0.865, Wilcoxon p = 1.19e-16

paired_P_minus_S1: n=91, mean diff -0.01146, P lower in 100% of task pairs, ratio of means 0.813, Wilcoxon p = 1.19e-16

θ_min by layer type (all populations): mlp.down_proj: min 16.5°, median 74.1°; mlp.gate_proj: min 48.2°, median 85.2°; mlp.up_proj: min 33.7°, median 81.3°; self_attn.k_proj: min 17.8°, median 48.6°; self_attn.o_proj: min 24.1°, median 74.7°; self_attn.q_proj: min 40.8°, median 67.9°; self_attn.v_proj: min 13.9°, median 60.1°

### S2: same-task seed pairs (t@s0 × t@s1)

Gate (θ★ = 30°) fires in **13/14** same-task pairs; min θ_min over all S2 pairs/layers = 21.96°.

| task | min θ_min (°) | argmin layer | #layers < 30° | #layers < 45° | O_A | null z | tv cos | λ* | D | TA@λ* | gate@λ*TA | gate@own |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cola | 26.49 | model.layers.0.self_attn.k_proj | 1 | 13 | 0.06345 | 95.0 | 0.0143 | 0.5 | -0.0118 | 1.0118 | 1.0124 | 1.0124 |
| sst2 | 28.43 | model.layers.0.self_attn.k_proj | 2 | 18 | 0.07863 | 129.9 | 0.0174 | 1.0 | 0.0079 | 0.9921 | 0.9921 | 0.9921 |
| mrpc | 25.87 | model.layers.0.self_attn.k_proj | 1 | 17 | 0.06941 | 108.7 | 0.0145 | 0.7 | 0.0056 | 0.9944 | 1.0015 | 1.0015 |
| stsb | 27.49 | model.layers.0.self_attn.k_proj | 1 | 20 | 0.07842 | 129.4 | 0.0273 | 0.7 | 0.0045 | 0.9955 | 0.9955 | 0.9955 |
| mnli | 24.59 | model.layers.19.self_attn.k_proj | 1 | 21 | 0.07377 | 118.7 | 0.0276 | 1.0 | 0.0770 | 0.9230 | 0.9237 | 0.9237 |
| qnli | 27.72 | model.layers.21.self_attn.k_proj | 1 | 25 | 0.07610 | 124.1 | 0.0203 | 0.7 | 0.0194 | 0.9806 | 0.9812 | 0.9812 |
| rte | 22.01 | model.layers.0.self_attn.k_proj | 1 | 22 | 0.06720 | 103.6 | 0.0073 | 1.0 | 0.0538 | 0.9462 | 0.9510 | 0.9510 |
| wic | 25.58 | model.layers.0.self_attn.k_proj | 1 | 17 | 0.06473 | 97.9 | 0.0159 | 0.7 | 0.0285 | 0.9715 | 0.9737 | 0.9737 |
| snli | 31.30 | model.layers.16.self_attn.k_proj | 0 | 16 | 0.07465 | 120.7 | 0.0180 | 1.0 | 0.1587 | 0.8413 | 0.8413 | 0.8413 |
| scitail | 28.50 | model.layers.0.self_attn.k_proj | 2 | 21 | 0.07293 | 116.8 | 0.0119 | 0.7 | -0.0004 | 1.0004 | 0.9988 | 0.9988 |
| ag_news | 26.15 | model.layers.0.self_attn.k_proj | 1 | 16 | 0.06894 | 107.6 | 0.0184 | 0.7 | 0.0100 | 0.9900 | 0.9902 | 0.9902 |
| imdb | 28.80 | model.layers.0.self_attn.k_proj | 1 | 20 | 0.07030 | 110.7 | 0.0179 | 0.7 | 0.0079 | 0.9921 | 0.9919 | 0.9919 |
| trec | 21.96 | model.layers.0.self_attn.k_proj | 1 | 12 | 0.06757 | 104.5 | 0.0175 | 0.7 | 0.0031 | 0.9969 | 0.9959 | 0.9959 |
| yelp_polarity | 28.02 | model.layers.0.self_attn.k_proj | 1 | 25 | 0.07489 | 121.3 | 0.0213 | 0.7 | 0.0017 | 0.9983 | 0.9983 | 0.9983 |

E1b-grid gate@lam*TA − TA: mean +0.095 pp, CI [-0.006, +0.221] pp, sign-flip p = 0.159, W/T/L 9/2/3

E1b-grid gate@own − TA: mean +0.095 pp, CI [-0.006, +0.221] pp, sign-flip p = 0.159, W/T/L 9/2/3

**E3 code (amendment-A2 grids) on S2, 14 pairs.** Mean normalized score: TA 0.9781, PICO_TA 0.9760, GATE 0.9782, FORCEGATE 0.9790, TA4_score 0.9739, PICO_TA_c1_score 0.9621.

| contrast | mean (pp) | 95% CI over tasks (pp) | sign-flip p (2-sided) | W/T/L |
|---|---:|---|---:|---|
| GATE-TA | +0.008 | [-0.148, +0.159] | 0.8786 | 7/1/6 |
| FORCEGATE-TA | +0.081 | [-0.190, +0.362] | 0.5884 | 7/0/7 |
| PICO_TA-TA | -0.217 | [-0.696, +0.170] | 0.4395 | 7/0/7 |
| GATE-PICO_TA | +0.225 | [-0.135, +0.687] | 0.4197 | 6/0/8 |
| FORCEGATE-PICO_TA | +0.297 | [-0.132, +0.774] | 0.2544 | 11/0/3 |
| PICO_TA_c1_score-TA | -1.600 | [-3.339, -0.372] | 0.0149 | 5/0/9 |
| TA4_score-TA | -0.429 | [-1.153, +0.134] | 0.4601 | 4/6/4 |

Gate vs TA on S2 pairs with ≥ 1 FAIL layer: n = 13; per pair (pp): cola -0.704, sst2 -0.060, mrpc +0.140, stsb -0.017, mnli +0.047, qnli +0.054, rte +0.698, wic +0.229, scitail -0.159, ag_news +0.021, imdb -0.042, trec -0.102, yelp_polarity +0.010; mean +0.009 pp, sign-flip p = 0.886.

Selected-config boundary rate: {'TA': 0.07142857142857142, 'PICO_TA': 0.14285714285714285, 'GATE': 0.07142857142857142, 'FORCEGATE': 0.07142857142857142}. Sanity (E3 TA at E1b λs = stage-2 values): max |Δ| = 0.00e+00.


### Method comparison (E1b stage-2 exploratory methods)

**P**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9594 | — | — | — | — |
| TIES-lite@lam*TA | 0.9293 | -0.0301 | [-0.0436, -0.0188] | 9/0/82 | 2.79e-15 |
| TIES-lite@own-lam | 0.9389 | -0.0206 | [-0.0350, -0.0092] | 17/0/74 | 8.41e-12 |
| gate30@lam*TA | 0.9594 | -0.0000 | [-0.0006, +0.0004] | 23/32/36 | 0.751 |
| gate30@own-lam | 0.9593 | -0.0001 | [-0.0006, +0.0003] | 22/32/37 | 0.497 |

**R0**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9587 | — | — | — | — |
| TIES-lite@lam*TA | 0.9204 | -0.0382 | [-0.0628, -0.0203] | 11/0/80 | 1.82e-14 |
| TIES-lite@own-lam | 0.9521 | -0.0065 | [-0.0141, -0.0003] | 39/1/51 | 0.0193 |
| gate30@lam*TA | 0.9575 | -0.0011 | [-0.0045, +0.0006] | 38/17/36 | 0.676 |
| gate30@own-lam | 0.9590 | +0.0003 | [-0.0017, +0.0032] | 40/16/35 | 0.788 |

**S1**

| method | mean norm. score | mean diff vs TA | task-block 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---|---|---:|
| TA@lam* | 0.9547 | — | — | — | — |
| TIES-lite@lam*TA | 0.9296 | -0.0251 | [-0.0465, -0.0111] | 10/0/81 | 1.77e-14 |
| TIES-lite@own-lam | 0.9432 | -0.0116 | [-0.0310, -0.0008] | 32/0/59 | 0.000125 |
| gate30@lam*TA | 0.9547 | -0.0001 | [-0.0005, +0.0004] | 28/20/43 | 0.259 |
| gate30@own-lam | 0.9548 | +0.0001 | [-0.0009, +0.0010] | 30/19/42 | 0.497 |

### Cross-backbone (exploratory): Spearman over common task pairs

- D: E1b R0 (BERT) ~ e4b R0: ρ = 0.375 (n = 91)
- O_A: E1b R0 (BERT) ~ e4b R0: ρ = 0.187 (n = 91)
- tv_cosine: E1b R0 (BERT) ~ e4b R0: ρ = 0.546 (n = 91)
- D: E1c P (BERT) ~ e4b P: ρ = 0.430 (n = 91)
- O_A: E1c P (BERT) ~ e4b P: ρ = 0.133 (n = 91)
- tv_cosine: E1c P (BERT) ~ e4b P: ρ = 0.610 (n = 91)
- D: E4a R0 (RoBERTa) ~ e4b R0: ρ = 0.246 (n = 78)
- O_A: E4a R0 (RoBERTa) ~ e4b R0: ρ = 0.265 (n = 78)
- tv_cosine: E4a R0 (RoBERTa) ~ e4b R0: ρ = 0.323 (n = 78)
- D: E4a P (RoBERTa) ~ e4b P: ρ = 0.312 (n = 78)
- O_A: E4a P (RoBERTa) ~ e4b P: ρ = 0.211 (n = 78)
- tv_cosine: E4a P (RoBERTa) ~ e4b P: ρ = 0.307 (n = 78)

## Integrity (stage0, per seed; rule (b) not applicable)

| task | seed | lr | steps | micro-bs | train loss (mean) | last logged loss | prior entropy | eval | (a) thr | (a) | (c) max diff / tol | (c2) hold bf16 / fp32 | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| cola | 0 | 0.0003 | 2360 | 32 | 0.2147 | 0.0045 | 0.6046 | 0.8082 | 0.7913 | True | 6.6e-05 / 1.0e-02 True | 0.8190 / 0.8170 True | **True** |
| sst2 | 0 | 0.0003 | 1563 | 32 | 0.3182 | 0.1216 | 0.6867 | 0.9323 | 0.6092 | True | 1.2e-05 / 5.6e-03 True | 0.9560 / 0.9560 True | **True** |
| mrpc | 0 | 0.0003 | 840 | 32 | 0.2518 | 0.0000 | 0.6267 | 0.8578 | 0.7838 | True | 9.3e-05 / 1.3e-02 True | 0.8540 / 0.8510 True | **True** |
| stsb | 0 | 0.0003 | 1490 | 32 | 4.1404 | 0.0452 | nan | 0.9082 | 0.7000 | True | 5.2e-06 / 4.9e-03 True | 0.8859 / 0.8861 True | **True** |
| mnli | 0 | 0.0003 | 1563 | 16 | 1.2820 | 0.8434 | 1.0986 | 0.8486 | 0.4560 | True | 1.6e-05 / 6.0e-03 True | 0.8450 / 0.8430 True | **True** |
| qnli | 0 | 0.0003 | 1563 | 16 | 0.9586 | 0.4501 | 0.6931 | 0.9190 | 0.6056 | True | 6.4e-06 / 5.0e-03 True | 0.9030 / 0.9050 True | **True** |
| rte | 0 | 0.0003 | 470 | 16 | 0.8793 | 0.0000 | 0.6931 | 0.7690 | 0.5729 | True | 1.4e-04 / 1.8e-02 True | 0.7830 / 0.7830 True | **True** |
| wic | 0 | 0.0003 | 1390 | 32 | 0.3104 | 0.0000 | 0.6931 | 0.6834 | 0.6000 | True | 2.0e-04 / 1.2e-02 True | 0.7700 / 0.7670 True | **True** |
| snli | 0 | 0.0003 | 1563 | 32 | 0.5510 | 0.3592 | 1.0986 | 0.8778 | 0.4328 | True | 1.2e-05 / 7.5e-03 True | 0.8650 / 0.8660 True | **True** |
| scitail | 0 | 0.0003 | 1382 | 16 | 0.6524 | 0.0709 | 0.6571 | 0.9594 | 0.5962 | True | 4.0e-05 / 7.1e-03 True | 0.9730 / 0.9730 True | **True** |
| ag_news | 0 | 0.0003 | 1563 | 16 | 0.6805 | 0.3385 | 1.3863 | 0.9418 | 0.3494 | True | 1.4e-04 / 2.2e-02 True | 0.9480 / 0.9480 True | **True** |
| imdb | 0 | 0.0003 | 1500 | 16 | 0.6112 | 0.2121 | 0.6931 | 0.9464 | 0.5932 | True | 1.2e-05 / 5.9e-03 True | 0.9230 / 0.9250 True | **True** |
| trec | 0 | 0.0003 | 1400 | 32 | 0.1661 | 0.0000 | 1.6528 | 0.9760 | 0.2880 | True | 4.6e-05 / 3.4e-02 True | 0.9580 / 0.9580 True | **True** |
| yelp_polarity | 0 | 0.0003 | 1563 | 16 | 0.5096 | 0.2032 | 0.6931 | 0.9712 | 0.6052 | True | 2.5e-05 / 8.4e-03 True | 0.9770 / 0.9770 True | **True** |
| cola | 1 | 0.0003 | 2360 | 32 | 0.1840 | 0.0054 | 0.6046 | 0.8245 | 0.7913 | True | 1.1e-04 / 1.0e-02 True | 0.8210 / 0.8200 True | **True** |
| sst2 | 1 | 0.0003 | 1563 | 32 | 0.2358 | 0.1492 | 0.6867 | 0.9415 | 0.6092 | True | 1.2e-05 / 5.4e-03 True | 0.9470 / 0.9470 True | **True** |
| mrpc | 1 | 0.0003 | 840 | 32 | 0.2081 | 0.0000 | 0.6267 | 0.8725 | 0.7838 | True | 1.9e-04 / 1.6e-02 True | 0.8530 / 0.8560 True | **True** |
| stsb | 1 | 0.0003 | 1490 | 32 | 0.5434 | 0.0444 | nan | 0.9082 | 0.7000 | True | 7.2e-06 / 5.0e-03 True | 0.8855 / 0.8865 True | **True** |
| mnli | 1 | 0.0003 | 1563 | 16 | 1.1100 | 0.7264 | 1.0986 | 0.8432 | 0.4560 | True | 2.3e-05 / 7.6e-03 True | 0.8450 / 0.8480 True | **True** |
| qnli | 1 | 0.0003 | 1563 | 16 | 0.7068 | 0.5299 | 0.6931 | 0.9226 | 0.6056 | True | 1.4e-05 / 4.3e-03 True | 0.9010 / 0.9060 True | **True** |
| rte | 1 | 0.0003 | 470 | 16 | 0.4382 | 0.0006 | 0.6931 | 0.7762 | 0.5729 | True | 2.0e-04 / 1.7e-02 True | 0.7470 / 0.7510 True | **True** |
| wic | 1 | 0.0003 | 1390 | 32 | 0.2295 | 0.0000 | 0.6931 | 0.6928 | 0.6000 | True | 1.8e-04 / 1.1e-02 True | 0.7710 / 0.7700 True | **True** |
| snli | 1 | 0.0003 | 1563 | 32 | 0.4803 | 0.3484 | 1.0986 | 0.8906 | 0.4328 | True | 1.9e-05 / 8.1e-03 True | 0.8610 / 0.8650 True | **True** |
| scitail | 1 | 0.0003 | 1382 | 16 | 0.3501 | 0.1626 | 0.6571 | 0.9663 | 0.5962 | True | 2.0e-05 / 8.5e-03 True | 0.9750 / 0.9760 True | **True** |
| ag_news | 1 | 0.0003 | 1563 | 16 | 0.6132 | 0.3503 | 1.3863 | 0.9408 | 0.3494 | True | 3.4e-05 / 1.4e-02 True | 0.9480 / 0.9480 True | **True** |
| imdb | 1 | 0.0003 | 1500 | 16 | 0.4592 | 0.2808 | 0.6931 | 0.9444 | 0.5932 | True | 1.6e-05 / 5.7e-03 True | 0.9260 / 0.9260 True | **True** |
| trec | 1 | 0.0003 | 1400 | 32 | 0.1559 | 0.0000 | 1.6528 | 0.9780 | 0.2880 | True | 5.7e-05 / 3.4e-02 True | 0.9580 / 0.9590 True | **True** |
| yelp_polarity | 1 | 0.0003 | 1563 | 16 | 0.3190 | 0.1665 | 0.6931 | 0.9718 | 0.6052 | True | 2.2e-05 / 7.4e-03 True | 0.9750 / 0.9750 True | **True** |

## Deviations and notes

# E4b — deviations and notes (append-only; KST)

Pre-registration: `PREREG_E4B.md` / `prereg_e4b.json`, hashed 2026-09-26 10:02:03 KST (`prereg_e4b.sha256`). No E4b GPU work happened before E4a finished (E4a merges ended 15:01:14).

## Before the pilot (smoke test in a scratch copy `~/e4_smoke/e4b_decoder`, 20-step throwaway adapters, 15:02–15:30; deleted afterwards; no study numbers)
- **D1 (15:05, code: smoke-only path).** `train_e4b.py` smoke mode used `recipe.lr = null` (the lr comes from the pilot) and crashed. Smoke mode now sets lr = 3e-4 for the throwaway
  adapters. The pilot and main paths are unchanged.
- **D2 (15:30, execution; within the prereg).** At micro-batch 32, RTE ran out of memory (15.5 GB in use at ≤ 256 tokens). Qwen activations with fp32 master weights take ≈ 2.3 MB/token.
  The **pre-registered OOM fallback** (halve the micro-batch, double the accumulation; effective batch 32) handled it automatically: rte ran at micro-batch 16, peak 9.1 GB.
  Measured 20-step peaks: sst2 5.9 GB, stsb 8.0 GB, trec 4.4 GB, imdb (micro-batch 16) 9.7 GB.
  Two concurrent training processes would not fit reliably, so **main training runs as a single GPU process** (the prereg allows at most 2). The fallback may therefore apply to other long-sequence tasks
  (e.g. qnli, mnli, ag_news, scitail). It is logged per adapter (`train_meta.json: micro_batch, micro_div`; failed attempts in `logs/*_fail_div*.log`).
- **D3 (15:30, pipeline orchestration only; no computation changed).** The E3 subset (`e3.py run()`, unmodified) needs ≈ 12 GB for Qwen. With `--mem-frac 0.5` it ran out of memory. A stage-2
  process peaks at ≈ 6.7 GB. So `run_pipeline_e4b.sh` now uses a phased schedule that keeps the pre-registered priority P → S2+E3 → R0 → S1:
  - phase 1: P ∥ (S2, then R0 in reverse order, stopped through a stop-file once P is done);
  - phase 2: E3 subset alone on the GPU (`--mem-frac 0.9`);
  - phase 3: R0 → S1 in both directions.
  `e4b.past_deadline()` also honors the stop-file (`E4B_STOPFILE`). All code was re-hashed afterwards in `code_e4b.sha256` (before the pilot).
- Smoke results (pipeline only): (c) max logit diffs 2e-5–8e-5, far below tolerance; (c2) within 0.5 pp. All stages ran: pilot rule, stage0, stage1, stage2 P/S2/R0/S1, forced-gate test with θ★ = 89° (168 layers gated),
  E3 subset, stage3, verdict. The analysis-code equivalence check on E1b reproduced E1b exactly.

## Pilot (pre-registered pre-step; 15:29:24–15:44:53 KST)
- Outcome, applied under the pre-registered rule (`pilot/pilot_decision.json`, sha256 in `pilot/pilot_decision.sha256`, written 15:44:52):
  - Held-out accuracy (sst2 / rte) by lr:
    - 1e-4: 0.942 / 0.735
    - 3e-4: 0.956 / 0.783
    - 1e-3: 0.920 / 0.746
  - None of the lrs diverged.
  - **lr_main = 3e-4.**
  - Projected core compute for 2 seeds = 5.52 h (≤ 11 h threshold), so we use **seeds [0, 1]**, and the **primary population is P** (mixed seed).
- rte hit the expected OOM at micro-batch 32 in every pilot run (see D2). Each time it fell back to micro-batch 16 (rc=1 then rc=0 in `pilot.log`). Its peak was 9.7 GB.
- **N1 (15:46:56, operational; no computation).** The first main-training launch passed the 28 queue items as a single argument, because the remote login shell (zsh) does not word-split.
  `train_e4b.py` exited at argument parsing with rc=1. No adapter or file was created. The line stays in `train_e4b.log`.
  We relaunched correctly at **15:47:19** (`train_launch_time.txt`), with one queue Q1 containing the 14 tasks @s0 followed by the 14 tasks @s1. `run_pipeline_e4b.sh` started at 15:46:58 and is waiting for the adapters.

## Main run
- Training ran 15:47:19–18:00:30 KST as a single queue (2.22 h, both seeds).
  - Every adapter finished on its first successful attempt. The pre-registered micro-batch fallback (div 2) was used for mnli, qnli, rte, scitail and ag_news in both seeds; imdb and yelp use micro-batch 16 by design.
  - No collapse. The sst2@s0 main run reproduced the pilot lr=3e-4 run exactly (same loss trajectory, held-out 0.956).
- Stage0 (18:05:43): **all 14 tasks valid in both seeds** under (a), (c) and (c2). There were no exclusions. Population sizes: P 91, R0 91, S1 91, S2 14.
- Stage1 froze the predictors at 18:09:32 (`predictors_e4b.sha256`).
- **D4 (18:21, orchestration only; no computation changed).** In phase 1, the stage-2 P process reached ≈ 13 GB with the real adapters and evaluation sets (the smoke test had measured ≈ 6.7 GB).
  - Process B (S2, then R0 in reverse) therefore ran out of memory 3×3 times between 18:09 and 18:13. The first S2 pair (`cola@s0__cola@s1`) completed and is kept (stage 2 is resumable per pair). Every other attempt failed while loading or merging and wrote nothing.
  - Unchanged, the original runner would never run S2 and would start two concurrent processes in phase 3. So its parent (pid 2026925) was stopped at 18:21:22. The P subshell kept running untouched.
  - `run_pipeline_e4b_serial.sh` (sha256 `89e2ffa0…`, in `run_pipeline_e4b_serial.sha256`) takes over once P finishes (stop-file). It re-checks the prereg, code and predictor hashes, then runs **one GPU process at a time**: S2 → E3 subset → R0 → S1 → stage3.
  - This keeps the pre-registered priority P → S2+E3 → R0 → S1 and the same budget deadline (pilot start + 13.6 h). All merge, evaluation and analysis code is unchanged.
- Observation (reported in secondary results): unlike BERT and RoBERTa, Qwen2.5-0.5B cross-task adapter pairs have layers with θ_min < 30°. Across 48,216 pair-layers, 351 have θ_min < 30° (minimum 13.9°), in 235 of 287 pairs. They are concentrated in `layers.0.self_attn.k_proj` (d_out = 128 under GQA) and in layer-23 v_proj/down_proj. So the pre-registered 30° gate fires in 1 layer for most pairs.


## Compute

Timing files (s): {"timing_R0": {"stage2_gpu_wall_s": 5314.1}, "timing_S1": {"stage2_gpu_wall_s": 5203.0}, "timing_P": {"stage2_gpu_wall_s": 5005.9}, "timing_main": {"stage0_s": 284.0, "stage1_s": 227.7}, "timing_S2": {"stage2_gpu_wall_s": 850.0, "e3same_s": 1467.7}}
