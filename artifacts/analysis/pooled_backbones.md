# Per-backbone and across-backbone pooled Spearman ρ — EXPLORATORY (all values derived)

Generated 2026-09-27 00:04:03 KST by `analysis/pooled_backbones.py`. Not pre-registered; written after all E1b, E1c, E4a and E4b results were known. Within a backbone the three populations (R0 seed 0, P mixed seed, S1 seed 1) share tasks, evaluation sets and adapters, so they are not independent; across backbones 13 of 14 tasks and all evaluation examples are shared, so the backbones are not independent either.

## 1. Per-backbone pooled ρ̄ (mean of R0, P, S1; joint within-backbone task-block bootstrap, 2,000 reps)

| backbone | K | predictor | ρ R0 | ρ P | ρ S1 | **ρ̄** | 95% CI | joint perm. p (1-sided / 2-sided) | LOTO ρ̄ range |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| BERT | 14 | O_A | -0.194 | -0.025 | -0.029 | **-0.083** | [-0.460, +0.314] | 0.707 / 0.597 | [-0.193, +0.096] |
| BERT | 14 | tv_cosine | -0.044 | +0.127 | +0.057 | **+0.047** | [-0.308, +0.347] | 0.341 / 0.679 | [-0.031, +0.111] |
| BERT | 14 | mean_theta_min_A_deg | +0.140 | -0.001 | +0.011 | **+0.050** | [-0.314, +0.431] | 0.352 / 0.719 | [-0.095, +0.147] |
| BERT | 14 | min_theta_min_A_deg | +0.136 | +0.059 | +0.067 | **+0.087** | [-0.173, +0.358] | 0.196 / 0.405 | [+0.046, +0.159] |
| BERT | 14 | sign_conflict_top20 | +0.048 | -0.152 | -0.044 | **-0.049** | [-0.333, +0.302] | 0.666 / 0.665 | [-0.101, +0.029] |
| BERT | 14 | sign_conflict_all | +0.061 | -0.050 | +0.004 | **+0.005** | [-0.288, +0.340] | 0.479 / 0.963 | [-0.080, +0.098] |
| BERT | 14 | norm_ratio | +0.111 | +0.211 | +0.189 | **+0.171** | [-0.155, +0.441] | 0.084 / 0.163 | [+0.077, +0.252] |
| RoBERTa | 13 | O_A | +0.146 | +0.161 | +0.037 | **+0.115** | [-0.331, +0.528] | 0.269 / 0.540 | [+0.006, +0.264] |
| RoBERTa | 13 | tv_cosine | +0.130 | +0.112 | +0.038 | **+0.093** | [-0.350, +0.464] | 0.287 / 0.578 | [+0.002, +0.207] |
| RoBERTa | 13 | mean_theta_min_A_deg | -0.117 | -0.155 | -0.041 | **-0.104** | [-0.507, +0.330] | 0.715 / 0.574 | [-0.247, +0.005] |
| RoBERTa | 13 | min_theta_min_A_deg | -0.242 | -0.299 | -0.184 | **-0.242** | [-0.477, +0.113] | 0.888 / 0.226 | [-0.332, -0.159] |
| RoBERTa | 13 | sign_conflict_top20 | -0.154 | -0.141 | -0.057 | **-0.117** | [-0.493, +0.321] | 0.758 / 0.488 | [-0.228, -0.023] |
| RoBERTa | 13 | sign_conflict_all | -0.047 | -0.088 | +0.033 | **-0.034** | [-0.416, +0.382] | 0.584 / 0.835 | [-0.128, +0.054] |
| RoBERTa | 13 | norm_ratio | +0.249 | +0.231 | +0.395 | **+0.292** | [-0.028, +0.555] | 0.025 / 0.052 | [+0.212, +0.368] |
| Qwen | 14 | O_A | +0.366 | +0.345 | +0.211 | **+0.308** | [-0.073, +0.549] | 0.035 / 0.065 | [+0.170, +0.381] |
| Qwen | 14 | tv_cosine | +0.395 | +0.368 | +0.105 | **+0.289** | [-0.117, +0.573] | 0.036 / 0.070 | [+0.151, +0.366] |
| Qwen | 14 | mean_theta_min_A_deg | -0.369 | -0.333 | -0.184 | **-0.295** | [-0.567, +0.102] | 0.953 / 0.088 | [-0.369, -0.139] |
| Qwen | 14 | min_theta_min_A_deg | +0.074 | +0.052 | -0.014 | **+0.038** | [-0.169, +0.262] | 0.321 / 0.640 | [-0.030, +0.071] |
| Qwen | 14 | sign_conflict_top20 | -0.450 | -0.387 | -0.133 | **-0.323** | [-0.598, +0.101] | 0.981 / 0.037 | [-0.400, -0.197] |
| Qwen | 14 | sign_conflict_all | -0.253 | -0.289 | -0.061 | **-0.201** | [-0.486, +0.184] | 0.892 / 0.209 | [-0.265, -0.038] |
| Qwen | 14 | norm_ratio | +0.271 | +0.234 | +0.209 | **+0.238** | [-0.177, +0.560] | 0.089 / 0.174 | [+0.141, +0.373] |

Sanity (per-population CIs and one-sided permutation p from the joint draws vs the pre-registered analysis files; max |Δ|): {"BERT": {"max_abs_diff_ci": 0.0, "max_abs_diff_perm_p": 0.0, "n_checked_ci": 6, "n_checked_p": 6}, "RoBERTa": {"max_abs_diff_ci": 0.0, "max_abs_diff_perm_p": 0.0, "n_checked_ci": 21, "n_checked_p": 21}, "Qwen": {"max_abs_diff_ci": 0.0, "max_abs_diff_perm_p": 0.0, "n_checked_ci": 21, "n_checked_p": 21}}

## 2. Across-backbone summary and heterogeneity

Joint draws over the 14-task union: 2000 replicates used (0 skipped). Independent within-backbone draws: 2000 replicates.

| predictor | quantity | observed | 95% CI (joint task draws) | approx. 2-sided boot p (joint) | 95% CI (independent draws) | approx. 2-sided boot p (indep.) |
|---|---|---:|---|---:|---|---:|
| O_A | across_mean | +0.113 | [-0.210, +0.399] | 0.479 | [-0.128, +0.320] | 0.356 |
| O_A | encoder_mean | +0.016 | [-0.376, +0.384] | 0.931 | [-0.281, +0.310] | 0.914 |
| O_A | Qwen_minus_BERT | +0.390 | [-0.141, +0.840] | 0.155 | [-0.154, +0.846] | 0.174 |
| O_A | Qwen_minus_RoBERTa | +0.193 | [-0.320, +0.629] | 0.435 | [-0.366, +0.708] | 0.505 |
| O_A | Qwen_minus_encoder_mean | +0.292 | [-0.212, +0.703] | 0.253 | [-0.175, +0.675] | 0.228 |
| O_A | RoBERTa_minus_BERT | +0.197 | [-0.066, +0.449] | 0.152 | [-0.401, +0.759] | 0.532 |
| tv_cosine | across_mean | +0.143 | [-0.176, +0.397] | 0.364 | [-0.090, +0.321] | 0.228 |
| tv_cosine | encoder_mean | +0.070 | [-0.290, +0.380] | 0.707 | [-0.207, +0.310] | 0.627 |
| tv_cosine | Qwen_minus_BERT | +0.242 | [-0.113, +0.596] | 0.172 | [-0.255, +0.681] | 0.339 |
| tv_cosine | Qwen_minus_RoBERTa | +0.196 | [-0.259, +0.591] | 0.370 | [-0.369, +0.675] | 0.485 |
| tv_cosine | Qwen_minus_encoder_mean | +0.219 | [-0.163, +0.559] | 0.221 | [-0.245, +0.616] | 0.345 |
| tv_cosine | RoBERTa_minus_BERT | +0.046 | [-0.264, +0.368] | 0.760 | [-0.467, +0.565] | 0.889 |
| mean_theta_min_A_deg | across_mean | -0.116 | [-0.393, +0.204] | 0.455 | [-0.321, +0.127] | 0.339 |
| mean_theta_min_A_deg | encoder_mean | -0.027 | [-0.381, +0.371] | 0.892 | [-0.304, +0.265] | 0.864 |
| mean_theta_min_A_deg | Qwen_minus_BERT | -0.345 | [-0.807, +0.177] | 0.194 | [-0.809, +0.197] | 0.210 |
| mean_theta_min_A_deg | Qwen_minus_RoBERTa | -0.191 | [-0.623, +0.322] | 0.414 | [-0.705, +0.371] | 0.500 |
| mean_theta_min_A_deg | Qwen_minus_encoder_mean | -0.268 | [-0.692, +0.228] | 0.272 | [-0.664, +0.199] | 0.264 |
| mean_theta_min_A_deg | RoBERTa_minus_BERT | -0.155 | [-0.387, +0.113] | 0.216 | [-0.704, +0.420] | 0.623 |
| min_theta_min_A_deg | across_mean | -0.039 | [-0.194, +0.147] | 0.679 | [-0.183, +0.133] | 0.617 |
| min_theta_min_A_deg | encoder_mean | -0.077 | [-0.277, +0.192] | 0.475 | [-0.260, +0.154] | 0.445 |
| min_theta_min_A_deg | Qwen_minus_BERT | -0.050 | [-0.378, +0.215] | 0.837 | [-0.402, +0.298] | 0.827 |
| min_theta_min_A_deg | Qwen_minus_RoBERTa | +0.280 | [-0.163, +0.627] | 0.202 | [-0.106, +0.611] | 0.136 |
| min_theta_min_A_deg | Qwen_minus_encoder_mean | +0.115 | [-0.227, +0.391] | 0.445 | [-0.185, +0.401] | 0.437 |
| min_theta_min_A_deg | RoBERTa_minus_BERT | -0.329 | [-0.632, +0.038] | 0.084 | [-0.679, +0.090] | 0.118 |
| sign_conflict_top20 | across_mean | -0.163 | [-0.415, +0.152] | 0.308 | [-0.338, +0.073] | 0.168 |
| sign_conflict_top20 | encoder_mean | -0.083 | [-0.396, +0.273] | 0.658 | [-0.315, +0.187] | 0.554 |
| sign_conflict_top20 | Qwen_minus_BERT | -0.274 | [-0.640, +0.102] | 0.144 | [-0.708, +0.223] | 0.283 |
| sign_conflict_top20 | Qwen_minus_RoBERTa | -0.206 | [-0.578, +0.244] | 0.349 | [-0.673, +0.342] | 0.477 |
| sign_conflict_top20 | Qwen_minus_encoder_mean | -0.240 | [-0.586, +0.161] | 0.209 | [-0.622, +0.241] | 0.316 |
| sign_conflict_top20 | RoBERTa_minus_BERT | -0.068 | [-0.369, +0.225] | 0.619 | [-0.576, +0.439] | 0.798 |
| sign_conflict_all | across_mean | -0.077 | [-0.329, +0.226] | 0.592 | [-0.260, +0.131] | 0.494 |
| sign_conflict_all | encoder_mean | -0.014 | [-0.320, +0.319] | 0.924 | [-0.257, +0.242] | 0.880 |
| sign_conflict_all | Qwen_minus_BERT | -0.206 | [-0.561, +0.157] | 0.272 | [-0.652, +0.299] | 0.412 |
| sign_conflict_all | Qwen_minus_RoBERTa | -0.167 | [-0.533, +0.260] | 0.406 | [-0.659, +0.372] | 0.546 |
| sign_conflict_all | Qwen_minus_encoder_mean | -0.187 | [-0.509, +0.181] | 0.294 | [-0.585, +0.279] | 0.439 |
| sign_conflict_all | RoBERTa_minus_BERT | -0.039 | [-0.368, +0.295] | 0.828 | [-0.523, +0.475] | 0.876 |
| norm_ratio | across_mean | +0.233 | [+0.022, +0.414] | 0.033 | [+0.034, +0.399] | 0.024 |
| norm_ratio | encoder_mean | +0.231 | [-0.061, +0.444] | 0.121 | [+0.008, +0.420] | 0.042 |
| norm_ratio | Qwen_minus_BERT | +0.067 | [-0.476, +0.547] | 0.738 | [-0.409, +0.490] | 0.771 |
| norm_ratio | Qwen_minus_RoBERTa | -0.054 | [-0.499, +0.455] | 0.767 | [-0.529, +0.406] | 0.817 |
| norm_ratio | Qwen_minus_encoder_mean | +0.007 | [-0.449, +0.485] | 0.958 | [-0.437, +0.398] | 0.962 |
| norm_ratio | RoBERTa_minus_BERT | +0.121 | [-0.187, +0.449] | 0.405 | [-0.305, +0.542] | 0.539 |

Random-effects (DerSimonian–Laird, Fisher z; assumes independent backbones, k = 3; descriptive):

| predictor | fixed-effect ρ | random-effects ρ [95% CI] | τ² (z) | Q (df 2) | Q p | I² |
|---|---:|---|---:|---:|---:|---:|
| O_A | +0.138 | +0.137 [-0.101, +0.360] | 0.0021 | 2.10 | 0.350 | 0.05 |
| tv_cosine | +0.136 | +0.136 [-0.083, +0.343] | 0.0000 | 0.95 | 0.620 | 0.00 |
| mean_theta_min_A_deg | -0.125 | -0.125 [-0.345, +0.108] | 0.0000 | 1.62 | 0.445 | 0.00 |
| min_theta_min_A_deg | -0.010 | -0.018 [-0.196, +0.161] | 0.0077 | 2.85 | 0.241 | 0.30 |
| sign_conflict_top20 | -0.151 | -0.151 [-0.356, +0.067] | 0.0000 | 1.19 | 0.550 | 0.00 |
| sign_conflict_all | -0.072 | -0.072 [-0.275, +0.138] | 0.0000 | 0.73 | 0.695 | 0.00 |
| norm_ratio | +0.232 | +0.232 [+0.040, +0.407] | 0.0000 | 0.31 | 0.856 | 0.00 |

## 3. Layer-exclusion check for O_A (Qwen; RoBERTa for comparison)

**Qwen** (recompute check max |Δ| vs frozen O_A: R0 6.0e-12, P 6.9e-12, S1 3.0e-11)

| O_A variant | layers kept | ρ R0 [CI] | ρ P [CI] (perm p) | ρ S1 [CI] | ρ̄ [joint CI] |
|---|---:|---|---|---|---|
| O_A (all layers; recomputed) | 168 | +0.366 [-0.078, +0.635] | +0.345 [-0.096, +0.609] (0.030) | +0.211 [-0.242, +0.551] | +0.308 [-0.073, +0.549] |
| excl. layer-0 k_proj | 167 | +0.369 [-0.085, +0.627] | +0.354 [-0.096, +0.619] (0.028) | +0.211 [-0.238, +0.553] | +0.311 [-0.072, +0.553] |
| excl. all k_proj | 144 | +0.303 [-0.171, +0.631] | +0.333 [-0.146, +0.634] (0.043) | +0.139 [-0.326, +0.504] | +0.258 [-0.142, +0.541] |
| excl. all d_out = 128 layers (k_proj, v_proj) | 120 | +0.333 [-0.112, +0.629] | +0.373 [-0.077, +0.658] (0.027) | +0.147 [-0.310, +0.518] | +0.284 [-0.108, +0.554] |
| excl. hot spots (L0 k_proj, L23 v_proj, L23 down_proj) | 165 | +0.379 [-0.076, +0.632] | +0.350 [-0.080, +0.613] (0.029) | +0.216 [-0.249, +0.557] | +0.315 [-0.067, +0.561] |
| excl. every layer sub-30 deg in any pair | 150 | +0.411 [-0.032, +0.646] | +0.365 [-0.050, +0.631] (0.018) | +0.232 [-0.211, +0.573] | +0.336 [-0.049, +0.570] |
| excl. pair-specific sub-30 deg layers | pair-specific | +0.390 [-0.070, +0.651] | +0.330 [-0.103, +0.603] (0.040) | +0.221 [-0.200, +0.555] | +0.314 [-0.073, +0.552] |
| only d_out = 896/768 layers | 72 | +0.352 [-0.087, +0.634] | +0.377 [-0.064, +0.665] (0.026) | +0.163 [-0.292, +0.514] | +0.297 [-0.084, +0.553] |
| only d_out = 4864/3072 layers | 48 | +0.251 [-0.200, +0.586] | +0.292 [-0.190, +0.609] (0.075) | +0.072 [-0.403, +0.479] | +0.205 [-0.207, +0.522] |
| chance-normalized O_A (mean cos2 / (r/d_out)) | all | +0.290 [-0.162, +0.612] | +0.345 [-0.134, +0.624] (0.040) | +0.114 [-0.354, +0.492] | +0.250 [-0.158, +0.536] |

**RoBERTa** (recompute check max |Δ| vs frozen O_A: R0 5.1e-12, P 5.4e-12, S1 5.7e-12)

| O_A variant | layers kept | ρ R0 [CI] | ρ P [CI] (perm p) | ρ S1 [CI] | ρ̄ [joint CI] |
|---|---:|---|---|---|---|
| O_A (all layers; recomputed) | 72 | +0.146 [-0.368, +0.601] | +0.161 [-0.342, +0.589] (0.209) | +0.037 [-0.422, +0.500] | +0.115 [-0.331, +0.528] |
| excl. layer-0 k_proj | 71 | +0.141 [-0.371, +0.595] | +0.155 [-0.350, +0.583] (0.217) | +0.030 [-0.426, +0.496] | +0.109 [-0.334, +0.529] |
| excl. all k_proj | 60 | +0.092 [-0.414, +0.549] | +0.097 [-0.382, +0.511] (0.312) | +0.007 [-0.428, +0.461] | +0.065 [-0.368, +0.472] |
| excl. all d_out = 128 layers (k_proj, v_proj) | 72 | +0.146 [-0.368, +0.601] | +0.161 [-0.342, +0.589] (0.209) | +0.037 [-0.422, +0.500] | +0.115 [-0.331, +0.528] |
| excl. hot spots (L0 k_proj, L23 v_proj, L23 down_proj) | 72 | +0.146 [-0.368, +0.601] | +0.161 [-0.342, +0.589] (0.209) | +0.037 [-0.422, +0.500] | +0.115 [-0.331, +0.528] |
| excl. every layer sub-30 deg in any pair | 72 | +0.146 [-0.368, +0.601] | +0.161 [-0.342, +0.589] (0.209) | +0.037 [-0.422, +0.500] | +0.115 [-0.331, +0.528] |
| excl. pair-specific sub-30 deg layers | pair-specific | +0.146 [-0.368, +0.601] | +0.161 [-0.342, +0.589] (0.209) | +0.037 [-0.422, +0.500] | +0.115 [-0.331, +0.528] |
| only d_out = 896/768 layers | 60 | +0.151 [-0.368, +0.597] | +0.163 [-0.350, +0.595] (0.207) | +0.042 [-0.417, +0.504] | +0.119 [-0.325, +0.534] |
| only d_out = 4864/3072 layers | 12 | -0.087 [-0.575, +0.329] | +0.050 [-0.383, +0.405] (0.379) | +0.052 [-0.372, +0.366] | +0.005 [-0.388, +0.316] |
| chance-normalized O_A (mean cos2 / (r/d_out)) | all | +0.131 [-0.383, +0.598] | +0.148 [-0.338, +0.567] (0.228) | +0.030 [-0.424, +0.482] | +0.103 [-0.345, +0.516] |

Qwen sub-30° census: {"n_pair_layers": 48216, "n_sub30": 351, "min_theta": 13.93573047, "n_pairs_with_sub30": 235, "by_population": {"P": 72, "R0": 138, "S1": 126, "S2": 15}, "by_layer_P": {"model.layers.0.self_attn.k_proj": 66, "model.layers.23.self_attn.k_proj": 3, "model.layers.19.self_attn.k_proj": 3}, "share_layer0_kproj_all": 0.5954415954415955, "share_d_out_128_all": 0.9116809116809117}

