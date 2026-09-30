# Pooled Spearman rho across R0 (E1b), P and S1 (E1c) — EXPLORATORY

Generated 2026-09-26 09:19:19 KST by `analysis/pooled_rho.py`. Not pre-registered; written after all E1b/E1c results. The three populations share all 14 tasks, the evaluation sets and, for P, the adapters themselves (P's t1 adapters are R0 adapters, P's t2 adapters are S1 adapters), so they are not independent. The bootstrap resamples tasks jointly across populations (2000 replicates).

| predictor | ρ R0 | ρ P | ρ S1 | **ρ̄ (mean)** | joint task-block 95% CI | joint perm. p (1-sided / 2-sided) | Fisher-z mean | LOTO ρ̄ range | ρ(pair-mean predictor, pair-mean D) [95% CI] |
|---|---:|---:|---:|---:|---|---|---:|---|---|
| O_A | -0.194 | -0.025 | -0.029 | **-0.083** | [-0.460, 0.314] | 0.707 / 0.597 | -0.083 | [-0.193, 0.096] | -0.098 [-0.494, 0.338] |
| tv_cosine | -0.044 | 0.127 | 0.057 | **0.047** | [-0.308, 0.347] | 0.341 / 0.679 | 0.047 | [-0.031, 0.111] | 0.064 [-0.366, 0.442] |
| mean_theta_min_A_deg | 0.140 | -0.001 | 0.011 | **0.050** | [-0.314, 0.431] | 0.352 / 0.719 | 0.050 | [-0.095, 0.147] | 0.065 [-0.350, 0.488] |
| min_theta_min_A_deg | 0.136 | 0.059 | 0.067 | **0.087** | [-0.173, 0.358] | 0.196 / 0.405 | 0.087 | [0.046, 0.159] | 0.070 [-0.292, 0.432] |
| sign_conflict_top20 | 0.048 | -0.152 | -0.044 | **-0.049** | [-0.333, 0.302] | 0.666 / 0.665 | -0.050 | [-0.101, 0.029] | -0.042 [-0.430, 0.378] |
| sign_conflict_all | 0.061 | -0.050 | 0.004 | **0.005** | [-0.288, 0.340] | 0.479 / 0.963 | 0.005 | [-0.080, 0.098] | -0.063 [-0.429, 0.346] |
| norm_ratio | 0.111 | 0.211 | 0.189 | **0.171** | [-0.155, 0.441] | 0.084 / 0.163 | 0.171 | [0.077, 0.252] | 0.174 [-0.213, 0.517] |

Sanity check (per-population CIs from the joint draws vs the pre-registered analysis files; max |Δ|): {"R0:O_A": 0.0, "R0:tv_cosine": 0.0, "P:O_A": 0.0, "P:tv_cosine": 0.0, "S1:O_A": 0.0, "S1:tv_cosine": 0.0}
