# Reliability, noise ceiling and detectability across three backbones — EXPLORATORY (derived)

Generated 2026-09-27 00:15:08 KST by `analysis/noise_ceiling_backbones.py`. Supplements `analysis/NOISE_CEILING.md` (BERT; unchanged).

## 1. Seed test–retest reliability (Spearman over the same unordered task pairs)

| backbone | K | D R0~S1 | D R0~P | D P~S1 | O_A R0~S1 | O_A R0~P | O_A P~S1 | cos R0~S1 | cos R0~P | cos P~S1 | check vs files (max abs diff) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT | 14 | 0.670 | 0.669 | 0.808 | 0.940 | 0.956 | 0.937 | 0.729 | 0.609 | 0.512 | 0.0e+00 |
| RoBERTa | 13 | 0.648 | 0.793 | 0.801 | 0.933 | 0.941 | 0.975 | 0.898 | 0.748 | 0.781 | 0.0e+00 |
| Qwen | 14 | 0.760 | 0.690 | 0.653 | 0.852 | 0.784 | 0.850 | 0.863 | 0.704 | 0.712 | 0.0e+00 |

Evaluation-sampling reliability (BERT only; from `noise_ceiling_results.json`): {"E1b": {"mean_pair_se": 0.009138970637697405, "split_half_r": 0.8770423499776125, "spearman_brown": 0.9344939393488624}, "E1c-mixed": {"mean_pair_se": 0.008984449804455574, "split_half_r": 0.8626201376743408, "spearman_brown": 0.9262437576256471}, "E1c-seed1": {"mean_pair_se": 0.008712283991237066, "split_half_r": 0.8726001689541109, "spearman_brown": 0.9319663464961424}}. RoBERTa/Qwen: not computable on the box: per-example cross-task predictions for E4a/E4b are not mirrored (GPU machine only).

## 2. Attenuation ceiling and disattenuated pooled ρ̄

| backbone | predictor | rel_D (R0~S1) | rel_X (R0~S1) | ceiling | ceiling range (3 comparisons) | expected observed ρ for latent 0.4 | pooled ρ̄ | disattenuated ρ̄ [CI / ceiling] |
|---|---|---:|---:|---:|---|---:|---:|---|
| BERT | O_A | 0.670 | 0.940 | 0.793 | [0.793, 0.870] | 0.317 | -0.083 | -0.104 [-0.579, +0.396] |
| BERT | tv_cosine | 0.670 | 0.729 | 0.699 | [0.639, 0.699] | 0.280 | +0.047 | +0.067 [-0.441, +0.497] |
| RoBERTa | O_A | 0.648 | 0.933 | 0.777 | [0.777, 0.883] | 0.311 | +0.115 | +0.147 [-0.426, +0.679] |
| RoBERTa | tv_cosine | 0.648 | 0.898 | 0.763 | [0.763, 0.791] | 0.305 | +0.093 | +0.122 [-0.459, +0.608] |
| Qwen | O_A | 0.760 | 0.852 | 0.805 | [0.736, 0.805] | 0.322 | +0.308 | +0.382 [-0.091, +0.682] |
| Qwen | tv_cosine | 0.760 | 0.863 | 0.810 | [0.681, 0.810] | 0.324 | +0.289 | +0.357 [-0.144, +0.707] |

## 3. Detectability simulation (single population, pre-registered statistics approximated)

1000 simulated studies per row; task-block bootstrap 500; permutation 1000. PASS (liberal) uses one-sided p < 0.05, PASS (conservative) p < 0.025.

| condition | backbone | predictor | K | target observed ρ | mean observed ρ | mean CI [lower, upper] | P(CI lower > 0) | P(ρ ≥ 0.4) | P(p < 0.05) | P(PASS) lib. / cons. | P(FAIL) | P(INCONCLUSIVE) |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| latent 0.4, attenuated (R0~S1 ceiling) | BERT | O_A | 14 | 0.317 | 0.311 | [-0.032, +0.592] | 0.396 | 0.195 | 0.909 | 0.195 / 0.195 | 0.000 | 0.805 |
| latent 0.4, attenuated (R0~S1 ceiling) | BERT | tv_cosine | 14 | 0.280 | 0.279 | [-0.070, +0.570] | 0.274 | 0.106 | 0.860 | 0.106 / 0.106 | 0.002 | 0.892 |
| latent 0.4, attenuated (R0~S1 ceiling) | RoBERTa | O_A | 13 | 0.311 | 0.304 | [-0.080, +0.614] | 0.259 | 0.172 | 0.867 | 0.172 / 0.172 | 0.000 | 0.828 |
| latent 0.4, attenuated (R0~S1 ceiling) | RoBERTa | tv_cosine | 13 | 0.305 | 0.300 | [-0.085, +0.613] | 0.261 | 0.171 | 0.857 | 0.171 / 0.171 | 0.001 | 0.828 |
| latent 0.4, attenuated (R0~S1 ceiling) | Qwen | O_A | 14 | 0.322 | 0.322 | [-0.023, +0.603] | 0.428 | 0.220 | 0.933 | 0.220 / 0.220 | 0.001 | 0.779 |
| latent 0.4, attenuated (R0~S1 ceiling) | Qwen | tv_cosine | 14 | 0.324 | 0.321 | [-0.024, +0.602] | 0.434 | 0.216 | 0.929 | 0.216 / 0.216 | 0.002 | 0.782 |
| observed 0.4 (no attenuation) | any | - | 13 | 0.400 | 0.392 | [+0.016, +0.675] | 0.550 | 0.471 | 0.975 | 0.471 / 0.471 | 0.000 | 0.529 |
| rho = 0 | any | - | 13 | 0.000 | -0.007 | [-0.380, +0.369] | 0.000 | 0.000 | 0.041 | 0.000 / 0.000 | 0.271 | 0.729 |
| observed 0.4 (no attenuation) | any | - | 14 | 0.400 | 0.394 | [+0.053, +0.653] | 0.673 | 0.496 | 0.984 | 0.496 / 0.496 | 0.000 | 0.504 |
| rho = 0 | any | - | 14 | 0.000 | 0.001 | [-0.340, +0.343] | 0.001 | 0.000 | 0.060 | 0.000 / 0.000 | 0.344 | 0.656 |

Caveats: Pearson attenuation formula applied to Spearman coefficients; seed variation is counted as measurement noise (it is partly real variation between adapters); simulated pairs carry no task random effects; the conservative/liberal PASS rows bracket the Holm step.
