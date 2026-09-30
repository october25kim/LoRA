# Reliability and noise ceiling for merge loss `D`

Exploratory, CPU-only analysis of the saved E1b and E1c evaluation predictions. `D` is recomputed exactly as `1 - 0.5 * (merged_eval_t1 / single_eval_t1 + merged_eval_t2 / single_eval_t2)` using the original held-out-selected TA lambda; lambda is not reselected inside a bootstrap. Classification tasks use accuracy and STS-B uses the saved prediction-vs-label Spearman metric, matching `e1b/results/e1b.py`.

## Sampling-noise reliability

Each of 1,000 replicates resamples evaluation examples independently within each task. The same indices are used for the corresponding single and merged predictions and are shared across all pairs containing that task. `D` SEs are the SD of the 1,000 bootstrap values. Split-half reliability is the mean Spearman correlation across 1,000 independent bootstrap-half rank vectors; the Spearman–Brown value is `2r/(1+r)`.

| population | pairs | mean D SE | median D SE | 95th-pct D SE | split-half r | Spearman–Brown |
|---|---:|---:|---:|---:|---:|---:|
| E1b | 91 | 0.0091 | 0.0064 | 0.0234 | 0.877 | 0.934 |
| E1c-mixed | 91 | 0.0090 | 0.0062 | 0.0225 | 0.863 | 0.926 |
| E1c-seed1 | 91 | 0.0087 | 0.0062 | 0.0220 | 0.873 | 0.932 |

The full per-pair bootstrap table is `analysis/noise_ceiling_pair_bootstrap.csv`; the summary is `analysis/noise_ceiling_sampling_summary.csv`.

## Seed test–retest reliability

| comparison | pairs | Spearman rho(D) |
|---|---:|---:|
| E1b~E1c-mixed | 91 | 0.669 |
| E1b~E1c-seed1 | 91 | 0.670 |
| E1c-mixed~E1c-seed1 | 91 | 0.808 |

These are the correlations of D across the same unordered task pairs; they reproduce the reported 0.67–0.81 range without inventing or refitting any values.

## Attenuation ceiling

The predictor reliabilities below are E1b versus E1c seed-1 values for the same 91 unordered task pairs. Using the E1b~E1c-seed1 D test–retest coefficient as `rel_D`, the classical attenuation ceiling is `sqrt(rel_D * rel_predictor)`. The mixed-seed D coefficient is shown as a sensitivity in the CSV/JSON outputs.

| predictor | rel_D (E1b~E1c seed1) | rel_predictor | ceiling sqrt(product) |
|---|---:|---:|---:|
| O_A | 0.670 | 0.940 | 0.793 |
| tv_cosine | 0.670 | 0.729 | 0.699 |

## Detectability check for a true rho = 0.4

For orientation only, a Gaussian-copula simulation converts a latent rho=.4 to an expected observed rho of `.4 × ceiling`, then applies the paper's 14-task task-block bootstrap (1,000 simulations, 300 block resamples per simulation). This is an assumption-based detectability check, not a new confirmatory result.

| predictor | expected observed rho | mean task-block CI lower | P(CI lower > 0) | P(point rho >= .4) | P(both) |
|---|---:|---:|---:|---:|---:|
| O_A | 0.317 | -0.036 | 0.369 | 0.180 | 0.173 |
| tv_cosine | 0.280 | -0.068 | 0.285 | 0.099 | 0.098 |

Under this approximation, a latent rho=.4 would not be reliably detectable with n=91 and task-block CIs: the positive-CI detection rate is 36.9% for O_A and 28.5% for task-vector cosine, and the expected observed rho remains below the preregistered .4 PASS threshold.

## Manuscript-ready limitation paragraph

The reliability analysis suggests that merge loss is not measured without noise. Resampling evaluation examples within each task produced non-zero pair-level uncertainty in `D`, although the resulting split-half rank reliability was high after Spearman–Brown correction. More importantly, test–retest reliability across the same 91 task pairs was moderate: Spearman rho was approximately 0.67 for E1b versus the mixed-seed E1c population and 0.67 for E1b versus the seed-1 population (0.81 for the two E1c populations). Predictor test–retest reliability was high for output-subspace overlap and lower for task-vector cosine. Thus, classical attenuation alone places the maximum observable predictor–`D` association below one (approximately the values in the ceiling table), and a latent association of 0.4 would be expected to appear attenuated in a 91-pair study. These figures are exploratory and do not replace the preregistered task-block intervals; they reinforce that null or near-zero observed associations should not be interpreted as proof that arbitrarily small effects are absent.
