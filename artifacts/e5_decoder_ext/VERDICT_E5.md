# E5 VERDICT — acceptance-odds experiments on the E4b Qwen2.5-0.5B adapters

Generated 2026-09-27 20:16 KST by `make_verdict_e5.py` (formatting only). Pre-registration: `PREREG_E5.md` + `prereg_e5.json` (sha256 in `prereg_e5.sha256`; record `PREREG_RECORD_E5.txt`; local git commit `d0975c4` on branch `e5-prereg`, not pushed). **Disclosure: the pre-registration was written after the E4b verdict was known** (see PREREG_E5.md §0). Deviations: `DEVIATIONS_E5.md`.

## E5a — method comparison vs tuned TA on Qwen P (91/91 pairs; equal budget of 8 configurations per method)

Decision rule (pre-registered): a method beats TA iff mean gain ≥ +0.5 pp AND Holm-adjusted one-sided Wilcoxon p < 0.05 (Holm over 5 methods). TA mean normalized score 0.9600; TA boundary rate 0.02; TA selected λ: {'{"lam": 0.7}': 26, '{"lam": 0.85}': 25, '{"lam": 1.0}': 24, '{"lam": 1.15}': 13, '{"lam": 1.5}': 1, '{"lam": 0.5}': 1, '{"lam": 0.3}': 1}.

| method | mean score | mean Δ vs TA (pp) | median Δ (pp) | task-block 95% CI (pp) | W/T/L | Wilcoxon p (1-sided >) | **Holm p** | sign-flip Holm p (E3 test) | Wilcoxon p (2-sided) | boundary | **beats TA?** |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| PICO_TA | 0.9618 | +0.19 | +0.05 | [-0.21, +0.70] | 46/0/45 | 0.2308 | 0.6924 | 0.2007 | 0.462 | 0.02 | **no** |
| GATE | 0.9603 | +0.03 | +0.00 | [-0.01, +0.07] | 34/28/29 | 0.07575 | 0.303 | 0.1104 | 0.151 | 0.02 | **no** |
| FORCEGATE | 0.9635 | +0.35 | +0.20 | [+0.04, +0.79] | 61/2/28 | 4.708e-05 | 0.0002354 | 0.0005 | 9.42e-05 | 0.02 | **no** |
| TIES | 0.9448 | -1.52 | -0.69 | [-2.76, -0.64] | 12/0/79 | 1 | 1 | 1 | 1.56e-12 | 0.08 | **no** |
| PICO_GATE | 0.9615 | +0.15 | +0.03 | [-0.27, +0.65] | 46/0/45 | 0.2685 | 0.6924 | 0.2016 | 0.537 | 0.03 | **no** |

**Any method beats TA: NO.** GATE: CI upper < +0.5 pp → a ≥ 0.5 pp gain is not supported. TIES: CI upper < +0.5 pp → a ≥ 0.5 pp gain is not supported.

Wilcoxon vs E3 sign-flip decision disagreement: none.

Selected configurations: PICO_TA: {'{"c": 1.41}': 32, '{"c": 1.2}': 20, '{"c": 0.99}': 18, '{"c": 1.63}': 15, '{"c": 1.84}': 2, '{"c": 2.12}': 2, '{"c": 0.71}': 2}; GATE: {'{"theta": 30.0, "beta": 1.0, "lam": 0.7}': 26, '{"theta": 30.0, "beta": 1.0, "lam": 1.0}': 26, '{"theta": 30.0, "beta": 1.0, "lam": 0.85}': 24, '{"theta": 30.0, "beta": 1.0, "lam": 1.15}': 11, '{"theta": 30.0, "beta": 1.0, "lam": 1.3}': 1, '{"theta": 30.0, "beta": 1.0, "lam": 1.5}': 1, '{"theta": 30.0, "beta": 1.0, "lam": 0.5}': 1, '{"theta": 30.0, "beta": 1.0, "lam": 0.3}': 1}; FORCEGATE: {'{"force_k": 1, "beta": 1.0, "lam": 1.0}': 32, '{"force_k": 1, "beta": 1.0, "lam": 0.7}': 26, '{"force_k": 1, "beta": 1.0, "lam": 0.85}': 18, '{"force_k": 1, "beta": 1.0, "lam": 1.15}': 11, '{"force_k": 1, "beta": 1.0, "lam": 1.5}': 2, '{"force_k": 1, "beta": 1.0, "lam": 0.5}': 2}; TIES: {'{"k": 20, "lam": 1.25}': 31, '{"k": 20, "lam": 1.0}': 28, '{"k": 20, "lam": 1.5}': 21, '{"k": 20, "lam": 0.7}': 7, '{"k": 20, "lam": 1.75}': 3, '{"k": 20, "lam": 2.0}': 1}; PICO_GATE: {'{"theta": 30.0, "c": 1.41}': 33, '{"theta": 30.0, "c": 1.2}': 22, '{"theta": 30.0, "c": 0.99}': 18, '{"theta": 30.0, "c": 1.63}': 12, '{"theta": 30.0, "c": 1.84}': 2, '{"theta": 30.0, "c": 2.12}': 2, '{"theta": 30.0, "c": 0.71}': 1, '{"theta": 30.0, "c": 0.42}': 1}

### Secondary (no multiplicity correction)

| contrast | n | mean (pp) | median (pp) | task-block 95% CI (pp) | Wilcoxon p (2-sided) | sign-flip p (2-sided) | W/T/L |
|---|---:|---:|---:|---|---:|---:|---|
| GATE_minus_TA_on_gate_active | 66 | +0.04 | +0.01 | [-0.01, +0.10] | 0.151 | 0.0514 | 34/3/29 |
| GATE_minus_PICO_TA | 91 | -0.16 | -0.05 | [-0.67, +0.24] | 0.576 | 0.201 | 44/1/46 |
| FORCEGATE_minus_PICO_TA | 91 | +0.16 | +0.10 | [-0.22, +0.70] | 0.117 | 0.132 | 49/0/42 |
| PICO_GATE_minus_PICO_TA | 91 | -0.04 | +0.00 | [-0.21, +0.05] | 0.973 | 0.546 | 30/27/34 |
| PICO_GATE_minus_GATE | 91 | +0.12 | -0.01 | [-0.30, +0.63] | 0.797 | 0.3 | 44/0/47 |
| PICO_GATE_minus_PICO_TA_on_gate_active | 66 | -0.05 | -0.01 | [-0.27, +0.08] | 0.973 | 0.544 | 30/2/34 |
| TA8_minus_TA4_E1bgrid | 91 | +0.05 | +0.00 | [-0.09, +0.21] | 0.176 | 0.249 | 21/52/18 |
| PICO_c1_untuned_minus_TA | 91 | -0.92 | -0.33 | [-2.02, -0.08] | 9.47e-06 | 0.0001 | 27/2/62 |

Gate-active pairs (θ★ = 30°): 66. Sanity (E3 TA at the E4b λs vs E4b stage 2): {'max_abs_diff': 0.0, 'lam_selected_mismatches': 0, 'exact': True}.

Exploratory Spearman(predictor, gain): {"PICO_TA": {"O_A": -0.2775441949354993, "tv_cosine": -0.30767638158942506}, "GATE": {"O_A": -0.006101527164874169, "tv_cosine": -0.0988124140992271}, "FORCEGATE": {"O_A": 0.04118506603419344, "tv_cosine": 0.061881119132195905}, "TIES": {"O_A": -0.0017040930084408345, "tv_cosine": -0.029081063863672565}, "PICO_GATE": {"O_A": -0.29504698200350377, "tv_cosine": -0.37364229972925633}}

## E5b — λ grid extended to {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0} (pre-registered SENSITIVITY; the E4b primary verdict is unchanged)

| population | H | predictor | ρ original grid | ρ extended grid | task-block CI (ext) | Holm p (ext) | LOTO frac (ext) | rule (ext, descriptive) | rule (orig) |
|---|---|---|---:|---:|---|---:|---:|---|---|
| P | H1 | O_A | 0.3452 | 0.3475 | [-0.096, 0.610] | 0.0294 | 1.00 | INCONCLUSIVE | INCONCLUSIVE |
| P | H2 | tv_cosine | 0.3684 | 0.3699 | [-0.079, 0.653] | 0.0238 | 1.00 | INCONCLUSIVE | INCONCLUSIVE |
| R0 | H1 | O_A | 0.3665 | 0.3665 | [-0.078, 0.635] | 0.0220 | 1.00 | INCONCLUSIVE | INCONCLUSIVE |
| R0 | H2 | tv_cosine | 0.3948 | 0.3948 | [-0.076, 0.665] | 0.0220 | 1.00 | INCONCLUSIVE | INCONCLUSIVE |
| S1 | H1 | O_A | 0.2112 | 0.2062 | [-0.251, 0.544] | 0.2956 | 0.64 | INCONCLUSIVE | INCONCLUSIVE |
| S1 | H2 | tv_cosine | 0.1046 | 0.0955 | [-0.386, 0.502] | 0.3168 | 0.07 | INCONCLUSIVE | INCONCLUSIVE |

- **P**: λ\* counts original {'0.3': 1, '0.5': 1, '0.7': 41, '1.0': 48} → extended {'0.3': 1, '0.5': 1, '0.7': 40, '1.0': 47, '1.3': 1, '2.0': 1}; λ changed in 2/91 pairs; share at the new top λ = 2.0: 0.01; mean D 0.0406 → 0.0402 (mean D − D₊ = 0.0003); Spearman(D, D₊) = 1.000; fixed-λ ρ at the new λs: {"O_A": {"lam1.3": 0.17928014014970536, "lam1.5": 0.07768752986144291, "lam2.0": -0.06891224717311674}, "tv_cosine": {"lam1.3": 0.21753463927376973, "lam1.5": 0.12274247491638797, "lam2.0": -0.06604554865424431}}; sanity λ = 1.0 max |Δ| = 0.0.
- **R0**: λ\* counts original {'0.3': 3, '0.5': 34, '0.7': 43, '1.0': 11} → extended {'0.3': 3, '0.5': 34, '0.7': 43, '1.0': 11}; λ changed in 0/91 pairs; share at the new top λ = 2.0: 0.00; mean D 0.0413 → 0.0413 (mean D − D₊ = 0.0000); Spearman(D, D₊) = 1.000; fixed-λ ρ at the new λs: {"O_A": {"lam1.3": 0.32173913043478264, "lam1.5": 0.2688007644529384, "lam2.0": 0.11146679407548973}, "tv_cosine": {"lam1.3": 0.31122790253225036, "lam1.5": 0.26351329829590703, "lam2.0": 0.13924191750278708}}; sanity λ = 1.0 max |Δ| = None.
- **S1**: λ\* counts original {'0.3': 3, '0.5': 18, '0.7': 45, '1.0': 25} → extended {'0.3': 3, '0.5': 18, '0.7': 44, '1.0': 24, '1.3': 2}; λ changed in 2/91 pairs; share at the new top λ = 2.0: 0.00; mean D 0.0453 → 0.0443 (mean D − D₊ = 0.0010); Spearman(D, D₊) = 0.997; fixed-λ ρ at the new λs: {"O_A": {"lam1.3": 0.16018474279343844, "lam1.5": 0.09324733237776715, "lam2.0": 0.00613154960981048}, "tv_cosine": {"lam1.3": -0.005844879757923236, "lam1.5": -0.07238413760152891, "lam2.0": -0.14362159579550884}}; sanity λ = 1.0 max |Δ| = None.

Cross-check E5a TA held-out at λ = 1.3/1.5 vs E5b: {'n_compared': 182, 'max_abs_diff': 0.0}.

## E5c — evaluation-sampling reliability of D (CPU; saved per-example predictions at the original λ\*)

| exp | pop | n | SD(D) between pairs | mean / median / p95 SE(D) | 1 − mean SE²/Var D | indep.-replicate r → SB (E1b method) | true split-half r [95%] → SB | √SB | ρ(O_A,D): obs / boot SD / [2.5, 97.5] | ρ(cos,D): obs / boot SD / [2.5, 97.5] |
|---|---|---:|---:|---|---:|---|---|---:|---|---|
| e4a | P | 78 | 0.0182 | 0.0057 / 0.0039 / 0.0133 | 0.856 | 0.819 → 0.900 | 0.653 [0.484, 0.775] → 0.790 | 0.889 | 0.161 / 0.057 / [0.033, 0.248] | 0.112 / 0.040 / [0.042, 0.200] |
| e4a | R0 | 78 | 0.0194 | 0.0056 / 0.0039 / 0.0131 | 0.876 | 0.859 → 0.924 | 0.729 [0.548, 0.847] → 0.843 | 0.918 | 0.146 / 0.050 / [0.034, 0.233] | 0.130 / 0.044 / [0.035, 0.213] |
| e4a | S1 | 78 | 0.0151 | 0.0058 / 0.0039 / 0.0131 | 0.786 | 0.809 → 0.894 | 0.639 [0.445, 0.782] → 0.779 | 0.883 | 0.037 / 0.063 / [-0.079, 0.157] | 0.038 / 0.053 / [-0.066, 0.142] |
| e4b | P | 91 | 0.0431 | 0.0075 / 0.0053 / 0.0177 | 0.955 | 0.902 → 0.949 | 0.814 [0.713, 0.879] → 0.897 | 0.947 | 0.345 / 0.038 / [0.251, 0.401] | 0.368 / 0.035 / [0.285, 0.425] |
| e4b | R0 | 91 | 0.0383 | 0.0080 / 0.0059 / 0.0193 | 0.936 | 0.916 → 0.956 | 0.843 [0.750, 0.910] → 0.915 | 0.956 | 0.366 / 0.040 / [0.253, 0.415] | 0.395 / 0.039 / [0.295, 0.445] |
| e4b | S1 | 91 | 0.0460 | 0.0078 / 0.0055 / 0.0177 | 0.957 | 0.918 → 0.957 | 0.840 [0.736, 0.912] → 0.913 | 0.955 | 0.211 / 0.032 / [0.138, 0.264] | 0.105 / 0.035 / [0.032, 0.171] |

D recomputed from saved predictions equals the reported D in every population (max |Δ| = 0.0e+00). R = 1000, split-halves = 200, seeds [20260927, 20260928].

## E5d — lemma quantities on Qwen gate layers (derived; Prop. 2–3 of `paper/LEMMA_PROJECTION.md`)

| set | layers | pairs | \|cos φ\| median [IQR] (collinearity of X1, X2) | fraction of gate edit not a coefficient change (1 − cos²φ) median [IQR] | mean | median c | share c > 0 | share \|cos\| ≥ 0.5 / ≥ 0.9 | median ε | median edit/‖M_sum‖ | max identity residual |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|
| θ★=30° FAIL layers, all pops | 351 | 235 | 0.172 [0.047, 0.428] | 0.970 [0.817, 0.998] | 0.865 | 0.064 | 0.73 | 0.20 / 0.01 | 0.64 | 0.295 | 1.3e-15 |
|   gate30 P | 72 | 66 | 0.026 [0.012, 0.046] | 0.999 [0.998, 1.000] | 0.998 | 0.006 | 0.58 | 0.00 / 0.00 | 0.78 | 0.254 | 1.8e-16 |
|   gate30 R0 | 138 | 78 | 0.242 [0.117, 0.440] | 0.941 [0.806, 0.986] | 0.865 | 0.122 | 0.75 | 0.18 / 0.00 | 0.63 | 0.308 | 1.3e-15 |
|   gate30 S1 | 126 | 78 | 0.275 [0.116, 0.611] | 0.924 [0.626, 0.986] | 0.775 | 0.161 | 0.78 | 0.35 / 0.02 | 0.53 | 0.321 | 2.5e-16 |
|   gate30 S2 | 15 | 13 | 0.064 [0.019, 0.147] | 0.996 [0.978, 1.000] | 0.988 | 0.053 | 0.80 | 0.00 / 0.00 | 0.68 | 0.289 | 1.6e-16 |
|   gate30 down_proj | 20 | 19 | 0.254 [0.051, 0.609] | 0.935 [0.628, 0.997] | 0.804 | 0.212 | 0.85 | 0.35 / 0.00 | 0.64 | 0.247 | 4.3e-16 |
|   gate30 k_proj | 236 | 219 | 0.086 [0.029, 0.212] | 0.993 [0.955, 0.999] | 0.964 | 0.017 | 0.62 | 0.03 / 0.00 | 0.74 | 0.273 | 2.4e-16 |
|   gate30 o_proj | 11 | 8 | 0.772 [0.557, 0.817] | 0.405 [0.332, 0.690] | 0.519 | 0.719 | 1.00 | 0.82 / 0.18 | 0.35 | 0.386 | 1.3e-16 |
|   gate30 v_proj | 84 | 66 | 0.552 [0.375, 0.761] | 0.695 [0.420, 0.859] | 0.648 | 0.453 | 0.96 | 0.55 / 0.00 | 0.35 | 0.397 | 1.3e-15 |
| forced k=1, all 168 layers of P | 15288 | 91 | 0.023 [0.011, 0.041] | 0.999 [0.998, 1.000] | 0.998 | 0.001 | 0.56 | 0.00 / 0.00 | 0.26 | 0.080 | 1.7e-15 |
|   forced down_proj | 2184 | 91 | 0.014 [0.007, 0.024] | 1.000 [0.999, 1.000] | 0.999 | 0.001 | 0.58 | 0.00 / 0.00 | 0.24 | 0.073 | 1.7e-15 |
|   forced gate_proj | 2184 | 91 | 0.026 [0.012, 0.044] | 0.999 [0.998, 1.000] | 0.998 | 0.001 | 0.60 | 0.00 / 0.00 | 0.08 | 0.021 | 8.9e-16 |
|   forced k_proj | 2184 | 91 | 0.024 [0.011, 0.045] | 0.999 [0.998, 1.000] | 0.998 | 0.003 | 0.57 | 0.00 / 0.00 | 0.57 | 0.171 | 2.4e-16 |
|   forced o_proj | 2184 | 91 | 0.022 [0.011, 0.040] | 0.999 [0.998, 1.000] | 0.999 | 0.001 | 0.54 | 0.00 / 0.00 | 0.24 | 0.063 | 2.1e-16 |
|   forced q_proj | 2184 | 91 | 0.028 [0.013, 0.051] | 0.999 [0.997, 1.000] | 0.997 | 0.002 | 0.58 | 0.00 / 0.00 | 0.34 | 0.089 | 3.4e-16 |
|   forced up_proj | 2184 | 91 | 0.025 [0.012, 0.043] | 0.999 [0.998, 1.000] | 0.998 | 0.001 | 0.57 | 0.00 / 0.00 | 0.14 | 0.044 | 8.4e-16 |
|   forced v_proj | 2184 | 91 | 0.025 [0.012, 0.043] | 0.999 [0.998, 1.000] | 0.998 | 0.000 | 0.51 | 0.00 / 0.00 | 0.45 | 0.119 | 1.0e-15 |

Reference (BERT seed pair): paper/verify_lemma_seed_pair.json: 2 FAIL layers, cos 0.107 / 0.184, eps 0.857 / 0.850, c 0.102 / 0.190.

## Compute

Logged GPU process seconds: {"e5d_s": 3086, "e5a_s": 12983, "e5b_P_s": 2054, "e5b_R0_s": 2027, "e5b_S1_s": 2025}; GPU jobs (E5a + E5b) total 5.30 h (serial). E5d ran on CPU; E5c on the box CPU.


## Interpretation (hand-written by the executor on 2026-09-27 ~20:20 KST; all numbers are copied from the tables above, and none are new)

1. **E5a (primary):** with an equal 8-configuration held-out tuning budget on all 91 Qwen mixed-seed cross-task pairs (P), **no method beats tuned TA** under the pre-registered rule (≥ +0.5 pp AND Holm p < 0.05).
   - FORCEGATE (k = 1) gives a small, statistically reliable gain: +0.35 pp, Holm p = 2.4e-4, 61/2/28 W/T/L, task-block CI [+0.04, +0.79]. It is below the pre-registered +0.5 pp practical threshold, so it counts as "significant but not practically beating TA".
   - GATE30 is effectively TA: +0.03 pp, and the CI upper bound is below 0.5 pp.
   - Tuned Pico soft reweighting (+0.19 pp) and gate+Pico (+0.15 pp) are indistinguishable from TA. Untuned Pico (c = 1) is worse than TA (−0.92 pp).
   - TIES-lite is clearly worse (−1.52 pp).
2. **E5b (sensitivity):** extending the λ grid to 2.0 is immaterial.
   - λ\* moved in 2/91 (P), 0/91 (R0) and 2/91 (S1) pairs.
   - Spearman(D, D₊) ≥ 0.997.
   - H1/H2 ρ changed by ≤ 0.01. P: 0.345→0.348 / 0.368→0.370. S1: 0.211→0.206 / 0.105→0.096.
   - Every rule outcome stays INCONCLUSIVE, the same as E4b.
   - The "λ = 1.0 at grid top for 53% of pairs" concern does not hold: once λ > 1 is allowed, it is almost never selected.
3. **E5c:** D is measured reliably.
   - On E4b: true split-half Spearman–Brown 0.90–0.92, and 1 − SE²/Var 0.94–0.96.
   - On E4a it is lower but still high: split-half SB 0.78–0.84, and 1 − SE²/Var 0.79–0.88.
   - Eval-sampling noise in ρ(O_A, D) is small next to the task-block CI width. E4b P boot SD is 0.038, versus a CI of about [−0.10, 0.61].
   - So the inconclusive verdicts come from between-task heterogeneity, not from eval-set noise.
4. **E5d:** on mixed-seed P pairs the θ★ = 30° gate edit is almost purely a removal of the non-shared component. Median |cos φ| is 0.026, and the median fraction of the edit that is not a coefficient change is 0.999.
   - Some collinearity appears only on shared-seed populations (R0/S1) and in v_proj/o_proj layers. o_proj median |cos| is 0.77, with a non-coefficient fraction of 0.41.
   - This fits GATE ≈ TA on P: the deleted component carries little of the merged function's utility. It also fits FORCEGATE's small gain.
