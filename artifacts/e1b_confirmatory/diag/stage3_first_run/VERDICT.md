# E1b VERDICT — H1 (O_A): **FAIL**; H2 (task-vector cosine): **INCONCLUSIVE**

Generated 2026-09-26 00:22 KST on NVIDIA GeForce RTX 4070 Ti SUPER. Pre-registration: PREREG.md / prereg.json (sha256 in prereg.sha256); deviations: deviations.md.

## Primary hypotheses (pre-registered; Holm over H1, H2)

Valid tasks: 14 → 91 pairs. Excluded: {'qqp': ['(a) 0.6240 < 0.7240', '(b) accuracy 0.6240 < 0.8571'], 'boolq': ['(a) 0.6217 < 0.7217']}.

| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | verdict |
|---|---|---:|---|---:|---:|---:|---|
| H1 | O_A | -0.1935 | [-0.521, 0.203] | 0.9292 | 1.0000 | 0.00 | **FAIL** |
| H2 | tv_cosine | -0.0441 | [-0.425, 0.316] | 0.6344 | 1.0000 | 0.00 | **INCONCLUSIVE** |

Rule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE.

H1 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola -0.17, sst2 -0.20, mrpc -0.26, stsb -0.17, mnli -0.04, qnli -0.24, rte -0.24, wic -0.24, snli -0.22, scitail -0.17, ag_news -0.21, imdb -0.19, trec -0.19, yelp_polarity -0.15

H2 criteria: {'rho_ge_0.4': False, 'holm_p_lt_0.05': False, 'loto_ge_75pct_gt_0.2': False}. LOTO ρ: cola -0.03, sst2 -0.02, mrpc -0.09, stsb -0.04, mnli -0.02, qnli -0.10, rte 0.06, wic -0.08, snli -0.11, scitail -0.02, ag_news -0.03, imdb 0.01, trec -0.14, yelp_polarity -0.01

D: mean 0.0480, median 0.0435, range [-0.0420, 0.1893]; λ* counts {'0.5': 22, '0.7': 60, '1.0': 9}. ρ(O_A, tv_cosine) = 0.539. Min θ_min over all pairs/layers = 45.1°.

## Secondary / exploratory predictors (not used for verdicts; uncorrected)

| predictor | ρ | 95% CI | perm p one-sided | perm p two-sided | LOTO frac>0.2 |
|---|---:|---|---:|---:|---:|
| O_A | -0.194 | [-0.52, 0.20] | 0.9292 | 0.1425 | 0.00 |
| tv_cosine | -0.044 | [-0.43, 0.32] | 0.6344 | 0.7256 | 0.00 |
| O_B | -0.194 | [-0.52, 0.20] | 0.9292 | 0.1425 | 0.00 |
| mean_theta_min_A_deg | 0.140 | [-0.24, 0.48] | 0.1334 | 0.2726 | 0.14 |
| min_theta_min_A_deg | 0.136 | [-0.19, 0.47] | 0.0999 | 0.2062 | 0.07 |
| n_layers_theta_min_lt30_A | nan | [nan, nan] | 0.0001 | 0.0001 | 0.00 |
| sign_conflict_top20 | 0.048 | [-0.31, 0.41] | 0.3558 | 0.7103 | 0.00 |
| norm_ratio | 0.111 | [-0.28, 0.44] | 0.1771 | 0.3599 | 0.00 |
| null_z_O_A | -0.194 | [-0.52, 0.20] | 0.9292 | 0.1425 | 0.00 |
| sign_conflict_all | 0.061 | [-0.32, 0.42] | 0.3150 | 0.6261 | 0.00 |

Fixed-λ ρ (eval D at each λ): {"O_A": {"lam0.3": -0.418, "lam0.5": -0.247, "lam0.7": -0.193, "lam1.0": -0.152}, "tv_cosine": {"lam0.3": -0.163, "lam0.5": -0.228, "lam0.7": -0.01, "lam1.0": 0.194}}

## Method comparison (exploratory)

| method | mean norm. score | median | min | mean diff vs TA | task-block 95% CI | pair-bootstrap 95% CI | win/tie/loss | Wilcoxon p |
|---|---:|---:|---:|---:|---|---|---|---:|
| TA@lam* | 0.9520 | 0.9565 | 0.8107 | — | — | — | — | — |
| TIES-lite@lam*TA | 0.8813 | 0.9048 | 0.4890 | -0.0707 | [-0.1116, -0.0368] | [-0.0841, -0.0583] | 2/0/89 | 2.82e-16 |
| TIES-lite@own-lam | 0.9322 | 0.9376 | 0.6262 | -0.0198 | [-0.0348, -0.0105] | [-0.0261, -0.0142] | 12/0/79 | 4.02e-13 |
| gate30@lam*TA | 0.9520 | 0.9565 | 0.8107 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/91/0 | nan |
| gate30@own-lam | 0.9520 | 0.9565 | 0.8107 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/91/0 | nan |

Normalized score = mean over the two tasks of merged/single eval metric (= 1 − D for TA). Gate active (≥1 layer with θ_min < 30°) in 0/91 pairs; elsewhere gate ≡ TA (tie).

## Integrity (stage0)

| task | trained | eval metric | score | (a) thr | (a) | (b) ref−5pp | (b) | (c) max diff | valid |
|---|---|---|---:|---:|---|---|---|---:|---|
| cola | prereg (2360 steps) | accuracy | 0.8236 | 0.7913 | True | mcc 0.5654 ≥ 0.5153 | True | 9.7e-06 | **True** |
| sst2 | prereg (5625 steps) | accuracy | 0.9186 | 0.6092 | True | accuracy 0.9186 ≥ 0.8732 | True | 1.8e-06 | **True** |
| mrpc | prereg (840 steps) | accuracy | 0.8505 | 0.7838 | True | accuracy 0.8505 ≥ 0.7907 | True | 8.7e-06 | **True** |
| qqp | D1 (deviations.md) (34017 steps) | accuracy | 0.6240 | 0.7240 | False | accuracy 0.6240 ≥ 0.8571 | False | 8.9e-08 | **False** |
| stsb | prereg (1490 steps) | spearman | 0.8894 | 0.7000 | True | spearman 0.8894 ≥ 0.8348 | True | 1.9e-06 | **True** |
| mnli | D1 (deviations.md) (36723 steps) | accuracy | 0.8237 | 0.4274 | True | accuracy 0.8237 ≥ 0.7891 | True | 3.0e-06 | **True** |
| qnli | prereg (5625 steps) | accuracy | 0.8977 | 0.6054 | True | accuracy 0.8977 ≥ 0.8566 | True | 5.8e-06 | **True** |
| rte | prereg (470 steps) | accuracy | 0.6426 | 0.5729 | True | accuracy 0.6426 ≥ 0.6070 | True | 1.2e-05 | **True** |
| boolq | prereg (2640 steps) | accuracy | 0.6217 | 0.7217 | False | n/a | None | 1.8e-07 | **False** |
| wic | prereg (1390 steps) | accuracy | 0.6959 | 0.6000 | True | n/a | None | 2.6e-05 | **True** |
| snli | prereg (5625 steps) | accuracy | 0.8702 | 0.4331 | True | n/a | None | 4.2e-06 | **True** |
| scitail | prereg (2073 steps) | accuracy | 0.9333 | 0.5962 | True | n/a | None | 4.3e-06 | **True** |
| ag_news | prereg (5625 steps) | accuracy | 0.9387 | 0.3500 | True | n/a | None | 4.6e-06 | **True** |
| imdb | prereg (2250 steps) | accuracy | 0.8898 | 0.5977 | True | n/a | None | 4.4e-06 | **True** |
| trec | prereg (1400 steps) | accuracy | 0.9660 | 0.2880 | True | n/a | None | 3.3e-06 | **True** |
| yelp_polarity | prereg (5625 steps) | accuracy | 0.9407 | 0.6108 | True | n/a | None | 3.8e-06 | **True** |

## Pair results

| pair | λ* | merged t1 | merged t2 | norm t1 | norm t2 | D | O_A | tv cos | TIES@λ* | gate@λ* (layers) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cola-sst2 | 0.7 | 0.7517 | 0.8830 | 0.9127 | 0.9613 | 0.0630 | 0.01712 | 0.0012 | 0.8975 | 0.9370 (0) |
| cola-mrpc | 0.7 | 0.8092 | 0.7990 | 0.9825 | 0.9395 | 0.0390 | 0.01798 | 0.0015 | 0.8970 | 0.9610 (0) |
| cola-stsb | 0.7 | 0.7967 | 0.8678 | 0.9674 | 0.9757 | 0.0284 | 0.01751 | 0.0004 | 0.9638 | 0.9716 (0) |
| cola-mnli | 0.7 | 0.7306 | 0.7573 | 0.8871 | 0.9194 | 0.0968 | 0.01295 | 0.0005 | 0.7764 | 0.9032 (0) |
| cola-qnli | 0.7 | 0.7641 | 0.8770 | 0.9278 | 0.9770 | 0.0476 | 0.01572 | 0.0007 | 0.8666 | 0.9524 (0) |
| cola-rte | 0.7 | 0.8092 | 0.6101 | 0.9825 | 0.9494 | 0.0340 | 0.01927 | 0.0014 | 0.9113 | 0.9660 (0) |
| cola-wic | 0.7 | 0.7977 | 0.6646 | 0.9686 | 0.9550 | 0.0382 | 0.01798 | 0.0009 | 0.9335 | 0.9618 (0) |
| cola-snli | 0.7 | 0.7756 | 0.8267 | 0.9418 | 0.9499 | 0.0541 | 0.01505 | 0.0007 | 0.8507 | 0.9459 (0) |
| cola-scitail | 1.0 | 0.7910 | 0.9064 | 0.9604 | 0.9712 | 0.0342 | 0.01661 | 0.0005 | 0.9013 | 0.9658 (0) |
| cola-ag_news | 0.7 | 0.7469 | 0.9257 | 0.9069 | 0.9861 | 0.0535 | 0.01600 | 0.0006 | 0.9135 | 0.9465 (0) |
| cola-imdb | 1.0 | 0.7872 | 0.7869 | 0.9558 | 0.8844 | 0.0799 | 0.01743 | 0.0008 | 0.9102 | 0.9201 (0) |
| cola-trec | 0.7 | 0.7709 | 0.9580 | 0.9360 | 0.9917 | 0.0362 | 0.01859 | 0.0012 | 0.9379 | 0.9638 (0) |
| cola-yelp_polarity | 0.7 | 0.7459 | 0.9078 | 0.9057 | 0.9650 | 0.0646 | 0.01621 | 0.0008 | 0.9278 | 0.9354 (0) |
| sst2-mrpc | 0.7 | 0.9312 | 0.8137 | 1.0137 | 0.9568 | 0.0147 | 0.01586 | 0.0009 | 0.9175 | 0.9853 (0) |
| sst2-stsb | 0.5 | 0.9151 | 0.8499 | 0.9963 | 0.9555 | 0.0241 | 0.01594 | 0.0014 | 0.9626 | 0.9759 (0) |
| sst2-mnli | 0.7 | 0.9025 | 0.7353 | 0.9825 | 0.8926 | 0.0624 | 0.01384 | 0.0008 | 0.8760 | 0.9376 (0) |
| sst2-qnli | 0.5 | 0.9255 | 0.8380 | 1.0075 | 0.9335 | 0.0295 | 0.01486 | 0.0007 | 0.8683 | 0.9705 (0) |
| sst2-rte | 0.7 | 0.9163 | 0.5523 | 0.9975 | 0.8596 | 0.0715 | 0.01595 | 0.0014 | 0.9185 | 0.9285 (0) |
| sst2-wic | 0.5 | 0.9151 | 0.6379 | 0.9963 | 0.9167 | 0.0435 | 0.01447 | 0.0006 | 0.9292 | 0.9565 (0) |
| sst2-snli | 0.7 | 0.9083 | 0.8026 | 0.9888 | 0.9222 | 0.0445 | 0.01500 | 0.0011 | 0.8955 | 0.9555 (0) |
| sst2-scitail | 0.7 | 0.9151 | 0.8712 | 0.9963 | 0.9334 | 0.0352 | 0.01545 | 0.0012 | 0.8814 | 0.9648 (0) |
| sst2-ag_news | 0.7 | 0.8865 | 0.9263 | 0.9650 | 0.9868 | 0.0241 | 0.01529 | 0.0005 | 0.9382 | 0.9759 (0) |
| sst2-imdb | 0.7 | 0.9163 | 0.8668 | 0.9975 | 0.9742 | 0.0142 | 0.02232 | 0.0071 | 0.9853 | 0.9858 (0) |
| sst2-trec | 0.7 | 0.8968 | 0.9600 | 0.9763 | 0.9938 | 0.0150 | 0.01661 | 0.0007 | 0.9270 | 0.9850 (0) |
| sst2-yelp_polarity | 0.5 | 0.9140 | 0.9277 | 0.9950 | 0.9862 | 0.0094 | 0.02186 | 0.0058 | 0.9686 | 0.9906 (0) |
| mrpc-stsb | 0.7 | 0.8480 | 0.8202 | 0.9971 | 0.9222 | 0.0403 | 0.02467 | 0.0064 | 0.9159 | 0.9597 (0) |
| mrpc-mnli | 0.5 | 0.8235 | 0.7424 | 0.9683 | 0.9013 | 0.0652 | 0.01379 | 0.0010 | 0.7365 | 0.9348 (0) |
| mrpc-qnli | 0.7 | 0.8186 | 0.8257 | 0.9625 | 0.9199 | 0.0588 | 0.01887 | 0.0016 | 0.9197 | 0.9412 (0) |
| mrpc-rte | 0.5 | 0.7941 | 0.5848 | 0.9337 | 0.9101 | 0.0781 | 0.02550 | 0.0055 | 0.9081 | 0.9219 (0) |
| mrpc-wic | 1.0 | 0.7843 | 0.6301 | 0.9222 | 0.9054 | 0.0862 | 0.02166 | 0.0021 | 0.9109 | 0.9138 (0) |
| mrpc-snli | 0.5 | 0.7868 | 0.8032 | 0.9251 | 0.9229 | 0.0760 | 0.01668 | 0.0013 | 0.7594 | 0.9240 (0) |
| mrpc-scitail | 0.7 | 0.8456 | 0.8781 | 0.9942 | 0.9408 | 0.0325 | 0.02366 | 0.0057 | 0.8949 | 0.9675 (0) |
| mrpc-ag_news | 0.7 | 0.7402 | 0.9091 | 0.8703 | 0.9685 | 0.0806 | 0.01678 | 0.0001 | 0.8753 | 0.9194 (0) |
| mrpc-imdb | 0.7 | 0.7941 | 0.8633 | 0.9337 | 0.9702 | 0.0480 | 0.01814 | 0.0013 | 0.9110 | 0.9520 (0) |
| mrpc-trec | 1.0 | 0.8603 | 0.9560 | 1.0115 | 0.9896 | -0.0006 | 0.01719 | 0.0003 | 0.9514 | 1.0006 (0) |
| mrpc-yelp_polarity | 0.7 | 0.8064 | 0.9319 | 0.9481 | 0.9906 | 0.0306 | 0.01679 | 0.0009 | 0.9048 | 0.9694 (0) |
| stsb-mnli | 0.7 | 0.7727 | 0.8070 | 0.8688 | 0.9797 | 0.0757 | 0.01472 | 0.0016 | 0.8537 | 0.9243 (0) |
| stsb-qnli | 0.7 | 0.8280 | 0.8838 | 0.9309 | 0.9845 | 0.0423 | 0.02057 | 0.0030 | 0.9138 | 0.9577 (0) |
| stsb-rte | 1.0 | 0.8138 | 0.6679 | 0.9149 | 1.0393 | 0.0229 | 0.02482 | 0.0048 | 0.9599 | 0.9771 (0) |
| stsb-wic | 0.7 | 0.8708 | 0.6881 | 0.9791 | 0.9887 | 0.0161 | 0.02169 | 0.0022 | 0.9642 | 0.9839 (0) |
| stsb-snli | 0.7 | 0.7831 | 0.8500 | 0.8805 | 0.9768 | 0.0714 | 0.01987 | 0.0035 | 0.9056 | 0.9286 (0) |
| stsb-scitail | 0.7 | 0.8734 | 0.9248 | 0.9819 | 0.9910 | 0.0135 | 0.02286 | 0.0051 | 0.9120 | 0.9865 (0) |
| stsb-ag_news | 0.5 | 0.8608 | 0.9112 | 0.9678 | 0.9707 | 0.0308 | 0.01616 | 0.0008 | 0.9045 | 0.9692 (0) |
| stsb-imdb | 0.7 | 0.8875 | 0.8861 | 0.9978 | 0.9958 | 0.0032 | 0.01670 | 0.0006 | 0.9807 | 0.9968 (0) |
| stsb-trec | 0.7 | 0.8047 | 0.9300 | 0.9048 | 0.9627 | 0.0662 | 0.01792 | 0.0007 | 0.8976 | 0.9338 (0) |
| stsb-yelp_polarity | 0.5 | 0.8675 | 0.9254 | 0.9754 | 0.9837 | 0.0205 | 0.01489 | 0.0005 | 0.9436 | 0.9795 (0) |
| mnli-qnli | 0.5 | 0.7484 | 0.6399 | 0.9086 | 0.7129 | 0.1893 | 0.01680 | 0.0036 | 0.7536 | 0.8107 (0) |
| mnli-rte | 0.7 | 0.7958 | 0.7184 | 0.9661 | 1.1180 | -0.0420 | 0.01410 | 0.0020 | 0.8988 | 1.0420 (0) |
| mnli-wic | 0.7 | 0.7482 | 0.6160 | 0.9083 | 0.8851 | 0.1033 | 0.01386 | 0.0005 | 0.7766 | 0.8967 (0) |
| mnli-snli | 0.5 | 0.7990 | 0.8096 | 0.9699 | 0.9303 | 0.0499 | 0.01743 | 0.0046 | 0.7506 | 0.9501 (0) |
| mnli-scitail | 0.5 | 0.7533 | 0.8336 | 0.9145 | 0.8932 | 0.0961 | 0.01423 | 0.0016 | 0.7169 | 0.9039 (0) |
| mnli-ag_news | 0.7 | 0.7340 | 0.9049 | 0.8910 | 0.9640 | 0.0725 | 0.01229 | 0.0006 | 0.7239 | 0.9275 (0) |
| mnli-imdb | 0.7 | 0.7807 | 0.8513 | 0.9478 | 0.9567 | 0.0477 | 0.01367 | 0.0005 | 0.8155 | 0.9523 (0) |
| mnli-trec | 0.7 | 0.7803 | 0.7500 | 0.9473 | 0.7764 | 0.1381 | 0.01225 | 0.0007 | 0.4890 | 0.8619 (0) |
| mnli-yelp_polarity | 0.7 | 0.7372 | 0.9245 | 0.8950 | 0.9828 | 0.0611 | 0.01359 | 0.0013 | 0.8538 | 0.9389 (0) |
| qnli-rte | 0.5 | 0.8559 | 0.6029 | 0.9535 | 0.9382 | 0.0541 | 0.02060 | 0.0042 | 0.9102 | 0.9459 (0) |
| qnli-wic | 0.7 | 0.8331 | 0.6364 | 0.9280 | 0.9144 | 0.0788 | 0.01868 | 0.0009 | 0.8739 | 0.9212 (0) |
| qnli-snli | 0.5 | 0.8614 | 0.7870 | 0.9596 | 0.9044 | 0.0680 | 0.01711 | 0.0021 | 0.7477 | 0.9320 (0) |
| qnli-scitail | 0.5 | 0.8602 | 0.8919 | 0.9582 | 0.9556 | 0.0431 | 0.01923 | 0.0026 | 0.8513 | 0.9569 (0) |
| qnli-ag_news | 0.7 | 0.7904 | 0.9262 | 0.8805 | 0.9867 | 0.0664 | 0.01410 | 0.0008 | 0.8432 | 0.9336 (0) |
| qnli-imdb | 0.7 | 0.8768 | 0.8773 | 0.9768 | 0.9860 | 0.0186 | 0.01582 | 0.0009 | 0.9042 | 0.9814 (0) |
| qnli-trec | 0.7 | 0.7820 | 0.9560 | 0.8711 | 0.9896 | 0.0696 | 0.01546 | 0.0015 | 0.8518 | 0.9304 (0) |
| qnli-yelp_polarity | 0.5 | 0.8210 | 0.9297 | 0.9146 | 0.9883 | 0.0486 | 0.01461 | 0.0005 | 0.8627 | 0.9514 (0) |
| rte-wic | 0.7 | 0.6029 | 0.6614 | 0.9382 | 0.9505 | 0.0557 | 0.02338 | 0.0029 | 0.9207 | 0.9443 (0) |
| rte-snli | 0.5 | 0.6715 | 0.8110 | 1.0449 | 0.9319 | 0.0116 | 0.01811 | 0.0029 | 0.7870 | 0.9884 (0) |
| rte-scitail | 0.7 | 0.6643 | 0.8919 | 1.0337 | 0.9556 | 0.0053 | 0.02325 | 0.0046 | 0.9425 | 0.9947 (0) |
| rte-ag_news | 1.0 | 0.5343 | 0.9349 | 0.8315 | 0.9959 | 0.0863 | 0.01851 | 0.0006 | 0.9183 | 0.9137 (0) |
| rte-imdb | 0.7 | 0.6173 | 0.8883 | 0.9607 | 0.9983 | 0.0205 | 0.01927 | 0.0024 | 0.9383 | 0.9795 (0) |
| rte-trec | 0.7 | 0.5776 | 0.9700 | 0.8989 | 1.0041 | 0.0485 | 0.01877 | 0.0007 | 0.9352 | 0.9515 (0) |
| rte-yelp_polarity | 0.5 | 0.5668 | 0.9313 | 0.8820 | 0.9900 | 0.0640 | 0.01692 | 0.0007 | 0.8995 | 0.9360 (0) |
| wic-snli | 0.7 | 0.6489 | 0.8040 | 0.9324 | 0.9239 | 0.0718 | 0.01664 | 0.0014 | 0.8205 | 0.9282 (0) |
| wic-scitail | 1.0 | 0.5502 | 0.8696 | 0.7905 | 0.9318 | 0.1388 | 0.02042 | 0.0014 | 0.8906 | 0.8612 (0) |
| wic-ag_news | 0.7 | 0.6834 | 0.9332 | 0.9820 | 0.9941 | 0.0120 | 0.01642 | 0.0001 | 0.9407 | 0.9880 (0) |
| wic-imdb | 0.7 | 0.6489 | 0.8737 | 0.9324 | 0.9819 | 0.0428 | 0.01645 | 0.0004 | 0.9331 | 0.9572 (0) |
| wic-trec | 0.7 | 0.6552 | 0.9580 | 0.9414 | 0.9917 | 0.0334 | 0.01819 | 0.0011 | 0.9274 | 0.9666 (0) |
| wic-yelp_polarity | 0.5 | 0.6426 | 0.9246 | 0.9234 | 0.9829 | 0.0468 | 0.01504 | 0.0008 | 0.9128 | 0.9532 (0) |
| snli-scitail | 0.5 | 0.8152 | 0.8957 | 0.9367 | 0.9597 | 0.0518 | 0.01813 | 0.0020 | 0.6999 | 0.9482 (0) |
| snli-ag_news | 0.7 | 0.8026 | 0.9351 | 0.9222 | 0.9962 | 0.0408 | 0.01511 | 0.0009 | 0.8305 | 0.9592 (0) |
| snli-imdb | 0.7 | 0.8549 | 0.8673 | 0.9824 | 0.9747 | 0.0215 | 0.01479 | 0.0010 | 0.8817 | 0.9785 (0) |
| snli-trec | 0.7 | 0.8389 | 0.9620 | 0.9639 | 0.9959 | 0.0201 | 0.01579 | 0.0004 | 0.8443 | 0.9799 (0) |
| snli-yelp_polarity | 0.5 | 0.7216 | 0.9291 | 0.8292 | 0.9877 | 0.0916 | 0.01503 | 0.0014 | 0.7608 | 0.9084 (0) |
| scitail-ag_news | 0.7 | 0.8213 | 0.9300 | 0.8800 | 0.9907 | 0.0646 | 0.01610 | 0.0004 | 0.8185 | 0.9354 (0) |
| scitail-imdb | 1.0 | 0.9103 | 0.8674 | 0.9753 | 0.9748 | 0.0249 | 0.01676 | 0.0010 | 0.9449 | 0.9751 (0) |
| scitail-trec | 1.0 | 0.8880 | 0.9580 | 0.9515 | 0.9917 | 0.0284 | 0.01742 | 0.0003 | 0.9411 | 0.9716 (0) |
| scitail-yelp_polarity | 0.7 | 0.8758 | 0.9364 | 0.9384 | 0.9954 | 0.0331 | 0.01526 | 0.0009 | 0.8757 | 0.9669 (0) |
| ag_news-imdb | 0.7 | 0.9363 | 0.8337 | 0.9975 | 0.9370 | 0.0328 | 0.01671 | 0.0008 | 0.9176 | 0.9672 (0) |
| ag_news-trec | 0.7 | 0.9271 | 0.9060 | 0.9877 | 0.9379 | 0.0372 | 0.01807 | 0.0008 | 0.8797 | 0.9628 (0) |
| ag_news-yelp_polarity | 0.7 | 0.8961 | 0.8969 | 0.9546 | 0.9534 | 0.0460 | 0.01550 | 0.0004 | 0.9123 | 0.9540 (0) |
| imdb-trec | 0.7 | 0.8169 | 0.9620 | 0.9181 | 0.9959 | 0.0430 | 0.01694 | 0.0006 | 0.9054 | 0.9570 (0) |
| imdb-yelp_polarity | 0.5 | 0.8847 | 0.9361 | 0.9943 | 0.9951 | 0.0053 | 0.02767 | 0.0135 | 0.9734 | 0.9947 (0) |
| trec-yelp_polarity | 0.7 | 0.9520 | 0.9314 | 0.9855 | 0.9901 | 0.0122 | 0.01531 | 0.0010 | 0.9412 | 0.9878 (0) |

## Deviations

# E1b deviations from PREREG.md (append-only; each entry timestamped KST)

## D1 — full-data MNLI and QQP (written 2026-09-25 21:03 KST, before any evaluation-set scoring)
MNLI and QQP retrained on full train minus the same 1,000 held-out examples, 3 epochs, same recipe otherwise, because the 60k
cap made the pre-registered published-reference check (−5pp vs HF run_glue bert-base dev) structurally unreachable; decided
before any evaluation-set scoring; the capped adapters are kept under a separate name and not used.

- Decision by the study owner at 2026-09-25 21:02 KST, after reading the pre-registered risk note (PREREG.md §2). At that point
  only training losses existed. No adapter had been scored on any evaluation set or held-out set.
- Integrity rule, thresholds, λ grid, analysis, and all other tasks' budgets are **unchanged**. BoolQ and RTE keep their
  pre-registered budgets and are excluded if they fail the check.
- New budgets (machine-readable: `deviations_d1.json`; code: `train_e1b_d1.py`, which differs from `train_e1b.py` only in
  reading these budgets and using the uncapped pool; `run_train_d1.sh`):
  - MNLI: 391,702 training examples (392,702 − 1,000 held-out), 3 epochs, 12,241 steps/epoch → **36,723 steps**, warmup 500
    (pre-registered: 60,000 examples, 5,625 steps).
  - QQP: 362,846 training examples (363,846 − 1,000 held-out), 3 epochs, 11,339 steps/epoch → **34,017 steps**, warmup 500
    (pre-registered: 60,000 examples, 5,625 steps).
  - The held-out 1,000 examples are identical to the pre-registered ones (asserted in code). The pre-registered step cap of 5,625 does not apply to these two tasks.
  - Total training steps: 54,288 → 113,403. Projected training GPU time is still far below the 12 h cap.
- The capped adapters from the original run are renamed `adapters/mnli_cap60k` and `adapters/qqp_cap60k` and are not used for
  integrity, predictors, merges, or verdicts. After the verdict they may be scored as a clearly labelled exploratory sensitivity check.


## Compute

Training (final adapters): 1.34 GPU-h; stages (s): {"stage0_gpu_wall_s": 140.8, "stage1_gpu_wall_s": 62.0, "stage2_gpu_wall_s": 6358.3, "stage3_wall_s": 15.3}.
