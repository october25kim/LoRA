| method | mean norm. score | median | min | mean diff vs TA | task-block 95% CI | pair-bootstrap 95% CI | win/tie/loss | Wilcoxon p |
|---|---:|---:|---:|---:|---|---|---|---:|
| TA@lam* | 0.9520 | 0.9565 | 0.8107 | — | — | — | — | — |
| TIES-lite@lam*TA | 0.8813 | 0.9048 | 0.4890 | -0.0707 | [-0.1116, -0.0368] | [-0.0841, -0.0583] | 2/0/89 | 2.82e-16 |
| TIES-lite@own-lam | 0.9322 | 0.9376 | 0.6262 | -0.0198 | [-0.0348, -0.0105] | [-0.0261, -0.0142] | 12/0/79 | 4.02e-13 |
| gate30@lam*TA | 0.9520 | 0.9565 | 0.8107 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/91/0 | nan |
| gate30@own-lam | 0.9520 | 0.9565 | 0.8107 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/91/0 | nan |

Normalized score = mean over the two tasks of merged/single eval metric (= 1 − D for TA). Gate active (≥1 layer with θ_min < 30°) in 0/91 pairs; elsewhere gate ≡ TA (tie).
