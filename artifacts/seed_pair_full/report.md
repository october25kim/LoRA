# MNLI seed7 × seed42 — full train (3 epochs)

Machine: ubuntu-4070 (RTX 4070 Ti SUPER). Full GLUE MNLI train split, r=8, α=16, batch 32, 3 epochs.

| θ★ | PASS | FAIL | MNLI sum | MNLI corrected | gap |
|---:|---:|---:|---:|---:|:---:|
| 30 | 73 | 0 | 0.4668 | 0.4668 | no |
| 45 | 72 | 1 | 0.4668 | 0.4766 | yes |
| 60 | 13 | 60 | 0.4668 | 0.4707 | yes |
| 75 | 0 | 73 | 0.4668 | 0.4941 | yes |

θ_min min ≈ 42.4°, median ≈ 55.4°. Still no natural FAIL at paper default θ★=30°.
