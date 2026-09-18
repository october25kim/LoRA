# MNLI seed7 × seed42 (ubuntu-4070)

Trained on RTX 4070 Ti SUPER: `bert-base-uncased` LoRA r=8, α=16, 20k train samples, 1 epoch, batch 32.

| θ★ | PASS | FAIL | MNLI sum | MNLI corrected | gap |
|---:|---:|---:|---:|---:|:---:|
| 30 | 73 | 0 | 0.4629 | 0.4629 | no |
| 45 | 66 | 7 | 0.4629 | 0.4570 | yes |
| 60 | 26 | 47 | 0.4629 | 0.4531 | yes |
| 75 | 5 | 68 | 0.4629 | 0.4434 | yes |

θ_min min ≈ 39.95° across layers. At the paper default θ★=30°, no natural FAIL for this same-task different-seed pair.
