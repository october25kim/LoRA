# Next step: baselines + constructed conflict on full seeds

## Single-adapter MNLI baselines (ubuntu-4070)
| model | n=512 | full val |
|---|---:|---:|
| base (no adapter) | 0.338 | 0.320 |
| mnli_seed7_full | 0.506 | 0.494 |
| mnli_seed42_full | 0.506 | 0.486 |
| Hub `prateeky2806/...-mnli-lora-...` | **0.811** | — |

Local 3-epoch full train underfits vs Hub; merge ~0.47 was not an eval bug.

## Constructed conflict (share 3 B-cols of seed42 → seed7)
θ_min ≈ 0 on all 73 layers.

| θ★ | PASS | FAIL | MNLI sum | MNLI corrected |
|---:|---:|---:|---:|---:|
| 30 | 0 | 73 | 0.398 | 0.510 |
| 45 | 0 | 73 | 0.398 | 0.510 |
| 60 | 0 | 73 | 0.398 | 0.510 |

Certificate correction improves MNLI under forced shared directions.
