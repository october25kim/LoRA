# Hub-ish local training follow-up (ubuntu-4070)

Recipe: lr=1e-3, fp16, warmup_steps=500, r=8, α=16, 3 epochs MNLI / 5 epochs RTE.

## Single adapters
| adapter | n=512 | full val |
|---|---:|---:|
| mnli_s7_hubish | 0.461 | 0.452 |
| mnli_s42_hubish | 0.523 | 0.490 |
| Hub MNLI control | 0.811 | — |

## Certificates
| pair | θ★ | PASS | FAIL | θ_min min | MNLI sum→corr |
|---|---:|---:|---:|---:|---|
| seed hubish | 30 | 71 | 2 | 20.6 | 0.428→0.428 |
| mnli+rte local | 30 | 73 | 0 | 66.3 | 0.465→0.465 |

Natural FAILs: layer.11.output.dense (26.5°), pooler.dense (20.6°).
