# Week-1 LoRA Merge Certificate — θ★ / conflict follow-up

Eval n=512, subspace A, `bert-base-uncased`, LoRA r=8.

## A) Higher-interference pair

### Hub search — no θ_min < 30°
Author `prateeky2806` GLUE LoRAs:

| pair | n_fail@30° | min θ_min | mean θ_min | n&lt;60° |
|---|---:|---:|---:|---:|
| `mnli_rte` | 0 | 69.9 | 78.5 | 0 |
| `mnli_qnli` | 0 | 66.5 | 77.3 | 0 |
| `mnli_qqp` | 0 | 59.7 | 77.0 | 1 |
| `qnli_rte` | 0 | 53.7 | 74.7 | 1 |
| `sst2_cola` | 0 | 58.0 | 75.6 | 2 |

Closest: **qnli_rte** (~53.7°) — still 0 FAIL at θ★=30°.

### Training blocker
CPU train ~17s/step; fixed `nyu-mll/glue`. Killed mid-run; `adapters/train_*` incomplete — not used.

### Working conflict pair (constructed)
Align n_shared=3 B-columns of SST-2 LoRA to `orth(B_MNLI)`.

| | |
|---|---|
| Adapters | `adapters/conflict_mnli`, `adapters/conflict_sst2_shared` |
| Summary | `artifacts/conflict_A/summary.json` |
| n_pass / n_fail @30° | **0 / 73** |
| θ_min | min=0.0000°, mean=0.0018°, n&lt;30°=73 |
| MNLI sum → corrected | **0.8184 → 0.8145** |
| SST-2 sum → corrected | **0.6055 → 0.6152** |
| angle–metric gap | **YES** |

## B) θ★ sweep (`artifacts/theta_sweep/summary.csv`)

### Original Hub MNLI+SST-2

| θ★ | n_pass | n_fail | min θ_min | MNLI sum | MNLI corr | SST-2 sum | SST-2 corr | gap? |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20.0 | 73 | 0 | 65.84 | 0.7969 | 0.7969 | 0.8770 | 0.8770 | False |
| 30.0 | 73 | 0 | 65.84 | 0.7969 | 0.7969 | 0.8770 | 0.8770 | False |
| 45.0 | 73 | 0 | 65.84 | 0.7969 | 0.7969 | 0.8770 | 0.8770 | False |
| 60.0 | 73 | 0 | 65.84 | 0.7969 | 0.7969 | 0.8770 | 0.8770 | False |
| 75.0 | 61 | 12 | 65.84 | 0.7969 | 0.7949 | 0.8770 | 0.8770 | True |

### Conflict constructed

Cert counts recomputed per θ★; GLUE metrics reused from conflict_A@30° (identical U★ when θ_min≈0). Per-θ★ full re-eval aborted (CPU re-SVD).

| θ★ | n_pass | n_fail | min θ_min | MNLI sum | MNLI corr | SST-2 sum | SST-2 corr | gap? |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20.0 | 0 | 73 | 0.00 | 0.8184 | 0.8145 | 0.6055 | 0.6152 | True |
| 30.0 | 0 | 73 | 0.00 | 0.8184 | 0.8145 | 0.6055 | 0.6152 | True |
| 45.0 | 0 | 73 | 0.00 | 0.8184 | 0.8145 | 0.6055 | 0.6152 | True |
| 60.0 | 0 | 73 | 0.00 | 0.8184 | 0.8145 | 0.6055 | 0.6152 | True |
| 75.0 | 0 | 73 | 0.00 | 0.8184 | 0.8145 | 0.6055 | 0.6152 | True |

## Angle–metric gap (correction ≠ sum)

| setting | gap? |
|---|---|
| Original θ★ ∈ {20,30,45,60} | **No** |
| Original θ★ = 75° | **Yes** (12 FAIL; MNLI 0.7969→0.7949) |
| Conflict θ★ ∈ {20,30,45,60,75} | **Yes** (MNLI & SST-2) |

## Paths
- `artifacts/conflict_A/summary.json`
- `artifacts/hub_pair_probe.json`
- `artifacts/theta_sweep/summary.csv`
- `artifacts/theta_sweep/report.md`
