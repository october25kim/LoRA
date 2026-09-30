# LoRA Merge Certificate (1주차)

두 개의 학습 완료 LoRA를 **재학습 없이** 병합할 때, 레이어마다 부분공간 주성분 각도로 **PASS/FAIL 인증서**를 만들고 FAIL만 닫힌 식으로 보정합니다.

- 태스크1(앵커): **MNLI**
- 태스크2: **SST-2**
- 기본 베이스: `bert-base-uncased`, 랭크 `r=8`
- 임계각: **θ★ = 30°** (전역 고정)
- 부분공간: 정의 **A** (`orth(B)`) / 정의 **B** (`ΔW=BA`의 좌특이벡터)

공식은 `METHOD.md`(상위 `lora-merge-cert/METHOD.md`)를 그대로 따릅니다.

## 설치

```bash
cd lora-merge-certificate  # or clone root
python -m venv .venv && source .venv/bin/activate   # 선택
pip install -r requirements.txt
pip install -e .
```

## Dry-run (네트워크/Hub 불필요)

합성 랜덤 LoRA A/B로 인증서 표·산점도·요약 JSON을 생성합니다.

```bash
cd lora-merge-certificate  # or clone root
python scripts/week1_run.py --dry-run --subspace A --theta-star-deg 30 --output-dir artifacts
```

정의 B로 돌리려면 `--subspace B`를 넣습니다.

## Real-run (HF 베이스 + PEFT 어댑터)

공개 Hub에 MNLI/SST-2용 BERT LoRA가 안정적으로 잡히지 않으면 아래 스텁으로 직접 학습하세요.

```bash
# (GPU 권장) 어댑터 학습 스텁
python scripts/train_lora_glue.py --task mnli --output-dir adapters/mnli-lora --r 8
python scripts/train_lora_glue.py --task sst2 --output-dir adapters/sst2-lora --r 8

# 병합 + (가능하면) GLUE eval
python scripts/week1_run.py \
  --base-model bert-base-uncased \
  --lora-mnli adapters/mnli-lora \
  --lora-sst2 adapters/sst2-lora \
  --subspace A \
  --theta-star-deg 30 \
  --output-dir artifacts
```

### 권장 Hub 어댑터 ID

동일 작성자·동일 `r=8`·동일 `target_modules=['query','key','value','dense']`·`bert-base-uncased` 페어:

| 역할 | Hub ID | 비고 |
|------|--------|------|
| MNLI (앵커) | `prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001` | r=8, α=16 |
| SST-2 | `prateeky2806/bert-base-uncased-sst2-lora-epochs-2-lr-0.0005` | r=8, α=16 |

대체 MNLI: `mia-project-2025/bert-base-uncased-LoRA-glue-mnli` (서브폴더 `bert-lora-glue-mnli`, r=8, α=64, 동일 target_modules).

Hub 404/호환 실패 시:

```bash
python scripts/train_lora_glue.py --task mnli --output-dir adapters/mnli-lora --r 8
python scripts/train_lora_glue.py --task sst2 --output-dir adapters/sst2-lora --r 8
```

예시 real-run:

```bash
python scripts/week1_run.py \
  --base-model bert-base-uncased \
  --lora-mnli prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001 \
  --lora-sst2 prateeky2806/bert-base-uncased-sst2-lora-epochs-2-lr-0.0005 \
  --subspace A --theta-star-deg 30 --output-dir artifacts
```

> PEFT `target_modules`가 서로 맞아야 레이어 페어가 수집됩니다. 학습 스텁은 `query,key,value,dense`를 사용합니다.

## 산출물 (`artifacts/`)

| 파일 | 의미 |
|------|------|
| `certificate_table.csv` / `.md` | 레이어별 PASS/FAIL, θ_min(°), overlap, n_shared |
| `theta_vs_mnli_drop.png` | x=θ_min, y=MNLI 정확도 하락(프록시) 산점도 |
| `summary.json` | PASS/FAIL 개수, 산술합 vs 인증서 보정 요약 숫자 |

- **PASS** (`θ_min ≥ 30°`): `ΔW = ΔW1 + ΔW2` (그대로 합)
- **FAIL** (`θ_min < 30°`): `ΔW2_corr = (I − P_{S★}) ΔW2`, `ΔW = ΔW1 + ΔW2_corr`

Dry-run의 accuracy 숫자는 Hub/데이터 없이 만든 **휴리스틱 프록시**입니다 (`fake: true`).

## 테스트

```bash
cd lora-merge-certificate  # or clone root
pytest -q
```

순수 PyTorch 단위 테스트(네트워크 없음): 각도/투영/PASS 항등/FAIL 공유방향 제거.

## 패키지 API

```python
from lora_merge_cert import (
    certify_and_merge_layer,
    principal_angles,
    extract_basis,
    merge_lora_models,
)
```

## 라이선스

MIT, see `LICENSE`. Base models and datasets keep their own licenses.

## Segment-ID (token_type_ids) convention — important for evaluation

Evaluate every adapter with the segment-id convention it was **trained** with. `lora_merge_cert.eval.evaluate_glue(..., segment_ids=...)` takes the mode explicitly:

| mode | meaning | adapters |
|---|---|---|
| `"bert"` | pass the tokenizer's `token_type_ids` | trained with `scripts/train_lora_glue.py` (`adapters/*_hubish`, `adapters/mnli_seed*`, `conflict_full_*`) |
| `"none"` | drop `token_type_ids` (all-zero segments) | `prateeky2806/*` Hub adapters and `conflict_mnli` / `conflict_sst2_shared` built from them |

`resolve_segment_ids(adapter)` picks the mode in this order: a `segment_ids.json` file in the adapter dir (`train_lora_glue.py` now writes one), then `SEGMENT_ID_REGISTRY`, then `conflict_meta.json` provenance. If none of these apply, it falls back to `"bert"` and issues a loud warning. Calling with `segment_ids=None` keeps the legacy behaviour (`"none"`) and also warns. The scripts accept `--segment-ids {auto,bert,none}`.

Before this fix (2026-09-25), the evaluator always used `"none"`. The locally trained adapters were therefore mis-evaluated (mnli_s7_hubish 0.452 → 0.821, mnli_s42_hubish 0.490 → 0.824, rte_s42_hubish 0.433 → 0.675). Details: `artifacts/seed_fix_segid/`, `artifacts/e1_predictive/diag/`.

## Data and code availability (Neurocomputing submission)

This repository accompanies "Label-Free Choice of the Merge Coefficient and the Merge Decision for LoRA Adapters:
A Pre-Registered Study". `artifacts/` holds, per study (E1–E7), the pre-registration documents with their SHA-256
hashes and time stamps, deviations logs, verdicts, pair-level result files and the analysis scripts that produce
every number in the paper. E7 is exploratory/post hoc (plan hashed before evaluation).

Trained LoRA adapters: not in this repository (about 1.9 GB of fp32 `adapter_model.safetensors`). They will be
published on the Hugging Face Hub with a `MANIFEST.sha256`, and the link will be added here. The sha256 of every
adapter is already recorded in each study's `predictors_aux_*.json` (`adapter_sha256`).

Code: MIT (see `LICENSE`). Base models (bert-base-uncased, roberta-base, Qwen2.5-0.5B/1.5B) and datasets keep their
own licenses.

### Studies added for the submission (E4–E7)

| study | backbone / scope | start here |
|---|---|---|
| `artifacts/e4a_roberta/` | E4a: RoBERTa-base, 14 tasks × 2 seeds | `PREREG_E4A.md` (+`prereg_e4a.sha256`), `DEVIATIONS_E4A.md`, `VERDICT_E4A.md`; code `e4a.py`, `e4a_analysis.py`, `make_verdict_e4a.py` |
| `artifacts/e4b_decoder/` | E4b: Qwen2.5-0.5B, 14 tasks × 2 seeds | `PREREG_E4B.md` (+`prereg_e4b.sha256`), `DEVIATIONS_E4B.md`, `VERDICT_E4B.md`; code `e4b.py`, `e4b_analysis.py` |
| `artifacts/e5_decoder_ext/` | E5: decoder extensions (e5a–e5d) | `PREREG_E5.md`, `PREREG_RECORD_E5.txt`, `DEVIATIONS_E5.md`, `VERDICT_E5.md`; code `e5.py`, `e5_analysis.py` |
| `artifacts/e6_decoder2/` | E6: Qwen2.5-1.5B confirmatory, label-free λ and merge decision | `results/PREREG_E6.md` (+`prereg_e6.sha256`, `PREREG_RECORD_E6.txt`), `VERDICT_E6_DECODER2.md`, `results/lam_confirm_e6.json`; code in `code/` |
| `artifacts/e6_lambda/` | exploratory label-free rule analysis | `RULES.md` (+`RULES.sha256`, `RULES_timestamp.txt`), `DEVIATIONS_E6.md`, `VERDICT_E6.md`, `OUTPUTS.sha256` |
| `artifacts/e7_baselines/` | E7 (exploratory/post hoc): label-free baselines, cost, 3-/4-adapter merges | `PLAN_E7.md` + `PLAN_E7.stamp`, `DEVIATIONS_E7.md`, `RESULTS_E7.md`, `summary_e7.json`; code `e7.py`, `e7_q05.py`, `e7_analysis.py` |
| `artifacts/analysis/` | cross-study numbers in the manuscript | `pooled_backbones.py`, `pooled_rho.py`, `noise_ceiling*.py`, `lambda_decomp*.py`, `compute_cost_u1_m3.py`, `e6_prereg_extract.py`, `e7_extract.py` |

Reproduction: verify the frozen inputs with `cd <dir> && shasum -a 256 -c <manifest>.sha256` (for
`e6_decoder2/results/code_e6.sha256`, run it from `e6_decoder2/code/`). The analysis and verdict scripts re-derive
the reported numbers from the committed pair-level files. Re-training needs the adapters or a GPU, and the launch
scripts assume `~/Desktop/Workspace/LoRA` (paths in hashed files are left unchanged). Logs, tokenized caches and
per-example prediction dumps are not included.

## Experiments

Pre-registered studies live under `artifacts/<study>/`. They are kept at their original paths because the pre-registration
hash manifests (`*.sha256`) use relative paths (e.g. `e1c_seed_diverse/code_e1c.sha256` hashes `../e1b_confirmatory/e1b.py`).
Every file listed in a manifest is committed **byte-identical** to the run machine; verify with
`cd artifacts/<study>/<dir> && shasum -a 256 -c <manifest>.sha256`.

Only code and small metadata are committed. **Not** included (regenerable or too large): trained adapters (`adapters/`, `adapters_s1/`),
tokenized-dataset caches (`cache/`), per-example prediction dumps (`preds/`), training logs (`logs/`), and the raw
`e3_baselines/e1b_run/results.jsonl` (1.5 MB; aggregated in `method_by_pair.csv` / `analysis.json`). Launch scripts assume the
repo at `~/Desktop/Workspace/LoRA` (hard-coded in hashed files, so left unchanged; pass `--repo-root` / edit `REPO` to run elsewhere).

| study | question | verdict | start here |
|---|---|---|---|
| `artifacts/e1_predictive/` | E1: does per-layer principal-angle overlap O_A predict task-arithmetic merge loss D on 21 Hub GLUE adapter pairs? | **INVALID** (single adapters >2 pp below model card); statistical outcome would be INCONCLUSIVE (ρ = 0.17) | `VERDICT.md`, `e1.py`, `analysis.json`, `pair_results.csv`, `predictors.csv`, `diag/` (segment-id diagnosis) |
| `artifacts/e1b_confirmatory/` | E1b: confirmatory re-run with 18 self-trained BERT LoRA tasks (14 valid → 91 pairs), held-out selection | H1 (O_A) **FAIL** (ρ = −0.19, CI [−0.52, 0.20]); H2 (task-vector cosine) **INCONCLUSIVE** | `PREREG.md`/`prereg.json` (+`prereg.sha256`), `deviations.md`, `VERDICT.md`, `post_run_notes.md`, `analysis.json`, `stage0.json`, `predictors.csv`, `pair_results.csv`, `method_comparison.md`, `fig_f1_scatter.png`, `fig_f2_heatmap.png`; code `e1b.py`, `tasks.py`, `train_e1b*.py`, `run_*.sh` |
| `artifacts/e3_baselines/` | E3: pre-registered merge-method baselines (TA, TIES, DARE, TSVM, KNOTS, PICO, GATE, …) on the E1b adapters | No method meets the pre-registered improvement rule vs TA; GATE ≡ TA (gate never active) | `code/PREREG_E3.md`, `code/PREREG_E3_AMENDMENTS.md`, `PREREG_E3_AMENDMENTS_ERRATUM.md` (+`.sha256`), `code/IMPLEMENTATION_NOTES.md`, `e1b_run/E3_E1B_REPORT.md`, `e1b_run/analysis.json`, `e1b_run/method_by_pair.csv`; pilot in `pilot_e1/`; code `code/e3.py`, `code/merges.py`, `code/test_merges.py` |
| `artifacts/e1c_seed_diverse/` | E1c: E1b replicated on mixed-seed cross-task pairs (seed-0 × seed-1 adapters) | H1c (O_A) **INCONCLUSIVE** (ρ = −0.02); H2c **INCONCLUSIVE** (ρ = 0.13) | `PREREG_E1C.md`/`prereg_e1c.json` (+`.sha256`), `DEVIATIONS_E1C.md`, `VERDICT_E1C.md`, `analysis_e1c.json`, `predictors_e1c.csv`, `pair_results_e1c.csv`, `fig_e1c_*.png`; code `e1c.py`, `e1c_analysis.py`, `make_verdict_e1c.py`, `train_e1c.py`, `run_*_e1c.sh` |
| `artifacts/seed_fix_segid/` | Re-evaluation of the seed-pair / MNLI×RTE certificates after the segment-id fix (`seg_bert` vs `seg_none`) | see `TABLE_SEED_FIXED.md` | `TABLE_SEED_FIXED.md`, `summary.json`, `seed_fix_segid.py` |
| `artifacts/lemma_check/` | Per-layer certificate for the `mnli_s7_hubish` × `mnli_s42_hubish` seed pair (θ★ = 30°, subspace A) | 2 / 73 layers FAIL | `seed_pair.json` |

Hash-manifest status: all manifests verify except those superseded by later, documented versions — `e1b_confirmatory/code_pipeline.sha256`
(`e1b.py` was patched for the stage-3 NaN-p guard; current hash in `code_stage3_rerun.sha256`, see `post_run_notes.md`) and
`e3_baselines/pilot_e1/code_at_launch.sha256` (pilot ran an earlier `code/` version). `e3_baselines/*/code_at_launch.sha256`
paths are relative to `e3_baselines/code/`.

### Software / hardware

Runs were produced with Python 3.11.15, torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.0, datasets 5.0.1,
NVIDIA driver 570.211.01, NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), Ubuntu 24.04.4.

### Withdrawn result

The constructed-conflict result stated in commit `80f5ec7` ("MNLI sum 0.398 → corrected 0.510") is **withdrawn** and should not be
cited; see `CHANGELOG.md`.
