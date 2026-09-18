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

MIT (연구용 Week-1 패키지)
