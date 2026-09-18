# LoRA Merge Certificate — 방법 상세

이 노트는 아이디어 개요가 아니라 **방법**만 적는다.
입력은 이미 학습이 끝난 두 LoRA다. 재학습하지 않는다.

## 0. 한 줄

레이어마다 두 어댑터의 열공간을 잡고, 주성분 각도가 임계보다 작은 방향만 "공유 방향"으로 본다. 그 방향은 한 번만 넣고, 나머지는 그냥 더한다. 각도와 pass/fail 표가 인증서다.

## 1. 기호

베이스 가중치 W0 ∈ R^{m×n}.
태스크 i ∈ {1,2}의 LoRA: ΔWi = Bi Ai, Bi ∈ R^{m×ri}, Ai ∈ R^{ri×n}.
산술 merge: Wsum = W0 + ΔW1 + ΔW2.
방법이 고치는 것은 ΔW1+ΔW2 자리다. W0는 그대로.
레이어 L마다 독립. 트랜스포머면 WQ,WK,WV,WO,Wup,Wdown 각각이 한 레이어 단위.

## 2. 부분공간

정의 A (어댑터 출력 방향): Ui = orth(Bi)
정의 B (실제 업데이트 방향): ΔWi = Ũi Σi Ṽi^T, Ui = Ũi
정의 C는 데이터 필요 — 1주차 기본은 A 또는 B.

Ui^T Ui = I 가정.

## 3. 주성분 각도

U1^T U2 = P Σ Q^T, cos θk = σk, θk = arccos(σk) ∈ [0, π/2]
θmin = θ1, overlap = ||U1^T U2||_F^2
정준 벡터: uk^(1) = U1 pk, uk^(2) = U2 qk

## 4. 근사 공유 방향

집합론적 교집합은 쓰지 않음.
S★ = span{ uk^(1) : θk < θ★ }
θ★ 기본: 30° 또는 cos θ★ = 0.9. 전역 하나 고정. 레이어별 튜닝 금지.
P_S★ = U★ U★^T

## 5. 인증서

cert(L) = PASS if θmin(L) ≥ θ★, else FAIL
PASS: ΔW = ΔW1+ΔW2
FAIL만 보정

## 6. 보정 (닫힌 식)

기본(태스크1 앵커): ΔW2^corr = (I - P_S★) ΔW2, ΔWmerge = ΔW1 + ΔW2^corr
선택으로 평균/argmax share 버전 있음. 기본은 앵커.

## 7. 알고리즘

각 레이어:
1. dW1,dW2 = B1@A1, B2@A2
2. U1,U2 = left_basis (QR of B or left SVD of dW)
3. SVD(U1.T @ U2) → thetas
4. if any theta < theta_star: Ustar = qr(U1 @ P[:,idx]); dW2 -= Ustar @ (Ustar.T @ dW2)
5. dW = dW1 + dW2
6. log theta_min, overlap, FAIL if idx.any()

## 8–11. (이론/TIES) — 1주차는 산술합 vs 인증서 보정만 필수

## 12. 1주차 최소 단위

모델: BERT-base 또는 RoBERTa-base
태스크: MNLI LoRA, SST-2 LoRA, r=8
부분공간: 정의 A와 B
임계: θ★=30° 고정
출력: (1) 레이어별 인증서 표 (2) x=θmin, y=MNLI acc drop 산점도 (3) 산술합 vs 보정 두 숫자
