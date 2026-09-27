---
name: training-budget-estimator
description: Estimate (N, D, hours, GPU count) for a new transformer training run given compute budget and deployment constraints.
version: 1.0.0
phase: 7
lesson: 13
tags: [scaling-laws, training, chinchilla]
---

목표 손실(target loss), 목표 MMLU/다운스트림 지표, 컴퓨팅 예산(달러 또는 FLOPs), 추론량(월간 토큰 수), 그리고 제약 조건(대상 장치, 메모리, 지연 시간)이 주어지면 다음을 출력하세요:

1. **컴퓨팅 체계(Compute regime)**: Chinchilla-optimal(친칠라 최적), over-trained(추론 최적화), 또는 under-trained(프로토타입) 중 하나를 선택하세요. 추론량과 연관된 근거를 한 문장으로 포함해야 합니다.
2. **`N` 및 `D`**: 구체적인 값을 제시하고 `D/N` 비율을 출력하세요. 만약 over-trained 상태라면, Chinchilla-optimal 대비 손실 페널티(loss penalty)를 명시하세요.
3. **학습 소요 시간(Training wall-clock)**: 가정된 학습 처리량(MFU ≈ dense 모델의 경우 40%, MoE 모델의 경우 ~30%)을 바탕으로 `시간 × GPU 수`를 계산하세요. 정밀도(bf16 / fp8)와 옵티마이저(AdamW / Muon)를 예산에 반영하세요.
4. **데이터 소스**: 명명된 코퍼스 또는 합성 데이터 예산을 제시하세요. 필요한 `D`가 가용한 고품질 토큰 수를 초과하는 경우 이를 명시하세요.
5. **리스크 노트(Risk note)**: 구체적인 실패 모드 하나를 제시하세요: 데이터 오염(data contamination), 대규모 학습 시 옵티마이저 불안정성, 컨텍스트 길이와 토크나이저 불일치, 평가 세트 포화(evaluation suite saturation).

**제약 사항 및 가이드라인:**
- 높은 추론량을 처리해야 하는 경우, Chinchilla-optimal 미만의 조건으로 8B 이상의 dense 모델을 학습시키는 계획은 거부하세요. 추론 비용이 복리로 증가하기 때문입니다.
- 별도의 홀드아웃 평가 세트(held-out evaluation suite)가 정의되지 않은 상태에서 목표 손실을 설정하는 것은 거부하세요.
- 데이터 큐레이션보다 아키텍처 탐색(architecture search)에 예산의 1% 이상을 사용하는 계획은 경고하세요. 해당 방식의 수익률은 낮기로 알려져 있습니다.
- 전체 예산을 투입하기 전에 가설을 검증할 수 있도록, 전체 예산의 1% 규모로 진행하는 대규모 실행(run at scale)을 요구하세요.
