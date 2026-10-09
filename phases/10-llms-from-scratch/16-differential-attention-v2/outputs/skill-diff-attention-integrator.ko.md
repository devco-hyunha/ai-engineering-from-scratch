---
name: diff-attention-integrator
description: 새로운 사전 학습 실행이나 LoRA 미세 조정에 Differential Attention V2를 추가하기 위한 통합 계획입니다.
version: 1.0.0
phase: 10단계
lesson: 16강
tags: [differential-attention, diff-transformer, long-context, flash-attention, pre-training, lora]
---

모델 아키텍처(hidden, heads, KV heads, layers, d_head), 목표 컨텍스트 길이, 환각(Hallucination) 또는 긴 컨텍스트 프로필(기존 평가에서의 실패 모드), 그리고 학습 예산(사용 가능한 토큰, GPU 시간)이 주어지면, DIFF V2를 위한 통합 계획을 생성합니다.

생성할 항목:

1. 통합 모드. 처음부터 하는 사전 학습, 중간 학습 중 아키텍처 교체, 또는 Q 투사(Q projections)에 대한 LoRA 미세 조정 중 하나를 선택합니다. 학습 예산과 사용 가능한 기존 가중치에 비추어 선택 이유를 정당화합니다.
2. 아키텍처 차이. 필드별 구체적인 변경 목록: 어떤 투사(projections)가 커지는지, 어떤 투사가 그대로 유지되는지, 추가되는 매개변수(Parameter) 수, 그리고 어텐션 블록에서 차감(subtraction)이 어디에 배치되는지 포함합니다. `lambda_init` 스케줄을 레이어 깊이별로 포함합니다(`0.8 - 0.6 * exp(-0.3 * (depth - 1))`는 논문에서의 기본값입니다. 레이어별 텔레메트리(telemetry)가 불안정성을 보이면 깊이별로 조정합니다).
3. 커널 선택. V2의 헤드 수(head-count)가 두 배가 되는 상황에서 FlashAttention 2 또는 3의 지원 여부를 확인합니다. 사용자가 재현성(reproducibility)을 위해 명시적으로 필요하지 않는 한 V1의 커스텀 커널(custom-kernel) 경로를 거부합니다.
4. 메모리 예산. KV 캐시(KV Cache)는 기준선(baseline)을 유지합니다(KV heads는 변경되지 않음). 토큰별 활성화 메모리 델타(extra Q heads, extra compute)를 계산합니다. 목표 컨텍스트에서의 절대적인 수치를 보고합니다.
5. 학습 안정성 계획. 모니터링할 항목을 설명합니다: 레이어별 `lambda` 드리프트(drift), 헤드별 어텐션 엔트로피(attention entropy), Q 투사(Q projections)의 기울기 분산(gradient variance). 텔레메트리(telemetry)가 발산(divergence)을 나타낼 경우 기준선 어텐션으로 롤백(Rollback)을 트리거해야 하는 특정 지표(metric)를 명시합니다.

하드 리젝트(Hard rejects):
- 사전 학습된 모델에 DIFF 어텐션을 추가하면서 지속 사전 학습(continued pre-training)을 하지 않는 경우. 출력 분포가 드리프트(drift)됩니다. 즉, 즉시 적용 가능한(drop-in) 수정이 아닙니다.
- 2026년 4월 이후의 모든 새로운 실행에 DIFF V1을 사용하는 경우. V2는 모든 측정된 차원에서 엄격하게 더 좋습니다.
- 긴 컨텍스트 학습 데이터를 활성화하지 않고 DIFF를 통합하는 경우. 이 혜택은 32k 이상에서만 나타납니다.
- 통제된 실험 없이 `lambda_init`를 음수 값으로 변경하는 경우. 음수 초기값은 노이즈 바닥(noise floor)보다 더 많이 차감하며 학습을 붕괴(collapse)시킵니다.

거절 규칙:
- 타겟 컨텍스트가 16k 미만인 경우, 통합을 거절하고 표준 어텐션을 권장합니다. 잡음 바닥(noise-floor) 논거로는 추가 매개변수 비용이 정당화되지 않습니다.
- 사용자가 장문 컨텍스트 평가 데이터(RULER, needle-in-haystack, MultiNeedle)를 제공할 수 없는 경우, 거절하고 먼저 보정 데이터를 요청합니다.
- 사용자가 FlashAttention-2 이전 스택을 사용하는 경우, 거절하고 통합을 시도하기 전에 스택을 업그레이드할 것을 권장합니다.

출력: 모드, 매개변수 수 차이, KV 캐시 영향, FlashAttention 확인, `lambda` 스케줄 및 3가지 지표 모니터링 보드를 나열한 한 페이지 분량의 통합 계획을 생성합니다. 아키텍처에 DIFF V2를 유지할지 되돌릴지 정당화하는 특정 장문 컨텍스트 평가 수치(RULER 64k의 백분율 포인트 차이 또는 동등한 지표)를 명시한 "성공 기준" 단락으로 마무리합니다.
