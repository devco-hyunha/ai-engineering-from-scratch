---
name: mha-configurator
description: 새로운 트랜스포머 모델을 위한 헤드 수, KV 헤드 수, 그리고 프로젝션 전략(MHA / MQA / GQA / MLA)을 추천합니다.
version: 1.0.0
phase: 7
lesson: 3
tags: [transformers, attention, mha, gqa]
---

트랜스포머 사양(파라미터 예산, 은닉층 크기 `d_model`, 목표 컨텍스트 길이, 추론 장치 메모리, 학습 vs 추론 우선순위)이 주어지면 다음을 출력합니다:

1. 프로젝션 변형(Projection variant): MHA, GQA, MQA, MLA 중 하나를 선택합니다. KV 캐시 제약 조건과 관련된 이유를 한 문장으로 포함합니다.
2. 헤드 기하학(Head geometry): `n_heads`, `n_kv_heads`, `d_head`를 제안합니다. 값은 `d_model = n_heads * d_head` 및 `n_heads % n_kv_heads == 0`을 만족해야 합니다.
3. KV 캐시 추정치(KV cache estimate): 선택한 변형에 대해 목표 컨텍스트 길이에서 레이어당 토큰당 바이트 수(fp16)를 계산합니다. 단일 배치가 목표 장치 메모리를 초과하는 경우 이를 표시합니다.
4. 초기화(Initialization): Q, K, V, O 행렬에 대한 Xavier / Kaiming 스케일을 제공합니다. 편향(bias) 항 포함 여부를 명시합니다(대부분의 2026년 모델은 이를 제거합니다).
5. 테스트 가능성 훅(Testability hook): 이 설정으로 학습된 2개 레이어 버전이 95% 이상의 정확도로 해결해야 하는 단일 합성 작업(예: 유도 헤드 패턴 `A B A ? → B`)을 제안합니다.

`d_head < 32`인 경우는 추천을 거부합니다. 어텐션 역학(attention dynamics)이 붕괴되기 때문입니다. 컨텍스트 길이가 32K를 초과하는 경우, KV 캐시 비용을 명시적으로 산출해야 하며, GQA 또는 MLA를 제안하지 않는 한 `n_heads > 16`인 MHA 추천을 거부합니다. 사용자가 명시적으로 벤치마킹을 요청하지 않는 한, 1B 미만 파라미터 모델에 대해 MLA를 제안하는 것을 거부합니다.
