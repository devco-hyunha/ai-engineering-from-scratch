---
name: sequence-architecture-picker
description: Pick sequence architecture (RNN, transformer, SSM, hybrid) given length, throughput, and training budget.
version: 1.0.0
phase: 7
lesson: 1
tags: [transformers, architecture, rnn, ssm]
---

시퀀스 문제(최대 길이, 배치 형태, 할당된 학습 토큰 수, 추론 지연 시간 목표, 디바이스 클래스)가 주어지면 다음을 출력하세요:

1. **주요 아키텍처(Primary architecture).** 다음 중 하나를 선택: `transformer`, 상태 공간 모델(`state-space model`, Mamba/RWKV), 하이브리드 `SSM+attention`, `RNN`. 지배적인 제약 조건과 연결하여 한 문장으로 이유를 기술하세요.
2. **컨텍스트 길이 전략(Context length strategy).** `transformer`인 경우: 전체 어텐션 컷오프(full attention cutoff), 슬라이딩 윈도우 크기, `RoPE` 스케일링 계수. `SSM`인 경우: 스캔 청크 크기(scan chunk size). `RNN`인 경우: 은닉층 너비(hidden width).
3. **학습 FLOP 프로필(Training FLOP profile).** 아키텍처와 컨텍스트를 기반으로 토큰당 근사 FLOPs를 계산하고, 해당 사양이 컴퓨팅 예산에 부합하는지 명시하세요.
4. **추론 메모리 프로필(Inference memory profile).** `transformer`의 경우 `KV cache`, `SSM`의 경우 상태 크기(state size), `RNN`의 경우 토큰당 메모리. 대상 디바이스가 배치 크기 1을 수용할 수 있는지 여부를 표시하세요.
5. **리스크 노트(Risk note).** 해당 사양 규모에서 이 선택이 가질 수 있는 알려진 특정 실패 모드(예: Flash Attention 없이 24GB GPU에서 64K 컨텍스트를 처리할 때 `transformer`의 OOM 발생)를 기술하세요.

1B 토큰 이상의 학습 실행에 대해 순수 `RNN`을 추천할 때는 반드시 그래디언트 흐름(gradient-flow) 및 병렬성 페널티(parallelism penalties)를 명시해야 하며, 그렇지 않으면 추천을 거부하세요. 64K 이상의 컨텍스트에 대해 전체 어텐션(full-attention) `transformer`를 추천할 때는 `O(N^2)` 메모리 비용을 명시해야 하며, 그렇지 않으면 추천을 거부하세요. 출시된 지 12개월 미만인 완전히 새로운 아키텍처를 프로덕션용으로 추천할 때는 반드시 대체안(fallback)을 명시해야 하며, 그렇지 않으면 추천을 거부하세요.
