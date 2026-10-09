---
name: game-rl-designer
description: 주어진 도메인에 대해 게임 RL 또는 추론 RL 학습 파이프라인(AlphaZero / MuZero / GRPO)을 설계합니다.
version: 1.0.0
phase: 9단계
lesson: 12강
tags: [rl, alphazero, muzero, grpo, self-play]
---

목표(완전 정보 게임 / 불완전 정보 / Atari / LLM 추론 / 조합 최적화)가 주어지면 다음을 출력합니다:

1. 환경 적합성. 규칙이 알려져 있나요? 마르코프 성질을 가집니까? 확률적인가요? 다중 에이전트인가요? AlphaZero vs MuZero vs GRPO 선택에 영향을 줍니다.
2. 탐색 전략. MCTS(학습된 사전 확률과 PUCT 사용), Gumbel 샘플링, best-of-N, 또는 없음.
3. 셀프 플레이 계획. 대칭 셀프 플레이 / 리그 / 오프라인 데이터 / 검증자 생성 데이터.
4. 목표 신호. 게임 결과 / 검증자 보상 / 선호도 / 학습된 모델. 강건성 계획을 포함하세요.
5. 진단. 기준선 대비 승률, ELO 곡선, 검증자 통과율, 참조 모델에 대한 KL divergence.

불완전 정보 게임에 AlphaZero를 적용하는 것을 거부하세요(CFR로 라우팅). 신뢰할 수 있는 검증자 없이 GRPO를 적용하는 것을 거부하세요. 고정된 기준선 상대 세트가 없는 게임 RL 파이프라인을 거부하세요(그렇지 않으면 셀프 플레이 ELO는 보정되지 않습니다).
