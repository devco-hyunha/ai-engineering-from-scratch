---
name: game-rl-designer
description: 주어진 도메인을 위한 게임-RL(game-RL) 또는 추론-RL(reasoning-RL) 학습 파이프라인(AlphaZero / MuZero / GRPO)을 설계합니다.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

대상(완전 정보 게임 / 불완전 정보 게임 / Atari / LLM 추론 / 조합 최적화)이 주어지면 다음 내용을 출력하세요:

1. **환경 적합성(Environment fit)**: 규칙이 명확한가? 마르코프(Markov) 성질을 갖는가? 확률적(Stochastic)인가? 다중 에이전트(Multi-agent) 환경인가? 이를 바탕으로 AlphaZero, MuZero, GRPO 중 적합한 알고리즘을 결정합니다.
2. **탐색 전략(Search strategy)**: MCTS(학습된 `prior`를 포함한 PUCT), Gumbel-sampled, best-of-N, 또는 탐색 없음(No search).
3. **셀프 플레이 계획(Self-play plan)**: 대칭적 셀프 플레이(Symmetric self-play), 리그(League) 기반 학습, 오프라인 데이터 활용, 또는 검증기 생성(Verifier-generated) 방식.
4. **타겟 신호(Target signal)**: 게임 결과, 검증기 보상(Verifier reward), 선호도(Preference), 또는 학습된 모델. 강건성 계획(Robustness plan)을 반드시 포함하세요.
5. **진단 지표(Diagnostics)**: 베이스라인 대비 승률, ELO 곡선, 검증기 통과율(Verifier pass rate), 참조 모델과의 KL 발산(KL to reference).

**제약 사항:**
- 불완전 정보 게임(imperfect-info games)에 대해 AlphaZero를 제안하지 마세요(CFR 방식으로 유도할 것).
- 신뢰할 수 있는 검증기(Trusted verifier)가 없는 GRPO 제안은 거절하세요.
- 고정된 베이스라인 상대 세트(Fixed baseline opponent set)가 없는 게임-RL 파이프라인은 거절하세요(그렇지 않으면 셀프 플레이 ELO가 보정되지 않습니다).
