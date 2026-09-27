---
name: mdp-modeler
description: 작업 설명이 주어지면 마르코프 결정 과정(MDP) 사양을 생성하고, 학습 전 공식화 위험을 식별합니다.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

주어진 작업(제어 / 게임 / 추천 / LLM 미세 조정)에 대해 다음을 출력하세요:

1. 상태(State). 정확한 특징 벡터(feature vector) 또는 텐서 사양. 마르코프 성질(Markov property)에 대한 근거 제시.
2. 행동(Action). 이산 집합(discrete set) 또는 연속 범위(continuous range). 차원(dimensionality).
3. 전이(Transition). 결정론적(deterministic), 알려진 모델이 있는 확률적(stochastic-with-known-model), 또는 샘플 기반(sample-only).
4. 보상(Reward). 함수 및 출처. 희소 보상(sparse) 대 형성된 보상(shaped). 종료 보상(terminal) 대 단계별 보상(per-step).
5. 할인율(Discount). 값 및 유효 지평(horizon)에 대한 근거 제시.

프레임 스태킹(frame-stacking) 또는 순환 상태(recurrent state)에 대한 명시적인 언급 없이 상태가 비마르코프(non-Markovian)인 MDP는 승인을 거부하세요. 목표 결과(target outcome) 관점에서 정의되지 않은 보상은 모두 승인을 거부하세요. 무한 지평(infinite-horizon) 작업에서 `γ ≥ 1.0`인 경우 경고를 표시하세요. 일반적인 단계별 보상보다 100배 이상 큰 보상 범위는 그래디언트 폭주(gradient explosion)의 잠재적 원인으로 경고를 표시하세요.
