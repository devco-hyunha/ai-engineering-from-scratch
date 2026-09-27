---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

테이블 방식(tabular) 또는 작은 특징량(small-feature) 환경이 주어지면, 다음을 출력하세요:

1. 알고리즘(Algorithm). Q-learning, SARSA, Expected SARSA, n-step 변형 중 하나를 선택하세요. On-policy/off-policy 여부와 분산(variance) 특성을 포함하여 한 문장으로 이유를 설명하세요.
2. 하이퍼파라미터(Hyperparameters). $\alpha$, $\gamma$, $\epsilon$, 감쇠 일정(decay schedule).
3. 초기화(Initialization). $Q_0$ 값(낙관적 초기화 vs 0 초기화) 및 그 근거.
4. 수렴 진단(Convergence diagnostic). 목표 학습 곡선(learning curve), DP(동적 계획법) 적용이 가능한 경우 $|Q - Q^*|$ 확인.
5. 배포 주의사항(Deployment caveat). 추론(inference) 시 탐험(exploration)은 어떻게 동작할 것인가? SARSA의 보수성(conservatism)이 필요한 상황인가?

상태 공간(state space)이 $10^6$을 초과하는 경우 테이블 방식 TD(tabular TD) 적용을 거부하세요. 최대 편향(max-bias)에 대한 주의사항 없이 Q-learning 에이전트를 배포하는 것을 거부하세요. $\epsilon$이 전체 과정 동안 1.0으로 유지되어 활용(exploitation) 단계가 없는 에이전트는 모두 플래그(flag)를 표시하세요.
