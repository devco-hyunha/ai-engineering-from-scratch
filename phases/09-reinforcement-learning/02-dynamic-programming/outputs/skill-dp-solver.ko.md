---
name: dp-solver
description: 정책 반복(policy iteration) 또는 가치 반복(value iteration)을 통해 소규모 테이블형 MDP를 정확하게 해결하고 수렴 동작을 보고합니다.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

모델이 알려진 MDP를 제공하면 다음 항목을 출력합니다:

1. 선택(Choice). 정책 반복(policy iteration)과 가치 반복(value iteration) 중 무엇을 사용할 것인지, 그리고 $|S|$, $|A|$, $\gamma$를 고려한 근거.
2. 초기화(Initialization). $V_0$ 및 초기 정책. 수렴 민감도.
3. 중단(Stopping). Sup-norm 허용 오차 $\epsilon$. 예상 스윕(sweep) 횟수.
4. 검증(Verification). 정확하게 계산된 $V^*(s_0)$. 추출된 탐욕 정책(greedy policy).
5. 활용(Use). 이 베이스라인을 샘플링 기반 방법론의 디버깅 또는 평가에 어떻게 활용할 것인지.

상태 공간(state spaces)의 크기가 $10^7$보다 큰 경우 DP 실행을 거부합니다. Sup-norm 확인 없이 수렴을 주장하는 것을 거부합니다. 무한 지평(infinite-horizon) 작업에서 $\gamma \ge 1$인 경우 보장 위반(guarantee violation)으로 표시합니다.
