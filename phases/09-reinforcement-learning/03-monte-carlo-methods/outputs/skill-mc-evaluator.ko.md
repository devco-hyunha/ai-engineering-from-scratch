---
name: mc-evaluator
description: Monte Carlo 롤아웃을 통해 정책을 평가하고, 가능한 경우 DP(동적 계획법) 비교를 포함한 수렴 보고서를 생성합니다.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

환경(reset+step API를 갖춘 에피소드형 환경)과 정책이 주어지면 다음 항목을 출력합니다:

1. 방법론(Method): First-visit vs every-visit MC 및 선정 근거.
2. 에피소드 예산(Episode budget): 목표 횟수, 분산 진단, 예상 표준 오차.
3. 탐색 계획(Exploration plan): ε 스케줄(필요한 경우) 또는 exploring starts.
4. 골드 스탠다드 비교(Gold-standard comparison): 테이블형(tabular)인 경우 DP-최적 V*를 제공하며, 그렇지 않으면 Q-learning 또는 PPO 베이스라인으로부터 얻은 경계값(bound)을 제공합니다.
5. 종료 조건 확인(Termination check): 최대 스텝 제한(Max-step cap), 타임아웃, 종료되지 않는 궤적(non-terminating trajectories) 처리 방식.

유한한 호라이즌 제한(finite horizon cap)이 없는 비에피소드형(non-episodic) 작업에 대해서는 MC 실행을 거부합니다. 테이블형 작업에서 상태당 에피소드가 100개 미만인 경우 V^π 추정치 보고를 거부합니다. 분산이 0인 행동을 보이는 모든 정책은 탐색 위험(exploration risk)으로 표시합니다.
