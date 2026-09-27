---
name: policy-gradient-trainer
description: 주어진 작업에 대한 REINFORCE / actor-critic / PPO 학습 설정을 생성하고 분산(variance) 문제를 진단합니다.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

환경(이산/연속 행동, 호라이즌, 보상 통계)이 주어지면 다음을 출력하세요:

1. 정책 헤드(Policy head): 파라미터 수를 포함한 Softmax(이산형) 또는 Gaussian(연속형).
2. 베이스라인(Baseline): 없음(vanilla), 이동 평균(running mean), 학습된 `V̂(s)`, 또는 A2C critic.
3. 분산 제어(Variance controls): 기본적으로 Reward-to-go 적용, 리턴 정규화(return normalization), 그래디언트 클리핑(gradient clipping) 값.
4. 엔트로피 보너스(Entropy bonus): 계수 $\beta$ 및 감쇠 스케줄(decay schedule).
5. 배치 크기(Batch size): 업데이트당 에피소드 수; 온폴리시(on-policy) 데이터 신선도 계약.

호라이즌이 500단계를 초과하는 경우 REINFORCE-no-baseline을 거부하세요. Softmax 헤드를 사용하는 연속 행동 제어(continuous-action control)를 거부하세요. `β = 0`이고 관찰된 정책 엔트로피가 0.1 미만인 모든 실행은 엔트로피 붕괴(entropy-collapsed)로 표시하세요.
