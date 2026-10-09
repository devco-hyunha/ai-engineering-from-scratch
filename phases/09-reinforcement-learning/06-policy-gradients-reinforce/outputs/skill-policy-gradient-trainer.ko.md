---
name: policy-gradient-trainer
description: 주어진 작업에 대해 REINFORCE / actor-critic / PPO 학습 구성을 생성하고 분산(variance) 문제를 진단합니다.
version: 1.0.0
phase: 9단계
lesson: 06강
tags: [rl, policy-gradient, reinforce]
---

환경(이산/연속 행동, 기간, 보상 통계)이 주어지면 다음을 출력합니다:

1. 정책 헤드. Softmax(이산) 또는 Gaussian(연속)과 매개변수 개수.
2. 기준선(Baseline). 없음(vanilla), 이동 평균, 학습된 `V̂(s)`, 또는 A2C 크리틱.
3. 분산 제어. Reward-to-go 기본 활성화, 리턴 정규화, 기울기 클리핑 값.
4. 엔트로피 보너스. 계수 β와 감쇠 스케줄.
5. 배치 크기. 업데이트당 에포크 수; 온폴리시(on-policy) 데이터 신선도 계약.

500 스텝 이상의 기간에 대해 REINFORCE-no-baseline을 거부합니다. Softmax 헤드를 사용하는 연속 행동 제어는 거부합니다. `β = 0`이 있고 관측된 정책 엔트로피가 0.1 미만인 모든 실행은 엔트로피 붕괴(entropy-collapsed)로 플래그를 지정합니다.
