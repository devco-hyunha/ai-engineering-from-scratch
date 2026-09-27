---
name: dqn-trainer
description: 이산 행동(discrete-action) 강화 학습 작업을 위한 DQN 학습 설정(버퍼, 타겟 동기화, ε 스케줄, 보상 클리핑)을 생성합니다.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

이산 행동 환경(관측 형태, 행동 수, 호라이즌, 보상 스케일)이 주어지면 다음을 출력하세요:

1. 네트워크(Network): 아키텍처(MLP / CNN / Transformer), 특징 차원(feature dim), 깊이(depth).
2. 리플레이 버퍼(Replay buffer): 용량(Capacity), 미니배치 크기(minibatch size), 웜업 크기(warmup size).
3. 타겟 네트워크(Target network): 동기화 전략(C 스텝마다 수행하는 hard sync 또는 soft $\tau$).
4. 탐험(Exploration): $\epsilon$ 시작값 / 종료값 / 스케줄 길이.
5. 손실(Loss): Huber vs MSE, 그래디언트 클리핑 값(gradient clip value), 보상 클리핑 규칙(reward clipping rule).
6. Double DQN: 명시적으로 비활성화해야 할 이유가 없는 한 기본적으로 활성화(On)합니다.

타겟 네트워크가 없거나, 리플레이 버퍼가 없거나, $\epsilon$ 값이 1로 유지되는 DQN은 생성을 거부하세요. 연속 행동(continuous-action) 작업은 거부하세요(SAC / TD3로 안내). 스텝당 평균 보상의 10배를 초과하는 보상 범위가 발견되면 클리핑(clipping) 또는 스케일 정규화(scale normalization)가 필요하다고 표시하세요.
