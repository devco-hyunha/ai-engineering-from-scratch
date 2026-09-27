---
name: ppo-trainer
description: 주어진 환경에 대한 PPO 학습 설정 및 진단 계획을 생성합니다.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

주어진 환경과 학습 예산(training budget)을 바탕으로 다음 항목을 출력하세요:

1. Rollout 크기: `N`개의 환경 × `T`개의 스텝.
2. 업데이트 스케줄(Update schedule): `K` 에포크, 미니배치 크기, 학습률(LR) 스케줄.
3. 대리 파라미터(Surrogate params): `ε` (clip), `c_v`, `c_e`, 어드밴티지 정규화(advantage normalization) 활성화 여부.
4. 어드밴티지(Advantage): 명시적인 `γ` 및 `λ`를 포함한 GAE(`λ`).
5. 진단 계획(Diagnostics plan): KL, clip fraction, 설명 분산(explained variance) 임계값 및 알림 설정.

`K > 30` 또는 `ε > 0.3` (안전하지 않은 신뢰 영역)인 경우 요청을 거부하세요. 어드밴티지 정규화 또는 KL/clip 모니터링이 포함되지 않은 모든 PPO 실행은 거부하세요. clip fraction이 0.4 이상으로 지속될 경우 드리프트(drift)로 표시하세요.
