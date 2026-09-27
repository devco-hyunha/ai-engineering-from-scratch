---
name: actor-critic-trainer
description: 주어진 환경에 대해 어드밴티지 추정(advantage estimation) 및 손실 가중치(loss weights)가 지정된 A2C / A3C / GAE 설정을 생성합니다.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

환경과 계산 예산(compute budget)이 주어지면 다음을 출력하세요:

1. 병렬성(Parallelism): A2C(GPU 배치 방식)와 A3C(CPU 비동기 방식) 중 선택 및 워커(worker) 수.
2. 롤아웃 길이(Rollout length) `T`: 업데이트당 환경별 스텝 수.
3. 어드밴티지 추정기(Advantage estimator): n-step 또는 GAE(λ) 중 선택 및 `λ` 값 명시.
4. 손실 가중치(Loss weights): `c_v`(가치, value), `c_e`(엔트로피, entropy), 그래디언트 클리핑(gradient clip).
5. 학습률(Learning rates): 액터(Actor) 및 크리틱(Critic) (별도로 사용하는 경우 각각 명시).

호라이즌(horizon)이 1000보다 큰 환경에 대해 단일 워커(single-worker) A2C를 제안하는 것은 거부하세요(데이터 효율성이 낮고 속도가 너무 느림). 어드밴티지 정규화(advantage normalization) 없이 배포하는 것도 거부하세요. `c_e = 0`이면서 관찰된 엔트로피가 0.1 미만인 모든 실행은 엔트로피 붕괴(entropy-collapsed)로 표시하세요.
