# 액터-크리틱 — A2C와 A3C

> REINFORCE는 잡음이 심합니다. `V̂(s)`을 학습하는 크리틱을 추가하고, 이를 리턴에서 빼면 기대값은 동일하지만 분산이 훨씬 낮은 어드밴티지를 얻을 수 있습니다. 이것이 액터-크리틱입니다. A2C는 이를 동기적으로 실행하고, A3C는 여러 스레드에서 실행합니다. 둘 다 모든 현대적 심층 RL 방법의 개념적 모델입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 04강 (TD 학습), 9단계 · 06강 (REINFORCE)
**시간:** 약 75분

## 문제점

기본 REINFORCE는 작동하지만 분산이 매우 큽니다. 몬테카를로 리턴 `G_t`은 에피소드 간에 10배 이상 변동할 수 있습니다. 그 잡음을 `∇ log π`과 곱하고 평균 내면, DQN 업데이트로 훨씬 적은 횟수로 정책을 이동할 수 있는 거리와 같은 거리를 이동하기 위해 수천 개의 에피소드가 필요한 기울기 추정기가 생성됩니다.

분산은 원시 리턴을 사용해서 발생합니다. 기준선 `b(s_t)`을 빼면 — 상태의 함수, 학습된 가치 포함 — 기대값은 변하지 않고 분산이 감소합니다. 가장 다루기 쉬운 기준선은 `V̂(s_t)`입니다. 이제 `∇ log π`에 곱해지는 양은 *어드밴티지*입니다:

`A(s, a) = G - V̂(s)`

행동이 평균 이상의 리턴을 생성했다면 좋은 행동이고, 평균 이하라면 나쁜 행동입니다. 학습된 크리틱을 사용하는 REINFORCE는 *액터-크리틱*입니다. 크리틱은 액터에게 분산이 낮은 교사 역할을 합니다. 2015년 이후의 모든 심층 정책 방법 (A2C, A3C, PPO, SAC, IMPALA)이 이 방식입니다.

## 개념

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**두 개의 네트워크, 하나의 공유 손실:**

- **액터** `π_θ(a | s)`: 정책입니다. 행동을 위해 샘플링됩니다. 정책 기울기로 학습됩니다.
- **크리틱** `V_φ(s)`: 상태로부터 기대 리턴을 추정합니다. `(V_φ(s) - target)²`을 최소화하도록 학습됩니다.

**어드밴티지.** 두 가지 표준 형태가 있습니다:

- *MC 어드밴티지:* `A_t = G_t - V_φ(s_t)`. 편향되지 않았으며, 분산이 더 높습니다.
- *TD 어드밴티지:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. 편향되어 있습니다 (`V_φ` 사용), 분산이 훨씬 낮습니다. *TD 잔차* `δ_t`라고도 불립니다.

**n-단계 어드밴티지.** 두 형태 사이를 보간합니다:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`은 순수 TD입니다. `n = ∞`은 MC입니다. 대부분의 구현에서는 Atari에 `n = 5`을, MuJoCo의 PPO에 `n = 2048`을 사용합니다.

**일반화 이점 추정 (GAE).** Schulman et al. (2016)는 모든 n-step 이점에 대한 지수 가중 평균을 제안했습니다:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

`λ ∈ [0, 1]`를 사용합니다. `λ = 0`는 TD (낮은 분산, 높은 편향)입니다. `λ = 1`는 MC (높은 분산, 편향 없음)입니다. `λ = 0.95`은 2026년 기본값입니다 — 편향/분산 조절이 원하는 위치에 올 때까지 조정해 보세요.

**A2C: 동기식 이점 actor-critic.** `N`개의 병렬 환경에서 `T` 스텝을 수집합니다. 각 스텝에 대한 이점을 계산합니다. 결합된 배치에 대해 actor와 critic을 업데이트합니다. 반복합니다. A3C의 더 단순하고 확장 가능한 형제입니다.

**A3C: 비동기식 이점 actor-critic.** Mnih et al. (2016). `N`개의 워커 스레드를 생성하여 각각 환경(env)을 실행합니다. 각 워커는 자신의 롤아웃에서 로컬로 기울기를 계산한 후, 이를 공유 매개변수 서버에 비동기적으로 적용합니다. 리플레이 버퍼가 필요 없습니다 — 워커들이 서로 다른 궤적을 실행하여 상관을 제거합니다. A3C는 CPU에서 대규모로 학습할 수 있음을 증명했습니다. 2026년에는 GPU가 큰 배치를 선호하기 때문에, GPU 기반 A2C (배치된 병렬 환경)가 지배적입니다.

**결합된 손실.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

세 가지 항: 정책 기울기 손실, 가치 회귀, 엔트로피 보너스. `c_v ~ 0.5`, `c_e ~ 0.01`은 표준적인 시작점입니다.

```figure
actor-critic
```

## 구현하기

### 1단계: critic

MSE로 업데이트되는 선형 critic `V_φ(s) = w · features(s)`:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

테이블형 환경에서는 critic이 몇 백 에포크 만에 수렴합니다. Atari에서는 선형 critic을 공유 CNN 트렁크 + 가치 헤드(value head)로 교체하세요.

### 2단계: n-step 이점

길이 `T`인 롤아웃과 부트스트랩된 최종 `V(s_T)`가 주어지면:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`는 critic의 목표입니다. `advantages`는 `∇ log π`에 곱해지는 값입니다.

### 3단계: 결합된 업데이트

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

온 폴리시(on-policy), 업데이트당 하나의 롤아웃, actor와 critic에 대한 별도의 학습률.

### 4단계: 병렬화 (A3C vs A2C)

- **A3C:** `N`개의 스레드를 시작합니다. 각각이 자신의 환경과 자신의 순방향 전파(forward pass)를 실행합니다. 주기적으로 공유 마스터에 기울기 업데이트를 푸시합니다. 마스터에는 잠금이 없습니다 — 레이스(race)는 허용되며, 이는 단지 노이즈를 추가할 뿐입니다.
- **A2C:** 단일 프로세스에서 `N` env 인스턴스를 실행하고, 관측값을 `[N, obs_dim]` 배치로 쌓은 후 배치 단위 순전파와 배치 단위 역전파를 수행합니다. GPU 활용률이 높고, 결정적이며, 이해하기 쉽습니다. 2026년 기본값입니다.

우리의 장난감 코드는 명확성을 위해 단일 스레드이며, 배치형 A2C로 재작성하는 것은 numpy로 세 줄이면 됩니다.

## 함정

- **크리틱 편향과 액터 기울기.** 크리틱이 랜덤이면 그 기준선(baseline)은 정보량이 없으며, 순수한 잡음으로 학습하게 됩니다. 정책 기울기를 켜기 전에 크리틱을 몇백 스텝 동안 워밍업하거나, 액터의 학습률을 낮게 설정하세요.
- **어드밴티지 정규화.** 어드밴티지를 배치별로 평균 0/표준편차 1로 정규화하세요. 거의 비용 없이 학습을 대폭 안정화합니다.
- **공유 트렁크.** 이미지 입력에서는 액터와 크리틱이 공유 피처 추출기를 사용하세요. 헤드는 분리합니다. 공유 피처는 두 손실 함수의 혜택을 함께 받습니다.
- **온-폴리시 계약.** A2C는 데이터를 정확히 한 번의 업데이트에만 재사용합니다. 그 이상을 재사용하면 기울기가 편향됩니다 (중요도 샘플링 보정은 PPO가 추가하는 부분입니다).
- **엔트로피 붕괴.** `c_e > 0`이 없으면, 정책은 몇백 업데이트 만에 거의 결정적으로 변하고 탐험을 멈추게 됩니다.
- **보상 스케일.** 어드밴티지 크기는 보상 스케일에 의존합니다. 작업 간에 일관된 기울기 크기를 위해 보상을 정규화하세요 (예: 이동 표준편차로 나누기).

## 사용하기

A2C/A3C는 2026년 최종 선택인 경우는 드물지만, 이후 모든 기법이 정교화하는 아키텍처입니다:

| 기법 | A2C와의 관계 |
|--------|----------------|
| PPO | 다중 에포크 업데이트를 위한 클립된 중요도 비율을 더한 A2C |
| IMPALA | V-trace 오프-폴리시 보정을 더한 A3C |
| SAC (9단계 · 07) | 소프트 가치 크리틱을 사용하는 오프-폴리시 A2C (다음 강) |
| GRPO (9단계 · 12) | 크리틱이 없는 A2C — 그룹 상대 어드밴티지 |
| DPO | 선호 순위 손실로 축약된 A2C, 샘플링 없음 |
| AlphaStar / OpenAI Five | 리그 훈련과 모방 사전 훈련을 결합한 A2C |

2026년 논문에서 "어드밴티지"를 보게 되면, 액터-크리틱을 떠올리세요.

## 출시하기

`outputs/skill-actor-critic-trainer.md`로 저장하세요:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 MC advantage (`G_t - V(s_t)`)를 사용하여 actor-critic을 학습해 보세요. 06강의 running-mean-baseline을 사용한 REINFORCE와 샘플 효율성을 비교해 보세요.
2. **중간.** TD-residual advantage (`r + γ V(s') - V(s)`)로 전환해 보세요. advantage 배치의 분산을 측정해 보세요. 얼마나 감소하나요?
3. **어려움.** GAE(λ)를 구현해 보세요. `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`를 스윕해 보세요. 최종 리턴과 샘플 효율성을 플롯해 보세요. 이 작업에서 bias/variance의 최적 지점은 어디인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Actor | "정책 네트워크" | `π_θ(a\|s)`, 정책 기울기로 업데이트됨. |
| Critic | "가치 네트워크" | `V_φ(s)`, 리턴/TD 타겟에 대한 MSE 회귀로 업데이트됨. |
| Advantage | "평균보다 얼마나 좋은지" | `A(s, a) = Q(s, a) - V(s)` 또는 그 추정량. `∇ log π`의 곱셈 인자. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; 1단계 advantage 추정량. |
| GAE | "보간 조절자" | n-step advantage의 지수 가중 합으로, `λ`로 매개변수화됨. |
| A2C | "동기식 actor-critic" | 환경 간에 배치 처리; 롤아웃당 하나의 기울기 단계. |
| A3C | "비동기식 actor-critic" | 워커 스레드가 공유 매개변수 서버에 기울기를 푸시함. 원 논문; 2026년에는 덜 흔함. |
| Bootstrap | "수평선에서 V 사용" | 롤아웃을 잘라내고, 합을 닫기 위해 `γ^n V(s_{t+n})`를 더함. |

## 추가 읽기

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, 원본 비동기식 actor-critic 논문.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 기초; critic이 신경망일 때 함수 근사에 대한 9장과 함께 읽어 보세요.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — V-trace off-policy 보정을 사용하는 확장 가능한 분산 actor-critic.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 읽어 볼 가치가 있는 프로덕션 A2C/PPO 구현.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 2-시간 스케일(two-timescale) actor-critic 분해에 대한 기초적인 수렴 결과.
