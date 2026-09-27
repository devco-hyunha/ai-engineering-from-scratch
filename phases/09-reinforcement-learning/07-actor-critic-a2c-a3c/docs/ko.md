# Actor-Critic — A2C 및 A3C

> REINFORCE는 노이즈가 심합니다. `V̂(s)`를 학습하는 크리틱(critic)을 추가하고, 이를 리턴(return)에서 빼면 기대값은 동일하면서 분산은 훨씬 낮은 어드밴티지(advantage)를 얻을 수 있습니다. 이것이 바로 액터-크리틱(actor-critic)입니다. A2C는 이를 동기식(synchronously)으로 실행하며, A3C는 여러 스레드에 걸쳐 비동기식(across threads)으로 실행합니다. 두 방식 모두 현대의 모든 심층 강화 학습(deep-RL) 방법론의 사고 모델입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## 문제점 (The Problem)

Vanilla REINFORCE는 작동하지만, 분산(variance)이 매우 심각합니다. 몬테카를로 리턴(Monte Carlo returns)인 `G_t`는 에피소드 사이에 10배 넘게 요동칠 수 있습니다. 이 노이즈를 `∇ log π`에 곱하고 평균을 내면, DQN 업데이트 한 번으로도 충분히 이동할 수 있는 거리를 이동시키기 위해 수천 번의 에피소드가 필요한 그래디언트 추정기(gradient estimator)가 생성됩니다.

이러한 분산은 가공되지 않은 리턴(raw returns)을 사용하기 때문에 발생합니다. 만약 베이스라인(baseline) `b(s_t)`—학습된 가치 함수를 포함하여 상태의 임의의 함수—를 빼준다면, 기댓값(expectation)은 변하지 않으면서 분산은 줄어듭니다. 다루기 가장 좋은 베이스라인은 `V̂(s_t)`입니다. 이제 `∇ log π`에 곱해지는 값은 *어드밴티지(advantage)*가 됩니다:

`A(s, a) = G - V̂(s)`

어떤 행동이 평균 이상의 리턴을 생성했다면 좋은 행동이고, 평균 미만이라면 나쁜 행동입니다. 학습된 크리틱(critic)을 사용하는 REINFORCE는 *액터-크리틱(actor-critic)* 방식입니다. 크리틱은 액터(actor)에게 분산이 낮은 스승 역할을 합니다. 이는 2015년 이후 등장한 모든 심층 정책(deep-policy) 방법론(A2C, A3C, PPO, SAC, IMPALA)의 근간이 됩니다.

## 개념 (The Concept)

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**두 개의 네트워크, 하나의 공유 손실(shared loss):**

- **Actor** `π_θ(a | s)`: 정책(policy)입니다. 행동을 위해 샘플링되며, 정책 경사(policy gradient)로 학습됩니다.
- **Critic** `V_φ(s)`: 상태로부터 기대 수익(expected return)을 추정합니다. `(V_φ(s) - target)²`을 최소화하도록 학습됩니다.

**어드밴티지(The advantage).** 두 가지 표준 형태가 있습니다:

- *MC 어드밴티지:* `A_t = G_t - V_φ(s_t)`. 편향되지 않았으나(unbiased), 분산이 높습니다.
- *TD 어드밴티지:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. 편향되어 있지만(`V_φ` 사용), 분산이 훨씬 낮습니다. *TD 잔차(TD residual)* `δ_t`라고도 불립니다.

**n-step 어드밴티지(n-step advantage).** 두 방식 사이를 보간(interpolate)합니다:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`은 순수 TD입니다. `n = ∞`는 MC입니다. 대부분의 구현체는 Atari의 경우 `n = 5`, MuJoCo에서의 PPO의 경우 `n = 2048`을 사용합니다.

**일반화된 어드밴티지 추정(Generalized Advantage Estimation, GAE).** Schulman et al. (2016)은 모든 n-step 어드밴티지에 대해 지수 가중 평균(exponentially weighted average)을 제안했습니다:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

여기서 `λ ∈ [0, 1]`입니다. `λ = 0`은 TD(낮은 분산, 높은 편향)입니다. `λ = 1`은 MC(높은 분산, 편향 없음)입니다. `λ = 0.95`는 2026년의 기본값입니다. 편향과 분산 사이의 균형을 맞추기 위해 튜닝해 보세요.

**A2C: 동기식 어드밴티지 액터-크리틱(synchronous advantage actor-critic).** `N`개의 병렬 환경에서 `T`단계만큼 데이터를 수집합니다. 각 단계에 대한 어드밴티지를 계산한 뒤, 결합된 배치(combined batch)로 액터와 크리틱을 업데이트합니다. 이를 반복합니다. A3C보다 단순하고 확장성이 더 뛰어난 모델입니다.

**A3C: 비동기식 어드밴티지 액터-크리틱(asynchronous advantage actor-critic).** Mnih et al. (2016). `N`개의 워커 스레드를 생성하며, 각 스레드는 환경을 실행합니다. 각 워커는 자신의 롤아웃(rollout)에서 로컬하게 그래디언트를 계산한 다음, 공유 파라미터 서버에 비동기적으로 적용합니다. 워커들이 서로 다른 궤적(trajectory)을 실행함으로써 상관관계를 제거(decorrelate)하기 때문에 리플레이 버퍼(replay buffer)가 필요하지 않습니다. A3C는 CPU에서도 대규모 학습이 가능하다는 것을 증명했습니다. 2026년에는 GPU가 큰 배치를 선호하기 때문에 GPU 기반의 A2C(배치 병렬 환경)가 주를 이룹니다.

**결합된 손실(The combined loss).**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

세 가지 항으로 구성됩니다: 정책 경사 손실(policy-gradient loss), 가치 회귀(value regression), 엔트로피 보너스(entropy bonus). `c_v ~ 0.5`, `c_e ~ 0.01`이 전형적인 시작점입니다.

```figure
actor-critic
```

## 직접 구현해 보기 (Build It)

### 1단계: 크리틱 (a critic)

MSE(평균 제곱 오차)로 업데이트되는 선형 크리틱(Linear critic) `V_φ(s) = w · features(s)`:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

테이블형 환경(tabular env)에서 크리틱은 수백 에피소드 내에 수렴합니다. Atari 환경에서는 선형 크리틱을 공유된 CNN trunk + 가치 헤드(value head)로 교체하세요.

### 2단계: n-step 어드밴티지 (n-step advantage)

길이가 `T`인 롤아웃(rollout)과 부트스트랩된 최종 `V(s_T)`가 주어졌을 때:

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

`returns`는 크리틱 타겟(critic target)입니다. `advantages`는 `∇ log π`에 곱해지는 값입니다.

### 3단계: 결합 업데이트 (combined update)

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

온폴리시(On-policy) 방식이며, 업데이트당 하나의 롤아웃(rollout)을 수행하고, 액터(actor)와 크리틱(critic)에 대해 별도의 학습률을 사용합니다.

### 4단계: 병렬화 (A3C vs A2C)

- **A3C:** `N`개의 스레드를 실행합니다. 각 스레드는 자신만의 환경(env)과 순전파(forward pass)를 수행하며, 주기적으로 공유 마스터(shared master)에 그래디언트 업데이트를 푸시합니다. 마스터에 락(lock)을 걸지 않으므로 경합 조건(race condition)이 발생해도 괜찮습니다. 이는 단지 노이즈를 추가할 뿐입니다.
- **A2C:** 단일 프로세스 내에서 `N`개의 환경 인스턴스를 실행하고, 관측값(observations)을 `[N, obs_dim]` 배치로 쌓은 뒤, 배치 단위의 순전파 및 역전파(backward pass)를 수행합니다. GPU 활용도가 더 높고, 결정론적(deterministic)이며, 논리적으로 이해하기 더 쉽습니다. 2026년의 기본 방식입니다.

우리의 예제 코드는 명확성을 위해 싱글 스레드로 작성되었습니다. 이를 배치 방식의 A2C로 재작성하는 것은 `numpy`로 단 세 줄이면 충분합니다.

## 주의 사항 (Pitfalls)

- **Actor 그래디언트 계산 전 Critic 편향 (Critic bias before actor gradient).** 만약 Critic이 무작위 상태라면, 해당 베이스라인(baseline)은 정보 가치가 없으며 순수한 노이즈를 바탕으로 학습하게 됩니다. 정책 그래디언트(policy gradient)를 활성화하기 전에 Critic을 수백 단계 동안 먼저 학습(warm up)시키거나, Actor의 학습률(learning rate)을 낮게 설정해 보세요.
- **어드밴티지 정규화 (Advantage normalization).** 배치(batch)별로 어드밴티지의 평균을 0, 표준편차를 1로 정규화하세요. 거의 비용이 들지 않으면서도 학습을 매우 안정적으로 만들어 줍니다.
- **공유 트렁크 (Shared trunk).** 이미지 입력의 경우 Actor와 Critic이 공통의 특징 추출기(feature extractor)를 사용하도록 하세요. 헤드(head)는 분리합니다. 공유된 특징은 두 손실 함수(loss) 모두로부터 이득을 얻습니다.
- **온-폴리시 계약 (On-policy contract).** A2C는 데이터를 정확히 단 한 번의 업데이트에만 재사용합니다. 그 이상 사용하면 그래디언트에 편향이 발생합니다(PPO가 추가하는 것이 바로 중요도 샘플링(importance-sampling) 보정입니다).
- **엔트로피 붕괴 (Entropy collapse).** `c_e > 0` 설정이 없다면, 정책은 수백 번의 업데이트 만에 거의 결정론적(deterministic)인 상태가 되어 탐험(exploration)을 중단합니다.
- **보상 스케일 (Reward scale).** 어드밴티지의 크기는 보상 스케일에 따라 달라집니다. 다양한 태스크에서 일관된 그래디언트 크기를 유지할 수 있도록 보상을 정규화하세요(예: 이동 표준편차(running-std)로 나누기).

## 활용하기 (Use It)

A2C/A3C는 2026년 기준으로 최종 선택지로 쓰이는 경우는 드물지만, 이후의 모든 알고리즘이 발전해 나가는 기초 아키텍처입니다.

| 방법론 (Method) | A2C와의 관계 |
|--------|----------------|
| PPO | A2C + 멀티 에포크 업데이트를 위한 클리핑된 중요도 비율(clipped importance ratio) |
| IMPALA | A3C + V-trace 오프-폴리시(off-policy) 보정 |
| SAC (Phase 9 · 07) | 소프트 가치 크리틱(soft-value critic)을 사용하는 오프-폴리시 A2C (다음 레슨) |
| GRPO (Phase 9 · 12) | 크리틱(critic)이 없는 A2C — 그룹 상대적 어드밴티지(group-relative advantage) |
| DPO | 샘플링 없이 선호도 순위 손실(preference-ranking loss)로 축약된 A2C |
| AlphaStar / OpenAI Five | 리그 트레이닝(league training) + 모방 사전 학습(imitation pre-training)이 결합된 A2C |

2026년 논문에서 "어드밴티지(advantage)"라는 단어를 본다면, 액터-크리틱(actor-critic) 구조를 떠올려 보세요.

## Ship It

`outputs/skill-actor-critic-trainer.md`로 저장하세요:

```markdown
---
name: actor-critic-trainer
description: 주어진 환경에 대해 어드밴티지 추정(advantage estimation) 및 손실 가중치(loss weights)가 지정된 A2C / A3C / GAE 설정을 생성합니다.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

주어진 환경과 연산 예산(compute budget)을 바탕으로 다음을 출력하세요:

1. 병렬성(Parallelism). A2C (GPU 배치 방식) 대 A3C (CPU 비동기 방식) 및 워커(worker) 수.
2. 롤아웃 길이(Rollout length) `T`. 업데이트당 환경별 스텝 수.
3. 어드밴티지 추정기(Advantage estimator). n-step 또는 GAE(λ); `λ` 값을 명시할 것.
4. 손실 가중치(Loss weights). `c_v` (가치), `c_e` (엔트로피), 그래디언트 클리핑(gradient clip).
5. 학습률(Learning rates). 액터(Actor) 및 크리틱(Critic) (별도로 사용하는 경우 각각 명시).

호라이즌(horizon)이 1000보다 큰 환경에 대해 단일 워커(single-worker) A2C를 제안하는 것은 거부하세요 (너무 온-폴리시(on-policy)이며 속도가 너무 느림). 어드밴티지 정규화(advantage normalization) 없이는 결과물을 제공하지 마세요. `c_e = 0`이고 관찰된 엔트로피가 0.1 미만인 모든 실행은 엔트로피 붕괴(entropy-collapsed)로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 4×4 GridWorld 환경에서 MC 어드밴티지(`G_t - V(s_t)`)를 사용하는 Actor-Critic을 학습시켜 보세요. Lesson 06에서 다룬 running-mean-baseline을 사용하는 REINFORCE 방식과 샘플 효율성(sample efficiency)을 비교해 보세요.
2. **중간 (Medium).** TD-residual 어드밴티지(`r + γ V(s') - V(s)`)로 전환해 보세요. 어드밴티지 배치(advantage batches)의 분산을 측정해 보세요. 분산이 얼마나 감소하나요?
3. **어려움 (Hard).** GAE(λ)를 구현해 보세요. `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 범위에서 탐색(sweep)을 수행하세요. 최종 리턴(final return) 대 샘플 효율성(sample efficiency)을 그래프로 그려 보세요. 이 작업에서 편향(bias)과 분산(variance) 사이의 최적 지점(sweet spot)은 어디인가요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Actor (액터) | "정책 네트워크(policy net)" | 정책 경사(policy gradient)에 의해 업데이트되는 `π_θ(a\|s)`. |
| Critic (크리틱) | "가치 네트워크(value net)" | 리턴(returns) 또는 TD 타겟에 대한 MSE 회귀로 업데이트되는 `V_φ(s)`. |
| Advantage (어드밴티지) | "평균보다 얼마나 더 나은가" | `A(s, a) = Q(s, a) - V(s)` 또는 그 추정치. `∇ log π`에 곱해지는 승수. |
| TD residual (TD 잔차) | "δ" | `δ_t = r + γ V(s') - V(s)`; 1단계 어드밴티지 추정치. |
| GAE | "보간 조절 노브(interpolation knob)" | `λ`로 매개변수화된 n-step 어드밴티지의 지수 가중 합. |
| A2C | "동기식 액터-크리틱(Synchronous actor-critic)" | 환경(envs) 전체에 걸쳐 배치(batch) 처리됨; 롤아웃(rollout)당 한 번의 경사 단계 수행. |
| A3C | "비동기식 액터-크리틱(Async actor-critic)" | 워커 스레드가 공유 파라미터 서버로 경사를 푸시함. 원본 논문 방식; 2026년 기준으로는 덜 일반적임. |
| Bootstrap (부트스트랩) | "지평선(horizon)에서 V를 사용함" | 롤아웃을 절단하고, 합계를 닫기 위해 `γ^n V(s_{t+n})`을 더함. |

## 추가 읽을거리 (Further Reading)

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, 원본 비동기 액터-크리틱(async actor-critic) 논문입니다.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE(Generalized Advantage Estimation)에 관한 논문입니다.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 기초 이론입니다. 크리틱(critic)이 신경망인 경우, 함수 근사(function approximation)를 다루는 9장을 함께 읽어 보세요.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — V-trace off-policy 교정을 사용하는 확장 가능한 분산형 액터-크리틱(distributed actor-critic) 모델입니다.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 살펴볼 가치가 있는 실무용 A2C/PPO 구현체입니다.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 2-timescale 액터-크리틱 분해(actor-critic decomposition)에 대한 기초적인 수렴 결과(convergence result)를 다룹니다.
