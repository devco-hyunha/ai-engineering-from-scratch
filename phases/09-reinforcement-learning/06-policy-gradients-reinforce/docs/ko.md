# 정책 경사(Policy Gradient) — REINFORCE 밑바닥부터 구현하기

> 가치(Value)를 추정하는 것을 멈추세요. 정책(Policy)을 직접 매개변수화하고, 기대 보상(Expected Return)의 경사(Gradient)를 계산하여 보상이 높아지는 방향으로 이동하세요. Williams(1992)는 이를 하나의 정리로 정리했습니다. 이것이 바로 PPO, GRPO, 그리고 모든 LLM RL 루프가 존재하는 이유입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## 문제점 (The Problem)

Q-learning과 DQN은 *가치(Value)* 함수를 매개변수화합니다. 여러분은 `argmax Q`를 통해 행동을 선택합니다. 이는 이산적(discrete)인 행동과 상태에서는 괜찮습니다. 하지만 행동이 연속적일 때(예: 10차원 토크에 대한 `argmax`를 구하는 경우)나 확률적 정책(stochastic policy)을 원할 때( `argmax`는 구조적으로 결정론적입니다) 문제가 발생합니다.

정책 경사(Policy gradients)는 대신 *정책(Policy)*을 매개변수화합니다. `π_θ(a | s)`는 행동에 대한 분포를 출력하는 신경망입니다. 이 분포에서 샘플링하여 행동합니다. `θ`에 대한 기대 수익(expected return)의 기울기(gradient)를 계산하여 경사 상승법(gradient ascent)을 수행합니다. `argmax`도 필요 없고, 벨만 재귀(Bellman recursion)도 필요 없습니다. 그저 `J(θ) = E_{π_θ}[G]`에 대한 경사 상승법을 수행할 뿐입니다.

REINFORCE 정리(Williams 1992)는 이 기울기를 계산할 수 있음을 알려줍니다: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. 에피소드를 실행하고, 수익(return)을 계산합니다. 매 단계마다 `∇ log π_θ(a | s)`를 곱하고 평균을 낸 뒤, 경사 상승법을 적용합니다. 끝입니다.

2026년의 모든 LLM-RL 알고리즘 — PPO, DPO, GRPO — 은 REINFORCE의 개선판입니다. 이를 완벽히 이해하는 것은 이 단계의 나머지 과정과, Phase 10 · 07 (RLHF 구현) 및 Phase 10 · 08 (DPO)을 위한 필수 전제 조건입니다.

## 개념 (The Concept)

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**정책 경사 정리 (The policy gradient theorem).** `θ`로 매개변수화된 임의의 정책 `π_θ`에 대하여:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

여기서 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`는 `t` 단계부터의 할인된 수익(discounted return)입니다. 기댓값은 `π_θ`로부터 샘플링된 전체 궤적(trajectory) `τ`에 대해 계산됩니다.

**증명은 간단합니다.** 기댓값 하에서 `J(θ) = Σ_τ P(τ; θ) G(τ)`를 미분합니다. `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (로그 미분 트릭, log-derivative trick)를 사용합니다. `log P(τ; θ) = Σ log π_θ(a_t | s_t) + θ에 의존하지 않는 환경 항(environment terms)`으로 인수분해하면 환경 항은 사라집니다. 두 줄의 대수 계산만으로 정리가 도출됩니다.

**분산 감소 기법 (Variance reduction tricks).** 기본 REINFORCE는 치명적인 분산을 가집니다. 수익(return)이 노이즈가 심하고, `∇ log π`도 노이즈가 심하며, 이 둘의 곱은 매우 심한 노이즈를 가집니다. 두 가지 표준적인 해결책은 다음과 같습니다:

1. **베이스라인 차감 (Baseline subtraction).** `a_t`에 의존하지 않는 임의의 베이스라인 `b(s_t)`에 대해 `G_t`를 `G_t - b(s_t)`로 대체합니다. `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`이므로 편향되지(unbiased) 않습니다. 일반적인 선택은 크리틱(critic)에 의해 학습된 `b(s_t) = V̂(s_t)`입니다. 이는 액터-크리틱(actor-critic, Lesson 07)으로 이어집니다.
2. **보상-투-고 (Reward-to-go).** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`를 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`로 대체합니다. 특정 행동에 대해서는 미래의 수익만이 중요하며, 과거의 보상은 평균이 0인 노이즈로 작용할 뿐입니다.

이들을 결합하면 다음과 같습니다:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

이는 베이스라인을 사용하는 REINFORCE이며, A2C (Lesson 07) 및 PPO (Lesson 08)의 직접적인 조상입니다.

**소프트맥스 정책 매개변수화 (Softmax policy parameterization).** 이산 행동(discrete actions)의 경우, 표준적인 선택은 다음과 같습니다:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

여기서 `f_θ`는 각 행동에 대한 점수(score)를 출력하는 임의의 신경망입니다. 경사는 깔끔한 형태를 가집니다:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

즉, 취해진 행동의 점수에서 정책에 따른 해당 점수의 기대값을 뺀 값입니다.

**연속 행동을 위한 가우시안 정책 (Gaussian policy for continuous actions).** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)`는 닫힌 형태(closed form)를 가집니다. 이것이 Phase 9 · 07의 SAC에 필요한 전부입니다.

```figure
policy-gradient-landscape
```

## 구축하기 (Build It)

### 1단계: softmax 정책 네트워크 (softmax policy network)

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

테이블형 환경(tabular env)의 경우 선형 정책(action당 하나의 가중치 벡터)을 사용하세요. Atari 환경의 경우 CNN으로 교체하되, softmax 헤드는 그대로 유지하세요.

### 2단계: 샘플링 및 로그 확률 (sampling and log-probability)

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 3단계: 로그 확률(log-probs)을 캡처한 롤아웃(rollout)

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 4단계: REINFORCE 업데이트 (REINFORCE update)

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

그래디언트 `∇ log π(a|s) = e_a - π(·|s)` (`a`의 원-핫 벡터에서 확률값을 뺀 것)는 소프트맥스 정책 그래디언트(softmax policy gradients)의 핵심입니다. 이를 근육 기억(muscle memory)에 새겨두세요.

### 5단계: 베이스라인(baselines)

최근 에피소드들에 대한 `G`의 이동 평균(running mean)만으로도 4×4 GridWorld를 실행하기에 충분한 분산 감소(variance reduction) 효과를 얻을 수 있으며, 수렴하는 데 약 500 에피소드가 소요됩니다. 베이스라인을 학습 가능한 `V̂(s)`로 업그레이드하면 액터-크리틱(actor-critic) 구조가 됩니다.

## 주의 사항 (Pitfalls)

- **기울기 폭주 (Exploding gradients).** 리턴(Returns) 값이 매우 커질 수 있습니다. `∇ log π`를 곱하기 전에 항상 배치 전체에 대해 `G`를 `~N(0, 1)`로 정규화하세요.
- **엔트로피 붕괴 (Entropy collapse).** 정책이 너무 일찍 거의 결정론적인(deterministic) 행동으로 수렴하여 탐색을 멈추고 정체될 수 있습니다. 해결책: 목적 함수에 엔트로피 보너스 `β · H(π(·|s))`를 추가하세요.
- **높은 분산 (High variance).** 바닐라 REINFORCE는 수천 번의 에피소드가 필요합니다. 크리틱 베이스라인(Lesson 07)이나 TRPO/PPO의 신뢰 영역(Trust region, Lesson 08)을 사용하는 것이 표준적인 해결 방법입니다.
- **샘플 효율성 저하 (Sample inefficiency).** 온폴리시(On-policy) 방식은 한 번의 업데이트 후에 모든 전이(transition)를 버려야 함을 의미합니다. 중요도 샘플링(Importance sampling)을 통한 오프폴리시(Off-policy) 보정은 분산이 커지는 대가로 데이터를 재사용할 수 있게 해줍니다(PPO의 비율은 클리핑된 IS 가중치입니다).
- **비정상성 기울기 (Non-stationary gradients).** 100 에피소드 전의 동일한 기울기는 오래된 `π`를 사용하게 됩니다. 온폴리시 방법론이 몇 번의 롤아웃(rollout)마다 업데이트를 수행하는 이유가 바로 이것입니다.
- **신용 할당 (Credit assignment).** 리워드 투 고(reward-to-go)를 사용하지 않으면 과거의 리워드가 노이즈로 작용합니다. 항상 리워드 투 고를 사용하세요.

## 활용하기 (Use It)

2026년 현재, REINFORCE가 직접적으로 실행되는 경우는 드물지만 그 그래디언트(gradient) 공식은 어디에나 존재합니다:

| 활용 사례 (Use case) | 파생된 방법 (Derived method) |
|----------|---------------|
| 연속 제어 (Continuous control) | 가우시안 정책(Gaussian policy)을 사용하는 PPO / SAC |
| LLM RLHF | 토큰 레벨 정책에서 실행되는 KL 페널티(KL penalty) 기반 PPO |
| LLM 추론 (DeepSeek) | GRPO — 크리틱(critic) 없이 그룹 상대적 베이스라인(group-relative baseline)을 사용하는 REINFORCE |
| 멀티 에이전트 (Multi-agent) | 중앙 집중형 크리틱 REINFORCE (MADDPG, COMA) |
| 이산 행동 로보틱스 (Discrete action robotics) | A2C, A3C, PPO |
| 선호도 전용 설정 (Preference-only settings) | DPO — 샘플링 없이 선호도-우도 손실(preference-likelihood loss)로 재작성된 REINFORCE |

2026년의 학습 스크립트에서 `loss = -advantage * log_prob`를 읽게 된다면, 그것은 바로 베이스라인을 사용하는 REINFORCE입니다. 수많은 논문들(DPO, GRPO, RLOO)은 이 한 줄의 코드 위에 구현된 분산 감소(variance-reduction) 기법들입니다.

## Ship It

`outputs/skill-policy-gradient-trainer.md`로 저장하세요:

```markdown
---
name: policy-gradient-trainer
description: 주어진 작업에 대한 REINFORCE / actor-critic / PPO 학습 설정을 생성하고 분산(variance) 문제를 진단합니다.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

환경(이산/연속 동작, 호라이즌, 보상 통계)이 주어지면 다음을 출력하세요:

1. 정책 헤드(Policy head). 파라미터 수를 포함한 Softmax(이산) 또는 Gaussian(연속).
2. 베이스라인(Baseline). 없음(vanilla), 이동 평균(running mean), 학습된 `V̂(s)`, 또는 A2C critic.
3. 분산 제어(Variance controls). 기본적으로 적용되는 Reward-to-go, 리턴 정규화(return normalization), 그래디언트 클리핑(gradient clip) 값.
4. 엔트로피 보너스(Entropy bonus). 계수 $\beta$ 및 감쇠 스케줄(decay schedule).
5. 배치 크기(Batch size). 업데이트당 에피소드 수; 온폴리시(on-policy) 데이터 신선도 계약.

호라이즌이 500단계를 초과하는 경우 REINFORCE-no-baseline을 거부하세요. Softmax 헤드를 사용하는 연속 동작 제어(continuous-action control)를 거부하세요. `β = 0`이고 관찰된 정책 엔트로피가 0.1 미만인 모든 실행은 엔트로피 붕괴(entropy-collapsed)로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 선형 소프트맥스 정책(linear softmax policy)을 사용하여 4×4 GridWorld 환경에서 REINFORCE를 구현해 보세요. 베이스라인(baseline) 없이 1,000 에피소드 동안 학습을 진행합니다. 학습 곡선(learning curve)을 그리고, 분산(수익의 표준편차)을 측정해 보세요.
2. **중간 (Medium).** 이동 평균 베이스라인(running-mean baseline)을 추가해 보세요. 다시 학습을 진행합니다. 기본 실행(vanilla run)과 비교하여 샘플 효율성(sample efficiency)과 분산을 비교해 보세요. 베이스라인을 사용했을 때 수렴까지 걸리는 단계(steps to convergence)가 얼마나 단축되었나요?
3. **어려움 (Hard).** 엔트로피 보너스 `β · H(π)`를 추가해 보세요. `β ∈ {0, 0.01, 0.1, 1.0}` 범위에 대해 스윕(sweep)을 수행합니다. 최종 수익(final return)과 정책 엔트로피(policy entropy)를 그래프로 그리세요. 이 작업에서 가장 적절한 지점(sweet spot)은 어디인가요?

## 주요 용어 (Key Terms)

| 용어 (Term) | 흔히 하는 말 (What people say) | 실제 의미 (What it actually means) |
|------|-----------------|-----------------------|
| Policy gradient (정책 경사) | "정책을 직접 학습시킨다" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; log-derivative trick에서 유도되었습니다. |
| REINFORCE | "오리지널 PG 알고리즘" | Williams (1992); Monte Carlo 리턴에 log-policy gradient를 곱한 방식입니다. |
| Log-derivative trick (로그 미분 트릭) | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; 기댓값의 경사(gradient)를 계산 가능하게 만듭니다. |
| Baseline (베이스라인) | "분산 감소(Variance reduction)" | `G`에서 빼주는 임의의 `b(s)`; `E[b · ∇ log π] = 0`이므로 편향되지(unbiased) 않습니다. |
| Reward-to-go (보상 합계) | "미래의 리턴만 고려한다" | 전체 `G_0` 대신 `G_t^{from t}`를 사용합니다. 정확하며 분산이 더 낮습니다. |
| Entropy bonus (엔트로피 보너스) | "탐험(Exploration)을 장려한다" | `+β · H(π(·\|s))` 항을 추가하여 정책이 붕괴되는 것을 방지합니다. |
| On-policy (온-폴리시) | "방금 본 데이터로 학습한다" | 경사(gradient)의 기댓값이 현재 정책에 대한 것입니다. 즉, 과거 데이터를 직접 재사용할 수 없습니다. |
| Advantage (어드밴티지) | "평균보다 얼마나 더 좋은가" | `A(s, a) = G(s, a) - V(s)`; REINFORCE-with-baseline에서 곱해지는 부호가 있는 수치입니다. |

## 추가 읽을거리 (Further Reading)

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 알고리즘의 원전 논문입니다.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 함수 근사(function approximation)를 포함한 현대적인 정책 경사 정리(policy-gradient theorem)를 다룹니다.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서적인 설명이 담긴 자료입니다.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorch 코드가 포함된 명확하고 교육적인 해설입니다.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 분산 감소(variance-reduction)와 REINFORCE를 신뢰 영역(trust-region) 계열(TRPO, PPO)로 연결하는 자연 경사(natural-gradient) 관점을 다룹니다.
