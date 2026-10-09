# 정책 기울기 — REINFORCE를 처음부터 구현하기

> 가치를 추정하는 것을 멈추세요. 정책을 직접 매개변수화하고, 기대 보상의 기울기를 계산하여 상승 방향으로 이동하세요. Williams (1992)는 이를 하나의 정리로 작성했습니다. 이것이 PPO, GRPO 및 모든 LLM RL 루프가 존재하는 이유입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 03강 (역전파), 9단계 · 03강 (몬테카를로), 9단계 · 04강 (TD 학습)
**시간:** 약 75분

## 문제점

Q-학습과 DQN은 *가치* 함수를 매개변수화합니다. `argmax Q`을 통해 행동을 선택합니다. 이는 이산적인 행동과 이산적인 상태에서는 문제가 없습니다. 하지만 행동이 연속적인 경우 (`argmax`이 10차원 토크에 대해 어떻게 작동합니까?)나 확률적 정책을 원할 때 (`argmax`는 구조적으로 결정론적임)에는 작동하지 않습니다.

정책 기울기는 대신 *정책*을 매개변수화합니다. `π_θ(a | s)`은 행동에 대한 분포를 출력하는 신경망입니다. 여기서 샘플링하여 행동하세요. `θ`에 대한 기대 보상의 기울기를 계산하세요. 상승 방향으로 이동하세요. `argmax`는 필요 없습니다. Bellman 재귀도 필요 없습니다. `J(θ) = E_{π_θ}[G]`에 대한 단순한 기울기 상승만 수행하세요.

REINFORCE 정리 (Williams 1992)는 이 기울기가 계산 가능하다는 것을 알려줍니다: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. 에피소드를 실행하세요. 보상을 계산하세요. 모든 단계에서 `∇ log π_θ(a | s)`을 곱하세요. 평균을 내세요. 기울기 상승을 수행하세요. 끝입니다.

2026년의 모든 LLM-RL 알고리즘 — PPO, DPO, GRPO —은 REINFORCE의 개선입니다. 이를 손에 익히는 것은 이 단계의 나머지 부분과 10단계 · 07강 (RLHF 구현) 및 10단계 · 08강 (DPO)의 선수 요건입니다.

## 개념

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**정책 기울기 정리.** `θ`으로 매개변수화된 모든 정책 `π_θ`에 대해:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

여기서 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`는 `t` 단계부터의 할인된 보상입니다. 기대값은 `π_θ`에서 샘플링된 전체 궤적 `τ`에 대한 것입니다.

**증명은 짧습니다.** 기대값 아래에서 `J(θ) = Σ_τ P(τ; θ) G(τ)`을 미분하세요. `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (로그 미분 트릭)을 사용하세요. `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`을 분해하세요. 환경 항은 사라집니다. 두 줄의 대수 연산으로 정리를 얻을 수 있습니다.

**분산 감소 기법.** 순수 REINFORCE는 살인적인 분산을 가집니다 — 보상은 잡음이 많고, `∇ log π`도 잡음이 많으며, 그 곱은 매우 잡음이 많습니다. 두 가지 표준적인 수정 방법:

1. **기준선 차감.** `a_t`에 의존하지 않는 임의의 기준선 `b(s_t)`에 대해 `G_t`을 `G_t - b(s_t)`으로 대체하세요. `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`이므로 편향되지 않습니다. 일반적인 선택: 크리틱이 학습한 `b(s_t) = V̂(s_t)` → actor-critic (07강).
2. **Reward-to-go.** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`을 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`으로 대체하세요. 주어진 행동에 대해서는 미래의 보상만 의미가 있습니다 — 과거의 보상은 평균이 0인 잡음에 기여합니다.

두 기법을 결합하면 다음을 얻습니다:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

이는 기준선을 사용한 REINFORCE로, A2C (07강)와 PPO (08강)의 직접적인 조상입니다.

**소프트맥스 정책 파라미터화.** 이산 행동의 경우 표준적인 선택:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

여기서 `f_θ`은 각 행동에 대한 점수를 출력하는 임의의 신경망입니다. 기울기는 깔끔한 형태를 가집니다:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

즉, 선택된 행동의 점수에서 정책 하의 기대값을 뺀 값입니다.

**연속 행동을 위한 가우시안 정책.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)`은 닫힌 형태를 가집니다. 이것이 9단계 · 07강의 SAC에 필요한 전부입니다.

```figure
policy-gradient-landscape
```

## 구현하기

### 1단계: 소프트맥스 정책 네트워크

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

테이블형 환경에서는 선형 정책(각 행동에 하나의 가중치 벡터)을 사용하세요. Atari의 경우 CNN으로 교체하고 소프트맥스 헤드를 유지하세요.

### 2단계: 샘플링 및 로그 확률

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

### 3단계: 로그 확률을 기록한 롤아웃

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

### 4단계: REINFORCE 업데이트

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

기울기 `∇ log π(a|s) = e_a - π(·|s)` (`a`의 onehot에서 확률을 뺀 값)는 소프트맥스 정책 기울기의 핵심입니다. 근육 기억에 새기세요.

### 5단계: 기준선

최근 에피소드에 대한 `G`의 이동 평균은 4x4 GridWorld를 실행하는 데 충분한 분산 감소입니다; 수렴하는 데 약 500 에피소드가 걸립니다. 기준선을 학습된 `V̂(s)`으로 업그레이드하면 actor-critic을 얻습니다.

## 문제점

- **기울기 폭발.** 보상은 매우 클 수 있습니다. `∇ log π`을 곱하기 전에 항상 배치 전체에서 `G`을 `~N(0, 1)`으로 정규화하세요.
- **엔트로피 붕괴.** 정책이 너무 일찍 거의 결정적인 행동으로 수렴하여 탐험을 멈추고 갇히게 됩니다. 해결책: 목적 함수에 엔트로피 보너스 `β · H(π(·|s))`를 추가해 보세요.
- **높은 분산.** 기본 REINFORCE는 수천 개의 에피소드가 필요합니다. 크리틱 베이스라인 (07강)이나 TRPO/PPO의 신뢰 영역 (08강)이 표준적인 해결책입니다.
- **샘플 비효율성.** 온-폴리시(on-policy)는 한 번의 업데이트 후 모든 전이를 버립니다. 중요도 샘플링(importance sampling)을 통한 오프-폴리시(off-policy) 보정은 분산의 대가로 데이터를 되살립니다 (PPO의 비율은 클리핑된 IS 가중치입니다).
- **비정상적(non-stationary) 기울기.** 100 에피소드 전의 동일한 기울기는 오래된 `π`를 사용합니다. 온-폴리시 방법은 이 때문에 몇 번의 롤아웃마다 업데이트를 수행합니다.
- **신용 할당(Credit assignment).** Reward-to-go가 없으면 과거의 보상이 잡음을 기여합니다. 항상 reward-to-go를 사용하세요.

## 사용하기

2026년, REINFORCE는 거의 직접 실행되지 않지만 그 기울기 공식은 어디에나 있습니다:

| 사용 사례 | 파생된 방법 |
|----------|---------------|
| 연속 제어 | 가우시안 정책을 사용하는 PPO / SAC |
| LLM RLHF | KL 페널티가 있는 PPO, 토큰 수준 정책에서 실행 |
| LLM 추론 (DeepSeek) | GRPO — 그룹 상대 베이스라인을 사용하는 REINFORCE, 크리틱 없음 |
| 다중 에이전트 | 중앙 집중식 크리틱 REINFORCE (MADDPG, COMA) |
| 이산 행동 로봇공학 | A2C, A3C, PPO |
| 선호도 전용 설정 | DPO — 선호도 가능성 손실로 재작성된 REINFORCE, 샘플링 없음 |

2026년 훈련 스크립트에서 `loss = -advantage * log_prob`를 읽는다면, 그것은 베이스라인이 있는 REINFORCE입니다. 전체 논문(DPO, GRPO, RLOO)은 이 한 줄 위에 분산 감소 트릭을 쌓은 것입니다.

## 출시하기

`outputs/skill-policy-gradient-trainer.md`로 저장하세요:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 연습 문제

1. **쉬움.** 선형 소프트맥스 정책을 사용하여 4×4 GridWorld에서 REINFORCE를 구현하세요. 베이스라인 없이 1,000 에피소드 동안 훈련하세요. 학습 곡선을 플롯하고 분산(보상의 표준 편차)을 측정하세요.
2. **중간.** 이동 평균 베이스라인을 추가하세요. 다시 훈련하세요. 샘플 효율성과 분산을 기본 실행과 비교하세요. 베이스라인이 수렴까지의 스텝을 얼마나 줄여줍니까?
3. **난이도: 높음.** 엔트로피 보너스 `β · H(π)`를 추가하세요. `β ∈ {0, 0.01, 0.1, 1.0}`를 스윕해 보세요. 최종 리턴과 정책 엔트로피를 플롯하세요. 이 작업에서 최적의 균형점은 어디인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 정책 기울기 | "정책을 직접 학습한다" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; 로그 미분 트릭에서 유도됩니다. |
| REINFORCE | "원래의 PG 알고리즘" | Williams (1992); 몬테 카를로 리턴에 로그 정책 기울기를 곱합니다. |
| 로그 미분 트릭 | "스코어 함수 추정기" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; 기댓값의 기울기를 계산 가능하게 만듭니다. |
| 베이스라인 | "분산 감소" | `b(s)`를 `G`에서 차감한 값; `E[b · ∇ log π] = 0`이므로 편향되지 않습니다. |
| Reward-to-go | "미래 리턴만 계산한다" | 전체 `G_0` 대신 `G_t^{from t}`를 사용; 정확하고 분산이 낮습니다. |
| 엔트로피 보너스 | "탐색을 장려한다" | `+β · H(π(·\|s))` 항은 정책이 붕괴하는 것을 방지합니다. |
| 온-폴리시 | "방금 본 데이터로 학습한다" | 기울기 기댓값은 현재 정책에 대한 것 — 이전 데이터를 직접 재사용할 수 없습니다. |
| 어드밴티지 | "평균보다 얼마나 좋은가" | `A(s, a) = G(s, a) - V(s)`; REINFORCE-with-baseline이 곱하는 부호 있는 값입니다. |

## 추가 읽기

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 원 논문.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 함수 근사를 포함한 현대적 정책 기울기 정리.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서적 설명.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorch 코드를 포함한 명확한 교육적 설명.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 분산 감소 및 REINFORCE를 신뢰 영역 계열(TRPO, PPO)과 연결하는 자연 기울기 관점.
