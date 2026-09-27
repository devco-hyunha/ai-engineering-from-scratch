# 몬테카를로 방법(Monte Carlo Methods) — 전체 에피소드를 통한 학습

> 동적 계획법(Dynamic Programming)은 모델이 필요합니다. 하지만 몬테카를로 방법은 에피소드 외에는 아무것도 필요하지 않습니다. 정책을 실행하고, 리턴(returns)을 관찰하며, 그 평균을 구하세요. 강화학습(RL)에서 가장 단순한 아이디어이자, 이후의 모든 과정을 가능하게 하는 핵심 원리입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## 문제점 (The Problem)

동적 계획법(Dynamic Programming)은 우아하지만, 모든 상태와 행동에 대해 `P(s' | s, a)`를 쿼리할 수 있다고 가정합니다. 현실 세계에서 이런 방식으로 작동하는 경우는 거의 없습니다. 로봇은 관절 토크(joint torque) 이후의 카메라 픽셀 분포를 분석적으로 계산할 수 없습니다. 가격 책정 알고리즘은 가능한 모든 고객의 반응에 대해 적분할 수 없습니다. LLM은 토큰 이후에 올 수 있는 모든 연속된 문장을 열거할 수 없습니다.

환경으로부터 *샘플링(sample)*할 수 있는 능력만 있으면 되는 방법이 필요합니다. 정책(policy)을 실행하세요. 궤적(trajectory) `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`를 얻으세요. 이를 사용하여 가치를 추정하세요. 이것이 바로 몬테카를로(Monte Carlo)입니다.

DP에서 MC로의 전환은 철학적으로 중요합니다. 우리는 *알려진 모델 + 정확한 백업(exact backup)*에서 *샘플링된 롤아웃(sampled rollouts) + 평균 리턴(averaged return)*으로 이동합니다. 분산(variance)은 급증하지만, 적용 가능성은 폭발적으로 늘어납니다. 이 레슨 이후의 모든 강화 학습 알고리즘 — TD, Q-learning, REINFORCE, PPO, GRPO — 은 본질적으로 몬테카를로 추정기이며, 때로는 그 위에 부트스트래핑(bootstrapping)이 계층적으로 쌓여 있을 뿐입니다.

## 개념 (The Concept)

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**핵심 아이디어 (한 줄 요약):** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` 여기서 `G^{(i)}(s)`는 정책 `π`에 따라 `s`를 방문한 후 관찰된 리턴(returns)입니다.

**First-visit vs every-visit MC.** 상태 `s`를 여러 번 방문하는 에피소드가 주어졌을 때, first-visit MC는 첫 번째 방문으로부터의 리턴만 계산합니다. 반면 every-visit MC는 모든 방문을 계산합니다. 두 방식 모두 극한에서는 편향되지 않습니다(unbiased). First-visit은 분석하기 더 간단합니다(iid 샘플). Every-visit은 에피소드당 더 많은 데이터를 사용하며, 실제로는 일반적으로 더 빠르게 수렴합니다.

**점진적 평균 (Incremental mean).** 모든 리턴을 저장하는 대신, 실행 평균(running average)을 업데이트합니다:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

재구성하면: `V_new = V_old + α · (target - V_old)`이며, 여기서 `α = 1/n`입니다. `1/n`을 상수 단계 크기(constant step-size) `α ∈ (0, 1)`로 바꾸면, `π`의 변화를 추적하는 비정상(non-stationary) MC 추정기를 얻을 수 있습니다. 이 변화가 바로 MC에서 TD로, 그리고 모든 현대적 RL 알고리즘으로 도약하는 결정적인 지점입니다.

**탐험(Exploration)의 문제.** DP는 열거(enumeration)를 통해 모든 상태를 다루었습니다. 하지만 MC는 정책이 방문하는 상태만을 봅니다. 만약 `π`가 결정론적(deterministic)이라면, 상태 공간의 전체 영역이 샘플링되지 않으며, 해당 상태들의 가치 추정치는 영원히 0으로 남게 됩니다. 역사적 순서에 따른 세 가지 해결책은 다음과 같습니다:

1. **탐험적 시작 (Exploring starts).** 각 에피소드를 무작위 `(s, a)` 쌍에서 시작합니다. 모든 영역의 커버리지를 보장하지만, 실제로는 비현실적입니다(로봇을 임의의 상태로 "리셋"할 수는 없습니다).
2. **ε-greedy.** 현재 `Q`에 대해 탐욕적(greedy)으로 행동하되, 확률 `ε`으로 무작위 행동을 선택합니다. 점근적으로 모든 상태-행동 쌍이 샘플링됩니다.
3. **Off-policy MC.** 행동 정책(behavior policy) `μ`를 통해 데이터를 수집하고, 중요도 샘플링(importance sampling)을 통해 목표 정책(target policy) `π`에 대해 학습합니다. 분산(variance)은 높지만, DQN과 같은 리플레이 버퍼(replay-buffer) 방식의 가교 역할을 합니다.

**Monte Carlo Control.** 정책 반복(policy iteration)과 마찬가지로 평가(evaluate) → 개선(improve) → 평가 과정을 거치지만, 평가는 샘플링 기반으로 이루어집니다:

1. `π`를 실행하여 에피소드를 얻습니다.
2. 관찰된 리턴으로부터 `Q(s, a)`를 업데이트합니다.
3. `π`를 `Q`에 대해 ε-greedy하게 만듭니다.
4. 반복합니다.

완만한 조건(모든 쌍이 무한히 자주 방문되고, `α`가 Robbins-Monro 조건을 만족함) 하에서 확률 1로 `Q*`와 `π*`로 수렴합니다.

```figure
epsilon-greedy
```

## 구축하기 (Build It)

### 1단계: rollout → (s, a, r) 리스트

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

모델은 없으며 오직 `env.reset()`과 `env.step(s, a)`만 사용합니다. Gym 환경과 인터페이스는 같지만 핵심만 남겼습니다.

### 2단계: 리턴 계산 (역방향 스윕, reverse sweep)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

단 한 번의 패스로 `O(T)`의 시간 복잡도를 가집니다. 역방향 재귀식 `G_t = r_{t+1} + γ G_{t+1}`을 사용하여 다시 합산하는 과정을 방지합니다.

### 3단계: first-visit MC 평가 (first-visit MC evaluation)

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

세 줄의 코드가 핵심 작업을 수행합니다: 상태의 첫 방문을 표시하고, 카운트를 증가시키며, 실행 평균(running mean)을 업데이트합니다.

### 4단계: ε-greedy MC 제어 (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 5단계: DP 골드 스탠다드(gold standard)와 비교하기

에피소드 수가 무한히 증가함에 따라(episodes → ∞), `V^π`에 대한 MC 추정치는 Lesson 02에서 구한 DP 결과와 일치해야 합니다. 실제로 4×4 GridWorld에서 50,000개의 에피소드를 실행하면 DP 정답과 `~0.1` 이내의 오차로 일치하는 결과를 얻을 수 있습니다.

## 주의 사항 (Pitfalls)

- **무한 에피소드 (Infinite episodes).** MC는 에피소드가 반드시 *종료(terminate)*되어야 합니다. 만약 정책이 무한 루프를 돌 수 있다면, `max_steps`를 설정하고 해당 제한에 도달하는 것을 암묵적인 실패로 처리하세요. 무작위 정책을 사용하는 GridWorld는 정기적으로 타임아웃이 발생하며, 이는 정상적인 현상입니다. 다만 이를 올바르게 카운트하도록 구성하세요.
- **분산 (Variance).** MC는 전체 리턴(full returns)을 사용합니다. 에피소드가 길어질수록 분산이 매우 커집니다. 에피소드 끝에서 발생한 단 한 번의 불운한 보상이 `V(s_0)`를 동일한 양만큼 변화시킬 수 있습니다. TD 방식(Lesson 04)은 부트스트래핑(bootstrapping)을 통해 이를 줄입니다.
- **상태 커버리지 (State coverage).** 초기 Q 값이 동일할 때 탐욕적(Greedy) MC를 사용하면 단 하나의 행동만 시도하게 됩니다. 반드시 탐색(exploration)을 수행해야 합니다 (ε-greedy, exploring starts, UCB 등).
- **비정상 정책 (Non-stationary policies).** 만약 `π`가 변경된다면(MC 제어에서와 같이), 이전의 리턴들은 서로 다른 정책으로부터 얻은 것입니다. 상수-α MC(Constant-α MC)는 이를 처리할 수 있지만, 표본 평균(sample-average) MC는 처리할 수 없습니다.
- **오프-정책 중요도 샘플링 (Off-policy importance sampling).** 가중치 `π(a|s)/μ(a|s)`는 궤적(trajectory) 전체에 걸쳐 곱해집니다. 이로 인해 호라이즌(horizon)이 길어질수록 분산이 폭발합니다. 결정별 가중 중요도 샘플링(per-decision weighted IS)으로 제한하거나 TD 방식으로 전환하세요.

## 활용 방법 (Use It)

2026년 기준 몬테카를로(Monte Carlo) 방법의 역할:

| 활용 사례 (Use case) | MC를 사용하는 이유 (Why MC) |
|----------|--------|
| 단기 호라이즌 게임 (블랙잭, 포커) | 에피소드가 자연스럽게 종료되며, 리턴(returns)이 명확합니다. |
| 기록된 정책의 오프라인 평가 (Offline evaluation) | 저장된 궤적(trajectories)에 대한 평균 할인 리턴을 계산합니다. |
| 몬테카를로 트리 탐색 (Monte Carlo Tree Search, AlphaZero) | 트리 리프(leaf) 노드에서의 MC 롤아웃(rollouts)이 선택을 가이드합니다. |
| LLM RL 평가 | 주어진 정책에 대해 샘플링된 완성 문장들의 평균 보상을 계산합니다. |
| PPO에서의 베이스라인 추정 (Baseline estimation) | 어드밴티지 타겟 `A_t = G_t - V(s_t)`에서 MC `G_t`를 사용합니다. |
| 강화학습(RL) 교육 | 실제로 작동하는 가장 단순한 알고리즘입니다. 부트스트래핑(bootstrapping)을 제거하여 핵심 원리를 파악할 수 있습니다. |

현대적인 심층 강화학습(Deep-RL) 알고리즘(PPO, SAC)은 `n`-step 리턴 또는 GAE를 통해 순수 MC(전체 리턴)와 순수 TD(1단계 부트스트래핑) 사이를 보간(interpolate)합니다. 두 끝점은 모두 동일한 추정기(estimator)의 인스턴스입니다.

## Ship It

`outputs/skill-mc-evaluator.md`로 저장하세요:

```markdown
---
name: mc-evaluator
description: Monte Carlo 롤아웃을 통해 정책을 평가하고, 가능한 경우 DP(동적 계획법) 비교를 포함한 수렴 보고서를 생성합니다.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

환경(에피소드형, reset+step API 제공)과 정책이 주어지면 다음을 출력합니다:

1. 방법론(Method). First-visit vs every-visit MC. 근거.
2. 에피소드 예산(Episode budget). 목표 횟수, 분산 진단, 예상 표준 오차.
3. 탐색 계획(Exploration plan). ε 스케줄(필요한 경우) 또는 exploring starts.
4. 골드 표준 비교(Gold-standard comparison). 테이블형(tabular)인 경우 DP-최적 V*를, 그렇지 않으면 Q-learning / PPO 베이스라인으로부터 얻은 경계값(bound)을 제시합니다.
5. 종료 확인(Termination check). 최대 스텝 제한, 타임아웃, 종료되지 않는 궤적(trajectory) 처리 방식.

유한한 호라이즌(finite horizon) 제한이 없는 비에피소드형(non-episodic) 작업에 대해 MC를 실행하는 것을 거부합니다. 테이블형 작업의 경우 상태당 100회 미만의 에피소드에서 얻은 V^π 추정치를 보고하는 것을 거부합니다. 분산이 0인 액션을 가진 정책은 탐색 위험(exploration risk)으로 표시합니다.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 4×4 GridWorld에서 균등 무작위 정책(uniform-random policy)에 대한 First-visit MC 평가를 구현해 보세요. 10,000 에피소드를 실행합니다. 에피소드 횟수에 따른 `V(0,0)`의 변화를 DP(Dynamic Programming) 정답과 비교하여 그래프로 그려 보세요.
2. **중간 (Medium).** `ε ∈ {0.01, 0.1, 0.3}`인 ε-greedy MC 제어(control)를 구현해 보세요. 20,000 에피소드 이후의 평균 리턴(mean return)을 비교해 보세요. 곡선은 어떤 형태를 띠나요? 편향-분산 트레이드오프(bias-variance tradeoff)는 어디에서 나타나나요?
3. **어려움 (Hard).** 중요도 샘플링(importance sampling)을 사용하는 *오프-폴리시(off-policy)* MC를 구현해 보세요: 균등 무작위 정책 `μ`를 통해 데이터를 수집하고, 결정론적 최적 정책 `π`에 대한 `V^π`를 추정합니다. 일반 IS(plain IS), 결정 단계별 IS(per-decision IS), 가중 IS(weighted IS)를 비교해 보세요. 어떤 방식의 분산(variance)이 가장 낮습니까?

## 주요 용어 (Key Terms)

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| 몬테카를로 (Monte Carlo) | "무작위 샘플링" | 분포로부터 추출한 iid 샘플들의 평균을 통해 기댓값을 추정합니다. |
| 리턴 `G_t` (Return `G_t`) | "미래 보상" | 단계 `t`부터 에피소드 종료 시점까지의 할인된 보상 합계: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "각 상태를 한 번만 카운트" | 에피소드 내의 첫 번째 방문만이 가치 추정에 기여합니다. |
| Every-visit MC | "모든 방문을 사용" | 모든 방문이 기여합니다. 약간의 편향(bias)이 있을 수 있으나 샘플 효율성이 더 높습니다. |
| ε-greedy | "탐험 노이즈" | 확률 `1-ε`로 탐욕적(greedy) 행동을 선택하고, 확률 `ε`로 무작위 행동을 선택합니다. |
| 중요도 샘플링 (Importance sampling) | "잘못된 분포로부터 샘플링하는 것을 교정" | `μ` 데이터로부터 `V^π`를 추정하기 위해 리턴에 `π(a\|s)/μ(a\|s)` 곱을 곱하여 가중치를 재조정합니다. |
| 온-폴리시 (On-policy) | "나 자신의 데이터로부터 학습" | 타겟 정책(Target policy) = 행동 정책(Behavior policy). Vanilla MC, PPO, SARSA가 해당됩니다. |
| 오프-폴리시 (Off-policy) | "다른 사람의 데이터로부터 학습" | 타겟 정책(Target policy) ≠ 행동 정책(Behavior policy). 중요도 샘플링 MC, Q-learning, DQN이 해당됩니다. |

## 추가 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 가장 권위 있는 기본 교재(canonical treatment).
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit 대 every-visit 분석.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 및 분산 제어(variance control).
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 현대적인 저분산 중요도 샘플링(low-variance IS) 추정기.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 셀프 플레이(self-play)가 초인적인 수준으로 수렴함을 보여준 최초의 대규모 실증적 사례; 이 페이즈 후반부의 모든 레슨에 대한 개념적 선구자 역할을 합니다.
