# 몬테 카를로 방법 — 완전한 에피소드에서 학습하기

> 동적 프로그래밍은 모델이 필요합니다. 몬테 카를로 방법은 에피소드만 있으면 됩니다. 정책을 실행하고, 리턴을 관찰하고, 평균을 내 보세요. RL에서 가장 단순한 아이디어이며, 이후 모든 것을 가능하게 하는 방법입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 01강 (MDP), 9단계 · 02강 (동적 프로그래밍)
**시간:** 약 75분

## 문제점

동적 프로그래밍은 우아하지만, 모든 상태와 행동에 대해 `P(s' | s, a)`을 쿼리할 수 있다고 가정합니다. 현실 세계에서는 거의 모든 것이 그렇게 작동하지 않습니다. 로봇은 관절 토크 이후 카메라 픽셀의 분포를 분석적으로 계산할 수 없습니다. 가격 책정 알고리즘은 모든 가능한 고객 반응에 대해 적분할 수 없습니다. LLM은 토큰 이후 모든 가능한 연속을 열거할 수 없습니다.

환경에서 *샘플링*하는 능력만 필요한 방법이 필요합니다. 정책을 실행하세요. 궤적 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`을 얻으세요. 이를 사용하여 값을 추정하세요. 이것이 몬테 카를로 방법입니다.

DP에서 MC로의 전환은 철학적으로 중요합니다: *알려진 모델 + 정확한 백업*에서 *샘플링된 롤아웃 + 평균화된 리턴*으로 이동합니다. 분산은 증가하지만, 적용 가능성은 폭발적으로 증가합니다. 이 강의 이후의 모든 RL 알고리즘 — TD, Q-learning, REINFORCE, PPO, GRPO —은 본질적으로 몬테 카를로 추정자이며, 때로는 부트스트래핑이 그 위에 겹쳐져 있습니다.

## 개념

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**핵심 아이디어, 한 줄로:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`, 여기서 `G^{(i)}(s)`은 정책 `π` 하에서 `s`를 방문한 후 관찰된 리턴입니다.

**첫 방문 vs 모든 방문 MC.** 상태 `s`를 여러 번 방문하는 에피소드가 주어지면, 첫 방문 MC는 첫 방문에서의 리턴만 계산합니다. 모든 방문 MC는 모든 방문을 계산합니다. 둘 다 극한에서 편향되지 않습니다. 첫 방문은 분석하기 더 쉽습니다 (iid 샘플). 모든 방문은 에피소드당 더 많은 데이터를 사용하며, 일반적으로 실제에서 더 빠르게 수렴합니다.

**증분 평균.** 모든 리턴을 저장하는 대신, 진행 중인 평균을 업데이트하세요:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

재구성: `V_new = V_old + α · (target - V_old)`을 `α = 1/n`과 함께 사용하세요. `1/n`를 상수 스텝 크기 `α ∈ (0, 1)`으로 바꾸면 `π`의 변화를 추적하는 비정상(non-stationary) MC 추정기를 얻을 수 있습니다. 이 변화는 MC에서 TD로, 그리고 모든 현대 RL 알고리즘으로 이어지는 핵심적인 도약입니다.

**탐색이 이제 문제가 됩니다.** DP는 열거를 통해 모든 상태를 다루었습니다. MC는 정책이 방문하는 상태만 볼 수 있습니다. `π`이 결정적(deterministic)이라면, 상태 공간의 전체 영역이 샘플링되지 않으며, 해당 가치 추정치는 영원히 0으로 남습니다. 역사적 순서대로 세 가지 해결책이 있습니다:

1. **탐색 시작(Exploring starts).** 각 에피소드를 랜덤한 (s, a) 쌍에서 시작합니다. 커버리지를 보장하지만, 실제에서는 비현실적입니다(로봇을 임의의 상태로 '리셋'할 수 없습니다).
2. **ε-그리디(ε-greedy).** 현재 Q에 대해 그리디하게 행동하되, 확률 `ε`으로 랜덤한 행동을 선택합니다. 모든 상태-행동 쌍이 점진적으로 샘플링됩니다.
3. **오프-폴리시 MC(Off-policy MC).** 행동 정책 `μ` 아래에서 데이터를 수집하고, 중요도 샘플링(importance sampling)을 통해 목표 정책 `π`에 대해 학습합니다. 분산이 높지만, DQN 같은 리플레이 버퍼(replay-buffer) 방법론으로 이어지는 다리 역할을 합니다.

**Monte Carlo Control.** 정책 반복(policy iteration)처럼 평가 → 개선 → 평가를 수행하되, 평가는 샘플링 기반입니다:

1. `π`을 실행하여 에피소드를 얻습니다.
2. 관측된 리턴(return)으로부터 `Q(s, a)`을 업데이트합니다.
3. `Q`에 대해 `π`을 ε-그리디로 만듭니다.
4. 반복합니다.

온건한 조건(모든 쌍이 무한히 자주 방문되고, `α`가 Robbins-Monro 조건을 만족) 하에서 확률 1로 `Q*`과 `π*`에 수렴합니다.

```figure
epsilon-greedy
```

## 구현하기

### 1단계: 롤아웃(rollout) → (s, a, r) 목록

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

모델이 없으며, `env.reset()`과 `env.step(s, a)`만 사용합니다. gym 환경과 동일한 인터페이스를 사용하되 단순화되었습니다.

### 2단계: 리턴 계산(역방향 스윕)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

한 번의 패스, `O(T)`. 역점화식 `G_t = r_{t+1} + γ G_{t+1}`을 통해 재합산을 피합니다.

### 3단계: 첫 방문(first-visit) MC 평가

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

세 줄 코드가 작업을 수행합니다: 첫 방문 시 상태를 표시하고, 카운트를 증가시키며, 이동 평균을 업데이트합니다.

### 4단계: ε-그리디 MC 제어(on-policy)

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

### 5단계: DP 골든 스탠더드(gold standard)와 비교

`V^π`의 MC 추정값은 에피소드 수 → ∞에서 02강의 DP 결과와 일치해야 합니다. 실제로는 4×4 GridWorld에서 50,000 에피소드를 실행하면 DP 답과 `~0.1` 이내의 차이를 보입니다.

## 함정

- **무한 에피소드.** MC는 에피소드가 *종료*되어야 합니다. 정책이 무한 루프에 빠질 수 있다면 `max_steps`에 상한을 설정하고, 상한 도달을 암묵적인 실패로 처리하세요. GridWorld에서 랜덤 정책을 사용하면 시간 초과가 흔하게 발생하는데, 이는 정상적인 현상이므로 정확히 계산하도록 하세요.
- **분산.** MC는 전체 리턴을 사용합니다. 긴 에피소드에서는 분산이 매우 커지며, 마지막에 운이 나쁜 보상이 하나 발생하면 `V(s_0)`가 동일한 크기로 변합니다. TD 방법(04강)은 부트스트래핑을 통해 이를 줄입니다.
- **상태 커버리지.** 새로운 Q에서 동점이 있는 경우 Greedy MC는 항상 하나의 행동만 시도합니다. *반드시* 탐색(ε-greedy, 탐색 시작, UCB)을 해야 합니다.
- **비정상 정책.** `π`가 변하면(MC 제어의 경우처럼), 이전 리턴은 다른 정책에서 나온 것입니다. Constant-α MC는 이를 처리하지만, 표본 평균 MC는 처리하지 못합니다.
- **오프 정책 중요도 샘플링.** 가중치 `π(a|s)/μ(a|s)`는 궤적 전체에 걸쳐 곱해집니다. 호리즌이 길어지면 분산이 폭발합니다. 의사결정별 가중치 IS로 상한을 설정하거나 TD로 전환하세요.

## 사용하기

2026년 몬테 카를로 방법의 역할:

| 사용 사례 | MC를 사용하는 이유 |
|----------|--------|
| 짧은 호리즌 게임(블랙잭, 포커) | 에피소드가 자연스럽게 종료되며, 리턴이 명확합니다. |
| 기록된 정책의 오프라인 평가 | 저장된 궤적에 대한 평균 할인 리턴을 계산합니다. |
| 몬테 카를로 트리 검색(AlphaZero) | 트리 잎에서의 MC 롤아웃이 선택을 안내합니다. |
| LLM RL 평가 | 주어진 정책에 대해 샘플링된 완성의 평균 보상을 계산합니다. |
| PPO에서의 기준선 추정 | 이점 목표 `A_t = G_t - V(s_t)`는 MC `G_t`을 사용합니다. |
| RL 교육 | 실제로 작동하는 가장 단순한 알고리즘 — 부트스트래핑을 제거하여 핵심을 확인하세요. |

현대 딥 RL 알고리즘(PPO, SAC)은 `n`-스텝 리턴이나 GAE를 통해 순수 MC(전체 리턴)와 순수 TD(1스텝 부트스트랩) 사이를 보간합니다. 두 끝점 모두 동일한 추정기의 인스턴스입니다.

## 출시하기

`outputs/skill-mc-evaluator.md`로 저장하세요:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 균일 랜덤 정책의 최초 방문 MC 평가를 구현하세요. 10,000 에피소드를 실행하고, DP 정답에 대해 에피소드 수의 함수로 `V(0,0)`을 플롯하세요.
2. **중간.** `ε ∈ {0.01, 0.1, 0.3}`을 사용하여 ε-그리디 MC 제어 구현을 하세요. 20,000 에피소드 후 평균 리턴을 비교하세요. 곡선은 어떤 모양인가요? 편차-분산 트레이드오프는 어디에 위치하나요?
3. **어려움.** 중요 샘플링을 사용하는 *오프 정책* MC를 구현하세요. 균일 랜덤 정책 `μ` 아래에서 데이터를 수집하고, 결정적 최적 정책 `π`에 대해 `V^π`을 추정하세요. 순수 IS, 결정별 IS, 가중 IS를 비교하세요. 어느 것이 분산이 가장 낮은가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 몬테 카를로 | "랜덤 샘플링" | 분포에서 iid 샘플을 평균 내어 기대값을 추정합니다. |
| 리턴 `G_t` | "미래 보상" | 단계 `t`부터 에피소드 종료까지의 할인된 보상 합: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| 최초 방문 MC | "각 상태를 한 번만 세기" | 에피소드 내의 최초 방문만 가치 추정에 기여합니다. |
| 모든 방문 MC | "모든 방문 사용" | 모든 방문이 기여합니다. 약간 편향되어 있지만 샘플 효율성이 더 높습니다. |
| ε-그리디 | "탐색 잡음" | 확률 `1-ε`로 그리디 행동을 선택하고, 확률 `ε`로 랜덤 행동을 선택합니다. |
| 중요 샘플링 | "잘못된 분포에서 샘플링하는 것을 보정하기" | `π(a\|s)/μ(a\|s)` 곱으로 리턴을 재가중하여 `μ` 데이터로부터 `V^π`을 추정합니다. |
| 온 정책 | "내 자신의 데이터에서 학습하기" | 목표 정책 = 행동 정책. 순수 MC, PPO, SARSA. |
| 오프 정책 | "다른 사람의 데이터에서 학습하기" | 목표 정책 ≠ 행동 정책. 중요 샘플링 MC, Q-학습, DQN. |

## 추가 읽기

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 표준적인 처리.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 최초 방문 vs 모든 방문 분석.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 오프 정책 MC 및 분산 제어.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 현대적인 저분산 IS 추정자.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 자기 대국이 초인적 플레이로 수렴하는 최초의 대규모 실증적 시연; 이 단계 후반부의 모든 강의에 대한 개념적 선구자.
