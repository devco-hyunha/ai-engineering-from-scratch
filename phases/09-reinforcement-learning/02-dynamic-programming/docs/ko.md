# 동적 계획법 (Dynamic Programming) — 정책 반복(Policy Iteration) & 가치 반복(Value Iteration)

> 동적 계획법(Dynamic Programming)은 '치팅(cheating)'을 사용하는 강화학습입니다. 여러분은 이미 전이 함수(transition function)와 보상 함수(reward function)를 알고 있습니다. 그저 `V` 또는 `π`가 더 이상 변하지 않을 때까지 벨만 방정식(Bellman equation)을 반복할 뿐입니다. 이는 모든 샘플링 기반 방법론이 도달하고자 하는 기준점(benchmark)이 됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## 문제 (The Problem)

당신은 모델을 알고 있는 MDP를 가지고 있습니다. 즉, 어떤 상태-행동 쌍에 대해서도 `P(s' | s, a)`와 `R(s, a, s')`를 조회할 수 있습니다. 재고 관리자는 수요 분포를 알고 있습니다. 보드게임은 결정론적 전이(deterministic transitions)를 가집니다. 그리드월드(Gridworld)는 네 줄의 파이썬 코드로 이루어져 있습니다. 당신에게는 *모델*이 있습니다.

Model-free RL(Q-learning, PPO, REINFORCE)은 모델이 없는 경우, 즉 환경으로부터 샘플링만 가능한 경우를 위해 발명되었습니다. 하지만 모델이 있다면 더 빠르고 더 나은 방법이 있습니다. 바로 동적 계획법(Dynamic Programming, DP)입니다. 벨만(Bellman)은 1957년에 이를 설계했습니다. DP는 여전히 정답의 기준을 정의합니다. 사람들이 "이 MDP에 대한 최적 정책(optimal policy)"이라고 말할 때, 그것은 DP가 반환할 정책을 의미합니다.

2026년에도 DP가 필요한 세 가지 이유가 있습니다. 첫째, RL 연구의 모든 테이블형 환경(GridWorld, FrozenLake, CliffWalking)은 골드 표준(gold-standard) 정책을 생성하기 위해 DP로 해결됩니다. 둘째, 정확한 값은 샘플링 방식의 *디버깅(debug)*을 가능하게 합니다. 만약 `V*(s_0)`에 대한 Q-learning의 추정치가 DP의 답과 30% 차이가 난다면, 당신의 Q-learning에는 버그가 있는 것입니다. 셋째, 현대의 오프라인 RL(offline RL) 및 계획(planning) 방법론(MCTS, AlphaZero의 탐색, Phase 9 · 10의 모델 기반 RL)은 모두 학습된 모델 또는 주어진 모델 위에서 벨만 백업(Bellman backup)을 반복 수행합니다.

## 개념 (The Concept)

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**두 알고리즘 모두 벨만(Bellman) 방정식에 대한 고정점 반복(fixed-point iteration) 방식입니다.**

**정책 반복 (Policy iteration).** 정책이 더 이상 변하지 않을 때까지 다음 두 단계를 교대로 수행합니다.

1. *평가 (Evaluation):* 주어진 정책 `π`에 대하여, `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`를 수렴할 때까지 반복 적용하여 `V^π`를 계산합니다.
2. *개선 (Improvement):* 주어진 `V^π`를 바탕으로, `π`가 `V^π`에 대해 탐욕적(greedy)이 되도록 만듭니다: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

수렴이 보장되는 이유는 (a) 각 개선 단계에서 `π`가 유지되거나 특정 상태에 대해 `V^π`가 엄격하게 증가하며, (b) 결정론적 정책(deterministic policies)의 공간이 유한하기 때문입니다. 일반적으로 상태 공간이 크더라도 약 5~20회의 외부 반복(outer iterations) 내에 수렴합니다.

**가치 반복 (Value iteration).** 평가와 개선을 하나의 과정으로 통합합니다. 벨만 *최적성(optimality)* 방정식을 적용합니다:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`가 될 때까지 반복합니다. 마지막에 탐욕적 행동을 취함으로써 정책을 추출합니다. 내부 평가 루프가 없으므로 반복당 속도는 엄격하게 더 빠르지만, 일반적으로 수렴하기까지 더 많은 반복이 필요합니다.

**일반화된 정책 반복 (Generalized policy iteration, GPI).** 이를 통합하는 프레임워크입니다. 가치 함수와 정책은 양방향 개선 루프에 묶여 있습니다. 두 요소가 상호 일관성을 향해 나아가도록 유도하는 모든 방법(비동기 가치 반복, 수정된 정책 반복, Q-learning, actor-critic, PPO)은 GPI의 사례입니다.

**`γ < 1`이 중요한 이유.** 벨만 연산자는 sup-norm에 대해 `γ`-축약(contraction) 성질을 갖습니다: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. 축약 성질은 유일한 고정점과 기하학적 수렴을 의미합니다. `γ < 1` 조건이 없다면 이 보장이 사라지며, 이 경우 유한한 지평선(finite horizon)이나 흡수 종단 상태(absorbing terminal state)가 필요합니다.

```figure
value-iteration-gamma
```

## 직접 구현해 보기 (Build It)

### 1단계: GridWorld MDP 모델 구축 (build the GridWorld MDP model)

Lesson 01에서 사용했던 것과 동일한 4×4 GridWorld를 사용합니다. 여기에 확률적 변형(stochastic variant)을 추가합니다: `0.1`의 확률로 에이전트가 무작위 수직 방향으로 미끄러집니다.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`는 `(s', r, p)`의 리스트를 반환합니다. 이것이 전체 모델입니다.

### 2단계: 정책 평가 (Policy Evaluation)

정책 `π(s) = {action: prob}`가 주어졌을 때, `V`의 변화가 멈출 때까지 벨만 방정식(Bellman equation)을 반복합니다.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 3단계: 정책 개선 (Policy Improvement)

`π`를 `V`에 대한 탐욕적 정책(greedy policy)으로 교체합니다. 만약 `π`가 변경되지 않았다면, 최적 상태에 도달한 것이므로 종료합니다.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 4단계: 하나로 결합하기 (Stitch them together)

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 임의의 시작 정책
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 그리드에서의 일반적인 수렴 속도는 4~6회의 외부 반복(outer iterations)입니다. 결과값으로 `V*(0,0) ≈ -6`과 단계 수를 엄격하게 줄이는 정책(policy)을 출력합니다.

### 5단계: 가치 반복 (value iteration, 단일 루프 버전)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

동일한 고정점(fixed point)을 가지며, 코드 줄 수는 더 적습니다.

## 주의 사항 (Pitfalls)

- **종단 상태(Terminals) 처리 누락.** 흡수 상태(absorbing state)에 벨만 방정식을 적용하면, 아무것도 변화시키지 않는 "최적의 행동"을 여전히 선택하게 됩니다. `if s == terminal: V[s] = 0`과 같이 방어 코드를 작성하세요.
- **Sup-norm vs L2 수렴.** 평균이 아닌 `max |V_new - V|`를 사용하세요. 이론적 보장은 sup-norm에 대해 이루어집니다.
- **In-place vs 동기식 업데이트(Synchronous updates).** `V[s]`를 제자리에서 업데이트하는 방식(Gauss-Seidel)이 별도의 `V_new` 딕셔너리를 사용하는 방식(Jacobi)보다 더 빠르게 수렴합니다. 실제 프로덕션 코드에서는 in-place 방식을 사용합니다.
- **정책 동률(Policy ties).** 두 행동의 Q-값이 동일할 경우, `argmax`가 매 반복마다 서로 다른 방식으로 동률을 해결하여 "정책 안정성(policy stable)" 체크가 진동할 수 있습니다. 안정적인 tie-break 방식(고정된 순서의 첫 번째 행동 선택)을 사용하세요.
- **상태 공간 폭발(State-space explosion).** DP는 1회 스윕(sweep)당 `O(|S| · |A|)`의 복잡도를 가집니다. 약 $10^7$개의 상태까지는 작동하지만, 그 이상에서는 함수 근사(function approximation, 9단계 · 05 이후)가 필요합니다.

## 활용 방법 (Use It)

2026년, DP(동적 계획법)는 정답의 기준점(correctness baseline)이자 플래너(planner)의 내부 루프(inner loop) 역할을 수행합니다.

| 활용 사례 (Use case) | 방법 (Method) |
|----------|--------|
| 작은 규모의 테이블형 MDP를 정확히 해결 | 가치 반복(Value iteration, 더 단순함) 또는 정책 반복(Policy iteration, 더 적은 외부 단계) |
| Q-learning / PPO 구현 검증 | 토이 환경(toy environment)에서 DP 최적값인 `V*`와 비교 |
| 모델 기반 RL (Phase 9 · 10) | 학습된 전이 모델(transition model)에 대한 벨만 백업(Bellman backup) |
| AlphaZero / MuZero의 계획(Planning) | 몬테카를로 트리 탐색(MCTS) = 비동기 벨만 백업(async Bellman backup) |
| 오프라인 RL (CQL, IQL) | 보수적 Q-반복(Conservative Q-iteration) — OOD(분포 외) 행동에 페널티를 부여하는 DP |

누군가 "최적 가치 함수(the optimal value function)"라고 말할 때마다, 그들은 "DP 고정점(DP fixed point)"을 의미하는 것입니다. 논문에서 `V*` 또는 `Q*`를 본다면, 이 루프를 머릿속에 그려 보세요.

## Ship It

`outputs/skill-dp-solver.md`로 저장하세요:

```markdown
---
name: dp-solver
description: 정책 반복(policy iteration) 또는 가치 반복(value iteration)을 통해 소규모 테이블형 MDP를 정확하게 해결합니다. 수렴 동작을 보고합니다.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

알려진 모델을 가진 MDP가 주어지면, 다음을 출력합니다:

1. 선택(Choice). 정책 반복(policy iteration) 대 가치 반복(value iteration). $|S|$, $|A|$, $\gamma$와 관련된 이유.
2. 초기화(Initialization). $V_0$, 시작 정책. 수렴 민감도.
3. 중단(Stopping). Sup-norm 허용 오차 $\epsilon$. 예상 스윕(sweep) 횟수.
4. 검증(Verification). 정확하게 계산된 $V^*(s_0)$. 추출된 탐욕 정책(greedy policy).
5. 용도(Use). 이 베이스라인이 샘플링 기반 방법론을 디버깅하거나 평가하는 데 어떻게 사용될 것인지.

상태 공간(state spaces)이 $10^7$보다 큰 경우 DP 실행을 거부합니다. Sup-norm 확인 없이 수렴을 주장하는 것을 거부합니다. 무한 시계(infinite-horizon) 작업에서 $\gamma \ge 1$인 경우 보장 위반(guarantee violation)으로 표시합니다.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `γ ∈ {0.9, 0.99}`인 4×4 GridWorld에서 가치 반복(value iteration)을 실행해 보세요. `max |ΔV| < 1e-6`이 될 때까지 몇 번의 스윕(sweep)이 필요한가요? `V*`를 4×4 그리드 형태로 출력해 보세요.
2. **중간 (Medium).** *확률적(stochastic)* GridWorld(미끄러질 확률 `0.1`)에서 정책 반복(policy iteration)과 가치 반복(value iteration)을 비교해 보세요. 다음 항목들을 기록하세요: 스윕 횟수, 실제 실행 시간(wall-clock time), 최종 `V*(0,0)`. 반복 횟수 측면에서는 어느 것이 더 빨리 수렴하나요? 실제 실행 시간 측면에서는요?
3. **어려움 (Hard).** 수정된 정책 반복(modified policy iteration)을 구현해 보세요: 평가(evaluation) 단계에서 수렴할 때까지 반복하는 대신 `k`번의 스윕만 수행합니다. `k ∈ {1, 2, 5, 10, 50}`에 대해 `k` 대비 `V*(0,0)` 오차를 그래프로 그려 보세요. 이 곡선은 평가(evaluation)와 개선(improvement) 사이의 트레이드오프(tradeoff)에 대해 무엇을 알려주나요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 정책 반복 (Policy iteration) | "DP 알고리즘" | 정책이 더 이상 변하지 않을 때까지 평가(`V^π`)와 개선(`V^π`에 대한 탐욕적 `π`)을 교대로 수행합니다. |
| 가치 반복 (Value iteration) | "더 빠른 DP" | 한 번의 스윕(sweep)에 벨만 최적성 백업(Bellman optimality backup)을 적용하며, `V*`로 기하급수적으로 수렴합니다. |
| 벨만 연산자 (Bellman operator) | "재귀식" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; sup-norm에서 `γ`-축약(contraction) 성질을 가집니다. |
| 축약 (Contraction) | "DP가 수렴하는 이유" | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`를 만족하는 모든 연산자 `T`는 유일한 고정점(fixed point)을 가집니다. |
| GPI | "모든 것이 DP다" | 일반화된 정책 반복(Generalized Policy Iteration): `V`와 `π`를 상호 일관된 상태로 유도하는 모든 방법론을 의미합니다. |
| 동기식 업데이트 (Synchronous update) | "Jacobi 방식" | 한 번의 스윕 동안 이전의 `V` 값을 사용합니다. 분석이 용이하지만 속도가 느립니다. |
| 인플레이스 업데이트 (In-place update) | "Gauss-Seidel 방식" | 업데이트 중인 `V` 값을 즉시 사용합니다. 실제 환경에서 더 빠르게 수렴합니다. |

## 추가 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 정책 반복(policy iteration)과 가치 반복(value iteration)에 대한 표준적인 설명입니다.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 축소 사상(contraction-mapping) 논증에 대한 엄밀한 다룸을 제공합니다.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 수정된 정책 반복(modified policy iteration)과 그 수렴 분석을 다룹니다.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 정책 반복(policy iteration)에 관한 최초의 논문입니다.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — DP에서 근사 DP(approximate-DP) 및 심층 강화 학습(deep RL)으로 넘어가는 가교 역할을 하며, 이후 모든 강의의 토대가 됩니다.
