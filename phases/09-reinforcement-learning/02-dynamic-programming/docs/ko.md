# 동적 프로그래밍 — 정책 반복 및 가치 반복

> 동적 프로그래밍은 반칙을 쓰는 RL입니다. 이미 전이 함수와 보상 함수를 알고 있으며, `V` 또는 `π`이 더 이상 변하지 않을 때까지 Bellman 방정식을 반복하는 것뿐입니다. 모든 샘플링 기반 방법이 접근하려고 하는 벤치마크입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 01강 (MDP)
**시간:** 약 75분

## 문제점

모델이 알려진 MDP가 있습니다. 모든 상태-행동 쌍에 대해 `P(s' | s, a)`과 `R(s, a, s')`을 쿼리할 수 있습니다. 재고 관리자는 수요 분포를 알고 있습니다. 보드 게임은 결정론적 전이를 가집니다. 그리드월드는 Python 네 줄로 표현됩니다. *모델*이 있습니다.

모델 프리 RL (Q-learning, PPO, REINFORCE)는 모델이 없는 경우, 즉 환경에서만 샘플링할 수 있는 경우를 위해 발명되었습니다. 하지만 모델이 있을 때는 더 빠르고 더 나은 방법, 즉 동적 프로그래밍이 있습니다. Bellman은 1957년에 이 방법들을 설계했습니다. 이들은 여전히 정확성을 정의합니다. 사람들이 "이 MDP에 대한 최적 정책"이라고 말할 때, 동적 프로그래밍이 반환하는 정책을 의미합니다.

2026년에도 이 방법들이 필요한 이유는 세 가지입니다. 첫째, RL 연구의 모든 테이블형 환경(GridWorld, FrozenLake, CliffWalking)은 동적 프로그래밍으로 해결되어 표준 정책을 생성합니다. 둘째, 정확한 값은 샘플링 방법을 *디버깅*하는 데 사용할 수 있습니다. Q-learning의 `V*(s_0)` 추정치가 동적 프로그래밍 답과 30% 차이가 난다면, Q-learning에 버그가 있는 것입니다. 셋째, 현대의 오프라인 RL 및 계획 방법(MCTS, AlphaZero의 탐색, 9단계 · 10강의 모델 기반 RL)은 모두 학습된 모델이나 주어진 모델에 대해 Bellman 백업을 반복합니다.

## 개념

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**두 가지 알고리즘, 모두 Bellman에 대한 고정점 반복입니다.**

**정책 반복.** 정책이 더 이상 변하지 않을 때까지 두 단계를 번갈아 수행합니다.

1. *평가:* 정책 `π`이 주어지면, `V^π`이 수렴할 때까지 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`를 반복 적용하여 을 계산합니다.
2. *개선:* `V^π`이 주어지면, `V^π`에 대해 `π`을 탐욕적으로 만듭니다: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

수렴이 보장되는 이유는 (a) 각 개선 단계가 `π`을 동일하게 유지하거나 특정 상태에서 `V^π`을 엄격하게 증가시키며, (b) 결정론적 정책의 공간이 유한하기 때문입니다. 일반적으로 큰 상태 공간에서도 약 5~20번의 외부 반복으로 수렴합니다.

**가치 반복.** 평가와 개선을 하나의 스윕으로 통합합니다. Bellman *최적성* 방정식을 적용하세요:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`이 될 때까지 반복하세요. 마지막에贪婪(greedy) 행동을 선택하여 정책을 추출합니다. 반복당 속도가 더 빠릅니다 — 내부 평가 루프가 없음 — 하지만 수렴에는 일반적으로 더 많은 반복이 필요합니다.

**일반화된 정책 반복(GPI).** 통합된 프레임워크입니다. 가치 함수와 정책은 양방향 개선 루프에 묶여 있으며, 둘 모두를 상호 일관성으로 이끄는 모든 방법(비동기 가치 반복, 수정된 정책 반복, Q-학습, actor-critic, PPO)은 GPI의 인스턴스입니다.

**`γ < 1`이 중요한 이유.** Bellman 연산자는 sup-norm에서 `γ`-축소 연산자입니다: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. 축소는 유일한 고정점과 기하급수적 수렴을 의미합니다. `γ < 1`을 제거하면 보장이 사라집니다 — 유한 기간이나 흡수 상태가 필요합니다.

```figure
value-iteration-gamma
```

## 구현하기

### 1단계: GridWorld MDP 모델 구축

1강의와 동일한 4×4 GridWorld를 사용하세요. 확률적 변형을 추가합니다: 확률 `0.1`로 에이전트가 랜덤한 수직 방향으로 미끄러집니다.

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

`transitions(s, a)`은 `(s', r, p)`의 리스트를 반환합니다. 이것이 전체 모델입니다.

### 2단계: 정책 평가

정책 `π(s) = {action: prob}`이 주어지면, `V`이 더 이상 변하지 않을 때까지 Bellman 방정식을 반복하세요:

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

### 3단계: 정책 개선

`π`을 `V`에 대한贪婪(greedy) 정책으로 교체하세요. `π`이 변하지 않았다면 반환하세요 — 최적점에 도달했습니다.

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

### 4단계: 두 단계를 결합하기

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 임의의 시작점
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4에서의 전형적인 수렴: 4~6번의 외부 반복. `V*(0,0) ≈ -6`과 단계 수를 엄격하게 감소시키는 정책을 출력합니다.

### 5단계: 가치 반복 (단일 루프 버전)

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

동일한 고정점, 더 짧은 코드 라인.

## 주의할 점

- **터미널 처리를 잊는 경우.** 흡수 상태에 Bellman을 적용하면 아무것도 변하지 않는 "최적 행동"을 선택합니다. `if s == terminal: V[s] = 0`로 보호하세요.
- **Sup-norm vs L2 수렴.** 평균이 아닌 `max |V_new - V|`을 사용하세요. 이론적 보장은 sup-norm에 대한 것입니다.
- **제자리 업데이트 vs 동기 업데이트.** `V[s]`을 제자리에서 업데이트(Gauss-Seidel)하면 별도의 `V_new` dict(Jacobi)보다 더 빠르게 수렴합니다. 프로덕션 코드는 제자리 업데이트를 사용합니다.
- **정책 동점.** 두 행동의 Q-value가 같으면 `argmax`이 매 반복마다 동점을 다르게 처리하여 "정책 안정" 체크가 진동할 수 있습니다. 안정적인 동점 처리(고정된 순서에서 첫 번째 행동)를 사용하세요.
- **상태 공간 폭발.** DP는 스윕당 `O(|S| · |A|)`입니다. 약 10⁷개의 상태까지 작동합니다. 그 이상에서는 함수 근사(9단계 · 05강 이후)가 필요합니다.

## 사용하기

2026년, DP는 정확성 기준선이며 플래너의 내부 루프입니다:

| 사용 사례 | 방법 |
|----------|--------|
| 작은 표형 MDP를 정확히 해결 | 가치 반복(더 단순) 또는 정책 반복(외부 단계가 더 적음) |
| Q-learning / PPO 구현 검증 | 장난감 환경에서 DP 최적 V*와 비교 |
| 모델 기반 RL (9단계 · 10강) | 학습된 전이 모델에 대한 Bellman 백업 |
| AlphaZero / MuZero에서의 플래닝 | Monte Carlo Tree Search = 비동기 Bellman 백업 |
| 오프라인 RL (CQL, IQL) | 보수적 Q-반복 — OOD 행동에 페널티가 있는 DP |

"최적 가치 함수"라고 말할 때마다 "DP 고정점"을 의미합니다. 논문에서 `V*` 또는 `Q*`을 보면 이 루프를 상상해 보세요.

## 출시하기

`outputs/skill-dp-solver.md`로 저장하세요:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 연습 문제

1. **쉬움.** `γ ∈ {0.9, 0.99}`을 사용하여 4×4 GridWorld에서 가치 반복을 실행하세요. `max |ΔV| < 1e-6`이 될 때까지 몇 번의 스윕이 필요합니까? `V*`를 4×4 그리드로 출력하세요.
2. **중간.** *확률적* GridWorld(슬립 확률 `0.1`)에서 정책 반복과 가치 반복을 비교하세요. 스윕 수, 실시간(wall-clock time), 최종 `V*(0,0)`을 세어 보세요. 반복 횟수에서는 어떤 것이 더 빠르게 수렴합니까? 실시간에서는요?
3. **어려움.** 수정된 정책 반복을 구축해 보세요: 평가 단계에서 수렴까지 진행하지 않고 `k` 스윕만 실행합니다. `k ∈ {1, 2, 5, 10, 50}`에 대해 `V*(0,0)` 오차와 `k`의 관계를 그래프로 그려 보세요. 이 곡선은 평가/개선 트레이드오프에 대해 무엇을 알려주나요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 정책 반복 | "DP 알고리즘" | 정책이 더 이상 변하지 않을 때까지 평가(`V^π`)와 개선(`V^π`에 대한 greedy `π`)을 번갈아 수행하는 것. |
| 가치 반복 | "더 빠른 DP" | 벨만 최적성 백업을 한 번의 스윕에 적용; `V*`에 기하급수적으로 수렴. |
| 벨만 연산자 | "재귀식" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; sup-norm에서 `γ`-축소 연산자. |
| 축소 | "DP가 수렴하는 이유" | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`을 만족하는 모든 연산자 `T`는 유일한 고정점을 가짐. |
| GPI | "모든 것이 DP" | 일반화된 정책 반복: `V`과 `π`을 상호 일관성으로 이끄는 모든 방법. |
| 동기 업데이트 | "Jacobi 스타일" | 스윕 전체 동안 이전 `V`을 사용; 분석이 깔끔하지만 더 느림. |
| 제자리 업데이트 | "Gauss-Seidel 스타일" | 업데이트되는 `V`을 즉시 사용; 실제로 더 빠르게 수렴. |

## 추가 읽기

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 정책 반복과 가치 반복의 표준적인 설명.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — 축소 사상 논증에 대한 엄밀한 처리.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 수정된 정책 반복과 그 수렴 분석.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 정책 반복의 원 논문.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — DP에서 근사 DP / 심층 RL로 이어지는 다리이며, 이후 모든 강의에서 사용됨.
