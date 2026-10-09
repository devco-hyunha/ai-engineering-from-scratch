# 시간 차분 — Q-Learning & SARSA

> 몬테 카를로(Monte Carlo)는 에피소드가 끝날 때까지 기다립니다. TD는 부트스트래핑을 통해 다음 값 추정치를 사용하여 매 단계마다 업데이트합니다. Q-learning은 오프-폴리시(off-policy)이며 낙관적이고, SARSA는 온-폴리시(on-policy)이며 신중합니다. 둘 다 한 줄의 코드로 구현됩니다. 둘 다 이 단계의 모든 심층 RL(deep-RL) 방법의 기반이 됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 01강 (MDP), 9단계 · 02강 (동적 프로그래밍), 9단계 · 03강 (몬테 카를로)
**시간:** 약 75분

## 문제점

몬테 카를로(MC)는 작동하지만 두 가지 비싼 요구 사항이 있습니다. 종료되는 에피소드가 필요하며, 최종 리턴이 들어온 후에만 업데이트합니다. 에피소드가 1,000단계라면 MC는 무언가를 업데이트하기 위해 1,000단계를 기다립니다. 이는 고분산, 저편향이며 실제로는 느립니다.

동적 프로그래밍은 반대 프로필을 가집니다 — 부트스트랩된 백업의 분산이 0입니다 — 하지만 알려진 모델이 필요합니다.

시간 차분(TD) 학습은 차이를 분할합니다. 단일 전이 `(s, a, r, s')`에서 1단계 목표 `r + γ V(s')`를 형성하고 `V(s)`를 그쪽으로 살짝 조정합니다. 모델이 필요 없습니다. 완전한 에피소드도 필요 없습니다. RHS에서 근사 `V`를 사용하므로 편향이 발생하지만, MC보다 분산이 극적으로 낮으며 첫 단계부터 온라인 업데이트가 가능합니다.

이것은 모든 현대 RL — DQN, A2C, PPO, SAC —이 의존하는 전환점입니다. 9단계의 나머지 부분은 이 강에서 작성할 1단계 TD 업데이트 위에 구축된 함수 근사 및 트릭의 레이어입니다.

## 개념

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V에 대한 TD(0) 업데이트:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

괄호 안의 양은 TD 오차 `δ = r + γ V(s') - V(s)`입니다. 이는 MC의 `G_t - V(s_t)`에 대한 온라인 유사체입니다. 수렴하려면 `α`가 Robbins-Monro 조건(`Σ α = ∞`, `Σ α² < ∞`)을 만족해야 하며 모든 상태가 무한히 자주 방문되어야 합니다.

**Q-learning.** 제어(control)를 위한 오프-폴리시 TD 방법입니다:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`는 에이전트가 실제로 어떤 행동을 취하든 `s'` 이후부터 *탐욕적(greedy)* 정책이 따를 것이라고 가정합니다. 이러한 분리(decoupling)로 인해 Q-learning은 에이전트가 ε-탐욕적(ε-greedy)으로 탐색하는 동안 `Q*`를 학습합니다. Mnih et al. (2015)는 이를 Atari에서의 심층 Q-learning으로 변환했습니다 (05강).

**SARSA.** 온-정책 TD 방법입니다:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

이름은 튜플 `(s, a, r, s', a')`입니다. SARSA는 에이전트가 *실제로* 다음에 취하는 행동 `a'`을 사용하며, 탐욕적인 `argmax`을 사용하지 않습니다. 실행 중인 ε-그리디 `π`에 대해 `Q^π`으로 수렴하며, 극한에서는 `ε → 0`이 `Q*`이 됩니다.

**절벽 걷기에서의 차이.** 고전적인 절벽 걷기 작업(절벽에서 떨어짐 = 보상 -100)에서 Q-러닝은 절벽 가장자리를 따라 최적 경로를 학습하지만, 탐색 중에 가끔 페널티를 받습니다. SARSA는 Q-값에 탐색 잡음을 고려하므로 절벽에서 한 칸 떨어진 더 안전한 경로를 학습합니다. 학습을 통해 둘 다 `ε → 0`에서 최적에 도달합니다. 실제로는 중요합니다. 배포 시 탐색이 실제로 일어나고 있을 때, SARSA의 행동은 더 보수적입니다.

**기대 SARSA.** `Q(s', a')`을 `π` 하에서의 기대값으로 대체합니다:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA보다 분산이 낮습니다(`a'`의 샘플이 없음). 동일한 온-정책 목표를 가집니다. 현대 교과서에서는 종종 기본값으로 사용됩니다.

**n-단계 TD 및 TD(λ).** 부트스트랩하기 전에 `n` 단계를 기다림으로써 TD(0)과 MC를 보간합니다. `n=1`은 TD이고, `n=∞`는 MC입니다. TD(λ)는 기하학적 가중치 `(1-λ)λ^{n-1}`를 사용하여 모든 `n`을 평균냅니다. 대부분의 딥 RL은 `n`을 03강 20 사이에서 사용합니다.

```figure
qlearning-gridworld
```

## 구현하기

### 1단계: ε-그리디 정책에서의 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

여덟 줄입니다. Q-러닝과의 *유일한* 차이는 목표 라인입니다.

### 2단계: Q-러닝

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`은 목표와 행동을 분리합니다. 그 하나의 기호가 온-정책과 오프-정책의 차이입니다.

### 3단계: 학습 곡선

100 에포크마다 평균 보적을 추적합니다. Q-러닝은 단순한 결정론적 GridWorld에서 더 빠르게 수렴합니다. SARSA는 절벽 걷기에서 더 보수적입니다. `code/main.py`의 4×4 GridWorld에서는 `α=0.1, ε=0.1`로 약 2,000 에포크 후 둘 다 최적에 가깝습니다.

### 4단계: DP 진실과 비교

값 반복(02강)을 실행하여 `Q*`을 얻습니다. `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`을 확인하세요. 건강한 테이블형 TD 에이전트는 10,000 에포크 후 4×4 GridWorld에서 `~0.5` 이내에 위치합니다.

## 문제점

- **초기 Q 값이 중요합니다.** 낙관적 초기화(`Q = 0`, 음의 보상 작업의 경우)는 탐험을 장려합니다. 비관적 초기화는 탐욕적 정책을 영원히 가둘 수 있습니다.
- **α 스케줄.** 비정상(non-stationary) 문제에는 상수 `α`이 적합합니다. 감쇠하는 `α_n = 1/n`은 이론적으로 수렴을 보장하지만, 실제에서는 너무 느립니다. `[0.05, 0.3]`에서 `α`을 고정하고 학습 곡선을 모니터링하세요.
- **ε 스케줄.** 높은 값(`ε=1.0`)에서 시작하여 `ε=0.05`으로 감쇠하세요. "GLIE"(무한한 탐험을 통해 극한에서 탐욕적)는 수렴 조건입니다.
- **Q-learning의 최대 편향.** `Q`이 잡음이 많을 때 `max` 연산자는 상향 편향을 가집니다. 과대추정으로 이어지며, Hasselt의 Double Q-learning(05강에서 DDQN이 사용)은 두 개의 Q 테이블을 통해 이를 해결합니다.
- **비종결 에피소드.** TD는 종결점 없이 학습할 수 있지만, 단계 수를 제한하거나 제한점에서의 부트스트랩을 올바르게 처리해야 합니다. 표준은 제한점을 비종결점으로 취급하고 부트스트랩을 유지하는 것입니다.
- **상태 해싱.** 상태가 튜플이나 텐서인 경우, 해시 가능한 키를 사용하세요(리스트가 아닌 튜플, 원시 값이 아닌 반올림된 floats의 튜플).

## 사용하기

2026년 TD 현황:

| 작업 | 방법 | 이유 |
|------|--------|--------|
| 작은 표형 환경 | Q-learning | 최적 정책을 직접 학습합니다. |
| 온-폴리시 안전 중시 | SARSA / Expected SARSA | 탐험 중 보수적입니다. |
| 고차원 상태 | DQN (9단계 · 05) | 리플레이 및 타겟 네트워크를 사용하는 신경망 Q 함수. |
| 연속 동작 | SAC / TD3 (9단계 · 07) | Q 네트워크에 TD 업데이트 적용; 정책 네트워크가 동작을 생성합니다. |
| LLM RL (보상 모델 기반) | PPO / GRPO (9단계 · 08, 12) | GAE를 통해 TD 스타일의 이점을 사용하는 액터-크리틱. |
| 오프라인 RL | CQL / IQL (9단계 · 08) | 보수적 정규화를 적용한 Q-learning. |

2026년 논문에서 읽는 "RL"의 90%는 Q-learning이나 SARSA의 변형입니다. 심층적으로 읽기 전에 표형 업데이트를 손에 익히세요.

## 출시하기

`outputs/skill-td-agent.md`으로 저장하세요:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 Q-learning과 SARSA를 구현하세요. 2,000 에포크에 대해 학습 곡선(100 에포크당 평균 리턴)을 플롯하세요. 누가 더 빨리 수렴하나요?
2. **중간.** 절벽 걷기(cliff-walking) 환경을 구축하세요 (4×12, 마지막 행은 절벽이며 보상 -100을 받고 시작점으로 리셋됨). Q-learning과 SARSA의 최종 정책을 비교하세요. 각 알고리즘이 취하는 경로를 스크린샷으로 찍어보세요. 어느 쪽이 절벽에 더 가까운가요?
3. **어려움.** Double Q-learning을 구현하세요. 잡음 있는 보상 GridWorld (단계별 보상에 가우시안 잡음 σ=5 추가)에서, Q-learning이 `V*(0,0)`을 의미 있는 정도로 과대평가하는 반면 Double Q-learning은 그렇지 않음을 보이세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| TD error | "업데이트 신호" | `δ = r + γ V(s') - V(s)`, 부트스트랩된 잔차. |
| TD(0) | "1단계 TD" | 모든 전이(transitions) 후 다음 상태의 추정치만 사용하여 업데이트. |
| Q-learning | "Off-policy RL 입문" | 다음 상태의 행동에 대해 `max`을 사용하는 TD 업데이트; 행동 정책과 무관하게 `Q*`을 학습. |
| SARSA | "On-policy Q-learning" | 실제 다음 행동을 사용하는 TD 업데이트; 현재 ε-greedy π에 대해 `Q^π`을 학습. |
| Expected SARSA | "저분산 SARSA" | 샘플링된 `a'`을 π 하에서의 기댓값으로 대체. |
| GLIE | "올바른 탐색 스케줄" | Greedy in the Limit with Infinite Exploration; Q-learning 수렴에 필요. |
| Bootstrapping | "타겟에서 현재 추정치 사용" | TD를 MC와 구분하는 점. 편향의 원천이지만 분산을 대폭 줄임. |
| Maximization bias | "Q-learning이 과대평가함" | 잡음 있는 추정치에 대한 `max`은 상향 편향됨; Double Q-learning으로 수정. |

## 추가 읽기

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 원 논문 및 수렴 증명.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — maximization bias에 대한 수정.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA의 동기.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSA를 명명한 논문 (당시 "modified connectionist Q-learning"라 불림).
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)를 TD(n)으로 일반화하며, Q-learning에서 eligibility traces 및 이후 PPO의 GAE로 이어지는 경로.
