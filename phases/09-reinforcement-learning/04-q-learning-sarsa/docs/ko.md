# 시간차 학습(Temporal Difference) — Q-Learning & SARSA

> 몬테카를로(Monte Carlo) 방식은 에피소드가 끝날 때까지 기다립니다. 반면 TD(Temporal Difference)는 다음 가치 추정치를 부트스트래핑(bootstrapping)하여 매 단계마다 업데이트를 수행합니다. Q-learning은 오프-폴리시(off-policy)이며 낙관적(optimistic)인 반면, SARSA는 온-폴리시(on-policy)이며 신중(cautious)합니다. 두 방식 모두 단 한 줄의 코드로 구현 가능하며, 이 단계의 모든 심층 강화 학습(deep-RL) 방법론의 근간이 됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## 문제점 (The Problem)

몬테카를로(Monte Carlo) 방식은 작동하지만, 두 가지 비용이 많이 드는 요구 사항이 있습니다. 에피소드가 반드시 종료되어야 하며, 최종 반환값(return)이 들어온 후에만 업데이트가 이루어진다는 점입니다. 만약 에피소드가 1,000단계라면, MC는 무언가를 업데이트하기 위해 1,000단계를 기다려야 합니다. 이는 분산(variance)은 높고 편향(bias)은 낮지만, 실제 적용 시 속도가 느립니다.

동적 계획법(Dynamic Programming)은 이와 반대되는 특성을 가집니다. 부트스트랩(bootstrapped) 백업을 통해 분산이 0이지만, 알려진 모델(known model)이 필요합니다.

시간차(Temporal Difference, TD) 학습은 그 중간 지점을 찾습니다. 단일 전이 `(s, a, r, s')`로부터 1단계 타겟(one-step target)인 `r + γ V(s')`를 형성하고, `V(s)`를 그 방향으로 조금씩 이동시킵니다. 모델이 필요하지 않으며, 완전한 에피소드도 필요하지 않습니다. 우변(RHS)에 근사된 `V`를 사용하기 때문에 편향이 발생하지만, MC보다 분산이 극적으로 낮으며 첫 단계부터 온라인 업데이트가 가능합니다.

이것이 바로 DQN, A2C, PPO, SAC와 같은 현대 강화학습(RL)의 모든 핵심이 되는 전환점입니다. 9단계(Phase 9)의 나머지 부분은 이번 레슨에서 작성할 1단계 TD 업데이트를 기반으로 구축된 함수 근사(function approximation) 계층과 다양한 기법들로 구성됩니다.

## 개념 (The Concept)

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V에 대한 TD(0) 업데이트:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

괄호 안의 값은 TD 오차(TD error) `δ = r + γ V(s') - V(s)`입니다. 이는 MC(Monte Carlo)에서의 `G_t - V(s_t)`에 대응하는 온라인 방식의 아날로그입니다. 수렴을 위해서는 `α`가 Robbins-Monro 조건(`Σ α = ∞`, `Σ α² < ∞`)을 만족해야 하며, 모든 상태가 무한히 자주 방문되어야 합니다.

**Q-learning.** 제어를 위한 off-policy TD 방법입니다:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 연산은 에이전트가 실제로 어떤 행동을 취하든 관계없이, `s'` 이후부터는 *탐욕적(greedy)* 정책을 따를 것이라고 가정합니다. 이러한 분리 덕분에 Q-learning은 에이전트가 ε-greedy를 통해 탐색하는 동안에도 `Q*`를 학습할 수 있습니다. Mnih et al. (2015)은 이를 Atari 게임에서의 Deep Q-learning으로 발전시켰습니다 (Lesson 05).

**SARSA.** on-policy TD 방법입니다:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

이 이름은 튜플 `(s, a, r, s', a')`에서 유래되었습니다. SARSA는 탐욕적인 `argmax`가 아니라, 에이전트가 다음에 *실제로* 취하는 행동 `a'`를 사용합니다. 실행 중인 ε-greedy 정책 `π`에 따른 `Q^π`로 수렴하며, 극한 상황인 `ε → 0`에서는 `Q*`가 됩니다.

**절벽 걷기(cliff-walking)의 차이점.** 고전적인 절벽 걷기 과제(절벽에서 떨어짐 = 보상 -100)에서, Q-learning은 절벽 가장자리를 따라가는 최적의 경로를 학습하지만 탐색 과정에서 가끔 페널티를 받습니다. 반면 SARSA는 탐색 노이즈를 Q-값에 반영하기 때문에 절벽에서 한 걸음 떨어진 더 안전한 경로를 학습합니다. 학습이 진행됨에 따라 `ε → 0`일 때 두 방법 모두 최적에 도달합니다. 실제 적용 시에는 차이가 있습니다. 배포 시 실제로 탐색이 일어나고 있다면, SARSA의 행동이 더 보수적입니다.

**Expected SARSA.** `Q(s', a')`를 `π` 하에서의 기대값으로 대체합니다:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA보다 분산(variance)이 낮으며( `a'`의 샘플을 사용하지 않음), 동일한 on-policy 타겟을 가집니다. 현대 교과서에서는 주로 이 방식을 기본으로 사용합니다.

**n-step TD 및 TD(λ).** 부트스트래핑(bootstrapping)을 하기 전에 `n` 단계를 기다림으로써 TD(0)와 MC 사이를 보간(interpolate)합니다. `n=1`은 TD이고, `n=∞`는 MC입니다. TD(λ)는 기하학적 가중치 `(1-λ)λ^{n-1}`를 사용하여 모든 `n`에 대해 평균을 냅니다. 대부분의 Deep-RL은 `n`을 3에서 20 사이로 사용합니다.

```figure
qlearning-gridworld
```

## 구현하기 (Build It)

### 1단계: ε-greedy 정책 기반 SARSA (SARSA on ε-greedy policy)

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

여덟 줄입니다. Q-learning과의 *유일한* 차이점은 `target`을 계산하는 줄입니다.

### 2단계: Q-learning

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

`max` 연산자는 타겟(target)을 행동(behavior)으로부터 분리합니다. 이 기호 하나가 온-폴리시(on-policy)와 오프-폴리시(off-policy)를 가르는 차이점입니다.

### 3단계: 학습 곡선 (learning curves)

100 에피소드당 평균 보상(mean return)을 추적합니다. Q-learning은 단순한 결정론적 GridWorld에서 더 빠르게 수렴하며, SARSA는 절벽 걷기(cliff-walking) 환경에서 더 보수적인 경향을 보입니다. `code/main.py`에 구현된 4×4 GridWorld에서 `α=0.1, ε=0.1` 설정 시, 두 알고리즘 모두 약 2,000 에피소드 이후 최적에 가까운 성능을 보입니다.

### 4단계: DP 정답(truth)과 비교하기

`Q*`를 얻기 위해 가치 반복(Value Iteration, Lesson 02)을 실행합니다. `max_{s,a} |Q_learned(s,a) - Q*(s,a)|` 값을 확인해 보세요. 정상적인 테이블형 TD 에이전트라면 4×4 GridWorld 환경에서 10,000 에피소드 수행 후 `~0.5` 이내의 오차 범위로 수렴합니다.

## 주의 사항 (Pitfalls)

- **초기 Q 값의 중요성 (Initial Q values matter).** 낙관적 초기화(Optimistic init, 예: 음수 보상 작업에서 `Q = 0`)는 탐험(exploration)을 촉진합니다. 비관적 초기화는 탐욕적 정책(greedy policy)을 영원히 가두어 버릴 수 있습니다.
- **$\alpha$ 스케줄 ($\alpha$ schedule).** 비정상적(non-stationary) 문제에서는 상수 `α`를 사용해도 괜찮습니다. 감쇠하는 `α_n = 1/n`은 이론적으로는 수렴을 보장하지만 실제로는 너무 느립니다. `α`를 `[0.05, 0.3]` 범위 내로 고정하고 학습 곡선을 모니터링해 보세요.
- **$\epsilon$ 스케줄 ($\epsilon$ schedule).** 높은 값(`ε=1.0`)에서 시작하여 `ε=0.05`까지 감쇠시키세요. "GLIE"(Greedy in the limit with infinite exploration, 무한한 탐험을 통한 점진적 탐욕 정책)가 수렴 조건입니다.
- **Q-learning의 최대값 편향 (Max bias in Q-learning).** `Q` 값에 노이즈가 있을 때 `max` 연산자는 위쪽으로 편향됩니다. 이는 과대평가(overestimation)로 이어집니다. Hasselt의 Double Q-learning(레슨 05의 DDQN에서 사용됨)은 두 개의 Q 테이블을 사용하여 이 문제를 해결합니다.
- **종료되지 않는 에피소드 (Non-terminating episodes).** TD는 종료 상태 없이도 학습할 수 있지만, 단계(steps)를 제한하거나 제한 시점에 부트스트랩(bootstrap)을 올바르게 처리해야 합니다. 표준 방식은 제한 지점을 비종료 상태로 취급하고 부트스트랩을 유지하는 것입니다.
- **상태 해싱 (State hashing).** 상태가 튜플이나 텐서인 경우, 해시 가능한 키를 사용하세요(리스트가 아닌 튜플 사용, 가공되지 않은 값이 아닌 반올림된 부동 소수점 튜플 사용).

## 활용하기 (Use It)

2026년 TD(Temporal Difference) 지형:

| 작업 (Task) | 방법 (Method) | 이유 (Reason) |
|------|--------|--------|
| 작은 정형 데이터 환경 (Small tabular environments) | Q-learning | 최적 정책을 직접 학습합니다. |
| 온폴리시 안전 필수 작업 (On-policy safety-critical) | SARSA / Expected SARSA | 탐색(exploration) 과정에서 보수적입니다. |
| 고차원 상태 (High-dimensional state) | DQN (Phase 9 · 05) | 리플레이(replay)와 타겟 네트워크를 사용하는 신경망 Q-함수입니다. |
| 연속적 행동 (Continuous actions) | SAC / TD3 (Phase 9 · 07) | Q-네트워크에 TD 업데이트를 적용하며, 정책 네트워크가 행동을 출력합니다. |
| LLM RL (보상 모델 기반) | PPO / GRPO (Phase 9 · 08, 12) | GAE를 통해 TD 방식의 어드밴티지(advantage)를 사용하는 액터-크리틱(Actor-critic)입니다. |
| 오프라인 RL (Offline RL) | CQL / IQL (Phase 9 · 08) | 보수적 정규화(conservative regularization)를 적용한 Q-learning입니다. |

2026년 논문에서 접하게 될 "RL"의 90%는 Q-learning이나 SARSA를 정교하게 확장한 것입니다. 더 깊이 공부하기 전에, 테이블 기반 업데이트(tabular update)를 손에 익을 정도로 완벽히 이해해 보세요.

## Ship It

`outputs/skill-td-agent.md`로 저장하세요:

```markdown
---
name: td-agent
description: 테이블형(tabular) 또는 소규모 특징(small-feature) RL 작업에 대해 Q-learning, SARSA, Expected SARSA 중 하나를 선택합니다.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

테이블형 또는 소규모 특징 환경이 주어지면, 다음을 출력하세요:

1. 알고리즘(Algorithm). Q-learning / SARSA / Expected SARSA / n-step 변형 중 선택. On-policy 대 off-policy 여부 및 분산(variance)과 관련된 한 문장 분량의 이유를 포함하세요.
2. 하이퍼파라미터(Hyperparameters). $\alpha$, $\gamma$, $\epsilon$, 감쇠 일정(decay schedule).
3. 초기화(Initialization). $Q_0$ 값(낙관적 초기화 vs 0 초기화) 및 근거.
4. 수렴 진단(Convergence diagnostic). 목표 학습 곡선, DP(Dynamic Programming)가 가능한 경우 $|Q - Q^*|$ 확인.
5. 배포 주의사항(Deployment caveat). 추론(inference) 시 탐험(exploration)이 어떻게 동작할 것인가? SARSA의 보수성(conservatism)이 필요한가?

상태 공간(state space)이 $10^6$보다 큰 경우 테이블형 TD 적용을 거부하세요. 최대 편향(max-bias) 주의사항 없이 Q-learning 에이전트를 배포하는 것을 거부하세요. $\epsilon$이 전체 과정 동안 1.0으로 유지된(활용(exploitation) 단계가 없는) 에이전트는 모두 플래그(flag)를 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 4×4 GridWorld에서 Q-learning과 SARSA를 구현해 보세요. 2,000 에피소드 동안의 학습 곡선(100 에피소드당 평균 리턴)을 그래프로 그려 보세요. 어떤 알고리즘이 더 빠르게 수렴하나요?
2. **중간 (Medium).** 절벽 걷기(cliff-walking) 환경(4×12 크기, 마지막 행은 보상 -100을 받고 시작 지점으로 리셋되는 절벽)을 구축해 보세요. Q-learning과 SARSA의 최종 정책(policy)을 비교해 보세요. 각 알고리즘이 이동하는 경로를 스크린샷으로 찍어 보세요. 어떤 경로가 절벽에 더 가깝나요?
3. **어려움 (Hard).** Double Q-learning을 구현해 보세요. 노이즈가 섞인 보상이 주어지는 GridWorld(매 단계 보상에 가우시안 노이즈 $\sigma=5$ 추가)에서, Q-learning은 `V*(0,0)`을 유의미한 수준으로 과대평가(overestimate)하는 반면, Double Q-learning은 그렇지 않음을 보여 주세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| TD error | "업데이트 신호" | `δ = r + γ V(s') - V(s)`, 부트스트랩된 잔차(bootstrapped residual). |
| TD(0) | "1단계 TD (One-step TD)" | 다음 상태의 추정치만을 사용하여 매 전이(transition) 후에 업데이트함. |
| Q-learning | "오프-폴리시(Off-policy) RL의 기초" | 다음 상태 행동들에 대해 `max`를 적용한 TD 업데이트; 행동 정책(behavior policy)과 관계없이 `Q*`를 학습함. |
| SARSA | "온-폴리시(On-policy) Q-learning" | 실제 수행한 다음 행동을 사용하는 TD 업데이트; 현재의 ε-greedy π에 대한 `Q^π`를 학습함. |
| Expected SARSA | "저분산(low-variance) SARSA" | 샘플링된 `a'`를 π 하에서의 기대값(expectation)으로 대체함. |
| GLIE | "올바른 탐색 스케줄" | Greedy in the Limit with Infinite Exploration; Q-learning의 수렴을 위해 필요함. |
| Bootstrapping | "타겟에 현재 추정치를 사용함" | TD를 MC(Monte Carlo)와 구분 짓는 특징. 편향(bias)의 원인이 되지만 분산(variance)을 크게 줄여줌. |
| Maximization bias | "Q-learning의 과대평가" | 노이즈가 섞인 추정치에 대해 `max`를 취하면 상향 편향됨; Double Q-learning으로 해결 가능. |

## 추가 읽을거리 (Further Reading)

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 원본 논문 및 수렴 증명.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 최대화 편향(maximization bias) 해결 방법.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA의 동기.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSA라는 용어를 처음 만든 논문 (당시에는 "modified connectionist Q-learning"이라 불림).
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)를 TD(n)으로 일반화하며, Q-learning에서 적격성 흔적(eligibility traces)을 거쳐 이후 PPO의 GAE로 이어지는 경로를 다룹니다.
