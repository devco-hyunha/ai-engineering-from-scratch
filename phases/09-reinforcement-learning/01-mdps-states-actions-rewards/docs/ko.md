# MDP, 상태(States), 행동(Actions) 및 보상(Rewards)

> 마르코프 결정 과정(Markov Decision Process)은 다섯 가지 요소로 구성됩니다: 상태(states), 행동(actions), 전이(transitions), 보상(rewards), 그리고 할인율(discount)입니다. Q-learning, PPO, DPO, GRPO를 포함한 강화학습(RL)의 모든 것은 이 구조를 바탕으로 최적화됩니다. 이것을 한 번 제대로 배우면, 나머지 강화학습 내용은 거저 얻는 것과 같습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## 문제 (The Problem)

당신은 체스 봇을 작성하고 있습니다. 혹은 재고 계획가, 트레이딩 에이전트, 또는 추론 모델을 학습시키는 PPO 루프를 만들고 있을 수도 있습니다. 네 가지 서로 다른 도메인이지만, 한 가지 놀라운 사실이 있습니다. 이 네 가지 모두 동일한 수학적 객체로 수렴한다는 점입니다.

지도 학습(Supervised learning)은 `(x, y)` 쌍을 제공하며 함수를 맞추라고 요구합니다. 반면 강화 학습(Reinforcement learning)은 레이블을 제공하지 않습니다. 오직 상태(states)의 흐름, 당신이 취한 행동(actions), 그리고 스칼라 보상(scalar reward)만을 제공할 뿐입니다. 그 수가 게임에서 승리했나요? 재고 보충 결정이 비용을 절감했나요? 거래가 이익을 냈나요? LLM이 방금 생성한 토큰이 판정자로부터 더 높은 보상을 이끌어냈나요?

이 흐름을 공식화(formalize)하기 전까지는 이를 통해 학습할 수 없습니다. "내가 본 것", "내가 한 행동", "다음에 일어난 일", "그것이 얼마나 좋았는지" — 이 각각은 당신이 추론할 수 있는 하나의 객체가 되어야 합니다. 그 공식화된 형태가 바로 마르코프 결정 과정(Markov Decision Process)입니다. 마지막 단계의 RLHF 및 GRPO 루프를 포함하여, 이 단계의 모든 RL 알고리즘은 이 구조를 바탕으로 최적화를 수행합니다.

## 개념 (The Concept)

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**다섯 가지 요소.**

- **상태 (States)** `S`. 에이전트가 결정을 내리는 데 필요한 모든 것. GridWorld에서는 셀(cell), 체스에서는 보드(board), LLM에서는 컨텍스트 창(context window)과 모든 메모리를 의미합니다.
- **행동 (Actions)** `A`. 선택지. 위/아래/왼쪽/오른쪽으로 이동하기, 수를 두기, 토큰 생성하기 등입니다.
- **전이 (Transitions)** `P(s' | s, a)`. 상태 `s`와 행동 `a`가 주어졌을 때, 다음 상태에 대한 확률 분포입니다. 체스에서는 결정론적(deterministic)이고, 인벤토리 관리에서는 확률적(stochastic)이며, LLM 디코딩에서는 거의 결정론적(almost-deterministic)입니다.
- **보상 (Rewards)** `R(s, a, s')`. 스칼라 신호입니다. 승리 = +1, 패배 = -1, 매출에서 비용을 뺀 값, 또는 GRPO에서의 로그 우도비(log-likelihood ratio) 항이 될 수 있습니다.
- **할인율 (Discount)** `γ ∈ [0, 1)`. 미래의 보상이 현재와 비교해 얼마나 중요한지를 나타냅니다. `γ = 0.99`는 약 100단계의 지평(horizon)을 확보하며, `γ = 0.9`는 약 10단계를 확보합니다.

**마르코프 성질 (The Markov property)** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. 미래는 오직 현재의 상태에만 의존합니다. 만약 그렇지 않다면, 상태 표현(state representation)이 불완전한 것입니다. 이는 방법론의 실패가 아니라 상태의 실패입니다.

**정책과 수익 (Policies and returns).** 정책 `π(a | s)`는 상태를 행동 분포로 매핑합니다. 수익(return) `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`은 미래 보상의 할인된 합계입니다. 가치(value) `V^π(s) = E[G_t | s_t = s]`는 정책 `π` 하에서 `s`로부터 시작했을 때의 기대 수익입니다. Q-값 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`는 특정 행동으로 시작했을 때의 기대 수익입니다. 모든 RL 알고리즘은 이 두 가지 중 하나를 추정하고, 그에 따라 `π`를 개선합니다.

**벨만 방정식 (The Bellman equations).** 이 단계의 모든 것이 사용하는 고정점 방정식(fixed-point equations)입니다:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

이 방정식들은 기대 수익을 "이번 단계의 보상"과 "도착한 지점의 할인된 가치"로 나눕니다. 재귀적(Recursive) 구조입니다. Phase 9의 모든 알고리즘은 이 방정식을 수렴할 때까지 반복(동적 계획법, dynamic programming)하거나, 이로부터 샘플링(몬테카를로, Monte Carlo)하거나, 한 단계씩 부트스트랩(시간차 학습, temporal difference)합니다.

```figure
discount-horizon
```

## 직접 구현해 보기 (Build It)

### 1단계: 아주 작은 결정론적 MDP (a tiny deterministic MDP)

4×4 GridWorld입니다. 에이전트는 왼쪽 상단에서 시작하며, 오른쪽 하단이 종료(terminal) 지점입니다. 매 스텝마다 -1의 보상을 받으며, 가능한 행동은 `{up, down, left, right}`입니다. `code/main.py`를 참조하세요.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

단 다섯 줄입니다. 이것이 환경의 전부입니다. 결정론적 전이(deterministic transitions), 일정한 스텝 페널티(constant step penalty), 그리고 흡수 상태인 종료 지점(absorbing terminal state)으로 구성되어 있습니다.

### 2단계: 정책 실행 (roll out a policy)

정책(Policy)은 상태(state)에서 행동 분포(action distribution)로 매핑되는 함수입니다. 가장 단순한 형태는 균등 무작위(uniform random) 정책입니다.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

무작위 정책을 1,000번 실행해 보세요. 이 4×4 보드에서 평균 보상(average return)은 약 -60에서 -80 사이입니다. 최적의 보상은 -6(오른쪽 아래로 향하는 직선 경로)입니다. 9단계(Phase 9)의 핵심은 바로 그 격차를 줄이는 것입니다.

### 3단계: 벨만 방정식(Bellman equation)을 통해 `V^π`를 정확하게 계산하기

규모가 작은 MDP의 경우, 벨만 방정식은 선형 시스템(linear system)이 됩니다. 상태(states)를 열거하고, 기댓값(expectation)을 적용하며, 가치(values)가 더 이상 변하지 않을 때까지 반복합니다.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

이것은 반복적 정책 평가(iterative policy evaluation)입니다. 이는 Sutton & Barto 교재에 등장하는 첫 번째 알고리즘이며, 이후 등장하는 모든 강화 학습(RL) 방법론의 이론적 토대가 됩니다.

### 4단계: `γ`는 물리적 의미를 갖는 하이퍼파라미터입니다

유효 지평(Effective horizon)은 대략 `1 / (1 - γ)`입니다. `γ = 0.9`이면 약 10단계, `γ = 0.99`이면 약 100단계, `γ = 0.999`이면 약 1000단계가 됩니다.

`γ` 값이 너무 낮으면 에이전트가 근시안적(myopically)으로 행동합니다. 반대로 너무 높으면 초기 단계의 많은 행동이 먼 미래의 보상에 대해 책임을 공유하게 되므로, 신용 할당(credit assignment)에 노이즈가 발생합니다. LLM RLHF는 에피소드가 짧고 제한적이기 때문에 일반적으로 `γ = 1`을 사용합니다. 제어(Control) 작업에는 `0.95–0.99`를 사용하며, 장기 지평(Long-horizon) 전략 게임에는 `0.999`를 사용합니다.

## 함정 (Pitfalls)

- **비마르코프 상태 (Non-Markovian state).** 결정을 내리기 위해 마지막 3개의 관측치가 필요하다면, "상태"는 단순히 현재의 관측치만이 아닙니다. 해결책: 프레임을 쌓거나(Atari DQN에서 4개의 스택 사용), 순환 상태(관측치에 대한 LSTM/GRU 사용)를 활용해 보세요.
- **희소 보상 (Sparse rewards).** 승리 시에만 보상을 주는 방식은 거대한 상태 공간에서 학습을 거의 불가능하게 만듭니다. 보상 형성(Reward shaping, 중간 신호)을 적용하거나 모방 학습으로 부트스트랩(Phase 9 · 09)해 보세요.
- **보상 해킹 (Reward hacking).** 대리 보상(Proxy reward)을 최적화하면 종종 병리적인 행동이 발생합니다. OpenAI의 보트 레이싱 에이전트는 경주를 마치는 대신 파워업 아이템을 모으기 위해 영원히 제자리에서 회전했습니다. 항상 대리 보상이 아닌 최종 목표 결과로부터 보상을 정의하세요.
- **할인율 설정 오류 (Discount mis-spec).** 무한 시계열(Infinite-horizon) 작업에서 `γ = 1`로 설정하면 모든 가치가 무한대가 됩니다. 항상 유한한 시계열(Finite horizon)을 설정하거나 `γ < 1`로 제한하세요.
- **보상 스케일 (Reward scale).** {+100, -100} 보상과 {+1, -1} 보상은 동일한 최적 정책을 생성하지만, 그래디언트 크기(Gradient magnitudes)는 매우 다릅니다. PPO/DQN에 입력하기 전에 `[-1, 1]` 정도로 정규화해 보세요.

## 사용 방법 (Use It)

2026년형 스택은 코드를 작성하기 전에 모든 RL 파이프라인을 MDP로 축소합니다:

| 상황 (Situation) | 상태 (State) | 행동 (Action) | 보상 (Reward) | γ |
|-----------|-------|--------|--------|---|
| 제어 (Control) (보행, 조작) | 관절 각도 + 속도 | 연속적인 토크 | 작업 특화형 보상 설계 (shaped) | 0.99 |
| 게임 (Games) (체스, 바둑, 포커) | 보드 + 이력 | 허용된 수 (Legal move) | 승리=+1 / 패배=-1 | 1.0 (유한) |
| 재고 / 가격 책정 (Inventory / pricing) | 재고 + 수요 | 주문 수량 | 수익 - 비용 | 0.95 |
| LLM을 위한 RLHF | 컨텍스트 토큰 | 다음 토큰 | 마지막 시점의 보상 모델 점수 | 1.0 (에피소드 ~200 토큰) |
| 추론을 위한 GRPO | 프롬프트 + 부분 응답 | 다음 토큰 | 마지막 시점의 검증기(Verifier) 0/1 | 1.0 |

훈련 루프를 작성하기 전에 이 다섯 가지 튜플을 먼저 작성해 보세요. "RL이 작동하지 않는다"라는 대부분의 버그 보고서는 종이 위에서부터 잘못된 MDP 설계로 인해 발생합니다.

## Ship It

`outputs/skill-mdp-modeler.md`로 저장하세요:

```markdown
---
name: mdp-modeler
description: 작업 설명이 주어지면 마르코프 결정 과정(MDP) 사양을 생성하고, 학습 전 공식화 위험 요소를 식별합니다.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

주어진 작업(제어 / 게임 / 추천 / LLM 미세 조정)에 대해 다음을 출력하세요:

1. 상태(State). 정확한 피처 벡터 또는 텐서 사양. 마르코프 성질(Markov property)에 대한 근거 제시.
2. 행동(Action). 이산 집합 또는 연속 범위. 차원(Dimensionality).
3. 전이(Transition). 결정론적(Deterministic), 알려진 모델을 가진 확률적(Stochastic-with-known-model), 또는 샘플 기반(Sample-only).
4. 보상(Reward). 함수 및 출처. 희소 보상(Sparse) 대 형성된 보상(Shaped). 종료 보상(Terminal) 대 단계별 보상(Per-step).
5. 할인율(Discount). 값 및 유효 지평(Horizon) 설정 근거.

프레임 스태킹(Frame-stacking) 또는 순환 상태(Recurrent state)에 대한 명시적 언급 없이 상태가 비마르코프(Non-Markovian)인 MDP는 승인을 거부하세요. 목표 결과(Target outcome) 관점에서 정의되지 않은 보상은 승인을 거부하세요. 무한 지평(Infinite-horizon) 작업에서 `γ ≥ 1.0`인 경우 경고를 표시하세요. 일반적인 단계별 보상보다 100배 이상 큰 보상 범위는 그래디언트 폭주(Gradient-explosion)의 잠재적 원인으로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 4×4 GridWorld와 무작위 정책 롤아웃(random-policy rollout)을 구현해 보세요. 10,000 에피소드를 실행합니다. 리턴(return)의 평균과 표준편차를 보고하세요. 최적 리턴(-6)과 비교해 보세요.
2. **중간 (Medium).** 균등 무작위 정책(uniform-random policy)에 대해 `γ ∈ {0.5, 0.9, 0.99}` 값으로 `policy_evaluation`을 실행해 보세요. 각 경우에 대해 `V`를 4×4 그리드 형태로 출력하세요. 왜 종료 상태(terminal state) 근처의 상태 가치(state values)가 더 큰 `γ` 값에서 더 빠르게 증가하는지 설명해 보세요.
3. **어려움 (Hard).** GridWorld를 확률적(stochastic)으로 변형해 보세요: 각 행동은 `p = 0.1`의 확률로 인접한 방향으로 미끄러집니다. 균등 정책을 다시 평가해 보세요. `V[start]` 값은 좋아지나요, 아니면 나빠지나요? 그 이유는 무엇인가요?

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| MDP | "강화 학습 설정" | 마르코프 성질(Markov property)을 만족하는 튜플 `(S, A, P, R, γ)`. |
| 상태 (State) | "에이전트가 보는 것" | 선택된 정책 클래스 하에서 미래 역학(future dynamics)을 설명하기 위한 충분 통계량(sufficient statistic). |
| 정책 (Policy) | "에이전트의 행동 방식" | 조건부 분포 `π(a \| s)` 또는 결정론적 매핑 `s → a`. |
| 리턴 (Return) | "총 보상" | 현재 단계부터의 할인된 합계 `Σ γ^t r_t`. |
| 가치 (Value) | "상태가 얼마나 좋은지" | `s`에서 시작하여 `π`를 따랐을 때의 기대 리턴. |
| Q-값 (Q-value) | "행동이 얼마나 좋은지" | `s`에서 시작하여 첫 번째 행동으로 `a`를 취하고 `π`를 따랐을 때의 기대 리턴. |
| 벨만 방정식 (Bellman equation) | "동적 계획법 재귀" | 가치/Q-값을 '한 단계의 보상 + 할인된 다음 상태의 가치'로 분해하는 고정점 분해(fixed-point decomposition). |
| 할인율 (Discount `γ`) | "미래 대 현재" | 먼 미래 보상에 대한 기하학적 가중치; 유효 지평(effective horizon)은 `~1/(1-γ)`. |

## 추가 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 강화학습의 교과서입니다. 3장은 MDP와 벨만 방정식(Bellman equations)을 다루며, 1장은 이후 모든 학습의 기초가 되는 보상 가설(reward hypothesis)에 대한 동기를 부여합니다.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 벨만 방정식(Bellman equation)의 기원이 되는 문헌입니다.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 심층 강화학습(deep-RL) 관점에서 작성된 간결한 MDP 입문서입니다.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP와 정확한 해법(exact solution methods)에 관한 운영 과학(operations-research) 분야의 참고 문헌입니다.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — MDP를 동적 계획법(dynamic-programming)의 특수 사례로 가장 깔끔하게 유도해낸 박사 학위 논문입니다.
