# MDP, 상태, 행동 및 보상

> 마르코프 결정 과정(Markov Decision Process)은 다섯 가지 요소로 구성됩니다: 상태, 행동, 전이, 보상, 할인율. RL의 모든 것 — Q-learning, PPO, DPO, GRPO —은 이 구조에 대해 최적화합니다. 이 구조를 한 번만 학습하면 나머지 강화 학습 내용도 쉽게 이해할 수 있습니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 1단계 · 06강 (확률 및 분포), 2단계 · 01강 (ML 분류 체계)
**시간:** 약 45분

## 문제점

체스 봇을 작성하고 있습니다. 또는 재고 계획자, 거래 에이전트, 추론 모델을 학습하는 PPO 루프를 작성하고 있을 수도 있습니다. 네 가지 서로 다른 영역이지만, 놀라운 사실은 이 네 가지가 모두 동일한 수학적 객체로 축소된다는 점입니다.

지도 학습은 `(x, y)` 쌍을 제공하며 함수를 적합(fit)하도록 요청합니다. 강화 학습은 레이블을 제공하지 않습니다. 상태의 스트림, 취한 행동, 스칼라 보상만 주어집니다. 그 수가 게임에서 승리했나요? 재고 결정이 비용을 절감했나요? 거래가 이익을 냈나요? LLM이 방금 생성한 토큰이 판독기로부터 더 높은 보상을 이끌어냈나요?

이 스트림을 형식화하기 전까지는 학습할 수 없습니다. "내가 본 것", "내가 한 것", "다음에 일어난 것", "그것이 얼마나 좋았는지"는 각각 추론할 수 있는 객체가 되어야 합니다. 이 형식화가 마르코프 결정 과정입니다. 이 단계의 모든 RL 알고리즘, 마지막에 있는 RLHF 및 GRPO 루프를 포함하여, 이 구조에 대해 최적화합니다.

## 개념

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**다섯 가지 객체.**

- **상태** `S`. 에이전트가 결정하는 데 필요한 모든 것. GridWorld에서는 셀, 체스에서는 보드, LLM에서는 컨텍스트 윈도우 및 메모리입니다.
- **행동** `A`. 선택 사항. 위/아래/왼쪽/오른쪽으로 이동. 수를 두기. 토큰을 생성하기.
- **전이** `P(s' | s, a)`. 상태 `s`과 행동 `a`가 주어졌을 때, 다음 상태에 대한 분포. 체스에서는 결정론적, 재고에서는 확률적, LLM 디코딩에서는 거의 결정론적입니다.
- **보상** `R(s, a, s')`. 스칼라 신호. 승리 = +1, 패배 = -1. 수익에서 비용을 뺀 값. GRPO의 로그 우도비(log-likelihood ratio) 항.
- **할인율** `γ ∈ [0, 1)`. 미래의 보상이 현재와 비교하여 얼마나 중요한지. `γ = 0.99`은 약 100단계의 기간을, `γ = 0.9`는 약 10단계의 기간을 설정합니다.

**마르코프 성질** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. 미래는 오직 현재 상태에만 의존합니다. 만약 그렇지 않다면, 상태 표현이 불완전한 것입니다 — 방법의 실패가 아니라, 상태의 실패입니다.

**정책과 리턴.** 정책 `π(a | s)`은 상태를 행동 분포에 매핑합니다. 리턴 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`은 미래 보상의 할인된 합입니다. 가치 `V^π(s) = E[G_t | s_t = s]`는 정책 `π` 하에서 `s`에서 시작하는 기대 리턴입니다. Q-값 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`은 특정 행동으로 시작하는 기대 리턴입니다. 모든 RL 알고리즘은 이 두 가지 중 하나를 추정한 후, `π`을 이에 따라 개선합니다.

**벨만 방정식.** 이 단계의 모든 내용이 사용하는 고정점 방정식입니다:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

이 방정식은 기대 리턴을 "이번 단계의 보상"과 "도착한 위치의 할인된 가치"로 분리합니다. 재귀적입니다. 9단계의 모든 알고리즘은 이 방정식을 수렴할 때까지 반복(dynamic programming), 샘플링(Monte Carlo), 또는 한 단계 부트스트랩(temporal difference)합니다.

```figure
discount-horizon
```

## 구현하기

### 1단계: 작은 결정론적 MDP

4×4 GridWorld입니다. 에이전트는 좌상단에서 시작하고, 우하단에 종단 상태가 있으며, 각 단계당 보상은 -1이고, 행동은 `{up, down, left, right}`입니다. `code/main.py`을 참고하세요.

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

5줄입니다. 이것이 환경 전체입니다. 결정론적 전이, 일정한 단계 페널티, 흡수 종단 상태입니다.

### 2단계: 정책 롤아웃

정책은 상태에서 행동 분포로 가는 함수입니다. 가장 단순한 것은 균일한 랜덤입니다.

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

랜덤 정책을 1000번 실행하세요. 이 4×4 보드에서 평균 리턴은 약 -60에서 -80 사이입니다. 최적 리턴은 -6 (우하단으로 직선 경로)입니다. 이 격차를 줄이는 것이 9단계의 모든 내용입니다.

### 3단계: 벨만 방정식으로 `V^π`을 정확히 계산

작은 MDP에서는 벨만 방정식이 선형 시스템입니다. 상태를 열거하고, 기대값을 적용하며, 값이 더 이상 변하지 않을 때까지 반복하세요.

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

이것은 반복적 정책 평가입니다. Sutton & Barto의 첫 번째 알고리즘이며, 이후 모든 RL 방법의 이론적 기초입니다.

### 4단계: `γ`은 물리적 의미를 가진 하이퍼파라미터입니다

효과적인 기간은 대략 `1 / (1 - γ)`입니다. `γ = 0.9` → 10단계. `γ = 0.99` → 100단계. `γ = 0.999` → 1000단계.

너무 낮으면 에이전트(Agent)가 근시안적으로 행동합니다. 너무 높으면 크레딧 할당(credit assignment)이 노이즈가 많아집니다. 많은 초기 단계가 먼 미래의 보상에 대한 책임을 공유하기 때문입니다. LLM RLHF는 에피소드가 짧고 유한하므로 일반적으로 `γ = 1`을 사용합니다. 제어 작업은 `0.95–0.99`을 사용합니다. 장기 기간 전략 게임은 `0.999`을 사용합니다.

## 함정

- **비마르코프 상태.** 마지막 세 관측값을 결정에 사용해야 한다면, "상태"는 현재 관측값만 포함하지 않습니다. 해결책: 프레임을 스택(stack)으로 쌓기(DQN은 Atari에서 4개를 스택) 또는 순환 상태(recurrent state) 사용(관측값에 대해 LSTM/GRU 사용).
- **희소한 보상.** 승리 전용 보상은 큰 상태 공간에서 학습을 거의 불가능하게 만듭니다. 보상을 형성(shape rewards)(중간 신호)하거나 모방으로 부트스트랩(bootstrap)(9단계 · 09강)하세요.
- **보상 해킹.** 대리(proxy) 보상을 최적화하면 병적인 행동이 생성되는 경우가 많습니다. OpenAI의 보트 경주 에이전트(Agent)는 경주를 완주하지 않고, 파워업을 영원히 수집하며 원을 돌았습니다. 항상 대리(proxy)가 아닌 목표 결과로부터 보상을 정의하세요.
- **할인율 설정 오류.** `γ = 1`은 무한 기간 작업에서 모든 값을 무한하게 만듭니다. 항상 유한 기간 또는 `γ < 1`로 상한을 설정하세요.
- **보상 스케일.** {+100, -100}과 {+1, -1}의 보상은 동일한 최적 정책을 산출하지만, 기울기 크기는 크게 다릅니다. PPO/DQN에 연결하기 전에 `[-1, 1]` 근처로 정규화(normalization)하세요.

## 사용하기

2026년 스택은 코드를 작성하기 전에 모든 RL 파이프라인을 MDP로 줄입니다:

| 상황 | 상태 | 행동 | 보상 | γ |
|-----------|-------|--------|--------|---|
| 제어 (이동, 조작) | 관절 각도 + 속도 | 연속 토크 | 작업별 형성된 | 0.99 |
| 게임 (체스, 바둑, 포커) | 보드 + 기록 | 합법적 이동 | 승리=+1 / 패배=-1 | 1.0 (유한) |
| 재고 / 가격 | 재고 + 수요 | 주문 수량 | 수익 - 비용 | 0.95 |
| LLM용 RLHF | 컨텍스트 토큰(Token) | 다음 토큰(Token) | 보상 모델 점수 (종료 시) | 1.0 (에피소드 ~200 토큰(Token)) |
| 추론용 GRPO | 프롬프트 + 부분 응답 | 다음 토큰(Token) | 검증자 0/1 (종료 시) | 1.0 |

훈련 루프를 작성하기 전에 5개 튜플을 먼저 작성하세요. "RL가 작동하지 않는다"는 버그 보고의 대부분은 종이 위에서 MDP 공식화가 깨진 데서 기인합니다.

## 출시하기

`outputs/skill-mdp-modeler.md`로 저장하세요:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 4×4 GridWorld와 랜덤 정책 롤아웃을 구현하세요. 10,000 에피소드를 실행하고 리턴의 평균과 표준 편차를 보고하세요. 최적 리턴(-6)과 비교하세요.
2. **중간.** `policy_evaluation`에서 `γ ∈ {0.5, 0.9, 0.99}`을 사용하여 균일 랜덤 정책을 실행하세요. 각각에 대해 `V`를 4×4 그리드로 출력하세요. `γ`이 클수록 종단 상태 근처의 상태 값이 더 빠르게 증가하는 이유를 설명하세요.
3. **어려움.** GridWorld를 확률적으로 만드세요: 각 행동이 확률 `p = 0.1`으로 인접한 방향으로 미끄러집니다. 균일 정책을 재평가하세요. `V[start]`이 더 좋아지나요, 나빠지나요? 왜 그런가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| MDP | "강화 학습 설정" | 마르코프 성질을 만족하는 튜플 `(S, A, P, R, γ)`. |
| 상태 | "에이전트가 보는 것" | 선택된 정책 클래스 하에서 미래 동역학에 대한 충분 통계량. |
| 정책 | "에이전트의 행동" | 조건부 분포 `π(a \| s)` 또는 결정적 맵 `s → a`. |
| 리턴 | "총 보상" | 현재 단계부터의 할인된 합 `Σ γ^t r_t`. |
| 가치 | "상태가 얼마나 좋은지" | `s`에서 시작하여 `π` 하에서의 기대 리턴. |
| Q-값 | "행동이 얼마나 좋은지" | `s`에서 시작하여 첫 행동이 `a`인 `π` 하에서의 기대 리턴. |
| 벨만 방정식 | "동적 프로그래밍 재귀" | 가치 / Q를 1단계 보상과 할인된 후속 가치의 고정점 분해로 표현한 것. |
| 할인 `γ` | "미래 vs 현재" | 먼 미래의 보상에 대한 기하학적 가중치; 유효 기간 `~1/(1-γ)`. |

## 추가 읽기

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서. Ch. 3은 MDP와 벨만 방정식을 다루며, Ch. 1은 모든 후속 강의의 기반이 되는 보상 가설을 동기 부여합니다.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 벨만 방정식의 기원.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 심층 RL 관점의 간결한 MDP 입문서.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP와 정확한 해법에 대한 오퍼레이션 리서치 참고 자료입니다.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) — MDP를 동적 프로그래밍의 특수화 형태로 가장 깔끔하게 유도한 자료입니다.
