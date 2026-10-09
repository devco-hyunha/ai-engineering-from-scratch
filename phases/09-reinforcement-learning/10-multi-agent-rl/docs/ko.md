# 멀티 에이전트 RL

> 단일 에이전트 RL은 환경이 정적이라고 가정합니다. 같은 세계에 두 개의 학습 에이전트를 배치하면 이 가정이 깨집니다. 각 에이전트는 서로의 환경의 일부이며, 둘 다 변화하고 있습니다. 멀티 에이전트 RL은 마르코프 가정이 더 이상 성립하지 않을 때 학습이 수렴하도록 하는 기법들의 집합입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 04강 (Q-learning), 9단계 · 06강 (REINFORCE), 9단계 · 07강 (Actor-Critic)
**시간:** 약 45분

## 문제점

방을 탐색하는 법을 배우는 로봇은 단일 에이전트 RL 문제입니다. 축구 팀은 그렇지 않습니다. AlphaStar 대 StarCraft 상대는 그렇지 않습니다. 입찰 에이전트들의 시장은 그렇지 않습니다. 네 방향 정지 표지판에서 두 자동차가 협상하는 것은 그렇지 않습니다. 다대다의 실제 세계 문제는 그렇지 않습니다.

모든 멀티 에이전트 환경에서, 임의의 한 에이전트의 관점에서 다른 에이전트들은 환경의 일부*입니다*. 그들이 학습하고 행동을 변화함에 따라 환경은 비정적(non-stationary)이 됩니다. 마르코프 성질 — "다음 상태는 현재 상태와 내 행동에만 의존한다" —는 위반됩니다. 왜냐하면 다음 상태는 *다른* 에이전트들이 선택한 것에도 의존하며, 그들의 정책은 계속 변하는 목표이기 때문입니다.

이는 표형 수렴 증명(Q-learning의 보장은 정적 환경을 가정합니다)을 깨뜨립니다. 또한 단순한 딥 RL도 깨뜨립니다. 에이전트들이 서로를 따라다니며 루프에 빠지고, 안정적인 정책으로 수렴하지 못합니다. 멀티 에이전트 전용 기법이 필요합니다. 중앙 집중식 학습 / 분산 실행, 반사실적 기준선(counterfactual baselines), 리그 플레이, 자기 대결(self-play) 등입니다.

2026년 응용 분야: 로봇 스웜, 교통 라우팅, 자율 차량 플릿, 시장 시뮬레이터, 멀티 에이전트 LLM 시스템(16단계), 그리고 한 명 이상의 지능적인 플레이어가 있는 모든 게임.

## 개념

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**형식화: 마르코프 게임.** MDP의 일반화입니다. 상태 `S`, 결합 행동 `a = (a_1, …, a_n)`, 전이 `P(s' | s, a)`, 그리고 에이전트별 보상 `R_i(s, a, s')`이 있습니다. 각 에이전트 `i`는 자신의 정책 `π_i` 하에서 자신의 리턴을 최대화합니다. 보상이 동일하면 **완전 협력(fully cooperative)**입니다. 합이 0이면 **적대(adversarial)**입니다. 혼합되어 있으면 **일반 합(general-sum)**입니다.

**핵심 과제:**

- **비정상성.** 에이전트 `i`의 관점에서 `P(s' | s, a_i)`는 변화하는 `π_{-i}`에 의존합니다.
- **신용 할당.** 공유된 보상에서, 어떤 에이전트가 그 원인을 발생시켰습니까?
- **탐색 조정.** 에이전트들은 동일한 상태를 중복적으로 탐색하는 것이 아니라, 상호 보완적인 전략을 탐색해야 합니다.
- **확장성.** 결합된 행동 공간은 `n`에 대해 지수적으로 증가합니다.
- **부분 관측 가능성.** 각 에이전트는 자신의 관측만 볼 수 있으며, 전역 상태는 숨겨져 있습니다.

**네 가지 지배적 영역:**

**1. 독립 Q-학습 / 독립 PPO (IQL, IPPO).** 각 에이전트는 다른 에이전트를 환경의 일부로 간주하며 자신의 Q 또는 정책을 학습합니다. 단순하며, 때로는 효과가 있습니다 (특히 경험 리플레이가 에이전트 모델링 기법으로 작동하는 경우). 이론적 수렴: 없음. 실제 사용: 느슨하게 결합된 작업에는 적합하지만, 밀접하게 결합된 작업에는 적합하지 않습니다.

**2. 중앙화 학습, 분산 실행 (CTDE).** 가장 일반적인 현대적 패러다임입니다. 각 에이전트는 로컬 관측 `o_i`에 조건을 거는 자체 *정책* `π_i`을 가지며, 배포 시 표준 분산 실행을 수행합니다. *학습* 중에는 중앙화 크리틱 `Q(s, a_1, …, a_n)`이 전체 전역 상태와 결합된 행동에 조건을 겁니다. 예시:
- **MADDPG** (Lowe et al. 2017): 각 에이전트마다 중앙화 크리틱을 사용하는 DDPG.
- **COMA** (Foerster et al. 2017): 반사실적 기준선 — "내가 행동 `a'`을 취했다면 내 보상이 어떻게 되었을까?"라고 물어보며 — 나의 기여를 분리합니다.
- **MAPPO** / 공유 크리틱을 사용하는 **IPPO** (Yu et al. 2022): 중앙화 가치 함수를 사용하는 PPO. 2026년 협력 MARL에서 지배적입니다.
- **QMIX** (Rashid et al. 2018): 가치 분해 — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`를 단조적 혼합으로 처리합니다.

**3. 자가 대결(Self-play).** 동일한 에이전트의 두 복제본이 서로 대결합니다. 상대의 정책은 과거 스냅샷에서의 내 정책입니다. AlphaGo / AlphaZero / MuZero. OpenAI Five. 합이 0인 게임에 가장 잘 작동하며, 학습 신호는 대칭적입니다.

**4. 리그 플레이.** 자기 대결(self-play)을 일반 합/적대적 환경으로 확장한 것입니다. 과거 및 현재 정책의 개체군을 유지하고, 리그에서 상대를 샘플링하여 이를 상대로 학습합니다. 익스플로이터(현재 최강자를 이기는 데 특화)와 메인 익스플로이터(익스플로이터를 이기는 데 특화)를 추가합니다. AlphaStar (StarCraft II). 게임이 "가위바위보" 전략 순환을 허용할 때 필요합니다.

**커뮤니케이션.** 에이전트가 학습된 메시지를 서로에게 전송하도록 허용합니다. 협력적 환경에서 작동합니다. Foerster et al. (2016)는 미분 가능한 에이전트 간 통신을 엔드투엔드(end-to-end)로 학습할 수 있음을 보여 주었습니다. 오늘날의 LLM 기반 다중 에이전트 시스템(16단계)은 본질적으로 자연어로 소통합니다. `m_i`

```figure
f3-marl-orbit
```

## 구현하기

이 강의는 두 협력 에이전트를 사용하는 6×6 GridWorld를 사용합니다. 에이전트는 반대 코너에서 시작하여 공유된 목표에 도달해야 합니다. 공유 보상: `-1`는 에이전트 중 하나가 아직 이동 중일 때 단계별로, `+10`는 두 에이전트가 모두 도착했을 때 지급됩니다. `code/main.py`를 참조하세요.

### 1단계: 다중 에이전트 환경

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 두 에이전트

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*공동* 행동 공간은 `|A|² = 16`입니다. 전역 상태는 두 위치로 구성됩니다.

### 2단계: 독립 Q-학습

각 에이전트는 공동 상태를 키로 사용하는 자체 Q-테이블을 실행합니다. 각 단계에서: 두 에이전트 모두 ε-그리디 행동을 선택하고, 공동 전이를 수집하며, 각 에이전트는 공유된 보상으로 자신의 Q를 업데이트합니다.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

보상이 밀집하고 정렬되어 있으므로 이 작업에서 작동합니다. tightly-coupled 작업(예: 한 에이전트가 다른 에이전트를 *기다려야* 하는 경우)에서는 실패합니다.

### 3단계: 분해된 가치 업데이트를 사용하는 중앙 집중식 Q

공동 행동 `Q(s, a_1, a_2)`에 대해 하나의 Q를 사용합니다. 공유된 보상으로 업데이트합니다. 실행 시 주변화(marginalizing)를 통해 분산화합니다: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. 지수적으로 커지는 공동 행동 공간을 *올바른* 전역 관점과 교환합니다.

### 4단계: 단순 자기 대결 (적대적 2-에이전트)

동일한 에이전트, 두 역할. 에이전트 A를 에이전트 B를 상대로 학습합니다. `K` 에포크 후, A의 가중치를 B에 복사합니다. 대칭적인 학습, 일관된 진행. AlphaZero 레시피의 축소판입니다.

## 함정

- **비정상적 리플레이.** 독립적인 에이전트를 사용하는 경험 리플레이는 단일 에이전트보다 더 나쁜데, 이는 오래된 전이가 현재는 폐기된 상대에 의해 생성되었기 때문입니다. 해결책: 최신성에 따라 재라벨링하거나 가중치를 부여하세요.
- **신용 할당 모호성.** 긴 에피소드 후 공유된 보상; 어떤 에이전트(Agent)가 기여했는지 명확히 말할 방법이 없습니다. 해결책: 반사실적 기준선(COMA) 또는 에이전트별 보상 형성(reward shaping).
- **정책 드리프트 / 추격.** 각 에이전트(Agent)의 최적 응답이 서로의 업데이트에 따라 변합니다. 해결책: 중앙화 크리틱, 느린 학습률, 또는 하나씩 동결.
- **조정을 통한 보상 해킹.** 에이전트(Agent)들이 설계자가 예상하지 못한 조정된 악용을 발견합니다. 경매 에이전트들이 입찰을 0으로 수렴합니다. 해결책: 신중한 보상 설계, 행동 제약.
- **탐색 중복.** 두 에이전트(Agent)가 동일한 상태-행동 쌍을 탐색합니다. 해결책: 에이전트별 엔트로피 보너스, 또는 역할 조건부.
- **리그 사이클.** 순수 자가 플레이는 지배 사이클에 갇힐 수 있습니다. 해결책: 다양한 상대를 포함한 리그 플레이.
- **샘플 폭발.** `n` 에이전트(Agent) × 상태 공간 × 결합 행동. 함수 근사로 근사화; 분할된 행동 공간(에이전트별 정책 출력 헤드).

## 사용하기

2026년 MARL 적용 지도:

| 영역 | 방법 | 비고 |
|--------|--------|-------|
| 협력적 내비게이션 / 조작 | MAPPO / QMIX | CTDE; 공유 크리틱 + 분산화된 액터. |
| 2인 게임 (체스, 바둑, 포커) | MCTS를 활용한 자가 플레이 (AlphaZero) | 영합 게임; 대칭적 훈련. |
| 복잡한 다인 게임 (Dota, StarCraft) | 리그 플레이 + 모방 사전 훈련 | OpenAI Five, AlphaStar. |
| 자율주행 차량 플릿 | CTDE MAPPO / PPO + 어텐션 | 부분 관측; 가변 팀 크기. |
| 경매 시장 | 게임 이론적 균형 + RL | `n` → ∞일 때 평균장 RL. |
| LLM 다중 에이전트 시스템 (16단계) | 자연어 통신 + 역할 조건부 | 에이전트 계획 층에서의 RL 루프. |

2026년, MARL의 가장 큰 성장 영역은 LLM 기반입니다: 언어 모델 에이전트(Agent)들이 협상, 토론, 소프트웨어 구축을 수행하는 스웜(Swarm). RL은 토큰 수준이 아닌 *궤적 수준* 출력에 대한 선호 최적화로 나타납니다 (16단계 · 03강).

## 출시하기

`outputs/skill-marl-architect.md`로 저장:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 연습 문제

1. **쉬움.** 2 에이전트 협력 GridWorld에서 독립 Q-learning을 학습해 보세요. 평균 보상이 0을 초과할 때까지 몇 에포크가 걸리나요? 결합 학습 곡선을 그려 보세요.
2. **중간.** "조율(coordination)" 작업을 추가해 보세요: 두 에이전트가 같은 턴에 목표 지점에 도달해야만 목표가 달성됩니다. 독립 Q-learning이 여전히 수렴하나요? 무엇이 깨지나요?
3. **어려움.** MAPPO 스타일 학습을 위해 중앙화 크리틱(centralized critic)을 구현하고, 조율 작업에서 독립 PPO와 수렴 속도를 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Markov game | "멀티 에이전트 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; 각 에이전트는 자체 보상을 가집니다. |
| CTDE | "중앙화 학습, 분산 실행" | 학습 시점에 결합 크리틱을 사용하며, 각 에이전트의 정책은 로컬 관측만 사용합니다. |
| IPPO | "독립 PPO" | 각 에이전트가 PPO를 개별적으로 실행합니다. 단순한 베이스라인이며, 종종 과소평가됩니다. |
| MAPPO | "멀티 에이전트 PPO" | 전역 상태에 조건을 둔 중앙화 가치 함수를 사용하는 PPO입니다. |
| QMIX | "단조 가치 분해" | `Q_tot = f_monotone(Q_1, …, Q_n)`는 분산 argmax를 허용합니다. |
| COMA | "반사실 멀티 에이전트" | Advantage는 내 Q에서 내 행동을 주변화(marginalizing)한 기대 Q를 뺀 값입니다. |
| Self-play | "에이전트 vs 과거의 나" | 단일 에이전트, 두 역할; 제로섬 게임의 표준입니다. |
| League play | "인구 훈련" | 과거 정책을 캐시하고 풀(pool)에서 상대를 샘플링합니다; 전략 순환을 처리합니다. |

## 추가 읽기

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 중앙화 크리틱을 사용하는 CTDE입니다.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 크레딧 할당(credit assignment)을 위한 반사실적 베이스라인입니다.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 단조성을 가진 가치 분해입니다.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO는 MARL에서 놀라울 정도로 강력합니다.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 대규모 league play입니다.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 제로섬 게임에서의 순수 self-play입니다.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서의 멀티 에이전트 설정에 대한 짧은 설명과 CTDE가 해결하도록 설계된 비정상성(non-stationarity) 문제를 포함합니다.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 협력, 경쟁, 혼합 MARL을 수렴 결과와 함께 다루는 서베이입니다.
