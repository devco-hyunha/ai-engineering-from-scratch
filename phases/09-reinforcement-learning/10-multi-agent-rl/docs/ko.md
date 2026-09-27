# 멀티 에이전트 강화학습 (Multi-Agent RL)

> 단일 에이전트 강화학습(Single-agent RL)은 환경이 정적(stationary)이라고 가정합니다. 하지만 동일한 세계에 두 명의 학습 에이전트를 배치하면 이 가정은 깨집니다. 각 에이전트가 서로의 환경 일부가 되며, 두 에이전트 모두 계속해서 변화하기 때문입니다. 멀티 에이전트 강화학습(Multi-agent RL)은 마르코프 가정(Markov assumption)이 더 이상 성립하지 않을 때 학습이 수렴하도록 만드는 일련의 기법들을 의미합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## 문제점 (The Problem)

방 안을 탐색하는 법을 배우는 로봇은 단일 에이전트 강화학습(single-agent RL) 문제입니다. 하지만 축구 팀은 그렇지 않습니다. AlphaStar와 스타크래프트 상대방의 대결도 그렇지 않습니다. 입찰 에이전트들이 모인 시장도 그렇지 않습니다. 사거리에서 협상하는 두 대의 자동차도 그렇지 않습니다. 다수 대 다수의 실제 세계 문제들은 모두 이에 해당하지 않습니다.

모든 멀티 에이전트 설정(multi-agent setting)에서, 개별 에이전트의 관점에서는 다른 에이전트들이 환경의 일부가 됩니다. 다른 에이전트들이 학습하고 행동을 변화시킴에 따라, 환경은 비정상성(non-stationarity)을 띠게 됩니다. "다음 상태는 현재 상태와 나의 행동에만 의존한다"라는 마르코프 성질(Markov property)이 위배되는데, 이는 다음 상태가 *다른* 에이전트들이 선택한 것에 의해서도 결정되며, 그들의 정책(policy)이 계속해서 변하는 표적(moving targets)이기 때문입니다.

이로 인해 테이블 방식의 수렴 증명(tabular convergence proofs)이 무너집니다(Q-learning의 보장은 정적인 환경을 가정합니다). 단순한 심층 강화학습(deep RL) 또한 실패합니다. 에이전트들이 서로를 쫓으며 루프를 돌 뿐, 안정적인 정책으로 수렴하지 못하기 때문입니다. 따라서 다음과 같은 멀티 에이전트 특화 기술이 필요합니다: 중앙 집중식 학습/분산 실행(centralized training / decentralized execution), 인과적 베이스라인(counterfactual baselines), 리그 플레이(league play), 셀프 플레이(self-play).

2026년의 응용 분야: 로봇 군집(robot swarms), 교통 경로 최적화(traffic routing), 자율주행 차량 플릿(autonomous vehicle fleets), 시장 시뮬레이터, 멀티 에이전트 LLM 시스템(Phase 16), 그리고 둘 이상의 지능형 플레이어가 존재하는 모든 게임.

## 개념 (The Concept)

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**형식화: 마르코프 게임 (Markov Game).** MDP의 일반화된 형태입니다: 상태 `S`, 결합 행동(joint action) `a = (a_1, …, a_n)`, 전이 확률 `P(s' | s, a)`, 그리고 에이전트별 보상 `R_i(s, a, s')`로 구성됩니다. 각 에이전트 `i`는 자신의 정책 `π_i` 하에서 자신의 리턴(return)을 최대화합니다. 보상이 모두 동일하다면 이는 **완전 협력(fully cooperative)** 모델입니다. 제로섬(zero-sum) 게임이라면 **적대적(adversarial)** 모델이며, 혼합된 형태라면 **일반 합 게임(general-sum)** 모델입니다.

**핵심 과제 (Core challenges):**

- **비정상성 (Non-stationarity).** 에이전트 `i`의 관점에서 본 `P(s' | s, a_i)`는 계속 변화하는 다른 에이전트들의 정책 `π_{-i}`에 의존합니다.
- **신용 할당 (Credit assignment).** 보상이 공유될 때, 어떤 에이전트가 그 보상을 유도했는지 판단해야 합니다.
- **탐색 협력 (Exploration coordination).** 에이전트들은 동일한 상태를 중복해서 탐색하는 것이 아니라, 서로 보완적인 전략을 탐색해야 합니다.
- **확장성 (Scalability).** 결합 행동 공간(joint action space)은 에이전트 수 `n`에 따라 기하급수적으로 증가합니다.
- **부분 관측 가능성 (Partial observability).** 각 에이전트는 자신의 관측값(observation)만 볼 수 있으며, 전체 글로벌 상태(global state)는 숨겨져 있습니다.

**4가지 주요 체계 (Four dominant regimes):**

**1. 독립적 Q-러닝 / 독립적 PPO (Independent Q-learning / independent PPO, IQL, IPPO).** 각 에이전트는 다른 에이전트들을 환경의 일부로 취급하며 자신만의 Q 함수나 정책을 학습합니다. 단순하며, 때로는 효과적입니다(특히 경험 재현(experience replay)이 에이전트 모델링을 매끄럽게 만드는 기법으로 작용할 때). 이론적 수렴성은 보장되지 않습니다. 실제로는 느슨하게 결합된(loosely-coupled) 작업에는 괜찮지만, 긴밀하게 결합된(tightly-coupled) 작업에는 부적합합니다.

**2. 중앙 집중식 학습, 분산 실행 (Centralized training, decentralized execution, CTDE).** 가장 일반적인 현대적 패러다임입니다. 각 에이전트는 로컬 관측값 `o_i`를 조건으로 하는 자신만의 *정책* `π_i`를 가집니다. 이는 배포 시 표준적인 분산 실행 방식입니다. *학습* 단계에서는 중앙 집중식 크리틱(centralized critic) `Q(s, a_1, …, a_n)`이 전체 글로벌 상태와 결합 행동을 조건으로 합니다. 예시:
- **MADDPG** (Lowe et al. 2017): 에이전트당 중앙 집중식 크리틱을 갖는 DDPG.
- **COMA** (Foerster et al. 2017): 반사실적 베이스라인(counterfactual baseline) — "만약 내가 행동 `a'`를 취했다면 내 보상이 어땠을까?"라고 질문하여 나의 기여도를 분리합니다.
- **MAPPO / IPPO** 공유 크리틱 방식 (Yu et al. 2022): 중앙 집중식 가치 함수를 사용하는 PPO. 2026년 협력적 MARL 분야의 주류입니다.
- **QMIX** (Rashid et al. 2018): 가치 분해(value decomposition) — 단조적 믹싱(monotonic mixing)을 사용하는 `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`.

**3. 셀프 플레이 (Self-play).** 동일한 에이전트의 두 복사본이 서로 대결합니다. 상대방의 정책은 과거 스냅샷 시점의 나의 정책입니다. AlphaGo / AlphaZero / MuZero, OpenAI Five가 이에 해당합니다. 학습 신호가 대칭적인 제로섬 게임에서 가장 잘 작동합니다.

**4. 리그 플레이 (League play).** 셀프 플레이를 일반 합 / 적대적 환경으로 확장한 것입니다. 과거와 현재의 정책 인구(population)를 유지하고, 리그에서 상대방을 샘플링하여 그들과 학습합니다. 여기에는 현재 최강자를 이기는 데 특화된 익스플로이터(exploiters)와 익스플로이터를 이기는 데 특화된 메인 익스플로이터(main exploiters)가 추가됩니다. AlphaStar (StarCraft II)가 대표적입니다. 게임이 "가위바위보" 식의 전략 순환을 허용할 때 필요합니다.

**통신 (Communication).** 에이전트들이 학습된 메시지 `m_i`를 서로 주고받을 수 있도록 합니다. 협력적 환경에서 효과적입니다. Foerster et al. (2016)은 미분 가능한 에이전트 간 통신을 엔드투엔드(end-to-end)로 학습할 수 있음을 보여주었습니다. 오늘날의 LLM 기반 멀티 에이전트 시스템(Phase 16)은 본질적으로 자연어로 통신합니다.

```figure
f3-marl-orbit
```

## 구현하기 (Build It)

이 레슨에서는 두 명의 협력 에이전트(cooperative agents)가 포함된 6×6 GridWorld를 사용합니다. 에이전트들은 서로 반대편 모서리에서 시작하여 공동의 목표 지점에 도달해야 합니다. 공유 보상(Shared reward) 체계는 다음과 같습니다: 에이전트 중 하나라도 움직이고 있는 동안에는 단계당 `-1`을 받으며, 두 에이전트가 모두 도착하면 `+10`을 받습니다. 자세한 내용은 `code/main.py`를 참조하세요.

### 1단계: 멀티 에이전트 환경 (the multi-agent env)

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 두 명의 에이전트

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*결합(joint)* 행동 공간은 `|A|² = 16`입니다. 전역 상태(global state)는 두 개의 위치로 구성됩니다.

### 2단계: 독립적 Q-러닝 (independent Q-learning)

각 에이전트는 결합 상태(joint state)를 키로 사용하는 자신만의 Q-테이블을 실행합니다. 매 단계마다: 두 에이전트 모두 $\epsilon$-greedy 행동을 선택하고, 결합 전이(joint transition)를 수집하며, 공유된 보상을 바탕으로 각자의 Q-값을 업데이트합니다.

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

이 작업에서는 보상이 조밀(dense)하고 정렬(aligned)되어 있기 때문에 잘 작동합니다. 하지만 긴밀하게 결합된(tightly-coupled) 작업(예: 한 에이전트가 다른 에이전트를 위해 *기다려야* 하는 경우)에서는 실패합니다.

### 3단계: 분해된 가치 업데이트를 이용한 중앙 집중식 Q (centralized Q with decomposed-value update)

결합 행동(joint actions)에 대한 단일 Q 함수인 `Q(s, a_1, a_2)`를 사용합니다. 공유된 보상(shared reward)으로부터 업데이트를 수행합니다. 실행 시에는 주변화(marginalizing)를 통해 분산화합니다: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. 이는 지수적으로 증가하는 결합 행동 공간(joint action space)을 대가로 지불하는 대신, *정확한* 글로벌 관점(global view)을 얻습니다.

### 4단계: 단순 셀프 플레이 (simple self-play, 적대적 2-에이전트)

동일한 에이전트가 두 가지 역할을 수행합니다. 에이전트 A를 에이전트 B에 맞서 학습시키고, `K` 에피소드가 지나면 A의 가중치를 B로 복사합니다. 대칭적 학습(Symmetric training)을 통해 일관된 발전을 도모합니다. 이는 AlphaZero 레시피를 축소해 놓은 형태입니다.

## 함정 (Pitfalls)

- **비정상성 리플레이 (Non-stationary replay).** 독립적인 에이전트들을 사용하는 경험 리플레이(Experience replay)는 단일 에이전트 방식보다 성능이 떨어집니다. 과거의 전이(transitions) 데이터가 현재는 더 이상 존재하지 않는 구식 상대방에 의해 생성되었기 때문입니다. 해결책: 레이블을 다시 지정(relabel)하거나 최신성(recency)에 따라 가중치를 부여하세요.
- **신용 할당 모호성 (Credit assignment ambiguity).** 긴 에피소드 후에 공유된 보상이 주어지면, 어떤 에이전트가 기여했는지 명확히 알 수 있는 방법이 없습니다. 해결책: 반사실적 베이스라인(counterfactual baselines, COMA)을 사용하거나 에이전트별로 보상 형성(reward shaping)을 수행하세요.
- **정책 표류/추격 (Policy drift / chasing).** 각 에이전트의 최적 대응(best response)은 서로의 업데이트에 따라 계속 변합니다. 해결책: 중앙 집중식 크리틱(centralized critic), 느린 학습률(learning rates), 또는 한 번에 하나씩 동결(freeze-one-at-a-time)하는 방식을 사용하세요.
- **협업을 통한 보상 해킹 (Reward hacking via coordination).** 에이전트들이 설계자가 예상하지 못한 협동형 취약점(exploits)을 찾아냅니다. 예를 들어, 경매 에이전트들이 입찰가를 0으로 수렴시키는 경우가 있습니다. 해결책: 세심한 보상 설계 및 행동 제약(behavioral constraints)을 적용하세요.
- **탐색 중복 (Exploration redundancy).** 두 에이전트가 동일한 상태-행동 쌍을 탐색합니다. 해결책: 에이전트별 엔트로피 보너스(entropy bonuses)를 부여하거나 역할 조건화(role-conditioning)를 사용하세요.
- **리그 순환 (League cycles).** 순수 자기 대전(self-play)은 지배 순환(dominance cycle)에 빠질 수 있습니다. 해결책: 다양한 상대와 대결하는 리그 플레이(league play)를 수행하세요.
- **샘플 폭발 (Sample explosion).** `n` 에이전트 × 상태 공간 × 결합 행동(joint actions)의 규모로 문제가 발생합니다. 해결책: 함수 근사(function approximation)를 사용하거나, 행동 공간을 분해(factored action spaces, 에이전트당 하나의 정책 출력 헤드)하여 근사하세요.

## 활용하기 (Use It)

2026년 MARL(Multi-Agent Reinforcement Learning) 응용 지도:

| 도메인 (Domain) | 방법론 (Method) | 비고 (Notes) |
|--------|--------|-------|
| 협력적 내비게이션 / 조작 (Cooperative navigation / manipulation) | MAPPO / QMIX | CTDE; 공유된 critic + 분산된 actors. |
| 2인 게임 (체스, 바둑, 포커) | MCTS를 활용한 셀프 플레이 (AlphaZero) | 제로섬(Zero-sum); 대칭적 학습. |
| 복잡한 멀티플레이어 (Dota, StarCraft) | 리그 플레이 + 모방 사전 학습 (imitation pretraining) | OpenAI Five, AlphaStar. |
| 자율주행 차량 플릿 (Autonomous-vehicle fleets) | CTDE MAPPO / 어텐션을 활용한 PPO | 부분 관측(Partial obs); 가변적인 팀 규모. |
| 경매 시장 (Auction markets) | 게임 이론적 평형 + RL | `n` → ∞일 때의 평균장 RL (Mean-field RL). |
| LLM 멀티 에이전트 시스템 (Phase 16) | 자연어 통신 + 역할 조건화 (role conditioning) | 에이전트 계획 계층에서의 RL 루프. |

2026년 MARL의 가장 큰 성장 분야는 LLM 기반 분야입니다. 즉, 언어 모델 에이전트 군집(swarms)이 협상하고, 토론하며, 소프트웨어를 구축하는 영역입니다. 여기서 RL은 토큰 수준이 아닌 *궤적 수준(trajectory-level)*의 출력에 대한 선호도 최적화(preference optimization) 형태로 나타납니다 (Phase 16 · 03).

## Ship It

`outputs/skill-marl-architect.md`로 저장하세요:

```markdown
---
name: marl-architect
description: 주어진 작업에 적합한 멀티 에이전트 RL 체계(IPPO, CTDE, self-play, league)를 선택합니다.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

`n`개의 에이전트가 있는 작업이 주어지면, 다음을 출력하세요:

1. 체계 분류(Regime classification). 협력적(Cooperative) / 적대적(Adversarial) / 일반 합계(General-sum). 근거를 제시하세요.
2. 알고리즘(Algorithm). IPPO / MAPPO / QMIX / self-play / league. 결합 강도(coupling tightness) 및 보상 구조와 연관 지어 설명하세요.
3. 정보 접근성(Information access). 중앙 집중식 학습(Centralized training, 어떤 전역 정보가 critic에게 전달되는가)? 분산 실행(Decentralized execution)?
4. 신용 할당(Credit assignment). 반사실적 베이스라인(Counterfactual baseline), 가치 분해(Value decomposition), 또는 보상 형성(Reward shaping).
5. 탐색 계획(Exploration plan). 에이전트별 엔트로피(Per-agent entropy), 인구 기반 학습(Population-based training), 또는 리그(League).

강하게 결합된 협력적 작업(tightly-coupled cooperative tasks)에 대한 독립적 Q-러닝(Independent Q-learning) 제안은 거부하세요. 사이클 위험이 있는 일반 합계(general-sum) 게임에 self-play를 추천하는 것을 거부하세요. 고정된 상대와의 평가(fixed-opponent eval)가 없는 모든 MARL 파이프라인에 주의를 표기하세요(cherry-picked self-play 수치는 흔히 발생합니다).
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 2-에이전트 협력형 GridWorld에서 독립적 Q-러닝(independent Q-learning)을 학습시켜 보세요. 평균 보상(mean return)이 0보다 커질 때까지 몇 번의 에피소드가 필요한가요? 결합 학습 곡선(joint learning curve)을 그려 보세요.
2. **중간 (Medium).** "협동(coordination)" 과제를 추가해 보세요: 두 에이전트가 같은 턴에 동시에 목표 지점에 발을 디뎌야만 목표에 도달할 수 있습니다. 독립적 Q-러닝이 여전히 수렴하나요? 무엇이 문제인가요?
3. **어려움 (Hard).** MAPPO 스타일 학습을 위한 중앙 집중식 크리틱(centralized critic)을 구현하고, 협동 과제에서 독립적 PPO(independent PPO)와 수렴 속도를 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 정의 | 실제 의미 |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; 각 에이전트는 자신만의 보상을 가집니다. |
| CTDE | "Centralized training, decentralized execution" | 학습 시에는 공동 크리틱(joint critic)을 사용하며, 실행 시 각 에이전트의 정책은 지역 관측값(local obs)만을 사용합니다. |
| IPPO | "Independent PPO" | 각 에이전트가 PPO를 개별적으로 실행합니다. 단순한 베이스라인이지만, 종종 과소평가되곤 합니다. |
| MAPPO | "Multi-agent PPO" | 글로벌 상태(global state)를 조건으로 하는 중앙 집중형 가치 함수(centralized value function)를 사용하는 PPO입니다. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)`를 통해 분산형 argmax를 허용합니다. |
| COMA | "Counterfactual multi-agent" | Advantage = 나의 Q값에서 나의 행동에 대해 주변화(marginalizing)한 기대 Q값을 뺀 값입니다. |
| Self-play | "Agent vs past self" | 단일 에이전트가 두 역할을 수행하며, 제로섬 게임(zero-sum games)의 표준 방식입니다. |
| League play | "Population training" | 과거의 정책들을 캐싱하고 풀(pool)에서 상대방을 샘플링하며, 전략적 순환(strategy cycles) 문제를 해결합니다. |

## 추가 읽을거리 (Further Reading)

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 중앙 집중식 크리틱(centralized critic)을 사용하는 CTDE 방식.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 기여도 할당(credit assignment)을 위한 반사실적 베이스라인(counterfactual baselines).
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 단조성(monotonicity)을 이용한 가치 분해(value decomposition).
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — MARL에서 놀라울 정도로 강력한 성능을 보이는 PPO.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 대규모 리그 플레이(league play).
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 제로섬 게임에서의 순수 자기 대전(self-play).
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 다중 에이전트 환경에 대한 교재의 간략한 설명과 CTDE가 해결하고자 하는 비정상성(non-stationarity) 문제를 포함합니다.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 협력적, 경쟁적, 혼합형 MARL 및 수렴 결과(convergence results)를 다루는 서베이 논문.
