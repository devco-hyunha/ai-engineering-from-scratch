# Deep Q-Networks (DQN)

> 2013년: Mnih는 가공되지 않은 픽셀(raw pixels) 데이터로 하나의 Q-learning 네트워크를 학습시켜 7개의 아타리(Atari) 게임에서 모든 고전적 RL 에이전트를 이겼습니다. 2015년: 이를 49개의 게임으로 확장하여 Nature지에 발표했으며, 이는 심층 강화학습(deep-RL) 시대의 서막을 알렸습니다. DQN은 Q-learning에 함수 근사(function approximation)를 안정화하는 세 가지 기법을 더한 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## 문제점 (The Problem)

Tabular Q-learning은 모든 (상태, 행동) 쌍에 대해 별도의 Q-값을 필요로 합니다. 체스판은 약 $10^{43}$개의 상태를 가집니다. Atari 프레임은 $210 \times 160 \times 3 = 100,800$개의 특징(feature)을 가집니다. Tabular RL은 수십억 개는커녕 수천 개의 상태만 되어도 작동이 불가능해집니다.

지나고 보면 해결책은 명확합니다. Q-테이블을 신경망 `Q(s, a; θ)`로 교체하는 것입니다. 하지만 이 명확한 해결책을 찾기까지 수십 년이 걸렸습니다. Q-learning을 이용한 단순한 함수 근사(naive function approximation)는 "치명적인 삼중주(deadly triad)" — 함수 근사(function approximation) + 부트스트래핑(bootstrapping) + 오프-폴리시 학습(off-policy learning) — 상황에서 발산합니다. Mnih et al. (2013, 2015)은 학습을 안정화하는 세 가지 엔지니어링 기법을 찾아냈습니다:

1. **경험 재현(Experience replay)**은 전이(transition) 간의 상관관계를 제거합니다.
2. **타겟 네트워크(Target network)**는 부트스트랩 타겟을 고정합니다.
3. **보상 클리핑(Reward clipping)**은 그래디언트 크기를 정규화합니다.

Atari 환경에서의 DQN은 단일 아키텍처와 단일 하이퍼파라미터 세트만으로 가공되지 않은 픽셀(raw pixels)로부터 수십 개의 제어 문제를 해결한 첫 사례였습니다. 이후 등장한 모든 "Deep-RL" 기술들 — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 —은 이 세 가지 기법이 적용된 기초 위에 쌓아 올려진 것들입니다.

## 개념 (The Concept)

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**목표.** DQN은 신경망 Q-함수에 대한 1단계 TD 손실(one-step TD loss)을 최소화합니다:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 온라인 네트워크(online network)로, 매 단계 경사 하강법(gradient descent)을 통해 업데이트됩니다. `θ^-` = 타겟 네트워크(target network)로, 주기적으로 `θ`로부터 복사됩니다(약 10,000단계마다). `D` = 과거 전이(transitions)들의 리플레이 버퍼(replay buffer)입니다.

**중요도 순으로 정리한 세 가지 트릭:**

**경험 리플레이(Experience replay).** 약 `~10⁶`개의 전이로 구성된 링 버퍼입니다. 각 학습 단계에서는 미니배치를 무작위로 균등하게 샘플링합니다. 이는 시간적 상관관계(연속된 프레임은 거의 동일함)를 끊어주고, 희귀한 보상 전이로부터 네트워크가 여러 번 학습할 수 있게 하며, 연속적인 경사 업데이트 간의 상관관계를 제거합니다. 이것이 없다면, 신경망을 사용하는 온폴리시(on-policy) TD는 Atari 게임에서 발산합니다.

**타겟 네트워크(Target network).** 벨만 방정식의 양변에 동일한 네트워크 `Q(·; θ)`를 사용하면 업데이트할 때마다 타겟이 움직이는 "자신의 꼬리를 쫓는(chasing your own tail)" 현상이 발생합니다. 해결책은 가중치가 고정된 두 번째 네트워크 `Q(·; θ^-)`를 유지하는 것입니다. `C` 단계마다 `θ → θ^-`로 복사합니다. 이를 통해 수천 번의 경사 단계 동안 회귀 타겟(regression target)을 안정화할 수 있습니다. 소프트 업데이트(Soft updates) `θ^- ← τ θ + (1-τ) θ^-` (DDPG, SAC에서 사용)는 이보다 더 부드러운 변형 방식입니다.

**보상 클리핑(Reward clipping).** Atari 보상의 크기는 1에서 1000 이상까지 다양합니다. 이를 `{-1, 0, +1}`로 클리핑하면 특정 게임이 경사를 지배하는 것을 방지할 수 있습니다. 보상의 크기가 중요한 경우에는 잘못된 방법이지만, 부호(sign)만이 중요한 Atari에서는 적절합니다.

**Double DQN.** Hasselt (2016)는 최대화 편향(maximization bias)을 해결합니다. 온라인 네트워크를 사용하여 행동을 *선택(select)*하고, 타겟 네트워크를 사용하여 해당 행동을 *평가(evaluate)*합니다.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

기존 방식을 그대로 대체할 수 있으며, 일관되게 더 나은 성능을 보입니다. 기본적으로 이를 사용하세요.

**기타 개선 사항 (Rainbow, 2017):** 우선순위 리플레이(prioritized replay, TD 오차가 큰 전이를 더 많이 샘플링), 듀얼링 아키텍처(dueling architecture, `V(s)`와 advantage 헤드를 분리), 노이즈 네트워크(noisy networks, 학습된 탐색), n-step 리턴(n-step returns), 분포형 Q(distributional Q, C51/QR-DQN), 멀티스텝 부트스트래핑(multi-step bootstrapping). 각 요소는 몇 퍼센트의 성능을 추가하며, 그 이득은 대략적으로 합산됩니다.

```figure
f3-dqn-stability
```

## 구현하기 (Build It)

이곳의 코드는 외부 라이브러리 없이 표준 라이브러리(stdlib)만을 사용하며 `numpy`를 사용하지 않습니다. 아주 작은 연속형 GridWorld 환경에서 직접 구현한 단일 은닉층 MLP(Multi-Layer Perceptron)를 사용하므로, 모든 학습 단계는 마이크로초(microseconds) 단위로 실행됩니다. 알고리즘은 대규모 환경에서의 Atari DQN과 동일합니다.

### 1단계: 리플레이 버퍼 (Replay Buffer)

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari의 경우 약 50,000의 용량(capacity)이 필요하지만, 우리의 토이 환경(toy env)에서는 5,000 정도면 충분합니다.

### 2단계: 아주 작은 Q-네트워크 (수동 MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

순전파(Forward pass): 선형(linear) → ReLU → 선형(linear) 과정입니다. 이것이 네트워크의 전부입니다.

### 3단계: DQN 업데이트 (the DQN update)

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

이 구조는 Lesson 04의 Q-learning과 형태는 유사하지만 두 가지 차이점이 있습니다: (a) 테이블에 인덱싱하는 대신 미분 가능한 `Q(·; θ)`를 통해 역전파(backprop)를 수행하며, (b) 타겟(target)으로 `Q(·; θ^-)`를 사용합니다.

### 4단계: 외부 루프 (the outer loop)

각 에피소드마다 `Q(·; θ)`에 대해 $\epsilon$-greedy 방식으로 행동하고, 전환(transition)을 버퍼에 저장하며, 미니배치를 샘플링하고, 경사 하강법(gradient step)을 수행하며, 주기적으로 `θ^- ← θ`를 동기화합니다. 패턴은 다음과 같습니다:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16차원 원-핫(one-hot) 상태를 가진 작은 GridWorld 환경에서 에이전트는 약 500 에피소드 만에 최적에 가까운 정책을 학습합니다. Atari 환경에서는 이를 2억(200M) 프레임 규모로 확장하고 CNN 특징 추출기(feature extractor)를 추가해 보세요.

## 주의 사항 (Pitfalls)

- **치명적 삼중주 (Deadly triad).** 함수 근사(Function approximation) + 오프-폴리시(off-policy) + 부트스트래핑(bootstrapping)이 결합되면 발산할 수 있습니다. DQN은 타겟 네트워크(target net)와 리플레이 버퍼(replay buffer)를 통해 이를 완화합니다. 둘 중 어느 것도 제거하지 마세요.
- **탐험 (Exploration).** $\epsilon$은 반드시 감소해야 하며, 일반적으로 학습 초기 약 10% 구간 동안 1.0에서 0.01까지 감소시킵니다. 초기 탐험이 충분하지 않으면 Q-네트워크가 지역 최적점(local basin)으로 수렴해 버립니다.
- **과대평가 (Overestimation).** 노이즈가 섞인 Q값에 대해 `max` 연산을 수행하면 상향 편향(upward-biased)이 발생합니다. 실제 서비스 환경에서는 항상 Double DQN을 사용하세요.
- **보상 스케일 (Reward scale).** 보상을 클리핑(clip)하거나 정규화(normalize)하세요. 그래디언트의 크기는 보상의 크기에 비례합니다.
- **리플레이 버퍼 콜드 스타트 (Replay buffer coldstart).** 버퍼에 수천 개의 전이(transition)가 쌓이기 전까지는 학습을 시작하지 마세요. 약 20개의 샘플만 있는 상태에서의 초기 그래디언트는 과적합(overfit)을 유발합니다.
- **타겟 동기화 빈도 (Target sync frequency).** 너무 빈번하면 타겟 네트워크를 사용하는 의미가 없고, 너무 드물면 타겟이 너무 오래된(stale) 상태가 됩니다. Atari DQN은 10,000 환경 스텝을 사용합니다. 경험적인 규칙(Rule of thumb)으로는 전체 학습 기간의 약 1/100마다 동기화하세요.
- **관측값 전처리 (Observation preprocessing).** Atari DQN은 상태가 마르코프(Markov) 성질을 갖게 하기 위해 4개의 프레임을 스택(stack)합니다. 속도(velocity) 정보가 필요한 모든 환경은 프레임 스태킹 또는 순환 상태(recurrent state)가 필요합니다.

## 활용하기 (Use It)

2026년 기준으로 DQN은 최첨단(state-of-the-art) 기술인 경우는 드물지만, 오프-폴리시(off-policy) 알고리즘의 기준점으로 남아 있습니다.

| 작업 (Task) | 권장 방법 (Method of choice) | DQN을 사용하지 않는 이유 (Why not DQN?) |
|------|------------------|--------------|
| 이산 행동(Discrete-action) Atari 스타일 | Rainbow DQN 또는 Muesli | 동일한 프레임워크 내에서 더 많은 기법(tricks)을 사용함. |
| 연속 제어 (Continuous control) | SAC / TD3 (Phase 9 · 07) | DQN에는 정책 네트워크(policy network)가 없음. |
| 온-폴리시(On-policy) / 고처리량(high-throughput) | PPO (Phase 9 · 08) | 리플레이 버퍼(replay buffer)가 필요 없음; 확장(scale)이 더 쉬움. |
| 오프라인 RL (Offline RL) | CQL / IQL / Decision Transformer | 보수적인 Q 타겟(Conservative Q targets)을 사용하여 부트스트래핑 폭발(bootstrapping blowups)이 없음. |
| 대규모 이산 행동 공간 (추천 시스템) | 액션 임베딩(action embedding)을 적용한 DQN 또는 IMPALA | 사용 가능함; 세부적인 조정(decoration)이 중요함. |
| LLM RL | PPO / GRPO | 스텝 단위가 아닌 시퀀스 단위이며, 손실 함수(loss)가 다름. |

여기서 배운 교훈들은 여전히 유효합니다. 리플레이(Replay)와 타겟 네트워크(target networks)는 SAC, TD3, DDPG, SAC-X, AlphaZero의 셀프 플레이 버퍼(self-play buffer), 그리고 모든 오프라인 RL 방법론에서 등장합니다. 리워드 클리핑(Reward clipping)은 PPO의 어드밴티지 정규화(advantage normalization)로 이어집니다. 이 구조가 바로 설계도(blueprint)입니다.

## Ship It

`outputs/skill-dqn-trainer.md`로 저장하세요:

```markdown
---
name: dqn-trainer
description: 이산 행동(discrete-action) RL 작업을 위한 DQN 학습 설정(버퍼, 타겟 동기화, ε 스케줄, 보상 클리핑)을 생성합니다.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

이산 행동 환경(관측 형태, 행동 수, 호라이즌, 보상 스케일)이 주어지면 다음을 출력하세요:

1. Network. 아키텍처(MLP / CNN / Transformer), 특징 차원(feature dim), 깊이.
2. Replay buffer. 용량(Capacity), 미니배치 크기(minibatch size), 웜업 크기(warmup size).
3. Target network. 동기화 전략(C 스텝마다 수행하는 Hard sync 또는 소프트 $\tau$ sync).
4. Exploration. $\epsilon$ 시작값 / 종료값 / 스케줄 길이.
5. Loss. Huber vs MSE, 그래디언트 클리핑 값, 보상 클리핑 규칙.
6. Double DQN. 명시적으로 비활성화해야 할 이유가 없는 한 기본적으로 활성화.

타겟 네트워크가 없거나, 리플레이 버퍼가 없거나, $\epsilon$이 1로 유지되는 DQN은 생성을 거부하세요. 연속 행동(continuous-action) 작업은 거부하세요(SAC / TD3로 안내). 스텝당 평균 보상의 10배를 초과하는 보상 범위가 발견되면 클리핑 또는 스케일 정규화가 필요하다고 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하세요. 에피소드당 리턴(per-episode return) 곡선을 그리세요. 이동 평균(running mean)이 -10을 초과할 때까지 몇 번의 에피소드가 소요됩니까?
2. **중간 (Medium).** 타겟 네트워크(target network)를 비활성화하세요 (Bellman target의 양쪽 모두에 online net을 사용합니다). 학습 불안정성(training instability)을 측정하세요. 리턴이 진동(oscillate)합니까, 아니면 발산(diverge)합니까?
3. **어려움 (Hard).** Double DQN을 추가하세요: `argmax a'`를 선택할 때는 online net을 사용하고, 값을 평가할 때는 target net을 사용합니다. 노이즈가 있는 보상(noisy-reward) 환경의 GridWorld에서, Double DQN을 사용했을 때와 사용하지 않았을 때 1,000 에피소드 후 `Q(s_0, best_a)`의 편향(bias)을 실제 `V*(s_0)`와 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 정의 | 실제 의미 |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | 신경망 Q-함수(neural Q-function), 리플레이 버퍼(replay buffer), 타겟 네트워크(target network)를 사용하는 Q-learning. |
| Experience replay | "셔플된 전이(Shuffled transitions)" | 매 그래디언트 단계마다 균등하게 샘플링되는 링 버퍼(ring buffer); 데이터 간의 상관관계를 제거함. |
| Target network | "Frozen bootstrap" | 벨만 타겟(Bellman target)에 사용되는 Q의 주기적 복사본; 학습을 안정화함. |
| Deadly triad | "RL이 발산하는 이유" | 함수 근사(Function approximation) + 부트스트래핑(bootstrapping) + 오프-폴리시(off-policy) = 수렴 보장 없음. |
| Double DQN | "최대화 편향(maximization bias) 해결책" | 온라인 네트워크가 행동을 선택하고, 타겟 네트워크가 해당 행동을 평가함. |
| Dueling DQN | "V와 A 헤드" | $Q = V + A - \text{mean}(A)$로 분해; 동일한 출력, 더 나은 그래디언트 흐름. |
| Rainbow | "모든 기법의 집합" | DDQN + PER + dueling + n-step + noisy + distributional을 하나로 통합. |
| PER | "우선순위 리플레이(Prioritized Replay)" | TD-오차(TD-error) 크기에 비례하여 전이를 샘플링함. |

## 추가 학습 자료 (Further Reading)

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 심층 강화 학습(deep RL)의 시작을 알린 2013년 NeurIPS 워크숍 논문입니다.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — 49개의 게임을 수행한 DQN을 다룬 Nature 논문입니다.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN에 관한 논문입니다.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — Dueling DQN에 관한 논문입니다.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 여러 개선 사항을 결합한 Rainbow 논문입니다.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 명확하고 현대적인 설명이 담긴 자료입니다.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — DQN의 타겟 네트워크(target network)와 리플레이 버퍼(replay buffer)가 제어하고자 하는 "치명적 삼중주(deadly triad)"(함수 근사 + 부트스트래핑 + off-policy)를 교과서적으로 다룹니다.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 어블레이션 연구(ablation studies)에서 사용되는 참조용 단일 파일 DQN 구현체입니다. 본 강의의 밑바닥부터 구현하는 버전과 함께 읽어보시면 좋습니다.
