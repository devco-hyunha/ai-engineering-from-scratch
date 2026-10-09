# Deep Q-Networks (DQN)

> 2013년: Mnih는 원시 픽셀(raw pixels)에 대해 하나의 Q-학습 네트워크를 훈련하여 7개의 Atari 게임에서 모든 고전 RL 에이전트보다 높은 성능을 달성했습니다. 2015년: 이를 49개 게임으로 확장하여 Nature에 게재했으며, 딥 RL 시대를 촉발했습니다. DQN은 Q-학습에 함수 근사를 안정적으로 만드는 세 가지 트릭을 더한 것입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 03강 (역전파(Backpropagation)), 9단계 · 04강 (Q-학습, SARSA)
**시간:** 약 75분

## 문제점

테이블형 Q-학습은 모든 (상태, 행동) 쌍에 대해 별도의 Q-값을 필요로 합니다. 체스판은 약 10⁴³개의 상태를 가지며, Atari 프레임은 210×160×3 = 100,800개의 특징을 가집니다. 테이블형 RL은 수천 개의 상태에서도 한계에 부딪히며, 수십억 개의 상태는 말할 것도 없습니다.

사후에 보면 해결책은 명확합니다: Q-테이블을 신경망으로 대체하는 것, `Q(s, a; θ)`. 하지만 사후에 명확한 해결책은 수십 년이 걸렸습니다. Q-학습과 함께 naive한 함수 근사를 사용하면 "치명적 삼중주(deadly triad)" — 함수 근사 + 부트스트래핑 + off-policy 학습 —로 인해 발산합니다. Mnih et al. (2013, 2015)은 학습을 안정화하는 세 가지 엔지니어링 트릭을 식별했습니다:

1. **경험 리플레이(Experience replay)**는 전이(transitions)의 상관관계를 제거합니다.
2. **타겟 네트워크(Target network)**는 부트스트래핑 타겟을 고정합니다.
3. **보상 클리핑(Reward clipping)**은 기울기 크기를 정규화합니다.

Atari에서의 DQN은 하나의 아키텍처와 하나의 하이퍼파라미터 세트가 원시 픽셀로부터 수십 개의 제어 문제를 해결한 최초의 사례였습니다. 이후 구축된 모든 "deep-RL" — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 —는 이 세 가지 트릭 기반 위에 쌓여 있습니다.

## 개념

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**목표.** DQN은 신경망 Q-함수에 대해 1단계 TD 손실을 최소화합니다:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 온라인 네트워크, 매 단계마다 경사 하강법으로 업데이트됩니다. `θ^-` = 타겟 네트워크, `θ`에서 주기적으로 복사됩니다 (약 10,000 단계마다). `D` = 과거 전이(transitions)의 리플레이 버퍼입니다.

**세 가지 트릭, 중요도 순서대로:**

**경험 재생.** `~10⁶` 트랜지션의 링 버퍼입니다. 각 학습 단계에서 미니배치를 균일하게 무작위로 샘플링합니다. 이는 시간적 상관관계(연속 프레임이 거의 동일함)를 깨뜨리고, 네트워크가 희소한 보상 트랜지션에서 여러 번 학습할 수 있게 하며, 연속적인 기울기 업데이트를 비상관화합니다. 이것이 없으면 신경망 기반의 온폴리시 TD는 Atari에서 발산합니다.

**타겟 네트워크.** 벨만 방정식의 양쪽에 동일한 네트워크 `Q(·; θ)`를 사용하면 타겟이 매 업데이트마다 이동하여 "자기 꼬리를 쫓는" 현상이 발생합니다. 해결책: 가중치가 고정된 두 번째 네트워크 `Q(·; θ^-)`를 유지하는 것입니다. `C` 단계마다 `θ → θ^-`를 복사합니다. 이는 수천 개의 기울기 단계 동안 회귀 타겟을 안정화합니다. 부드러운 업데이트 `θ^- ← τ θ + (1-τ) θ^-` (DDPG, SAC에서 사용)는 더 매끄러운 변형입니다.

**보상 클리핑.** Atari 보상 크기는 1에서 1000+까지 다양합니다. `{-1, 0, +1}`로 클리핑하면 단일 게임이 기울기를 지배하는 것을 방지합니다. 보상 크기가 중요한 경우에는 부적절하지만, 부호만 중요한 Atari에서는 문제가 없습니다.

**Double DQN.** Hasselt (2016)는 최대화 편향을 수정합니다: 온라인 네트워크로 *동작을 선택*하고, 타겟 네트워크로 *평가*합니다.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

드롭인 대체품이며, 일관되게 더 좋습니다. 기본으로 사용하세요.

**기타 개선 사항 (Rainbow, 2017):** 우선순위 재생 (높은 TD 오차 트랜지션을 더 많이 샘플링), 듀얼링 아키텍처 (`V(s)`와 어드밴티지 헤드를 분리), 노이즈 네트워크 (학습된 탐색), n-단계 리턴, 분포 Q (C51/QR-DQN), 다단계 부트스트래핑. 각각 몇 퍼센트의 성능 향상을 추가하며, 이 향상은 대략적으로 가산적입니다.

```figure
f3-dqn-stability
```

## 구현하기

여기 있는 코드는 표준 라이브러리 전용이며 numpy-free입니다. 작은 연속 GridWorld에 수동으로 만든 단일 은닉층 MLP를 사용하므로 모든 학습 단계가 마이크로초 단위로 실행됩니다. 알고리즘은 대규모 Atari DQN과 동일합니다.

### 1단계: 재생 버퍼

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

Atari의 경우 약 50,000 용량; 우리의 장난감 환경에는 5,000이면 충분합니다.

### 2단계: 작은 Q-네트워크 (수동 MLP)

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

순방향 패스: 선형 → ReLU → 선형. 이것이 전체 네트워크입니다.

### 3단계: DQN 업데이트

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

구조는 04강의 Q-학습과 두 가지 차이점이 있습니다: (a) 테이블을 인덱싱하는 대신 미분 가능한 `Q(·; θ)`를 통해 역전파하며, (b) 타겟은 `Q(·; θ^-)`를 사용합니다.

### 4단계: 외부 루프

각 에피소드마다 `Q(·; θ)`에 대해 ε-탐욕(ε-greedy)으로 행동하고, 전이(transitions)를 버퍼에 푸시하고, 미니배치를 샘플링하고, 기울기 스텝을 수행하고, `θ^- ← θ`를 주기적으로 동기화합니다. 패턴은 다음과 같습니다:

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

16차원 원-hot 상태인 작은 GridWorld에서는 에이전트가 약 500 에피소드 안에 준최적 정책을 학습합니다. Atari에서는 이를 2억 프레임으로 확장하고 CNN 특징 추출기를 추가합니다.

## 함정

- **치명적 삼중주.** 함수 근사 + 오프-폴리시(off-policy) + 부트스트래핑은 발산할 수 있습니다. DQN은 타겟 네트워크 + 리플레이로 이를 완화하므로, 둘 중 하나를 제거하지 마세요.
- **탐색.** ε는 학습의 첫 ~10% 동안 1.0에서 0.01로 감소해야 합니다. 초기 탐색이 충분하지 않으면 Q 네트워크가 지역적 최소점(local basin)에 수렴합니다.
- **과대평가.** `max`는 잡음이 있는 Q에 대해 상향 편향됩니다. 프로덕션에서는 항상 Double DQN을 사용하세요.
- **보상 스케일.** 보상을 클리핑하거나 정규화하세요. 기울기 크기는 보상 크기에 비례합니다.
- **리플레이 버퍼 콜드 스타트.** 버퍼에 수천 개의 전이가 쌓일 때까지 학습하지 마세요. ~20개 샘플에 대한 초기 기울기는 과적합됩니다.
- **타겟 동기화 빈도.** 너무 빈번하면 ≈ 타겟 네트워크가 없는 것과 같고, 너무 드물면 ≈ 오래된 타겟이 됩니다. Atari DQN은 10,000 환경 스텝을 사용합니다. 경험칙: 학습 기간의 ~1/100마다 동기화하세요.
- **관측 전처리.** Atari DQN은 상태를 마르코프(Markov)로 만들기 위해 4 프레임을 스택합니다. 속도 정보가 있는 환경은 프레임 스택이나 순환 상태가 필요합니다.

## 사용하기

2026년 현재, DQN은 거의 최첨단(state-of-the-art)이 아니지만, 여전히 기준이 되는 오프-폴리시(off-policy) 알고리즘입니다:

| 작업 | 선택 방법 | DQN을 쓰지 않는 이유 |
|------|------------------|--------------|
| 이산 행동 Atari 유사 | Rainbow DQN 또는 Muesli | 같은 프레임워크, 더 많은 트릭. |
| 연속 제어 | SAC / TD3 (9단계 · 07강) | DQN에는 정책 네트워크가 없습니다. |
| 온-폴리시 / 고처리량 | PPO (9단계 · 08강) | 리플레이 버퍼 없음; 확장하기 더 쉬움. |
| 오프라인 RL | CQL / IQL / Decision Transformer | 보수적인 Q 타겟, 부트스트래핑 폭발 없음. |
| 대규모 이산 행동 공간 (추천 시스템) | 행동 임베딩을 사용한 DQN 또는 IMPALA | 괜찮음; 세부 사항이 중요합니다. |
| LLM RL | PPO / GRPO | 시퀀스 단위, 스텝 단위 아님; 손실 함수가 다름. |

이 교시들은 여전히 유효합니다. 리플레이 및 타겟 네트워크는 SAC, TD3, DDPG, SAC-X, AlphaZero의 자기 대국(self-play) 버퍼, 그리고 모든 오프라인 RL 방법론에 등장합니다. 보상 클리핑은 PPO에서의 어드밴티지 정규화(advantage normalization)로 이어집니다. 이 아키텍처는 청사진입니다.

## 출시하기

`outputs/skill-dqn-trainer.md`로 저장하세요:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 에피소드별 리턴(return) 곡선을 플롯하세요. 이동 평균이 -10을 초과할 때까지 몇 에피소드가 걸리나요?
2. **중간.** 타겟 네트워크를 비활성화하세요 (벨만 타겟의 양쪽 모두에 온라인 네트워크를 사용하세요). 학습 불안정성을 측정하세요 — 리턴이 진동(oscillate)하거나 발산(diverge)하나요?
3. **어려움.** Double DQN을 추가하세요: 온라인 네트워크로 `argmax a'`를 선택하고, 타겟 네트워크로 평가하세요. 노이즈가 있는 보상 GridWorld에서 Double DQN을 사용했을 때와 사용하지 않았을 때, 1,000 에피소드 후 `Q(s_0, best_a)`의 편향(bias)과 실제 `V*(s_0)`를 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | 신경망 Q-함수, 리플레이 버퍼, 타겟 네트워크를 사용하는 Q-learning. |
| Experience replay | "Shuffled transitions" | 각 기울기 단계마다 균일하게 샘플링하는 링 버퍼; 데이터를 비상관화(decorrelate)합니다. |
| Target network | "Frozen bootstrap" | 벨만 타겟에 사용되는 Q의 주기적 복사본; 학습을 안정화합니다. |
| Deadly triad | "Why RL diverges" | 함수 근사 + 부트스트래핑 + 오프-폴리시 = 수렴 보장 없음. |
| Double DQN | "Fix for maximization bias" | 온라인 네트워크가 액션을 선택하고, 타겟 네트워크가 이를 평가합니다. |
| Dueling DQN | "V and A heads" | Q = V + A - mean(A)로 분해; 동일한 출력, 더 나은 기울기 흐름. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional을 하나로 통합. |
| PER | "Prioritized Replay" | TD-error 크기에 비례하여 전이(transitions)를 샘플링합니다. |

## 추가 읽기

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 딥 RL의 시작을 알린 2013 NeurIPS 워크숍 논문.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 논문, 49개 게임 DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 스택된(trick) 논문.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 함수 근사 + 부트스트래핑 + 오프 정책의 "치명적 삼중주"에 대한 교과서적 처리로, DQN의 타겟 네트워크와 리플레이 버퍼가 이를 제어하기 위해 설계되었습니다.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — Ablation 연구에서 사용되는 단일 파일 DQN 참조 구현입니다. 이 강의의 처음부터 구현한 버전과 함께 읽어 보세요.
