# 게임을 위한 강화학습(RL for Games) — AlphaZero, MuZero, 그리고 LLM 추론 시대

> 1992년: TD-Gammon이 순수 TD(Temporal Difference) 학습만으로 백개먼 인간 챔피언을 이겼습니다. 2016년: AlphaGo가 이세돌을 이겼습니다. 2017년: AlphaZero가 아무런 사전 지식 없이 체스, 쇼기, 바둑을 지배했습니다. 2024년: DeepSeek-R1은 PPO를 대체한 GRPO를 통해 동일한 레시피가 추론(Reasoning) 작업에서도 작동함을 증명했습니다. 게임은 이 모든 단계의 돌파구를 이끄는 벤치마크입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## 문제 (The Problem)

게임은 강화학습(RL)이 요구하는 모든 요소를 갖추고 있습니다. 명확한 보상(승리/패배), 무한한 에피소드(셀프 플레이를 통한 리셋), 완벽한 시뮬레이션(게임 자체가 시뮬레이터), 이산적(discrete)이거나 작은 연속적(continuous) 액션 공간, 그리고 적대적 강건성(adversarial robustness)을 강제하는 멀티 에이전트 구조까지 말이죠.

또한 게임은 모든 주요 RL 혁신 기술이 테스트되는 장이었습니다. TD-Gammon (백개먼, 1992), Atari-DQN (2013), AlphaGo (2016), AlphaZero (2017), OpenAI Five (Dota 2, 2019), AlphaStar (StarCraft II, 2019), MuZero (학습된 모델, 2019), AlphaTensor (행렬 곱셈, 2022), AlphaDev (정렬 알고리즘, 2023), 그리고 DeepSeek-R1 (수학적 추론, 2025) — 이는 게임-RL 기술이 텍스트 영역에서도 작동한다는 것을 보여주는 최신 사례입니다.

본 캡스톤 과정은 세 가지 이정표적인 아키텍처인 AlphaZero, MuZero, GRPO를 하나의 통합된 관점인 **셀프 플레이(self-play) + 탐색(search) + 정책 개선(policy improvement)**을 통해 살펴봅니다. 각 모델은 이전 모델을 일반화합니다. 특히 GRPO는 AlphaZero의 레시피를 LLM 추론에 적용한 것으로, 토큰을 액션으로, 수학적 검증을 승리 신호로 사용합니다.

## 개념 (The Concept)

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**통합된 루프 (The unifying loop).**

```
while True:
    trajectory = self_play(current_policy, search)     # 자신과 대국(self-play)
    policy_target = search.improved_policy(trajectory) # 탐색을 통해 원시 정책(raw policy) 개선
    policy_net.update(policy_target, value_target)     # 탐색 결과로 지도 학습(supervised)
```

**AlphaZero (2017).** Silver et al. 규칙이 알려진 게임(체스, 쇼기, 바둑)이 주어졌을 때:

- 정책-가치 네트워크(Policy-value network): 하나의 타워 `f_θ(s) → (p, v)`를 가집니다. `p`는 가능한 수들에 대한 사전 확률(prior)이며, `v`는 예상되는 게임 결과입니다.
- 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS): 매 수마다 가능한 진행 경로의 트리를 확장합니다. `(p, v)`를 사전 확률(prior) 및 부트스트랩(bootstrap)으로 사용합니다. UCB(PUCT)를 사용하여 노드를 선택합니다: `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- 셀프 플레이(Self-play): 에이전트 대 에이전트로 게임을 진행합니다. `t`번째 수에서 MCTS 방문 분포 `π_t`가 정책 학습의 타겟이 됩니다.
- 손실 함수(Loss): `L = (v - z)² - π · log p + c · ||θ||²`. 여기서 `z`는 게임 결과(+1 / 0 / -1)입니다.

인간의 지식은 전혀 사용되지 않습니다. 수작업으로 만든 휴리스틱도 없습니다. 단 하나의 레시피로 각각 수천만 번의 셀프 플레이 게임을 거친 후 체스, 쇼기, 바둑을 마스터했습니다.

**MuZero (2019).** Schrittwieser et al. 규칙을 반드시 알아야 한다는 요구 사항을 제거했습니다.

- 고정된 환경 대신, *잠재 역학 모델(latent dynamics model)* `(h, g, f)`를 학습합니다:
  - `h(s)`: 관측값을 잠재 상태(latent state)로 인코딩합니다.
  - `g(s_latent, a)`: 다음 잠재 상태와 보상을 예측합니다.
  - `f(s_latent)`: 정책 사전 확률(policy prior)과 가치를 예측합니다.
- MCTS는 *학습된 잠재 공간(learned latent space)* 내에서 실행됩니다. 탐색 방식과 학습 루프는 동일합니다.
- 바둑, 체스, 쇼기뿐만 아니라 *Atari* 게임에서도 작동합니다. 하나의 알고리즘으로 규칙 지식 없이 수행합니다.

**Stochastic MuZero (2022).** 확률적 역학(stochastic dynamics)과 확률 노드(chance nodes)를 추가하여 백개먼(backgammon) 급의 게임으로 확장했습니다.

**Muesli, Gumbel MuZero (2022-2024).** 샘플 효율성(sample efficiency)과 결정론적 탐색(deterministic search)을 개선했습니다.

**GRPO (2024-2025).** DeepSeek-R1 레시피입니다. AlphaZero와 유사한 형태의 루프를 언어 모델 추론(reasoning)에 적용했습니다:

- "게임": 수학 / 코딩 / 추론 문제를 푸는 것입니다. "승리"는 검증기(verifier, 테스트 케이스 통과 또는 수치적 정답 일치)가 1을 반환하는 것을 의미합니다.
- 정책(Policy): LLM. 행동(Actions): 토큰. 상태(State): 프롬프트 + 지금까지의 응답.
- 비평가(Critic, PPO 방식의 `V_φ`)가 없습니다. 대신, 각 프롬프트에 대해 정책으로부터 `G`개의 완성을 샘플링합니다. 각각의 보상을 계산합니다. **그룹 상대적 어드밴티지(group-relative advantage)** `A_i = (r_i - mean_r) / std_r`을 REINFORCE 방식 업데이트를 위한 신호로 사용합니다.
- 드리프트(drift)를 방지하기 위해 참조 정책(reference policy)에 대한 KL 페널티를 적용합니다(RLHF와 유사).
- 전체 손실 함수:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

보상 모델(reward model), 비평가(critic), MCTS가 없습니다. 그룹 상대적 베이스라인(group-relative baseline)이 이 세 가지를 모두 대체합니다. 매우 적은 연산량으로 추론 벤치마크에서 PPO-RLHF 수준에 도달하거나 이를 능가합니다.

**R1 레시피 전체 구성.** DeepSeek-R1 (DeepSeek 2025)은 한 논문에 담긴 두 개의 모델입니다:

- **R1-Zero.** DeepSeek-V3 베이스 모델에서 시작합니다. SFT(지도 미세 조정)를 거치지 않습니다. 두 가지 보상 구성 요소를 사용하여 GRPO를 직접 적용합니다: *정확도 보상(accuracy reward)* (규칙 기반 — 최종 답변이 올바른 숫자로 파싱되었는가 / 코드가 유닛 테스트를 통과했는가) 및 *형식 보상(format reward)* (완성된 답변이 사고 과정(CoT)을 `<think>…</think>` 태그로 감쌌는가). 수천 단계에 걸쳐 평균 응답 길이는 약 100토큰에서 약 10,000토큰으로 증가하며, 수학 벤치마크 점수는 o1-preview 수준에 근접합니다. 모델은 아무것도 없는 상태에서 추론하는 법을 배웁니다. 단점은 사고 과정이 읽기 어렵고, 언어가 섞여 있으며, 문체적 세련미가 부족하다는 점입니다.
- **R1.** 4단계 파이프라인을 통해 R1-Zero의 가독성 문제를 해결합니다:
  1. **콜드 스타트 SFT (Cold-start SFT).** 깔끔한 형식을 갖춘 수천 개의 긴 CoT 데모를 수집합니다. 이를 통해 베이스 모델을 지도 미세 조정합니다. 이는 읽기 좋은 시작점을 제공합니다.
  2. **추론 중심 GRPO (Reasoning-oriented GRPO).** 정확도+형식 보상에 더해, 언어 전환(code-switching)을 방지하기 위한 *언어 일관성(language-consistency)* 보상을 추가하여 GRPO를 적용합니다.
  3. **거절 샘플링 + 2차 SFT (Rejection sampling + SFT round 2).** RL 체크포인트에서 약 60만 개의 추론 궤적을 샘플링하여, 최종 답변이 정확하고 CoT가 읽기 쉬운 것만 남깁니다. 이를 약 20만 개의 비추론 SFT 예시(작문, QA, 자기 인식)와 결합합니다. 베이스 모델을 다시 미세 조정합니다.
  4. **전 영역 GRPO (Full-spectrum GRPO).** 추론(규칙 기반 보상)과 일반적 정렬(도움됨/무해함 기반 선호도 보상)을 모두 아우르는 마지막 RL 단계를 수행합니다.

그 결과, 오픈 웨이트 모델임에도 AIME 및 MATH-500에서 o1과 대등한 성능을 보이며, 증류(distill)하기에 충분히 작은 크기입니다. 또한 동일한 논문에서 R1의 추론 흔적(reasoning traces)을 사용하여 SFT를 수행함으로써 6개의 증류된 밀집 모델(Qwen-1.5B부터 Llama-70B까지)을 출시했습니다. 학생 모델 단계에서는 RL을 수행하지 않았습니다. 강력한 RL 교사로부터의 증류는 학생 모델 규모에서 처음부터 RL을 수행하는 것보다 일관되게 우수한 성능을 보입니다.

**추론에서 PPO 대신 GRPO를 사용하는 이유.** DeepSeekMath 논문(2024년 2월)에서 제시한 세 가지 이유입니다: (1) 학습할 가치 네트워크(value network)가 없어 메모리를 절반으로 줄일 수 있습니다; (2) 그룹 베이스라인은 추론 작업에서 발생하는 희소한 궤적 끝단 보상(sparse end-of-trajectory reward)을 자연스럽게 처리합니다; (3) 프롬프트별 정규화(per-prompt normalization)를 통해 난이도가 판이하게 다른 문제들 사이에서도 어드밴티지(advantages)를 비교 가능하게 만듭니다. 이는 PPO의 단일 비평가로는 불가능한 일입니다.

**탐색 미사용(Search-free) vs 탐색 사용(Search-based).** 게임의 양상이 갈라졌습니다:

- *긴 호흡을 가진 완전 정보 게임* (바둑, 체스): 여전히 탐색 기반입니다. AlphaZero / MuZero가 지배적입니다.
- *LLM 추론*: 아직 프로덕션 환경에서 MCTS를 사용하지는 않습니다. 전체 롤아웃(rollouts)에 GRPO를 적용하거나, 추론 연산을 위해 Best-of-N 방식을 사용합니다. 프로세스 보상 모델(PRMs)은 단계별 탐색이 다시 도입될 가능성을 시사합니다.

```figure
f3-selfplay-ladder
```

## 구현하기 (Build It)

`code/main.py`에 있는 코드는 **미니어처 GRPO (GRPO in miniature)**를 구현합니다. 이는 여러 샘플 그룹을 가진 밴딧(bandit) 알고리즘입니다. 알고리즘 자체는 LLM에서 사용하는 것과 동일하며, 정책(policy)과 환경(environment)만 더 단순화되었습니다. 이 코드는 2025년의 혁신 기술인 *손실 함수(loss)*와 *그룹 상대적 어드밴티지(group-relative advantage)*의 개념을 학습합니다.

### 1단계: 아주 작은 검증기(Verifier) 환경

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

실제 GRPO에서는 검증기가 유닛 테스트를 실행하거나 수학적 등식을 확인합니다.

### 2단계: policy: 프롬프트당 K개 답변 토큰에 대한 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

프롬프트가 주어졌을 때 LLM의 최종 레이어 출력과 동일합니다.

### 3단계: 그룹 샘플링 및 그룹 상대적 어드밴티지 (group sampling and group-relative advantage)

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 페널티: theta를 참조 모델(reference) 쪽으로 끌어당김
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

그룹 상대적 어드밴티지(group-relative advantage)는 2024년 DeepSeek이 선보인 기법입니다. 별도의 비평가(critic) 모델이 필요하지 않습니다. 여기서 "베이스라인(baseline)"은 그룹 평균이며, 정규화에는 그룹 표준 편차를 사용합니다.

### 4단계: REINFORCE 베이스라인(가치 함수 미사용)과 비교

동일한 설정, 동일한 연산량으로 일반적인 REINFORCE를 적용합니다. GRPO가 더 빠르고 안정적으로 수렴합니다.

### 5단계: 엔트로피(Entropy) 및 KL 발산(KL Divergence) 관찰

RLHF와 동일한 진단 지표를 사용합니다: 참조 모델(Reference model)과의 평균 KL 발산, 정책 엔트로피(Policy entropy), 시간 경과에 따른 보상(Reward-over-time)을 확인하세요. 이 지표들이 안정화되면 학습이 완료된 것입니다.

## 주의 사항 (Pitfalls)

- **검증기 악용을 통한 보상 해킹(Reward hacking via verifier gaming).** GRPO는 RLHF의 위험성을 그대로 물려받습니다. 만약 검증기(verifier)가 잘못되었거나 악용 가능하다면, LLM은 그 취약점을 찾아낼 것입니다. 따라서 강력한 검증기(다양한 테스트 케이스, 형식적 증명 등)를 구축하는 것이 중요합니다.
- **너무 작은 그룹 크기(Group size too small).** 그룹 베이스라인의 분산은 `1/√G`에 비례합니다. `G = 4` 미만일 경우 어드밴티지(advantage) 신호에 노이즈가 심해지며, 일반적으로 `G = 8`에서 `64` 사이를 선택합니다.
- **길이 편향(Length bias).** 길이가 서로 다른 LLM 생성 결과물은 서로 다른 로그 확률(log-probabilities)을 가집니다. 토큰 수로 정규화하거나, 시퀀스 수준의 로그 확률을 사용하거나, 최대 길이에 맞춰 자르는(truncate) 방식을 사용하세요.
- **순수 자기 대국 사이클(Pure self-play cycles).** AlphaZero 방식의 학습은 일반 합 게임(general-sum games)에서 지배 루프(dominance loops)에 빠질 수 있습니다. 이는 다양한 상대 풀(diverse opponent pools, 리그 플레이 방식, Lesson 10 참조)을 통해 완화할 수 있습니다.
- **탐색-정책 불일치(Search-policy mismatch).** AlphaZero는 정책(policy)이 탐색(search) 결과를 모방하도록 학습합니다. 만약 정책 네트워크가 탐색의 분포를 표현하기에 너무 작다면 학습이 정체됩니다.
- **컴퓨팅 하한선(Compute floor).** MuZero / AlphaZero는 막대한 컴퓨팅 자원을 필요로 합니다. 단일 어블레이션(ablation) 실험에도 수백 GPU 시간이 소요되는 경우가 많습니다. 학습을 위해 소규모 데모(예: Connect Four에서의 AlphaZero)를 활용해 보세요.
- **검증기 커버리지(Verifier coverage).** 버그가 있는 솔루션이 통과되는 유닛 테스트는 해당 버그를 강화합니다. 엣지 케이스(edge cases)를 잡아낼 수 있도록 검증기를 설계하세요.

## 활용하기 (Use It)

도메인별 2026년 게임-RL(game-RL) 지형:

| 도메인 (Domain) | 지배적인 방법론 (Dominant method) |
|--------|-----------------|
| 2인 제로섬 보드 게임 (바둑, 체스, 쇼기) | AlphaZero / MuZero / KataGo |
| 불완전 정보 카드 게임 (포커) | CFR + 딥러닝 (DeepStack, Libratus, Pluribus) |
| Atari / 픽셀 게임 | Muesli / MuZero / IMPALA-PPO |
| 대규모 멀티플레이어 전략 (Dota, StarCraft) | PPO + 셀프 플레이(self-play) + 리그(league) (OpenAI Five, AlphaStar) |
| LLM 수학/코드 추론 | GRPO (DeepSeek-R1, Qwen-RL, 오픈 복제 모델들) |
| LLM 정렬 (Alignment) | DPO / RLHF-PPO (GRPO가 아님; 검증기(verifier)가 검증 가능한 것이 아닌 선호도 기반임) |
| 로보틱스 (Robotics) | PPO + DR (게임-RL은 아니지만, 동일한 정책 경사(policy-gradient) 도구를 사용함) |
| 조합 최적화 문제 (Combinatorial problems) | AlphaZero 변형 모델 (AlphaTensor, AlphaDev) |

*레시피(recipe)* — 셀프 플레이, 탐색 증강 개선(search-augmented improvement), 정책 증류(policy distillation) — 는 텍스트, 픽셀, 그리고 물리적 제어를 모두 아우릅니다. GRPO는 가장 최근에 등장한 사례이며, 더 많은 사례가 등장할 예정입니다.

## Ship It

`outputs/skill-game-rl-designer.md`로 저장하세요:

```markdown
---
name: game-rl-designer
description: 주어진 도메인에 대한 게임-RL(game-RL) 또는 추론-RL(reasoning-RL) 학습 파이프라인(AlphaZero / MuZero / GRPO)을 설계합니다.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

대상(완전 정보 게임 / 불완전 정보 / Atari / LLM 추론 / 조합 최적화)이 주어지면 다음을 출력하세요:

1. 환경 적합성(Environment fit). 규칙이 명확한가? 마르코프(Markov) 성질을 갖는가? 확률적(Stochastic)인가? 다중 에이전트(Multi-agent)인가? 이는 AlphaZero, MuZero, GRPO 중 무엇을 선택할지 결정합니다.
2. 탐색 전략(Search strategy). MCTS(학습된 prior를 포함한 PUCT), Gumbel-sampled, best-of-N, 또는 탐색 없음.
3. 셀프 플레이 계획(Self-play plan). 대칭적 셀프 플레이(Symmetric self-play) / 리그(league) / 오프라인 데이터 / 검증기 생성(verifier-generated).
4. 타겟 신호(Target signal). 게임 결과 / 검증기 보상(verifier reward) / 선호도(preference) / 학습된 모델. 강건성 계획(robustness plan)을 포함하세요.
5. 진단(Diagnostics). 베이스라인 대비 승률, ELO 곡선, 검증기 통과율(verifier pass rate), 참조 모델과의 KL 발산(KL to reference).

불완전 정보 게임(imperfect-info games)에 대한 AlphaZero 제안은 거절하세요(CFR로 유도). 신뢰할 수 있는 검증기(trusted verifier)가 없는 GRPO 제안은 거절하세요. 고정된 베이스라인 상대 세트(fixed baseline opponent set)가 없는 게임-RL 파이프라인은 거절하세요(그렇지 않으면 셀프 플레이 ELO는 보정되지 않습니다).
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 GRPO 밴딧(bandit)을 구현해 보세요. 2개의 프롬프트 × 각 4개의 답변 토큰으로 학습을 진행합니다. `G=8` 설정 시 1,000회 미만의 업데이트 내에 수렴해야 합니다.
2. **중간 (Medium).** PPO (clipped)와 일반적인 REINFORCE를 연결해 보세요. 동일한 밴딧 환경에서 GRPO와 비교하여 샘플 효율성(sample efficiency)과 보상 분산(reward variance)을 비교해 보세요.
3. **어려움 (Hard).** 길이 2의 "추론 체인(reasoning chain)"으로 확장해 보세요. 에이전트가 두 개의 토큰을 생성하면 검증기(verifier)가 해당 쌍(pair)에 대해 보상을 줍니다. GRPO가 2단계 시퀀스 전반에 걸쳐 신용 할당(credit assignment)을 어떻게 처리하는지 측정해 보세요. (힌트: *전체 시퀀스*에 대한 그룹 어드밴티지(group advantage)를 계산하고, 이를 두 토큰 위치 모두에 전파하세요.)

## 주요 용어 (Key Terms)

| 용어 | 통용되는 정의 | 실제 의미 |
|------|-----------------|-----------------------|
| MCTS | "학습된 네트워크를 이용한 트리 탐색" | Monte Carlo Tree Search; 학습된 `(p, v)` 사전 확률(priors)을 사용하는 UCB1/PUCT 선택 방식. |
| AlphaZero | "Self-play + MCTS" | MCTS 방문 횟수 및 게임 결과와 일치하도록 학습된 정책-가치 네트워크(Policy-value net). |
| MuZero | "학습된 모델 기반의 AlphaZero" | 동일한 루프를 따르되, 학습된 역학(dynamics)을 통해 잠재 공간(latent space)에서 수행됨. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; 그룹 평균 베이스라인(group-mean baseline)과 KL을 사용하는 REINFORCE 방식. |
| PUCT | "AlphaZero의 UCB" | `Q + c · p · √N / (1 + N_a)` — 가치 추정치와 사전 확률(prior) 사이의 균형을 맞춤. |
| Self-play | "에이전트 vs 과거의 자신" | 제로섬(zero-sum) 게임의 표준; 대칭적인 학습 신호 제공. |
| League play | "인구 기반 Self-play" | 과거, 현재, 그리고 약점 공략자(exploiters)를 대전 상대로 샘플링함. |
| Verifier reward | "검증 가능한 RL" | 결정론적 체크기(테스트 통과, 정답 일치 여부)로부터 보상이 발생함. |
| Process reward | "PRM" | 최종 정답뿐만 아니라 각 추론 단계(reasoning step)에 점수를 부여함. |

## 추가 학습 자료 (Further Reading)

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — GRPO와 그룹 상대적 베이스라인(group-relative baseline)을 소개한 논문입니다.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 전체 4단계 R1 레시피와 R1-Zero 절제 연구(ablation)를 포함합니다.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 대규모 CFR(Counterfactual Regret Minimization)과 딥러닝의 결합을 다룹니다.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 이 모든 흐름의 시작이 된 논문입니다.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 커스텀 보상 함수를 사용하여 GRPO를 적용할 때 참고할 수 있는 프로덕션 레퍼런스입니다.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 다양한 규모에서 R1 레시피를 오픈 소스로 재현한 프로젝트입니다.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — R1이 LLM 규모에서 구현해낸 셀프 플레이(self-play), 탐색(search), "설계된 보상(designed reward)"에 대한 교과서적 프레임워크를 제공합니다.
