# 게임에서의 RL — AlphaZero, MuZero, 그리고 LLM 추론 시대

> 1992년: TD-Gammon이 순수 TD로 인간 챔피언을 백gammon에서 이겼습니다. 2016년: AlphaGo가 이세돌을 이겼습니다. 2017년: AlphaZero가 체스, 쇼기, 바둑을 처음부터 장악했습니다. 2024년: DeepSeek-R1은 GRPO가 PPO를 대체한 동일한 레시피가 추론에도 작동함을 증명했습니다. 게임은 이 단계의 모든 돌파구를 이끄는 벤치마크입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 05강 (DQN), 9단계 · 08강 (PPO), 9단계 · 09강 (RLHF), 9단계 · 10강 (MARL)
**시간:** 약 120분

## 문제점

게임은 RL이 원하는 모든 것을 갖추고 있습니다. 명확한 보상(승/패). 무한한 에피소드(자기 대국으로 리셋). 완벽한 시뮬레이션(게임 자체가 시뮬레이터). 이산적이거나 작은 연속 행동 공간. 적대적 강건성을 강제하는 다중 에이전트 구조.

그리고 게임은 모든 주요 RL 돌파구가 테스트된 방식입니다. TD-Gammon(백gammon, 1992). Atari-DQN(2013). AlphaGo(2016). AlphaZero(2017). OpenAI Five(Dota 2, 2019). AlphaStar(StarCraft II, 2019). MuZero(학습된 모델, 2019). AlphaTensor(행렬 곱셈, 2022). AlphaDev(정렬 알고리즘, 2023). DeepSeek-R1(수학 추론, 2025) — 게임 RL 기술이 텍스트에도 작동함을 보여주는 최신 시연입니다.

이 캡스톤은 세 가지 주요 아키텍처인 AlphaZero, MuZero, GRPO를 **자기 대국 + 탐색 + 정책 개선**이라는 단일 통합 렌즈를 통해 조사합니다. 각각이 이전 것을 일반화하며, 특히 GRPO는 AlphaZero의 레시피를 LLM 추론에 적용한 것으로, 토큰을 행동으로, 수학적 검증을 승리 신호로 사용합니다.

## 개념

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**통합 루프.**

```
while True:
    trajectory = self_play(current_policy, search)     # 자기와 게임 진행
    policy_target = search.improved_policy(trajectory) # 탐색이 원시 정책을 개선
    policy_net.update(policy_target, value_target)     # 탐색 출력에 대한 지도 학습
```

**AlphaZero (2017).** Silver et al. 알려진 규칙을 가진 게임(체스, 쇼기, 바둑)이 주어지면:

- 정책-가치 네트워크: 하나의 타워 `f_θ(s) → (p, v)`. `p`은 합법적 수에 대한 사전 확률입니다. `v`는 예상 게임 결과입니다.
- Monte Carlo Tree Search (MCTS): 각 수에서 가능한 연속 수의 트리를 확장합니다. `(p, v)`을 사전 확률 + 부트스트랩으로 사용하세요. UCB(PUCT)로 노드를 선택하세요: `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- 자기 대국: 에이전트 대 에이전트로 게임을 진행합니다. 수 `t`에서 MCTS 방문 분포 `π_t`가 정책 학습 목표가 됩니다.
- 손실: `L = (v - z)² - π · log p + c · ||θ||²`. `z`은 게임 결과(+1 / 0 / -1)입니다.

인간 지식은 제로, 수작업 휴리스틱은 제로입니다. 단일 레시피로 각각 수천만 번의 자기 대국 후 체스, 쇼기, 바둑을 마스터했습니다.

**MuZero (2019).** Schrittwieser 등. 규칙이 알려져야 한다는 요구 사항을 제거했습니다.

- 고정된 환경 대신 *잠재 동역학 모델* `(h, g, f)`을 학습합니다:
  - `h(s)`: 관측을 잠재 상태로 인코딩합니다.
  - `g(s_latent, a)`: 다음 잠재 상태와 보상을 예측합니다.
  - `f(s_latent)`: 정책 사전 분포와 가치를 예측합니다.
- MCTS는 *학습된 잠재 공간*에서 실행됩니다. 동일한 탐색, 동일한 학습 루프입니다.
- 바둑, 체스, 쇼기 *그리고* Atari에서 작동합니다. 하나의 알고리즘, 규칙 지식 없음.

**Stochastic MuZero (2022).** 확률적 동역학과 기회 노드를 추가하여 바커가몬류 게임으로 확장했습니다.

**Muesli, Gumbel MuZero (2022-2024).** 샘플 효율성과 결정적 탐색에 대한 개선 사항입니다.

**GRPO (2024-2025).** DeepSeek-R1 레시피. AlphaZero 형태의 루프를 언어 모델 추론에 적용했습니다:

- "게임": 수학 / 코딩 / 추론 문제를 답합니다. "승리" = 검증자(테스트 케이스 통과, 수치 답 일치)가 1을 반환합니다.
- 정책: LLM. 행동: 토큰. 상태: 프롬프트 + 현재까지의 응답.
- 비판자(PPO 스타일 V_φ)가 없습니다. 대신 각 프롬프트에 대해 정책에서 `G`개의 완성을 샘플링합니다. 각각에 대해 보상을 계산합니다. **그룹 상대적 이점** `A_i = (r_i - mean_r) / std_r`을 REINFORCE 스타일 업데이트의 신호로 사용합니다.
- 드리프트를 방지하기 위해 참조 정책에 대한 KL 페널티를 적용합니다(RLHF와 유사).
- 전체 손실:

`L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

보상 모델 없음, 비판자 없음, MCTS 없음. 그룹 상대적 기준선이 이 세 가지를 모두 대체합니다. 연산량의 일부로 추론 벤치마크에서 PPO-RLHF 품질과 일치하거나 이를 초과합니다.

**R1 레시피 전체.** DeepSeek-R1 (DeepSeek 2025)은 한 논문 안에 두 모델이 있습니다:

- **R1-Zero.** DeepSeek-V3 기본 모델에서 시작합니다. SFT는 없습니다. 두 가지 보상 구성 요소와 함께 GRPO를 직접 적용합니다: *정확도 보상* (규칙 기반 — 최종 답이 올바른 숫자로 파싱되었는지, 코드가 단위 테스트를 통과했는지) 및 *형식 보상* (완료된 내용이 `<think>…</think>` 태그 안에 사고의 연쇄(CoT)를 감싸고 있는지). 수천 단계에 걸쳐 평균 응답 길이가 약 100에서 약 10,000 토큰으로 증가하고 수학 벤치마크 점수가 o1-preview 수준에 근접합니다. 모델은 처음부터 추론하는 법을 배웁니다. 단점: 사고의 연쇄(CoT)가 종종 읽기 어렵고, 언어가 혼합되며, 스타일적 다듬음이 부족합니다.
- **R1.** 4단계 파이프라인으로 R1-Zero의 가독성 문제를 해결합니다:
  1. **콜드 스타트 SFT.** 깔끔한 형식의 몇 천 개의 긴 사고의 연쇄(CoT) 시연 데이터를 수집합니다. 기본 모델을 이를 사용하여 지도 미세 조정(Fine-tuning)합니다. 이는 읽기 쉬운 시작점을 제공합니다.
  2. **추론 중심 GRPO.** 정확도+형식 보상과 언어 전환(code-switching)을 방지하는 *언어 일관성* 보상을 포함하여 GRPO를 적용합니다.
  3. **거부 샘플링 + SFT 2차.** RL 체크포인트에서 약 60만 개의 추론 궤적을 샘플링하고, 최종 답이 정확하고 사고의 연쇄(CoT)가 읽기 쉬운 것만 유지하며, 약 20만 개의 비추론 SFT 예제(글쓰기, QA, 자기 인식)와 결합합니다. 기본 모델을 다시 미세 조정합니다.
  4. **전 스펙트럼 GRPO.** 추론(규칙 기반 보상)과 일반 정렬(유용성/무해성 선호 기반 보상)을 모두 포함하는 RL 라운드를 한 번 더 수행합니다.

결과는 오픈 가중치에서 AIME와 MATH-500에 대해 o01강 일치하며, 증류(distillation)할 수 있을 만큼 충분히 작습니다. 같은 논문은 R1의 추론 궤적에 대해 SFT를 수행하여 6개의 증류된 밀집 모델(Qwen-1.5B부터 Llama-70B까지)도 공개합니다 — 학생 모델에서는 RL이 없습니다. 강력한 RL 교사로부터의 증류(distillation)는 학생 모델의 규모에서 처음부터 수행하는 RL을 일관되게 능가합니다.

**추론에 PPO 대신 GRPO를 사용하는 이유.** DeepSeekMath 논문(2024년 2월)의 세 가지 이유: (1) 학습할 가치 네트워크가 없어 메모리가 절반으로 줄어듭니다; (2) 그룹 기준선이 추론 작업이 생성하는 희소한 궤적 종료 보상을 자연스럽게 처리합니다; (3) 프롬프트별 정규화(normalization)는 난이도가 극도로 다른 문제들 간에 이점(advantage)을 비교 가능하게 만들며, 이는 PPO의 단일 비평가(critic)가 할 수 없는 부분입니다.

**검색 없는 방식 vs 검색 기반 방식.** 게임은 분기되었습니다:

- *긴Horizon을 가진 완전 정보 게임* (바둑, 체스): 여전히 탐색 기반입니다. AlphaZero / MuZero가 지배합니다.
- *LLM 추론*: 프로덕션 환경에서는 아직 MCTS가 사용되지 않습니다. 전체 롤아웃에 대한 GRPO, 추론 컴퓨팅을 위한 best-of-N이 사용됩니다. 프로세스 보상 모델 (PRM)은 단계 수준 탐색이 다시 추가될 가능성을 시사합니다.

```figure
f3-selfplay-ladder
```

## 구현하기

`code/main.py`의 코드는 **축소된 GRPO**를 구현합니다 — 여러 그룹의 샘플을 가진 밴디트입니다. 알고리즘은 LLM에서의 경우와 동일하며, 정책과 환경만 더 단순합니다. 이는 *손실*과 *그룹 상대적 이점*을 가르치며, 이는 2025년의 혁신입니다.

### 1단계: 작은 검증자 환경

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

실제 GRPO에서는 검증자가 단위 테스트를 실행하거나 수학 등식을 확인합니다.

### 2단계: 정책: 프롬프트당 K개의 답변 토큰에 대한 소프트맥스

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

프롬프트에 조건을 부여한 LLM의 최종 레이어 출력과 동일합니다.

### 3단계: 그룹 샘플링 및 그룹 상대적 이점

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
    # KL 페널티: theta를 참조 모델 쪽으로 당깁니다.
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

그룹 상대적 이점은 2024년 DeepSeek의 트릭입니다. 크리틱이 필요하지 않습니다. "기준선"은 그룹 평균이며, 정규화는 그룹 표준 편차를 사용합니다.

### 4단계: REINFORCE 기준선 (가치 없음)과 비교

동일한 설정, 동일한 컴퓨팅, 순수 REINFORCE. GRPO는 더 빠르고 안정적으로 수렴합니다.

### 5단계: 엔트로피 및 KL 관찰

RLHF와 동일한 진단: 참조 모델에 대한 평균 KL, 정책 엔트로피, 시간에 따른 보상. 이러한 지표가 안정화되면 학습이 완료됩니다.

## 함정

- **검증자 조작을 통한 보상 해킹.** GRPO는 RLHF의 위험을 상속받습니다: 검증자가 잘못되었거나 악용될 여지가 있으면 LLM이 그 취약점을 찾아냅니다. 견고한 검증자 (다수의 테스트 케이스, 형식적 증명)가 중요합니다.
- **그룹 크기가 너무 작음.** 그룹 기준선의 분산은 `1/√G`에 비례합니다. `G = 4` 미만에서는 이점 신호가 노이즈가 많습니다. 표준적인 선택은 `G = 8`에서 `64`입니다.
- **길이 편향.** 길이가 다른 LLM 완성물은 서로 다른 로그 확률을 가집니다. 토큰 수로 정규화하거나, 시퀀스 수준 로그 확률을 사용하거나, 최대 길이로 잘라내세요.
- **순수 자기 대국 사이클.** AlphaZero 스타일 학습은 일반 합 게임(general-sum games)에서 지배 루프에 갇힐 수 있습니다. 다양한 상대 풀(리그 플레이, 10강)로 완화됩니다.
- **탐색-정책 불일치.** AlphaZero는 정책 네트워크가 탐색 출력의 분포를 표현하기 너무 작으면 학습이 정체됩니다. 정책 네트워크가 탐색의 분포를 표현할 수 없다면 학습이 멈추게 됩니다.
- **컴퓨팅 하한.** MuZero / AlphaZero는 대규모 컴퓨팅이 필요합니다. 단일 애블레이션(ablation)은 보통 수백 GPU-hours가 소요됩니다. 학습용으로는 Miniature demos (예: Connect Four에서의 AlphaZero)가 존재합니다.
- **검증자 커버리지.** 버그가 있는 솔루션에 대해 통과하는 단위 테스트는 버그를 강화합니다. 엣지 케이스를 잡을 수 있는 검증자를 설계해 보세요.

## 사용하기

2026년 게임 RL 현황, 도메인별:

| 도메인 | 지배적 방법 |
|--------|-----------------|
| 2인 제로합 보드 게임 (Go, 체스, 쇼기) | AlphaZero / MuZero / KataGo |
| 불완전 정보 카드 게임 (포커) | CFR + 딥러닝 (DeepStack, Libratus, Pluribus) |
| Atari / 픽셀 게임 | Muesli / MuZero / IMPALA-PPO |
| 대규모 멀티플레이어 전략 (Dota, StarCraft) | PPO + 자기 대국 + 리그 (OpenAI Five, AlphaStar) |
| LLM 수학/코드 추론 | GRPO (DeepSeek-R1, Qwen-RL, 오픈 복제) |
| LLM 정렬 | DPO / RLHF-PPO (GRPO 아님; 검증자는 선호도이며 검증 가능하지 않음) |
| 로보틱스 | PPO + DR (게임 RL은 아니지만, 동일한 정책 기울기 도구 사용) |
| 조합 문제 | AlphaZero 변형 (AlphaTensor, AlphaDev) |

*레시피* — 자기 대국, 탐색 증강 개선, 정책 증류 —는 텍스트, 픽셀, 물리적 제어에 걸쳐 있습니다. GRPO는 가장 젊은 사례이며, 더 많은 사례가 나오고 있습니다.

## 출시하기

`outputs/skill-game-rl-designer.md`로 저장:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 GRPO bandit을 구현하세요. 2개 프롬프트 × 각 4개 답변 토큰으로 학습하세요. `G=8`로 1,000 업데이트 미만에서 수렴합니다.
2. **중간.** PPO (clipped)와 vanilla REINFORCE를 연결하세요. 동일한 bandit에서 GRPO와 샘플 효율성 및 보상 분산을 비교하세요.
3. **난이도: 높음.** 길이 2의 "추론 체인"으로 확장해 보세요: 에이전트가 두 개의 토큰을 생성하고 검증기가 이 쌍에 대해 보상합니다. GRPO가 2단계 시퀀스에서 크레딧 할당(credit assignment)을 어떻게 처리하는지 측정해 보세요. (힌트: *전체 시퀀스*에 대한 그룹 어드밴티지를 계산하고, 두 토큰 위치 모두에 전파하세요.)

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| MCTS | "학습된 네트워크를 사용한 트리 탐색" | Monte Carlo Tree Search; 학습된 `(p, v)` 사전 확률(priors)을 사용하여 UCB1/PUCT 선택을 수행합니다. |
| AlphaZero | "셀프 플레이 + MCTS" | MCTS 방문 수와 게임 결과에 맞춰 학습된 정책-가치(policy-value) 네트워크. |
| MuZero | "학습된 모델 기반 AlphaZero" | 학습된 동학(dynamics)을 통해 잠재 공간(latent space)에서 동일한 루프를 수행합니다. |
| GRPO | "크리틱 없는 PPO" | Group Relative Policy Optimization; 그룹 평균 기반선(group-mean baseline)과 KL을 사용하는 REINFORCE. |
| PUCT | "AlphaZero의 UCB" | `Q + c · p · √N / (1 + N_a)` — 가치 추정치와 사전 확률(prior)의 균형을 맞춥니다. |
| Self-play | "에이전트 vs 과거의 나" | 영합 게임(zero-sum)의 표준; 대칭적인 학습 신호. |
| League play | "인구 기반 셀프 플레이" | 과거 에이전트 + 현재 에이전트 + 익스플로이터(exploiters)를 상대로 샘플링합니다. |
| Verifier reward | "검증 가능한 RL" | 보상(reward)이 결정론적 검사기(테스트 통과, 정답 일치)에서 나옵니다. |
| Process reward | "PRM" | 최종 답변뿐만 아니라 각 추론 단계에 점수를 매깁니다. |

## 추가 읽기

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — GRPO와 그룹 상대적 기반선(group-relative baseline)을 도입한 논문입니다.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 전체 4단계 R1 레시피와 R1-Zero Ablation을 포함합니다.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 대규모 CFR + 딥러닝.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 모든 것을 시작한 논문입니다.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 커스텀 보상 함수와 함께 GRPO를 적용하기 위한 생산 환경 참고 자료입니다.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 여러 규모에서 R1 레시피를 오픈 소스로 재현한 것입니다.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — R1이 LLM 규모에서 구현하는 셀프 플레이, 탐색 및 "설계된 보상(designed reward)"에 대한 교과서적 관점입니다.
