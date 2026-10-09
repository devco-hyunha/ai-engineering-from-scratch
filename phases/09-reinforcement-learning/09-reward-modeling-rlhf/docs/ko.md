# 보상 모델링 및 RLHF

> 사람은 "좋은 어시스턴트 응답"에 대한 보상 함수를 직접 작성할 수 없지만, 두 응답을 비교하여 더 나은 것을 선택할 수 있습니다. 이러한 비교에 보상 모델을 적합(fit)한 후, 언어 모델을 보상 모델에 대해 RL(강화 학습)합니다. Christiano 2017. InstructGPT 2022. GPT-3를 ChatGPT로 만든 레시피입니다. 2026년에는 대부분 DPO로 대체되고 있지만, 개념적 모델은 그대로 유지됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 05강 (감성 분석), 9단계 · 08강 (PPO)
**시간:** 약 45분

## 문제점

다음 토큰 예측 목적 함수로 언어 모델을 학습했습니다. 문법적으로 올바른 영어를 작성합니다. 또한 거짓말을 하고, 장황하게 말하며, 거절을 거부합니다. 추가 사전 학습으로는 이를 고칠 수 없습니다. 웹 텍스트가 문제이지, 해결책이 아닙니다.

"지시문 X에 대해 응답 A가 응답 B보다 낫다"라고 말하는 *스칼라 보상*을 원합니다. 그 보상 함수를 직접 작성하는 것은 불가능합니다. "유용성"은 토큰에 대한 닫힌 형식(closed-form) 표현이 아닙니다. 하지만 사람은 두 출력물을 비교하여 선호도를 표시할 수 있습니다. 이는 대규모로 수집하기 저렴합니다.

RLHF (Christiano et al. 2017; Ouyang et al. 2022)는 선호도를 보상 모델로 변환한 후, 그 보상 모델에 대해 PPO로 언어 모델을 최적화합니다. 세 단계로 진행됩니다: SFT → RM → PPO. 2023–2025년 ChatGPT, Claude, Gemini 및 모든 정렬된 LLM을 출시한 레시피입니다.

2026년에는 PPO 단계가 대부분 DPO (10단계 · 08강)로 대체되었습니다. 비용이 저렴하고 정렬 튜닝에 거의 동일한 성능을 내기 때문입니다. 하지만 *보상 모델* 부분은 여전히 모든 Best-of-N 샘플러, 모든 검증 가능한 보상 기반 RL 파이프라인, 그리고 모든 프로세스 보상 모델을 사용하는 추론 모델의 기반이 됩니다. RLHF를 이해하면 전체 정렬 스택을 이해하게 됩니다.

## 개념

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**1단계: 지도 미세 조정 (SFT).** 사전 학습된 기본 모델에서 시작합니다. 목표 행동에 대한 사람이 작성한 시연(demonstration)(지시문 따르기 응답, 유용한 답변 등)으로 미세 조정합니다. 결과: *좋은 행동으로 편향된* 모델 `π_SFT`이 생성되지만, 여전히 무한한 행동 공간을 가집니다.

**2단계: 보상 모델 학습.**

- 프롬프트 `x`에 대한 응답 쌍 `(y_+, y_-)`을 수집하고, 인간이 "y_+가 y_-보다 선호됨"이라고 라벨링합니다.
- 보상 모델 `R_φ(x, y)`을 학습하여 `y_+`에 더 높은 점수를 부여합니다.
- 손실: **Bradley-Terry 쌍별 로지스틱**:

`L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

σ는 시그모이드입니다. 보상 차이는 선호도의 로그 오즈(log-odds)를 의미합니다. BT는 1952년(Bradley-Terry)부터 표준이었으며, 현대 RLHF에서 지배적인 선택입니다.

- `R_φ`은 일반적으로 SFT 모델에서 스칼라 헤드(scalar head)를 추가하여 초기화합니다. 동일한 트랜스포머 백본을 사용하며, 단일 선형 레이어가 보상을 출력합니다.

**3단계: KL 페널티가 포함된 RM에 대한 PPO.**

- 학습 가능한 정책 `π_θ`을 `π_SFT`에서 초기화합니다. 동결된 *참조(reference)* `π_ref = π_SFT`를 유지합니다.
- 응답 `y`의 끝에 보상:

`r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

KL 페널티는 `π_θ`이 `π_SFT`에서 임의로 벗어나는 것을 방지합니다. 이는 *정규화자(regularizer)*이며, 하드 신뢰 영역(hard trust region)이 아닙니다. `β`는 일반적으로 `0.01`-`0.05`입니다.
- 이 보상으로 PPO (08강)를 실행합니다. 이점(advantages)은 토큰 단위 궤적에서 계산되지만, RM은 전체 응답만 점수화합니다.

**KL이 왜 필요할까요?** KL이 없으면 PPO는 보상 해킹(reward-hacking) 전략을 쉽게 찾아냅니다. RM은 분포 내(in-distribution) 완성본으로만 학습되었습니다. 분포 외(out-of-distribution) 응답은 인간이 작성한 어떤 응답보다 높은 점수를 받을 수 있습니다. KL은 `π_θ`이 RM이 학습된 매니폴드(manifold) 근처에 머무르게 합니다. 이는 RLHF에서 가장 중요한 조절 변수(knob)입니다.

**2026년 현황:**

- **DPO** (Rafailov 2023): 폐형(closed-form) 대수적 접근으로 2단계+3단계를 선호 데이터에 대한 단일 지도 학습 손실로 통합합니다. RM이 없고, PPO도 없습니다. 연산량의 일부로 정렬 벤치마크에서 동일한 품질을 달성합니다. 10단계 · 08강에서 다룹니다.
- **GRPO** (DeepSeek 2024–2025): 크리틱(critic) 대신 그룹 상대적 기준선(group-relative baseline)을 사용하는 PPO이며, 인간이 학습한 RM 대신 *검증자(verifier)* (코드 실행 / 수학 답 일치)에서 보상을 얻습니다. 추론 모델에서 지배적입니다. 9단계 · 12강에서 다룹니다.
- **프로세스 보상 모델 (PRM):** 부분 해법(각 추론 단계)에 점수를 부여하며, 추론을 위한 RLHF 및 GRPO 변형 모두에서 사용됩니다.
- **헌법 AI / RLAIF:** 정렬된 LLM을 사용하여 인간 대신 선호도를 생성합니다. 선호도 예산을 확장합니다.

```figure
reward-model
```

## 구현하기

이 강의는 문자열로 표현된 작은 합성 "프롬프트"와 "응답"을 사용합니다. RM은 토큰 주머니(bag-of-tokens) 표현에 대한 선형 스코어입니다. 실제 LLM은 없으며, 파이프라인의 *형태*가 중요하지 규모는 중요하지 않습니다. `code/main.py`을 참조하세요.

### 1단계: 합성 선호 데이터

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

실제 RLHF에서는 인간 라벨러가 이를 대체합니다. 형태 — `(prompt, preferred_response, rejected_response)` —는 동일합니다.

### 2단계: Bradley-Terry 보상 모델

선형 점수: `R(x, y) = w · bag(y)`. BT 쌍별 로그 손실을 최소화하도록 학습합니다:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

수백 번의 업데이트 후, `w`은 좋은 단어 토큰에 양의 가중치를, 나쁜 단어 토큰에 음의 가중치를 할당합니다.

### 3단계: RM 위에 PPO 유사 정책 적용

우리의 장난감 정책은 어휘에서 단일 토큰을 생성합니다. RM 아래에서 토큰을 점수화하고, `log π_θ(token | prompt)`을 계산하며, 참조 모델에 대한 KL 페널티를 추가하고, 클리핑된 PPO 대리 함수를 적용합니다.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 보상을 리턴으로 간주하여 theta에 대한 ppo 스타일 업데이트
    ...
```

### 4단계: KL 모니터링

매 업데이트마다 평균 `KL(π_θ || π_ref)`을 추적합니다. `~5-10`을 넘어서면 정책이 `π_SFT`에서 크게 벗어났다는 의미이며, `β`이 상승하거나 보상 해킹이 시작되고 있습니다. 이는 실제 RLHF에서 가장 중요한 진단 지표입니다.

### 5단계: TRL을 사용한 프로덕션 레시피

장난감 파이프라인을 이해했다면, 실제 라이브러리 사용자가 작성하는 것과 동일한 루프가 여기 있습니다. Hugging Face의 [TRL](https://huggingface.co/docs/trl)는 참조 구현입니다. 2단계에는 `RewardTrainer`을, 3단계에는 (KL-to-reference가 내장된) `PPOTrainer`을 사용합니다.

```python
# 2단계: 쌍별 선호도로부터 보상 모델 생성
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 데이터셋 행: {"prompt", "chosen", "rejected"} — Bradley-Terry 형식
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 3단계: SFT 참조 모델에 대한 KL 페널티가 있는 RM에 대한 PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 동결(frozen)

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # 통계에는 mean_kl, clip_frac, value_loss가 포함됩니다. 이는 세 가지 PPO 진단 지표입니다.
```

라이브러리가 대신 처리하는 세 가지가 있습니다. `adap_kl_ctrl=True`는 적응형 β 스케줄을 구현합니다. 관측된 KL이 `target_kl`를 초과하면 β가 두 배가 되고, 절반 미만이면 β가 절반으로 줄어듭니다. 참조 모델은 관례에 따라 고정(frozen)되어 있으며, `policy`와 실수로 매개변수를 공유해서는 안 됩니다. 그리고 가치 헤드는 정책과 동일한 백본 위에 위치합니다(`AutoModelForCausalLMWithValueHead`는 스칼라 MLP 헤드를 부착합니다). 이 때문에 TRL은 `policy/kl`와 `value/loss`를 별도로 보고합니다.

## 함정

- **과도한 최적화 / 보상 해킹.** RM은 불완전합니다. `π_θ`는 점수가 높지만 실제로는 나쁜 적대적 완성(adversarial completions)을 찾아냅니다. 증상: 보상이 무한히 상승하는 동안 인간 평가 점수는 정체되거나 하락합니다. 해결책: 일찍 멈추고, `β`를 높이고, RM 학습 데이터를 확장하세요.
- **길이 해킹.** 유용한 응답으로 학습된 RM은 종종 길이를 암묵적으로 보상합니다. 정책은 응답을 패딩하는 것을 학습합니다. 해결책: 길이 정규화된 보상, 또는 길이 인식 RM을 사용하는 RLAIF.
- **너무 작은 RM.** RM은 정책만큼 크거나 더 커야 합니다. 작은 RM은 정책의 출력을 충실히 점수 매길 수 없습니다.
- **KL 튜닝.** β가 너무 낮으면 드리프트와 보상 해킹이 발생합니다. β가 너무 높으면 정책이 거의 변하지 않습니다. 표준적인 방법은 고정된 KL을 단계별로 목표로 하는 *적응형* β를 사용하는 것입니다.
- **선호 데이터 노이즈.** 인간 레이블의 약 30%는 노이즈가 있거나 모호합니다. 일치로 필터링된 데이터로 RM을 학습하거나 BT에 온도를 적용하여 보정하세요.
- **오프 정책 문제.** PPO 데이터는 첫 에포크 이후 약간 오프 정책(off-policy)이 됩니다. 08강에서처럼 클립 비율을 모니터링하세요.

## 사용하기

2026년의 RLHF는 계층화되어 있습니다:

| 계층 | 목표 | 방법 |
|-------|--------|--------|
| 지시문 따르기, 유용성, 무해성 | 정렬 | RLHF-PPO보다 DPO (10단계 · 08강)가 선호됩니다. |
| 추론 정확성 (수학, 코드) | 능력 | 검증자 보상과 함께 GRPO (9단계 · 12강). |
| 장기 다단계 작업 | 에이전트 | 단계별 프로세스 보상 모델과 함께 PPO / GRPO. |
| 안전 / 거절 행동 | 안전 | 별도의 안전 RM을 사용하는 RLHF-PPO, 또는 Constitutional AI. |
| 추론 시 Best-of-N | 빠른 정렬 | 디코딩 시 RM을 사용; 정책 학습이 필요 없습니다. |
| 보상 증류 | 추론 연산량 | 동결된 LM 위에 작은 "보상 헤드"를 학습합니다. |

RLHF는 2022–2024년 동안 *가장* 중요한 방법이었습니다. 2026년에는 생산 환경의 정렬 파이프라인이 DPO 우선이며, RM 집약적이거나 안전이 중요한 단계에서만 PPO를 사용합니다.

## 출시하기

`outputs/skill-rlhf-architect.md`로 저장하세요:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 500개의 합성 선호 쌍에 대해 Bradley-Terry 보상 모델을 학습하세요. 유지된 100개 쌍에 대한 쌍별 정확도를 측정하세요. 90%를 초과해야 합니다.
2. **중간.** `β ∈ {0.0, 0.1, 1.0}`로 장난감 PPO-RLHF 루프를 실행하세요. 각 실행에 대해 업데이트 동안의 RM 점수 대 참조 모델 대비 KL을 플롯하세요. 어떤 실행이 보상 해킹을 합니까?
3. **어려움.** 동일한 선호 데이터에 DPO (폐형 선호 가능성 손실)를 구현하고, 사용된 연산량과 달성된 최종 RM 점수에서 RLHF-PPO 파이프라인과 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| RLHF | "정렬 RL" | 3단계 SFT + RM + PPO 파이프라인 (Christiano 2017, Ouyang 2022). |
| 보상 모델 (RM) | "채점 네트워크" | Bradley-Terry를 통해 쌍별 선호에 맞춰 학습된 스칼라 함수. |
| Bradley-Terry | "쌍별 로지스틱 손실" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; 표준 RM 목적 함수. |
| KL 페널티 | "참조 모델 근처에 머무르기" | 보상에서의 `β · KL(π_θ \|\| π_ref)`; 보상 해킹 방지 정규화자. |
| 보상 해킹 | "굿하트의 법칙" | 정책이 RM 결함을 이용합니다; 증상: 보상 상승, 인간 평가 평평. |
| RLAIF | "AI 라벨 선호" | 라벨이 인간 대신 다른 LM에서 나오는 RLHF. |
| PRM | "과정 보상 모델" | 부분 추론 단계를 채점합니다; 추론 파이프라인에서 사용됩니다. |
| Constitutional AI | "Anthropic의 방법" | 명시적 규칙에 의해 안내되는 AI 생성 선호. |

## 추가 읽기

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHF를 시작한 논문.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 뒤의 레시피.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 요약에 대한 초기 RLHF.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; 2026년 RLHF 이후의 기본값.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 및 자기 비판 루프.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 논문.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — `RewardTrainer` 및 `PPOTrainer`의 프로덕션 환경입니다. 적응형 KL 및 가치 헤드 세부 사항에 대해 트레이너 소스를 읽어 보세요.
- Lambert, Castricato, von Werra, Havrilla의 [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) — 다이어그램과 함께 3단계 파이프라인을 상세히 설명하는 표준 가이드입니다.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 라이브러리입니다. `examples/`에는 Llama, Mistral, Qwen용 엔드투엔드 RLHF 스크립트가 포함되어 있습니다.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 보상 가설 관점입니다. 보상 해킹에 대해 고려하는 데 필수적인 선수 요건입니다.
