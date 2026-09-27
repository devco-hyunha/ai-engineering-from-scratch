# 보상 모델링 및 RLHF (Reward Modeling & RLHF)

> 인간은 "좋은 비서의 응답"에 대한 보상 함수를 직접 작성할 수는 없지만, 두 응답을 비교하여 더 나은 것을 선택할 수는 있습니다. 이러한 비교 데이터를 바탕으로 보상 모델(Reward Model)을 학습시킨 뒤, 이를 활용해 언어 모델을 강화학습(RL) 시킵니다. Christiano 2017. InstructGPT 2022. GPT-3를 ChatGPT로 탈바꿈시킨 레시피입니다. 2026년 현재는 대부분 DPO로 대체되고 있지만, 그 사고 모델(Mental Model)은 여전히 유효합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## 문제점 (The Problem)

당신은 다음 토큰 예측(next-token-prediction) 목표로 언어 모델을 학습시켰습니다. 모델은 문법적으로 올바른 영어를 구사합니다. 하지만 거짓말을 하고, 횡설수설하며, 거절해야 할 상황에서도 거절하지 않습니다. 사전 학습(pretraining)을 더 한다고 해서 이 문제를 해결할 수는 없습니다. 웹 텍스트 자체가 문제이지, 해결책이 아니기 때문입니다.

당신은 "지시 사항 X에 대해 응답 A가 응답 B보다 더 낫다"라고 말해줄 수 있는 *스칼라 보상(scalar reward)*을 원합니다. 이러한 보상 함수를 직접 손으로 작성하는 것은 불가능합니다. "도움이 되는 정도(Helpfulness)"는 토큰에 대한 폐쇄형 수식(closed-form expression)으로 표현될 수 없습니다. 하지만 인간은 두 개의 출력을 비교하고 선호도를 표시할 수 있습니다. 이는 대규모로 수집하기에 비용이 저렴합니다.

RLHF (Christiano et al. 2017; Ouyang et al. 2022)는 이러한 선호도를 보상 모델(reward model)로 변환한 다음, 해당 보상을 바탕으로 PPO를 통해 언어 모델을 최적화합니다. 이는 세 단계로 이루어집니다: SFT → RM → PPO. 이것이 바로 2023~2025년 사이 ChatGPT, Claude, Gemini 및 기타 모든 정렬된 LLM(aligned-LLM)을 탄생시킨 레시피입니다.

2026년에는 PPO 단계가 대부분 DPO (Phase 10 · 08)로 대체되었습니다. DPO는 비용이 더 저렴하면서도 정렬 튜닝(alignment tuning) 성능이 거의 대등하기 때문입니다. 하지만 *보상 모델(reward model)* 요소는 여전히 모든 Best-of-N 샘플러, 모든 검증 가능한 보상 기반 RL(RL-from-verifiable-rewards) 파이프라인, 그리고 프로세스 보상 모델(process reward model)을 사용하는 모든 추론 모델의 근간을 이룹니다. RLHF를 이해하면 정렬 스택(alignment stack) 전체를 이해하게 됩니다.

## 개념 (The Concept)

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**1단계: 지도 미세 조정 (Supervised Fine-Tuning, SFT).** 사전 학습된 베이스 모델(pretrained base model)에서 시작합니다. 목표 행동(지시 이행 응답, 유용한 답변 등)에 대해 사람이 작성한 데모 데이터를 사용하여 미세 조정합니다. 결과물: *좋은 행동을 하도록 편향되어* 있지만, 여전히 제한되지 않은 행동 공간(unbounded action space)을 가진 모델 `π_SFT`가 생성됩니다.

**2단계: 보상 모델 학습 (Reward Model training).**

- 프롬프트 `x`에 대한 응답 쌍 `(y_+, y_-)`을 수집하고, 사람이 "y_+가 y_-보다 선호됨"이라고 라벨을 지정합니다.
- `y_+`에 더 높은 점수를 부여하도록 보상 모델 `R_φ(x, y)`를 학습시킵니다.
- 손실 함수(Loss): **Bradley-Terry 쌍체 로지스틱(Bradley-Terry pairwise logistic)**을 사용합니다:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  여기서 σ는 시그모이드(sigmoid) 함수입니다. 보상의 차이는 선호도의 로그 오즈(log-odds)를 의미합니다. BT 모델은 1952년(Bradley-Terry)부터 표준이었으며, 현대 RLHF에서 지배적인 선택지입니다.

- `R_φ`는 일반적으로 SFT 모델 위에 스칼라 헤드(scalar head)를 얹어 초기화합니다. 트랜스포머 백본(backbone)은 동일하며, 단일 선형 레이어(`linear layer`)가 보상을 출력합니다.

**3단계: KL 페널티를 적용한 RM 대상 PPO (PPO against the RM with KL penalty).**

- 학습 가능한 정책(trainable policy) `π_θ`를 `π_SFT`로부터 초기화합니다. 고정된 *참조 모델(reference model)* `π_ref = π_SFT`를 유지합니다.
- 응답 `y`의 끝에서 계산되는 보상:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 페널티는 `π_θ`가 `π_SFT`로부터 임의로 벗어나는 것을 방지합니다. 이는 하드한 신뢰 영역(hard trust region)이 아니라 *정규화 도구(regularizer)* 역할을 합니다. `β` 값은 통상 `0.01`-`0.05` 사이입니다.
- 이 보상을 사용하여 PPO(Lesson 08)를 실행합니다. 어드밴티지(Advantages)는 토큰 수준의 궤적(token-level trajectory)에서 계산되지만, RM 점수는 전체 응답에 대해서만 부여됩니다.

**왜 KL을 사용하는가?** KL이 없다면, PPO는 기꺼이 보상 해킹(reward-hacking) 전략을 찾아낼 것입니다. RM은 분포 내(in-distribution) 완성형 데이터로만 학습되었기 때문입니다. 분포 외(out-of-distribution) 응답이 사람이 작성한 그 어떤 응답보다 높은 점수를 받을 수도 있습니다. KL은 `π_θ`를 RM이 학습된 매니폴드(manifold) 근처에 머물게 합니다. 이는 RLHF에서 가장 중요한 조절 장치(knob)입니다.

**2026년 현황:**

- **DPO** (Rafailov 2023): 폐쇄형 대수식(closed-form algebra)을 통해 2단계와 3단계를 선호도 데이터에 대한 단일 지도 학습 손실로 통합합니다. RM도, PPO도 필요하지 않습니다. 훨씬 적은 연산량으로 정렬 벤치마크에서 동일한 품질을 보여줍니다. Phase 10 · 08에서 다룹니다.
- **GRPO** (DeepSeek 2024–2025): 크리틱(critic) 대신 그룹 상대적 베이스라인(group-relative baseline)을 사용하는 PPO 방식이며, 사람이 학습시킨 RM 대신 *검증기(verifier)* (코드 실행 결과 / 수학 정답 일치 여부)로부터 보상을 받습니다. 추론 모델(reasoning models)에서 지배적입니다. Phase 9 · 12에서 다룹니다.
- **과정 보상 모델 (Process Reward Models, PRMs):** 부분적인 해결 과정(각 추론 단계)에 점수를 부여하며, 추론을 위한 RLHF 및 GRPO 변형 모델 모두에서 사용됩니다.
- **Constitutional AI / RLAIF:** 사람 대신 정렬된 LLM을 사용하여 선호도를 생성합니다. 선호도 데이터 예산(preference budget)을 확장할 수 있습니다.

```figure
reward-model
```

## 구현하기 (Build It)

이 레슨에서는 문자열로 표현된 아주 작은 합성 "프롬프트(prompts)"와 "응답(responses)"을 사용합니다. RM(Reward Model)은 토큰 주머니(bag-of-tokens) 표현에 대한 선형 스코어러(linear scorer)입니다. 실제 LLM을 사용하지는 않으며, 규모가 아닌 파이프라인의 *구조(shape)*가 중요합니다. `code/main.py`를 참조해 보세요.

### 1단계: 합성 선호 데이터 (synthetic preference data)

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

실제 RLHF에서는 이 과정이 인간 레이블러(human labelers)로 대체됩니다. 데이터의 형태 — `(prompt, preferred_response, rejected_response)` — 는 동일합니다.

### 2단계: Bradley-Terry 보상 모델 (Bradley-Terry reward model)

선형 점수(Linear score): `R(x, y) = w · bag(y)`. BT 쌍별 로그 손실(BT pairwise log-loss)을 최소화하도록 학습합니다:

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

수백 번의 업데이트를 거치면, `w`는 좋은 단어 토큰에는 양수 가중치를, 나쁜 단어 토큰에는 음수 가중치를 할당하게 됩니다.

### 3단계: RM 상단의 PPO 스타일 정책 (PPO-like policy on top of RM)

우리의 토이 정책(toy policy)은 어휘 사전(vocabulary)에서 단일 토큰을 생성합니다. RM(Reward Model)을 통해 해당 토큰의 점수를 매기고, `log π_θ(token | prompt)`를 계산한 뒤, 참조 모델(reference model)에 대한 KL 페널티를 더하고, 클리핑된 PPO 대리 목적 함수(clipped PPO surrogate)를 적용합니다.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 보상(reward)을 리턴(return)으로 취급하여 theta에 대해 ppo 스타일 업데이트 수행
    ...
```

### 4단계: KL 모니터링 (monitor the KL)

매 업데이트마다 평균 `KL(π_θ || π_ref)` 값을 추적하세요. 만약 이 값이 `~5-10`을 넘어간다면, 정책(policy)이 `π_SFT`로부터 멀리 이탈했다는 의미입니다. 이는 `β` 값이 너무 낮거나 보상 해킹(reward hacking)이 시작되고 있음을 나타냅니다. 이는 실제 RLHF 과정에서 가장 중요한 진단 지표입니다.

### 5단계: TRL을 활용한 프로덕션 레시피(Production Recipe)

토이 파이프라인을 이해했다면, 이제 실제 라이브러리 사용자가 작성하는 것과 동일한 루프를 살펴보겠습니다. Hugging Face의 [TRL](https://huggingface.co/docs/trl)은 표준 구현체입니다. 2단계에서는 `RewardTrainer`를, 3단계에서는 (참조 모델에 대한 KL 페널티가 내장된) `PPOTrainer`를 사용합니다.

```python
# 2단계: 쌍체 선호도(pairwise preferences)를 이용한 보상 모델 학습
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 데이터셋 행 구성: {"prompt", "chosen", "rejected"} — Bradley-Terry 형식
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 3단계: SFT 참조 모델에 대한 KL 페널티를 적용하여 RM을 대상으로 PPO 수행
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 동결(frozen) 상태

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats에는 다음 세 가지 PPO 진단 지표가 포함됩니다: mean_kl, clip_frac, value_loss
```

라이브러리가 대신 처리해 주는 세 가지 핵심 사항이 있습니다. 첫째, `adap_kl_ctrl=True`는 적응형-β 스케줄(adaptive-β schedule)을 구현합니다. 관찰된 KL이 `target_kl`을 초과하면 β가 두 배로 증가하고, 절반 미만이면 β가 절반으로 감소합니다. 둘째, 관례에 따라 참조 모델(reference model)은 동결됩니다. 즉, `policy`와 파라미터를 실수로 공유해서는 안 됩니다. 셋째, 가치 헤드(value head)는 정책(policy)과 동일한 백본에 존재합니다(`AutoModelForCausalLMWithValueHead`가 스칼라 MLP 헤드를 부착함). 이것이 TRL이 `policy/kl`과 `value/loss`를 별도로 보고하는 이유입니다.

## 주의 사항 (Pitfalls)

- **과최적화 / 보상 해킹 (Over-optimization / reward hacking).** RM(보상 모델)은 불완전합니다. `π_θ`는 점수는 높지만 실제로는 좋지 않은 적대적 완성형(adversarial completions)을 찾아낼 수 있습니다. 증상: 보상 점수는 무한히 상승하지만 인간 평가 점수는 정체되거나 하락합니다. 해결책: 조기 종료(early stopping)를 적용하거나, `β` 값을 높이거나, RM의 학습 데이터를 확장하세요.
- **길이 해킹 (Length hacking).** 도움이 되는 답변으로 학습된 RM은 종종 암묵적으로 답변의 길이에 보상을 줍니다. 이로 인해 정책(policy)은 답변을 길게 늘리는 법을 배웁니다. 해결책: 길이를 정규화한 보상(length-normalized reward)을 사용하거나, 길이를 인지하는 RM을 사용한 RLAIF를 적용해 보세요.
- **너무 작은 RM (Too-small RM).** RM은 최소한 정책 모델만큼 커야 합니다. 너무 작은 RM은 정책 모델의 출력을 충실하게 평가할 수 없습니다.
- **KL 튜닝 (KL tuning).** `β`가 너무 낮으면 모델이 표류(drift)하고 보상 해킹이 발생합니다. `β`가 너무 높으면 정책이 거의 변하지 않습니다. 표준적인 기법은 매 단계마다 고정된 KL을 목표로 하는 *적응형(adaptive)* `β`를 사용하는 것입니다.
- **선호 데이터 노이즈 (Preference-data noise).** 인간의 라벨링 중 약 30%는 노이즈가 있거나 모호합니다. 일치 여부로 필터링된 데이터로 RM을 학습시켜 교정하거나, BT(Bradley-Terry) 모델에 온도를 적용하여 교정해 보세요.
- **오프-폴리시 문제 (Off-policy problems).** PPO 데이터는 첫 번째 에포크 이후 약간의 오프-폴리시(off-policy) 상태가 됩니다. 레슨 08에서 다룬 것처럼 클립 비율(clip fraction)을 모니터링하세요.

## 활용하기 (Use It)

2026년의 RLHF는 다음과 같이 계층화되어 있습니다:

| 계층 (Layer) | 대상 (Target) | 방법 (Method) |
|-------|--------|--------|
| 지시 이행, 유용성, 무해성 (Instruction following, helpfulness, harmlessness) | 정렬 (Alignment) | RLHF-PPO보다 DPO (Phase 10 · 08) 선호. |
| 추론 정확도 (수학, 코드) (Reasoning correctness) | 역량 (Capability) | 검증기 보상(verifier reward)을 사용하는 GRPO (Phase 9 · 12). |
| 장기 다단계 작업 (Long-horizon multi-step tasks) | 에이전트 (Agentic) | 단계별 프로세스 보상 모델(process reward models)을 사용하는 PPO / GRPO. |
| 안전성 / 거절 행동 (Safety / refusal behavior) | 안전성 (Safety) | 별도의 안전 RM을 사용하는 RLHF-PPO, 또는 Constitutional AI. |
| 추론 시 Best-of-N (Best-of-N at inference) | 빠른 정렬 (Fast alignment) | 디코딩 시점에 RM 사용; 정책(policy) 학습 불필요. |
| 보상 증류 (Reward distillation) | 추론 연산 (Inference compute) | 동결된 LM 위에 작은 "보상 헤드(reward head)"를 학습. |

RLHF는 2022~2024년의 *핵심* 방법론이었습니다. 2026년의 프로덕션 정렬 파이프라인은 DPO를 우선적으로 사용하며, RM 집약적이거나 안전이 중요한 단계에서만 PPO를 사용합니다.

## Ship It

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

기본 언어 모델(base LM), 목표 행동(정렬 / 추론 / 거절 / 에이전트), 그리고 선호도(preference) 또는 검증기(verifier) 예산이 주어졌을 때, 다음을 출력하세요:

1. 단계(Stage). SFT? RM? DPO? GRPO? 근거와 함께 제시하세요.
2. 선호도 또는 검증기 소스(Preference or verifier source). 인간, AI 피드백, 규칙 기반(rule-based), 유닛 테스트 통과(unit-test-pass), 또는 보상 증류(reward distillation).
3. KL 전략(KL strategy). 고정 $\beta$, 적응형 $\beta$, 또는 DPO(암시적 KL).
4. 진단(Diagnostics). 평균 KL, 보상 안정성, 과최적화 방지책(holdout human eval).
5. 안전 게이트(Safety gate). 레드팀 세트, 거절률, 유용성 RM과 분리된 안전 RM.

KL 모니터가 없는 RLHF-PPO 배포는 거부하세요. 타겟 정책(target policy)보다 작은 RM을 사용하는 것을 거부하세요. 길이만을 기준으로 하는 보상(length-only rewards)을 거부하세요. 별도의 블라인드 인간 평가 세트(blind human-eval set)를 확보하지 않은 파이프라인은 과최적화 보호 기능이 부족한 것으로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 있는 Bradley-Terry 보상 모델(reward model)을 500개의 합성 선호도 쌍(synthetic preference pairs)으로 학습시켜 보세요. 별도로 분리한 100개의 쌍에 대해 쌍별 정확도(pairwise accuracy)를 측정하세요. 90%를 초과해야 합니다.
2. **중간 (Medium).** `β ∈ {0.0, 0.1, 1.0}` 값에 대해 토이 PPO-RLHF 루프를 실행해 보세요. 각 경우에 대해 업데이트 과정에 따른 RM 점수(RM score) 대 참조 모델과의 KL 발산(KL-to-reference)을 그래프로 그리세요. 어떤 실행 결과에서 보상 해킹(reward-hack)이 발생하나요?
3. **어려움 (Hard).** 동일한 선호도 데이터에 대해 DPO(closed-form preference-likelihood loss)를 구현하고, 사용된 연산량(compute used)과 달성된 최종 RM 점수를 기준으로 RLHF-PPO 파이프라인과 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | SFT + RM + PPO로 이어지는 3단계 파이프라인 (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Bradley-Terry 모델을 통해 쌍체 비교 선호도(pairwise preferences)에 맞춰 학습된 스칼라 함수. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; 표준적인 RM 목적 함수. |
| KL penalty | "Stay near the reference" | 보상 함수 내의 `β · KL(π_θ \|\| π_ref)`; 보상 해킹(reward-hacking)을 방지하기 위한 정규화 도구. |
| Reward hacking | "Goodhart's law" | 정책(Policy)이 RM의 결함을 악용하는 현상; 증상: 보상은 상승하나 인간 평가 점수는 정체됨. |
| RLAIF | "AI-labeled preferences" | 인간 대신 다른 언어 모델(LM)로부터 레이블을 받아 수행하는 RLHF. |
| PRM | "Process Reward Model" | 추론 과정의 중간 단계별로 점수를 매기는 모델; 추론 파이프라인에서 사용됨. |
| Constitutional AI | "Anthropic's method" | 명시적인 규칙(rules)에 따라 AI가 생성한 선호도를 사용하는 방식. |

## 추가 읽을거리 (Further Reading)

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHF의 시작이 된 논문입니다.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT의 기반이 된 레시피입니다.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 요약 작업을 위한 초기 RLHF 연구입니다.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; 2026년 기준 RLHF 이후의 기본 방식입니다.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 및 자기 비판(self-critique) 루프에 관한 연구입니다.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 논문입니다.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 실무용 `RewardTrainer` 및 `PPOTrainer`를 제공합니다. 적응형 KL(adaptive-KL) 및 가치 헤드(value-head)에 대한 세부 사항은 트레이너 소스 코드를 읽어 보세요.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) (Lambert, Castricato, von Werra, Havrilla 저) — 다이어그램과 함께 3단계 파이프라인을 설명하는 표준 가이드입니다.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 해당 라이브러리입니다. `examples/` 디렉토리에 Llama, Mistral, Qwen을 위한 엔드투엔드(end-to-end) RLHF 스크립트가 포함되어 있습니다.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 보상 가설(reward-hypothesis) 관점을 다룹니다. 보상 해킹(reward hacking)을 이해하기 위한 필수 선행 지식입니다.
