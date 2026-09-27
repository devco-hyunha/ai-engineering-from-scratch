# 구조화된 출력 및 제약 디코딩 (Structured Outputs & Constrained Decoding)

> LLM에 JSON을 요청하면 대부분의 경우 JSON을 얻을 수 있습니다. 하지만 프로덕션 환경에서는 "대부분"이라는 점이 문제가 됩니다. 제약 디코딩(constrained decoding)은 샘플링 전 로짓(logits)을 수정하여 "대부분"을 "항상"으로 바꿉니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 문제점 (The Problem)

분류기가 LLM에 다음과 같이 프롬프트를 전달합니다: "하나를 선택하여 반환하세요: {positive, negative, neutral}." 모델은 "감정은 긍정적입니다 — 고객이 명시적으로 ...라고 언급했기 때문에 이 리뷰는 매우 우호적입니다"라고 응답합니다. 파서(parser)에서 오류가 발생하며, 분류기의 F1 점수는 0.0이 됩니다.

자유 형식 생성(Free-form generation)은 계약(contract)이 아니라, 단지 제안일 뿐입니다. 프로덕션 시스템에는 계약이 필요합니다.

2026년에는 세 가지 계층이 존재합니다.

1. **프롬프팅(Prompting).** 정중하게 요청하는 방식입니다. "JSON 객체만 반환하세요." 프론티어 모델에서는 약 80% 정도 작동하지만, 규모가 작은 모델에서는 그보다 낮습니다.
2. **네이티브 구조화된 출력 API(Native structured output APIs).** OpenAI의 `response_format`, Anthropic의 도구 사용(tool use), Gemini의 JSON 모드 등이 있습니다. 지원되는 스키마에 대해서는 신뢰할 수 있지만, 특정 벤더에 종속됩니다.
3. **제약 디코딩(Constrained decoding).** 모델이 유효하지 않은 토큰을 생성할 수 없도록 매 생성 단계마다 로짓(logits)을 수정합니다. 구조적으로 100% 유효하며, 모든 로컬 모델에서 작동합니다.

이 레슨에서는 이 세 가지 방식에 대한 직관을 기르고, 각각 어떤 상황에서 어떤 방식을 사용해야 하는지 알아봅니다.

## 개념 (The Concept)

![각 단계에서 유효하지 않은 토큰을 마스킹하는 제약 디코딩](../assets/constrained-decoding.svg)

**제약 디코딩(constrained decoding)의 작동 방식.** 각 생성 단계에서 LLM은 전체 어휘(약 10만 개의 토큰)에 대한 로짓(logit) 벡터를 생성합니다. *로짓 프로세서(logit processor)*는 모델과 샘플러 사이에 위치합니다. 이 프로세서는 대상 문법(JSON Schema, 정규 표현식, 문맥 자유 문법 등)의 현재 위치를 기준으로 어떤 토큰이 유효한지 계산하고, 유효하지 않은 모든 토큰의 로짓을 음의 무한대로 설정합니다. 남은 로짓에 대해 소프트맥스(softmax)를 적용하면 유효한 다음 토큰에만 확률 질량이 할당됩니다.

2026년 기준 구현 사례:

- **Outlines.** JSON Schema 또는 정규 표현식을 유한 상태 머신(FSM)으로 컴파일합니다. 모든 토큰은 $O(1)$의 유효한 다음 토큰 조회를 수행합니다. FSM 기반이므로 재귀적 스키마는 평탄화(flattening)가 필요합니다.
- **XGrammar / llguidance.** 문맥 자유 문법 엔진입니다. 재귀적 JSON Schema를 처리할 수 있습니다. 디코딩 오버헤드가 거의 없습니다. OpenAI는 2025년 구조화된 출력(structured output) 구현에서 llguidance의 기여를 언급했습니다.
- **vLLM guided decoding.** Outlines, XGrammar 또는 lm-format-enforcer 백엔드를 통해 `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` 기능을 내장하고 있습니다.
- **Instructor.** 모든 LLM에 대해 사용할 수 있는 Pydantic 기반 래퍼(wrapper)입니다. 검증 실패 시 재시도합니다. 여러 프로바이더를 지원하지만 로짓을 수정하지는 않으며, 재시도와 구조화된 출력 인지형 프롬프트(structured-output-aware prompts)에 의존합니다.

### 직관에 반하는 결과 (The counterintuitive result)

제약 디코딩(constrained decoding)은 종종 비제약 생성(unconstrained generation)보다 *더 빠릅니다*. 여기에는 두 가지 이유가 있습니다. 첫째, 다음 토큰의 탐색 공간(search space)을 축소합니다. 둘째, 효율적인 구현체는 강제된 토큰(forced tokens)에 대한 토큰 생성을 완전히 건너뜁니다 (`{"name": "`와 같은 스캐폴딩(scaffolding)의 경우, 모든 바이트가 이미 결정되어 있습니다).

### 비용을 초래하는 함정 (The pitfall that costs you)

필드 순서가 중요합니다. `answer`를 `reasoning`보다 앞에 두면, 모델은 추론하기도 전에 답을 결정해 버립니다. JSON 형식은 유효하지만, 답은 틀립니다. 어떤 검증도 이를 잡아내지 못합니다.

```json
// 나쁜 예
{"answer": "yes", "reasoning": "because ..."}

// 좋은 예
{"reasoning": "... therefore ...", "answer": "yes"}
```

스키마 필드 순서는 포맷팅이 아니라 로직입니다.

```figure
constrained-decoder
```

## 구현하기 (Build It)

### 1단계: 처음부터 시작하는 정규식 제약 생성 (regex-constrained generation from scratch)

FSM의 독립적인 구현은 `code/main.py`를 참조하세요. 핵심 아이디어를 30줄 내외로 요약하면 다음과 같습니다:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM은 현재 문법의 어느 단계까지 충족했는지를 추적합니다. `valid_tokens(state, tokenizer)`는 수락 경로(accepting path)를 벗어나지 않으면서 FSM을 전이시킬 수 있는 어휘 토큰(vocabulary tokens)을 계산합니다.

### 2단계: JSON 스키마를 위한 Outlines (Outlines for JSON Schema)

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

검증 오류가 전혀 없습니다. 단 한 번도 발생하지 않았습니다. FSM은 잘못된 출력이 발생하지 않도록 보장합니다.

### 3단계: 제공자 독립적(provider-agnostic) Pydantic 활용을 위한 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

작동 방식은 다릅니다. Instructor는 로짓(logits)을 조작하지 않습니다. 대신 스키마를 프롬프트에 구조화하여 포함하고, 출력을 파싱하며, 유효성 검사 실패 시 재시도합니다(기본 3회). 모든 제공자에서 사용할 수 있습니다. 재시도는 지연 시간(latency)과 비용을 증가시킵니다. 제공자 간 이식성(cross-provider portability)이 핵심 장점입니다.

### 4단계: 네이티브 벤더 API (native vendor APIs)

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

서버 측 제약 디코딩(constrained decoding) 방식입니다. 지원되는 스키마에 대해 Outlines와 대등한 신뢰성을 제공합니다. 로컬 모델을 관리할 필요가 없으나, 특정 벤더에 종속됩니다.

## 주의 사항 (Pitfalls)

- **재귀적 스키마 (Recursive schemas).** Outlines는 재귀 구조를 고정된 깊이로 평탄화(flatten)합니다. 트리 구조의 출력(중첩된 댓글, AST 등)을 생성하려면 XGrammar 또는 llguidance(CFG 기반)가 필요합니다.
- **거대한 열거형 (Huge enums).** 10,000개의 옵션을 가진 열거형은 컴파일 속도가 느려지거나 타임아웃이 발생할 수 있습니다. 리트리버(retriever) 방식을 고려해 보세요. 먼저 상위 k개의 후보를 예측한 뒤, 해당 후보들에 제약을 거는 방식입니다.
- **너무 엄격한 문법 (Grammar too strict).** `date: "YYYY-MM-DD"`와 같은 정규 표현식을 강제하면, 모델이 날짜가 누락된 경우 `"unknown"`을 출력할 수 없습니다. 이 경우 모델은 이를 해결하기 위해 임의의 날짜를 생성해 버립니다. `null` 또는 센티널(sentinel) 값을 허용해 보세요.
- **성급한 결정 (Premature commitment).** 위의 필드 순서(field-order) 함정을 참조하세요. 항상 추론(reasoning) 과정을 먼저 배치해야 합니다.
- **스키마 없는 벤더 제공 JSON 모드 (Vendor JSON mode without schema).** 순수 JSON 모드는 유효한 JSON 형식만을 보장할 뿐, *사용자의 사례(use case)*에 적합한 형식을 보장하지는 않습니다. 항상 전체 스키마를 제공하세요.

## 활용 방법 (Use It)

2026년 스택:

| 상황 | 선택 |
| :--- | :--- |
| OpenAI/Anthropic/Google 모델, 단순한 스키마 | 벤더 네이티브 구조화된 출력 (Native vendor structured output) |
| 모든 제공자, Pydantic 워크플로우, 재시도 가능 | Instructor |
| 로컬 모델, 100% 유효성 보장 필요, 평면 스키마 (flat schema) | Outlines (FSM) |
| 로컬 모델, 재귀적 스키마 (recursive schema) | XGrammar 또는 llguidance |
| 자체 호스팅 추론 서버 (Self-hosted inference server) | vLLM 가이드 디코딩 (vLLM guided decoding) |
| 재시도가 가능한 배치 처리 (Batch processing) | Instructor + 가장 저렴한 모델 |

## Ship It (실행해 보기)

`outputs/skill-structured-output-picker.md`로 저장하세요:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 연습 문제 (Exercises)

1. **쉬움(Easy).** `Review(sentiment, confidence, evidence_span)` 형식을 생성하도록 제약 디코딩(constrained decoding) 없이 소형 오픈 웨이트 모델(예: Llama-3.2-3B)에 프롬프트를 입력하세요. 100개의 리뷰 중 유효한 JSON으로 파싱되는 비율을 측정하세요.
2. **중간(Medium).** Outlines JSON 모드를 사용하여 동일한 코퍼스(corpus)를 대상으로 실험하세요. 준수율(compliance rate), 지연 시간(latency), 그리고 의미적 정확도(semantic accuracy)를 비교하세요.
3. **어려움(Hard).** 전화번호(`\d{3}-\d{3}-\d{4}`)를 위한 정규 표현식 제약 디코더(regex-constrained decoder)를 처음부터 구현하세요. 1,000개의 샘플에서 잘못된 출력이 0개인지 확인하세요.

## 주요 용어 (Key Terms)

| 용어 (Term) | 사람들이 말하는 방식 (What people say) | 실제 의미 (What it actually means) |
|------|-----------------|-----------------------|
| 제약 디코딩 (constrained decoding) | 유효한 출력을 강제하는 것 | 매 생성 단계마다 유효하지 않은 토큰의 로짓을 마스킹합니다. |
| 로짓 프로세서 (logit processor) | 제약을 적용하는 것 | 함수: (logits, state) -> masked_logits. |
| FSM | 유한 상태 머신 | 컴파일된 문법 표현; O(1)로 유효한 다음 토큰을 조회합니다. |
| CFG | 문맥 자유 문법 | 재귀를 처리하는 문법; FSM보다 느리지만 표현력이 더 뛰어납니다. |
| 스키마 필드 순서 (schema field order) | 중요할까요? | 네 — 첫 번째 필드가 먼저 결정됩니다. 항상 답변(answer) 전에 추론(reasoning)을 배치하세요. |
| 가이드 디코딩 (guided decoding) | vLLM에서 사용하는 명칭 | 동일한 개념이며, 추론 서버에 통합되어 있습니다. |
| JSON 모드 (JSON mode) | OpenAI의 초기 버전 | JSON 구문을 보장하지만, 스키마 일치까지 보장하지는 않습니다. |

## 추가 학습 자료 (Further Reading)

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — 논문의 개요를 다룹니다.
- [XGrammar 논문 (2024)](https://arxiv.org/abs/2411.15100) — 빠른 CFG 기반 제약 디코딩(constrained decoding)을 다룹니다.
- [vLLM — 구조화된 출력(Structured Outputs)](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 추론 서버 통합에 관한 내용입니다.
- [OpenAI — 구조화된 출력(Structured Outputs) 가이드](https://platform.openai.com/docs/guides/structured-outputs) — API 레퍼런스 및 주의 사항(gotchas)을 포함합니다.
- [Instructor 라이브러리](https://python.useinstructor.com/) — 다양한 프로바이더에 대한 Pydantic 및 재시도(retries) 기능을 제공합니다.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6개의 제약 디코딩(constrained decoding) 프레임워크를 벤치마킹합니다.
