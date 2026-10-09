# 구조화된 출력 및 제약 디코딩

> LLM에 JSON을 요청하면 대부분 JSON을 얻습니다. 프로덕션 환경에서는 "대부분"이 문제입니다. 제약 디코딩은 샘플링 전에 로짓을 수정하여 "대부분"을 "항상"으로 만듭니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 17강 (챗봇), 5단계 · 19강 (서브워드 토큰화)
**시간:** 약 60분

## 문제점

분류기가 LLM에 "positive, negative, neutral 중 하나를 반환하세요"라고 프롬프트합니다. 모델은 "감정은 positive입니다 — 이 리뷰는 고객이 명시적으로 ...라고 언급했기 때문에 압도적으로 긍정적입니다"라고 반환합니다. 파서가 충돌합니다. 분류기의 F1 점수가 0.0이 됩니다.

자유 형식 생성은 계약이 아닙니다. 제안일 뿐입니다. 프로덕션 시스템은 계약이 필요합니다.

2026년에는 세 가지 계층이 존재합니다.

1. **프롬프트.** 정중하게 요청합니다. "JSON 객체만 반환하세요." 프론티어 모델에서는 약 80% 작동하며, 작은 모델에서는 효과가 떨어집니다.
2. **네이티브 구조화된 출력 API.** OpenAI `response_format`, Anthropic 도구 사용, Gemini JSON 모드. 지원되는 스키마에서는 신뢰성이 높습니다. 벤더 종속적입니다.
3. **제약 디코딩.** 모든 생성 단계에서 로짓을 수정하여 모델이 *무효한* 토큰을 출력할 수 없게 만듭니다. 구성상 100% 유효합니다. 모든 로컬 모델에서 작동합니다.

이 강의는 세 가지 모두에 대한 직관을 구축하고, 어떤 상황에서 무엇을 선택해야 하는지 알려줍니다.

## 개념

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**제약 디코딩이 작동하는 방식.** 각 생성 단계에서 LLM은 전체 어휘(~100k 토큰)에 대한 로짓 벡터를 생성합니다. 모델과 샘플러 사이에 *로짓 프로세서*가 위치합니다. 이 프로세서는 대상 문법(JSON Schema, 정규식, 문맥 자유 문법)의 현재 위치에 따라 유효한 토큰을 계산하고, 모든 무효한 토큰의 로짓을 음의 무한대로 설정합니다. 남은 로짓에 대한 소프트맥스는 유효한 연속에만 확률 질량을 배분합니다.

2026년 구현체:

- **Outlines.** JSON Schema나 정규식을 유한 상태 기계(FSM)로 컴파일합니다. 모든 토큰에 대해 O(1) 유효 다음 토큰 조회가 가능합니다. FSM 기반이므로 재귀적 스키마는 평탄화(flattening)가 필요합니다.
- **XGrammar / llguidance.** 문맥 자유 문법 엔진입니다. 재귀적인 JSON Schema를 처리합니다. 디코딩 오버헤드가 거의 없습니다. OpenAI는 2025년 구조화된 출력 구현에서 llguidance의 기여를 인정했습니다.
- **vLLM 가이드 디코딩.** Outlines, XGrammar, lm-format-enforcer 백엔드를 통해 내장된 `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`을 지원합니다.
- **Instructor.** 모든 LLM에 대한 Pydantic 기반 래퍼입니다. 검증 실패 시 재시도합니다. 교차 제공자(cross-provider) 환경에서 작동하지만 로짓을 수정하지는 않으며, 재시도 및 구조화된 출력 인식 프롬프트에 의존합니다.

### 직관에 반하는 결과

제약 디코딩(제약 디코딩)은 종종 비제약 생성보다 *더 빠릅니다*. 두 가지 이유가 있습니다. 첫째, 다음 토큰 검색 공간을 줄입니다. 둘째, 영리한 구현은 강제 토큰(예: `{"name": "` — 모든 바이트가 결정됨)에 대해 토큰 생성을 완전히 생략합니다.

### 비용을 초래하는 함정

필드 순서가 중요합니다. `answer`을 `reasoning` 앞에 배치하면 모델은 생각하기 전에 답변을 확정합니다. JSON은 유효합니다. 답변은 틀립니다. 검증이 이를 잡아내지 못합니다.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 필드 순서는 형식 문제가 아니라 로직입니다.

```figure
constrained-decoder
```

## 구현하기

### 1단계: 처음부터 정규식 제약 생성 구현하기

독립적인 FSM 구현은 `code/main.py`을 참고하세요. 30줄의 핵심 아이디어는 다음과 같습니다:

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

FSM은 지금까지 문법의 어떤 부분을 충족했는지 추적합니다. `valid_tokens(state, tokenizer)`은 FSM을 수락 경로에서 벗어나지 않고 진행할 수 있는 어휘 토큰을 계산합니다.

### 2단계: JSON Schema를 위한 Outlines

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

검증 오류는 절대 없습니다. FSM이 유효하지 않은 출력을 도달 불가능하게 만듭니다.

### 3단계: 제공자 독립적인 Pydantic을 위한 Instructor

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

메커니즘이 다릅니다. Instructor는 로짓을 건드리지 않습니다. Schema를 프롬프트에 포맷하고, 출력을 파싱하며, 검증 실패 시 재시도합니다(기본값 3회). 모든 제공자와 작동합니다. 재시도는 지연과 비용을 추가합니다. 교차 제공자 이식성이 판매 포인트입니다.

### 4단계: 네이티브 벤더 API

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

서버 측 제약 디코딩입니다. 지원되는 Schema에 대해 Outlines와 신뢰성 동등성을 가집니다. 로컬 모델 관리가 필요 없습니다. 벤더에 종속됩니다.

## 함정

- **재귀 스키마.** Outlines는 재귀를 고정된 깊이로 평탄화합니다. 트리 구조 출력(중첩된 주석, AST)은 XGrammar 또는 llguidance(CFG 기반)가 필요합니다.
- **거대한 열거형.** 10,000개 옵션의 열거형은 컴파일 속도가 느리거나 시간 초과가 발생합니다. 검색기로 전환하세요: 먼저 상위 k개 후보를 예측하고, 그 후보들로 제한합니다.
- **문법이 너무 엄격함.** `date: "YYYY-MM-DD"` 정규식을 강제하면 모델은 누락된 날짜에 대해 `"unknown"`을 출력할 수 없습니다. 모델은 날짜를 임의로 만들어 보상합니다. `null` 또는 센티널을 허용하세요.
- **조기 확정.** 위의 필드 순서 함정을 참고하세요. 항상 추론을 먼저 배치하세요.
- **스키마 없는 벤더 JSON 모드.** 순수 JSON 모드는 유효한 JSON만 보장하며, *사용 사례에 대한 유효성*은 보장하지 않습니다. 항상 완전한 스키마를 제공하세요.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| OpenAI/Anthropic/Google 모델, 단순 스키마 | 벤더의 네이티브 구조화된 출력 |
| 모든 제공자, Pydantic 워크플로, 재시도 허용 가능 | Instructor |
| 로컬 모델, 100% 유효성 필요, 평탄한 스키마 | Outlines (FSM) |
| 로컬 모델, 재귀 스키마 | XGrammar 또는 llguidance |
| 자체 호스팅 추론 서버 | vLLM 가이드 디코딩 |
| 재시도가 허용되는 배치 처리 | Instructor + 가장 저렴한 모델 |

## 출시하기

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

## 연습 문제

1. **쉬움.** 제약 디코딩 없이 작은 오픈 웨이트 모델(예: Llama-3.2-3B)에 `Review(sentiment, confidence, evidence_span)`을 프롬프트하세요. 100개 리뷰에서 유효한 JSON으로 파싱되는 비율을 측정하세요.
2. **중간.** Outlines JSON 모드를 사용하여 동일한 코퍼스를 처리하세요. 준수율, 지연 시간, 의미적 정확도를 비교하세요.
3. **어려움.** 전화번호(`\d{3}-\d{3}-\d{4}`)를 위해 정규식 제약 디코더를 처음부터 구현하세요. 1000개 샘플에서 0개의 유효하지 않은 출력을 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 제약 디코딩 | 유효한 출력 강제 | 모든 생성 단계에서 유효하지 않은 토큰의 로짓을 마스킹합니다. |
| 로짓 프로세서 | 제약하는 요소 | 함수: `(logits, state) -> masked_logits`. |
| FSM | 유한 상태 기계 | 컴파일된 문법 표현; O(1) 유효 다음 토큰 조회. |
| CFG | 문맥 자유 문법 | 재귀를 처리하는 문법; FSM보다 느리지만 더 표현력이 높습니다. |
| 스키마 필드 순서 | 중요합니까? | 예 — 첫 번째 필드가 확정됩니다; 항상 답변 전에 추론을 배치하세요. |
| 가이드 디코딩 | vLLM의 명칭 | 동일한 개념이며, 추론 서버에 통합되어 있습니다. |
| JSON 모드 | OpenAI의 초기 버전 | JSON 문법을 보장합니다; 스키마 일치를 보장하지는 않습니다. |

## 추가 읽기

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 논문입니다.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 빠른 CFG 기반 제약 디코딩입니다.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 추론 서버 통합입니다.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 레퍼런스 + 주의 사항입니다.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + 프로바이더 간 재시도입니다.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6개 제약 디코딩 프레임워크 벤치마킹입니다.
