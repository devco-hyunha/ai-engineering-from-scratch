# 구조화된 출력(Structured Output) — JSON Schema, Pydantic, Zod, 제약 디코딩(제약 디코딩)

> "모델에게 JSON을 반환해 달라고 정중하게 요청하는" 방식은 프론티어 모델에서도 5~15%의 확률로 실패합니다. 구조화된 출력은 제약 디코딩(제약 디코딩)으로 이 격차를 해소합니다. 모델은 스키마를 위반하는 토큰을 방출하는 것이 문자 그대로 차단됩니다. OpenAI의 strict mode, Anthropic의 스키마 타입 도구 사용, Gemini의 `responseSchema`, Pydantic AI의 `output_type`, Zod의 `.parse`는 동일한 개념의 다섯 가지 표면 형태입니다. 이 강의에서는 학습자가 모든 프로덕션 추출 파이프라인에서 사용할 스키마 검증기와 strict-mode 계약을 구축합니다.

**유형:** Build
**언어:** Python (stdlib, JSON Schema 2020-12 하위 집합)
**선수 요건:** 13단계 · 02강 (함수 호출 심층 분석)
**시간:** 약 75분

## 학습 목표

- 적절한 제약 조건(enum, min/max, required, pattern)을 사용하여 추출 대상에 대한 JSON Schema 2020-12를 작성해 보세요.
- strict mode와 제약 디코딩(제약 디코딩)이 "생성 후 검증"과 다른 보장을 제공하는 이유를 설명해 보세요.
- 세 가지 실패 모드인 파싱 오류, 스키마 위반, 모델 거절을 구분해 보세요.
- 타입이 지정된 복구 및 타입이 지정된 거절 처리가 포함된 추출 파이프라인을 출시해 보세요.

## 문제점

구매 주문 이메일을 읽는 에이전트(Agent)는 자유 텍스트를 `{customer, line_items, total_usd}`로 변환해야 합니다. 세 가지 접근법이 있습니다.

**첫 번째 접근법: JSON을 요청하는 프롬프트.** "customer, line_items, total_usd 필드를 포함하여 JSON으로 응답하세요." 프론티어 모델에서는 85~95%의 확률로 작동합니다. 여는 중괄호 누락, 후행 쉼표, 잘못된 타입, 환각(Hallucination)된 필드, 토큰 한도에서의 잘림, "여기 JSON이 있습니다:"와 같은 산문 유출 등 여섯 가지 방식으로 실패합니다.

**두 번째 접근법: 생성 후 검증.** 자유롭게 생성하고, 파싱하고, 스키마에 대해 검증하고, 실패 시 재시도합니다. 신뢰할 수 있지만 비용이 많이 듭니다. 모든 재시도에 비용을 지불해야 하며, 잘림 버그는 발생마다 추가 턴(turn) 한 번의 비용이 발생합니다.

**세 번째 접근법: 제약 디코딩(제약 디코딩).** 제공자가 디코딩 시점에 스키마를 강제합니다. 유효하지 않은 토큰은 샘플링 분포에서 마스킹됩니다. 출력은 파싱이 보장되고 검증이 보장됩니다. 실패는 하나의 모드로 수렴합니다. 거절(모델이 입력이 스키마에 맞지 않는다고 판단하는 경우)입니다.

2026년 모든 프론트이어(profrontier) 제공자는 세 번째 접근 방식의 형태를 제공합니다.

- **OpenAI.** 모델이 거절할 경우 응답에 `response_format: {type: "json_schema", strict: true}` 및 `refusal`가 포함됩니다.
- **Anthropic.** `tool_use` 입력에 대한 스키마 강제 적용; `stop_reason: "refusal"`는 존재하지 않지만, 도구 호출이 없는 `end_turn`가 신호입니다.
- **Gemini.** 요청 수준에서 `responseSchema`를 적용합니다. 2026년 Gemini는 선택된 유형에 대해 토큰 수준 문법 제약 조건을 제공합니다.
- **Pydantic AI.** `output_type=InvoiceModel`는 `InvoiceModel`로 타입 지정된 구조화된 `RunResult`를 생성합니다.
- **Zod (TypeScript).** Zod 스키마에 대해 제공자 출력의 유효성을 검사하는 런타임 파서입니다. OpenAI의 `beta.chat.completions.parse`와 함께 사용됩니다.

공통적인 흐름은 스키마를 한 번 선언하고 끝까지 강제 적용하는 것입니다.

## 개념

### JSON Schema 2020-12 — 공통 언어

모든 제공자는 JSON Schema 2020-12를 허용합니다. 가장 많이 사용하는 구성 요소는 다음과 같습니다.

- `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` 중 하나입니다.
- `properties`: 필드 이름을 하위 스키마에 매핑합니다.
- `required`: 반드시 나타나야 하는 필드 이름 목록입니다.
- `enum`: 허용된 값의 닫힌 집합입니다.
- `minimum` / `maximum` (숫자), `minLength` / `maxLength` / `pattern` (문자열).
- `items`: 모든 배열 요소에 적용되는 하위 스키마입니다.
- `additionalProperties`: `false`는 추가 필드를 금지합니다 (기본값은 모드에 따라 다릅니다).

OpenAI strict mode는 세 가지 요구 사항을 추가합니다. 모든 속성은 `required`에 나열되어야 하며, `additionalProperties: false`가 모든 곳에 존재해야 하고, 미해결된 `$ref`가 없어야 합니다. 이를 위반하면 API는 요청 시 400을 반환합니다.

### Pydantic, Python 바인딩

Pydantic v2는 `model_json_schema()`를 통해 데이터 클래스 모양의 모델에서 JSON Schema를 생성합니다. Pydantic AI는 이를 래핑하여 다음과 같이 작성합니다.

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

그리고 에이전트 프레임워크는 스키마를 OpenAI strict mode, Anthropic `input_schema` 또는 Gemini `responseSchema`로 변환합니다. 모델의 출력은 타입 지정된 `Invoice` 인스턴스로 반환됩니다. 유효성 검사 오류는 타입 지정된 오류 경로와 함께 `ValidationError`를 발생시킵니다.

### Zod, TypeScript 바인딩

Zod (`z.object({customer: z.string(), ...})`)는 TypeScript의 동등한 도구입니다. OpenAI의 Node SDK는 API의 JSON Schema 페이로드에 매핑되는 `zodResponseFormat(Invoice)`을 노출합니다.

### 거절

엄격 모드(strict mode)는 모델이 답변을 강제할 수 없습니다. 입력이 스키마에 맞지 않는 경우("이메일은 청구서가 아니라 시였습니다"), 모델은 이유를 담은 `refusal` 필드를 생성합니다. 코드는 이를 실패가 아닌 일급 결과(first-class outcome)로 처리해야 합니다. 거절은 안전 신호로도 유용합니다. 보호된 콘텐츠의 이메일에서 신용카드 번호를 추출하라는 요청을 받은 모델은 안전 이유를 첨부한 거절을 반환합니다.

### 오픈 웨이트에서의 제약 디코딩

오픈 웨이트 구현은 세 가지 기법을 사용합니다.

1. **문법 기반 디코딩**(`outlines`, `guidance`, `lm-format-enforcer`): 스키마에서 결정적 유한 오토마톤(FSM)을 구축합니다. 모든 단계에서 FSM을 위반하는 토큰의 로짓을 마스킹합니다.
2. **JSON 파서와 함께 로짓 마스킹**: 모델과 동기화(lockstep)하여 스트리밍 JSON 파서를 실행합니다. 모든 단계에서 유효한 다음 토큰 집합을 계산합니다.
3. **검증자를 활용한 추론적 디코딩(Speculative Decoding)**: 저렴한 초안 모델이 토큰을 제안하면, 검증자가 스키마를 강제합니다.

상용 제공자는 이 중 하나를 내부적으로 선택합니다. 2026년 최신 기술은 짧은 구조화된 출력에서는 일반 생성보다 빠르고, 긴 출력에서는 속도가 거의 동일합니다.

### 세 가지 실패 모드

1. **파싱 오류.** 출력이 유효한 JSON이 아닙니다. 엄격 모드에서는 발생하지 않습니다. 비엄격(non-strict) 제공자에서는 여전히 발생할 수 있습니다.
2. **스키마 위반.** 출력은 파싱되지만 스키마를 위반합니다. 엄격 모드에서는 발생하지 않습니다. 그 외의 환경에서는 흔합니다.
3. **거절.** 모델이 답변을 거부합니다. 타입이 지정된 결과로 처리해야 합니다.

### 재시도 전략

엄격 모드 밖(Anthropic 도구 사용, 비엄격 OpenAI, 구버전 Gemini)에 있을 때, 복구 패턴은 다음과 같습니다:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

한 번의 재시도로 보통 충분합니다. 세 번의 재시도는 약한 모델의 불안정한 테스트(flaky)를 잡습니다. 세 번을 넘기는 것은 스키마가 나쁘다는 신호입니다. 모델이 일부 입력에 대해 스키마를 충족할 수 없으며, 프롬프트나 스키마를 수정해야 합니다.

### 소형 모델 지원

제약 디코딩은 소형 모델에서도 잘 작동합니다. 문법 강제 기능을 갖춘 30억 파라미터 오픈 모델은 구조화된 작업에서 원시 프롬프트를 사용하는 700억 파라미터 모델보다 더 잘 수행합니다. 이것이 프로덕션 환경에서 구조화된 출력이 중요한 주요 이유입니다. 신뢰성을 모델 크기와 분리하기 때문입니다.

```figure
constrained-decoding
```

## 사용하기

`code/main.py`은 표준 라이브러리에서 최소한의 JSON Schema 2020-12 검증기를 제공합니다 (타입, required, enum, min/max, pattern, items, additionalProperties). `Invoice` 스키마를 래핑하고 가짜 LLM 출력을 검증기에 통과시켜 파싱 오류, 스키마 위반, 거부 경로를 시연합니다. 프로덕션에서는 가짜 출력 대신 실제 제공자의 응답으로 교체하세요.

확인할 사항:

- 검증기는 경로와 메시지가 포함된 타입 지정된 `[ValidationError]` 목록을 반환합니다. 이 형태가 재시도 프롬프트에 노출되도록 하는 형태입니다.
- 거부 분기는 재시도하지 않습니다. 로깅하고 타입 지정된 거부를 반환합니다. 14단계 · 09강은 거부를 안전 신호로 사용합니다.
- `additionalProperties: false` 체크는 적대적 테스트 입력에서 발동하며, 엄격 모드가 환각된 필드에 대해 문을 닫는 이유를 보여줍니다.

## 출시하기

이 강의는 `outputs/skill-structured-output-designer.md`을 생성합니다. 자유 텍스트 추출 대상(청구서, 지원 티켓, 이력서 등)이 주어지면, 스킬은 엄격 모드 호환 JSON Schema 2020-12와 이를 미러링하는 Pydantic 모델을 생성하며, 타입 지정된 거부 및 재시도 처리가 스텁으로 포함됩니다.

## 연습 문제

1. `code/main.py`을 실행하세요. `total_usd`이 음수인 네 번째 테스트 케이스를 추가하세요. 검증기가 `minimum` 제약 경로로 이를 거부하는지 확인하세요.

2. 검증기를 확장하여 판별자(discriminator)가 있는 `oneOf`을 지원하세요. 일반적인 경우: `line_item`은 제품 또는 서비스이며, `kind`로 태그가 지정됩니다. 엄격 모드에는 미묘한 규칙이 있습니다. OpenAI의 구조화된 출력 가이드를 확인하세요.

3. 동일한 Invoice 스키마를 Pydantic BaseModel로 작성하고 `model_json_schema()` 출력과 수작업 스키마를 비교하세요. Pydantic이 기본으로 설정하는 필드 중 수작업 버전이 생략하는 필드를 식별하세요.

4. 거부율을 측정하세요. 추출할 수 없는 열 개의 입력(노래 가사, 수학 증명, 빈 이메일)을 구성하고 엄격 모드를 사용하는 실제 제공자를 통해 실행하세요. 거부와 환각된 출력을 세어 보세요. 이는 거부 인식 재시도를 위한 기준 데이터입니다.

5. OpenAI의 구조화된 출력 가이드를 처음부터 끝까지 읽어 보세요. 일반 JSON Schema가 허용하는 구성 중 strict 모드에서 명시적으로 금지하는 하나를 식별해 보세요. 그런 다음 금지된 구성을 필수적이지 않은 방식으로 사용하는 스키마를 설계하고, strict 모드와 호환되도록 리팩터링해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "스키마 사양" | 모든 최신 제공자가 사용하는 IETF 초안 스키마 방언 |
| Strict 모드 | "보장된 스키마" | 제약 디코딩(제약 디코딩)을 통해 스키마를 강제하는 OpenAI 플래그 |
| 제약 디코딩(제약 디코딩) | "로짓 마스킹" | 디코딩 시 유효하지 않은 다음 토큰을 마스킹하여 강제하는 방식 |
| 거절(Refusal) | "모델이 거부함" | 입력이 스키마에 맞지 않을 때 발생하는 타입 지정된 결과 |
| 파싱 오류 | "유효하지 않은 JSON" | 출력은 JSON으로 파싱되지 않았습니다. strict 모드에서는 불가능합니다 |
| 스키마 위반 | "잘못된 형태" | 파싱은 되었지만 타입 / 필수 필드 / enum / 범위를 위반했습니다 |
| `additionalProperties: false` | "추가 필드 허용 안 함" | 알 수 없는 필드를 금지합니다. OpenAI strict 모드에서 필수입니다 |
| Pydantic BaseModel | "타입 지정된 출력" | JSON Schema를 생성하고 검증하는 Python 클래스 |
| Zod 스키마 | "TypeScript 출력 타입" | 제공자 출력 검증을 위한 TS 런타임 스키마 |
| 문법 강제 | "오픈 웨이트 제약 디코딩" | outlines / guidance와 같이 FSM 기반 로짓 마스킹 |

## 추가 읽기

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict 모드, 거절 및 스키마 요구 사항
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 디코딩 보장을 설명하는 2024년 8월 출시 포스트
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 각 제공자로 직렬화되는 타입 지정된 output_type 바인딩
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 표준 사양
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 엔터프라이즈 배포 노트 및 strict 모드 주의 사항
