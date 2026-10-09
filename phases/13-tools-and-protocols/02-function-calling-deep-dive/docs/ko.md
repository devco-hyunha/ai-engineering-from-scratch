# 함수 호출 심층 분석 — OpenAI, Anthropic, Gemini

> 세 주요 프론트이어 제공자는 2024년에 동일한 도구 호출 루프로 수렴한 후, 나머지 모든 부분에서 갈라졌습니다. OpenAI는 `tools`과 `tool_calls`을 사용합니다. Anthropic은 `tool_use`과 `tool_result` 블록을 사용합니다. Gemini는 `functionDeclarations`과 고유 ID 상관관계를 사용합니다. 이 강의에서는 세 제공자를 나란히 비교하여 한 제공자에서 배포한 코드가 다른 제공자로 이식할 때 깨지지 않도록 합니다.

**유형:** Build
**언어:** Python (표준 라이브러리, 스키마 변환기)
**선수 요건:** 13단계 · 01강 (도구 인터페이스)
**시간:** 약 75분

## 학습 목표

- OpenAI, Anthropic, Gemini의 함수 호출 페이로드(선언, 호출, 결과)에서 세 가지 형태 차이를 설명하세요.
- 하나의 도구 선언을 세 제공자 형식으로 변환하고, 엄격 모드(strict-mode) 제약이 어디에서 달라질지 예측하세요.
- 각 제공자에서 `tool_choice`을 사용하여 도구 호출을 강제, 금지 또는 자동 선택하세요.
- 제공자별 하드 한도(도구 수, 스키마 깊이, 인자 길이)와 한도를 위반할 때 각 제공자가 생성하는 오류 시그니처를 파악하세요.

## 문제점

함수 호출 요청의 형태는 제공자마다 다릅니다. 2026년 프로덕션 스택에서 가져온 세 가지 구체적인 예시입니다:

**OpenAI Chat Completions / Responses API.** `tools: [{type: "function", function: {name, description, parameters, strict}}]`을 전달합니다. 모델의 응답에는 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`이 포함되며, 여기서 `arguments`는 파싱해야 하는 JSON 문자열입니다. 엄격 모드(`strict: true`)는 제약 디코딩(제약 디코딩)을 통해 스키마 준수를 강제합니다.

**Anthropic Messages API.** `tools: [{name, description, input_schema}]`을 전달합니다. 응답은 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`로 돌아옵니다. `input`는 이미 파싱된 상태(문자열이 아닌 객체)입니다. `{type: "tool_result", tool_use_id, content}` 블록을 포함하는 새로운 `user` 메시지로 응답합니다.

**Google Gemini API.** `tools: [{functionDeclarations: [{name, description, parameters}]}]`을 전달합니다(`functionDeclarations` 아래에 중첩됨). 응답은 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`로 도착하며, 여기서 `id`는 Gemini 3 이상에서 병렬 호출 상관관계를 위해 고유합니다. `{functionResponse: {name, id, response}}`로 응답합니다.

루프는 동일합니다. 필드 이름, 중첩 구조, 문자열 대 객체 관례, 상관관계 메커니즘이 모두 다릅니다. OpenAI에서 날씨 에이전트를 작성한 팀은 Anthropic으로 이식하는 데 이틀, Gemini로 이식하는 데 하루를 순수한 배관(plumbing) 작업으로만 소모합니다.

이 강의는 세 가지 형식을 하나의 표준 도구 선언으로 통합하고 가장자리에서 라우팅하는 번역기를 구축합니다. 13단계 · 17강은 동일한 패턴을 LLM 게이트웨이로 일반화합니다.

## 개념

### 공통 구조

모든 제공자는 다음 다섯 가지가 필요합니다:

1. **도구 목록.** 도구별 이름, 설명 및 입력 스키마.
2. **도구 선택.** 특정 도구를 강제하거나, 도구를 금지하거나, 모델이 결정하도록 허용합니다.
3. **호출 생성.** 도구와 인수를 지정하는 구조화된 출력.
4. **호출 ID.** 응답을 올바른 호출과 연결합니다(병렬 호출에서 중요합니다).
5. **결과 주입.** 결과를 호출과 연결하는 메시지 또는 블록.

### 형식 차이, 필드별 비교

| 측면 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 선언 엔벨로프 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 스키마 필드 | `parameters` | `input_schema` | `parameters` |
| 응답 컨테이너 | 어시스턴트 메시지의 `tool_calls[]` | `tool_use` 타입의 `content[]` | `functionCall` 타입의 `parts[]` |
| 인수 타입 | 문자열화된 JSON | 파싱된 객체 | 파싱된 객체 |
| ID 형식 | `call_...` (OpenAI가 생성) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| 결과 블록 | `tool` 역할, `tool_call_id` | `tool_result` 및 `tool_use_id`가 포함된 `user` | 일치하는 `id`가 포함된 `functionResponse` |
| 도구 강제 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 도구 금지 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 엄격한 스키마 | `strict: true` | 스키마는 스키마 (항상 강제됨) | 요청 수준의 `responseSchema` |

### 실제로 마주하게 되는 한계

- **OpenAI.** 요청당 128개 도구. 스키마 깊이 5. 인수 문자열 <= 8192 바이트. 엄격 모드에서는 `$ref`가 없어야 하고, 겹치는 `oneOf`/`anyOf`/`allOf`가 없어야 하며, 모든 속성이 `required`에 나열되어야 합니다.
- **Anthropic.** 요청당 64개 도구. 스키마 깊이는 사실상 무제한이지만 실용적 한계는 10입니다. 엄격 모드 플래그가 없으며, 스키마는 계약이며 모델은 이를 따르는 경향이 있습니다.
- **Gemini.** 요청당 64개 함수. 스키마 유형은 OpenAPI 3.0 하위 집합(JSON Schema 2020-12와 약간의 차이). Gemini 3부터 병렬 호출은 고유 ID를 사용.

### `tool_choice` 동작

모두가 지원하는 세 가지 모드, 이름이 다름.

- **Auto.** 모델이 도구 또는 텍스트를 선택. 기본값.
- **Required / Any.** 모델이 최소한 하나의 도구를 호출해야 함.
- **None.** 모델이 도구를 호출하지 않아야 함.

각 제공자 고유의 모드 하나 추가:

- **OpenAI.** 이름으로 특정 도구 강제.
- **Anthropic.** 이름으로 특정 도구 강제; `disable_parallel_tool_use` 플래그가 단일 vs 다중을 구분.
- **Gemini.** `mode: "VALIDATED"`가 모델 의도와 무관하게 모든 응답을 스키마 검증기를 거침.

### 병렬 호출

OpenAI의 `parallel_tool_calls: true` (기본값)는 하나의 어시스턴트 메시지에서 여러 호출을 생성. 모두 실행하고 `tool_call_id`당 하나의 항목을 포함하는 배치된 도구 역할 메시지로 응답. Anthropic은 역사적으로 단일 호출만 지원; `disable_parallel_tool_use: false` (Claude 3.5부터 기본값)가 다중 호출을 활성화. Gemini 2는 병렬 호출을 허용했으나 안정적 ID를 제공하지 않음; Gemini 3는 UUID를 추가하여 순서가 뒤섞인 응답을 깔끔하게 매칭.

### 스트리밍

세 가지 모두 스트리밍 도구 호출을 지원. 전송 형식이 다름:

- **OpenAI.** `tool_calls[i].function.arguments`의 델타 청크가 점진적으로 도착. `finish_reason: "tool_calls"`까지 누적.
- **Anthropic.** 블록 시작 / 블록 델타 / 블록 중지 이벤트. `input_json_delta` 청크가 부분적인 인수를 포함.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 신규)가 `functionCallId`를 포함하는 청크를 생성하여 여러 병렬 호출이 교차 가능.

13단계 · 03강은 병렬 + 스트리밍 재조립을 심층적으로 다룸. 이 강의는 선언과 단일 호출 형태에 집중.

### 오류 및 복구

잘못된 인수 오류도 형태가 다름.

- **OpenAI (non-strict).** 모델이 `arguments: "{bad json}"`를 반환, JSON 파싱 실패, 오류 메시지를 주입하고 재호출.
- **OpenAI (strict).** 디코딩 중 검증 수행; 잘못된 JSON은 불가능하지만 `refusal`가 나타날 수 있음.
- **Anthropic.** `input`에 예상치 못한 필드가 포함될 수 있음; 스키마는 권고 사항. 서버 측에서 검증.
- **Gemini.** OpenAPI 3.0의 특이점: `enum`은 객체 필드에서 조용히 무시되므로, 직접 검증해야 합니다.

### 번역 패턴

코드의 표준 도구 선언은 다음과 같은 형태입니다 (형태는 선택합니다):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

세 개의 작은 함수가 이를 세 가지 제공자 형태로 번역합니다. `code/main.py`의 하네스는 정확히 이 작업을 수행한 후, 각 제공자의 응답 형태를 통해 가짜 도구 호출을 왕복 처리합니다. 네트워크는 필요하지 않습니다 — 이 강의는 형태를 가르치며, HTTP를 가르치지 않습니다.

프로덕션 팀은 이 번역기를 `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), 또는 `BaseTool` (LlamaIndex)로 감쌉니다. 13단계 · 17강은 세 가지 중 하나 앞에 OpenAI 형태의 API를 노출하는 게이트웨이를 출시합니다.

```figure
function-call-args
```

## 사용하기

`code/main.py`은 하나의 표준 `Tool` 데이터 클래스와 OpenAI, Anthropic, Gemini 선언 JSON을 생성하는 세 개의 번역기를 정의합니다. 이후 각 형태의 손으로 만든 제공자 응답을 동일한 표준 호출 객체로 파싱하여, 내부적으로 의미가 동일함을 입증합니다. 실행하여 세 선언을 나란히 비교해 보세요.

확인할 사항:

- 세 선언 블록은 엔벨로프와 필드 이름만 다릅니다.
- 세 응답 블록은 호출이 위치하는 곳(최상위 `tool_calls`, `content[]` 블록, `parts[]` 항목)이 다릅니다.
- 하나의 `canonical_call()` 함수가 세 가지 응답 형태 모두에서 `{id, name, args}`을 추출합니다.

## 출시하기

이 강의는 `outputs/skill-provider-portability-audit.md`을 생성합니다. 하나의 제공자에 대한 함수 호출 통합이 주어지면, 이 스킬은 이식성 감사를 생성합니다: 어떤 제공자 제한에 의존하는지, 어떤 필드를 이름 변경해야 하는지, 다른 제공자로 이식할 때 무엇이 깨지는지 알려줍니다.

## 연습 문제

1. `code/main.py`을 실행하여 세 제공자 선언 JSON이 모두 동일한 기본 `Tool` 객체를 직렬화하는지 확인하세요. 표준 도구에 열거형 매개변수를 추가하고, OpenAPI 특이점을 처리해야 하는 번역기가 Gemini뿐임을 확인하세요.

2. 각 제공자에 대해 `ListToolsResponse` 파서를 추가하여, `list_tools` 또는 발견 호출 후 모델이 반환하는 도구 목록을 추출하세요. OpenAI는 이를 네이티브로 제공하지 않으므로, 이 비대칭성을 기록하세요.

3. `tool_choice` 변환을 구현하세요: 표준 `ToolChoice(mode="force", tool_name="x")`을 세 공급자의 모든 형식으로 매핑합니다. 그 다음 `mode="any"`와 `mode="none"`을 매핑하세요. 강의의 diff 표를 확인하세요.

4. 세 공급자 중 하나를 선택하여 함수 호출 가이드를 처음부터 끝까지 읽어보세요. 스키마 사양에서 다른 두 공급자가 지원하지 않는 필드를 하나 찾아보세요. 후보: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. 테스트 벡터를 작성하세요: 선언된 스키마를 위반하는 인수를 가진 도구 호출입니다. 각 공급자의 검증기(01강의 표준 라이브러리 검증기를 대리자로 사용)로 실행하고 어떤 오류가 발생하는지 기록하세요. 엄격성을 위해 프로덕션에서 사용할 공급자를 문서화하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 함수 호출(Function Calling) | "도구 사용" | 구조화된 도구 호출 생성을 위한 공급자 수준 API |
| 도구 선언 | "도구 사양" | 이름 + 설명 + JSON Schema 입력 페이로드 |
| `tool_choice` | "강제 / 금지" | 자동 / 필수 / 없음 / 특정 이름 모드 |
| 엄격 모드 | "스키마 강제" | 디코딩을 스키마에 일치하도록 제한하는 OpenAI 플래그 |
| `tool_use` 블록 | "Anthropic의 호출 형식" | id, name, input을 포함하는 인라인 콘텐츠 블록 |
| `functionCall` 파트 | "Gemini의 호출 형식" | name, args, id를 포함하는 `parts[]` 항목 |
| 문자열형 인수 | "문자열화된 JSON" | OpenAI는 인수를 객체가 아닌 JSON 문자열로 반환합니다 |
| 병렬 도구 호출 | "한 턴에서의 팬아웃" | 하나의 어시스턴트 메시지 내의 여러 도구 호출 |
| 거부 | "모델이 거절함" | 호출 대신 엄격 모드 전용 거부 블록 |
| OpenAPI 3.0 하위 집합 | "Gemini 스키마 특이점" | Gemini는 약간의 차이가 있는 JSON-Schema 유사 방언을 사용합니다 |

## 추가 읽기

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 엄격 모드 및 병렬 호출을 포함한 표준 참조
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 및 `tool_result` 블록 의미론
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 병렬 호출, 고유 id 및 OpenAPI 하위 집합
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini의 엔터프라이즈 표면
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 엄격 모드 스키마 강제 세부 사항
