# 도구와 함께 병렬 도구 호출 및 스트리밍

> 세 개의 독립적인 날씨 조회를 직렬로 실행하면 세 번의 왕복이 필요합니다. 병렬로 실행하면 총 시간이 가장 느린 단일 호출 시간으로 수렴합니다. 모든 프론티어 제공자는 이제 한 턴에서 여러 도구 호출을 생성합니다. 이점은 분명하지만, 구현은 미묘합니다. 이 강의는 병렬 분산(fan-out)과 스트리밍된 인자 재조립, 그리고 id 상관관계 함정에 중점을 두어 두 측면을 모두 다룹니다.

**유형:** Build
**언어:** Python (표준 라이브러리, 스레드 풀 + 스트리밍 하네스)
**선수 요건:** 13단계 · 02강 (함수 호출 심층 분석)
**시간:** 약 75분

## 학습 목표

- `parallel_tool_calls: true`가 존재하는 이유와 이를 비활성화해야 하는 시점을 설명해 보세요.
- 병렬 분산(fan-out) 중에 스트리밍된 인자 청크를 올바른 도구 호출 id와 상관관계 지을 수 있습니다.
- 조기 파싱 없이 부분적인 `arguments` 문자열을 완전한 JSON으로 재조립할 수 있습니다.
- 순차적 지연과 병렬 지연을 시연하는 세 도시 날씨 벤치마크를 실행할 수 있습니다.

## 문제점

병렬 호출이 없으면, "벵갈루루, 도쿄, 취리히의 날씨는 어떤가요?"라는 질문에 답하는 에이전트는 다음과 같이 동작합니다:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

세 번의 LLM 왕복이 필요하며, 각 왕복은 실행기(executor) 지연도 포함합니다. 이상적인 벽시계(wall-clock) 시간의 약 4배가 됩니다.

병렬 호출을 사용하면:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

한 번의 LLM 왕복이 필요합니다. 실행기 시간은 세 호출의 합이 아니라 최대값입니다. OpenAI, Anthropic, Gemini에서의 생산 벤치마크는 분산(fan-out) 워크로드에서 벽시계 시간이 60~70% 감소함을 보여줍니다.

대가는 상관관계 복잡성입니다. 세 호출이 순서대로 완료되지 않을 때, 모델이 결과를 정렬할 수 있도록 결과에 일치하는 `tool_call_id`가 포함되어야 합니다. 결과가 스트리밍될 때, 실행하기 전에 부분적인 인자 조각을 완전한 JSON으로 조립해야 합니다. Gemini 3는 두 개의 병렬 호출이 같은 도구일 때 구별할 수 없는 실제 문제를 해결하기 위해 부분적으로 고유 id를 추가했습니다.

## 개념

### 병렬 활성화

- **OpenAI.** `parallel_tool_calls: true`가 기본적으로 켜져 있습니다. `false`를 설정하여 직렬로 강제할 수 있습니다.
- **Anthropic.** `disable_parallel_tool_use: false`를 통해 병렬로 실행합니다 (Claude 3.5 이상에서 기본 켜짐). 직렬로 실행하려면 `true`를 설정하세요.
- **Gemini.** 항상 병렬 처리가 가능하며, `tool_config.function_calling_config.mode = "AUTO"`이 모델의 결정을 허용합니다.

도구 호출에 순서 의존성이 있는 경우(`create_file` 다음 `write_file`), 한 호출의 출력이 다른 호출의 입력에 영향을 주는 경우, 또는 속도 제한기가 팬아웃(fan-out)을 처리할 수 없는 경우 병렬 처리를 비활성화하세요.

### ID 상관관계

모델이 생성하는 모든 호출에는 `id`이 있습니다. 호스트가 반환하는 모든 결과에는 동일한 id가 포함되어야 합니다. 이 값이 없으면 결과가 모호해집니다.

- **OpenAI.** 각 도구 역할(tool-role) 메시지에 `tool_call_id`을 포함합니다.
- **Anthropic.** 각 `tool_result` 블록에 `tool_use_id`을 포함합니다.
- **Gemini.** 각 `functionResponse`에 `id`을 포함합니다 (Gemini 3 이상; Gemini 2는 이름으로 매칭했으며, 이는 동일한 이름의 병렬 호출에서 문제가 발생했습니다).

### 호출을 동시에 실행하기

호스트는 각 호출의 실행기를 자체 스레드, 코루틴(coroutine), 또는 원격 워커에서 실행합니다. 가장 단순한 하네스(harness)는 스레드 풀을 사용하며, 프로덕션 환경에서는 `asyncio.gather` 또는 구조적 동시성(structured concurrency)을 사용하는 asyncio를 활용합니다. 완료 순서는 예측할 수 없으므로 id가 식별자입니다.

흔한 버그 중 하나는 호출 목록 순서로 결과를 반환하는 것입니다. 모델은 `tool_call_id`만 중요하게 여기므로 보통은 잘 작동하지만, 결과가 누락되거나 중복될 경우 순서가 뒤섞이면 디버깅이 어려워집니다. 명시적인 id와 함께 완료 순서로 응답하는 것을 권장합니다.

### 스트리밍 도구 호출

모델이 스트리밍할 때 `arguments`은 조각으로 도착합니다. 세 개의 병렬 호출에 대한 세 개의 청크 스트림이 네트워크 상에서 뒤섞입니다. 각 id에 대해 하나의 누적기(accumulator)가 필요합니다.

제공자별 형태:

- **OpenAI.** 각 청크는 `choices[0].delta.tool_calls[i].function.arguments` (부분 문자열)입니다. 청크는 `index` (호출 목록에서의 위치)를 포함합니다. 인덱스별로 누적하고, `id`가 처음 나타날 때 읽으며, `finish_reason = "tool_calls"`일 때 JSON을 파싱합니다.
- **Anthropic.** 스트림 이벤트는 `message_start`이며, 그 후 각 블록에 대해 `tool_use` 타입의 `content_block_start`가 하나씩 옵니다 (id, name, 빈 입력 포함). `input_json_delta` 청크는 `content_block_delta` 이벤트에 포함됩니다. `content_block_stop`가 각 블록을 닫습니다.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 이상)는 `functionCallId`을 포함하는 청크를 방출하여 호출이 깔끔하게 뒤섞이도록 합니다. Gemini 3 이전에는 한 번에 하나의 완전한 호출만 스트리밍으로 반환했습니다.

### 부분 JSON과 조기 파싱 함정

`arguments`이 완전히 완료될 때까지는 파싱할 수 없습니다. `{"city": "Beng`와 같은 부분 JSON은 유효하지 않으며 예외를 발생시킵니다. 올바른 게이트는 제공자의 호출 종료 신호입니다: OpenAI의 `finish_reason = "tool_calls"`, Anthropic의 `content_block_stop`, 또는 Gemini의 스트림 종료 이벤트입니다. 그 후에만 `json.loads`를 시도해 보세요. 더 견고한 접근법은 구조가 완성될 때 이벤트를 생성하는 증분 JSON 파서를 사용하는 것입니다. OpenAI의 스트리밍 가이드는 실시간 "생각 중" 표시를 보여주는 UX를 위해 이 방법을 권장합니다. 중괄호 개수 세기는 완전성 테스트로는 신뢰할 수 없습니다(따옴표 안의 문자열이나 이스케이프된 내용으로 인해 오탐이 발생하므로) 비공식적인 디버깅 휴리스틱으로만 사용해야 합니다.

### 순서대로 완료되지 않는 경우

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

호스트 응답은 여전히 id를 인용해야 합니다:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

OpenAI나 Anthropic에서는 응답의 순서가 정확성에 영향을 미치지 않습니다. Gemini는 id가 일치하기만 하면 어떤 순서도 허용합니다.

### 벤치마크: 순차적 vs 병렬

`code/main.py`의 하네스는 400, 600, 800 ms 지연을 가진 세 실행기를 시뮬레이션합니다. 순차적 실행은 총 1800 ms가 걸립니다. 병렬 실행은 max(400, 600, 800) = 800 ms가 걸립니다. 차이는 비례적이지 않고 일정하므로, 도구 수가 증가할수록 절약되는 시간이 커집니다.

실제 환경 주의사항: 병렬 호출은 다운스트림 API에 부담을 줍니다. 속도 제한이 있는 서비스로 10-way 팬아웃을 수행하면 실패합니다. 13단계 · 17강은 게이트웨이 수준의 백프레셔를 다루며, 재시도 시맨틱은 향후 단계에서 계획되어 있습니다.

### 스트리밍 팬아웃의 실제 시간(wall-clock)

모델 자체가 스트리밍을 수행한다면, 모든 호출이 확정될 때까지 기다리지 않고 한 호출의 인수가 완료되는 즉시 실행을 시작할 수 있습니다. 이는 OpenAI가 문서화한 최적화이지만 모든 SDK가 이를 노출하지는 않습니다. 이 강의의 하네스는 이를 수행합니다: 시뮬레이션된 스트림이 완전한 인수 객체를 생성하는 즉시, 호스트가 해당 호출을 시작합니다.

```figure
tp-parallel-fanout
```

## 사용하기

`code/main.py`은 두 부분으로 나뉩니다. 첫 번째 부분은 `concurrent.futures.ThreadPoolExecutor`을 사용하여 세 개의 시뮬레이션된 날씨 호출을 순차적으로 및 병렬로 실행하고 실제 시간(wall-clock time)을 출력합니다. 두 번째 부분은 가짜 스트리밍 응답을 재생합니다 — 하나의 스트림에 세 개의 병렬 호출이 인터리브(interleaved)된 `arguments` 청크들 — `StreamAccumulator`를 사용하여 id별로 재조립합니다. LLM도 네트워크도 없으며, 오직 재조립 로직만 수행합니다.

확인할 사항:

- 순차적 타이머는 1.8초를 기록합니다. 병렬 타이머는 동일한 가짜 지연에서 0.8초를 기록합니다.
- 누적기는 id별로 버퍼링하여 순서가 뒤섞인 청크를 처리하며, 각 호출의 JSON이 완성되었을 때만 파싱합니다.
- 실행기는 모든 스트림이 끝난 후가 아니라, 특정 id의 인수가 확정되는 즉시 시작합니다.

## 출시하기

이 강의는 `outputs/skill-parallel-call-safety-check.md`를 생성합니다. 도구 레지스트리가 주어지면, 스킬은 병렬화할 수 있는 도구, 순서 의존성이 있는 도구, 다운스트림 속도 제한을 압도할 도구를 감사하여, 도구별 `parallel_safe` 플래그가 포함된 수정된 레지스트리를 반환합니다.

## 연습 문제

1. `code/main.py`를 실행하고 시뮬레이션된 지연 시간을 변경해 보세요. 병렬 대 순차 비율이 대략 `max/sum`임을 확인하세요. (실제 실행은 스레드 스케줄링, 직렬화, 하네스 오버헤드 때문에 이상적인 값에서 약간 벗어납니다.) 어떤 지연 시간 분포에서 병렬 처리가 중요하지 않게 되나요?

2. 누적기를 확장하여 "호출이 스트림 중간에 취소됨" 케이스를 처리해 보세요. 버퍼를 버리고 `cancelled` 이벤트를 방출합니다. 어떤 제공자가 이 케이스를 명시적으로 문서화하나요? Anthropic의 `content_block_stop` 의미론과 OpenAI의 `finish_reason: "length"` 동작을 확인하세요.

3. 스레드 풀을 `asyncio.gather`로 교체하고 두 가지를 벤치마킹하세요. 실행기가 실제 I/O를 수행하는 경우에만, 컨텍스트 스위칭 비용이 낮아 async에서 작은 이점을 볼 수 있어야 합니다.

4. 병렬화하면 안 되는 두 도구를 선택하세요 (예: `create_file` 후 `write_file`). 레지스트리에 `ordering_dependency` 그래프를 추가하고, 병렬 팬아웃을 해당 그래프에 게이트하세요. 이는 의존성 인식 스케줄링을 위한 최소한의 메커니즘이며, 향후 에이전트 엔지니어링 단계에서 이를 공식화합니다.

5. OpenAI의 병렬 함수 호출 섹션과 Anthropic의 `disable_parallel_tool_use` 문서를 읽어보세요. Anthropic이 병렬화를 비활성화할 것을 권장하는 실제 도구 유형 하나를 식별하세요. (힌트: 동일한 리소스에 대한 중대한 변경.)

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 병렬 도구 호출 | "한 턴에서의 팬아웃" | 모델이 단일 어시스턴트 메시지에서 여러 도구 호출을 방출 |
| `parallel_tool_calls` | "OpenAI의 플래그" | 다중 호출 방출을 활성화하거나 비활성화 |
| `disable_parallel_tool_use` | "Anthropic의 역 플래그" | 옵트아웃 플래그; 기본값은 병렬 활성화 |
| 도구 호출 id | "상관 핸들" | 결과 메시지가 에코해야 하는 호출별 식별자 |
| 누적기 | "스트림 버퍼" | 부분 `arguments` 청크용 id별 문자열 버퍼 |
| 순서 외 완료 | "가장 빠른 순서로 먼저" | 병렬 호출이 예측 불가능한 순서로 완료되며, id가 연결 고리 역할을 합니다 |
| 의존성 그래프 | "순서 제약 조건" | 출력값이 다른 도구의 입력값으로 이어지는 도구들; 병렬화할 수 없습니다 |
| 조기 파싱 함정 | "JSON.parse 폭발" | 불완전한 `arguments` 문자열을 파싱하려는 시도 |
| `streamFunctionCallArguments` | "Gemini 3 기능" | 호출마다 고유 id가 있는 스트리밍 인자 청크 |
| 완료 순서 응답 | "모두를 기다리지 마세요" | id를 키로 사용하여 결과가 도착하는 즉시 응답합니다 |

## 추가 읽기

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 기본 동작과 옵트아웃 플래그
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` 및 결과 배치
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3의 id 상관 병렬 호출
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 스트림용 청크 인자 재조립
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `input_json_delta`을 사용한 `content_block_delta`
