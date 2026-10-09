# OpenTelemetry GenAI — 도구 호출을 엔드투엔드로 추적하기

> 에이전트가 다섯 개의 도구, 세 개의 MCP 서버, 두 개의 하위 에이전트를 호출합니다. 이 모든 것을 하나의 추적(trace)으로 연결해야 합니다. OpenTelemetry GenAI 시맨틱 컨벤션(v1.37 이상에서 안정화된 속성)은 2026년 표준이며, Datadog, Langfuse, Arize Phoenix, OpenLLMetry, AgentOps가 네이티브로 지원합니다. 이 강의에서는 필수 속성을 명시하고, 스팬 계층 구조(에이전트 → LLM → 도구)를 순회하며, 임의의 OTel 익스포터에 연결할 수 있는 스탠다드 라이브러리 스팬 에미터를 제공합니다.

**유형:** Build
**언어:** Python (stdlib, OTel 스팬 에미터)
**선수 요건:** 13단계 · 07강 (MCP 서버), 13단계 · 08강 (MCP 클라이언트)
**시간:** 약 75분

## 학습 목표

- LLM 스팬과 도구 실행 스팬에 필요한 OTel GenAI 속성을 명시해 보세요.
- 에이전트 루프, LLM 호출, 도구 호출, MCP 클라이언트 디스패치를 포함하는 추적 계층 구조를 구축해 보세요.
- 포함할 콘텐츠(옵트인)와 삭제할 콘텐츠(기본값)를 결정해 보세요.
- 도구 코드를 수정하지 않고도 로컬 수집기(Jaeger, Langfuse)에 스팬을 전송해 보세요.

## 문제점

2026년 2월의 디버깅 사례입니다. 사용자가 "에이전트가 어떤 때는 30초, 어떤 때는 3초 만에 응답한다"고 보고합니다. 추적(trace)이 없습니다. 로그에는 LLM 호출만 있고, 도구 디스패치, MCP 서버 왕복, 하위 에이전트 호출은 기록되지 않습니다. 추측만 하다가 결국 한 MCP 서버가 콜드 스타트(cold-start) 때 간헐적으로 멈추는 것을 발견합니다.

엔드투엔드 추적이 없으면 이런 문제를 찾을 수 없습니다. OTel GenAI가 이를 해결합니다.

이 컨벤션은 2025-2026년 OpenTelemetry 시맨틱 컨벤션 그룹 아래에서 확정되었습니다. 안정화된 속성 이름을 정의하여 Datadog, Langfuse, Phoenix, OpenLLMetry, AgentOps가 동일한 스팬을 파싱할 수 있게 합니다. 한 번 계측(instrument)하면 어떤 백엔드로도 전송할 수 있습니다.

## 개념

### 스팬 계층 구조

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

전체 구조가 하나의 추적 ID 아래에 중첩됩니다. 스팬 ID가 부모-자식 관계를 연결합니다.

### 필수 속성

2025-2026 시맨틱 컨벤션에 따르면:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — 요청된 모델 문자열 (예: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — 실제로 서빙된 모델.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — 상관관계(correlation)를 위한 제공자 응답 ID.

도구(span)의 경우:

- `gen_ai.tool.name` — 도구 식별자.
- `gen_ai.tool.call.id` — 특정 호출 ID.
- `gen_ai.tool.description` — 도구 설명 (선택 사항).

에이전트(span)의 경우:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span 종류

- 프로세스 경계를 넘나드는 호출(LLM 제공자, MCP 서버)에 대해 `SpanKind.CLIENT`를 사용합니다.
- 에이전트의 자체 루프 단계 및 도구 실행에 대해 `SpanKind.INTERNAL`를 사용합니다.

### 선택적(content capture) 콘텐츠 캡처

기본적으로 span은 지표(metrics)와 타이밍(timing)만 포함하며 프롬프트나 완성(completions)은 포함하지 않습니다. 큰 페이로드와 PII는 기본적으로 꺼져 있습니다. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 및 특정 콘텐츠 캡처 환경 변수를 설정하여 콘텐츠를 포함하세요. 프로덕션 환경에서 활성화하기 전에 주의 깊게 검토해 보세요.

### Span의 이벤트

토큰 단위 이벤트는 span 이벤트로 추가할 수 있습니다:

- `gen_ai.content.prompt` — 입력 메시지.
- `gen_ai.content.completion` — 출력 메시지.
- `gen_ai.content.tool_call` — 기록된 도구 호출.

자세한 재생(replay)을 위해 span 내에서 이벤트를 시간 순서대로 정렬합니다.

### Exporters

OTel span은 다음으로 export됩니다:

- **Jaeger / Tempo.** OSS, 온프레미스.
- **Langfuse.** LLM 관측성 특화; 토큰 사용량을 시각화합니다.
- **Arize Phoenix.** 평가(Evals)와 추적(tracing)이 결합된 형태.
- **Datadog.** 상용; `gen_ai.*` 속성을 네이티브로 파싱합니다.
- **Honeycomb.** 열(column) 지향; 쿼리가 친화적입니다.

모두 OTLP, 즉 와이어 포맷을 따릅니다. 코드에서는 이를 신경 쓸 필요가 없습니다.

### MCP 전반에 걸친 전파

MCP 클라이언트가 서버를 호출할 때, 요청에 W3C traceparent 헤더를 주입(inject)합니다. Streamable HTTP는 표준 헤더를 지원합니다. Stdio는 HTTP 헤더를 네이티브로 전달하지 않으며, 스펙의 2026 로드맵은 JSON-RPC 호출에 `_meta.traceparent` 필드를 추가하는 것을 논의하고 있습니다.

해당 기능이 출시될 때까지: 모든 요청의 `_meta`에 traceparent를 수동으로 포함하세요. 서버는 trace ID를 로깅합니다.

### 메트릭

스팬과 함께, GenAI semconv는 메트릭을 정의합니다:

- `gen_ai.client.token.usage` — 히스토그램.
- `gen_ai.client.operation.duration` — 히스토그램.
- `gen_ai.tool.execution.duration` — 히스토그램.

호출별 세부 정보가 필요하지 않은 대시보드에 이 메트릭을 사용해 보세요.

### AgentOps 계층

AgentOps (2024년 설립)는 GenAI 관측 가능성에 특화되어 있습니다. LangGraph, Pydantic AI, CrewAI 등 인기 있는 프레임워크를 감싸서 OTel 스팬을 자동으로 생성합니다. 스택이 지원되는 프레임워크를 사용하는 경우 유용하며, 그렇지 않은 경우 수동 계측을 사용해 보세요.

```figure
t3-span-waterfall
```

## 사용하기

`code/main.py`는 LLM을 호출하고, 두 개의 도구를 디스패치하며, 한 번의 MCP 왕복 요청을 수행하는 에이전트에 대해 stdout으로 OTel 형식의 스팬을 생성합니다 (OTLP-JSON 유사 형식). 실제 익스포터는 없으며, 이 강의는 스팬 형식과 속성 세트에 집중합니다. 출력 내용을 OTLP 호환 뷰어에 붙여넣거나 단순히 읽어 보세요.

확인할 사항:

- 추적 ID는 모든 스팬에 공유됩니다.
- 부모-자식 링크는 `parentSpanId`를 통해 인코딩됩니다.
- 필수 `gen_ai.*` 속성이 채워집니다.
- 콘텐츠 캡처는 기본적으로 꺼져 있으며, 한 시나리오가 환경 변수를 통해 이를 켭니다.

## 출시하기

이 강의는 `outputs/skill-otel-genai-instrumentation.md`를 생성합니다. 에이전트 코드베이스가 주어지면, 이 스킬은 계측 계획을 생성합니다: 스팬을 추가할 위치, 채워야 할 속성, 타겟할 익스포터.

## 연습 문제

1. `code/main.py`를 실행하세요. 스팬을 세고 CLIENT와 INTERNAL을 식별하세요.

2. 콘텐츠 캡처를 켜고 (환경 변수) `gen_ai.content.prompt` 및 `gen_ai.content.completion` 이벤트가 나타나는지 확인하세요. PII에 대한 영향을 주의하세요.

3. 도구 실행 메트릭 `gen_ai.tool.execution.duration`를 추가하고 호출당 히스토그램 샘플로 생성하세요.

4. 부모 에이전트 스팬에서 MCP 요청의 `_meta.traceparent` 필드로 traceparent를 전파하세요. MCP 서버가 동일한 추적 ID를 볼 수 있는지 확인하세요.

5. OTel GenAI semconv 사양을 읽어 보세요. 이 강의의 코드가 NOT 생성하는 semconv에 나열된 속성 하나를 식별하세요. 이를 추가하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 스팬, 메트릭, 로그를 위한 개방 표준 |
| GenAI semconv | "GenAI 시맨틱 컨벤션" | LLM / 도구 / 에이전트 스팬에 대한 안정적 속성 이름 |
| `gen_ai.*` | "속성 네임스페이스" | 모든 GenAI 속성은 이 접두사를 공유합니다 |
| Span | "시간 측정 작업" | 시작, 종료 및 속성을 가진 작업 단위 |
| Trace | "스팬 간 계보" | 동일한 trace id를 공유하는 스팬의 트리 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 스팬 방향에 대한 힌트 |
| OTLP | "OpenTelemetry Line Protocol" | 익스포터용 와이어 형식 |
| Opt-in content | "프롬프트 / 완료 내용 캡처" | 기본적으로 꺼져 있으며, 환경 변수로 활성화 |
| traceparent | "W3C 헤더" | 서비스 간에 trace 컨텍스트를 전파 |
| Exporter | "백엔드 전용 전송 컴포넌트" | 스팬을 Jaeger / Datadog 등으로 전송하는 컴포넌트 |

## 추가 읽기

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 스팬, 메트릭 및 이벤트에 대한 표준 컨벤션
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 및 도구 실행 스팬 속성 목록
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 에이전트 수준의 `invoke_agent` 스팬
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub에서 호스팅되는 공식 출처
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 프로덕션 통합 가이드
