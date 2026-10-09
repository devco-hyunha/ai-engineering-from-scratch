# OpenTelemetry GenAI 시맨틱 컨벤션

> OpenTelemetry의 GenAI SIG (2024년 4월 출시)는 에이전트 텔레메트리를 위한 표준 스키마를 정의합니다. 스팬 이름, 속성, 콘텐츠 캡처 규칙이 벤더 전반에 걸쳐 수렴하므로 에이전트 추적이 Datadog, Grafana, Jaeger, Honeycomb에서 동일한 의미를 가집니다.

**유형:** Learn + Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 13강 (LangGraph), 14단계 · 24강 (관측 가능성 플랫폼)
**시간:** 약 60분

## 학습 목표

- GenAI 스팬 카테고리: 모델/클라이언트, 에이전트, 도구를 나열해 보세요.
- `invoke_agent` CLIENT 스팬과 INTERNAL 스팬을 구분하고 각각이 적용되는 시점을 설명해 보세요.
- GenAI 최상위 속성: 제공자 이름, 요청 모델, 데이터 소스 ID를 나열해 보세요.
- 콘텐츠 캡처 계약: 옵트인, `OTEL_SEMCONV_STABILITY_OPT_IN`, 외부 참조 권장 사항을 설명해 보세요.

## 문제점

모든 벤더가 자체 스팬 이름을 발명합니다. 운영 팀은 프레임워크별 대시보드를 구축해야 합니다. OpenTelemetry의 GenAI SIG는 전체 생태계가 따르는 하나의 표준을 정의하여 이 문제를 해결합니다.

## 개념

### 스팬 카테고리

1. **모델 / 클라이언트 스팬.** 원시 LLM 호출을 커버합니다. 제공자 SDK (Anthropic, OpenAI, Bedrock) 및 프레임워크 모델 어댑터에 의해 방출됩니다.
2. **에이전트 스팬.** `create_agent` (에이전트가 구성될 때) 및 `invoke_agent` (실행될 때).
3. **도구 스팬.** 도구 호출마다 하나씩 존재하며, 부모-자식 관계로 에이전트 스팬에 연결됩니다.

### 에이전트 스팬 이름 지정

- 스팬 이름: 이름이 지정된 경우 `invoke_agent {gen_ai.agent.name}`; 지정되지 않은 경우 `invoke_agent`로 대체됩니다.
- 스팬 종류:
  - **CLIENT** — 원격 에이전트 서비스 (OpenAI Assistants API, Bedrock Agents)용.
  - **INTERNAL** — 프로세스 내 에이전트 프레임워크 (LangChain, CrewAI, 로컬 ReAct)용.

### 핵심 속성

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — 모델 ID.
- `gen_ai.response.model` — 해결된 모델 (라우팅으로 인해 요청 모델과 다를 수 있음).
- `gen_ai.agent.name` — 에이전트 식별자.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — RAG의 경우, 참조한 코퍼스나 저장소.

Anthropic, Azure AI Inference, AWS Bedrock, OpenAI에는 기술별 관례가 존재합니다.

### 콘텐츠 캡처

기본 규칙: 계측은 기본적으로 입력/출력을 캡처하지 않아야(SHOULD NOT) 합니다. 캡처는 다음을 통해 옵트인(opt-in)합니다:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

권장되는 프로덕션 패턴: 콘텐츠를 외부(S3, 로그 저장소)에 저장하고, 스팬에 참조를 기록합니다(문서가 아닌 포인터 ID). 이는 관측 가능성에 연결된 27강의 콘텐츠 오염 방어입니다.

### 안정성

2026년 3월 기준 대부분의 관례는 실험적입니다. 다음을 통해 안정 프리뷰에 옵트인하세요:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+는 GenAI 속성을 LLM Observability 스키마에 네이티브로 매핑합니다. 다른 백엔드(Grafana, Honeycomb, Jaeger)는 원시 속성을 지원합니다.

### 이 패턴이 잘못되는 지점

- **스팬에 전체 프롬프트를 캡처하는 것.** PII, 비밀, 고객 데이터가 운영자가 읽을 수 있는 추적에 포함됩니다. 외부에 저장하세요.
- **`gen_ai.provider.name`가 없는 것.** 귀속(attribution)이 없으면 다중 제공자 대시보드가 깨집니다.
- **부모 링크가 없는 스팬.** 고립된 도구 스팬이 됩니다. 항상 컨텍스트를 전파하세요.
- **안정성 옵트인을 설정하지 않는 것.** 백엔드 업그레이드 시 속성이 이름이 바뀔 수 있습니다.

```figure
ae-genai-span-tree
```

## 구현하기

`code/main.py`는 GenAI 관례와 일치하는 stdlib 스팬 방출기를 구현합니다:

- GenAI 속성 스키마가 있는 `Span`.
- `start_span`가 있는 `Tracer`, 중첩된 컨텍스트.
- `create_agent`, `invoke_agent` (INTERNAL), 도구별 스팬, LLM 호출용 `chat` 스팬을 방출하는 스크립트된 에이전트 실행.
- 프롬프트를 외부에 저장하고 스팬에 ID를 기록하는 콘텐츠 캡처 모드.

실행하세요:

```
python3 code/main.py
```

출력: 모든 필수 GenAI 속성이 포함된 스팬 트리와 옵트인된 콘텐츠 참조가 표시된 "외부 저장소".

## 사용하기

- **Datadog LLM Observability** (v1.37+)는 속성을 네이티브로 매핑합니다.
- **Langfuse / Phoenix / Opik** (24강) — 생태계를 자동 계측합니다.
- **Jaeger / Honeycomb / Grafana Tempo** — 원시 OTel 추적; GenAI 속성으로 대시보드 구축.
- **셀프 호스트** — GenAI 프로세서가 포함된 OTel Collector 실행.

## 출시하기

`outputs/skill-otel-genai.md`는 기존 에이전트에 OTel GenAI 스팬을 연결하며, 콘텐츠 캡처 기본값과 외부 참조 저장을 적용합니다.

## 연습 문제

1. `invoke_agent` (INTERNAL) + 도구별 스팬으로 01강의 ReAct 루프를 계측해 보세요. Jaeger 인스턴스로 전송합니다.
2. "참조 전용" 모드에서 콘텐츠 캡처를 추가합니다: 프롬프트는 SQLite에 저장하고, 스팬 속성에는 행 ID만 포함합니다.
3. `gen_ai.data_source.id`의 사양을 읽어보고, 09강의 Mem0 검색에 연결해 보세요.
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`를 설정하고, 수집기가 속성을 이름 변경하지 않는지 확인합니다.
5. GenAI 속성만으로 "어떤 도구 오류가 어떤 모델과 상관관계가 있는지"를 보여주는 대시보드를 구축합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 그룹" | 스키마를 정의하는 OTel 워킹 그룹 |
| invoke_agent | "에이전트 스팬" | 에이전트 실행을 나타내는 스팬의 이름 |
| CLIENT 스팬 | "원격 호출" | 원격 에이전트 서비스 호출에 대한 스팬 |
| INTERNAL 스팬 | "프로세스 내" | 프로세스 내 에이전트 실행에 대한 스팬 |
| gen_ai.provider.name | "제공자" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 소스" | 검색이 도달한 코퍼스/저장소 |
| 콘텐츠 캡처 | "프롬프트 로깅" | 메시지 선택적 캡처; 프로덕션에서는 외부에 저장 |
| 안정성 옵트인 | "미리보기 모드" | 실험적 관례를 고정하는 환경 변수 |

## 추가 읽기

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 사양
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 기본값으로 GenAI 스팬
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 내장 OTel 스팬
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 추적 컨텍스트 전파
