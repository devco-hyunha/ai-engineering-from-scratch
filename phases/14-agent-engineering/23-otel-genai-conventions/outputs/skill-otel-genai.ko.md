---
name: otel-genai
description: OpenTelemetry GenAI 시맨틱 컨벤션으로 에이전트를 계측합니다 — invoke_agent, chat, tool_call 스팬을 올바른 속성과 옵트인 콘텐츠 캡처와 함께 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 23강
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

에이전트 런타임이 주어지면 OTel GenAI 시맨틱 컨벤션을 연결합니다.

생성물:

1. 에이전트 실행마다 `invoke_agent` 스팬을 생성합니다. 원격 에이전트 서비스의 경우 Kind는 CLIENT, 프로세스 내의 경우 INTERNAL입니다. 이름: `invoke_agent {gen_ai.agent.name}`.
2. LLM 호출마다 `chat` 스팬을 생성하며 `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`를 포함합니다.
3. 도구 호출마다 `tool_call` 스팬을 생성하며 `gen_ai.tool.name`를 포함하고, 해당되는 경우 `gen_ai.data_source.id` (RAG 코퍼스 / 메모리 저장소)를 포함합니다.
4. 옵트인 콘텐츠 캡처: 기본값은 OFF입니다. ON일 경우, 입력/출력을 외부에 저장하고 스팬에 `*.reference_id`를 기록합니다.
5. 컨텍스트 전파: W3C 트레이스 컨텍스트 헤더를 사용하여 다중 프로세스 실행(Claude Agent SDK CLI 하위 프로세스)이 하나의 트레이스로 연결되도록 합니다.

거부 조건:

- 기본적으로 전체 프롬프트/출력을 인라인으로 캡처하는 것. PII 및 비밀 유출 위험이 있으며, 또한 사양을 위반합니다.
- `gen_ai.provider.name`가 누락된 경우. 다중 제공자 대시보드가 작동하지 않습니다.
- 고아 도구 스팬. 항상 활성 컨텍스트를 통해 부모-자식 관계를 설정합니다.

거부 규칙:

- 런타임이 프로세스 경계를 넘어 컨텍스트를 전파할 수 없는 경우, 거부합니다. Claude Agent SDK + CLI 사용자에게는 다중 프로세스 트레이스 연결이 필수입니다.
- 제품에 규제 제약(HIPAA, GDPR)이 있는 경우, 인라인 콘텐츠 캡처를 거부합니다. 접근 제어만 있는 외부 저장소를 사용해야 합니다.
- 백엔드가 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`를 설정하지 않는 경우, 경고합니다: 수집기 업그레이드 시 속성 이름이 변경될 수 있습니다.

출력: `tracer.py`, `attributes.py`, `content_store.py`, `README.md`를 포함하여 스팬 구조, 안정성 옵트인 및 콘텐츠 캡처 정책을 설명합니다. "다음에 읽을 내용"으로 끝내며, 백엔드(Langfuse, Phoenix, Opik)에 대한 24강이나 Claude Agent SDK 트레이스 컨텍스트 전파에 대한 17강을 가리킵니다.
