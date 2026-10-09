---
name: otel-genai-instrumentation
description: 에이전트 코드베이스에 대해 OTel GenAI 스팬을 엔드투엔드로 방출하기 위한 계측 계획을 생성합니다.
version: 1.0.0
phase: 13단계
lesson: 19강
tags: [otel, observability, gen-ai, tracing]
---

에이전트 코드베이스(LLM 호출, 도구 디스패치, MCP 클라이언트, 하위 에이전트)가 주어지면 OTel GenAI 계측 계획을 생성합니다.

생성할 항목:

1. 스팬 계층 구조. 루트 `agent.invoke_agent` (INTERNAL) 및 하위 스팬: `llm.chat` (CLIENT), `tool.execute` (INTERNAL), `mcp.call` (CLIENT), `subagent.invoke` (INTERNAL).
2. 스팬별 속성 체크리스트. `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`, `gen_ai.tool.name`, `gen_ai.agent.name`.
3. 전파 규칙. 모든 원격 호출에 W3C traceparent를 주입합니다. MCP stdio의 경우 `_meta.traceparent`를 중간 필드로 사용합니다.
4. 콘텐츠 캡처 정책. 기본적으로 비활성화합니다. 활성화하는 환경 변수를 문서화하고 PII 위험을 명시합니다.
5. 익스포터 선택. Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb; 전송 프로토콜로 OTLP를 사용합니다.

허용 불가 조건:
- MCP 또는 하위 에이전트 경계를 넘어 추적 전파가 누락된 모든 계획.
- 콘텐츠 캡처가 기본적으로 활성화된 모든 계획. 프롬프트 및 PII가 유출됩니다.
- `gen_ai.` 또는 명시적인 벤더 접두어 없이 임의의 사용자 정의 속성을 방출하는 모든 계획.

거절 규칙:
- 코드베이스가 내장 OTel 자동 계측을 제공하는 프레임워크(Pydantic AI, LangGraph, AgentOps)를 사용하는 경우, 프레임워크 훅을 우선적으로 권장합니다.
- 익스포터 백엔드가 온프레미스이고 팀에 SRE 지원이 없는 경우, 관리형 백엔드를 권장합니다.
- 사용자가 프로덕션 디버깅을 위해 콘텐츠 캡처를 요청하는 경우, 타입이 지정된 동의 정책과 PII 마스킹 파이프라인이 없으면 거절합니다.

출력: 스팬 계층 구조, 스팬별 속성 체크리스트, 전파 규칙, 콘텐츠 캡처 정책 및 익스포터 선택을 포함한 한 페이지 계획입니다. 마지막에 알림해야 할 최상위 지표(일반적으로 p95 `gen_ai.client.operation.duration`)를 포함하여 마무리합니다.
