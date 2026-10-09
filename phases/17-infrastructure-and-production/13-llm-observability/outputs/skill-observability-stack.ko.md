---
name: observability-stack
description: 스택, 규모, 예산, 라이선스 정책, 자체 호스트 요구 사항에 따라 LLM 관측성 스택(개발 플랫폼 + 게이트웨이 + 선택적 확장 레이어)을 선택하고 OpenTelemetry GenAI 속성 집합을 정의합니다.
version: 1.0.0
phase: 17단계
lesson: 13강
tags: [observability, langfuse, langsmith, phoenix, arize, helicone, opik, opentelemetry, genai-conventions]
---

스택(LangChain / DSPy / 원시 SDK), 규모(트레이스/일), 예산, 라이선스 정책(MIT 전용 vs 상용 허용), 자체 호스트 요구 사항을 고려하여 관측성 계획을 작성해 보세요.

다음 내용을 작성합니다:

1. 개발 플랫폼 선택. Langfuse(OSS), LangSmith(LangChain 우선 상용), Opik(Comet OSS) 또는 없음. 스택과 라이선스를 근거로 정당화합니다.
2. 게이트웨이/텔레메트리 선택. Helicone(프록시 + 게이트웨이), SigNoz(전체 APM), OpenLLMetry(순수 OTel). 이미 AI 게이트웨이(17단계 · 19강)를 사용 중이라면 통합을 명시합니다.
3. 확장/데이터 레이크 레이어. 선택 사항; 장기 분석을 위해 Arize AX 또는 원시 Iceberg, RAG 드리프트를 위해 Phoenix를 사용합니다.
4. OTel GenAI 규약. 최소 속성 집합을 지정합니다: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.request.temperature`, `gen_ai.response.finish_reasons`, 그리고 조직별 속성(tenant_id, user_id, task)을 포함합니다.
5. 샘플링 정책. 오류는 100%, 고비용 호출(>$0.10/호출)은 100%, 성공은 N% 샘플링 비율로 수집합니다. 원본 보존 기간은 14일 / 30일 / 90일입니다. 집계 데이터는 더 오래 보존합니다.
6. 알림. 알림이 필수인 5가지 지표: 오류율, P99 TTFT, 요청당 비용, 프롬프트 캐시 적중률, 거절율.

하드 리젝트:
- OTel 폴백 없이 프레임워크 전용 SDK 내에서 계측하는 경우. 거부합니다 — 프레임워크 종속(lock-in)이 발생합니다.
- 규제되지 않은 워크로드에 대해 Datadog급 가격(>$500/월)으로 100%의 트레이스를 유지하는 경우. 거부합니다 — 샘플링을 권장합니다.
- OpenTelemetry GenAI 규약을 무시하는 경우. 거부합니다 — 2026년 상호 운용성(interop)은 이를 요구합니다.

거부 규칙:
- 트레이스/일이 5M 이상이고 팀이 Datadog 전체 보존을 고집한다면, 비용 예측 없이 거부합니다.
- 팀이 MIT 전용 라이선스를 요구하며 LangSmith를 선택한다면 거부합니다 — Langfuse가 MIT 동등한 옵션입니다.
- 팀이 AI 게이트웨이가 없으며 Helicone을 게이트웨이 AND 관측성 도구로 선택한다면, 승인합니다 — 프록시가 게이트웨이 역할을 겸하며 최대 ~500 RPS까지 지원됩니다(17단계 · 19강에서 게이트웨이 규모를 다룹니다).

출력: 개발 플랫폼, 게이트웨이, 확장 계층(있는 경우), OTel 속성 집합, 샘플링 규칙, 5개 경보가 명시된 한 장의 계획입니다. 스택 드리프트를 나타내는 단일 지표로 마무리하세요: 지난 7일 동안 완전한 OTel GenAI 속성을 가진 LLM 호출의 비율입니다.
