---
name: llm-observability
description: OpenTelemetry GenAI 스팬을 수집하고, 평가를 실행하며, 주입된 회귀를 5분 이내에 감지하는 자가 호스팅 LLM 관측성 대시보드를 구축합니다.
version: 1.0.0
phase: 19단계
lesson: 11강
tags: [capstone, observability, otel, langfuse, phoenix, evals, drift, clickhouse]
---

최소 6개 SDK 계열(OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM)에 걸친 프로덕션 LLM 트래픽을 고려하여, OTLP GenAI-semconv 스팬을 수집하고, 평가를 실행하며, 드리프트를 감지하고, 알림을 보내는 자가 호스팅 관측성 플레인(observability plane)을 배포해 보세요.

구축 계획:

1. OTLP HTTP 수신기, 테일 샘플링 프로세서(오류는 100%, 성공은 10%, 높은 독성/PII는 100% 유지), ClickHouse 및 S3로 내보내는 OpenTelemetry Collector.
2. GenAI semconv를 반영하는 ClickHouse 스팬 스키마: gen_ai.system, gen_ai.request.model, usage.input/output_tokens, latency_ms, user_id, app_id, 그리고 프롬프트/완성(completions)을 위한 JSON 백(bag).
3. 앱, 사용자, 세션, 주석 큐(annotation queue)를 위한 Postgres 메타데이터 저장소.
4. SDK 계열별 클라이언트 앱에 OpenLLMetry 자동 계측(auto-instrumentation)을 적용하고, 표준(canonical) 스팬이 정상적으로 도착하는지 확인합니다.
5. 샘플링된 추적(trace)에 대해 DeepEval + RAGAS + Phoenix 평가자(evaluator) 팩을 예약 실행하며, PII 및 off-policy에 대한 커스텀 LLM 판정(judge)을 사용합니다.
6. 풀링된(pooling) 프롬프트 임베딩(embedding)에 대한 주간 PSI / KL 드리프트 감지기를 설정하고, 알림 임계값을 0.2로 지정합니다.
7. 평가 점수 집계 및 지연(latency) 백분위수를 위한 Prometheus 내보내기(exporter); Slack(경고) 및 PagerDuty(치명적)로 Alertmanager를 연결합니다.
8. Next.js 15 App Router 대시보드: 개요, 추적 검색 + 워터폴(waterfall), 평가 추이, 드리프트 차트, 알림.
9. 회귀(probe) 테스트: 가짜 SSN을 1%의 확률로 유출하는 응답 패턴을 주입하고, MTTR(알림 발화 시간)을 측정합니다.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 스팬 스키마 커버리지 | 표준 GenAI 스팬을 생성하는 SDK 계열의 수 (목표 6개 이상) |
| 20 | 평가 정확도 | DeepEval / RAGAS 점수 vs 수동 라벨링된 세트 |
| 20 | 대시보드 UX | 주입된 회귀에 대한 MTTR (목표 5분 이내) |
| 20 | 비용 / 확장성 | 백로그 없이 지속 가능한 1k spans/sec 수집 |
| 15 | 알림 + 드리프트 감지 | Prometheus/Alertmanager 체인이 엔드 투 엔드로 실행됨 |

하드 리젝트:

- OpenTelemetry GenAI semconv에 없는 속성 이름을 임의로 만든 스팬 스키마.
- 오류를 드롭하는 테일 샘플링 정책 (잘 알려진 안티 패턴).
- 샘플링 없이 인제스트 속도로 실행되는 평가 (수락할 수 없는 비용).
- p50/p95/p99 분리 없이 "레이턴시"만 표시하는 대시보드.

거부 규칙:

- PII 마스킹 정책 없이 프롬프트나 완성을 저장하는 것을 거부합니다.
- SDK별 표준 스팬 회귀 테스트 없이 "멀티 SDK 지원"을 주장하는 것을 거부합니다.
- 기준 윈도우 없이 드리프트 감지를 출시하는 것을 거부합니다. 제로샷 드리프트는 쓸모가 없습니다.

출력: 수집기 구성, ClickHouse 스키마, Next.js 15 대시보드, 평가 작업, 드리프트 감지 도구, 알림 체인, 주석 처리된 회귀가 포함된 10k-추적 데모 데이터셋, 그리고 주입된 PII 회귀에 대한 MTTR과 반복 과정에서 MTTR을 낮춘 상위 세 가지 대시보드 UX 개선 사항을 문서화한 작성물을 포함하는 저장소입니다.
