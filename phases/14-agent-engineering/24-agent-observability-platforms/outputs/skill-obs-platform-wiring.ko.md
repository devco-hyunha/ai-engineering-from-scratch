---
name: obs-platform-wiring
description: 관측성 플랫폼(Langfuse, Phoenix, Opik, Datadog)을 선택하고 기존 에이전트에 추적(traces), 평가(evals), 프롬프트 버전을 연결합니다.
version: 1.0.0
phase: 14단계
lesson: 24강
tags: [observability, langfuse, phoenix, opik, datadog, tracing]
---

에이전트 런타임과 제품 요구사항이 주어지면, 관측성 플랫폼을 선택하고 연결 구조를 스캐폴딩합니다.

결정 사항:

1. 한 곳에서 프롬프트 관리와 세션 재생(session replay)이 필요하다면 **Langfuse**를 선택합니다.
2. 깊은 RAG 관련성(drift/anomaly detection)과 드리프트/이상치 탐지가 필요하다면 **Phoenix**를 선택합니다.
3. 자동화된 프롬프트 최적화와 PII 가드레일이 필요하다면 **Opik**를 선택합니다.
4. 이미 Datadog를 사용 중이라면 **Datadog LLM Observability**를 선택합니다(v1.37+부터 GenAI를 네이티브로 매핑합니다).
5. ELv2가 없는 라이선스가 필요하다면 **Langfuse**(MIT) 또는 **Opik**(Apache 2.0)를 선택합니다. 순수 OSS 배포를 위해 Phoenix는 피합니다.

산출물:

1. OTel GenAI 계측(23강) — 이는 공통 기반입니다.
2. 플랫폼별 SDK 또는 OTel exporter 구성.
3. 해당 도메인에 대한 LLM-judge 루브릭(사실적 정확성, 범위, 톤, 거절 품질).
4. 추적(traces)에 연결된 프롬프트 버전 관리(Langfuse) 또는 추적 클러스터링 구성(Phoenix) 또는 실험 정의(Opik).
5. 로그에 기록된 콘텐츠에 대한 가드레일: PII 마스킹, 비밀 정보 제거.
6. 대시보드: 세션 상태, 실패 분류 체계, 지연(latency) 분포, 세션당 비용.

허용되지 않는 사항:

- 평가(evals) 없이 출시하는 것. 추적(tracing)만으로는 비싼 로깅에 불과합니다.
- 외부 검증 없이 자체 작성한 LLM-judge를 사용하는 것. CRITIC 패턴(05강): judge는 사실적 근거(factual grounding)를 위해 외부 도구가 필요합니다.
- 스팬(span) 본문에 PII를 저장하는 것. 항상 외부 저장소 + 참조 ID를 사용해야 합니다.

거절 규칙:

- 사용자가 "모든 것을 위한 하나의 플랫폼"을 요청하면 거절하고 위의 결정 사항을 제시합니다. 단일 플랫폼이 세 가지 축을 모두 지배하지는 않습니다.
- 제품에 각 에이전트 작업에 대한 수용 기준(acceptance criteria)이 없다면 평가(evals) 출시를 거절합니다. LLM-judge는 루브릭이 필요하며, 루브릭은 제품 결정이 필요합니다.
- 사용자가 "샘플링 없이 모든 것을 캡처"하기를 원한다면 거절하세요. 추적(tracing) 볼륨은 트래픽에 선형적으로 비례하여 증가하므로, 대규모 환경에서는 샘플링(헤드 기반 또는 테일 기반)이 필수입니다.

출력: `instrumentation.py`, `judge.py`, `dashboards.md`, `README.md`를 통해 플랫폼 선택, 평가 기준(rubric), 샘플링 전략 및 인시던트 대응을 설명하세요. "다음에 읽을 내용"으로 30강(평가 주도 개발) 또는 26강(실패 모드 분류 체계)을 가리키며 마무리하세요.
