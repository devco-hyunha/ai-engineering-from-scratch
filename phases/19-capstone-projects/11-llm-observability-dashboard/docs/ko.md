# 캡스톤 11 — LLM 관측 가능성 및 평가 대시보드

> Langfuse는 오픈 코어 모델로 전환했습니다. Arize Phoenix는 2026 GenAI semconv 매핑을 공개했습니다. Helicone과 Braintrust는 모두 사용자별 비용 귀속(per-user cost attribution)에 집중했습니다. Traceloop의 OpenLLMetry는 사실상 SDK 계측 표준이 되었습니다. 프로덕션 형태는 추적(traces)을 위한 ClickHouse, 메타데이터를 위한 Postgres, UI를 위한 Next.js, 그리고 샘플링된 추적(traces)에 대해 실행되는 소규모 평가 작업(evals) 군단(DeepEval, RAGAS, LLM-judge)으로 구성됩니다. 자가 호스팅 대시보드를 구축하고, 최소 네 가지 SDK 계열에서 데이터를 수집하며, 주입된 회귀(regression)를 5분 이내에 감지하는 것을 시연해 보세요.

**유형:** Capstone
**언어:** TypeScript (UI), Python / TypeScript (수집 + 평가), SQL (ClickHouse)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구), 17단계 (인프라), 18단계 (안전)

**활용 단계:** P11 · P13 · P17 · P18
**시간:** 25시간

## 문제점

2026년 프로덕션 트래픽을 실행하는 모든 AI 팀은 모델과 함께 관측성 평면(observability plane)을 유지합니다. 비용 귀속, 환각(Hallucination) 감지, 드리프트 모니터링, 제일브레이크(Jailbreak) 신호, SLO 대시보드, PII 유출 알림 등이 필요합니다. Langfuse, Phoenix, OpenLLMetry 등 오픈소스 참조 구현들은 OpenTelemetry GenAI 시맨틱 컨벤션을 수집 스키마로 수렴했습니다. 이제 하나의 SDK로 OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM을 계측하고 호환되는 스팬(spans)을 전송할 수 있습니다.

최소 네 가지 SDK 계열에서 데이터를 수집하고, 샘플링된 추적(traces)에 대해 소규모 평가 작업(evals)을 실행하며, 드리프트를 감지하고 알림을 보내는 자가 호스팅 대시보드를 구축할 것입니다. 측정 기준은 다음과 같습니다: 의도적으로 주입된 회귀(regression)(PII를 생성하기 시작하는 프롬프트)이 발생했을 때, 대시보드가 이를 감지하고 5분 이내에 알림을 발동해야 합니다.

## 개념

수집은 OTLP HTTP를 사용합니다. SDK는 GenAI-semconv 스팬(spans)을 생성합니다: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. 스팬(spans)은 컬럼 분석을 위해 ClickHouse에 저장되며, 메타데이터(사용자, 세션, 앱)는 Postgres에 저장됩니다.

평가는 샘플링된 추적을 배치 작업으로 실행합니다. DeepEval은 충실도, 독성 및 답변 관련성을 점수화합니다. RAGAS는 추적이 검색 컨텍스트를 포함할 경우 검색 메트릭을 점수화합니다. 맞춤형 LLM 평가자는 도메인 특화 검사(PII 유출, 정책 이탈 응답)를 수행합니다. 평가 실행은 부모 추적에 연결된 평가 스팬과 함께 동일한 ClickHouse에 기록합니다.

드리프트 감지는 시간 경과에 따른 임베딩 공간 분포(프롬프트 임베딩에 대한 PSI 또는 KL 발산)와 평가 점수 추세를 모니터링합니다. 알림은 Prometheus Alertmanager를 통해 Slack / PagerDuty로 전달됩니다. UI는 Next.js 15와 Recharts를 사용합니다.

## 아키텍처

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 스택

- 수집: OpenTelemetry SDK + GenAI 시맨틱 컨벤션; OTLP HTTP 전송
- 컬렉터: 테일 샘플링 프로세서(비용 제어용)가 포함된 OpenTelemetry Collector
- 저장소: 스팬용 ClickHouse, 메타데이터용 Postgres, 원본 이벤트 아카이브용 S3
- 평가: DeepEval, RAGAS 0.2, Arize Phoenix 평가자 팩, 맞춤형 LLM 평가자
- 드리프트: 풀링된 프롬프트 임베딩(sentence-transformers)에 대한 PSI / KL 주간 실행
- 알림: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + 서버 액션
- 기본 지원 SDK: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## 구현하기

1. **컬렉터 구성.** OTLP HTTP 수신기, 오류 난 추적을 100% 유지하고 성공한 추적을 10% 유지하는 테일 샘플러, ClickHouse 및 S3로 내보내는 익스포터가 포함된 OpenTelemetry Collector를 구성합니다.

2. **ClickHouse 스키마.** GenAI 시맨틱 컨벤션을 반영하는 `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id` 열과 긴 페이로드용 JSON 백을 포함하는 `spans` 테이블을 생성합니다. user_id와 app_id에 대한 보조 인덱스를 추가합니다.

3. **SDK 커버리지 테스트.** 각 SDK(OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM)를 사용하는 작은 클라이언트 앱을 작성하고 OpenLLMetry 자동 계측을 적용합니다. 각 SDK가 ClickHouse에 저장되는 표준 GenAI 스팬을 생성하는지 확인합니다.

4. **평가 작업.** 예약된 작업이 최근 15분간의 샘플링된 추적을 읽어 DeepEval 충실도, 독성 및 답변 관련성을 실행합니다. 출력은 부모 추적에 연결된 평가 스팬입니다.

5. **사용자 정의 LLM 판정.** PII 유출 판정기: 응답을 받으면 가드 LLM을 호출하여 PII 유출 가능성을 점수화합니다. 점수가 높은 응답은 트리아지 큐로 이동합니다.

6. **드리프트 감지.** 주간 작업이 이번 주의 풀링된 프롬프트 임베딩과 최근 4주간의 기준선 간 PSI를 계산합니다. PSI가 임계값을 초과하면 알림이 발생합니다.

7. **대시보드.** Next.js 15로 구축하며, 페이지는 다음과 같습니다: 개요 (초당 스팬 수, 사용자당 비용, p95 지연 시간), 추적 (검색 + 워터폴), 평가 (충실도 추이, 독성), 드리프트 (시간에 따른 PSI), 알림.

8. **알림 체인.** Prometheus 익스포터가 평가 점수 집계 및 지연 시간 백분위수를 읽습니다. Alertmanager는 경고는 Slack으로, 중대한 위반은 PagerDuty로 라우팅합니다.

9. **회귀 프로브.** 버그를 주입합니다: 평가 대상 챗봇이 1%의 확률로 가짜 SSN을 유출하기 시작합니다. MTTR을 측정합니다: 버그가 배포된 시점부터 Slack 알림이 발생하기까지의 시간.

## 사용하기

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 출시하기

`outputs/skill-llm-observability.md`는 산출물입니다. LLM 애플리케이션이 주어지면, 대시보드는 해당 트래이스를 수집하고, 평가를 실행하며, 드리프트를 감지해 알림을 발생시키고, Next.js에서 사용자당 비용 내역을 표시합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 트래이스 스키마 커버리지 | 표준화된 GenAI 스팬을 생성하는 SDK 계열의 수 (목표: 6개 이상) |
| 20 | 평가 정확도 | DeepEval / RAGAS 점수를 수동 라벨링된 세트와 비교 |
| 20 | 대시보드 UX | 주입된 회귀에 대한 MTTR (목표: 5분 미만) |
| 20 | 비용 / 확장성 | 백로그 없이 초당 1,000 스팬을 지속해서 수집 |
| 15 | 알림 + 드리프트 감지 | Prometheus/Alertmanager 체인을 끝에서 끝까지 테스트 |
| **100** | | |

## 연습 문제

1. Haystack 프레임워크에 대한 사용자 정의 계측을 추가합니다. 표준화된 스팬이 ClickHouse에 faithful `gen_ai.*` 속성을 포함하여 저장되는지 확인합니다.

2. 동일한 트래이스에서 DeepEval을 Phoenix 평가기로 교체합니다. 두 평가 엔진 간의 점수 드리프트를 측정합니다.

3. 드리프트 감지기를 정밀화합니다: 전역이 아닌 app-id별로 PSI를 계산합니다. 앱별 드리프트 추적을 표시합니다.

4. "사용자 영향" 페이지를 추가합니다: 사용자당 비용 및 사용자당 실패율을 스파크라인으로 표시합니다.

5. 독성 점수가 0.5 이상인 추적(traces)은 100% 유지하고, 나머지는 10% 층화 샘플링(tail-sampling)하는 정책을 구축해 보세요. 도입된 샘플링 편향을 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM 속성" | LLM span 속성(system, model, tokens)에 대한 2025 OpenTelemetry 사양 |
| Tail sampling | "추적 후 샘플링" | 수집기가 추적(traces) 완료 후 유지 또는 버림을 결정 (오류 확인 가능) |
| PSI | "인구 안정성 지수" | 두 분포를 비교하는 드리프트 지표; > 0.2는 일반적으로 의미 있는 드리프트를 나타냄 |
| LLM-judge | "모델 기반 평가" | LLM이 다른 LLM의 출력을 평가 기준(faithfulness, toxicity, PII)에 따라 점수 매기기 |
| Tail-sampling policy | "유지 규칙" | 추적(traces)을 저장할지 버릴지 결정하는 규칙; 오류 발생 + 샘플링 비율 |
| Eval span | "연결된 평가 추적" | 원본 LLM 호출 span과 연결된 평가 점수를 포함하는 자식 span |
| Cost per user | "단위 경제학" | 기간 내 user_id에 귀속된 비용; 핵심 제품 지표 |

## 추가 읽기

- [Langfuse](https://github.com/langfuse/langfuse) — 참조 오픈 코어 관측성 플랫폼
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 강력한 드리프트 지원이 있는 대안 참조
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 자동 계측 SDK 계열
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 인제스트 스키마
- [Helicone](https://www.helicone.ai) — 대안 호스팅 관측성
- [Braintrust](https://www.braintrust.dev) — 대안 평가 우선 플랫폼
- [ClickHouse documentation](https://clickhouse.com/docs) — 컬럼형 span 저장소
- [DeepEval](https://github.com/confident-ai/deepeval) — 평가자 라이브러리
