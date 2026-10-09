# LLM 관측성 스택 선택

> 2026년 관측성 시장은 두 가지 범주로 나뉩니다. 개발 플랫폼(LangSmith, Langfuse, Comet Opik)은 모니터링을 평가(evals), 프롬프트 관리, 세션 리플레이와 함께 번들링합니다. 게이트웨이/계측 도구(Helicone, SigNoz, OpenLLMetry, Phoenix)는 텔레메트리에 집중합니다. Langfuse는 MIT 라이선스 코어를 사용하며 강력한 오픈소스 균형(클라우드 무료 월 50K 이벤트)을 제공합니다. Phoenix는 Elastic License 2.0 하의 OpenTelemetry 네이티브 도구로, 드리프트/RAG 시각화에 탁월하지만 영구적인 프로덕션 백엔드는 아닙니다. Arize AX는 제로 카피 Iceberg/Parquet 통합을 사용하며, 모놀리식 관측성 대비 100배 저렴하다고 주장합니다. LangSmith는 LangChain/LangGraph에 대해 선두를 달리고 있으며, 사용자당 월 $39이고, 셀프 호스팅은 Enterprise 플랜에서만 가능합니다. Helicone은 프록시 기반이며 설정 시간이 15-30분, 월 100K 요청이 무료지만 에이전트 추적에 대한 깊이가 부족합니다. 일반적인 프로덕션 패턴은 게이트웨이(Helicone/Portkey)와 평가 플랫폼(Phoenix/TruLens)을 OpenTelemetry로 연결하는 것입니다.

**유형:** Learn
**언어:** Python (stdlib, 장난감 추적 샘플링 시뮬레이터)
**선수 요건:** 17단계 · 08강 (추론 지표), 14단계 (에이전트 엔지니어링)
**시간:** 약 60분

## 학습 목표

- 개발 플랫폼(번들형: 평가 + 프롬프트 + 세션)과 게이트웨이/텔레메트리 도구(추적 + 지표만)를 구분해 보세요.
- 여섯 가지 주요 도구(Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik)를 라이선스, 가격 및 최적 사용 사례에 매핑해 보세요.
- 게이트웨이 도구와 별도의 평가 플랫폼을 결합할 수 있게 해주는 OpenTelemetry 글루(glue) 패턴을 설명해 보세요.
- 2026년 비용 차별화 요소(Arize AX의 제로 카피 방식 vs 모놀리식 인제스트)를 명시하고 대략적인 100배 배수를 언급해 보세요.

## 문제점

LLM 기능을 출시했습니다. 잘 작동합니다. 프롬프트 실패, 도구 루프, 지연 시간 퇴화, 비용 급증, 프롬프트 캐시 적중률에 대한 가시성이 없습니다. "LLM 관측성"을 검색하면 세 가지 다른 가격대에서 동일한 문제를 해결한다고 주장하는 여덟 가지 도구가 나옵니다.

이 도구들은 동일한 문제를 해결하지 않습니다. LangSmith는 "이 LangGraph 실행이 왜 실패했나요?"에 답하고, Phoenix는 "내 RAG 파이프라인이 드리프트(drift)되고 있나요?"에 답하며, Helicone은 "어떤 앱이 토큰을 태우고 있나요?"에 답하고, Langfuse는 "전체 시스템을 자체 호스팅할 수 있나요?"에 답합니다. 서로 다른 도구, 서로 다른 대상입니다.

선택에는 네 가지 축이 있습니다: 스택 (LangChain? 순수 SDK? 멀티 벤더?), 라이선스 허용 범위 (MIT 전용? Elastic 허용? 상용 라이선스 허용?), 예산 (무료 티어? $100/mo? $1000/월?), 그리고 자체 호스팅 (필수? 있으면 좋은? 절대 안 함?).

## 개념

### 두 가지 범주

**개발 플랫폼**은 관측 가능성, 평가, 프롬프트 관리, 데이터셋 버전 관리, 세션 재생을 함께 제공합니다. 실험을 실행하고, 어떤 프롬프트가 효과가 있었는지 확인하며, 새로운 프롬프트를 이전의 성공한 프롬프트에 대해 데이터셋 회귀 테스트를 수행합니다. LangSmith, Langfuse, Comet Opik.

**게이트웨이/텔레메트리 도구**는 추론 호출을 계측합니다 — 프롬프트, 응답, 토큰, 지연 시간, 모델, 비용. Helicone, SigNoz, OpenLLMetry, Phoenix. 미니멀리스트. OpenTelemetry를 통해 별도의 평가 도구와 결합할 수 있습니다.

### Langfuse — OSS 균형

- 코어는 Apache / MIT 라이선스이며, Docker를 통해 자체 호스팅합니다.
- 클라우드 무료 티어: 월 50K 이벤트. 유료: 팀용 월 $29.
- 평가, 프롬프트 관리, 추적, 데이터셋. 네 가지 개발 플랫폼 기능 모두에 대해 합리적인 커버리지를 제공합니다.
- 적합한 지점: LangSmith급 기능이 필요하지만 자체 호스팅해야 하거나 OSS 라이선스를 유지해야 하는 경우.

### Phoenix (Arize) — 텔레메트리 우선, OpenTelemetry 네이티브

- Elastic License 2.0; 자체 호스팅이 간단합니다.
- RAG 및 드리프트 시각화에 탁월합니다. 임베딩 공간 산점도가 일급 기능으로 제공됩니다.
- 영구적인 프로덕션 백엔드로 설계되지 않았습니다 — 주로 개발 시점의 관측 가능성을 위한 것입니다.
- 적합한 지점: RAG 파이프라인 개발, 드리프트 디버깅, 프로덕션에서는 별도의 게이트웨이와 조합하여 사용.

### Arize AX — 대규모 확장 전략

- 상용 제품입니다. Iceberg/Parquet를 통한 제로 카피 데이터 레이크 통합을 지원합니다.
- 대규모 환경에서 모놀리식 관측 가능성 (Datadog급) 대비 약 100배 저렴하다고 주장합니다. 계산 방식: 추적(traces)을 S3의 자체 Parquet에 저장하고, Arize가 직접 읽습니다.
- 적합한 지점: 일일 10M 이상의 추적(traces), 기존 데이터 레이크, Datadog 가격 없이 LLM 전용 대시보드를 원하는 경우.

### LangSmith — LangChain/LangGraph 우선

- 상용 제품, 사용자당 월 $39. Enterprise 플랜에서만 셀프 호스팅 가능.
- LangChain 및 LangGraph 스택에 대해 최고 수준의 성능을 제공합니다. 두 스택 중 하나를 사용하지 않는다면 매력도가 떨어집니다.
- 적합한 상황: LangChain을 채택한 팀이 비용을 지불할 의향이 있는 경우.

### Helicone — 프록시 기반 최소 기능 구현

- `OPENAI_API_BASE`을 Helicone 프록시로 교체하여 15~30분 만에 설정할 수 있습니다.
- MIT 라이선스; 월 100K 요청 무료, 유료는 월 $20부터 시작.
- 페일오버, 캐싱, 속도 제한 포함 — 게이트웨이 역할도 수행합니다.
- 에이전트 / 다단계 추적에 대한 깊이가 부족합니다.
- 적합한 상황: 빠른 시작, 단일 스택 앱, 게이트웨이와 관측 가능성을 한 번에 필요로 하는 경우.

### Opik (Comet) — OSS 개발 플랫폼

- Apache 2.0, 완전한 오픈 소스.
- Comet의 유산을 바탕으로 Langfuse와 유사한 기능 세트.
- 적합한 상황: Comet를 이미 사용 중인 ML 팀이 동일한 패널에서 LLM 관측 가능성을 원하는 경우.

### SigNoz — OpenTelemetry 우선의 완전한 APM

- Apache 2.0. OpenTelemetry를 통해 일반적인 APM과 LLM을 모두 처리합니다.
- 적합한 상황: 서비스와 LLM 호출 전반에 걸친 통합 관측 가능성을 원하는 경우.

### 접착제: OpenTelemetry + GenAI 시맨틱 컨벤션

OpenTelemetry는 2025년 말에 GenAI 시맨틱 컨벤션을 공개했습니다 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). OTel을 소비하는 도구들은 상호 운용이 가능합니다. 부상하는 프로덕션 패턴은 다음과 같습니다:

1. 모든 LLM 호출에서 GenAI 컨벤션을 사용하여 OTel을 방출합니다.
2. 일상적인 사용을 위해 게이트웨이(Helicone / Portkey)로 라우팅합니다.
3. 회귀 테스트를 위해 평가 플랫폼(Phoenix / Langfuse)으로 이중 전송합니다.
4. Arize AX 또는 DuckDB를 통해 장기 분석을 위해 데이터 레이크(Iceberg)에 아카이브합니다.

### 함정: 잘못된 계층에서 계측하는 경우

에이전트 프레임워크 내부(예: LangSmith 추적 추가)에서 계측하면 해당 프레임워크에 결합됩니다. HTTP/OpenAI-SDK 계층(OpenLLMetry 또는 게이트웨이 사용)에서 계측하면 이식성이 있습니다.

### 샘플링 — 모든 것을 유지할 수 없습니다

하루 100만 건 이상의 요청이 발생하면, 전체 추적(tracing) 보존 비용이 LLM 호출 비용보다 더 많이 듭니다. 규칙에 따라 샘플링하세요: 오류는 100%, 고비용 요청은 100%, 성공 요청은 5%. 집계 데이터는 항상 보존하고, 롱테일(long tail)에 대해서는 원시(raw) 데이터를 유지하세요.

### 기억해야 할 수치

- Langfuse 무료 클라우드: 월 5만 이벤트.
- LangSmith: 사용자당 월 $39.
- Helicone 무료: 월 10만 요청.
- Arize AX 주장: 대규모 환경에서 모놀리식(monolithic) 대비 약 100배 저렴.
- OpenTelemetry GenAI 컨벤션: 2025년 출시, 2026년 광범위하게 채택.

```figure
i4-otel-glue
```

## 사용하기

`code/main.py`는 보존 전략(100% 수집, 샘플링, 샘플링 + 오류)에 걸쳐 하루 100만 건의 추적을 시뮬레이션합니다. 각 전략에서의 저장 비용과 손실되는 내용을 보고합니다.

## 출시하기

이 강의는 `outputs/skill-observability-stack.md`를 생성합니다. 스택, 규모, 예산, 라이선스 정책을 고려하여 도구를 선택합니다.

## 연습 문제

1. LangChain을 사용하는 팀이 OSS 셀프 호스팅(self-hosted) 관측성(observability)을 원합니다. Langfuse 또는 Opik을 선택하고 그 이유를 정당화해 보세요.
2. 하루 500만 건의 추적(traces)에서 Datadog 견적이 월 $150K일 때, Arize AX의 손익분기점을 계산해 보세요.
3. LLM 호출마다 조직 지침이 의무화해야 할 OpenTelemetry GenAI 속성(attribute) 집합을 설계해 보세요.
4. Phoenix만으로는 프로덕션 환경에 충분한지 논증해 보세요. 어떤 경우에 충분하지 않을까요?
5. Helicone은 20ms의 프록시 오버헤드가 있습니다. P99 TTFT가 300ms일 때, 이것이 허용 가능한가요? SLA가 100ms라면 어떻게 될까요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| OpenLLMetry | "LLM용 OTel" | LLM을 위한 오픈소스 OpenTelemetry 계측(instrumentation) |
| GenAI 컨벤션 | "OTel 속성" | LLM 호출을 위한 표준 OTel 속성 이름 |
| LangSmith | "LangChain 관측성" | LangChain 생태계와 번들된 상용 플랫폼 |
| Langfuse | "OSS LangSmith" | 유사한 기능 세트의 MIT OSS |
| Phoenix | "Arize 개발 도구" | OpenTelemetry 네이티브 개발/평가 플랫폼 |
| Arize AX | "대규모 관측성" | 상용 제로 카피(zero-copy) Iceberg/Parquet 관측성 |
| Helicone | "프록시 관측성" | LLM 텔레메트리(telemetry) 수집 및 게이트웨이 기능을 제공하는 HTTP 프록시 |
| Opik | "Comet LLM" | Comet의 Apache 2.0 OSS 개발 플랫폼 |
| 세션 재생 | "추적 재실행" | 도구 호출을 포함한 전체 에이전트 세션을 재생 |
| 평가 | "오프라인 테스트" | 레이블이 지정된 데이터셋에 후보 모델/프롬프트를 실행 |

## 추가 읽기

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
