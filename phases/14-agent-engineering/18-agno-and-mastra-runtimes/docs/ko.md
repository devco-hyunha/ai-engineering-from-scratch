# 프로덕션 에이전트 런타임 — 빠른 인스턴스 생성 및 타입 지정 워크플로

> 프로덕션 에이전트 런타임은 프로토타이핑 프레임워크가 무시하는 요소, 즉 인스턴스 생성 비용, 타입 지정 워크플로 표면, 서빙 준비가 된 백엔드를 최적화합니다. 2026년 조합: Agno (Python)는 마이크로초 단위 에이전트 인스턴스 생성과 상태 비저장 FastAPI 백엔드를 목표로 합니다. Mastra는 Vercel AI SDK 기반 위에 에이전트, 도구, 워크플로, 통합 모델 라우팅, 복합 스토리지를 제공합니다.

**유형:** Learn
**언어:** Python, TypeScript
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 13강 (LangGraph)
**시간:** 약 45분

## 학습 목표

- Agno의 성능 목표와 이것이 중요한 시점을 식별해 보세요.
- Mastra의 세 가지 프리미티브 — 에이전트, 도구, 워크플로 — 와 지원되는 서버 어댑터를 나열해 보세요.
- 왜 상태 비저장 세션 범위 FastAPI 백엔드가 Agno의 권장 프로덕션 경로인지 설명해 보세요.
- 주어진 스택(Python 우선 vs TypeScript 우선)에 따라 Agno와 Mastra 중 하나를 선택해 보세요.

## 문제점

LangGraph, AutoGen, CrewAI는 프레임워크 중심적입니다. "에이전트 루프만, 빠르게, 내 런타임에서"를 원하는 팀은 Agno (Python) 또는 Mastra (TypeScript)를 선택합니다. 둘 다 프레임워크가 소유한 프리미티브의 일부와 교환하여 원시적인 속도와 주변 스택에 더 밀접한 적합성을 얻습니다.

## 개념

### Agno

- Python 런타임, 이전에는 Phi-data였습니다.
- "그래프, 체인, 복잡한 패턴은 없습니다 — 순수 Python만 있습니다."
- 문서에서의 성능 목표: 에이전트 인스턴스 생성 약 2μs, 에이전트당 메모리 약 3.75 KiB, 모델 제공자 약 23개.
- 프로덕션 경로: 상태 비저장 세션 범위 FastAPI 백엔드. 각 요청은 새로운 에이전트를 시작하며, 세션 상태는 DB에 저장됩니다.
- 네이티브 멀티모달(텍스트, 이미지, 오디오, 비디오, 파일) 및 에이전트형 RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)).

속도 목표는 초당 수천 개의 짧은 수명 에이전트(채팅 팬인, 평가 파이프라인)를 처리할 때 중요합니다. 하나의 에이전트가 10분 동안 실행될 때는 중요도가 낮습니다.

### Mastra

- TypeScript, Vercel AI SDK 위에 구축되었습니다.
- 세 가지 프리미티브: **에이전트**, **도구** (Zod 타입 지정), **워크플로**.
- 통합 모델 라우터 — 94개 제공업체의 3,300개 이상 모델 (2026년 3월).
- 복합 스토리지: 메모리, 워크플로우, 관측 가능성을 서로 다른 백엔드에 저장; 대규모 관측 가능성에는 ClickHouse를 권장합니다.
- Apache 2.0이며 `ee/` 디렉터리는 소스 공개(source-available) 엔터프라이즈 라이선스입니다.
- Express, Hono, Fastify, Koa용 서버 어댑터; Next.js 및 Astro와 일급(first-class) 통합을 지원합니다.
- 디버깅을 위해 Mastra Studio (localhost:4111)가 함께 제공됩니다.
- 1.0 버전 기준 (2026년 1월), GitHub 스타 22k+, 주간 npm 다운로드 300k+.

### 포지셔닝

둘 다 LangGraph가 되려 하지 않습니다. 다음 항목에서 경쟁합니다:

- **언어 적합성.** Python 우선 팀은 Agno; TypeScript 우선 팀은 Mastra.
- **런타임 인체공학(Ergonomics).** Agno는 거의 제로 오버헤드; Mastra는 Vercel 생태계와 통합됩니다.
- **관측 가능성.** 둘 다 Langfuse/Phoenix/Opik (24강)과 통합되지만, Mastra Studio는 자체 제품(first-party)입니다.

### 각각을 선택하는 시점

- **Agno** — Python 백엔드, 많은 단기 에이전트, 높은 성능 요구, FastAPI 환경.
- **Mastra** — TypeScript 백엔드, Next.js / Vercel 배포, 통합된 다중 제공업체 모델 라우팅, Zod 타입 지정 도구.
- **LangGraph** (13강) — 내구성 있는 상태와 명시적인 그래프 추론이 순수 속도보다 중요할 때.
- **OpenAI / Claude Agent SDK** — 제공업체의 제품화된 형태를 원할 때 (16–17강).

### 이 패턴이 잘못된 경우

- **성능을 위한 성능.** 요청당 느린 에이전트 호출 한 번뿐인 워크로드에서 "2μs"가 좋아 보인다고 Agno를 선택하는 것. 오버헤드가 병목이 아닙니다.
- **생태계 잠금(Lock-in).** Mastra의 Vercel 스타일 통합은 Vercel에서는 장점이지만, 그 외에서는 단점입니다.
- **엔터프라이즈 라이선스 혼동.** Mastra의 `ee/` 디렉터리는 소스 공개(source-available)이며 Apache 2.0이 아닙니다. 포크(fork)를 계획한다면 라이선스를 읽어 보세요.

```figure
wb-runtime-spawn
```

## 구현하기

이 강의는 주로 비교 중심입니다 — 단일 코드 산출물로는 두 프레임워크를 모두 공정하게 다루기 어렵습니다. `code/main.py`에서 나란히 놓인 장난감 예제를 보세요: "에이전트 실행, 출력 스트리밍, 세션 지속"이라는 최소 흐름을 두 번 구현했습니다 (한 번은 Agno 스타일, 한 번은 Mastra 스타일).

실행해 보세요:

```
python3 code/main.py
```

구조적으로 다르지만 기능적으로 동일한 두 개의 추적입니다.

## 사용하기

- **Agno** — 속도와 FastAPI 형태가 필요한 Python 백엔드입니다.
- **Mastra** — 다양한 제공자와 워크플로우 프리미티브를 갖춘 TypeScript 백엔드입니다.
- 두 도구 모두 자체 관측성 훅을 제공합니다. 두 도구 모두 Langfuse와 연동됩니다.

## 출시하기

`outputs/skill-runtime-picker.md`은 스택, 지연 시간 예산, 운영 형태에 따라 Agno, Mastra, LangGraph 또는 제공자 SDK를 선택합니다.

## 연습 문제

1. Agno의 문서를 읽어 보세요. 표준 라이브러리 ReAct 루프(01강)를 Agno로 이식해 보세요. 무엇이 사라졌고, 무엇이 남았나요?
2. Mastra의 문서를 읽어 보세요. 동일한 루프를 Mastra로 이식해 보세요. 도구 타입 지정(Zod 대 없음)에서 무엇이 변했나요?
3. 벤치마크: 사용 중인 스택에서 에이전트 인스턴스화 지연 시간을 측정해 보세요. Agno의 2μs가 작업 부하에 영향을 미치나요?
4. 마이그레이션을 설계해 보세요. Python에서 CrewAI를 실행해 왔다면, Agno로 이동할 때 무엇이 깨지나요?
5. Mastra의 `ee/` 라이선스 조건을 읽어 보세요. 오픈소스 포크에 영향을 미치는 제한 사항은 무엇인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agno | "빠른 Python 에이전트" | 상태 비저장 세션 범위 에이전트 런타임 |
| Mastra | "Vercel AI SDK 기반 TypeScript 에이전트" | 에이전트 + 도구 + 워크플로우 + 모델 라우터 |
| 통합 모델 라우터 | "다중 제공자 접근" | 94개 제공자의 3,300개 이상 모델을 위한 단일 클라이언트 |
| 복합 저장소 | "다중 백엔드" | 메모리/워크플로우/관측성을 각각 다른 저장소에 연결 |
| Mastra Studio | "로컬 디버거" | 에이전트를 내시하기 위한 localhost:4111 UI |
| 소스 공개 | "오픈소스가 아님" | 라이선스가 소스 읽기를 허용하지만 상업적 사용을 제한 |

## 추가 읽기

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 성능 목표, FastAPI 연동
- [Mastra docs](https://mastra.ai/docs) — 프리미티브, 서버 어댑터, 모델 라우터
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 상태 유지 그래프 대안
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 연동에서 인용된 관측성 비교
