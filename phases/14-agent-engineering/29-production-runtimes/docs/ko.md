# 프로덕션 런타임: 큐, 이벤트, 크론

> 프로덕션 에이전트(Agent)는 6가지 런타임 형태에서 실행됩니다: 요청-응답, 스트리밍(Streaming), 내구성 있는 실행(Durable Execution), 큐 기반 백그라운드, 이벤트 주도형, 스케줄링된 실행입니다. 프레임워크를 선택하기 전에 런타임 형태를 먼저 선택해 보세요. 모든 형태에서 관측 가능성(Observability)은 핵심적인 역할을 합니다.

**유형:** Learn
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 13강 (LangGraph), 14단계 · 22강 (Voice)
**시간:** 약 60분

## 학습 목표

- 6가지 프로덕션 런타임 형태를 나열하고, 각 형태를 프레임워크/제품 패턴과 매칭해 보세요.
- 장기적인 작업(long-horizon tasks)에 내구성 있는 실행(Durable Execution)(LangGraph)이 중요한 이유를 설명해 보세요.
- 이벤트 주도형 런타임을 설명하고, Claude Managed Agents가 적합한 시점을 설명해 보세요.
- 다단계 에이전트(Agent)에서 관측 가능성(Observability)이 핵심적인 역할을 한다는 주장을 설명해 보세요.

## 문제점

프로덕션 에이전트(Agent)는 Jupyter notebook이 드러내지 못하는 방식으로 실패합니다: 37번째 단계에서의 네트워크 시간 초과, 음성 통화 중 사용자 연결 끊김, 기계 재부팅으로 인한 크론(cron) 작업 종료, 백그라운드 워커의 메모리 부족. 런타임 형태는 어떤 실패가 생존 가능한지 결정합니다.

## 개념

### 요청-응답

- 동기식 HTTP입니다. 사용자가 완성을 기다립니다.
- 짧은 작업(<30초)에만 적용 가능합니다.
- 스택: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- 관측 가능성(Observability): 표준 HTTP 접근 로그 + OTel span.

### 스트리밍(Streaming)

- 점진적 출력을 위해 SSE 또는 WebSocket을 사용합니다.
- LiveKit은 이를 음성/비디오를 위한 WebRTC로 확장합니다 (22강).
- 스택: 스트리밍을 지원하는 모든 프레임워크 + SSE/WS를 처리하는 프론트엔드.
- 관측 가능성(Observability): 청크(chunk)별 타이밍, 첫 토큰까지의 시간 (TTFT)(Time to First Token (TTFT)), 꼬리 지연(Tail Latency).

### 내구성 있는 실행(Durable Execution)

- 모든 단계 후 상태가 체크포인트(Checkpoint)로 저장되며, 실패 시 자동으로 재개됩니다.
- AutoGen v0.4 액터 모델은 실패를 단일 에이전트(Agent)로 격리합니다 (14강).
- LangGraph의 핵심 차별점 (13강).
- 단계 수가 불명확하고 복구 비용이 높을 때 필수적입니다.

### 큐 기반 / 백그라운드

- 작업이 큐에 들어가고, 워커가 이를 가져가며, 결과는 웹훅이나 퍼블릭/서브(pub/sub)를 통해 전달됩니다.
- 긴 시간 범위의 에이전트(Anthropic의 컴퓨터 사용 발표에 따르면 작업당 수십에서 수백 단계)에 필수적입니다.
- 스택: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), 커스텀.
- 관측 가능성: 큐 깊이, 작업별 지연 분포, DLQ 크기.

### 이벤트 주도(Event-driven)

- 에이전트가 트리거를 구독합니다: 새 이메일, PR 열림, 크론 실행.
- Claude Managed Agents는 이 기능을 기본으로 제공합니다 (17강).
- CrewAI Flows (15강)는 이벤트 주도 결정적 워크플로우를 구조화합니다.
- 관측 가능성: 트리거 소스, 이벤트부터 시작까지의 지연, 에이전트 지연.

### 스케줄링(Scheduled)

- 주기적으로 실행되는 크론 형태의 에이전트입니다.
- 내구성 있는 실행(Durable Execution)과 결합하여, 실패한 야간 실행이 다음 틱(tick)에 재개되도록 하세요.
- 스택: Kubernetes CronJob + 내구성 프레임워크; 호스팅된 서비스 (Render cron, Vercel cron).

### 2026년 배포 패턴

- **CrewAI Flows**는 이벤트 주도 프로덕션 환경에 적합합니다.
- **Agno**는 Python 마이크로서비스용 상태 비저장(Stateless) FastAPI입니다.
- **Mastra**는 임베딩을 위해 서버 어댑터 (Express, Hono, Fastify, Koa)를 제공합니다.
- **Pipecat Cloud / LiveKit Cloud**는 관리형 음성(22강)을 위해 사용됩니다.
- **Claude Managed Agents**는 호스팅된 장기 실행 비동기 처리를 위해 사용됩니다.

### 관측 가능성은 핵심적인 요소입니다

OpenTelemetry GenAI 스팬 (23강)과 Langfuse/Phoenix/Opik 백엔드 (24강)가 없으면, 40단계에서 실패한 다단계 에이전트를 디버깅할 수 없습니다. 이는 프로덕션 환경에서 선택 사항이 아닙니다. "빠르게 디버깅한다"와 "더 많은 로깅으로 처음부터 재생한다"의 차이입니다.

### 프로덕션 런타임이 실패하는 지점

- **잘못된 형태 선택.** 5분짜리 작업에 요청-응답(request-response) 방식을 선택하는 경우. 사용자가 연결을 끊고, 워커가 쌓이며, 재시도가 누적됩니다.
- **DLQ 없음.** 데드 레터(dead-letter)가 없는 큐 워커. 실패한 작업이 사라집니다.
- **투명하지 않은 백그라운드 작업.** 추적(trace) 내보내기 없이 실행되는 백그라운드 에이전트. 사용자가 보고할 때까지 실패가 보이지 않습니다.
- **내구성 있는 상태(Durable State) 건너뛰기.** 30초 이상 실행되며 재시작이 허용되지 않는 모든 실행에는 내구성 있는 실행(Durable Execution)이 필요합니다.

```figure
wb-runtime-shapes
```

## 구현하기

`code/main.py`은 표준 라이브러리 다중 형태 데모입니다:

- 요청-응답 엔드포인트(일반 함수).
- 스트리밍 핸들러(제너레이터).
- DLQ가 포함된 큐 기반 워커.
- 이벤트 트리거 레지스트리.
- Cron 형태의 스케줄러.

실행해 보세요:

```bash
python3 code/main.py
```

출력: 동일한 작업에 대해 각 형태의 동작을 보여주는 5개의 추적(Trace)이 표시됩니다. 동일한 에이전트 로직, 서로 다른 외부 셸. 내구성 있는 실행(Durable Execution)(6번째 형태)은 LangGraph 체크포인팅(Checkpointing)과 함께 13강에서 의도적으로 다루어집니다.

## 사용하기

- **요청-응답(Request-Response)**은 채팅 스타일 UX에 적합합니다.
- **스트리밍(Streaming)**은 점진적 응답에 적합합니다.
- **내구성(Durable)**은 장기적인 작업에 적합합니다.
- **큐(Queue)**는 배치/비동기/장기 실행에 적합합니다.
- **이벤트(Event)**는 에이전트 반응성에 적합합니다.
- **Cron**은 하우스키핑(메모리 통합, 평가(Evals), 비용 보고서)에 적합합니다.

## 출시하기

`outputs/skill-runtime-shape.md`은 작업에 대한 런타임 형태를 선택하고 관측 가능성(Observability) 요구 사항을 연결합니다.

## 연습 문제

1. 1강의 ReAct 루프를 스택의 모든 6가지 형태로 이식해 보세요. 어떤 형태가 어떤 제품 표면에 적합합니까?
2. 큐 기반 데모에 DLQ를 추가하세요. 10% 작업 실패를 시뮬레이션하고 DLQ 크기를 표시하세요.
3. 하루의 상위 20개 추적(Trace)에 대해 야간에 실행되는 Cron 트리거 평가(Eval) 에이전트를 작성하세요.
4. 백프레셔(Backpressure)가 있는 스트리밍을 구현하세요: 클라이언트가 느리면 에이전트를 일시 중지합니다. 이것이 턴 예산(Turn Budget)과 어떻게 상호작용합니까?
5. Claude Managed Agents 문서를 읽어 보세요. 자가 호스팅 장기 에이전트를 관리형으로 언제 이동해야 합니까?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 요청-응답(Request-Response) | "동기식" | 사용자가 대기; 짧은 작업만 |
| 스트리밍(Streaming) | "SSE / WS" | 점진적 출력; 더 나은 UX; 청크별 지연 관측 가능 |
| 내구성 있는 실행(Durable Execution) | "실패 후 재개" | 체크포인팅된 상태; 마지막 단계에서 재시작 |
| 큐 기반(Queue-based) | "백그라운드 작업" | 생산자 / 워커 풀 / DLQ |
| 이벤트 기반 | "트리거 기반" | 에이전트가 외부 이벤트에 반응 |
| DLQ | "죽은 편지 큐" | 실패한 작업의 주차장 |
| Claude Managed Agents | "호스팅된 하네스" | 캐싱 + 압축을 지원하는 Anthropic 호스팅 장기 비동기 |

## 추가 읽기

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 내구성 있는 실행 세부 사항
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 호스팅된 장기 비동기
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "작업당 수십에서 수백 단계"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 액터 모델 장애 격리
