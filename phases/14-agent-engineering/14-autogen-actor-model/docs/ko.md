# 에이전트를 위한 액터 모델 — 비동기 메시지와 타입 지정 런타임

> 에이전트를 액터로 취급: 비동기 메시지 교환, 이벤트 기반 핸들러, 장애 격리, 자연스러운 동시성. AutoGen v0.4 (Microsoft Research, 2025년 1월)는 이 모델을 중심으로 에이전트 오케스트레이션을 재설계했습니다. 현재 이 프레임워크는 유지보수 모드에 있으며, Microsoft Agent Framework (2025년 10월 공개 프리뷰)가 그 생산용 후속 제품입니다.

**유형:** Learn + Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 12강 (워크플로 패턴)
**시간:** 약 75분

## 학습 목표

- 액터 모델을 설명하세요: 에이전트를 액터로, 메시지를 유일한 IPC로, 액터별 장애 격리로.
- AutoGen v0.4의 세 가지 API 계층 — Core, AgentChat, Extensions —과 각각의 용도를 나열하세요.
- 메시지 전달을 처리와 분리하는 것이 왜 장애 격리와 자연스러운 동시성을 제공하는지 설명하세요.
- Python의 stdlib로 액터 런타임을 구현하고, 두 에이전트 코드 리뷰 흐름을 이 런타임으로 이식하세요.

## 문제점

대부분의 에이전트 프레임워크는 동기적입니다: 한 에이전트가 생성하고, 한 에이전트가 소비하며, 호출 스택 안에서 동작합니다. 장애가 발생하면 스택이 크래시됩니다. 동시성은 억지로 붙여져 있습니다. 분산은 재작성을 요구합니다.

AutoGen v0.4의 답: 액터 모델. 각 에이전트는 비공개 인박스를 가진 액터입니다. 메시지가 유일한 상호작용입니다. 런타임은 전달과 처리를 분리합니다. 장애는 하나의 액터로 격리됩니다. 동시성은 네이티브입니다. 분산은 단지 다른 전송 수단일 뿐입니다.

## 개념

### 액터

액터는 다음을 가집니다:

- 비공개 상태 (외부에서 직접 건드리지 않음).
- 인박스 (메시지 큐).
- 핸들러: `receive(message) -> effects` 여기서 효과는 "답변", "다른 액터로 전송", "새 액터 생성", "상태 업데이트", "자기 자신 중지"가 될 수 있습니다.

두 액터는 메모리를 공유할 수 없습니다. 메시지만 전송할 수 있습니다.

### 세 가지 API 계층

AutoGen v0.4는 그 표면을 세 부분으로 나눕니다:

1. **Core.** 저수준 액터 프레임워크. `AgentRuntime`, `Agent`, `Message`, `Topic`. 비동기 메시지 교환, 이벤트 기반.
2. **AgentChat.** 작업 중심의 고수준 API (v0.2의 ConversableAgent 대체). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** 통합 — OpenAI, Anthropic, Azure, 도구, 메모리.

### 분리가 중요한 이유

v0.2 모델에서는 `agent_a.chat(agent_b)`를 동기적으로 호출하면 agent_b가 반환할 때까지 agent_a가 차단됩니다. v0.4에서는 `send(agent_b, msg)`가 메시지를 agent_b의收件함(inbox)에 넣고 반환합니다. 런타임이 나중에 전달합니다. 세 가지 결과:

- **장애 격리.** Agent B가 충돌해도 Agent A는 충돌하지 않습니다 — 런타임이 B의 핸들러에서 실패를 포착하고 처리 방식(로그, 재시도, 데드레터)을 결정합니다.
- **자연스러운 동시성.** 많은 메시지가 동시에 처리 중이며, 액터는收件함(inbox)을 병렬로 처리합니다.
- **분산 준비 완료.**收件함(inbox) + 전송은 액터가 프로세스 내인지 다른 호스트에 있는지와 관계없이 동일한 추상화입니다.

### 토폴로지

- **RoundRobinGroupChat.** 에이전트가 고정된 순서로 번갈아 진행합니다.
- **SelectorGroupChat.** 선택자 에이전트가 대화 컨텍스트에 따라 다음 진행자를 선택합니다.
- **Magentic-One.** 웹 브라우징, 코드 실행, 파일 처리를 위한 참조 다중 에이전트 팀입니다. AgentChat 위에 구축되었습니다.

### 관측 가능성

OpenTelemetry 지원이 내장되어 있습니다. 모든 메시지가 스팬(span)을 방출하며, 도구 호출은 2026 OTel GenAI 시맨틱 컨벤션(23강)에 따라 `gen_ai.*` 속성을 포함합니다.

### 상태: 유지보수 모드

2026년 초: AutoGen v0.7.x는 연구 및 프로토타이핑용으로 안정적입니다. Microsoft는 활발한 개발을 생산용 후속 제품인 Microsoft Agent Framework로 전환했습니다 (2025년 10월 1일 공개 프리뷰; 1.0 GA는 2026년 1분기 말로 목표). AutoGen 패턴은 앞으로 잘 이식됩니다 — 액터 모델은 지속 가능한 아이디어입니다.

```figure
actor-mailbox
```

## 구현하기

`code/main.py`는 표준 라이브러리 액터 런타임을 구현합니다:

- `Message` — `sender`, `recipient`, `topic`, `body`를 포함한 타입 지정 페이로드.
- `Actor` — `receive(message, runtime)`를 포함한 추상 클래스.
- `Runtime` — 공유 큐, 전달, 장애 격리를 포함한 이벤트 루프.
- 두 액터 데모: `ReviewerAgent`는 코드를 검토하고, `ChecklistAgent`는 체크리스트를 실행합니다. 합의에 도달할 때까지 메시지를 교환합니다.

실행:

```
python3 code/main.py
```

추적은 메시지 전달, 한 액터의 시뮬레이션된 실패가 다른 액터를 크래시시키지 않는 상황, 그리고 공유된 판정(convergence)에 수렴하는 과정을 보여줍니다.

## 사용하기

- **AutoGen v0.4/v0.7** (유지보수) — 연구, 프로토타이핑, 다중 에이전트 패턴에 안정적입니다.
- **Microsoft Agent Framework** — 생산용 후속 제품 (2025년 10월 공개 프리뷰); 갱신된 API에서 동일한 액터 모델 개념을 사용합니다.
- **LangGraph swarm 토폴로지** (13강) — 공유 도구 핸드오프를 통한 유사한 패턴입니다.
- **커스텀 액터 런타임** — 특정 전송(NATS, RabbitMQ, gRPC)이 필요할 때 사용합니다.

## 출시하기

`outputs/skill-actor-runtime.md`는 주어진 다중 에이전트 작업에 대해 최소한의 액터 런타임과 팀 템플릿(RoundRobin 또는 Selector)을 생성합니다.

## 연습 문제

1. 데드 레터 큐를 추가하세요: 핸들러가 예외를 발생시키면, 실패한 메시지를 인간 검사를 위해 보관합니다. 장난감(toy)에서 DLQ가 얼마나 자주 히트합니까?
2. `SelectorGroupChat`를 구현하세요: 셀렉터 액터가 대화 상태에 따라 다음 메시지를 누가 처리할지 선택합니다.
3. 분산 전송을 추가하세요: 프로세스 내 큐를 JSON-over-HTTP 서버로 교체하여 액터가 별도 프로세스에서 실행되도록 하세요.
4. 메시지마다 OTel span을 연결하세요(또는 no-op 대체). 23강에 따라 `gen_ai.agent.name`, `gen_ai.operation.name`를 방출하세요.
5. AutoGen v0.4의 아키텍처 포스트를 읽어보세요. 장난감(toy)을 실제 `autogen_core` API로 이식하세요. 생산 환경에서 중요한 것을 무엇을 생략했습니까?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 액터 | "에이전트" | 비공개 상태 + 인박스 + 핸들러; 공유 메모리 없음 |
| 메시지 | "이벤트" | 타입이 지정된 페이로드; 액터가 상호작용하는 유일한 방법 |
| 인박스 | "메일박스" | 액터별 대기 메시지 큐 |
| 런타임 | "에이전트 호스트" | 메시지를 라우팅하고 실패를 격리하는 이벤트 루프 |
| 토픽 | "채널" | 액터 간에 명명된 발행-구독 경로 |
| 장애 격리 | "크래시 허용" | 한 액터의 실패가 다른 액터를 크래시시키지 않음 |
| RoundRobinGroupChat | "고정 순환 팀" | 에이전트가 순서대로 번갈아 진행 |
| SelectorGroupChat | "컨텍스트 라우팅 팀" | 셀렉터가 다음 진행자를 선택 |
| Magentic-One | "참고 팀" | 웹 + 코드 + 파일을 위한 다중 에이전트 스쿼드 |

## 추가 읽기

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 재설계 포스트
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 그래프형 대안
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen이 기본적으로 생성하는 스팬
