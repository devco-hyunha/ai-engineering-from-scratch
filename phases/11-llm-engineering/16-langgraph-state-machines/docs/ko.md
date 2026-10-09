# 에이전트 상태 머신 — 그래프, 노드, 체크포인트

> 직접 작성한 ReAct 루프는 `while True`입니다. 동일한 루프를 명시적인 그래프로 작성하면 체크포인트 저장, 중단, 분기, 시간 여행이 가능합니다. 에이전트 자체는 변하지 않았습니다. 에이전트를 둘러싼 하네스가 변한 것입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 11단계 · 09강 (함수 호출), 11단계 · 14강 (모델 컨텍스트 프로토콜)
**시간:** 약 75분

## 문제점

함수 호출 에이전트를 출시했습니다. 세 번의 턴 동안은 잘 작동하지만, 그 후 문제가 발생합니다: 모델이 500을 반환하는 도구를 시도하거나, 사용자가 작업 중 마음을 바꾸거나, 에이전트가 인간의 승인 없이 주문을 환불하기로 결정합니다. `while True:` 루프에는 훅이 없습니다. 중단할 수 없고, 되돌릴 수 없으며, "모델이 다른 도구를 선택했다면 어떻게 되었을까?"와 같은 분기를 만들 수도 없습니다. 데모 단계를 넘겨 출시하는 순간, 에이전트는 작동했거나 작동하지 않은 블랙 박스가 됩니다.

이 문제를 인식하면 다음 단계는 명확합니다. 에이전트는 이미 상태 머신입니다 — 시스템 프롬프트(System Prompt) + 메시지 히스토리 + 대기 중인 도구 호출 + 다음 액션. 상태 머신을 명시적으로 만드세요: "모델이 생각하는", "도구가 실행되는", "인간이 승인하는" 노드와 그들 사이의 조건부 전환을 나타내는 엣지를 만드세요. 그래프가 명시적이 되면 하네스는 네 가지를 무료로 얻습니다: 체크포인팅(단계 간 상태 저장), 인터럽트(인간을 위해 중단), 스트리밍(토큰 및 중간 이벤트 스트리밍), 시간 여행(이전 상태로 되돌려 다른 분기를 시도).

이 추상화의 참조 구현은 LangGraph입니다. LangChain의 의미("여기 AgentExecutor가 있습니다, 행운을 빕니다")에서의 에이전트 프레임워크가 아닙니다. 일급 상태, 일급 지속성, 일급 인터럽트를 갖춘 그래프 런타임입니다. 에이전트 루프는 직접 작성하는 것이 아니라 그리는 것입니다.

## 개념

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph`는 세 가지를 가집니다.

1. **상태.** 그래프를 통해 흐르는 타입이 지정된 dict (TypedDict 또는 Pydantic 모델)입니다. 모든 노드는 전체 상태를 받고 부분 업데이트를 반환하며, LangGraph는 필드별 *리듀서*를 사용하여 이를 병합합니다 — 누적되어야 하는 리스트의 경우 `operator.add`, 기본값은 덮어쓰기입니다.
2. **노드(Node).** Python 함수 `state -> partial_state`. 각각은 "모델 호출", "도구 실행", "요약"과 같은 개별 단계입니다.
3. **엣지(Edge).** 노드 간의 전환입니다. 정적 엣지는 한 곳으로만 이동합니다. 조건부 엣지는 라우터 함수 `state -> next_node_name`를 사용하므로 그래프가 모델 출력에 따라 분기할 수 있습니다.

그래프를 컴파일합니다. 컴파일은 토폴로지를 바인딩하고, 체크포인터(checkpointer)를 연결하며(선택 사항이지만 프로덕션 환경에서는 필수), 실행 가능한(runnable) 객체를 반환합니다. 초기 상태와 `thread_id`를 사용하여 이를 호출합니다. 실행의 모든 단계는 `(thread_id, checkpoint_id)`를 키로 하는 체크포인트를 저장합니다.

### 네 가지 초능력

**체크포인팅(Checkpointing).** 모든 노드 전환은 새 상태를 저장소에 기록합니다(테스트에서는 메모리, 프로덕션에서는 Postgres/Redis/SQLite 사용). 동일한 `thread_id`로 그래프를 다시 호출하여 재개합니다. 그래프는 중단된 지점부터 이어집니다.

**인터럽트(Interrupts).** `interrupt_before=["human_review"]`로 노드를 표시하면 해당 노드가 실행되기 전에 실행이 중단됩니다. 상태는 유지됩니다. API는 사용자에게 "승인 대기 중"이라고 응답합니다. `Command(resume=...)`를 사용하여 동일한 `thread_id`에 대한 후속 요청이 실행을 재개합니다.

**스트리밍(Streaming).** `graph.stream(state, mode="updates")`는 상태 델타(state deltas)가 발생할 때 이를 생성합니다. `mode="messages"`는 모델 노드 내부의 LLM 토큰을 스트리밍합니다. `mode="values"`는 전체 스냅샷을 생성합니다. UI에 무엇을 표시할지 선택할 수 있습니다.

**타임 트래블(Time-travel).** `graph.get_state_history(thread_id)`는 전체 체크포인트 로그를 반환합니다. 이전 `checkpoint_id`를 `graph.invoke`에 전달하면 그 지점에서 포크(fork)할 수 있습니다. 디버깅("모델이 도구 B를 선택했다면 어떻게 되었을까?") 및 프로덕션 트레이스를 재생하는 회귀 테스트에 유용합니다.

### 리듀서(Reducers)가 핵심입니다

모든 상태 필드에는 리듀서가 있습니다. 대부분의 기본값은 새 값이 이전 값을 덮어쓰는 방식이며, 이는 문제가 없습니다. 하지만 메시지 목록은 `operator.add`가 필요하므로 새 메시지가 기존 것을 대체하는 대신 추가됩니다. 병렬 엣지는 리듀서를 통해 업데이트를 병합합니다. 두 노드가 모두 `messages`를 업데이트하는데 `Annotated[list, add_messages]`를 잊어버리면, 두 번째 업데이트가 조용히 승리하여 턴의 절반을 잃게 됩니다. 리듀서는 라이브러리에서 유일한 미묘한 부분입니다. 이를 올바르게 처리하면 나머지는 자연스럽게 조합됩니다.

### 네 노드로 구성된 ReAct 그래프

프로덕션 ReAct 에이전트는 네 개의 노드와 두 개의 엣지로 구성됩니다:

1. `agent` — 현재 메시지 히스토리로 LLM을 호출합니다. 어시스턴트 메시지(도구 호출(tool_calls)가 포함될 수 있음)를 반환합니다.
2. `tools` — 마지막 어시스턴트 메시지의 tool_calls를 실행하고, 도구 결과를 도구 메시지로 추가합니다.
3. `agent`에서 시작하는 조건부 엣지로, 마지막 메시지에 tool_calls가 있으면 `tools`로 라우팅하고, 그렇지 않으면 `END`로 라우팅합니다.
4. `tools`에서 `agent`로 돌아가는 정적 엣지입니다.

이것이 전부입니다. 체크포인팅, 인터럽트, 스트리밍을 포함한 완전한 ReAct 루프(Thought → Action → Observation → Thought → …)를 약 40줄의 코드로 구현할 수 있습니다.

### StateGraph vs Send (fanout)

`Send(node_name, state)`는 노드가 병렬 서브그래프를 디스패치할 수 있게 합니다. 예: 에이전트가 세 개의 리트리버를 동시에 쿼리하기로 결정합니다. 각 `Send`는 대상 노드의 병렬 실행을 생성하며, 그 출력은 상태 리듀서를 통해 병합됩니다. 이는 LangGraph가 스레딩 프리미티브 없이 오케스트레이터-워커 패턴을 표현하는 방법입니다.

### 서브그래프

컴파일된 그래프는 다른 그래프의 노드가 될 수 있습니다. 외부 그래프는 단일 노드로 인식하며, 내부 그래프는 자체 상태와 자체 체크포인팅을 가집니다. 이는 팀이 슈퍼바이저-워커 에이전트를 구축하는 방법입니다: 슈퍼바이저 그래프는 사용자 의도를 도메인별 워커 서브그래프로 라우팅합니다.

```figure
l5-state-graph-ledger
```

## 구현하기

### 1단계: 상태와 노드

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`는 메시지 목록이 덮어쓰기되지 않고 누적되게 하는 리듀서입니다. 이를 잊는 것이 가장 흔한 LangGraph 버그입니다.

### 2단계: 스레드로 실행하기

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

모든 업데이트는 dict `{node_name: state_delta}`입니다. 프론트엔드는 이를 UI로 스트리밍하여 사용자가 "에이전트가 생각 중… search_web 호출… 결과 획득… 답변 중."을 볼 수 있게 합니다.

### 3단계: 인간 개입 루프(HITL) 인터럽트 추가하기

실행이 노드 실행 전에 일시 중지되도록 노드를 표시합니다.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 모든 도구 호출 전에 일시 중지
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"]가 설정됩니다. 제안된 도구 호출을 검사합니다.
# 승인된 경우:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 거부된 경우: 거부 메시지를 작성하고 재개합니다
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

상태, 체크포인팅, 스레드 모두 인터럽트 동안 지속됩니다. 실행 중이 아닌 동안에는 메모리에 아무것도 저장되지 않습니다.

### 4단계: 디버깅을 위한 시간 여행

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 이전 체크포인팅에서 포크하기
target = history[3].config  # 세 단계 뒤로
for event in app.stream(None, target, stream_mode="values"):
    pass  # 그 시점부터 앞으로 재생하기
```

`None`을 입력으로 전달하면 주어진 체크포인트에서 재생됩니다. 값을 전달하면 해당 체크포인트의 상태에 업데이트로 추가한 후 재개합니다. 전체 대화를 다시 실행하지 않고도 불량한 에이전트 실행을 재현하는 방법입니다.

### 5단계: 프로덕션용 체크포인터로 교체하기

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, Postgres가 제공됩니다. `MemorySaver`는 테스트용입니다. 재시작 후에도 지속되는 모든 것은 실제 저장소가 필요합니다.

## 스킬

> 에이전트는 `while True` 루프가 아닌 그래프로 구축합니다.

LangGraph를 사용하기 전에 60초 설계 해 보세요:

1. **노드를 이름 짓습니다.** 모든 개별적인 결정이나 사이드 이펙트를 유발하는 행동은 노드입니다. "에이전트가 생각한다", "도구가 실행된다", "리뷰어가 승인한다", "응답이 스트리밍된다." 이들을 나열할 수 없다면, 그 작업은 아직 에이전트 형태가 아닙니다.
2. **상태를 선언합니다.** 모든 리스트 필드에 리듀서를 가진 최소한의 TypedDict를 사용하세요. 모든 것을 `messages`에 넣지 마세요. 작업 특화 필드(작업 중인 `plan`, `budget` 카운터, `retrieved_docs` 리스트)는 최상위 레벨로 끌어올리세요.
3. **엣지를 그립니다.** 다음 단계가 모델 출력에 의존하지 않는 한 정적으로 만드세요. 모든 조건부 엣지는 이름이 지정된 분기를 가진 라우터 함수가 필요합니다.
4. **체크포인터를 미리 선택합니다.** 테스트에는 `MemorySaver`, 그 외에는 Postgres/Redis/SQLite를 사용하세요. 체크포인터 없이 출시하지 마세요. 체크포인터가 없으면 재개, 인터럽트, 시간 여행이 불가능합니다.
5. **도구가 실행되기 전에 인터럽트를 결정하세요.** 승인(s)는 사이드 이펙트 노드로 들어가는 엣지에 배치하여 해를 끼치기 전에 취소할 수 있도록 하세요. 검증은 모델에서 나가는 엣지에 배치하여 불량한 호출을 저렴하게 거부할 수 있도록 하세요.
6. **기본적으로 스트리밍하세요.** UI에는 `mode="updates"`, 모델 노드 내부의 토큰 단위 스트리밍에는 `mode="messages"`, 평가 중 전체 스냅샷에는 `mode="values"`를 사용하세요.

체크포인터가 없는 LangGraph 에이전트는 출시하지 마세요. 사이드 이펙트 *후에* 인터럽트하는 에이전트도 출시하지 마세요. `add_messages`을 리듀서로 사용하지 않는 `messages` 필드도 출시하지 마세요.

## 연습 문제

1. **쉬움.** 위 네 노드 ReAct 그래프를 계산기 도구와 웹 검색 도구로 구현하세요. 두 턴 대화에 대해 `list(app.get_state_history(config))`가 최소 네 개의 체크포인트를 반환하는지 확인하세요.
2. **중간 난이도.** `agent`이 실행되기 전에 `planner` 노드를 추가하여 상태에 구조화된 `plan: list[str]`을 기록하세요. `agent`이 계획 단계를 완료로 표시하도록 하세요. 체크포인트 재개 시 `plan`이 손실되면 테스트가 실패하도록 하세요 (잘못된 리듀서).
3. **어려운 난이도.** `Send`을 사용하여 세 개의 서브그래프(`researcher`, `writer`, `reviewer`) 사이를 라우팅하는 감독자 그래프를 구축하세요. 각 서브그래프는 자체 상태와 체크포인터를 가집니다. 외부 그래프에 `interrupt_before=["writer"]`을 추가하여 인간이 연구 브리프를 승인할 수 있도록 하세요. 이전 체크포인트에서의 시간 여행이 분기된 브랜치만 다시 실행하는지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 그래프" | 컴파일하기 전에 노드와 엣지를 추가하는 빌더 객체입니다. |
| Reducer | "필드 병합 방식" | 노드가 해당 필드의 업데이트를 반환할 때 적용되는 함수 `(old, new) -> merged`입니다. 기본값은 덮어쓰기이며, `add_messages`은 추가합니다. |
| Thread | "대화 ID" | 한 세션의 모든 체크포인트를 범위로 지정하는 `thread_id` 문자열입니다. |
| Checkpoint | "일시 정지된 상태" | 노드 전환 후 전체 그래프 상태의 영속화된 스냅샷으로, `(thread_id, checkpoint_id)`을 키로 사용합니다. |
| Interrupt | "인간을 위한 일시 정지" | `interrupt_before` / `interrupt_after`이 노드 경계에서 실행을 중단합니다. `Command(resume=...)`로 재개하세요. |
| Time-travel | "이전 단계에서 분기" | `graph.invoke(None, config_with_old_checkpoint_id)`이 해당 체크포인트부터 앞으로 재생합니다. |
| Send | "병렬 서브그래프 디스패치" | 노드가 반환하여 대상 노드의 N개 병렬 실행을 생성하는 생성자입니다. |
| Subgraph | "노드로서의 컴파일된 그래프" | 다른 그래프의 노드로 사용되는 컴파일된 StateGraph입니다. 자체 상태 범위를 보존합니다. |

## 추가 읽기

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, 리듀서, 체크포인터 및 인터럽트에 대한 표준 참조입니다.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 원본에서 직접 가져온 이 강의가 사용하는 멘탈 모델입니다.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 저장소, 체크포인트 네임스페이스 및 스레드 ID에 대한 상세 정보입니다.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)` 및 상태 편집 패턴입니다.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 모든 LangGraph 에이전트가 구현하는 패턴입니다. 추론적 디코딩(Speculative Decoding)의 근거를 위해 읽어 보세요.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 어떤 그래프 형태(체인, 라우터, 오케스트레이터-워커, 평가자-최적화자)를 선호해야 하는지, 그리고 언제 사용해야 하는지입니다.
- 11단계 · 09강 (함수 호출) — 모든 LangGraph 에이전트 노드가 재사용하는 도구 호출 프리미티브입니다.
- 11단계 · 14강 (모델 컨텍스트 프로토콜) — MCP 어댑터를 통해 LangGraph `ToolNode`에 연결되는 외부 도구 발견입니다.
- 11단계 · 17강 (에이전트 프레임워크 트레이드오프) — CrewAI, AutoGen, Agno 대신 LangGraph를 선택해야 하는 시점입니다.
