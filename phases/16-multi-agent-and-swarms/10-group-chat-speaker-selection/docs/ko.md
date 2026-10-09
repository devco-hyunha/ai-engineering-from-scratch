# 그룹 채팅 및 화자 선택

> 공유 대화 오케스트레이션은 N개의 에이전트를 하나의 대화에 배치합니다. 선택 함수(LLM, 라운드 로빈, 또는 사용자 정의)가 다음에 누가 말할지 결정합니다. 이는 다중 에이전트 대화의 원형입니다. 에이전트는 정적 그래프에서의 역할을 알지 못하며, 공유된 풀에 단순히 반응할 뿐입니다. AutoGen GroupChat와 AG2 GroupChat은 참조 구현입니다. AutoGen v0.2의 GroupChat 의미론은 AG2 포크에서 유지되었으며, AutoGen v0.4는 이를 이벤트 기반 액터 모델로 재작성했습니다. Microsoft는 2026년 2월에 AutoGen을 유지보수 모드로 전환하고 Semantic Kernel과 병합하여 Microsoft Agent Framework(RC 2026년 2월)를 만들었습니다. GroupChat 프리미티브는 AG2와 Microsoft Agent Framework 모두에서 유지되므로, 한 번만 배우면 어디서나 사용할 수 있습니다.

**유형:** 학습 + 빌드
**언어:** Python (표준 라이브러리)
**선수 요건:** 16단계 · 04강 (프리미티브 모델)
**시간:** 약 60분

## 문제점

정적 그래프(LangGraph)는 워크플로우가 명확할 때 유용합니다. 실제 대화는 정적이지 않습니다. 코더가 리뷰어에게 물어볼 수도 있고, 연구자나 작가에게 물어볼 수도 있습니다. 모든 가능한 핸드오프를 하드코딩하면 엣지 폭발이 발생합니다. *에이전트가 공유된 풀에 반응*하고, 어떤 함수가 다음 화자를 결정하는 구조가 필요합니다.

AutoGen GroupChat이 바로 이 역할을 수행합니다.

## 개념

### 구조

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

모든 에이전트가 모든 메시지를 봅니다. 각 턴마다 선택 함수가 호출되어 다음 화자를 결정합니다.

### 세 가지 선택자 유형

**라운드 로빈.** 고정된 주기. 결정적입니다. N에 대해 선형적으로 확장되지만 컨텍스트를 무시합니다. 주제가 법률 검토인 경우에도 코더가 턴을 얻습니다.

**LLM 선택.** 최근 풀을 읽고 최적의 다음 화자를 반환하는 LLM 호출입니다. 컨텍스트를 인식하지만 느립니다. 매 턴마다 LLM 호출이 추가됩니다. AutoGen의 기본값입니다.

**사용자 정의.** 원하는 로직을 가진 Python 함수입니다. 일반적으로 LLM 선택과 폴백 규칙(예: "코더 이후에는 항상 검증자에게 턴을 준다")을 결합합니다.

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager`가 셀렉터를 저장합니다. 에이전트(Agent)가 한 턴을 완료하면, 매니저가 셀렉터를 호출하여 다음 에이전트를 반환합니다. 종료 조건(Termination Condition)이 충족될 때까지 루프가 계속됩니다.

### 종료

세 가지 일반적인 패턴:

- **최대 라운드.** 총 턴 수에 대한 하드 캡(Hard cap)입니다.
- **"TERMINATE" 토큰(Token).** 에이전트(Agent)가 센티널(sentinel) 메시지를 방출할 수 있으며, 매니저는 이 메시지가 나타나면 멈추는 방식입니다.
- **목표 달성 체크.** 가벼운 검증기가 매 턴마다 실행되며, 완료되면 채팅을 멈추는 방식입니다.

### 계보: 포크(Fork)와 병합(Merger)

2025년 초, Microsoft는 이벤트 기반 액터(actor) 모델을 중심으로 AutoGen(v0.4)의 대규모 재작성을 시작했습니다. 커뮤니티는 초기 채택자들이 통합했던 API를 보존하기 위해 AutoGen v0.2의 GroupChat 시맨틱을 AG2로 포크(Fork)했습니다.

2026년 2월, Microsoft는 AutoGen이 유지보수 모드(maintenance mode)로 전환되며, 이벤트 기반 액터 모델이 **Microsoft Agent Framework**에 병합될 것이라고 발표했습니다(RC 2026년 2월, 현재 Semantic Kernel과 병합됨). GroupChat 개념은 두 트랙 모두에서 살아남으며, 구현 세부 사항은 다릅니다. AG2는 v0.2 호환 코드를 위한 선호되는 상류(upstream)입니다.

### GroupChat이 적합한 경우

- **창발적 대화(Emergent conversations).** 모든 가능한 다음 화자를 사전에 연결(wire)하고 싶지 않은 경우입니다.
- **역할 혼합 작업.** 코더(Coder)가 리서처(Researcher)에게 묻고, 리서처가 아카이비스트(Archivist)에게 묻고, 아카이비스트가 코더에게 다시 묻는 식입니다. 흐름은 DAG가 아닙니다.
- **탐색적 문제 해결.** "조립 라인"이 아닌 "브레인스토밍 회의"를 생각하세요.

### 실패하는 경우

- **엄격한 결정론(Determinism).** LLM 셀렉터는 일관성이 없을 수 있습니다. 같은 프롬프트(Prompt)라도 실행마다 다음 화자가 달라질 수 있습니다.
- **아첨(Sycophancy) 연쇄.** 에이전트(Agent)가 가장 자신 있게 말한 사람에게 따르는 경향이 있습니다. 카운터 프롬프트(counter-prompt)를 명시적으로 사용하세요.
- **컨텍스트 팽창(Context bloat).** 모든 에이전트(Agent)가 모든 메시지를 읽기 때문에, 10턴 이후에는 컨텍스트가 거대해집니다. 투영(projections)(15강)을 사용하여 뷰를 범위로 제한하세요.
- **핫 스피커(Hot speakers).** 셀렉터가 특정 에이전트(Agent)의 전문 분야를 선호하기 때문에 한 에이전트가 대화를 지배합니다. 셀렉터 기능으로 화자 균형을 도입하세요.

### Group chat vs Supervisor

동일한 프리미티브(primitive)를 사용하지만 기본값이 다릅니다:

- Supervisor: 한 에이전트(Agent)가 계획하고 나머지가 실행합니다. 셀렉터는 "계획자에게 무엇을 해야 하는지 물어보는" 방식입니다.
- 그룹 채팅: 모든 에이전트(Agent)는 동등한 관계이며, 선택기는 공유 풀에 대한 함수입니다.

두 방식 모두 04강의 네 가지 기본 요소를 사용합니다. 그룹 채팅은 LLM 선택 기반 오케스트레이션과 전체 풀 공유 상태를 기본으로 합니다.

```figure
swarm-speaker
```

## 구현하기

`code/main.py`는 표준 라이브러리(stdlib)만으로 GroupChat을 처음부터 구현합니다. 세 개의 에이전트(코더, 리뷰어, 관리자), 라운드 로빈 및 LLM 선택 변형, 그리고 `TERMINATE` 토큰에 대한 종료 조건을 포함합니다.

데모는 두 변형에 대한 대화 기록과 선택기의 결정 추적(decision trace)을 출력합니다.

실행:

```
python3 code/main.py
```

## 사용하기

`outputs/skill-groupchat-selector.md`는 주어진 작업에 대한 GroupChat 선택기를 구성합니다. 라운드 로빈 vs LLM 선택 vs 사용자 정의 방식, 그리고 어떤 선택기 입력(최근 메시지, 에이전트 전문 분야, 턴 수)을 사용할지 결정합니다.

## 출시하기

체크리스트:

- **최대 라운드 상한.** 항상 설정하세요. 일반적인 작업에는 10-20 라운드가 적당합니다.
- **화자 균형 지표.** 에이전트별 턴 수를 추적하고, 불균형이 임계값을 초과하면 알림을 보내세요.
- **종료 토큰.** `TERMINATE` 또는 전용 검증 에이전트(Verifier Agent)를 사용하세요.
- **투영(Projection) 또는 범위 지정된 메모리.** 약 10개의 메시지 이후에는 컨텍스트 팽창을 방지하기 위해 각 에이전트에게 범위 지정된视图(scoped view)만 제공하는 것을 고려하세요.
- **선택기 로깅.** LLM 선택 변형의 경우, 선택기의 입력과 선택 결과를 모두 로깅하세요. 그렇지 않으면 디버깅이 불가능합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 라운드 로빈과 LLM 선택 방식에서의 대화를 비교하세요. 각 방식에서 어떤 에이전트가 주도권을 잡나요?
2. 선택기에 "agent별 최대 발화 횟수(max-speaks-per-agent)" 규칙을 추가하세요. 이것이 대화 기록(transcript)에 어떤 영향을 미치나요?
3. 목표 달성 종료(goal-reached termination)를 구현하세요. 리뷰어가 "approved"를 반환하면 중지하세요. 라운드 상한에 도달하기 전에 이 조건이 얼마나 자주 트리거되나요?
4. AutoGen 안정 버전 문서의 GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)를 읽어보세요. `GroupChatManager`가 사용하는 기본 선택기를 식별하세요.
5. AG2 저장소(https://github.com/ag2ai/ag2)를 읽고, v0.2 GroupChat을 v0.4 이벤트 기반 버전과 비교하세요. v0.4는 어떤 구체적인 속성(처리량, 내결함성, 구성 가능성)을 추가하나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| GroupChat | "하나의 채팅방에 있는 에이전트들" | 공유 메시지 풀 + 선택 함수. AutoGen / AG2 기본 요소. |
| Speaker selection | "다음에 누가 말할지" | 다음 에이전트를 선택하는 함수. 라운드 로빈, LLM 선택, 또는 사용자 정의. |
| GroupChatManager | "회의 진행자" | 선택자를 소유하고 턴을 반복하는 AutoGen 구성 요소. |
| ConversableAgent | "기본 에이전트" | AutoGen 기본 클래스; 메시지를 보내고 받을 수 있는 에이전트. |
| Termination token | "'중단' 단어" | 채팅을 종료하는 센티널 문자열 (일반적으로 `TERMINATE`). |
| Hot speaker | "한 에이전트가 지배함" | 선택자가 같은 에이전트를 계속 선택하는 실패 모드. |
| Context bloat | "풀이 무한히 커짐" | 각 에이전트가 모든 이전 메시지를 읽음; 컨텍스트가 턴에 따라 커짐. |
| Projection | "범위 한정된 뷰" | 컨텍스트 팽창을 방지하기 위해 공유 풀에 대한 역할별 뷰. |

## 추가 읽기

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 참조 구현
- [AG2 repo](https://github.com/ag2ai/ag2) — 커뮤니티 AutoGen v0.2 연속성
- [Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/) — 통합된 후속 제품, 2026년 2월 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 이벤트 기반 액터 모델 재작성 세부 사항
