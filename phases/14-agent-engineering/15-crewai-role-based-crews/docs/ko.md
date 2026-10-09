# 역할 기반 에이전트 팀 — 역할, 작업, 프로세스

> 네 가지 기본 요소: 에이전트(Agent), 작업(Task), 크루(Crew), 프로세스(Process). 두 가지 최상위 형태: 크루(Crews, 자율적 역할 기반 협업)와 플로우(Flows, 이벤트 기반 결정론적). CrewAI는 2026년 기준 참고 구현체이며, 공식 문서에서는 단호하게 "프로덕션 준비가 된 애플리케이션은 Flow로 시작하세요"라고 명시합니다.

**유형:** 학습 + 빌드
**언어:** Python (표준 라이브러리)
**선수 요건:** 14단계 · 12강 (워크플로우 패턴), 14단계 · 14강 (액터 모델)
**시간:** 약 75분

## 학습 목표

- CrewAI의 네 가지 기본 요소(Agent, Task, Crew, Process)를 나열하고 각각이 담당하는 영역을 설명해 보세요.
- 순차적(Sequential), 계층적(Hierarchical), 그리고 계획된 합의(Consensus) 프로세스를 구분하고, 각 워크로드에 하나를 선택해 보세요.
- 크루(Crews, 자율적 역할 기반)와 플로우(Flows, 이벤트 기반 결정론적)를 구분하고, 공식 문서의 프로덕션 권장 사항을 설명해 보세요.
- `@tool` 데코레이터와 `BaseTool` 하위 클래스로 도구를 연결하고, 구조화된 출력과 자유 텍스트를 비교해 보세요.
- CrewAI의 네 가지 메모리 유형을 나열하고 각각이 유용한 시점을 설명해 보세요.
- 연구자, 작가, 편집자로 구성된 세 에이전트 크루(표준 라이브러리 사용)를 구현하여 브리프를 생성해 보세요.
- CrewAI의 세 가지 실패 모드(프롬프트 팽창, 관리자 LLM 비용, 취약한 핸드오프)를 식별해 보세요.

## 문제점

멀티 에이전트 프레임워크를 도입한 팀은 동일한 벽에 부딪힙니다. "자율적 협업"은 데모에서는 훌륭해 보입니다. 하지만 고객이 버그를 신고하면 결정론적 재현이 필요해집니다. 재무 팀은 LLM 라우팅 크루의 실행당 비용이 얼마나 되는지 묻습니다. 온콜 담당자는 새벽 3시에 어떤 에이전트가 멈췄는지 알아야 합니다.

자유 형식 LLM 라우팅 크루는 이러한 질문에 명확하게 답하지 못합니다. 순수 DAG는 모든 질문에 답할 수 있지만, 브레인스토밍 에이전트가 필요한 탐색적 형태를 잃습니다.

CrewAI의 분리는 트레이드오프에 대해 솔직합니다. 협업적이고 역할 기반이며 탐색적인 작업에는 크루(Crews)를, 이벤트 기반이고 코드가 소유하며 감사 가능한 프로덕션에는 플로우(Flows)를 사용합니다. 동일한 프레임워크, 두 가지 형태, 표면(surface)에 따라 선택하세요.

## 개념

### 네 가지 기본 요소

CrewAI의 표면은 작습니다. 이것을 암기하면 나머지는 설정(config)입니다.

- **에이전트(Agent).** `role + goal + backstory + tools + (optional) llm`. 백스토리는 핵심 요소입니다. 톤, 판단, 에이전트가 멈추는 시점을 결정합니다. 도구는 에이전트가 호출할 수 있는 함수입니다 (아래에 더 설명).
- **태스크(Task).** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. 재사용 가능한 작업 단위입니다. `expected_output`이 계약입니다. `context`는 출력값이 전달되는 상류 태스크를 나열합니다. `output_pydantic`은 구조화된 형태를 강제합니다.
- **크루(Crew).** 컨테이너입니다. `agents` 목록, `tasks` 목록, `process`, 그리고 선택적인 `memory` + `verbose` + `manager_llm` 설정을 소유합니다.
- **프로세스(Process).** 실행 전략입니다. 순차적(Sequential), 계층적(Hierarchical), 합의(Consensus)(계획 중). 실행의 형태를 결정합니다.

에이전트들은 서로를 직접 보지 못합니다. 태스크는 에이전트를 참조합니다. 크루는 태스크를 순서대로 배치합니다. 프로세스는 다음 태스크를 누가 선택할지 결정합니다. 이것이 전체적인 개념 모델입니다.

> **검증 대상:** CrewAI 0.86 (2026-05). 최신 버전에서는 프로세스 유형이 이름이 변경되거나 병합될 수 있으므로, 특정 형태에 의존하기 전에 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)을 확인해 보세요.

### 순차적 vs 계층적 vs 합의

- **순차적(Sequential).** 태스크는 선언된 순서대로 실행됩니다. 태스크 N의 출력은 태스크 N+1에 `context`으로 사용 가능합니다. 비용이 가장 낮습니다. 가장 예측 가능합니다. 순서가 고정되어 있을 때 사용하세요.
- **계층적(Hierarchical).** 관리자 에이전트(Agent)(별도의 LLM 호출)가 전문가들 사이에서 라우팅합니다. CrewAI는 `manager_llm` 설정이나 기본값에서 관리자를 생성합니다. 관리자는 매 라운드마다 다음 태스크를 선택하며, 거부하거나 재라우팅할 수 있습니다. 네 명 이상의 전문가가 있고 순서가 이전 출력에 진정으로 의존할 때 사용하세요.
- **합의(Consensus).** 계획 중이며, 현재 공개 API에서 구현되지 않았습니다. 문서에서는 미래의 투표 기반 프로세스를 위해 이름을 예약하고 있습니다. 현재는 의존하지 마세요.

계층적 프로세스는 모든 전문가 호출 위에 라운드별 LLM 호출(관리자)을 추가합니다. 5단계 실행에서 토큰 비용이 3배가 될 수 있습니다. 라우팅이 필요할 때만 비용을 지불하세요.

### 크루(Crew) vs 플로우(Flow)

2026년 문서에서 주도하는 프레임입니다.

- **크루(Crew).** LLM 기반 자율성. 프레임워크가 런타임에 형태를 선택합니다. 연구, 브레인스토밍, 초안 작성, 경로가 답의 일부인 모든 곳에 적합합니다. 재현이 어렵습니다. 테스트가 어렵습니다. 프로토타입이 저렴합니다.
- **흐름.** 소유한 이벤트 기반 그래프입니다. `@start`이 진입점을 표시합니다. `@listen(topic)`은 다른 단계가 해당 주제를 방출할 때 실행되는 단계를 표시합니다. 각 단계는 순수 Python이며 (내부적으로 Crew를 호출할 수 있습니다). 장점: 프로덕션 환경에 적합합니다. 관측 가능하고, 테스트 가능하며, 결정적입니다.

문서에서 권장하는 2026년 프로덕션 전략: Flow로 시작하세요. 자율성이 비용을 정당화할 때 Flow 단계 내부에서 `Crew.kickoff()` 호출로 Crew를 통합하세요. Flow는 감사 추적(audit trail)을 제공하며, Crew는 탐색(exploration)을 제공합니다. 선택하는 것이 아니라 조합하세요.

### 도구 통합

Agent에 도구를 제공하는 세 가지 방법입니다. 가장 단순한 방법을 선택하세요.

1. **`@tool` 데코레이터.** 순수 함수가 도구가 됩니다. 시그니처가 스키마이며, docstring은 LLM이 보는 설명입니다. 일회성 헬퍼에 가장 적합합니다.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 하위 클래스.** 명시적인 args 스키마, async 지원, 재시도를 포함하는 클래스 기반 도구입니다. 도구가 상태(클라이언트, 캐시)를 가지거나 구조화된 args가 필요할 때 사용하세요.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **기본 제공 도구 키트.** CrewAI는 자체 어댑터를 제공합니다: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. 한 번의 import로 연결됩니다.

구조화된 출력은 Pydantic을 사용합니다. Task에 `output_pydantic=MyModel`을 전달하세요. CrewAI는 LLM 응답을 모델에 대해 검증하고, 강제 변환(coerce)하거나 재시도합니다. 이를 엄격한 `expected_output` 문자열과 함께 사용하세요. 자유 텍스트 출력은 초안 작성에 적합하며, 구조화된 출력은 하위 Flow가 소비할 수 있는 형태입니다.

### 메모리 훅

CrewAI는 네 가지 메모리 유형을 기본으로 제공합니다. 이들은 조합될 수 있습니다: Crew는 네 가지를 모두 활성화할 수 있습니다.

> **검증 대상:** CrewAI 0.86 (2026-05). 최근 릴리스는 이 네 가지 저장소를 래핑하는 통합 `Memory` 시스템을 통해 모든 것을 라우팅합니다. 아래 개념 모델은 여전히 유효하지만, 최신 버전에서는 공개 클래스 표면이 단일 `Memory` 진입점으로 축소될 수 있습니다. 현재 API는 [CrewAI memory docs](https://docs.crewai.com/concepts/memory)를 확인하세요.

- **단기.** 단일 실행 내의 대화 버퍼입니다. 종료 시 삭제됩니다.
- **장기.** 실행 간에 지속됩니다. 벡터 DB (기본은 Chroma, 교체 가능)에 저장됩니다. 현재 작업과의 유사성을 통해 검색됩니다.
- **엔티티.** 엔티티별 사실입니다. "고객 X는 엔터프라이즈 플랜에 있습니다." 유사성이 아닌 엔티티를 기준으로 키가 지정됩니다. 실행 간에 지속됩니다.
- **맥락적.** 조립 시점 검색. Agent가 필요로 하는 시점에 관련 메모리를 가져오며, 사전에 로드하지 않습니다.

Crew에서 `memory=True` 또는 유형별 구성으로 활성화하세요. 설정한 임베딩 제공자(기본값은 OpenAI, 로컬로 교체 가능)가 이를 뒷받침합니다. 메모리는 CrewAI가 더 얇은 프레임워크에 비해 가치를 입증하는 영역 중 하나입니다. 순수 LangGraph는 이러한 기능들을 직접 연결해야 합니다.

### 역할 기반 팀이 적합할 때

- 명명된 역할과 협업 워크플로우를 가진 3~6개의 에이전트. 초안 작성, 검토, 계획, 브레인스토밍.
- LLM의 다음 단계 판단이 가치의 일부인 라우팅(Hierarchical).
- 팀이 `role + goal + backstory`을 읽는 것이 그래프 정의를 읽는 것보다 더 만족스러운 모든 상황.

### 그들이 적합하지 않을 때

- 엄격한 순서를 가진 결정적 DAG. LangGraph를 사용하세요(13강). 그래프 형태가 올바른 추상화이며, CrewAI의 역할 프레이밍은 마찰을 유발합니다.
- 초 단위 지연 예산. Hierarchical은 왕복을 추가합니다. Sequential조차 백스토리 및 이전 출력을 포함하는 프롬프트를 직렬화합니다.
- 단일 에이전트 루프. 프레임워크를 건너뛰세요; 에이전트 루프(1강)와 도구 레지스트리가 더 짧습니다.

17강(Agent Framework Tradeoffs)은 이를 매트릭스로 정리합니다. 짧은 버전: CrewAI는 "협업 역할 기반" 코너에 위치합니다.

### 의존성 형태

LangChain과 독립적입니다. Python 3.10~3.13을 사용하며 `uv`을 사용합니다. 스타 수: [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) 참조(2026-05 스냅샷). AWS Bedrock 통합이 문서화되어 있습니다. 벤더 벤치마크는 QA 워크로드에서 LangGraph 대비 상당한 속도 향상을 보고하지만, 방법론(데이터셋, 하드웨어, 평가 지표)이 공개되지 않았으므로 프레임워크 벤더 수치는 방향성으로만 취급하세요.

### 이 패턴이 잘못된 곳

- **백스토리에서 오는 프롬프트 팽창.** 에이전트당 2000단어 백스토리와 5인 Crew는 첫 도구 호출 전에 컨텍스트 예산을 소진합니다. 백스토리를 200단어 미만으로 유지하세요. 에이전트 간에 문구를 재사용하세요; 하우스 스타일을 다섯 번 반복하지 마세요.
- **매니저 LLM 토큰 비용.** 계층적 프로세스는 모든 전문가 호출 전에 매니저 LLM 호출을 추가합니다. 5개 작업 크루의 경우 5개 대신 6개의 LLM 호출이 발생하며, 매니저 호출은 전체 작업 목록과 이전 출력 결과를 포함합니다. 라우팅이 출력에 의존하지 않는 한 Sequential로 전환하세요.
- **취약한 핸드오프.** 작업 N의 `expected_output`은 "개요"입니다. 작업 N+1은 이를 `context`로 읽고 세 섹션을 파싱하려고 시도합니다. LLM은 네 개를 생성했습니다. 다운스트림 Agent는 즉흥적으로 처리합니다. 작업 N에 `output_pydantic`를 적용하여 작업 N+1이 자유 텍스트가 아닌 타입 지정된 객체를 읽도록 수정하세요.
- **크루를 프로덕션으로.** Flow 래퍼 없이 자유 형식 Crew가 프로덕션에 배포되었습니다. 출력 변이가 높고, 재현이 불가능하며, 온콜 담당자가 나쁜 실행과 좋은 실행을 비교할 수 없습니다. Flow로 래핑하세요.

```figure
ae-crew-vs-flow
```

## 구현하기

`code/main.py`은 두 가지 형태와 세 에이전트 크루의 스탠다드 라이브러리 버전을 구현합니다.

형태:

- CrewAI의 인터페이스와 일치하는 `Agent`, `Task` 데이터 클래스.
- `SequentialCrew.kickoff(inputs)`은 선언 순서대로 작업을 실행하며, 출력을 `context`로 연결합니다.
- `HierarchicalCrew.kickoff(topic)`는 매니저 Agent가 매 라운드마다 다음 전문가를 선택하도록 추가하며, "done"에서 멈춥니다.
- `Flow`는 `@start` 및 `@listen(topic)` 데코레이터, 작은 이벤트 루프, 추적(trace)을 포함합니다.
- CrewAI의 `@tool` 형태를 반영하는 `tool(name)` 데코레이터.
- `Memory`는 `short_term`, `long_term`, `entity` 저장소를 포함하며, 모의 유사도는 numpy를 사용합니다.
- 모의 LLM 응답은 역할과 입력 접두어에 따라 하드코딩된 문자열입니다. 네트워크 없음. 결정적입니다.

구체적인 데모: 연구자, 작가, 편집자 크루가 "agent engineering 2026"에 대한 브리프를 생성합니다. 연구자는 (모의) 출처를 가져옵니다. 작가는 초안을 작성합니다. 편집자는 다듬습니다. 같은 크루가 Flow를 통해 실행되어 결정적인 형태를 보여줍니다.

실행하기:

```bash
python3 code/main.py
```

추적(trace)은 다음을 포함합니다: 순차적 크루가 `context`를 통해 출력을 연결하는 것, 매니저가 선택하는 계층적 크루(연구자, 작가, 편집자, 그 후 "done"), 명시적 주제(`researched`, `drafted`, `edited`)로 동일한 세 단계를 실행하는 Flow, `@tool`를 통해 라우팅된 도구 호출, 그리고 두 번의 킥오프에 걸쳐 유지되는 장기 메모리.

Crew 추적은 유동적입니다. 매니저는 원칙적으로 순서를 재배열할 수 있습니다. Flow 추적은 고정되어 있습니다. 그 선택이 교훈입니다.

## 사용하기

- 프로덕션 환경에서는 **CrewAI Flow**를 사용하세요. Flow가 `Crew.kickoff()`을 호출하는 단일 단계인 경우에도 Flow가 감사 경계를 제공합니다.
- 명확한 순서로 협업하는 작업, 특히 초안 작성 및 검토 루프에는 **CrewAI Crew (순차적)**를 사용하세요.
- 출력에 따라 라우팅이 결정되고 네 명 이상의 전문가가 있는 경우 **CrewAI Crew (계층적)**를 사용하세요.
- 명시적인 상태 머신, 내구성 있는 재개, 엄격한 순서 제어에는 **LangGraph** (13강)를 사용하세요.
- 액터 모델 동시성 및 장애 격리에는 **AutoGen v0.4** (14강)를 사용하세요.
- 핸드오프 및 가드레일을 갖춘 OpenAI 우선 제품에는 **OpenAI Agents SDK** (16강)를 사용하세요.
- 서브 에이전트 및 세션 저장소를 갖춘 Claude 우선 제품에는 **Claude Agent SDK** (17강)를 사용하세요.

## 출시하기

`outputs/skill-crew-or-flow.md`은 작업에 Crew와 Flow 중 하나를 선택하고 최소한의 구현을 스캐폴딩합니다. 백스토리가 없는 Crew, 명시적 주제가 없는 Flow, 세 명 미만의 전문가를 사용하는 계층적 구조에 대해 엄격하게 거부합니다.

## 문제점

- **백스토리를 단순한 맛으로 취급하지 마세요.** 백스토리는 출력에 영향을 미칩니다. 각 에이전트마다 세 가지 변형을 테스트해 보세요. 변동성은 실제로 존재합니다. 하나를 선택하고 고정하세요.
- **`expected_output`을 건너뛰지 마세요.** 작업별 계약이 없으면 하류 작업은 LLM이 생성한 모든 것을 가져옵니다. Crew는 실행되지만 감사에 실패합니다.
- **메모리를 항상 켜두지 마세요.** 장기 메모리는 모든 실행에서 기록합니다. 벡터 DB가 커지고 검색이 노이즈가 많아집니다. 사실이 지속되는 작업에만 기록을 범위를 지정하세요.
- **매니저 프롬프트 드리프트.** 계층적 구조의 매니저 프롬프트는 암묵적입니다. 라우팅이 이상해지면 상세 모드에서 덤프하고 읽어보세요.
- **Crew에서의 도구 부작용.** Crew는 예상보다 많은 횟수로 도구를 호출할 수 있습니다. POST, DELETE, 결제는 Flow 단계에 속하며 Crew 도구에 속하지 않습니다.

## 연습 문제

1. 순차적 Crew를 Flow로 변환하세요. 변동성이 감소하는 접점을 세어 보세요. 가독성이 감소한 부분을 기록하세요.
2. Crew에 엔티티 메모리를 추가하세요: 고객에 대한 사실이 킥오프 간에 지속됩니다. 검색이 올바른 엔티티를 가져오는지 확인하세요.
3. 매니저가 작성자의 출력에 최소 세 단락이 있을 때까지 편집자에게 라우팅을 거부하는 계층적 프로세스를 구현하세요. 재시도를 추적하세요.
4. `BaseTool` 서브클래스를 연결하여 (모의) 웹 검색을 구현해 보세요. `@tool` 데코레이터 버전과 추적(trace) 형태를 비교해 보세요.
5. 편집자(editor) 작업에 `output_pydantic=Brief`를 추가하세요. `Brief`는 `title`, `summary`, `sections`를 포함합니다. 작성자(writer) 작업이 한 번만 잘못된 JSON을 출력하도록 만들고, 추적(trace)에서 CrewAI의 재시도(retry) 동작을 확인해 보세요.
6. CrewAI 문서의 소개 부분을 읽어 보세요. 장난감(toy) 프로젝트를 실제 `crewai` API로 이식해 보세요. 표준 라이브러리 버전이 생략한 보장은 무엇인가요?
7. AgentOps 또는 Langfuse (24강)를 실제 실행에 연결해 보세요. 표준 라이브러리 버전에서 놓친 추적(trace)은 무엇인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 에이전트(Agent) | "페르소나" | 역할 + 목표 + 백스토리 + 도구 |
| 작업(Task) | "작업 단위" | 설명 + 예상 출력 + 담당자 + 선택적 구조화된 출력 |
| 크루(Crew) | "에이전트 팀" | 에이전트 + 작업 + 프로세스의 컨테이너 |
| 프로세스(Process) | "실행 전략" | 순차적 / 계층적 / 합의 (계획됨) |
| 플로우(Flow) | "결정적 워크플로" | 이벤트 주도, 코드 소유, 테스트 가능 |
| 백스토리(Backstory) | "페르소나 프롬프트" | 에이전트(Agent)의 어조와 판단을 형성하는 요소 |
| `@tool` | "함수 도구" | 함수를 에이전트(Agent)가 호출할 수 있는 도구로 변환하는 데코레이터 |
| `BaseTool` | "클래스 도구" | 인자 스키마, 재시도, 비동기 지원이 있는 클래스 기반 도구 |
| 엔티티 메모리(Entity memory) | "엔티티별 사실" | 고객 / 계정 / 이슈에 범위가 지정된 메모리 |
| 장기 메모리(Long-term memory) | "실행 간 메모리" | 킥오프(kickoff) 간에 유지되는 벡터 기반 메모리 |
| 컨텍스트 메모리(Contextual memory) | "적시 검색" | 에이전트(Agent)가 필요로 하는 순간에 가져오는 메모리 |
| 관리자 LLM(Manager LLM) | "라우터 에이전트" | 계층적 프로세스에서 다음 작업을 선택하는 추가 LLM |
| `expected_output` | "작업 계약" | 에이전트(Agent) (및 감사)에 반환 형태를 알려주는 문자열 |

## 추가 읽기

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): 개념 및 권장되는 프로덕션 경로
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): 이벤트 주도 형태, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, 내장 도구 키트
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): 단기, 장기, 엔티티, 컨텍스트
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 멀티 에이전트가 도움이 되는 경우와 도움이 되지 않는 경우
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 상태 머신 대안
