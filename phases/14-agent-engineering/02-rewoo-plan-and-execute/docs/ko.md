# ReWOO와 Plan-and-Execute: 분리된 계획

> ReAct는 생각과 행동을 하나의 스트림에서 교대로 수행합니다. ReWOO는 이를 분리합니다: 먼저 큰 계획을 세우고, 그 후 실행합니다. 토큰 사용량이 5배 감소하고, HotpotQA에서 정확도가 +4% 향상되며, 플래너를 7B 모델로 증류(distill)할 수 있습니다. Plan-and-Execute는 이를 일반화했고, Plan-and-Act는 이를 웹 내비게이션으로 확장했습니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프)
**시간:** 약 60분

## 학습 목표

- ReWOO의 Planner / Worker / Solver 분리가 ReAct의 교대로 수행되는 루프보다 토큰을 절약하고 강건성을 개선하는 이유를 설명해 보세요.
- 계획 DAG, 의존성 순서 기반 실행기, 워커 출력들을 조합하는 솔버를 모두 stdlib로 구현해 보세요.
- 2026년 Anthropic의 "5가지 워크플로우 패턴" 프레임워크를 사용하여, 작업이 plan-then-execute로 실행되어야 하는지 interleaved ReAct로 실행되어야 하는지 결정해 보세요.
- 장기적인 웹 또는 모바일 작업에 Plan-and-Act의 합성 계획 데이터가 필요한 시점을 인식해 보세요.

## 문제점

ReAct의 교대로 수행되는 thought-action-observation 루프는 단순하고 유연하지만, 각 도구 호출은 모든 이전 컨텍스트 — 모든 이전 생각을 포함 — 를 전달해야 합니다. 토큰 사용량은 깊이에 따라 2차적으로 증가합니다. 더 나쁜 점은: 루프 중간에 도구가 실패하면, 모델은 오류 관측치로부터 전체 계획을 다시 유도해야 합니다.

ReWOO (Xu et al., arXiv:2305.18323, 2023년 5월)는 이를 발견하고 한 가지 베팅을 했습니다: 전체를 미리 계획하고, 증거를 병렬로 수집하며, 마지막에 답을 조합하는 것입니다. 계획에 LLM 호출 1회, 증거 수집에 N개의 도구 호출 (병렬 가능), 해결에 LLM 호출 1회. 이 트레이드오프는 유연성 감소 (계획이 정적임)를 대가로 훨씬 더 나은 토큰 효율성과 명확한 실패 모드를 얻는 것입니다.

## 개념

### 세 가지 역할

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner는 DAG를 생성합니다. 각 노드는 도구, 그 인자, 그리고 의존하는 이전 노드들 (`#E1`, `#E2` 같은 참조)을 지정합니다. Workers는 위상 순서로 노드를 실행합니다. Solver는 모든 것을 함께 엮습니다.

### 토큰이 5배 감소하는 이유

ReAct는 단계 수에 따라 프롬프트 길이가 선형으로 증가합니다. 10단계에서는 프롬프트에 thought 1, action 1, observation 1, thought 2, action 2, observation 2 등이 포함되며, 각 중간 단계는 원본 프롬프트를 중복 포함합니다.

ReWOO는 하나의 큰 planner 프롬프트, N개의 작은 worker 프롬프트(각각 도구 호출만 포함, 체인 없음), 그리고 하나의 solver 프롬프트를 사용합니다. HotpotQA에서 논문은 토큰 수를 약 5배 줄이면서 절대 정확도를 +4 향상시켰다고 측정했습니다.

### 더 견고한 이유

ReAct에서 worker 3가 실패하면, 루프는 스트림 중간에서 오류를 추론하여 해결해야 합니다. ReWOO에서는 worker 3가 오류 문자열을 반환하며, solver는 원본 계획과 함께 컨텍스트에서 이를 보고 우아한 저하(Graceful Degradation)를 수행할 수 있습니다. 실패 위치 파악은 단계별이 아닌 노드별입니다.

### Planner 증류(Knowledge Distillation)

논문의 두 번째 결과: planner는 observation을 보지 않으므로, 175B teacher의 planner 출력으로 7B 모델을 미세 조정(Fine-tuning)할 수 있습니다. 작은 모델이 계획(planning)을 담당하며, 추론(Inference) 시 큰 모델은 필요하지 않습니다. 이는 현재 표준이 되었으며, 많은 2026년 생산 에이전트들은 작은 planner와 큰 executor를 사용하거나 그 반대로 사용합니다.

### Plan-and-Execute (2023)

LangChain 팀의 2023년 8월 게시물은 ReWOO를 Plan-and-Execute라는 패턴 이름으로 일반화했습니다. 사전 planner가 단계 목록을 생성하고, executor가 각 단계를 실행하며, 선택적 replanner가 결과를 관찰한 후 수정할 수 있습니다. 이는 replanner가 observation을 계획에 다시 가져오므로 ReWOO보다 ReAct에 가깝지만, 토큰 절감 효과는 유지합니다.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act는 이 패턴을 장기 웹 및 모바일 에이전트로 확장합니다. 주요 기여점은 합성 계획 데이터입니다. 레이블이 지정된 궤적 생성기가 계획이 명시된 훈련 데이터를 생성합니다. 이를 통해 WebArena 유사 작업에서 단일 ReAct 궤적이 일관성을 잃는 30~50 단계를 넘어서도 작동하는 planner 모델을 미세 조정(Fine-tuning)하는 데 사용합니다.

### 어떤 패턴을 선택해야 하는가

| 패턴 | 사용 시점 |
|---------|------|
| ReAct | 짧은 작업, 알 수 없는 환경, 반응적 예외 처리 필요 |
| ReWOO | 알려진 도구를 가진 구조화된 작업, 토큰 민감도, 병렬화 가능한 증거 |
| Plan-and-Execute | ReWOO와 유사하지만 부분 실행 후 재계획(replanning) 포함 |
| Plan-and-Act | 장기적 작업 (>30단계), 웹/모바일/컴퓨터 사용 |
| Tree of Thoughts | 탐색 비용이 정당화되는 경우 (04강) |

Anthropic의 2024년 12월 가이드라인: 가장 단순한 것부터 시작하세요. 작업이 도구 호출 한 번과 요약으로 이루어진다면 ReWOO를 구축하지 마세요. 작업이 40단계의 연구 과제라면 ReAct만으로는 수행하지 마세요.

```figure
rewoo-plan
```

## 구현하기

`code/main.py`는 장난감 ReWOO를 구현합니다:

- `Planner` — 프롬프트에서 계획 DAG를 생성하는 스크립트 정책입니다.
- `Worker` — 레지스트리를 통해 각 노드의 도구 호출을 디스패치합니다.
- `Solver` — 증거를 읽고 최종 답변을 생성하는 스크립트 조합입니다.
- 의존성 해결 — `#E1`와 같은 참조는 이전 워커 출력으로 대체됩니다.

데모는 "프랑스 수도의 인구를 백만 단위로 반올림하면 얼마인가요?"라는 질문에 두 단계 계획으로 답합니다: (1) 수도를 조회하고, (2) 인구를 조회한 후 해결합니다.

실행하세요:

```
python3 code/main.py
```

추적은 전체 계획을 먼저 표시한 후 워커 결과, 그리고 솔버 조합을 표시합니다. 토큰 수(대략적인 문자 수를 출력합니다)를 ReAct 스타일의 인터리브드 실행과 비교해 보세요. ReWOO는 이런 구조화된 작업에서 승리합니다.

## 사용하기

LangGraph는 Plan-and-Execute를 레시피로 제공합니다(ReAct용 `create_react_agent`, 계획-실행용 커스텀 그래프). CrewAI의 Flows는 이 패턴을 직접 인코딩합니다: 작업을 미리 정의하면 Flow DAG가 이를 실행합니다. Plan-and-Act의 합성 데이터 접근 방식은 여전히 대부분 연구 단계에 있으며, 런타임 패턴(명시적 계획 DAG)은 LangGraph와 CrewAI Flows를 통해 프로덕션에 출시됩니다.

## 출시하기

`outputs/skill-rewoo-planner.md`는 도구 카탈로그를 고려하여 사용자 요청에서 ReWOO 계획 DAG를 생성합니다. 실행기에 전달하기 전에 계획을 검증합니다(비순환적, 모든 참조가 해결됨, 모든 도구가 존재함).

## 연습 문제

1. 독립적인 계획 노드에 대해 워커 실행을 병렬화하세요. 2개의 병렬 그룹을 가진 6노드 DAG에서 어떤 이점을 얻을 수 있나요?
2. 워커가 오류를 반환하면 실행되는 재계획(replanner) 노드를 추가하세요. ReWOO를 Plan-and-Execute로 만드는 가장 작은 변경은 무엇인가요?
3. `Planner`을 작은 모델(7B급)로 교체하고 `Solver`은 프론티어 모델에 유지하세요. 엔드투엔드 품질을 비교해 보세요 — 분할이 실패하는 지점은 어디인가요?
4. ReWOO 논문 4절을 읽어보세요. 플래너 증류에 관한 내용입니다. 175B -> 7B 결과를 개념적으로 재현해 보세요: 어떤 학습 데이터가 필요하며, 계획 품질은 어떻게 점수화하나요?
5. 토이 프로젝트를 Plan-and-Act의 궤적 형태에 이식해 보세요: 계획은 DAG가 아니라 시퀀스입니다. 어떤 트레이드오프가 변하나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ReWOO | "관측 없는 추론" | 계획한 후, 병렬로 증거를 가져오고, 해결합니다 — 계획 프롬프트에 관측이 없습니다 |
| Plan-and-Execute | "LangChain의 plan-execute 패턴" | 실행 후 선택적 replanner 노드가 있는 ReWOO |
| Plan-and-Act | "확장된 plan-execute" | 장기 하위 작업(long-horizon tasks)을 위한 합성 계획 학습 데이터를 사용하여 플래너/실행자를 명시적으로 분리 |
| 증거 참조 | "#E1, #E2, ..." | 디스패치 시 이전 워커 출력으로 대체되는 계획 노드 자리표시자 |
| 플래너 증류 | "작은 플래너, 큰 실행자" | 큰 교사로부터 플래너 트레이스를 사용하여 작은 모델을 미세 조정 |
| 토큰 효율성 | "더 적은 왕복" | 논문에서 HotpotQA 기준 ReAct 대비 토큰이 5배 적음 |
| DAG 실행자 | "위상 디스패처" | 계획 노드를 의존성 순서로 실행; 각 레벨에서 병렬 실행 |

## 추가 읽기

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 표준 논문
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 합성 계획을 사용하는 확장된 플래너-실행자
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — 프레임워크 레시피
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 작동하는 가장 단순한 패턴을 선택하세요
