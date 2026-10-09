---
name: state-graph
description: LangGraph 형태의 상태 머신을 구축합니다. 타입이 지정된 상태, 조건부 엣지, 노드별 체크포인팅, 내구성 있는 재개 기능을 포함합니다.
version: 1.0.0
phase: 14단계
lesson: 13강
tags: [langgraph, state-machine, durable, checkpointing, human-in-the-loop]
---

목표 런타임, 상태 형태, 노드 함수 집합, 체크포인터 백엔드가 주어지면, 상태 유지 에이전트 그래프를 생성합니다.

생성 대상:

1. 타입이 지정된 `State` (dict 또는 Pydantic). 모든 필드를 문서화합니다. 노드는 상태를 읽으며, 업데이트를 반환합니다.
2. `StateGraph`와 `add_node`, `add_edge`, `add_conditional_edges`, `set_entry`, 그리고 `START`/`END` 센티넬을 포함합니다.
3. `Checkpointer` 인터페이스와 `save(session_id, node, state)` 및 `load_latest(session_id)`를 포함합니다. SQLite를 기본으로 사용하며, Postgres/Redis/커스텀을 허용합니다.
4. 그래프를 단계별로 실행하고, 모든 노드 실행 후 상태를 직렬화하며, 인간 개입 루프(Human-in-the-Loop)를 위해 `PausedAtNode`를 포착하고, 선택적 `state_override`와 함께 `resume_from`를 지원하는 `Runner`를 포함합니다.
5. 세 가지 토폴로지 헬퍼: supervisor (중앙 라우터), swarm (공유 도구 핸드오프), hierarchical (서브그래프).

엄격한 거부 조건:

- 명시적인 랜덤 시드나 벽시계(wall-clock) 캡처가 없는 비결정적 노드. 재개 기능은 입력 상태가 주어졌을 때 노드 출력이 재현 가능하다고 가정합니다.
- "요약" 상태만 저장하는 체크포인터. 전체 상태를 직렬화해야 하며, 그렇지 않으면 재개가 실패합니다.
- 모든 엣지가 조건부인 그래프. 간헐적인 분기를 가진 선형 체인을 선호합니다.

거부 규칙:

- 사용자가 지속성(persistence) 없는 상태 그래프를 요청하면 거부합니다. 핵심은 내구성 있는 재개(durable resume)입니다. 재개가 필요하지 않다면, 12강의 워크플로우 패턴을 사용하세요.
- 사용자가 "성공 시에만 체크포인팅"을 요청하면 거부합니다. 실패도 상태를 필요로 합니다. 디버깅이 시작되는 지점입니다.
- 그래프가 약 30개 이상의 노드를 포함하면, 평면(flat) 레이아웃을 거부하고 중첩된 서브그래프를 요구합니다. 평면 30노드 그래프는 검토가 불가능합니다.

출력: `state.py`, `graph.py`, `checkpointer.py`, `runner.py`, `README.md`. 상태 스키마, 체크포인터 선택, 재개 시맨틱을 설명합니다. "다음에 읽을 내용"으로 마무리하며, 액터 모델 대안인 14강, 핸드오프/가드레일 레이어인 16강, 그래프 단계의 OTel 스팬인 23강을 가리킵니다.
