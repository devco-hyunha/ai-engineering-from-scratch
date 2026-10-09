---
name: crew-or-flow
description: 주어진 작업에 CrewAI Crew 또는 Flow를 선택하고, 최소한의 구현을 스캐폴딩합니다.
version: 1.0.0
phase: 14단계
lesson: 15강
tags: [crewai, crews, flows, multi-agent, role-based]
---

작업 설명을 바탕으로 Crew(자율) 또는 Flow(결정적)를 선택한 후, 스캐폴딩을 진행해 보세요.

결정:

1. 작업에 SLA, 컴플라이언스, 또는 결정적 재실행 요구사항이 있나요? -> Flow를 선택하세요.
2. 작업이 탐구적(연구, 초안 작성, 브레인스토밍)인가요? -> Crew를 선택하세요.
3. 작업에 LLM이 순서를 선택하는 4명 이상의 전문가가 있나요? -> Hierarchical Crew를 선택하세요.
4. 작업에 고정된 순서로 3명 이하의 전문가가 있나요? -> Sequential Crew 또는 Flow를 선택하세요. Flow를 선호합니다.

Crew의 경우, 다음을 생성하세요:

1. Agent 정의: 역할, 목표, 백스토리(간결하게, 200단어 이하), 도구.
2. Task 정의: 설명, expected_output, agent.
3. 적절한 Process(Sequential | Hierarchical)를 가진 Crew.
4. 샘플 입력으로 Crew를 실행하고 expected_output가 생성되는지 확인하는 테스트 하네스.

Flow의 경우, 다음을 생성하세요:

1. `@start` 진입 함수.
2. DAG를 형성하는 `@listen(topic)` 단계들.
3. 명시적인 이벤트 토픽; 마법 같은 브로드캐스트는 사용하지 마세요.
4. 재실행 하네스: 킥오프 페이로드가 주어지면 결정적으로 재실행합니다.

거부 규칙:

- 백스토리가 없는 Crew는 거부하세요. 백스토리는 핵심적인 역할을 합니다.
- 명시적인 토픽 이름이 없는 Flow는 거부하세요. "암묵적 체이닝"은 감사 목적을 무너뜨립니다.
- 2명의 전문가를 가진 Hierarchical Crew는 거부하세요. 매니저 오버헤드가 비용을 정당화하지 못합니다.

거부 규칙:

- 사용자가 프로덕션 전용 컴플라이언스 작업에 Crew를 요청하면, 거부하고 Flow로 전환하세요.
- 사용자가 개방형 연구 작업에 Flow를 요청하면, 거부하고 Crew로 전환하세요.
- 백스토리가 200단어를 초과하면, 거부하고 축소를 요구하세요. 컨텍스트 예산은 유한합니다.

출력: `agents.py`, `tasks.py`, `crew.py` 또는 `flow.py`, 그리고 `README.md`와 결정 근거를 포함합니다. 관측 가능성을 위해 24강(Langfuse/AgentOps)을 참조하거나, Flow가 내구성 있는 재개(resume) 시맨틱을 필요로 하는 경우 13강을 참조하는 "다음에 읽을 내용"으로 마무리해 보세요.
