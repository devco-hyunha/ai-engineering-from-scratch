---
name: framework-picker
description: 에이전트 작업에 LangGraph, CrewAI, AutoGen, Agno 또는 순수 Python을 선택하는 도구입니다. 문제의 형태에 맞는 추상화를 매칭하여 선택합니다.
version: 1.0.0
phase: 11단계
lesson: 17강
tags: [langgraph, crewai, autogen, agno, agent-framework, orchestration, decision-matrix]
---

작업 설명(문제 형태, 실행당 총 LLM 호출 수, 분기 패턴, 내구성 및 재개 필요성, 인간 개입 루프(HITL) 체크포인트, 병렬 팬아웃, 세션 메모리, 예상 일일 실행량)가 주어지면 다음을 출력합니다:

1. 형태 매칭. 적합한 추상화를 한 문장으로 지정합니다: 그래프(타입 지정된 상태, 명명된 전환), 조직도(전문가 역할, 관리자 라우팅 핸드오프), 채팅(완료될 때까지 에이전트들이 대화), 도구와 함께 단일 에이전트. 하나를 선택할 수 없다면, 작업은 아직 에이전트 형태가 아닙니다. 중단하고 분해하세요.
2. 분기 권한. 다음 단계를 선택하는 주체는 누구입니까: 개발자(명시적 엣지), 관리자 LLM(CrewAI 계층형), 대화식 생성형(AutoGen GroupChat), 도구 호출 자가 라우팅(Agno). 해당되는 경우 LLM 선택 라우팅의 턴당 토큰 비용을 인용하세요.
3. 상태 예산. 재시작 후 재개, 시간 여행, 또는 인간 인터럽트가 필요한지 확인하세요. 필요하다면, LangGraph는 상태 우선 추상화에서 승리합니다. Agno는 세션 범위 메모리만 지원합니다.
4. 프레임워크 선택. langgraph, crewai, autogen, agno, plain_python 중 하나를 출력하세요. 형태 및 상태 답변을 프레임워크의 핵심 추상화에 매핑하는 한 문장 근거를 포함하세요.
5. 탈출구. 일일 실행량이 10_000을 넘거나, 상태 없이 LLM 호출이 2회 이하인 작업이라면, 대신 공급자 SDK와 함께 순수 Python을 권장하세요. 작업이 작을 때 프레임워크가 없는 것이 가장 빠른 프레임워크입니다.

알려진 DAG를 가진 결정적 워크플로우에 AutoGen을 추천하지 마세요. GroupChatManager는 개발자가 정적으로 연결할 수 있었던 화자 선택에 토큰을 낭비합니다. CrewAI는 `output_pydantic` / `output_json`을 통해 구조화된 작업 출력을 지원하지만([docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks) 참조), `context` 채널은 여전히 다음 작업의 프롬프트 문자열을 통해 흐릅니다. 작업 간에 구조화된 상태를 전달하기 위해 순수 `context`에 의존하는 워크플로우라면 CrewAI를 반대하세요. 두 번 호출하는 요약기에 LangGraph를 반대하세요. StateGraph 오버헤드는 순수한 세금입니다. 4개 이상의 병렬 하위 작업으로 리듀서 시맨틱을 통해 작업이 분산되는 경우 Agno를 반대하세요. Agno는 `Parallel` 블록을 제공하며, 그 출력은 단계 이름으로 키가 지정된 사전에 결합됩니다([docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview) 및 [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps) 참조). 그러나 LangGraph의 Send 스타일 분산 및 리듀스 API에 상응하는 API는 제공하지 않습니다.

예시 입력: "장기 실행 연구 워크플로우: 계획, 세 개의 리트리버로 분산, 합성, 인간이 브리프를 승인, 보고서 작성, 출처 인용. 크래시 후 재개해야 합니다. 하루 50회 실행으로 프로덕션에 바인딩됩니다."

예시 출력:
- 형태: 그래프. 타입이 지정된 계획, 세 개의 병렬 리트리버, 합성과 작성 사이의 명명된 전환.
- 분기: 조건부 엣지를 통해 개발자가 결정합니다. 턴별 관리자 LLM이 없습니다.
- 상태: 재개 및 인간 개입이 필요합니다. LangGraph가 필수입니다.
- 프레임워크: langgraph. State, Send 분산, interrupt_before, PostgresSaver가 모두 일급 객체입니다.
- 탈출구: 해당 없음. 하루 50회 실행은 순수 Python 임계값보다 훨씬 낮으며, 워크플로우가 너무 상태 의존적이어서 프레임워크 없이 방치할 수 없습니다.
