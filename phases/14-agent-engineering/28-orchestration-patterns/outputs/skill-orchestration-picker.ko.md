---
name: orchestration-picker
description: 주어진 문제에 대해 오케스트레이션 토폴로지(supervisor, swarm, hierarchical, debate, none 중 하나)를 선택하고 최소한으로 구현합니다.
version: 1.0.0
phase: 14단계
lesson: 28강
tags: [orchestration, supervisor, swarm, hierarchical, debate]
---

제품 도메인과 작업 클래스가 주어지면, 최소한의 토폴로지를 선택해 보세요.

결정 사항:

1. 1개의 에이전트 + 워크플로우 패턴(12강)으로 충분합니까? -> 토폴로지를 사용하지 마세요.
2. 명확한 책임이 있는 2~4명의 전문가가 필요합니까? -> **supervisor-worker**를 선택하세요.
3. 레이턴시가 중요하고 전문가 간에 깔끔한 핸드오프가 가능합니까? -> **swarm**을 선택하세요.
4. 10명 이상의 전문가가 필요하고 supervisor의 컨텍스트 예산이 부족합니까? -> **hierarchical**을 선택하세요.
5. 비용보다 정확성이 중요하고, 다중 제안자 + 비판이 도움이 됩니까? -> **debate**(25강)를 선택하세요.

생성물:

1. 선택된 토폴로지 스캐폴드.
2. swarm의 hop counter; hierarchical의 중첩 깊이 제한; debate의 라운드 상한.
3. 각 핸드오프 또는 각 단계별 관측 가능성 훅(OTel GenAI spans, 23강).
4. "왜 이것을 선택했고, 저것은 선택하지 않았는가"에 대한 README 섹션.

엄격한 거부 조건:

- 순차적인 3번의 LLM 호출을 "multi-agent"라고 부르는 것. 이는 프롬프트 체인입니다.
- hop counter가 없는 swarm. 바운싱(bouncing)은 필연적입니다.
- 각 브랜치가 1명의 전문가로 끝나는 hierarchical. 평탄화(flatten)하세요.

거부 규칙:

- 단일 ReAct 루프로 처리할 수 있는 작업에 대해 사용자가 multi-agent를 원한다면, 거부하고 01강을 제안하세요.
- 2단계 작업에 대해 사용자가 supervisor를 원한다면, 거부하고 프롬프트 체이닝(12강)을 제안하세요.
- 도메인에 컴플라이언스/감사 요건이 있는 경우, swarm을 거부하고 supervisor 또는 hierarchical을 제안하세요.

출력: 토폴로지 스캐폴드 + 결정 근거가 포함된 README. supervisor 구현을 위한 13강(LangGraph), 핸드오프를 도구로 사용하는 16강(OpenAI Agents SDK), debate 세부 사항을 위한 25강을 가리키는 "다음에 읽을 내용"으로 마무리하세요.
