---
name: groupchat-selector
description: AutoGen/AG2 스타일 GroupChat 선택기를 구성하여 선택기 변형, 종료 규칙, 그리고 핫 스피커 방지 규칙을 지정합니다.
version: 1.0.0
phase: 16단계
lesson: 10강
tags: [multi-agent, groupchat, autogen, ag2, speaker-selection]
---

작업과 에이전트 명단을 고려하여 GroupChat 구성을 생성합니다: 선택기 선택, 선택기 입력, 종료 규칙 및 가드레일(Guardrails)입니다.

생성 항목:

1. **선택기 변형.** 라운드 로빈(Round-robin) (저비용, 공정, 컨텍스트 비의존), LLM 선택 (컨텍스트 인식, 고비용), 또는 커스텀 (LLM + 규칙 기반 폴백(Fallback)).
2. **선택기 입력.** LLM 선택인 경우: 최근 N개 메시지, 에이전트 전문 분야, 턴 수. 커스텀인 경우: 명시적 규칙.
3. **종료 규칙.** 최대 라운드 수, TERMINATE 토큰, 목표 달성 검증기, 또는 조합.
4. **핫 스피커(Hot-speaker) 완화.** 에이전트별 턴 상한, 선택기 입력에 스피커 균형 점수 포함, K개 연속 턴 후 강제 로테이션.
5. **컨텍스트 팽창(Context bloat) 완화.** 투영 계획(역할별 범위 지정 뷰), 요약 체크포인트(Checkpoint), 에이전트별 컨텍스트 상한.
6. **관측 가능성(Observability).** 선택기 입력, 선택기 선택, 에이전트별 턴 지연을 로깅(Logging)합니다.

하드 리젝트(Hard rejects):

- 선택기 입력/출력 로깅이 없는 LLM 선택 구성은 허용되지 않습니다. 디버깅이 불가능해집니다.
- max_rounds 상한이 없는 구성은 허용되지 않습니다.
- 추론 작업에 대한 대칭 채팅(전문화 없음)은 허용되지 않습니다. 대신 디베이트(Debate)(10강)를 사용하세요.

거절 규칙:

- 작업에 알려진 DAG 구조가 있는 경우, GroupChat을 거절하고 결정성을 위해 LangGraph 정적 그래프를 권장합니다.
- 작업이 엄격한 감사 추적(Audit trails)을 요구하는 경우, GroupChat을 거절하고 체크포인터(Checkpointer)가 있는 LangGraph를 권장합니다.
- 에이전트 수가 5-6개를 초과하는 경우, 평면 GroupChat을 거절하고 중첩 그룹 또는 계층적 패턴을 권장합니다.

출력: 한 페이지 GroupChat 구성 브리프(Brief)입니다. 비용 추정(LLM 선택은 턴당 선택기 호출 한 번이 발생합니다)으로 마무리하세요.
