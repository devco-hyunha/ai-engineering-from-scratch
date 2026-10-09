# AGENTS.md

에이전트 워크벤치로 실행되는 저장소 내에서 작업하고 있습니다.

작업하기 전에 다음을 읽어 보세요:

1. `agent_state.json` — 마지막 세션이 중단된 위치.
2. `task_board.json` — 진행 중인 작업과 다음 단계.
3. `docs/agent-rules.md` — 시작, 금지 사항, 완료, 불확실성, 승인.
4. `docs/reliability-policy.md` — 이 워크벤치가 흡수하도록 설계된 실패 모드.
5. `docs/handoff-protocol.md` — 세션 종료 시 반드시 산출해야 하는 결과.
6. `docs/reviewer-rubric.md` — 완료된 작업이 평가되는 기준.

검증 명령: 보드의 활성 작업에서 `acceptance_criteria`을 참조하세요.

팩 버전: 1.0.0
