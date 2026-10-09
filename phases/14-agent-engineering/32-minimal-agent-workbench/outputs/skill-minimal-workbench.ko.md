---
name: minimal-workbench
description: 모든 저장소에 최소한의 실행 가능한 에이전트 작업대를 구축합니다. 짧은 AGENTS.md 라우터, 내구성 있는 agent_state.json, 그리고 프로젝트의 현재 백로그에 연결된 JSON task_board.json을 포함합니다.
version: 1.0.0
phase: 14단계
lesson: 32강
tags: [workbench, agents-md, state, task-board, scaffold]
---

저장소 경로와 짧은 백로그가 주어지면, 최소한의 실행 가능한 에이전트 작업대를 스캐폴딩해 보세요.

다음 내용을 생성합니다:

1. `AGENTS.md`는 80줄을 넘지 않아야 합니다. 이 파일은 상태 파일, 작업 보드, 더 깊은 규칙 문서(비어 있더라도), 그리고 검증 명령으로 라우팅되어야 합니다. 이 파일에는 산문 형식의 튜토리얼이 없어야 합니다.
2. `agent_state.json`는 다음 키를 포함합니다: `active_task_id`, `touched_files`, `assumptions`, `blockers`, `next_action`. 모든 선택적 필드는 빈 배열이나 빈 문자열로 기본 설정되며, 배열에 `null`를 사용하지 마세요.
3. `task_board.json`는 작업의 JSON 배열로 생성합니다. 각 작업은 `id`, `goal`, `owner` (`builder` | `reviewer` | `human`), `acceptance` (문자열 목록), `status` (`todo` | `in_progress` | `done` | `blocked`)를 포함합니다.
4. `docs/agent-rules.md`는 각 표면(surface)에 단일 H2를 포함하는 자리표시자로 생성하여, 이후 강의에서 이를 채울 수 있도록 합니다.

하드 리젝트(Hard rejects):

- `AGENTS.md`가 80줄을 넘거나 10줄 미만인 경우. 너무 길면 에이전트가 이를 건너뛰고, 너무 짧으면 라우팅 정보를 담지 못합니다.
- 저장소 대신 채팅 기록을 참조하는 상태 파일. 저장소가 시스템 오브 레코드(system of record)입니다.
- `acceptance`가 없는 작업 보드. 수용 기준이 없는 작업은 "looks good" 도장 찍기(rubber stamp)가 됩니다.
- `owner`가 `agent` 또는 `model`인 작업. 소유자는 엔티티가 아니라 역할입니다.

거부 규칙:

- 저장소에 검증 명령이 없다면, 검증 명령이 제공되거나 스텁(stub)될 때까지 `AGENTS.md`를 작성하는 것을 거부합니다. 존재하지 않는 게이트를 가리키는 라우터는 라우터가 없는 것보다 더 나쁩니다.
- 백로그에 열려 있는 작업이 12개 이상이라면, 거부하고 사용자에게 백로그를 분할하도록 요청합니다. 화면을 넘치는 보드는 계획 연극(planning theater)으로 변질됩니다.
- 추적되는 파일에 시크릿(secrets)이 포함된 프로젝트라면, 상태 파일 작성을 거부하고 시크릿 유출을 먼저 차단 요인으로 표출합니다.

출력 구조:

```
<repo>/
├── AGENTS.md
├── agent_state.json
├── task_board.json
└── docs/
    └── agent-rules.md
```

"다음에 읽을 내용"으로 다음을 가리키며 마무리합니다:

- 규칙 자리 표시자를 실행 가능한 제약 조건으로 변환하는 33강입니다.
- 내구성 있는 상태 스키마에 대한 34강입니다.
- 작업별 범위 계약에 대한 36강입니다.
