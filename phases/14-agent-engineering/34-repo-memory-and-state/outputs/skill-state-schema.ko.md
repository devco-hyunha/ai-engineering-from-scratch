---
name: state-schema
description: 에이전트 상태 및 작업 보드용 프로젝트별 JSON 스키마, 원자적 쓰기(atomic writes)를 지원하는 Python StateManager, 스키마 버전 상향 시 작업대(workbench)가 손상되지 않도록 하는 마이그레이션 스캐폴드를 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 34강
tags: [state, schema, json-schema, atomic-writes, migrations]
---

저장소와 그 안에서 실행되는 에이전트 제품을 고려하여, 작업대용 스키마 우선(state-first) 상태 파일을 생성합니다.

생성 대상:

1. `schemas/agent_state.schema.json`는 필수 키, 허용된 상태 값, 배열 대 null 규칙, `schema_version` 정수 필드를 포함합니다.
2. `schemas/task_board.schema.json`는 작업 ID 패턴, 허용된 소유자, 허용된 상태, 수용 배열을 포함합니다.
3. `tools/state_manager.py`는 `load`, `commit`, `update`를 노출하며, 임시 파일 생성 후 이름 변경(temp-and-rename) 방식의 원자적 쓰기를 사용합니다.
4. `tools/migrate_state.py`는 다음 스키마 버전 상향을 위한 스캐폴드이며, 파일이 알 수 없는 버전에서 온 경우 명확하게 실패(fail-loud)합니다.
5. `agent_state.json`와 `task_board.json`는 `schema_version: 1` 값으로 시드(seed)되며, 새로운 백로그(backlog)가 생성됩니다.

엄격한 거부 조건:

- `schema_version` 필드가 없는 스키마는 거부합니다. 마이그레이션은 선택 사항이 아닙니다.
- 배열이 예상되는 위치에서 `null`를 허용하지 않습니다. `null`는 데이터인 척하는 쓰기 시점 버그입니다.
- 단순 `open(path, "w")`를 사용하는 작성자는 거부합니다. 원자적 쓰기만 허용합니다. 부분적으로 손상된 파일은 신뢰할 수 있는 출처(source of truth)를 훼손합니다.
- 토큰, 원본 채팅 기록, 개인 식별 정보(PII)를 상태에 저장하지 않습니다. 상태는 저장소 관련 사실만 담습니다.

거부 규칙:

- 저장소에 버전 관리가 없다면 상태 파일 출시를 거부합니다. 원자적 쓰기 및 git diff가 내구성(durability)의 핵심입니다.
- `done` 전환을 검증하기 위한 수용 명령이 프로젝트에 하나도 없다면, `status: done` 열거형 값 추가를 거부합니다. 수용 검사 없이 `done`를 추가하는 것은 형식적인 조치에 불과합니다.
- 프로젝트가 잠금(lock) 전략 없이 프로세스 간에 상태를 공유하려는 경우, 출시 전에 해당 문제를 드러내야 합니다. 원자적 이름 변경은 필요하지만 충분하지 않습니다.

출력 구조:

```
<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py
```

다음에 읽을 내용으로 아래를 가리키며 마무리합니다:

- 35강: 시작 시 매니저를 호출하는 초기화 스크립트.
- 38강: 상태를 읽어 완료를 점수화하는 검증 게이트.
- 동일한 스키마를 사용하는 핸드오프 생성기의 40강입니다.
