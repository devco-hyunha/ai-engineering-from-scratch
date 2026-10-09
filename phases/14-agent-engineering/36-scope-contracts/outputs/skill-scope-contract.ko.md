---
name: scope-contract
description: 허용/금지 글롭, 수용 기준, 롤백 플랜을 포함하는 작업별 범위 계약을 생성하고, 모든 에이전트 diff에 대해 실행되는 CI 준비 글롭 인식 체크어를 포함합니다.
version: 1.0.0
phase: 14단계
lesson: 36강
tags: [scope, contract, globs, diff-check, ci]
---

작업 설명과 저장소 레이아웃을 바탕으로 범위 계약과 diff 인식 체크어를 생성해 보세요.

생성 대상:

1. 작업에 대한 `scope_contract.json`로, 필드는 `task_id`, `goal`, `allowed_files` (글롭), `forbidden_files` (글롭), `acceptance_criteria`, `rollback_plan`, `approvals_required`입니다.
2. 계약 경로와 수정된 파일 목록을 받아 `ScopeReport`를 반환하고, 위반이 있을 경우 비-0 종료 코드를 반환하는 `tools/scope_check.py`입니다.
3. 병합 diff에 대해 체크어를 실행하는 CI 단계 (`.github/workflows/scope-check.yml` 또는 동등한 것).
4. 계약이 변경 이력과 함께 출시되도록 하는 `outputs/scope/closed/<task_id>.json` 아카이빙 관례.

하드 리젝트:

- `forbidden_files`이 없는 계약. 음의 공간은 계약의 일부입니다.
- 코드 디렉터리에 대해 글롭 대신 원시 경로를 나열하는 계약. 리팩토링은 원시 경로를 밤새 무효화합니다.
- 비어 있거나 "runbook 참조"인 `rollback_plan` 필드. 명확히 명시하세요.
- "사례별"로 나열된 승인. 승인 경계는 열거 가능해야 합니다.

거부 규칙:

- 작업 설명이 저장소의 특정 영역을 제한하지 않으면, 설명만으로 `allowed_files`를 작성하는 것을 거부하세요. 작업이 위치한 디렉터리를 요청하세요.
- 저장소에 테스트 명령이 없으면, 테스트 명령이 제공되거나 스텁될 때까지 `acceptance_criteria`를 추가하는 것을 거부하세요. 검증할 수 없는 계약은 소원에 불과합니다.
- 에이전트 런타임이 승인 경계를 준수할 수 없다면 (인간 개입 루프(Human-in-the-Loop (HITL))가 없음), 출시 전에 공백을 드러내세요. 승인 필요 작업으로의 범위 확대가 지배적인 실패가 될 것입니다.

출력 구조:

```
<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml
```

다음에 읽을 내용으로 끝맺으며, 다음을 가리키세요:

- 실행된 명령을 계약에 연결하는 런타임 피드백에 대한 37강.
- 범위 보고서를 소비하는 검증 게이트(Verification Gate)에 대한 38강.
- 닫힌 계약 아카이브를 감사하는 리뷰어 에이전트(Reviewer Agent)에 대한 39강.
