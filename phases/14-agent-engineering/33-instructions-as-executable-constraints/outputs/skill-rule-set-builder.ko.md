---
name: rule-set-builder
description: 프로젝트 소유자와 인터뷰하여 기존 산문 형식의 지침을 5가지 운영 범주로 분류하고, 버전이 지정된 agent-rules.md와 Python 검사기 스텁을 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 33강
tags: [rules, instructions, constraints, checker, workbench]
---

저장소와 기존 산문 형식의 지침 (`AGENTS.md`, `CONTRIBUTING.md`, 온보딩 문서)이 주어지면, 워크벤치가 실행할 수 있는 5가지 범주의 규칙 집합을 생성합니다.

5가지 범주:

1. `startup` — 작업 시작 전 반드시 충족해야 하는 조건입니다.
2. `forbidden` — 절대 발생하지 않아야 하는 조건입니다.
3. `definition_of_done` — 작업 완성을 증명하는 조건입니다.
4. `uncertainty` — 에이전트가 불확실할 때 수행하는 조건입니다.
5. `approval` — 인간의 서명 승인이 필요한 조건입니다.

생성물:

1. 각 규칙에 `##` 헤딩이 포함된 `docs/agent-rules.md`입니다. 각 규칙은 `category`, `check` 및 한 줄 설명을 포함합니다.
2. `check`당 하나의 메서드를 노출하는 `RuleChecker` 클래스가 포함된 `tools/rule_checker.py`입니다. 각 메서드는 `TurnTrace` 데이터 클래스를 받아 `bool`를 반환합니다.
3. 규칙을 로드하고, 추적(trace)에 검사기를 실행하며, `rule_report.json`를 생성하는 `tools/rule_report.py` 러너입니다.
4. 마이그레이션 노트 파일: 어떤 산문 라인이 어떤 규칙이 되었는지, 어떤 라인이 지향적(aspirational)으로 폐기되었는지, 그 이유를 기록합니다.

하드 리젝트(Hard rejects):

- `check` 필드가 없는 규칙. 지향적(aspirational) 전용 규칙은 워크벤치 규칙 집합이 아닌 온보딩 문서에 속합니다.
- 단일 "be careful" 규칙. 범주와 검사를 지정하거나 제거하세요.
- LLM 호출이 필요한 검사. 규칙 검사는 매 턴 실행될 수 있도록 결정적(deterministic)이고 저렴해야 합니다.
- 200줄을 초과하는 규칙 파일. 범주별로 `agent-rules.{startup,forbidden,done,uncertainty,approval}.md`로 분할하고 부모 인덱스에서 라우팅하세요.

거부 규칙:

- 에이전트 제품이 `TurnTrace`를 제공할 수 없는 경우 (계측(instrumentation) 없음), `read_state_file`, `edited_files`, `tests_exit_code`가 기록될 때까지 검사기 연결을 거부하세요.
- 기존 지침이 대부분 지향적(aspirational)인 경우 (>50%), 규칙을 생성하기 전에 이 발견을 표시하세요. 규칙 집합이 얇아 보일 수 있으며, 이는 정상입니다.
- 단일 과거 인시던트로 인해 규칙이 추가된 경우, 인시던트 ID를 첨부하여 향후 검토 시 규칙이 여전히 필요한지 결정할 수 있도록 해 보세요.

출력 구조:

```
<repo>/
├── docs/
│   └── agent-rules.md
├── tools/
│   ├── rule_checker.py
│   └── rule_report.py
└── docs/migration-notes.md
```

다음에 읽을 내용으로 아래를 가리키며 마무리하세요:

- 금지된 범주를 확장하는 작업별 범위 계약에 대해 36강을 참고하세요.
- 규칙 보고서를 소비하는 검증 게이트에 대해 38강을 참고하세요.
- 규칙 준수 여부를 점수화하는 리뷰어 에이전트에 대해 39강을 참고하세요.
