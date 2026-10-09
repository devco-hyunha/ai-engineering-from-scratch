---
name: workbench-pack
description: 팀의 히스토리에 맞춰 규칙을 다듬고, 저장소에 맞는 스코프 글롭을 매칭하며, 도메인 특화 항목을 추가한 루브릭 차원을 확장하여 프로젝트에 맞춘 에이전트 워크벤치 팩을 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 42강
tags: [capstone, workbench-pack, installer, schemas, drop-in]
---

저장소, 팀의 인시던트 히스토리, 그리고 그 안에서 실행되는 에이전트 제품이 주어지면, 튜닝된 에이전트 워크벤치 팩과 설치 프로그램을 생성합니다.

다음 내용을 생성합니다:

1. AGENTS.md, docs/, schemas/, scripts/, bin/, README.md, VERSION을 포함하는 표준 레이아웃과 일치하는 `agent-workbench-pack/` 디렉토리입니다.
2. `--force` 없이 기존 팩을 덮어쓰는 것을 거부하며, `.workbench-version`를 대상 저장소에 기록하는 `bin/install.sh`입니다.
3. 팀의 최근 6개 인시던트에서 파생된 각 카테고리별 최소 한 개의 규칙을 포함하는 `agent-rules.md`, 여섯 번째 도메인 차원을 포함하는 `reviewer-rubric.md`, 프로젝트 특화 글롭을 포함하는 `scope_contract.schema.json`의 프로젝트 튜닝 버전입니다.
4. 스크립트와 스키마 간, 또는 VERSION과 스키마의 `schema_version` 간에 드리프트(drift)가 발생하면 실패하는 `lint_pack.py` 스크립트입니다.
5. 데모 브랜치에 팩을 설치하고 알려진 정상 작업에 대해 검증 게이트를 실행하는 선택적 CI 통합입니다.

하드 리젝트(Hard rejects):

- 프로젝트 특화 작업이 포함된 팩은 거부합니다. 작업은 대상 저장소의 보드에 존재합니다.
- 단일 벤더 SDK에 결합된 팩은 거부합니다. 프레임워크에 독립적이어야 하며, SDK 연결은 대상 저장소의 몫입니다.
- 상태 파일을 변경하는 설치 프로그램은 거부합니다. 설치 프로그램은 멱등적이며 표면(surface)만 다루며, 상태는 에이전트와 인간이 관리합니다.
- 해당 체크 함수가 없는 규칙은 거부합니다. 지향적인 규칙은 온보딩에 속하며 팩에 속하지 않습니다.

거부 규칙:

- 인시던트 히스토리가 비어 있으면 튜닝된 `agent-rules.md`를 출시하는 것을 거부합니다. 표준 기본값을 사용하고 공백을 드러내세요.
- 대상 저장소의 CI가 설치와 호환되지 않으면(`.github/workflows/` 없음, 등가물 없음), 선택적 CI 단계를 거부하고 수동 경로를 문서화하세요.
- 팀이 팩의 비공개 포크를 사용하는 경우, 공개 설치 프로그램을 작성하는 것을 거부합니다. 비공개 설치 프로그램은 비공개 불변 조건을 포함합니다.

출력 구조:

```
agent-workbench-pack/
├── AGENTS.md
├── docs/
├── schemas/
├── scripts/
├── bin/install.sh
├── lint_pack.py
├── VERSION
└── README.md
```

다음에 읽을 내용으로 다음을 가리키며 마무리하세요:

- 이 팩이 개선하는 before/after 벤치마크에 대한 41강
- 팩의 판정 결과를 소비하는 평가 루프에 대한 30강 (평가 주도 에이전트 개발)
- [SkillKit](https://github.com/rohitg00/skillkit)로 팩을 32개 AI 에이전트에 분배
