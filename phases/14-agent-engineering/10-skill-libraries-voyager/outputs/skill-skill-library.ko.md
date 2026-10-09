---
name: skill-library
description: 등록, 유사성 기반 검색, 합성 실행, 실패 기반 정제를 포함한 Voyager형 스킬 라이브러리를 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 10강
tags: [voyager, skills, library, composition, refinement]
---

목표 런타임과 도메인이 주어지면, Voyager의 세 가지 구성 요소인 커리큘럼 훅, 검색 가능한 스킬 저장소, 반복적 정제를 지원하는 스킬 라이브러리를 생성합니다.

생성 대상:

1. `Skill` 타입에 `name`, `description`, `code`, `version`, `tags`, `depends_on`, `history`를 포함합니다. 모든 쓰기 작업은 이전 코드를 기록합니다.
2. `SkillLibrary`는 `register(skill, dedup=True)` (신규 생성 또는 버전 상향), `search(query, top_k, tag_filter)`, `get(name)`, `topo_order(name)` (의존성 해결), `execute(name, context)` (위상 정렬 실행)와 함께 사용됩니다.
3. 검색은 전체 라이브러리에 대한 LLM 점수 매기기 대신 임베딩 유사성이나 BM25를 사용해야 합니다. 상위 k개 후보(shortlist)에 대한 LLM 재순위는 허용됩니다.
4. 실행은 스킬별로 예외를 잡아야 하며, 정제 루프가 소비할 수 있는 피드백으로 추적(trace)에 이를 표시해야 합니다.
5. 정제 훅: `execute` 실패 후, 런타임은 (task, skill_name, error, env_state)를 수집하여 모델에 전달하고, 재작성된 스킬에 `register`를 호출합니다. 버전이 상향되며, 히스토리는 이전 코드를 보존합니다.

허용되지 않는 사항:

- 스킬이 코드가 아닌 산문(string of prose)인 라이브러리. 스킬은 실행 가능해야 합니다. 산문은 `description`에 속합니다.
- 위상 정렬(topological sort) 없는 합성. 사이클 감지 없는 깊이 우선 탐색은 스킬 DAG에서 실패합니다.
- 침묵하는 버전 덮어쓰기. 모든 정제는 `version`의 버전을 상향하고, 감사(audit)를 위해 이전 코드를 `history`에 푸시해야 합니다.

거절 규칙:

- 목표 런타임에 스킬 실행용 샌드박스가 없다면, 스킬이 프로덕션 시스템을 건드리는 도메인에 대해 거절합니다. 출시(ship) 전에 샌드박스(10강 원칙)를 요구합니다.
- 사용자가 "정제 없이 모든 실패에 대해 자동 재시도"를 요청하면 거절합니다. 정제 없는 재시도는 버그를 증폭시킬 뿐, 수정하지 못합니다.
- 라이브러리가 평면 검색(flat retrieval)으로 약 200개 이상의 스킬을 초과하면 "프로덕션 준비 완료"라고 부르는 것을 거절합니다. 먼저 태그 필터와 계층적 네임스페이스를 추가합니다.

출력: `skill.py`, `library.py`, `execute.py`, `refine.py`, 그리고 중복 제거 규칙, 검색 백엔드, 정제 프롬프트, 버전 정책을 설명하는 `README.md`. "다음에 읽을 내용"으로 Claude Agent SDK 통합을 위한 17강, OpenAI Agents SDK 도구 번역을 위한 16강, 스킬 라이브러리 품질 평가를 위한 30강을 가리키며 마무리하세요.
