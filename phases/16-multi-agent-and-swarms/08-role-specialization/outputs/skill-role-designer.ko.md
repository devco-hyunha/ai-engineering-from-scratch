---
name: role-designer
description: 주어진 작업에 대해 다중 에이전트 시스템의 역할 목록을 생성합니다. planner/executor/critic/verifier를 명시적인 I/O 스키마와 함께 지정합니다.
version: 1.0.0
phase: 16단계
lesson: 08강
tags: [multi-agent, role-specialization, metagpt, chatdev, verification]
---

주어진 작업에 대해 I/O 스키마와 결정적 검증자(deterministic verifier)를 포함한 전문화된 역할 목록을 생성합니다. CrewAI, LangGraph, AutoGen 또는 커스텀 루프에 매핑할 준비가 되어 있습니다.

생성할 항목:

1. **역할 목록.** 3-5개의 역할. 각각의 이름을 지정합니다. 최소한 planner, executor, verifier가 포함되어야 합니다. Critic은 선택 사항입니다.
2. **역할별 I/O 스키마.** 각 역할에 대해: 상류 역할(upstream role)로부터 소비하는 것과 생성하는 것(산문(prose)이 아닌 스키마)을 명시합니다. dataclass 스타일 표기를 사용하세요.
3. **검증자 사양.** 결정적 검사(deterministic check)의 이름을 지정합니다: 테스트 스위트, 타입 체커, 스키마 검증자, 린터. 통과/실패 기준을 설명합니다.
4. **비평가(Critic) 사양 (선택 사항).** 포함하는 경우, 주관적 품질을 판단하는 기준을 지정합니다. "좋은 코드" 같은 모호한 표현이 아닌, 구체적인 체크리스트를 사용하세요.
5. **소통 기반 환각 방지 규칙.** 세부 정보가 누락되었을 때, 하류 역할(downstream role)이 상류 역할에 발명(invent)하지 않도록 허용되는 질문을 지정합니다.
6. **수정 루프 예산.** 인간에게 에스컬레이션하기 전의 최대 라운드 수. 기본값은 2입니다.
7. **프레임워크 매핑.** 한 줄씩: CrewAI, LangGraph, AutoGen에서 이 역할 목록을 표현하는 방법.

하드 리젝트(Hard rejects):

- 결정적 검증자(deterministic verifier)가 없는 역할 목록은 거부됩니다. 모든 LLM 기반 역할 목록은 MAST 검사를 통과하지 못합니다.
- 모호한 I/O ("executor가 output를 반환한다"). 항상 스키마를 명시하세요.
- Critic과 verifier가 혼동되는 경우. 이들은 서로 다른 버그를 잡습니다. 둘 다 필요하면 둘 다 존재해야 합니다.

거부 규칙:

- 작업에 결정적 정확성 검사(deterministic correctness check)가 없는 경우 (순수 생성 작업, 창의적 글쓰기), 거부하고 인간 리뷰어 루프나 다중 에이전트 토론(07강)을 권장하세요.
- 작업이 3개 이상의 역할에 비해 너무 작은 경우 (인간 작업 시간 10분 미만), 거부하고 단일 에이전트를 권장하세요.

출력: 한 페이지 분량의 역할 설계 브리프. MAST 실패-갭(MAST failure-gap) 체크로 마무리하세요: 최소한 하나의 결정적 검증자(deterministic verifier)가 존재함을 확인합니다.
