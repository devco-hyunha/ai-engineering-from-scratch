---
name: hybrid-planner
description: 하이브리드 플래너를 구축하세요 — 증명 가능한 계획에 ChatHTN을, 기계 검증 가능한 평가자를 사용하는 코드 검색에 AlphaEvolve를 사용하며, 문제에 맞는 플래너를 선택합니다.
version: 1.0.0
phase: 14단계
lesson: 11강
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

문제 유형(정책 기반 워크플로 vs 코드 최적화 vs 개방형 작업)에 따라 플래너를 선택하고 올바른 스캐폴드를 생성하세요.

결정 사항:

1. 문제가 하드 사전 조건 / 정책 / 스케줄링 제약이 있습니까? -> HTN (ChatHTN).
2. 문제가 결정론적이고 기계 검증 가능한 적합도 함수가 있습니까? -> 진화적 (AlphaEvolve).
3. 둘 다 해당하지 않습니까? -> ReAct (01강) 또는 ReWOO (02강)를 대신 사용하세요.

HTN의 경우 다음을 생성하세요:

1. `Operator` 유형에 `preconditions`, `effects_add`, `effects_remove`가 있습니다.
2. `Method` 유형에 `task`, `preconditions`, `subtasks`가 있습니다.
3. 먼저 메서드를 시도하고, LLM 분해로 폴백하며, 성공한 LLM 분해를 캐시하는 플래너.
4. 알 수 없는 연산자나 메서드를 참조하는 LLM 분해를 거부하는 검증 단계.

진화적 방식의 경우 다음을 생성하세요:

1. 후보 프로그램의 시드 개체군.
2. 스칼라 적합도를 반환하는 결정론적 평가자.
3. 변이 연산자(LLM 주도 또는 규칙 기반).
4. 상위 k개를 유지하고, 변이하고, 반복하는 선택 루프(조기 종료 포함).

거부 조건:

- 연산자 스키마 검증 없이 LLM 출력을 직접 적용하는 ChatHTN. 건전성 주장이 실패합니다.
- 평가자가 LLM 판정자를 호출하는 AlphaEvolve. 적합도는 결정론적이어야 하며, LLM 판정자는 루프가 복구할 수 없는 확률적 노이즈를 도입합니다.
- 개방형 작업("블로그 포스트 작성")에这两种 패턴을 사용하는 것. 평가자 없음, 사전 조건 없음 -> ReAct를 사용하세요.

거부 규칙:

- 도메인에 명확한 연산자 스키마가 없으면 ChatHTN을 거부하세요. ReWOO 또는 순수 ReAct를 제안하세요.
- 도메인에 기계 검증 가능한 적합도가 없으면 AlphaEvolve를 거부하세요. Self-Refine (05강)을 제안하세요.
- 사용자가 "planner + LLM이 최종 결정을 내린다"를 원한다면 거부하세요. 기호적 정확성과 LLM 탐색의 분리는 핵심적인 요소입니다.

출력: `operators.py`, `methods.py`, `planner.py` (HTN) 또는 `evaluator.py`, `mutator.py`, `loop.py` (진화적), 그리고 결정 근거가 담긴 `README.md`. 끝부분에 "다음에 읽을 내용"을 포함하여, 토론식 검증이 문제에 적합하다면 25강을, 과제가 실제로 ReWOO 형태라면 02강을 가리키세요.
