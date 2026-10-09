---
name: migration-agent
description: 결정적 레시피와 에이전트 폴백 루프를 결합하여 MigrationBench를 통과하고 실패 분류 체계를 공개하는 저장소 수준의 코드 마이그레이션 에이전트를 구축합니다.
version: 1.0.0
phase: 19단계
lesson: 09강
tags: [capstone, code-migration, openrewrite, libcst, migrationbench, agent, sandbox]
---

Java 8 또는 Python 2 저장소가 주어지면, 테스트 스위트가 통과하고 커버리지 감소가 최소한인 마이그레이션된 브랜치(Java 17 또는 Python 3.12로)를 생성합니다. 50개 저장소 MigrationBench 하위 집합에서 평가합니다.

구축 계획:

1. 결정적 패스: OpenRewrite (Java) 또는 libcst (Python)가 기계적 재작성을 먼저 실행합니다. 깨끗한 diff와 함께 "레시피" 커밋으로 커밋합니다.
2. Daytona 샌드박스: 대상 런타임이 사전 설치되어 있으며, 브랜치별 빌드 및 읽기 전용 소스 마운트를 사용합니다.
3. 에이전트 루프: Claude Opus 4.7 + GPT-5.4-Codex 위에서 LangGraph 또는 OpenAI Agents SDK를 사용합니다. 도구: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. 실패 유형(dep, syntax, test, build-tool)을 분류하고, 대상이 명확한 수정을 적용한 후 재실행합니다.
4. 예산 상한: 30분, $8, 20턴. 상한을 초과하면 중단하고 현재 diff와 함께 `budget_exhausted`에 파일을 저장합니다.
5. 테스트 + 커버리지 게이트: 빌드가 통과한 후 테스트가 통과해야 하며, 커버리지는 2% 이상 감소하지 않아야 합니다.
6. 레시피 커밋 + 에이전트 커밋 + 요약 댓글과 함께 PR을 엽니다.
7. 실패 분류 체계: `{dep_upgrade_required, build_tool_drift, custom_annotation, test_flake, syntax_edge_case, budget_exhausted, coverage_regression}`에서 저장소별 태그를 지정합니다.
8. MigrationBench 전체 50개 저장소를 실행합니다. 클래스별 통과율, 저장소당 비용, 커버리지 보존율을 공개하고, 결정적 도구 전용 기준선과 비교합니다.

평가 기준표:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | MigrationBench 통과율 | 50개 저장소 하위 집합 pass@1 |
| 20 | 테스트 커버리지 보존 | 기본 브랜치 대비 평균 커버리지 변화 |
| 20 | 마이그레이션된 저장소당 비용 | 통과한 실행의 평균 $/repo |
| 20 | 에이전트 / 결정적 도구 통합 | OpenRewrite가 처리한 수정의 비율 vs 에이전트 |
| 15 | 실패 분석 보고서 | 예시와 함께 분류 체계의 완전성 |

하드 리젝트:

- 결정적 패스를 건너뛰는 파이프라인. OpenRewrite는 기계적 변환을 에이전트보다 70-80% 더 저렴하고 신뢰성 있게 처리합니다.
- 커버리지 감소가 2% 이상인 경우를 통과로 간주하는 것.
- 기계적 변경과 에이전트가 작성한 변경을 하나의 커밋에 묶는 PR. 반드시 분리해야 합니다.
- 동일한 50개 저장소에 대한 결정적 전용 기준선과 일치하지 않는 상태로 통과율을 보고하는 것.

거부 규칙:

- 이동된 브랜치를 베이스 위에 강제 푸시하는 것을 거부합니다. 항상 새 브랜치 + PR을 생성합니다.
- 샌드박스에서 CI가 초록색으로 전환되지 않은 PR을 여는 것을 거부합니다.
- 수정 권한에 대한 명시적 라이선스 없이 기업 저장소에서 실행하는 것을 거부합니다.

출력: 두 계층 이동 파이프라인, 50개 저장소 MigrationBench 실행 로그, 실패 분류 대시보드, 일치하는 결정적 전용 기준선 실행, 그리고 가장 흔한 세 가지 실패 클래스와 각각을 제거할 레시피 변경에 대한 설명이 포함된 저장소입니다.
