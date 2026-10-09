# 캡스톤 09 — 코드 마이그레이션 에이전트 (저장소 단위 언어 / 런타임 업그레이드)

> Amazon의 MigrationBench (Java 8에서 17로)와 Google의 App Engine Py2-to-Py3 마이그레이터가 2026년의 기준을 설정했습니다. Moderne의 OpenRewrite는 대규모로 결정적인 AST 재작성을 수행합니다. Grit은 codemod 스타일 DSL로 동일한 문제를 목표로 합니다. 생산 패턴은 두 가지를 결합합니다: 안전한 재작성을 위한 결정적 기반, 모호한 케이스를 위한 에이전트 레이어, 브랜치별 빌드를 위한 샌드박스, 그리고 PR이 열리기 전에 테스트가 통과되도록 하는 테스트 하네스. 캡스톤은 50개의 실제 저장소를 마이그레이션하고 통과율과 실패 분류 체계를 공개하는 것입니다.

**유형:** Capstone
**언어:** Python (에이전트), Java / Python (대상), TypeScript (대시보드)
**선수 요건:** 5단계 (NLP), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 13단계 (도구), 14단계 (에이전트), 15단계 (자율), 17단계 (인프라)

**활용 단계:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**시간:** 30시간

## 문제점

대규모 코드 마이그레이션은 2026년 코딩 에이전트의 가장 명확한 생산 애플리케이션 중 하나입니다. 정답은 명확합니다 (마이그레이션 후 테스트 스위트가 통과하는가?), 보상은 실재합니다 (Java-8 함대 마이그레이션은 인원 규모 프로젝트입니다), 그리고 벤치마크는 공개되어 있습니다 (MigrationBench 50개 저장소 하위 집합). Moderne의 OpenRewrite는 결정적인 부분을 처리합니다. 에이전트 레이어는 OpenRewrite 레시피가 처리할 수 없는 모든 것을 처리합니다: 모호한 재작성, 빌드 시스템 드리프트, 롱테일 구문, 전이적 의존성 파손.

Java 8 저장소 (또는 Python 2 저장소)를 받아 CI가 통과하는 마이그레이션된 브랜치를 생성하는 에이전트를 구축해 보세요. 통과율, 테스트 커버리지 보존, 저장소당 비용을 측정하고 실패 분류 체계를 구축합니다. 결정적 전용 기준선과의 나란히 비교는 에이전트의 가치가 실제로 어디에 있는지를 알려줍니다.

## 개념

파이프라인은 두 계층으로 구성됩니다. **결정적 기반**(Java용 OpenRewrite, Python용 libcst)은 임포트, 메서드 시그니처, null 안전성 편집, try-with-resources, 폐기된 API 교체 등 대부분의 기계적 재작성을 안전하게 수행합니다. 이는 빠르고 감사 가능한 diff를 생성합니다. **에이전트 계층**(Claude Opus 4.7 및 GPT-5.4-Codex 기반의 OpenAI Agents SDK 또는 LangGraph)은 레시피가 처리할 수 없는 경우를 담당합니다: 빌드 파일 업그레이드(Maven/Gradle/pyproject), 전이 의존성 충돌, 테스트 불안정성(flakes), 사용자 정의 어노테이션.

각 저장소에는 대상 런타임이 사전 설치된 Daytona 샌드박스가 할당됩니다. 에이전트는 반복적으로 빌드 실행, 실패 분류, 수정 적용, 재실행을 수행합니다. 하드 제한: 저장소당 30분, 저장소당 $8, 에이전트 턴 20회. 모든 테스트가 통과하고 커버리지 델타가 음수가 아닌 경우, 브랜치는 PR을 엽니다. 그렇지 않으면, 저장소는 증거와 함께 실패 클래스로 분류됩니다.

실패 분류 체계가 산출물입니다. 50개 저장소에서 무엇이 깨졌습니까? 전이 의존성? 사용자 정의 어노테이션? 빌드 도구 버전? 마이그레이션과 무관한 테스트 flakes? 각 클래스는 개수와 예시 diff를 가집니다. 미래의 레시피 작성자는 상위 세 가지를 목표로 할 수 있습니다.

## 아키텍처

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 스택

- 결정적 기반: OpenRewrite (Java) 또는 libcst (Python)
- 에이전트: Claude Opus 4.7 + GPT-5.4-Codex 기반의 OpenAI Agents SDK 또는 LangGraph
- 샌드박스: 브랜치별 Daytona devcontainer, 대상 런타임(Java 17 / Python 3.12) 사전 설치
- 빌드 시스템: Maven, Gradle, uv (Python)
- 벤치마크: Amazon MigrationBench 50개 저장소 하위 집합(Java 8에서 17로), Google App Engine Py2-to-Py3 저장소
- 테스트 하네스: 병렬 실행기, Jacoco (Java) 또는 coverage.py (Python)를 통한 커버리지
- 관측 가능성: Langfuse + 모든 diff 청크가 포함된 저장소별 추적 번들
- 대시보드: 클래스별 개수와 예시 diff가 포함된 실패 분류 대시보드

```figure
ce-migration-funnel
```

## 구현하기

1. **레시피 패스.** 먼저 OpenRewrite (Java) 또는 libcst (Python) 레시피를 실행합니다. 마이그레이션의 70-80%를 차지하는 기계적 부분을 처리합니다. "recipe" 커밋으로 커밋하세요.

2. **빌드 시연.** Daytona 샌드박스: 대상 런타임을 설치하고 빌드를 실행합니다. 성공(green)하면 테스트로 건너뛰세요. 실패(red)하면 에이전트에게 전달하세요.

3. **에이전트 루프.** 도구와 함께 LangGraph 사용: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. 에이전트가 실패 유형(dep, syntax, test, build-tool)을 분류하고 targeted fix를 적용합니다. 재실행하세요.

4. **예산 상한.** 저장소당 30분 벽시계 시간, $8 비용, 20 에이전트 턴. 상한을 초과하면 중단하고 "budget_exhausted" 아래에 현재 diff를 파일로 남깁니다.

5. **테스트 + 커버리지 게이트.** 빌드가 green 상태가 되면 테스트 스위트를 실행하세요. base 저장소와 커버리지를 비교합니다. 커버리지가 2% 이상 감소하면 "coverage_regression" 아래에 파일로 남깁니다.

6. **PR 열기.** 성공 시 브랜치를 push하고, diff와 적용된 레시피 및 에이전트가 작성한 커밋에 대한 요약과 함께 PR을 여세요.

7. **실패 분류 체계.** 각 실패한 저장소에 클래스 태그를 붙이세요: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. 대시보드를 구축하세요.

8. **50개 저장소 실행.** MigrationBench 하위 집합 전체에서 실행하세요. 클래스별 통과율, 저장소당 비용, 커버리지 보존, deterministic-only baseline과의 비교를 보고하세요.

## 사용하기

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 출시하기

`outputs/skill-migration-agent.md`는 산출물입니다. 저장소가 주어지면 deterministic 레시피를 실행한 후 에이전트 루프를 통해 green 상태의 migrated branch를 생성하거나, 저장소를 분류 체계 클래스 아래에 파일로 남깁니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | MigrationBench 통과율 | 50개 저장소 하위 집합 pass@1 |
| 20 | 테스트 커버리지 보존 | base 대비 평균 커버리지 델타 |
| 20 | 저장소당 비용 | 통과한 실행에서의 $/repo |
| 20 | 에이전트 / deterministic 도구 통합 | OpenRewrite가 처리한 fix 비율 vs 에이전트가 작성한 fix |
| 15 | 실패 분석 보고서 | 예시와 함께 분류 체계의 완전성 |
| **100** | | |

## 연습 문제

1. OpenRewrite만 사용하여 migrate 파이프라인을 실행하세요(에이전트 없음). 전체 파이프라인과 통과율을 비교하세요. 에이전트가 유일한 차이인 경우를 식별하세요.

2. "lint-clean" 체크를 구현하세요: 마이그레이션 후 스타일 린터(Java의 경우 spotless, Python의 경우 ruff)를 실행하세요. 새로운 lint 오류가 발생하면 PR을 실패 처리하세요. 커버리지가 보존되었지만 스타일이 후퇴한 비율을 측정하세요.

3. "최소 diff" 옵티마이저를 추가하세요: 에이전트의 브랜치가 테스트를 통과한 후, 두 번째 패스로 불필요한 변경을 정리합니다. diff 크기 감소량을 보고하세요.

4. 세 번째 마이그레이션으로 확장하세요: Node 18에서 Node 22로. 샌드박스 래핑을 재사용하고, 레시피 레이어를 커스텀 codemod로 교체하세요.

5. UX 지표로서 첫 그린 빌드까지의 시간(TTFGB)을 측정하세요. 목표: p50가 10분 미만이어야 합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 결정적 기반(Deterministic substrate) | "레시피 엔진" | OpenRewrite / libcst: 안전 보장이 있는 선언적 AST 재작성 |
| Codemod | "코드 수정 프로그램" | 소스 코드를 기계적으로 변경하는 재작성 규칙 |
| 빌드 드리프트(Build drift) | "도구 버전 편차" | 주요 버전 간 Maven / Gradle / uv의 미묘한 동작 변화 |
| 실패 분류(Failure class) | "분류 버킷" | 저장소가 마이그레이션되지 않은 레이블 지정된 이유: 의존성, 구문, 테스트, 빌드 도구, 예산 |
| 커버리지 델타(Coverage delta) | "커버리지 보존" | 기본 브랜치에서 마이그레이션된 브랜치로 이동할 때 테스트 커버리지 %의 변화 |
| 에이전트 턴(Agent turn) | "도구 호출 라운드" | 에이전트 루프에서의 하나의 계획 -> 실행 -> 관찰 사이클 |
| 예산 소진(Budget exhaustion) | "상한 도달" | 저장소가 30분 / $8 / 20턴 한도를 소비했으나 통과하지 못함 |

## 추가 읽기

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026년 표준 벤치마크
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 결정적 기반(Deterministic substrate) 참조
- [OpenRewrite documentation](https://docs.openrewrite.org) — 레시피 작성
- [Grit.io](https://www.grit.io) — 대체 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 참조
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 대체 마이그레이션 벤치마크
- [libcst](https://github.com/Instagram/LibCST) — Python 결정적 기반(Deterministic substrate)
- [Daytona sandboxes](https://daytona.io) — 브랜치별 샌드박스 참조
