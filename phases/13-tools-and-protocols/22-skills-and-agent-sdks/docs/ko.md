# 에이전트 스킬: 이식 가능한 계약과 런타임 경계

> 스킬은 파일명이 더 좋은 긴 프롬프트가 아닙니다. 스킬은 런타임 계약을 통해 에이전트의 컨텍스트에 진입하는, 발견 가능한 지침, 리소스, 실행 가능한 헬퍼의 패키지입니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 13단계 · 01강 (도구 인터페이스), 13단계 · 05강 (도구 스키마 설계)
**시간:** 약 90분

## 학습 목표

- 프롬프트, 저장소 지침, 도구, 훅, 서브 에이전트, 플러그인과 혼동하지 않고 에이전트 스킬을 정의해 보세요.
- 이식 가능한 `SKILL.md` 계약을 읽고, 런타임별 확장 기능과 분리해 보세요.
- 발견, 선택, 활성화, 리소스 로드, 도구 사용, 검증을 각각 다른 라이프사이클 단계로 설명해 보세요.
- 런타임이 에이전트의 카탈로그에 스킬을 배치하기 전에 스킬 패키지를 검증해 보세요.
- 구체적인 작업에 대해 스킬, MCP 도구, 훅, 서브 에이전트, 일반 코드 중 하나를 선택해 보세요.

## 10분 내 첫 성공

긴 설명 전에 이 작업을 수행해 보세요. 작은 스킬을 생성하고, 완전한 리뷰어 번들을 실제 에이전트 호스트에 설치한 후, 호출하고, 결과를 검증하고, 제거합니다. 이를 통해 관찰 가능한 결과로 라이프사이클을 증명합니다.

### 실제 호스트 실습을 위한 사전 점검

실제 호스트 체크포인트에는 Node.js, `npx`, Python 3, 선택된 스킬 지원 호스트 하나, 그리고 설치 프로그램에서 선택한 프로젝트 또는 사용자 범위에 대한 쓰기 권한이 필요합니다. 먼저 로컬 명령을 검증하세요:

```bash
node --version
npx --version
python3 --version
```

설치 전에 사용할 호스트와 범위를 결정하세요. 요구 사항이 이용 불가능한 경우, 웹사이트에서 이 강의를 읽거나 아래 수동 패키지 연습을 계속 진행하세요. 이 폴백은 계약을 가르치지만, 호스트 발견, 호출, 번들 스크립트 실행, 제거 동작을 증명하지는 못합니다. 이러한 관찰 항목은 '보류'로 표시해 두세요.

### 1. 빈 작업 디렉토리에서 시작하기

학습 작업을 보관하는 임의의 부모 디렉토리에서 다음 명령을 실행하세요:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

마지막 명령은 아무것도 출력하지 않아야 합니다. 파일이 출력되면, 리뷰가 명확한 경계를 갖도록 다른 빈 디렉토리를 선택하세요.

첫 번째 스킬용 디렉터리를 생성합니다:

```bash
mkdir -p my-first-skill
```

`my-first-skill/SKILL.md`을 다음 내용으로 생성합니다:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

의도한 디렉터리에 파일이 생성되었는지 확인합니다:

```bash
test -f my-first-skill/SKILL.md
```

출력이 없고 종료 코드 0이면 파일이 존재합니다.

### 2. 완전한 리뷰어 번들 설치

`agent-skills-first-run`에 남아 다음을 실행합니다:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

사용 중인 에이전트 호스트와 범위를 선택합니다. 설치 프로그램은
`skill-contract-reviewer`과 기록한 대상 위치를 나열해야 합니다. `--full-depth`은
이 강의의 스킬이 참조, 스크립트, 자산을 포함하는 중첩 번들이므로
필수입니다.

`SKILL_ROOT`을 설치 프로그램이 보고한 절대 경로 디렉터리로 설정합니다. 이는
설치된 `SKILL.md`을 포함하는 디렉터리여야 하며, 강의 소스 디렉터리나
현재 작업 공간이어서는 안 됩니다:

```bash
# 자리표지자를 설치 프로그램이 출력한 대상 위치로 교체합니다.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

에이전트 세션이 이미 열려 있다면, 새 세션을 시작하거나 해당 호스트의
스킬 재스캔 명령을 사용하세요. 모든 호스트가 카탈로그를 핫 리로딩한다고
가정하지 마세요.

### 3. 명시적으로 호출하기

설치된 에이전트에서 `agent-skills-first-run`을 작업 디렉터리로 사용하며,
해당 호스트가 지원하는 문법을 사용하세요:

| 호스트 | 명시적 호출 |
|---|---|
| Codex | `skill-contract-reviewer`, 또는 `/skills`에서 선택한 후 리뷰 요청 제공 |
| Claude Code | `/skill-contract-reviewer` 뒤에 리뷰 요청 |
| 이식성 폴백 | `Use skill-contract-reviewer to review the target package.` |

요청에서 `SKILL_ROOT`과 `TARGET_ROOT`에 대해 출력된 절대 값을 사용하세요. 실행 전에
호스트가 이를 확장하고, 프로세스 작업 디렉터리에 의존하지 않는 정확히
해결된 명령을 표시하도록 요구하세요:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

해결된 명령은 자리표지자가 남지 않은 다음 형태여야 합니다:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

성공적인 결과는 다음 세 가지 속성을 모두 포함합니다:

1. 호스트가 `skill-contract-reviewer`을 이름으로 찾습니다.
2. 리뷰어가 패키지 계약을 읽고 번들된 검증기를 실행합니다.
3. 응답에는 샘플에 대한 구조적 오류가 없는 검증 보고서와
정당화된 원시 선택이 포함됩니다.

실행 증거에는 스크립트 경로, 대상 경로, cwd, 정확한 인자 벡터 및 종료 코드가 명시되어야 합니다. 이러한 필드가 없는 유창한 보고서는 설치된 동반 스크립트가 실행되었음을 증명하지 못합니다.

호스트가 스킬을 사용할 수 없다고 보고하는 경우, 설치 대상 경로를 확인하고 한 번 재스캔하거나 재시작한 후 명시적 요청을 다시 시도해 보세요. 설치 실패를 숨기기 위해 스킬 설명을 재작성하지 마세요.

### 4. 암묵적 선택을 탐지해 보세요

새로운 에이전트 턴을 시작하고 스킬 이름을 지정하지 않고 동일한 작업을 입력해 보세요:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

호스트가 선택된 스킬을 노출하는 경우, `skill-contract-reviewer`을 선택했는지 기록하세요. 호스트가 라우팅을 노출하지 않는 경우, 암묵적 선택을 검증되지 않은 것으로 표시하세요. 명시적 호출은 이식 가능한 폴백입니다.

### 5. 정리하기

설치된 리뷰어 번들만 제거하세요:

```bash
npx skills remove skill-contract-reviewer
```

설치 중에 사용했던 동일한 호스트와 범위를 선택하세요. 재스캔이나 새 세션 이후, `skill-contract-reviewer`에 대한 명시적 요청은 사용할 수 없다고 보고해야 합니다. `my-first-skill`은 이후 강의를 위해 유지하거나, 트랙을 완료한 후 실험실 디렉터리를 제거하세요.

## 문제점

팀이 신뢰할 수 있는 릴리스 워크플로우를 가지고 있다고 가정해 보세요. 이 워크플로우는 병합된 변경 사항을 찾고, 마이그레이션 노트를 확인하며, 변경 로그를 업데이트하고, 패키징 명령을 실행하고, 검토 체크리스트를 생성합니다.

이 워크플로우를 하나의 프롬프트에 넣으면 붙여넣기는 쉽지만 운영하기는 어렵습니다. 프롬프트에는 안정적인 식별자, 발견 규칙, 자원 경계, 테스트 가능한 패키지 형태가 없으며, 기본 질문에 대한 답도 없습니다. 누가 호출할 수 있나요? 모델은 언제 선택해야 하나요? 어떤 스크립트를 실행할 수 있나요? 어떤 파일이 신뢰되나요? 컨텍스트가 압축될 때 무엇이 남나요?

반대되는 실수는 모든 재사용 가능한 지시문을 스킬로 취급하는 것입니다. 저장소 관례, 결정적 자동화, 외부 도구, 이벤트 훅 및 위임된 에이전트는 서로 다른 문제를 해결합니다. 이 모든 것을 `SKILL.md`에 패키징하면 이식 가능한 것처럼 보이지만 한 호스트의 문서화되지 않은 동작에 의존하는 디렉터리가 생성됩니다.

첫 번째 엔지니어링 작업은 분류입니다. 패키징 방법을 결정하기 전에 산출물이 무엇인지 결정하세요.

## 개념

### 스킬은 절차적 지식을 인코딩합니다

에이전트 스킬은 진입점이 `SKILL.md`인 디렉토리입니다. 진입 파일은 YAML frontmatter에 이어 Markdown 지침을 포함합니다. 이 디렉토리에는 참조, 스크립트 및 자산도 포함될 수 있습니다.

```figure
skill-package-anatomy
```

배포 가능한 단위는 Markdown 파일 단독이 아니라 디렉토리입니다. 참조가 누락된 `SKILL.md`가 복사된 경우, frontmatter가 파싱되더라도 패키지가 손상된 것입니다.

### 인접한 추상화

| 산출물 | 주요 역할 | 로드 또는 실행 시점 | 모방하지 말아야 할 것 |
|---|---|---|---|
| 프롬프트 | 모델 상호작용을 형성 | 애플리케이션이나 사용자가 포함 | 리소스가 포함된 버전 관리 패키지 |
| 저장소 지침 | 코드베이스의 상시 규칙을 설명 | 코딩 런타임이 해당 범위에 진입 | 재사용 가능한 작업 워크플로 |
| 에이전트 스킬 | 재사용 가능한 절차적 지식을 제공 | 명시적 또는 암시적 활성화 | 엄격한 권한 경계 |
| MCP 도구 | 타입이 지정된 원격 기능을 노출 | 모델이나 애플리케이션이 호출 | 상세한 운영 절차 |
| 훅 | 이벤트에 대해 결정론적 로직을 실행 | 선언된 이벤트가 발생 | 확률적 모델 라우팅 |
| 서브 에이전트 | 별도의 컨텍스트와 상태로 작업을 위임 | 오케스트레이터가 생성하거나 호출 | 정적 지침 번들 |
| 플러그인 | 더 큰 런타임 확장을 배포 | 호스트가 설치하거나 활성화 | 이식 가능한 스킬 계약 자체 |
| 학습된 스킬 라이브러리 | 경험을 통해 발견된 행동을 저장 | 정책이 이전 프로그램이나 궤적을 검색 | 표준 기반 `SKILL.md` 패키지 |

릴리스 스킬은 에이전트에게 릴리스를 검사하는 방법을 알려줄 수 있습니다. MCP 서버는 릴리스 레지스트리를 노출할 수 있습니다. 훅은 직접 푸시를 금지할 수 있습니다. 서브 에이전트는 후보를 독립적으로 감사할 수 있습니다. 이러한 요소들은 서로 다른 책임을 유지하기 때문에 결합됩니다.

### "스킬"이라는 단어는 두 가지 다른 개념을 지칭합니다

연구 시스템은 때때로 학습된 프로그램, 성공적인 궤적, 환경 특정 정책 조각을 스킬이라고 부릅니다. 에이전트는 탐색 중에 이러한 산출물을 생성하고, 작업 유사성에 따라 검색하며, 실행하고, 피드백을 통해 라이브러리를 수정할 수 있습니다. 14단계 · 10은 이러한 종류의 평생 학습 라이브러리를 구축합니다.

이 미니 트랙에서의 에이전트 스킬(Agent Skill)은 다릅니다. 선언된 파일 시스템 계약, 카탈로그 메타데이터, 점진적 공개(Progressive Disclosure), 런타임 중개 호출, 호스트 제어 도구를 갖춘 제작된 패키지입니다. 에이전트가 생성하거나 개선할 수 있지만, 형식 자체를 학습할 필요는 없습니다.

| 차원 | 에이전트 스킬 패키지 | 학습된 스킬 라이브러리 |
|---|---|---|
| 주요 단위 | `SKILL.md` 디렉토리 | 프로그램, 정책, 궤적, 또는 메모리 레코드 |
| 생성 | 제작, 생성, 또는 큐레이션 | 일반적으로 환경 경험에서 발견 |
| 선택 | 카탈로그 설명 및 런타임 정책 | 작업 상태에 대한 검색 또는 정책 |
| 실행 | 모델이 지시문을 따르고 호스트 도구를 호출 | 환경이 저장된 행동이나 코드 산출물을 실행 |
| 이식성 | 패키지 계약이 호환 가능한 호스트 간에 이동 가능 | 종종 하나의 환경 및 행동 공간에 묶임 |
| 평가 | 라우팅, 산출물, 안전성, 호스트 호환성 | 보상, 성공률, 전이, 라이브러리 성장 |

두 아이디어 모두 재사용 가능한 역량을 패키징합니다. 이름이 같다는 이유만으로 구현 주장을 공유해서는 안 됩니다.

### 이식 가능한 핵심

에이전트 스킬 사양은 두 개의 프론트매터(frontmatter) 필드를 요구합니다:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name`는 안정적인 식별자입니다. 사양의 명명 규칙을 충족하고 부모 디렉토리와 일치해야 합니다. `description`는 문서화와 라우팅 메타데이터 모두를 겸합니다. 스킬이 무엇을 수행하는지, 언제 적용되는지 설명해야 합니다.

이식 가능한 선택 필드는 다음과 같습니다:

| 필드 | 목적 | 이식성 노트 |
|---|---|---|
| `license` | 패키지의 조건 명시 | 핵심 사양 |
| `compatibility` | 환경 요구 사항 명시 | 핵심 사양 |
| `metadata` | 문자열 값의 확장 데이터 전달 | 핵심 사양 |
| `allowed-tools` | 사전 승인된 도구 제안 | 실험적; 호스트 지원이 다름 |

Markdown 본문은 운영 지침을 담습니다. 워크플로, 결정 지점, 실패 동작, 지원 자원과의 직접적인 경로를 정의해야 합니다.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### 런타임 확장은 두 번째 계층입니다

일부 호스트는 추가 프론트매터나 동반 구성을 허용합니다. 이러한 필드는 유용할 수 있지만, 자동으로 이식되지는 않습니다.

| 동작 | 예시 호스트 확장 | 이식 가능한 핵심? |
|---|---|:---:|
| 모델 라우팅에서 스킬을 숨기되 직접 사용자 호출은 유지 | `disable-model-invocation` | 아니요 |
| 사용자의 명령 메뉴에서 스킬을 숨기되 모델 라우팅은 허용 | `user-invocable` | 아니요 |
| 명령 메뉴에서 인자 도움말 표시 | `argument-hint` | 아니요 |
| 위임된 컨텍스트에서 스킬 실행 | `context`, `agent` | 아니요 |
| 모델 또는 추론 설정 고정 | `model`, `effort` | 아니요 |
| 수명주기 자동화 등록 | `hooks` | 아니요 |
| Codex에서 암시적 호출 비활성화 | `agents/openai.yaml` 정책 | 아니요 |

각 확장을 어댑터로 취급하세요. 확장이 없어도 핵심 워크플로우가 유효하도록 유지하고, 폴백을 문서화하며, 이를 소비하는 호스트를 테스트하세요. 런타임은 알 수 없는 필드를 무시하거나, 거부하거나, 동작을 구현하지 않은 채 보존할 수 있습니다.

### 프론트매터는 실행 가능한 메타데이터입니다

메타데이터는 스킬 본문이 읽히기 전에 시스템 동작을 변경합니다.

- 잘못된 `name`는 발견(discovery) 실패를 유발할 수 있습니다.
- 모호한 `description`는 잘못된 요청을 라우팅할 수 있습니다.
- 인간 전용 플래그는 모델 카탈로그에서 스킬을 제거할 수 있습니다.
- 도구 허용은 호스트가 권한을 요청하는지 여부를 변경할 수 있습니다.
- 컨텍스트 설정은 실행을 별도의 에이전트 세션으로 이동할 수 있습니다.

프론트매터를 구성 코드처럼 검토하세요. 이를 검증하고, 버전을 관리하며, 그 동작을 평가(evals)에 포함하세요.

### 스킬 수명주기

```figure
skill-runtime-lifecycle
```

각 화살표는 자체 실패 모드를 가진 경계입니다.

1. **발견(Discovery)**은 구성된 위치에서 가능한 패키지를 찾습니다.
2. **검증(Validation)**은 카탈로그 게시 전에 잘못된 패키지나 안전하지 않은 패키지를 거부합니다.
3. **카탈로그화(Cataloging)**는 `name`와 `description`를 노출하며, 전체 패키지는 노출하지 않습니다.
4. **선택(Selection)**은 스킬이 관련 있는지 결정합니다.
5. **활성화(Activation)**는 본문을 모델 가시 컨텍스트에 로드합니다.
6. **공개(Progressive Disclosure)**는 브랜치가 필요로 할 때만 참조나 자산을 읽습니다.
7. **실행**은 호스트의 권한 및 격리 규칙 하에서 호스트 도구를 사용합니다.
8. **검증**은 모델의 주장과 독립적으로 생성된 산출물을 확인합니다.

이 단계를 통합하면 잘못된 개념 모델이 생깁니다. 발견된 스킬은 활성 상태가 아닙니다. 활성 스킬은 설명하는 모든 작업을 수행할 권한이 없습니다. 허용된 도구 호출은 결과가 정확하다는 증거가 아닙니다.

### 스킬과 도구는 직교합니다

MCP는 "이 애플리케이션이 호출할 수 있는 기능은 무엇이며, 그 스키마는 무엇인가?"에 답합니다. 스킬은 "에이전트가 이 유형의 작업에 어떻게 접근해야 하는가?"에 답합니다.

```figure
skill-tool-orthogonality
```

스킬이 도구를 언급할 수 있지만, 실제 기능 레지스트리는 호스트가 소유합니다. 도구가 없다면 스킬은 대체 수단(fallback)을 명시하거나 명확하게 실패해야 합니다. 기능 이름을 지정하는 것만으로 기능이 생성된다는 암시를 절대 해서는 안 됩니다.

### 스킬과 저장소 지침은 범위가 다릅니다

저장소 지침은 이미 있는 환경을 설명합니다: 명령어, 관례, 생성된 파일, 경계 등. 스킬은 여러 저장소에서 발생할 수 있는 작업에 대한 재사용 가능한 절차를 제공합니다.

둘 다 적용될 경우, 활성 사용자 요청과 저장소 규칙이 스킬을 제약합니다. 일반적인 리팩토링 스킬은 생성된 파일 편집을 금지하는 저장소 규칙을 덮어쓸 수 없습니다.

### 스킬은 서로를 가져오지(import) 않습니다

한 스킬이 에이전트에게 다른 스킬을 호출하도록 지시할 수 있지만, 이는 언어 수준의 import가 아닙니다. 두 번째 스킬은 여전히 런타임 발견, 자격, 활성화, 권한, 컨텍스트 처리를 거칩니다.

스킬 간 의존성은 관찰 가능한 워크플로우 엣지로 작성하세요:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

이렇게 하면 의존성을 테스트할 수 있고, 호스트가 정책을 강제할 기회를 가집니다.

## 구현하기

`code/main.py`는 표준 중심의 작은 검증기와 산출물 선택기를 구현합니다. 모든 규칙이 보이도록 stdlib 전용으로 유지합니다.

검증기는 다음을 노출합니다:

- `parse_frontmatter(text)`는 메타데이터를 본문과 분리합니다.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`를 사용하여 필수 필드, 명명 규칙, 알 수 없는 확장자, 본문 존재 여부 및 이식성 제한을 확인합니다.
- `ValidationIssue` 및 `SkillReport`를 사용하여 하나의 불투명한 부울 값 대신 구조화된 증거를 반환합니다.
- 안전하게 해석할 수 없는 입력에 대해 `FrontmatterSyntaxError`를 사용합니다.

선택기는 `TaskShape` 및 `select_primitives(task)`을 노출합니다. 이는 작업의 요구 사항을 일반 코드, 저장소 지침, 스킬, 훅, 서브 에이전트, 또는 MCP 도구로 매핑합니다.

실험을 실행하세요:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

이 명령 블록은 로컬 클론이 필요하며, `git rev-parse --show-toplevel`가 저장소 루트를 해석할 수 있도록 클론 내부의 아무 위치에서나 시작해야 합니다.

데모는 유효한 휴대용 스킬 하나, 호스트 확장 스킬 하나, 유효하지 않은 패키지, 그리고 여러 작업 형태 결정을 위한 JSON을 출력합니다. 이슈 코드를 확인하세요. 패키지 검증기는 작성자를 대신해 추측하지 않고 아티팩트를 수정하는 방법을 설명해야 합니다.

### 검증 순서가 중요합니다

더 깊은 콘텐츠 규칙보다 먼저 저렴한 구조적 사실을 검증하세요:

```figure
skill-validation-order
```

이 순서는 첫 번째 깨진 불변 조건을 숨기는 2차 오류를 방지합니다.

## 사용하기

스킬을 작성하기 전에 이 결정 카드를 작성하세요:

| 질문 | 예인 경우 | 가능성 있는 원시 요소 |
|---|---|---|
| 여러 단계에 걸쳐 재사용 가능한 모델 판단이 필요한가요? | 절차는 안정적이지만 결정이 다양함 | 스킬 |
| 이벤트가 발생할 때마다 반드시 실행되어야 하나요? | 한 번의 실행 누락은 허용되지 않음 | 훅 또는 애플리케이션 코드 |
| 모델이 타입이 지정된 입력을 가진 외부 기능이 필요합니까? | 작업이 모델 컨텍스트 외부에 위치함 | 도구 또는 MCP 서버 |
| 작업이 격리된 컨텍스트, 상태, 또는 소유권을 필요로 합니까? | 분리된 워커가 제한된 결과를 반환함 | 서브 에이전트 |
| 이 지침이 특정 저장소에 한정됩니까? | 로컬 명령 및 제약 조건을 설명함 | 저장소 지침 |
| 한 번의 상호작용으로 충분합니까? | 패키지 수명 주기가 필요하지 않음 | 프롬프트 |

많은 생산 워크플로우는 여러 행을 사용합니다. 이 카드는 하나의 아티팩트가 모든 속성을 제공하는 척하는 것을 방지합니다.

## 출시하기

이 강의는 `outputs/` 아래에 `skill-contract-reviewer` 번들을 생성합니다. 이는 다음을 포함합니다:

- 제안된 스킬 패키지를 검토하는 휴대용 `SKILL.md`;
- 이식 가능한 계약과 원시 선택에 대한 참조 체크리스트;
- 결정적 검증 스크립트;
- 프롬프트, 스킬, 도구, 훅, 일반 코드 및 서브 에이전트를 포함하는 작업 형태 픽스처.

엔트리 파일만 설치하지 말고 전체 번들을 설치하세요:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --13단계 --type skill
```

코스 설치 프로그램은 각 13단계 스킬을 복사하고 `/tmp/aiefs-skills/manifest.json`에 기록합니다. 이 깨끗한 목적지는 패키지 형태를 확인합니다. 위의 첫 성공 루프는 실제 호스트에서 발견 및 호출을 확인합니다.

다음 강의는 각 라이프사이클 단계를 심화합니다. 24강은 발견 및 점진적 공개를 구축합니다. 25강은 호출 정책 및 라우팅을 구축합니다. 26강은 권한을 샌드박싱과 분리합니다. 27강은 전체 패키지를 평가된 릴리스 아티팩트로 변환합니다.

## 연습 문제

1. `TaskShape`를 사용하여 팀의 다섯 가지 워크플로우를 분류하세요. 원시를 하나 이상 선택한 모든 경우를 방어하세요.
2. 500자 `compatibility` 값은 통과하고 501자 값은 사양 오류로 실패함을 증명하는 경계 테스트를 추가하세요.
3. 허용 목록에 런타임 확장 하나를 추가하세요. 동일한 파일이 이식 전용 스킬과 여전히 구별됨을 증명하는 테스트를 작성하세요.
4. 400자 프롬프트를 `SKILL.md`, 참조 하나, 스크립트 계약 하나, 출력 템플릿 하나로 분할하세요. 모든 파일이 한 종류의 정보만 담당하도록 유지하세요.
5. 사용할 수 없는 MCP 도구를 참조하는 스킬에 대한 실패 응답을 설계하세요. 더 넓은 권한을 가진 도구로 조용히 대체하지 마세요.
6. 기존 스킬을 검토하고 모든 문장을 라우팅, 절차, 정책, 참조 포인터 또는 출력 계약으로 라벨링하세요. 해당하지 않는 모든 것을 이동하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|---|---|---|
| 에이전트 스킬 | "저장된 프롬프트" | 절차적 지침과 선택적 리소스를 포함하는 발견 가능한 디렉토리 |
| 이식 가능한 코어 | "모든 런타임이 공유하는 필드" | Agent Skills 사양이 정의한 계약 |
| 런타임 확장 | "추가 프론트매터" | 호스트별 구성으로, 호환되는 어댑터가 필요한 동작을 요구함 |
| 활성화 | "스킬이 실행됨" | 스킬 본문이 모델 가시 컨텍스트에 진입함; 실행은 이후에 발생할 수 있음 |
| 스킬 의존성 | "다른 스킬 가져오기" | 가용성 및 정책 검사를 수행하는 런타임 중개 호출 엣지 |
| 도구 계약 | "함수 스키마" | 기능에 대한 입력, 출력, 권한, 사이드 이펙트, 오류 및 증거 |

## 추가 읽기

- [Agent Skills specification](https://agentskills.io/specification) 이식 가능한 디렉터리 및 frontmatter 계약에 대해.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) 범위, 지침 및 리소스 조직에 대해.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) 현재 Codex의 발견 및 호출 동작에 대해.
- [Claude Code skills](https://code.claude.com/docs/en/skills) 한 런타임의 호출, 인자, 도구 및 위임된 컨텍스트 확장 기능에 대해.
