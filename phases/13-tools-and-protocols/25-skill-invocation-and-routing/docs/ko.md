# 스킬 호출 및 라우팅

> 호출은 권한 결정에 이어 관련성 결정이 따릅니다. 좋은 설명은 모델이 선택하는 데 도움이 되며, 좋은 정책은 그 선택이 허용되는지 결정합니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 13단계 · 24강 (스킬 발견 및 점진적 공개)
**시간:** 약 105분

## 학습 목표

- 명시적 사용자 호출, 암시적 모델 호출, 애플리케이션 호출, 스킬 간 호출을 구분합니다.
- 인간 가시성과 모델 자격을 독립적인 정책 차원으로 모델링합니다.
- 긍정적 트리거와 근접 실패 경계를 포함하는 라우팅 설명을 작성합니다.
- 추적 및 테스트에서 자격, 선택, 활성화, 인자 바인딩, 실행을 분리합니다.
- 런타임별 호출 필드를 이식 가능한 프론트매터로 제시하지 않고 적응합니다.

## 문제점

`database-migration` 스킬을 설치합니다. 사용자는 이름으로 실행할 수 있지만, 모델도 그 설명을 보고 누군가가 일반적인 데이터베이스 질문에 대해 물었을 때 선택합니다. 스킬은 설명만 필요했던 작업에 대해 스키마 변경을 제안합니다.

`user-invocable: false`를 추가하여 사람들이 수동으로 실행하는 것을 차단하려고 기대합니다. 다른 런타임에서는 해당 필드가 무시됩니다. `disable-model-invocation: true`를 추가하여 스킬이 완전히 사라지기를 기대합니다. 이를 이해하는 런타임에서는 사용자가 명시적으로 호출할 수 있습니다.

필드 이름에는 문제가 없습니다. 모델이 잘못되었습니다. "사용자가 볼 수 있다", "모델이 선택할 수 있다", "애플리케이션이 사전 로드할 수 있다", "내부 도구가 실행할 수 있다"는 별개의 사실입니다. `invocable`라는 단일 부울 값은 이들을 표현할 수 없습니다.

라우팅에는 두 번째 실패 모드가 있습니다. 설명이 모호하면 여러 스킬이 유력해집니다. 설명에 키워드를 과도하게 채우면 관련 없는 작업이 트리거됩니다. 카탈로그는 확률적 인터페이스입니다: 충분히 컴팩트하여 적합하고, 충분히 구체적하여 라우팅됩니다.

## 개념

### 생애 주기를 시작할 수 있는 5가지 채널

| 행위자 | 호출 형태 | 일반적인 사용 | 주요 위험 |
|---|---|---|---|
| 인간 사용자 | UI 또는 프롬프트에서 스킬을 지정 | 의도적인 워크플로 선택 | 사용자가 호스트가 부여하지 않는 가용성이나 권한을 기대함 |
| 모델 또는 자율 에이전트 | 작업 컨텍스트에서 카탈로그 항목을 선택 | 자동화된 전문가 절차 | 오탐지 라우팅 |
| 애플리케이션 | 런타임 코드를 통해 스킬을 활성화하거나 사전 로드 | 고정된 제품 워크플로 | 하나의 호스트에 대한 숨겨진 결합 |
| 다른 스킬 또는 하위 에이전트 | 워크플로 의존성으로 정확한 스킬을 요청 | 구성 | 순환, 누락된 의존성, 또는 컨텍스트 누출 |
| 평가 하네스 | 고정된 시나리오에서 정확한 스킬을 활성화 | 반복 가능한 측정 | 연구 중인 생산 정책을 우연히 우회하면서 스킬을 테스트함 |

이식 가능한 Agent Skills 사양은 패키지를 정의합니다. 하나의 범용 슬래시 명령 UI, 암시적 라우팅 플래그, 애플리케이션 API, 또는 하위 에이전트 생명주기를 표준화하지는 않습니다.

### 5가지 호출 단계

```figure
skill-invocation-stages
```

이 용어를 정확히 사용하세요:

- **적격(Eligible)**은 정책이 이 행위자가 스킬을 요청하는 것을 허용함을 의미합니다.
- **선택(Selected)**은 사용자가 스킬을 지정했거나 라우터가 관련 있다고 판단했음을 의미합니다.
- **활성화(Activated)**는 스킬의 지침이 작업 컨텍스트에 들어갔음을 의미합니다.
- **실행(Executing)**은 에이전트가 해당 지침에 따라 모델 또는 도구 작업을 시작했음을 의미합니다.
- **완료(Completed)**는 출력이 독립적인 성공 검사를 충족했음을 의미합니다.

`skill_used=true`만 기록하는 추적은 실패가 발생한 경계를 숨깁니다.

### 인간 및 모델 호출은 2x2 매트릭스를 형성합니다

| 인간이 호출 가능 | 모델이 호출 가능 | 모드 | 적합한 예시 |
|:---:|:---:|---|---|
| 예 | 예 | 공유 | 코드 설명, 테스트 계획, 문서 검토 |
| 예 | 아니오 | 인간 전용 | 게시 준비, 청구 내보내기, 파괴적 정리 계획 |
| 아니오 | 예 | 모델 전용 | 내부 스타일 가이드, 도메인 참조, 자동화된 지원 절차 |
| 아니오 | 아니오 | 비활성화 또는 애플리케이션 전용 | 단계적 롤아웃, 폐기된 패키지, 프로그래밍 방식 사전 로드 |

이 매트릭스는 정책 모델이며, 표준 YAML이 아닙니다.

현재의 한 호스트는 인간 전용 행에 `disable-model-invocation: true`를, 모델 전용 행에 `user-invocable: false`를 사용합니다. 기본값은 둘 다입니다. 다른 호스트는 `agents/openai.yaml`와 `allow_implicit_invocation: false`를 사용하여 명시적 호출을 유지하면서 암시적 선택을 비활성화합니다. 이들은 런타임 어댑터입니다. 알 수 없는 호스트는 이를 무시할 수 있습니다.

혼란스러운 세부 사항이 중요합니다: `user-invocable: false`는 "모델이 이 스킬을 사용할 수 없다"는 의미가 아닙니다. 이는 해당 스킬을 정의한 호스트에서 직접적인 사용자 호출을 제거합니다. `disable-model-invocation: true`는 "스킬이 비활성화되었다"는 의미가 아닙니다. 이는 모델이 주도하는 선택을 제거하면서 명시적인 사용자 접근은 유지합니다.

### 명시적 호출은 신원 우선입니다

명시적 호출은 신원을 직접 제공합니다:

```text
/release-readiness v2.4.0
```

또는:

```text
release-readiness check v2.4.0 without publishing
```

현재 Codex 인터페이스는 선택을 위해 `/skills`를 문서화하며, 명시적 호출을 위한 요청에는 평문 스킬 이름을 사용합니다. Claude Code는 `/skill-name`와 호스트별 인수 확장을 문서화합니다. 정확한 구문, 메뉴 가시성, 인용 규칙 및 변수 확장은 호스트에 속합니다.

명시적 요청은 여전히 정책을 통과합니다. 스킬을 지정하는 것은 누락된 권한, 워크스페이스 제약, 승인 게이트 또는 런타임 격리를 우회해서는 안 됩니다.

### 암시적 호출은 설명 우선입니다

암시적 라우팅의 경우, 모델은 처음에 전체 본문이 아닌 카탈로그 메타데이터를 봅니다. 따라서 설명은 스킬의 라우팅 인터페이스입니다.

약한 예:

```yaml
description: Helps with releases.
```

너무 넓은 예:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

범위가 정해진 예:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

범위가 정해진 버전은 다음을 포함합니다:

1. **기능:** 준비된 후보를 검사합니다.
2. **출력:** 준비 상태 보고서.
3. **긍정적 경계:** 릴리스 아티팩트가 준비되었는지 묻습니다.
4. **부정적 경계:** 일반적인 빌드 및 개발은 범위 밖입니다.

부정적 경계는 두 개의 가까운 스킬이 어휘를 공유할 때 유용합니다. 이는 근접 실패(near-miss) 평가의 대체물이 아닙니다.

### 라우팅은 기각(abstain) 옵션이 있는 분류입니다

스킬 `s`과 요청 `x`에 대해 라우터 점수를 상상해 보세요:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

정확한 점수 산정은 산술 연산이 아니라 LLM의 결정일 수 있습니다. 엔지니어링 원칙은 여전히 유효합니다: 선택은 임계값과 경쟁 스킬을 능가해야 합니다. 증거가 약할 때는 기각(abstain)하세요.

```figure
skill-routing-abstention
```

높은 영향력을 가진 스킬의 경우, 강력한 설명이 있더라도 암시적 라우팅이 부적절할 수 있습니다. 오탐(false positive)의 비용이 자동 선택의 편의성을 초과하는 경우, 인간 전용 정책을 사용하세요.

### 적격성 판단이 순위 매기기보다 먼저 이루어져야 합니다

발견된 모든 스킬을 점수화하고, 가장 강한 매치를 선택한 후 해당 스킬의 정책을 확인하는 방식은 피하세요. 차단된 상위 매치가 적격인 하위 점수 후보의 검토를 부당하게 막을 수 있습니다.

암시적 라우팅에는 다음 순서를 사용하세요:

1. 요청한 행위자(actor)와 활성 호스트 어댑터 기준으로 발견된 스킬을 필터링하세요.
2. 적격 후보만 점수화하세요.
3. 임계값과 모호성 규칙을 통과하는 가장 강한 적격 매치를 선택하세요.
4. 적격 후보가 없거나 적격 점수가 충분히 강하지 않은 경우, 선택을 보류(abstain)하세요.

`incident-triage`가 `0.80` 점수를 받지만, 호스트 확장이 모델 호출을 비활성화한 경우를 가정해 보세요. `incident-review`는 `0.55` 점수를 받고 모델 호출을 허용합니다. 라우터는 `incident-review`를 최선의 적격 후보로 평가해야 합니다. `incident-triage`를 선택하고, 이를 거부하며, 중단해서는 안 됩니다.

이 순서는 정책 변경이 관련성 점수의 의미를 바꾸지 않도록 유지합니다. 적격성이 선택 집합을 정의하고, 관련성이 그 집합을 순위 매깁니다.

### 라우팅 평가에는 근접한 실패 사례(near miss)가 필요합니다

긍정 사례는 재현율(recall)을 증명합니다:

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

명확한 부정 사례는 기본 정밀도(precision)를 증명합니다:

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

근접한 실패 사례는 경계 품질을 드러냅니다:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

근접한 실패 사례는 릴리스 스킬과 `package` 및 `build`을 공유하지만, 다른 곳에 속합니다. 명백한 긍정 사례와 무관한 부정 사례만으로 구성된 라우팅 세트는 품질을 과대평가할 것입니다.

### 인수(argument)는 세 가지 표현을 가집니다

호출 인수는 여러 경계를 통과합니다:

```figure
skill-argument-boundaries
```

각 경계에서 의도를 보존하면서 텍스트를 코드로 취급하지 마세요.

- 호스트 파서(parser)는 명령어 구문과 인용(quoting)을 결정합니다.
- 스킬은 호스트 규칙에 따라 바인딩된 텍스트나 변수를 받습니다.
- 지시문은 필수 값과 기본값을 검증합니다.
- 도구 호출은 값을 타입이 지정된 스키마로 변환하고 재검증합니다.

원시 인자를 셸 명령에 직접 삽입하지 마세요. 인자 벡터로 호출되는 스크립트나 타입이 지정된 MCP 도구를 선호하세요.

### 애플리케이션 호출은 명시적인 오케스트레이션입니다

제품은 워크플로우가 이미 작업 유형을 알고 있으므로 스킬을 활성화할 수 있습니다. 예를 들어, 풀 리퀘스트 리뷰 서비스는 사용자가 Review를 누른 후 `pull-request-risk-review`를 사전 로드할 수 있습니다.

이는 라우팅 불확실성을 제거하지만 런타임 API에 대한 의존성을 생성합니다. 해당 어댑터는 이식 가능한 본문 외부에 유지하세요:

```figure
skill-host-adapter
```

스킬은 다른 호환 클라이언트가 열었을 때에도 이해할 수 있어야 합니다.

### 스킬 간 호출은 도구와 유사한 엣지입니다

`release-readiness`가 의존성 파일이 변경되었을 때 `security-change-review`를 요청한다고 가정해 보세요.

호출자는 다음을 제공해야 합니다:

- 대상 스킬 식별자;
- 범위가 한정된 작업 및 산출물 경로;
- 예상되는 응답 계약;
- 호출 이유;
- 사용 불가능할 경우의 폴백;
- 최대 깊이 또는 순환 규칙.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

두 번째 스킬은 첫 번째 스킬에 무분별하게 붙여넣지 마세요. 호스트는 스킬을 어떻게 활성화할지, 컨텍스트를 공유할지, 포크에서 실행할지, 도구 결과를 통해 반환할지 결정합니다.

### 컨텍스트 수명 주기는 호스트에 특화됩니다

활성화 후 스킬 본문은 대화에 남아 있을 수 있고, 압축(compaction) 중에 요약될 수 있으며, 위임된 컨텍스트에서 실행될 수 있습니다. 도구 허용은 한 턴 동안 지속될 수 있는 반면, 지침은 더 오래 지속될 수 있습니다. 서브 에이전트는 부모의 전체 히스토리 없이 스킬을 받을 수 있습니다.

보이지 않는 수명 주기 가정에 의존하는 스킬을 작성하지 마세요. 내구성 있는 출력은 파일이나 타입이 지정된 상태에 저장하고, 재진입을 안전하게 만들며, 중단 후 다시 로드해야 하는 내용을 명시하세요.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 구현하기

`code/main.py`는 정책과 라우팅을 분리된 어댑터로 구현합니다.

모델은 다음을 포함합니다:

- 인간, 모델, 자율 에이전트, 애플리케이션, 스킬 및 하네스 호출자를 위한 `Actor`;
- 라우팅 식별자를 위한 `SkillMetadata`;
- 인간/모델 매트릭스를 위한 `InvocationPolicy`;
- 추적 가능한 입력 및 결과를 위한 `InvocationRequest` 및 `InvocationDecision`;
- `CorePolicyAdapter`는 호스트 확장 기능이 없는 이식성 있는 동작을 위해 사용하세요;
- `ExtensionPolicyAdapter`는 인식된 런타임 필드를 위해 사용하세요;
- `build_invocation_matrix(policy)`는 2x2 뷰를 위해 사용하세요;
- `route_request(skills, request, adapter)`는 관련성 순위 매기기, 선택 및 거부 전에 자격 필터링을 위해 사용하세요.

실행해 보세요:

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

데모는 명시적 인간, 암시적 모델, 자율 에이전트, 애플리케이션, 스킬 조합 및 하네스 채널에 대한 하나의 매트릭스와 결정을 출력합니다. 확장 어댑터 결과는 자격을 갖춘 대안이 순위 매겨지기 전에 차단된 상위 어휘 매칭이 제거되는 것을 보여줍니다. 또한 정확한 이름 허용 목록을 포함합니다. 모델 API는 필요하지 않습니다. 결정적 라우터는 정책 경계를 검사 가능하게 만들기 위해 존재하며, 어휘 매칭이 프로덕션 모델 라우팅을 재현한다고 주장하기 위한 것이 아닙니다.

### 코어 및 확장 어댑터가 분리되는 이유

하나의 파서가 모든 관찰된 frontmatter 필드에 의미를 부여하면, 런타임 관례를 가짜 표준으로 조용히 승격시킵니다. 분리된 어댑터는 호출자가 어떤 호스트 시맨틱이 활성화되는지 명시적으로 지정하도록 강제합니다.

`CorePolicyAdapter`는 애플리케이션이 제공한 정책만 사용합니다. `ExtensionPolicyAdapter`는 명시적인 호스트 필드 집합을 인식하며, 어떤 필드가 결정을 변경했는지 기록합니다.

## 사용하기

스킬을 게시하기 전에 호출 계약을 작성하세요:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

이 계약은 어댑터 및 테스트를 위한 설계 문서입니다. 표준이 이를 명시적으로 채택하지 않는 한, 이식 가능한 `SKILL.md` frontmatter가 아닙니다.

## 출시하기

이 강의는 `skill-invocation-router` 번들을 생성합니다. 여기에는 호출 모델 참조, 예제 호스트 정책 및 하나의 인간, 모델, 자율 에이전트, 애플리케이션, 스킬 조합 또는 하네스 요청을 평가하고 채널, 어댑터, 점수 및 이유를 포함하는 JSON 결정을 반환하는 실행되지 않는 CLI가 포함됩니다.

단일 요청 CLI는 정책 프로브이며, 완전한 트리거 평가가 아닙니다. 27강의 레이블이 지정된 긍정 및 근접 실패 설계를 사용하여 혼동 수, 정밀도, 재현율 및 반복 실행 안정성을 계산하세요.

## 연습 문제

1. 인간/모델 매트릭스의 네 행을 모두 만들고 각각에 대해 하나의 합법적인 사용 사례를 작성하세요.
2. `CorePolicyAdapter`에 애플리케이션 전용 활성화를 추가하세요. 인간 및 모델 호출자가 계속 거부됨을 증명하세요.
3. 배포 스킬에 대해 10개의 근접 오탐(near miss)을 작성해 보세요. 각 프롬프트는 스킬과 어휘를 공유해야 하지만, 서로 다른 워크플로에 속해야 합니다.
4. 상위 두 라우팅 점수 사이에 모호성 마진을 추가하세요. 마진이 너무 작으면 `ask`을 반환합니다.
5. 스킬 간 요청에 최대 구성 깊이를 추가하고 두 스킬의 순환을 감지하세요.
6. 레이블이 지정된 동일한 데이터셋을 코어 및 확장 어댑터를 통해 실행하세요. 변경된 모든 결정을 설명해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|---|---|---|
| 명시적 호출 | "슬래시 명령" | 액터가 스킬 식별자를 직접 제공하며, 정책의 적용을 받습니다 |
| 암시적 호출 | "모델이 선택합니다" | 라우터가 작업 컨텍스트에 기반하여 자격 있는 카탈로그 메타데이터에서 선택합니다 |
| 사용자 호출 가능 | "사람이 사용할 수 있습니다" | 호스트별 메뉴나 직접 호출 속성이며, 코어 필드가 아닙니다 |
| 모델 호출 가능 | "에이전트가 사용할 수 있습니다" | 호스트 정책 하에서 암시적 모델 선택의 자격 요건입니다 |
| 호출 어댑터 | "프론트매터 파서" | 호스트의 필드와 API를 선언된 정책 모델로 매핑하는 코드입니다 |
| 근접 오탐 | "하드 네거티브" | 스킬의 의도된 입력과 유사하지만 트리거되지 않는 요청입니다 |
| Abstention | "스킬이 선택되지 않았습니다" | 증거가 없거나 모호할 때 의도적으로 선택하는 라우팅 결과입니다 |

## 추가 읽기

- 긍정적 트리거, 특이성 및 평가에 대한 [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)을 참고하세요.
- 트리거 및 출력 평가 설계에 대한 [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)을 참고하세요.
- 현재 Codex의 명시적 및 암시적 호출 제어에 대한 [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)을 참고하세요.
- 한 호스트의 `user-invocable`, `disable-model-invocation`, 인자 및 위임된 컨텍스트에 대한 [Claude Code skills](https://code.claude.com/docs/en/skills)을 참고하세요.
