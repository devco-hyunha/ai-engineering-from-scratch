# 캡스톤 86 — 헌법 규칙 엔진

> 규칙은 이름, 술어, 그리고 설명으로 구성됩니다. 이 세 가지 중 하나라도 빠진 것은 규칙이 아니라 단순한 분위기(vibe)일 뿐입니다.

**유형:** Build
**언어:** Python, YAML
**선수 요건:** 18단계 안전 관련 강의, 19단계 트랙 A 25-29강
**시간:** 약 90분

## 문제점

분류기는 인식 가능한 실패를 다루고, 규칙 엔진은 계약상 실패를 다룹니다. 코딩 어시스턴트를 작성하는 팀은 "코드를 포함하는 모든 응답은 실행 가능한 블록이나 명시된 가정을 포함해야 한다"와 같은 제약이 필요합니다. 고객 지원 봇을 운영하는 팀은 "모든 거절은 다음 단계를 제시해야 한다"는 제약이 필요합니다. 이러한 제약은 자연스러운 분류기 대상이 아닙니다. 이는 응답, 대화, 시스템 정책에 대한 술어이며, 엔지니어가 아닌 사람도 읽을 수 있어야 합니다.

정직한 표현은 선언적 파일입니다. 헌법은 코드와 함께 YAML로 버전 관리系统中에 위치하며, 별도의 리뷰 과정을 거칩니다. 각 규칙은 `name`, `predicate`, `severity`, `explanation` 템플릿을 가집니다. 엔진은 파일을 로드하고, 각 규칙을 후보 출력에 대해 평가하며, 발동된 규칙에 대해 구조화된 `Violation`를 반환합니다. 이 캡스톤의 규칙 엔진은 `all_of`, `any_of`, `not_`로 술어를 조합하므로, 단일 규칙으로 "응답에 코드가 포함되어 있다면, 실행 가능한 블록으로 끝나야 하며 내부 전용 라이브러리를 참조하지 않아야 한다"를 표현할 수 있습니다.

이 강의의 나머지 절반은 수정(revision)입니다. 차단만 하는 규칙 엔진은 절반만 완성된 것입니다. 수정을 제안하는 규칙 엔진은 운영적으로 유용합니다: 어시스턴트가 응답을 작성하고, 엔진이 위반을 표시하며, 수정기가 수정된 응답을 생성하고, 엔진이 수정이 규칙을 만족하는지 확인합니다. 이 강의는 최소한의 수정기(규칙별 정규식 교체)와 초안 및 수정본 간의 구조화된 diff(줄별 추가, 삭제, 편집)를 제공합니다.

## 개념

```mermaid
flowchart LR
  D["초안 응답"] --> RE["규칙 엔진"]
  RE -->|violations| F["fixer"]
  F --> R["수정된 응답"]
  R --> RE2["규칙 엔진 2차 패스"]
  RE2 -->|verdict| OUT["수락 또는 에스컬레이션"]
  D -.->|diff| R
```

규칙은 다음과 같은 형태를 가집니다

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "Code responses must end in either a closing fence or an explicit assumption."
  fix:
    append_if_missing: "\n\nAssumption: example inputs are valid."
```

술어는 원자적입니다: `contains_regex`, `not_contains_regex`, `ends_with_regex`, `starts_with_regex`, `max_words`, `min_words`. 구성은 `all_of`, `any_of`, `not_`입니다. 엔진은 `applies_when`를 먼저 평가합니다. 규칙이 적용되지 않으면 위반은 `not_applicable`으로 기록됩니다. 그렇지 않으면 엔진은 `must`를 평가하여 `pass` 또는 `violation`을 생성합니다.

심각도는 `low`, `medium`, `high`이며, 85강을 반영합니다. 다운스트림 게이트(87강)는 `high` 규칙 위반을 `high` 분류기 판정과 동일하게 처리합니다: 차단.

수정기는 선언적 연산 목록입니다: `append_if_missing`, `prepend_if_missing`, `replace_regex`. 각 연산은 이름으로 규칙을 변환에 매핑합니다. 수정기는 의도적으로 지역 편집으로만 제한됩니다. 구조적 재작성은 여기서 다루지 않는 별도의 거절 및 도움 계층에 속합니다.

diff는 원본과 수정본을 비교하여 계산됩니다. `op` (추가, 제거, 편집) 및 관련 텍스트를 포함하는 `Change` 레코드 목록입니다. 다운스트림 게이트는 diff를 기록할 수 있으며, 이를 통해 인간 리뷰어가 수정기의 동작을 시간에 걸쳐 감사할 수 있습니다.

```figure
cd-constitution-loop
```

## 구현하기

`code/rules.yml`는 헌법을 포함합니다. `code/main.py`의 로더는 YAML 파일(PyYAML이 사용 가능한 경우) 또는 JSON 파일(내장)을 허용합니다. 이 강의는 `rules.yml`를 포함하며, 강의 테스트는 두 코드 경로로 이를 파싱합니다. `code/main.py`는 `Engine` 및 `Fixer` 클래스와 `diff` 함수를 정의합니다. 구성은 `any_of`에서 단락 평가로 재귀적으로 평가됩니다.

출시된 헌법은 다음과 같습니다:

- `no-empty-refusal` (중간) - 거절은 제안이나 리디렉트를 포함해야 합니다
- `end-with-runnable-or-assumption` (중간) - 코드 응답은 깔끔하게 종료되어야 합니다
- `no-pii-in-examples` (높음) - 예제 데이터에는 이메일이나 전화번호 형식이 포함되지 않아야 합니다
- `cite-when-asserting-fact` (낮음) - "According to"로 시작하는 줄은 괄호 인용을 포함해야 합니다
- `no-internal-library-leak` (높음) - `internal-only` 및 `policybot-internal` 단어가 출력에 나타나지 않아야 합니다
- `bounded-length` (낮음) - 응답은 800 단어를 초과하지 않아야 합니다

## 사용하기

`python3 main.py`. 데모는 세 개의 초안 응답을 엔진에 실행하고, 위반 사항을 출력하며, 수정기를 실행하고, diff를 출력하고, `outputs/rules_report.json`를 작성합니다. 하나의 픽스처는 적용되지 않는 규칙(초안에 코드 블록이 없음)을 포함하며, 보고서는 해당 규칙에 대해 `not_applicable`를 표시하여 팀이 엔진이 이를 명시적으로 평가했음을 확인할 수 있습니다.

## 출시하기

`outputs/skill-constitutional-rules-engine.md`는 규칙 문법과 수정기 연산을 문서화합니다.

## 연습 문제

1. 프롬프트가 안전을 언급할 때 모든 응답에 "If this is urgent"라는 문구를 포함해야 하는 규칙을 추가하세요. 합성을 사용하세요.
2. 정규식 수정기를 이름 있는 슬롯을 받는 템플릿 수정기로 교체하세요. 새로운 설계에 따라 하나의 규칙을 재작성하여 시연하세요.
3. 초안 코퍼pus가 주어지면 규칙별 위반률을 반환하는 메트릭 엔드포인트를 추가하세요. 이를 통해 팀은 어떤 규칙이 과도하게 발동되는지 확인할 수 있습니다.

## 핵심 용어

| 용어 | 일반적인 사용 | 정확한 의미 |
|---|---|---|
| 헌법(constitution) | 모호한 정책 문서 | 술어, 심각도, 설명을 포함한 규칙을 담은 YAML 파일 |
| 술어(predicate) | 검사 | 텍스트에서 bool로 호출 가능한 함수이며, all_of/any_of/not_를 통해 원자적 또는 합성됨 |
| 위반(violation) | 실패 | 규칙 이름, 심각도, 설명, 매칭된 범위를 포함한 구조화된 레코드 |
| 수정기(fixer) | 모델 미세 조정 | 초안을 수정된 버전으로 매핑하는 규칙별 결정론적 변환 |
| diff | 문자열 비교 | 초안과 수정된 버전 간의 추가, 제거, 편집 연산의 구조화된 목록 |

## 추가 읽기

87강은 이 엔진을 입력 측 감지기와 출력 측 분류기를 결합하여 단일 안전 게이트(안전 게이트)로 구성합니다.
