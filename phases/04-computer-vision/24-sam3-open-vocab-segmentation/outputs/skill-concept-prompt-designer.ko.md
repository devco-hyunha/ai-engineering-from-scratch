---
name: skill-concept-prompt-designer
description: 사용자 발화를 분할, 모호성 해소, 폴백을 포함하는 잘 형성된 SAM 3 개념 프롬프트로 변환
version: 1.0.0
phase: 4단계
lesson: 24강
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# 개념 프롬프트 디자이너

SAM 3의 정확도는 개념 프롬프트의 표현 방식에 크게 의존합니다. 이 스킬은 자유 형식의 사용자 발화를 SAM 3가 잘 처리하는 프롬프트로 정규화합니다.

## 사용 시점

- 자연어 객체 쿼리를 허용하는 UI를 구축할 때.
- 상위 호출자가 문장을 보내는 API를 통해 SAM 3를 노출할 때.
- SAM 3의 매칭이 불량한 경우를 디버깅할 때 — 대부분 모델이 아니라 프롬프트가 잘못 형성된 것입니다.

## 입력

- `utterance`: 원시 사용자 문자열.
- `context`: 선택적 도메인 힌트 (예: "surveillance", "medical", "retail").
- `max_concepts`: 발화당 추출할 최대 개념 수; 기본값은 5입니다.

## SAM 3가 선호하는 규칙

- **문장이 아닌 짧은 명사구.** `"cat"`가 `"there is a cat"`보다 우선합니다.
- **구체적인 명사.** `"skateboard"`가 `"thing to ride on"`보다 우선합니다.
- **명사 바로 앞의 수식어.** `"red car"`가 `"car that is red"`보다 우선합니다.
- **소문자.** SAM 3는 견고하지만 경험적으로 소문자 입력에서 약간 더 잘 작동합니다.
- **단수 또는 복수.** 둘 다 작동합니다; 복수는 여러 인스턴스가 예상될 때 도움이 됩니다.

## 단계

1. **공통 구분자로 토큰화** — 쉼표, 세미콜론, "and", "or", "&".
2. **채워 넣기 접두어 제거** — "find", "show me", "segment", "detect", "locate", "a", "an", "the".
3. **전치사 수식어는 시각적인 경우에만 유지** — `"striped red umbrella"`는 유지, `"umbrella from yesterday"`는 제거 (`"from yesterday"`는 이미지 내부에 있지 않습니다).
4. 선택적 `context`를 사용하여 **충돌을 모호성 해소**:
   - 감시(surveillance) 컨텍스트에서의 `"window"` -> `"building window"`.
   - 의료(medical) 컨텍스트에서의 `"window"` -> 종종 오류; 사용자에게 명확히 하도록 제안하세요.
5. **폴백**으로 분할이 개념을 0개 산출하고 발화에 구체적 명사가 하나 이상 포함된 경우 원문 문자열을 그대로 사용하세요. 구체적 명사를 추출할 수 없다면 개념을 내보내지 말고 경고만 반환하고 사용자에게 명확히 해달라고 요청하세요 (규칙 참조).
6. **`max_concepts` 상한을 적용하세요.** 발화자 요청보다 더 많은 개념이 추출된 경우, 발화 순서대로 첫 `max_concepts`개를 유지하고 나머지는 `dropped` 아래에 `"exceeded max_concepts"` 이유와 함께 내보내세요. 이렇게 하면 사용자가 긴 나열을 붙여넣을 때 지연 시간이 제한됩니다.

## 출력 형식

```
[designed prompts]
  utterance:    <original>
  concepts:     ["concept_1", "concept_2", ...]
  dropped:      ["filler_1", ...]
  warnings:     ["concept too abstract", "may match many classes", ...]

[sam3 calls]
  For each concept run: sam3.detect(image, concept)
  Merge outputs with distinct concept tags per detection.
```

## 예시

```
in:  "can you find me a cat or two dogs?"
out: ["cat", "dogs"]
dropped: ["can you find me", "a", "or two", "?"]
note: "dogs" kept plural because the utterance says "two dogs" — plural hint preserved.

in:  "segment the big red truck and the blue sedan"
out: ["big red truck", "blue sedan"]
dropped: ["segment", "the", "and"]

in:  "thing near the door"
out: ["door"]
warnings: ["'thing' is too abstract for SAM 3; fell back to 'door'"]

in:  "striped red umbrella, green hat, pink balloon"
out: ["striped red umbrella", "green hat", "pink balloon"]
```

## 규칙

- 8단어 이상인 문장을 SAM 3에 전달하지 마세요. 그 이상에서는 정확도가 떨어집니다.
- 발화에 추출 가능한 구체적 명사가 없는 경우 SAM 3을 실행하지 말고, 경고를 반환하고 명확화를 요청하세요.
- 따옴표로 묶인 문자열 내부의 구두점으로 분할하지 마세요. `"black and white cat"`가 따옴표로 묶여 있다면 하나의 개념으로 보존하세요.
- 프로덕션 디버깅을 위해 원문 발화와 파생된 개념을 항상 기록하세요.
