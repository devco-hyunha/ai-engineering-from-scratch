---
name: skill-concept-prompt-designer
description: 분할·중의성 해소·폴백으로 사용자 발화를 잘 형성된 SAM 3 개념 프롬프트로 바꿉니다
version: 1.0.0
phase: 4
lesson: 24
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# 개념 프롬프트 디자이너 (Concept Prompt Designer)

SAM 3의 정확도는 개념 프롬프트 문구에 크게 의존합니다. 이 스킬은 자유형 사용자 발화를 SAM 3가 잘 다루는 프롬프트로 정규화합니다.

## 언제 쓰나요 (When to use)

- 자연어 객체 쿼리를 받는 UI를 만들 때.
- 업스트림 호출자가 문장을 보내는 API로 SAM 3를 노출할 때.
- 나쁜 SAM 3 매치를 디버깅할 때 — 종종 모델이 아니라 프롬프트가 잘못됨.

## 입력 (Inputs)

- `utterance`: 원시 사용자 문자열.
- `context`: 선택적 도메인 힌트(예: "surveillance", "medical", "retail").
- `max_concepts`: 발화당 추출할 최대 개념 수; 기본 5.

## SAM 3가 선호하는 규칙 (Rules SAM 3 prefers)

- **문장이 아니라 짧은 명사구.** `"cat"`이 `"there is a cat"`보다 낫습니다.
- **구체적 명사.** `"skateboard"`이 `"thing to ride on"`보다 낫습니다.
- **수식어는 명사 바로 앞.** `"red car"`이 `"car that is red"`보다 낫습니다.
- **소문자.** SAM 3는 견고하지만 경험적으로 소문자 입력이 약간 낫습니다.
- **단수 또는 복수.** 둘 다 동작; 여러 인스턴스가 기대되면 복수가 도움.

## 단계 (Steps)

1. **흔한 구분자로 토큰화** — 쉼표, 세미콜론, "and", "or", "&".
2. **필러 접두어 제거** — "find", "show me", "segment", "detect", "locate", "a", "an", "the".
3. **전치사 수식어는 시각적일 때만 유지** — `"striped red umbrella"` 예, `"umbrella from yesterday"` 아니오(`"from yesterday"`는 이미지에 없음).
4. **선택적 `context`로 충돌 중의성 해소**:
   - 감시 맥락의 `"window"` -> `"building window"`.
   - 의료 맥락의 `"window"` -> 종종 오류; 사용자에게 명확히 하라고 제안.
5. **폴백** — 분할이 0개 개념을 내고 *그리고* 발화에 구체적 명사가 하나라도 있으면 원문 그대로. 구체적 명사를 추출할 수 없으면 개념을 내보내지 말고 경고만 반환하고 사용자에게 명확히 하라고 요청(규칙 참고).
6. **`max_concepts`로 상한.** 호출자가 요청한 것보다 더 많은 개념이 추출되면, 발화 순서의 처음 `max_concepts`개를 유지하고 나머지를 `"exceeded max_concepts"` 이유로 `dropped`에 넣습니다. 사용자가 긴 열거를 붙여넣을 때 지연을 제한합니다.

## 출력 형식 (Output format)

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

## 예시 (Examples)

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

## 규칙 (Rules)

- 8단어를 넘는 문장을 SAM 3에 절대 넘기지 마세요 — 그 이상에서 정확도가 떨어집니다.
- 발화에 추출 가능한 구체적 명사가 없으면 SAM 3를 돌리지 말고; 경고를 반환하고 명확화를 요청하세요.
- 따옴표 안 구두점으로 나누지 마세요; `"black and white cat"`이 따옴표로 묶이면 하나의 개념으로 보존하세요.
- 프로덕션 디버깅을 위해 원 발화와 유도된 개념을 항상 로깅하세요.
