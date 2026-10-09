---
name: skill-constitutional-rules-engine
description: 심각도, 설명, 수정 작업 및 구조화된 diff를 포함하는 출력 제약 조건을 위한 선언적 YAML 규칙 엔진
version: 1.0.0
phase: 19
lesson: 86
tags: [safety, rules, constitutional]
---

# 헌법 규칙 엔진

헌법은 YAML 파일입니다. 각 규칙은 `name`, `severity` (low | medium | high), `applies_when` (술어), `must` (술어), `explanation` 및 선택적 `fix`를 포함합니다.

## 술어

원자적:

- `contains_regex` / `not_contains_regex`
- `starts_with_regex` / `ends_with_regex`
- `max_words` / `min_words`

합성적:

- `all_of: [...predicates]`
- `any_of: [...predicates]`
- `not_: predicate`

## 수정 작업

- `append_if_missing: <suffix>`
- `prepend_if_missing: <prefix>`
- `replace_regex: { pattern: <regex>, replacement: <text> }`

## 엔진 출력

`Engine.evaluate(text) -> EngineReport`는 `pass`의 `status`, `violation`, `not_applicable`를 포함하여 각 규칙에 대해 하나의 `RuleResult`를 반환합니다. `report.violations()`는 위반 사항으로 필터링하며 `report.max_severity()`는 존재하는 가장 심각한 심각도를 반환합니다.

## 산출물

`outputs/rules_report.json`는 각 케이스에 대해 초안, 수정본 및 구조화된 diff를 포함합니다.
