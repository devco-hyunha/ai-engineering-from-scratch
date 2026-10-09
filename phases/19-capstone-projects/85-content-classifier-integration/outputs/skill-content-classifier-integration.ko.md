---
name: skill-content-classifier-integration
description: 단일 심각도 라우터 뒤에 있는 세 개의 출력 측 분류기(유해성, PII, 지시문 유출)가 차단, 마스킹, 경고, 기록 동작을 수행합니다
version: 1.0.0
phase: 19
lesson: 85
tags: [safety, classifier, output-filter]
---

# 콘텐츠 분류기 통합

세 개의 분류기, 하나의 라우터, 네 가지 동작입니다.

## 판정 구조

```text
ClassifierVerdict
  name: str
  severity: none | low | medium | high
  score: float in [0, 1]
  findings: list[str]
```

## 동작 표

| 심각도 | 동작 | 효과 |
|---|---|---|
| 높음 | 차단 | 출력이 정책 거부 응답으로 대체됩니다 |
| 중간 | 마스킹 | 분류기별 마스커가 순서대로 적용됩니다 |
| 낮음 | 경고 | 출력에 부드러운 고지가 추가되어 전송됩니다 |
| 없음 | 기록 | 출력이 변경되지 않은 상태로 전송되며 판정이 기록됩니다 |

## 분류기별 동작

- 유해성 - 공백 경계와 작은 좌측 윈도우 부정 확인을 사용하는 괴롭힘 용어; `[redacted-language]`로 마스킹합니다
- PII - 이메일, 전화, SSN, Luhn 검증 카드, IPv4; SSN과 카드의 경우 심각도가 상승합니다; 각 형태를 태그로 마스킹합니다
- 지시문 유출 - 알려진 시스템 프롬프트와 트라이그램 코사인 유사도 비교; 심각도가 겹치는 정도에 따라 스케일링됩니다; 첫 번째 시스템 프롬프트 라인을 마스킹합니다

## 산출물

`outputs/classifier_report.json`는 각 케이스에 대해 동작 동사, 심각도, 마스킹된 출력, 전체 판정 목록을 포함합니다.
