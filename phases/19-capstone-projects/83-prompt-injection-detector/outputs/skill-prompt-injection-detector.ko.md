---
name: skill-prompt-injection-detector
description: 모든 프롬프트에 대해 범주와 신뢰도를 반환하는 계층적 탐지 파이프라인으로, 측정 가능한 정밀도 및 재현율을 제공합니다
version: 1.0.0
phase: 19
lesson: 83
tags: [safety, detector, prompt-injection]
---

# 프롬프트 인젝션 탐지기

여기서 탐지기는 프롬프트를 판정으로 매핑하는 함수입니다. 판정은 83강의 분류 체계에서 범주와 [0, 1] 범위의 신뢰도를 포함합니다.

## 파이프라인

1. 정규화 - 제로 폭 문자 제거, 동형 문자 복원, base64/hex 디코딩, leet-speak 숫자 변환, 공통 단어 Sanity Check를 수행하며 rot13 시도.
2. 부분 문자열 규칙 - `ignore previous`, `from now on you are`, `decode this base64`와 같은 수동 작성 니들.
3. 정규식 규칙 - `\bignor\w*\s+(all|prior|previous|earlier)\b`와 같은 토큰 수준 패턴.

집계는 범주별 최대 점수를 유지하며, 가장 큰 점수를 가진 범주를 반환하거나 아무것도 발동하지 않으면 `benign`을 반환합니다.

## 규칙 추가

`code/rules.py`을 편집하세요. 규칙은 `name`, `category` (6개 분류 체계 범주 중 하나), `score` (0에서 1까지의 float), `substring` 또는 `regex` 중 하나를 포함하는 사전입니다. `main.py`을 다시 실행하여 범주별 정밀도 및 재현율에 미치는 영향을 확인하세요.

## 산출물

`outputs/detector_report.json`은 범주별 지표 파일입니다. 87강의 엔드 투 엔드 게이트는 이를 읽어 신뢰도를 임계값으로 설정합니다.
