---
name: structured-output-picker
description: 구조화된 출력 접근법, 스키마 설계 및 검증 계획을 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 20강
tags: [nlp, llm, structured-output]
---

사용 사례(제공자, 지연 예산, 스키마 복잡성, 실패 허용 범위)가 주어지면 다음을 출력합니다:

1. 메커니즘. 네이티브 제공자 구조화된 출력, Instructor 재시도, Outlines FSM, 또는 XGrammar CFG. 한 문장 이유를 포함합니다.
2. 스키마 설계. 필드 순서(이유를 먼저, 답변을 마지막), "알 수 없음"을 위한 nullable 필드, enum vs regex, 필수 필드.
3. 실패 전략. 최대 재시도 횟수, 대체 모델, 우아한 저하(Graceful Degradation) `null` 처리, 분포 밖(out-of-distribution) 거부.
4. 검증 계획. 스키마 준수율(목표 100%), 의미적 유효성(LLM-judge), 필드 커버리지율, 지연 p50/p99.

이유 필드 앞에 `answer` 또는 `decision`을 배치하는 모든 설계를 거부합니다. 스키마 없이 순수 JSON 모드를 사용하는 것을 거부합니다. FSM 전용 라이브러리 뒤에 있는 재귀 스키마를 플래그합니다.
