---
name: structured-output-picker
description: 구조화된 출력(structured output) 방식, 스키마 설계 및 검증 계획을 선택합니다.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

사용 사례(제공자, 지연 시간 예산, 스키마 복잡도, 실패 허용 범위)가 주어지면 다음을 출력하세요:

1. 메커니즘(Mechanism): 벤더 제공 네이티브 구조화된 출력(Native vendor structured output), Instructor 재시도(Instructor retries), Outlines FSM 또는 XGrammar CFG 중 하나를 선택하고, 그 이유를 한 문장으로 작성하세요.
2. 스키마 설계(Schema design): 필드 순서(추론 필드를 먼저, 답변 필드를 마지막에 배치), 'unknown'을 위한 nullable 필드, enum vs regex, 필수 필드.
3. 실패 전략(Failure strategy): 최대 재시도 횟수, 폴백 모델(fallback model), 유연한 `null` 처리, 분포 외 거절(out-of-distribution refusal) 전략.
4. 검증 계획(Validation plan): 스키마 준수율(목표 100%), 의미론적 유효성(semantic validity, LLM-judge), 필드 커버리지 비율, 지연 시간(p50/p99).

`answer` 또는 `decision` 필드를 추론(reasoning) 필드보다 앞에 배치하는 설계는 배제하세요. 스키마가 없는 단순 JSON 모드(bare JSON mode) 사용도 배제하세요. FSM 전용 라이브러리에서 사용하는 재귀적 스키마(recursive schemas)는 주의(flag)를 표시하세요.
