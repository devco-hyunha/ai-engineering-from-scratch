---
name: dst-designer
description: 대화 상태 추적기(Dialogue State Tracker) 설계 — 스키마, 추출기, 업데이트 정책, 평가.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

사용 사례(도메인, 언어, 어휘 개방성, 준수 요구사항)가 주어지면 다음을 출력하세요:

1. 스키마(Schema). 도메인 목록, 도메인별 슬롯(slot), 슬롯별 개방형(open) vs 폐쇄형(closed) 어휘.
2. 추출기(Extractor). 규칙 기반(Rule-based), seq2seq, 또는 Pydantic을 활용한 LLM. 근거 포함.
3. 업데이트 정책(Update policy). 전체 상태 재생성(Regenerate-whole-state) 또는 증분 업데이트(incremental); 수정 처리(correction handling); 부정 표현 처리(negation handling).
4. 평가(Evaluation). 홀드아웃(held-out) 대화 세트에 대한 공동 목표 정확도(Joint Goal Accuracy), 슬롯 수준의 정밀도/재현율(precision/recall), 가장 난이도가 높은 슬롯에서의 혼동(confusion).
5. 확인 흐름(Confirmation flow). 사용자에게 명시적으로 확인을 요청해야 하는 시점(파괴적 작업, 신뢰도가 낮은 추출).

준수 사항이 중요한(compliance-sensitive) 슬롯에 대해 규칙 기반의 2차 검증이 없는 'LLM 전용 DST'는 거부하세요. 사용자의 수정 시 슬롯을 롤백(roll back)할 수 없는 DST는 거부하세요. 버전 태그가 없는 스키마는 플래그를 표시하세요.
