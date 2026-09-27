---
name: coref-picker
description: 상호 참조(coreference) 접근 방식, 평가 계획 및 통합 전략을 선택합니다.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

사용 사례(단일/다중 문서, 도메인, 언어)가 주어지면 다음 내용을 출력합니다:

1. 접근 방식(Approach): 규칙 기반(Rule-based), 신경망 스팬 기반(neural span-based), LLM 프롬프트 기반(LLM-prompted), 하이브리드(hybrid) 중 선택하고 그 이유를 한 문장으로 작성합니다.
2. 모델(Model): 신경망 방식을 선택한 경우, 명시된 체크포인트(checkpoint) 이름을 작성합니다.
3. 통합(Integration): 작업 순서는 다음과 같습니다: 토큰화(tokenize) $\rightarrow$ 개체명 인식(NER) $\rightarrow$ 상호 참조(coref) $\rightarrow$ 다운스트림 작업(downstream task).
4. 평가(Evaluation): 홀드아웃 세트(held-out set)에 대한 CoNLL F1(MUC + B³ + CEAF-φ4 평균) 및 20개 문서에 대한 수동 클러스터 검토(manual cluster review)를 수행합니다.

슬라이딩 윈도우 병합(sliding-window merge) 없이 2,000 토큰을 초과하는 문서에 LLM 전용 상호 참조(LLM-only coref)를 사용하는 것은 허용하지 않습니다. 언급 수준의 정밀도-재현율 보고서(mention-level precision-recall report) 없이 상호 참조를 수행하는 모든 파이프라인은 허용하지 않습니다. 인구통계학적으로 다양한 텍스트에 배포되는 성별 휴리스틱(gender-heuristic) 시스템은 주의 대상으로 표시(flag)합니다.
