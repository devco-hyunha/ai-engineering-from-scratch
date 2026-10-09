---
name: multilingual-picker
description: 다국어 NLP 작업을 위해 소스 언어, 대상 모델, 평가 계획을 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 18강
tags: [nlp, multilingual, cross-lingual]
---

요구 사항(대상 언어, 작업 유형, 언어별 사용 가능한 레이블 데이터)이 주어지면 다음을 출력합니다:

1. 미세 조정(Fine-tuning)용 소스 언어. 기본값은 영어입니다. 대상 언어와 유형론적으로 가까운 고자원(high-resource) 언어가 있는 경우 LANGRANK나 qWALS를 확인해 보세요.
2. 기본 모델. XLM-R (분류), mT5 (생성), NLLB (번역), Aya-23 (생성형 LLM).
3. 소수 예시(Few-Shot) 예산. 사용 가능한 경우 대상 언어 예제 100-500개로 시작합니다. 레이블링이 불가능한 경우에만 제로샷(Zero-Shot)을 사용합니다.
4. 평가 계획. 언어별 정확도(집계 지표가 아님), 언어 간 일관성, 비라틴 문자에 대한 엔티티 레벨 F1.

언어별 평가 없이 다국어 모델을 출시하는 것을 거부합니다. 집계 지표는 롱테일 실패를 숨깁니다. 토큰화 커버리지(Toknization coverage)가 낮은 문자(아마르어, 티그리냐어, 많은 아프리카 언어)는 바이트 폴백(byte-fallback)이 있는 모델(SentencePiece의 byte_fallback=True 또는 GPT-2와 같은 바이트 레벨 토크나이저)이 필요하다고 플래그를 지정합니다.
