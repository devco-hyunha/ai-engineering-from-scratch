---
name: lm-baseline
description: 신경망 언어 모델(neural LM)을 학습하기 전에 재현 가능한 n-gram 언어 모델 베이스라인을 구축합니다.
phase: 5
lesson: 16
---

코퍼스와 대상 용도(다음 단어 예측, 리스코어링, 퍼플렉서티 베이스라인)가 주어지면, 다음 사항을 출력합니다:

1. N-gram 차수(order). 일반적인 영어의 경우 Trigram, 코퍼스가 크면 4-gram, 음성 리스코어링(speech rescoring)의 경우 5-gram을 사용합니다.
2. 스무딩(Smoothing). Modified Kneser-Ney가 기본값이며, Laplace는 교육용으로만 사용합니다.
3. 라이브러리. 프로덕션용으로는 `kenlm`, 교육용으로는 `nltk.lm`을 사용하며, 수학적 원리를 배우기 위해서만 직접 구현해 보세요.
4. 평가. 훈련 세트와 테스트 세트 간에 일관된 토큰화(tokenization)를 적용한 홀드아웃 퍼플렉서티(held-out perplexity)를 측정합니다.

비교 대상 시스템 간에 서로 다른 토큰화를 사용하여 계산된 퍼플렉서티는 보고하지 마세요. 퍼플렉서티 수치는 동일한 토큰화 조건에서만 비교 가능합니다. 테스트 세트의 OOV(out-of-vocabulary) 비율을 표시하세요. 훈련 중에 특별한 `<UNK>` 토큰을 예약하지 않으면 KN은 OOV를 제대로 처리하지 못합니다.
