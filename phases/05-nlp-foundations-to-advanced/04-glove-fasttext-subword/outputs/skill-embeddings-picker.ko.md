---
name: skill-embeddings-picker
description: 새로운 언어 모델이나 텍스트 파이프라인에 대한 토큰화 접근 방식을 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 04강
tags: [nlp, tokenization, embeddings]
---

작업 및 데이터셋 설명이 주어지면 다음을 출력합니다:

1. 토큰화 전략 (word-level, BPE, WordPiece, SentencePiece, byte-level BPE). 한 문장 이유를 포함합니다.
2. 어휘 크기 목표. 영어 전용 LLM: 32k. 다국어: 64k-100k. 코드: 50k-100k.
3. 정확한 학습 명령이 포함된 라이브러리 호출. 라이브러리 이름 (Hugging Face `tokenizers`, `sentencepiece`)을 명시하고 인수를 인용합니다.
4. 재현성 관련 함정 하나. 토크나이저-모델 불일치는 가장 흔한 조용한 프로덕션 버그입니다. 어떤 토크나이저가 어떤 사전 학습된 체크포인트와 짝을 이루는지 명시하고, 교체하지 않도록 경고합니다.

사용자가 사전 학습된 LLM을 미세 조정할 때 사용자 정의 토크나이저 학습을 추천하지 마세요 (미세 조정은 사전 학습된 토크나이저를 사용해야 합니다). 프로덕션 추론 경로에 대해 word-level 토큰화를 추천하지 마세요. 비영어권이나 다중 스크립트 코퍼스는 바이트 폴백이 있는 SentencePiece가 필요함을 표시합니다.
