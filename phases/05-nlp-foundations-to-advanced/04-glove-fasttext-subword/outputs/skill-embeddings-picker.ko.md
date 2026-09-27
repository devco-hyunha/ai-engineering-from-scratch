---
name: skill-embeddings-picker
description: 새 언어 모델이나 텍스트 파이프라인을 위한 토큰화 방식을 고릅니다.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

작업과 데이터셋 설명이 주어지면 다음을 출력합니다:

1. 토큰화 전략(단어 수준, BPE, WordPiece, SentencePiece, 바이트 수준 BPE). 한 문장 이유.
2. 어휘 크기 목표. 영어 전용 LM: 32k. 다국어: 64k-100k. 코드: 50k-100k.
3. 정확한 학습 명령이 담긴 라이브러리 호출. 라이브러리 이름(Hugging Face `tokenizers`, `sentencepiece`). 인수를 인용.
4. 재현성 함정 하나. 토크나이저-모델 불일치는 가장 흔한 조용한 프로덕션 버그입니다. 어떤 토크나이저가 어떤 사전학습 체크포인트와 짝인지 밝히고, 바꾸지 말라고 경고하세요.

사용자가 사전학습 LLM을 파인튜닝할 때는 커스텀 토크나이저 학습을 추천하지 마세요(파인튜닝은 사전학습 토크나이저를 써야 함). 프로덕션 추론 경로에는 단어 수준 토큰화를 추천하지 마세요. 비영어 또는 다중 스크립트 코퍼스는 바이트 폴백이 있는 SentencePiece가 필요하다고 표시하세요.
