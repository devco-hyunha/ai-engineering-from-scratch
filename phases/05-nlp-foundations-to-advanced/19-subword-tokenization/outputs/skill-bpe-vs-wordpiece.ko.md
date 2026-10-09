---
name: skill-bpe-vs-wordpiece
description: 주어진 코퍼스와 배포 대상에 대해 토크나이저 알고리즘, 어휘 크기, 라이브러리를 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 19강
tags: [nlp, tokenization]
---

코퍼스(크기, 언어, 도메인)와 배포 대상(처음부터 학습 / 미세 조정 / API 호환 추론)이 주어지면 다음을 출력합니다:

1. 알고리즘. BPE, Unigram, 또는 WordPiece. 한 문장 이유.
2. 라이브러리. SentencePiece, HF Tokenizers, 또는 tiktoken. 이유.
3. 어휘 크기. 가장 가까운 1k 단위로 반올림. 모델 크기와 언어 커버리지와 연결된 이유.
4. 커버리지 설정. `character_coverage`, `byte_fallback`, 특수 토큰 목록.
5. 검증 계획. 홀드아웃 세트에서의 단어당 평균 토큰 수, OOV 비율, 압축률, 왕복 디코딩 동일성.

희귀 문자 스크립트 콘텐츠가 포함된 코퍼스에 대해 문자 커버리지 <0.995인 토크나이저를 학습하는 것을 거부합니다. CI에서 동결된 `tokenizer.json` 해시 체크가 없는 어휘를 출시하는 것을 거부합니다. 어휘 크기가 16k 미만인 단일 언어 토크나이저는 사양이 불충분할 가능성이 높으므로 플래그를 지정합니다.
