---
name: skill-bpe-vs-wordpiece
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

주어진 코퍼스(크기, 언어, 도메인)와 배포 대상(처음부터 학습 / 파인튜닝 / API 호환 추론)에 대해 다음을 출력하세요:

1. 알고리즘: BPE, Unigram 또는 WordPiece. 이유를 한 문장으로 제시.
2. 라이브러리: SentencePiece, HF Tokenizers 또는 tiktoken. 이유 제시.
3. 어휘 사전 크기: 가장 가까운 1k 단위로 반올림. 모델 크기 및 언어 커버리지와 연계된 이유 제시.
4. 커버리지 설정: `character_coverage`, `byte_fallback`, 특수 토큰 목록.
5. 검증 계획: 홀드아웃 세트에서의 단어당 평균 토큰 수, OOV 비율, 압축률, 라운드트립 디코딩(round-trip decode) 일치 여부.

희귀 문자 체계(rare-script) 콘텐츠가 포함된 코퍼스에 대해 `character_coverage`가 0.995 미만인 토크나이저 학습은 거부하세요. CI에서 고정된 `tokenizer.json` 해시 검증 없이 어휘 사전을 배포하는 것도 거부하세요. 어휘 사전 크기가 16k 미만인 단일 언어 토크나이저는 사양 미달(under-spec) 가능성이 높으므로 플래그를 지정하세요.
