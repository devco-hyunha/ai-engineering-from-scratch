---
name: embedding-probe
description: word2vec 모델을 검사합니다. 유추·이웃 탐색·품질 진단을 실행합니다.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

학습된 단어 임베딩이 작동하는지 검증하기 위해 프로브합니다. `gensim.models.KeyedVectors` 객체와 어휘가 주어지면 다음을 실행합니다:

1. 표준 유추 테스트 세 개. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. top-1 결과와 코사인을 보고하세요.
2. 사용자가 제공한 도메인 특화 단어에 대한 최근접 이웃 테스트 다섯 개. 코사인과 함께 top-5 이웃을 출력하세요.
3. 대칭성 검사 하나. float 정밀도 내에서 `similarity(a, b) == similarity(b, a)`.
4. 퇴화 검사 하나. 어떤 임베딩의 노름이 0.01 미만이거나 100 초과이면 학습 버그입니다. 표시하세요.

유추 정확도만으로 모델이 좋다고 선언하지 마세요. 유추 벤치마크는 조작 가능하고 하류 작업으로 전이되지 않습니다. 내재적(intrinsic) 평가와 하류(downstream) 평가를 함께 권하세요.
