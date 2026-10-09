---
name: embedding-probe
description: word2vec 모델을 검사합니다. 유추 테스트, 이웃 찾기, 품질 진단을 수행합니다.
version: 1.0.0
phase: 5단계
lesson: 03강
tags: [nlp, embeddings, debugging]
---

학습된 임베딩(Embedding)이 정상적으로 작동하는지 확인하기 위해 임베딩을 탐지합니다. `gensim.models.KeyedVectors` 객체와 어휘(Vocabulary)가 주어지면 다음을 실행합니다:

1. 세 가지 표준 유추 테스트. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. 상위 1개 결과와 그 코사인 유사도(Cosine Similarity)를 보고합니다.
2. 사용자가 제공하는 도메인 특화 단어에 대해 다섯 가지 최근접 이웃 테스트를 수행합니다. 코사인 유사도와 함께 상위 5개 이웃을 출력합니다.
3. 대칭성 체크 한 번. `similarity(a, b) == similarity(b, a)`가 부동 소수점 정밀도 범위 내에서 일치하는지 확인합니다.
4. 퇴화(degenerate) 체크 한 번. 임베딩의 노름(norm)이 0.01 미만이거나 100 이상이면 모델에 학습 버그가 있습니다. 이를 플래그로 표시합니다.

유추 정확도만으로 모델이 좋다고 선언하는 것을 거부합니다. 유추 벤치마크는 조작이 가능하며 다운스트림 작업으로 전이되지 않습니다. 내재적 평가와 다운스트림 평가를 함께 권장합니다.
