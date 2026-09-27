---
name: embedding-picker
description: 주어진 코퍼스(corpus)와 배포 환경에 적합한 임베딩 모델, 차원(dimension), 검색 모드(retrieval mode)를 선택합니다.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

코퍼스(크기, 언어, 도메인, 평균 길이), 배포 대상(클라우드 / 에지 / 온프레미스), 지연 시간 예산(latency budget), 저장 공간 예산(storage budget)이 주어졌을 때, 다음을 출력하세요:

1. 모델(Model): 명명된 체크포인트 또는 API. 한 문장으로 된 선정 이유 포함.
2. 차원(Dimension): 전체(Full) / 마트료시카 절단(Matryoshka-truncated) / int8 양자화(int8-quantized). 저장 공간 예산과 연관된 이유 포함.
3. 모드(Mode): 밀집(Dense) / 희소(Sparse) / 멀티 벡터(Multi-vector) / 하이브리드(Hybrid). 이유 포함.
4. 쿼리 접두사(Query prefix) 또는 템플릿: 모델 카드에서 요구하는 경우 작성.
5. 평가 계획(Evaluation plan): 도메인과 관련된 MTEB 태스크 + nDCG@10을 사용한 홀드아웃(held-out) 도메인 평가.

도메인 검증 없이 마트료시카(Matryoshka) 차원을 64 미만으로 축소하는 권장 사항은 거부하세요. 10k 미만의 구절(passage)을 가진 코퍼스에 대해 ColBERTv2를 권장하는 것은 거부하세요(오버헤드가 정당화되지 않음). 512 토큰 윈도우를 가진 모델로 라우팅되는 긴 문서 코퍼스(>8k 토큰)는 주의 표시(flag)를 하세요.
