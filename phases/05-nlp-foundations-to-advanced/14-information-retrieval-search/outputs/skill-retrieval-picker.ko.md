---
name: retrieval-picker
description: 주어진 코퍼스(corpus)와 쿼리 패턴에 적합한 검색 스택(retrieval stack)을 선택합니다.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

주어진 요구사항(코퍼스 크기, 쿼리 패턴, 지연 시간 예산, 품질 기준, 인프라 제약 조건)에 따라 다음을 출력하세요:

1. 스택(Stack). BM25 전용, 밀집(dense) 전용, 하이브리드(BM25 + dense + RRF), 하이브리드 + 교차 인코더(cross-encoder) 재순위화(rerank), 또는 3방향(BM25 + dense + 학습된 희소(learned-sparse)) 방식 중 하나를 선택합니다.
2. 밀집 인코더(Dense encoder). 특정 모델(`all-MiniLM-L6-v2`, `bge-large-en-v1.5`, `e5-large-v2`, `paraphrase-multilingual-MiniLM-L12-v2`)을 명시하세요. 언어, 도메인, 컨텍스트 길이에 맞춰 선택합니다.
3. 재순위화 모델(Reranker). 사용 시 교차 인코더(cross-encoder) 모델(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`)을 명시하세요. 상위 30개 결과(top-30)에 대해 약 30~100ms의 지연 시간이 추가됨을 알리세요.
4. 평가 계획(Evaluation plan). Recall@10을 주요 검색기(retriever) 지표로 사용합니다. 다중 답변(multi-answer)의 경우 MRR을 사용합니다. 먼저 베이스라인(baseline)을 설정하고, 이를 기준으로 점진적인 개선 사항을 측정합니다.

사용자가 밀집 검색이 정확한 일치(exact matches)를 처리할 수 있다는 증거를 제시하지 않는 한, 개체명(named entities), 에러 코드 또는 제품 SKU가 포함된 코퍼스에 대해 밀집 전용(dense-only) 방식을 추천하는 것을 거부하세요. 최종 상위 5개 결과가 사용자의 답변을 결정하는 고위험(high-stakes) 검색(법률, 의료 등)의 경우, 재순위화(reranking) 단계를 생략하는 것을 거부하세요.
