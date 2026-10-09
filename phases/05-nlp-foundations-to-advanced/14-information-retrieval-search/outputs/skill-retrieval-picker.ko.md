---
name: retrieval-picker
description: 주어진 코퍼스와 쿼리 패턴에 맞는 검색 스택을 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 14강
tags: [nlp, retrieval, rag, search]
---

요구 사항(코퍼스 크기, 쿼리 패턴, 지연 시간 예산, 품질 기준, 인프라 제약)이 주어지면 다음을 출력합니다:

1. 스택. BM25 전용, 밀집(dense) 전용, 하이브리드(BM25 + 밀집 + RRF), 하이브리드 + 교차 인코더 리랭크, 또는 3-way(BM25 + 밀집 + 학습된 희소)입니다.
2. 밀집 인코더. 특정 모델(`all-MiniLM-L6-v2`, `bge-large-en-v1.5`, `e5-large-v2`, `paraphrase-multilingual-MiniLM-L12-v2`)을 지정합니다. 언어, 도메인, 컨텍스트 길이에 맞춰 선택합니다.
3. 리랭커. 교차 인코더 모델을 사용하는 경우 지정합니다(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`). 상위 30개 항목에 대해 약 30-100ms의 지연 시간이 추가됨을 표시합니다.
4. 평가 계획. Recall@10이 주요 검색기 지표입니다. 다중 답변의 경우 MRR을 사용합니다. 먼저 기준선을 설정하고, 점진적 개선은 기준선 대비 측정합니다.

고유 명칭, 오류 코드, 제품 SKU가 포함된 코퍼스에 대해 사용자가 밀집 검색이 정확 일치(exact match)를 처리한다는 증거가 없는 한 밀집 전용을 추천하지 마세요. 최종 상위 5개가 사용자의 답변을 결정하는 고위험 검색(법률, 의료)에서는 리랭킹을 생략하지 마세요.
