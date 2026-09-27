---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

요구 사항(코퍼스 크기, 질문 유형, 사실성 제약, 지연 시간 예산)이 주어지면 다음을 출력하세요:

1. 아키텍처(Architecture). 추출형(Extractive), 추출형 리더를 사용하는 RAG, 생성형 리더를 사용하는 RAG, 또는 폐쇄형(closed-book) LLM 중 하나를 선택하세요. 한 문장으로 이유를 설명하세요.
2. 검색기(Retriever). 없음, BM25, 밀집(dense) 방식(`all-MiniLM-L6-v2`와 같은 인코더 명시), 또는 하이브리드 방식 중 하나를 선택하세요.
3. 리더(Reader). SQuAD로 튜닝된 모델(`deepset/roberta-base-squad2`), 특정 LLM 이름, 또는 도메인 미세 조정(fine-tuned)된 DistilBERT 중 하나를 선택하세요.
4. 평가(Evaluation). 추출형 벤치마크의 경우 EM + F1을 사용하고, 프로덕션 환경의 경우 답변 정확도(answer accuracy) + 인용 정확도(citation accuracy) + 거절 보정(refusal calibration)을 사용하세요. 무엇을 어떻게 측정하는지 명시하세요.

규제 또는 컴플라이언스에 민감한 질문에 대해서는 폐쇄형(closed-book) LLM 답변을 거부하세요. 검색 재현율(retrieval-recall) 베이스라인이 없는 QA 시스템은 거부하세요(검색기가 올바른 구절을 찾아냈는지 알지 못하면 리더를 평가할 수 없습니다). 멀티홉 추론(multi-hop reasoning)이 필요한 질문은 HotpotQA로 학습된 시스템과 같은 전문적인 멀티홉 검색기가 필요하다고 표시하세요.
