---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

코퍼스 설명(문서 수, 평균 길이, 도메인, 언어, 컴퓨팅 예산)이 주어지면 다음을 출력하세요:

1. 알고리즘(Algorithm). LDA / NMF / BERTopic / Top2Vec / FASTopic. 한 문장으로 된 이유.
2. 설정(Configuration). 토픽 수(문서 수의 제곱근 `~sqrt(n_docs)`에서 시작), `min_df` / `max_df` 필터, 신경망 접근 방식의 경우 임베딩 모델(embedding model).
3. 평가(Evaluation). `gensim.models.CoherenceModel`을 통한 토픽 일관성(Topic coherence, `c_v`), 토픽 다양성(topic diversity), 그리고 20개의 샘플에 대한 인간 검토(human read).
4. 조사할 실패 모드(Failure mode to probe). LDA의 경우, 불용어(stopwords)와 빈번한 용어를 흡수하는 "정크 토픽(junk topics)". BERTopic의 경우, 모호한 문서들을 삼켜버리는 -1 아웃라이어 클러스터(outlier cluster).

청킹(chunking) 전략 없이 임베딩 모델의 컨텍스트 윈도우(context window)보다 긴 문서에 대해서는 BERTopic 사용을 거부하세요. 일관성이 무너지는 매우 짧은 텍스트(트윗, 10 토큰 미만의 리뷰)에 대해서는 LDA 사용을 거부하세요. 실제 데이터에서 `n_topics`를 5 미만 또는 200 초과로 선택하는 경우, 잘못된 선택일 가능성이 높으므로 주의를 표시(flag)하세요.
