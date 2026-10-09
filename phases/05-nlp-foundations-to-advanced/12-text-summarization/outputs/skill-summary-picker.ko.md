---
name: summary-picker
description: 추출적 또는 생성적 요약 방식을 선택하고, 라이브러리를 지정하며, 사실성 검사를 추가합니다.
version: 1.0.0
phase: 5단계
lesson: 12강
tags: [nlp, summarization]
---

작업(문서 유형, 컴플라이언스 요구 사항, 길이, 컴퓨팅 예산)이 주어지면 다음을 출력합니다:

1. 접근 방식. 추출적 또는 생성적 요약. 한 문장으로 이유를 설명합니다.
2. 시작 모델 / 라이브러리. 이름을 지정합니다. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, 또는 LLM 프롬프트를 사용합니다.
3. 평가 계획. ROUGE-1, ROUGE-2, ROUGE-L (어간 추출(stemming)과 함께 `rouge-score`를 사용). 생성적 요약인 경우 사실성 검사도 포함합니다.
4. 조사할 하나의 실패 모드. 생성적 뉴스 요약에서 가장 흔한 것은 엔티티 교체입니다. 원본 엔티티가 요약에 나타나지 않는 샘플을 플래그합니다.

의료, 법률, 금융, 규제 대상 콘텐츠에 대해 사실성 게이트(factuality gate) 없이 생성적 요약(summarization)을 거부합니다. 모델의 컨텍스트 윈도우(context window)를 초과하는 입력은 단순한 잘라내기(truncation)가 아닌 청킹 맵-리듀스(chunked map-reduce) 요약이 필요함을 플래그합니다.
