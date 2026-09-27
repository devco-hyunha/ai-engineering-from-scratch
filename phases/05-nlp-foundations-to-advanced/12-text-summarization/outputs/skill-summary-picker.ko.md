---
name: summary-picker
description: 추출적(extractive) 또는 생성적(abstractive) 방식을 선택하고, 라이브러리 이름을 지정하며, 사실성 검증(factuality check)을 추가하세요.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

주어진 작업(문서 유형, 규정 준수 요구 사항, 길이, 컴퓨팅 예산)에 대해 다음을 출력하세요:

1. 접근 방식(Approach). 추출적(extractive) 또는 생성적(abstractive) 방식 중 하나를 선택하세요. 그 이유를 한 문장으로 설명하세요.
2. 시작 모델 / 라이브러리(Starting model / library). 이름을 지정하세요. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed` 또는 LLM 프롬프트를 사용하세요.
3. 평가 계획(Evaluation plan). ROUGE-1, ROUGE-2, ROUGE-L (`rouge-score`를 어간 추출(stemming)과 함께 사용). 생성적 방식인 경우 사실성 검증(factuality check)을 추가하세요.
4. 조사할 하나의 실패 모드(One failure mode to probe). 생성적 뉴스 요약에서 가장 흔한 것은 개체 교체(entity swap)입니다. 소스(source)의 개체명이 요약문에 나타나지 않는 샘플을 표시하세요.

사실성 게이트(factuality gate)가 없는 경우 의료, 법률, 금융 또는 규제 대상 콘텐츠에 대한 생성적(abstractive) 요약을 거부하세요. 모델의 컨텍스트 창(context window)을 초과하는 입력은 단순히 절단(truncation)하는 것이 아니라, 청크 단위의 맵-리듀스(chunked map-reduce) 요약이 필요함을 표시하세요.
