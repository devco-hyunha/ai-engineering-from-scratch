---
name: chunker
description: 주어진 코퍼스(corpus)와 쿼리 분포(query distribution)에 적합한 청킹 전략, 크기 및 오버랩(overlap)을 선택합니다.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

코퍼스(문서 유형, 평균 길이, 도메인)와 쿼리 분포(factoid / analytical / multi-hop)가 주어지면, 다음을 출력하세요:

1. 전략(Strategy): Recursive / sentence / semantic / parent-document / late / contextual 중 선택 및 근거(Reason) 제시.
2. 청크 크기(Chunk size): 토큰 수(Token count) 및 쿼리 유형과 연관된 근거 제시.
3. 오버랩(Overlap): 기본값은 0입니다. 0보다 클 경우 그 이유를 명시하세요.
4. 최소/최대 제한(Min/max enforcement): `min_tokens`, `max_tokens` 가드(guards) 설정.
5. 평가 계획(Evaluation plan): 50개의 쿼리로 구성된 층화 추출 평가 세트(factoid, analytical, multi-hop)에 대한 Recall@5.

최소/최대 청크 크기 제한이 없는 청킹 전략은 거부하세요. 성능 향상을 입증하는 어블레이션(ablation) 결과 없이 20%를 초과하는 오버랩은 거부하세요. 최소 토큰 하한선(min-token floor)이 없는 시맨틱 청킹(semantic chunking) 권장 사항은 플래그(flag)를 표시하세요.
