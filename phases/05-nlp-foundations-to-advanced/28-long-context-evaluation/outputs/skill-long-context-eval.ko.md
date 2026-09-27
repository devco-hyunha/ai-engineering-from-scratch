---
name: long-context-eval
description: 주어진 모델과 유스케이스에 적합한 롱 컨텍스트(long-context) 평가 세트를 설계합니다.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

대상 모델, 대상 컨텍스트 길이(context length), 그리고 유스케이스가 주어지면 다음을 출력합니다:

1. 테스트(Tests). NIAH(Needle In A Haystack) 깊이 × 길이 그리드; RULER 멀티홉(multi-hop); 커스텀 도메인 태스크.
2. 샘플링(Sampling). 각 길이별로 0, 0.25, 0.5, 0.75, 1.0의 깊이를 적용합니다.
3. 지표(Metrics). 검색 통과율(retrieval pass rate); 추론 통과율(reasoning pass rate); 첫 번째 토큰 생성 시간(time-to-first-token); 쿼리당 비용(cost-per-query).
4. 컷오프(Cutoff). 유효 검색 길이(90% 통과 기준) 및 유효 추론 길이(70% 통과 기준)를 모두 보고합니다.
5. 회귀 테스트(Regression). 고정된 하네스(harness)를 사용하여 모델이 업데이트될 때마다 재실행하고, 변화량(deltas)을 파악합니다.

모델 카드(model card)에 명시된 컨텍스트 창(context window) 수치만을 신뢰하지 마세요. 멀티홉(multi-hop) 작업에 대해 NIAH 전용 평가만을 수행하는 것을 지양하세요. 벤더(vendor)가 자체적으로 보고한 롱 컨텍스트 성능 수치를 독립적인 증거로 채택하지 마세요.
