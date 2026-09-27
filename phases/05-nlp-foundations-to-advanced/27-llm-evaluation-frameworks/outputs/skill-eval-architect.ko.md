---
name: eval-architect
description: 보정된 판사(calibrated judge)와 CI 게이트를 포함한 LLM 평가 계획을 설계합니다.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

사용 사례(RAG / 에이전트 / 생성 작업)가 주어지면 다음을 출력하세요:

1. **지표(Metrics).** Faithfulness / relevance / context-precision / context-recall 및 기준(criteria)이 포함된 모든 커스텀 G-Eval 지표.
2. **판사 모델(Judge model).** 모델명 및 버전, 비용 대비 정확도에 대한 근거.
3. **보정(Calibration).** 수동 레이블링 데이터셋 크기, 목표 Spearman rho 값(인간과의 상관계수 > 0.7).
4. **데이터셋 버전 관리(Dataset versioning).** 태그 전략, 변경 로그(change log), 층화(stratification).
5. **CI 게이트(CI gate).** 지표별 임계값(thresholds), 회귀 윈도우 로직(regression-window logic), 하위 분위수 경고(bottom-quantile alert).

50개 이상의 인간 레이블링 예시로 테스트되지 않은 판사 모델을 사용하는 것을 거부하세요. 자기 평가(동일한 모델이 생성하고 평가하는 방식)를 거부하세요. 하위 10% 결과가 드러나지 않는 집계 중심의 보고를 거부하세요. 병렬 베이스라인 평가 없이 판사 모델을 업그레이드하는 모든 파이프라인에는 경고를 표시하세요.
