---
name: skill-statistical-testing
description: ML 모델 비교 및 실험 평가를 위한 통계 검정 선택
version: 1.0.0
phase: 1단계
lesson: 15강
tags: [통계, 가설 검정, 모델 비교]
---

# ML을 위한 통계 검정

모델 비교, A/B 실험 수행, 결과 검증 시 올바른 검정 방법을 선택하는 방법입니다.

## 의사 결정 체크리스트

1. 무엇을 비교하고 있나요? 평균, 비율, 분포, 또는 상관관계인가요?
2. 그룹이 몇 개 있나요? 단일 샘플 대 기준, 두 그룹, 또는 여러 그룹인가요?
3. 관측값이 짝을 이루고 있나요(동일한 테스트 세트, 동일한 폴드) 아니면 독립적인가요?
4. 데이터가 정규 분포를 따르나요? n < 30이고 명확히 정규 분포가 아닌 경우, 비모수 검정을 사용하세요.
5. 데이터는 연속형, 순서형, 또는 범주형인가요?
6. 검정을 몇 번 수행하고 있나요? 하나 이상인 경우 보정을 적용하세요.

## 의사 결정 트리

```text
Comparing means?
  Two groups?
    Paired (same data splits)? --> Paired t-test (or Wilcoxon signed-rank if non-normal)
    Independent? --> Welch's t-test (or Mann-Whitney U if non-normal)
  Multiple groups?
    Paired? --> Repeated measures ANOVA (or Friedman test)
    Independent? --> One-way ANOVA (or Kruskal-Wallis)

Comparing proportions?
  Two groups? --> Chi-squared test or Fisher's exact test (small n)
  Multiple groups? --> Chi-squared test

Comparing distributions?
  Is one distribution a reference? --> Kolmogorov-Smirnov test
  Are both empirical? --> Two-sample KS test

Measuring association?
  Both continuous, roughly normal? --> Pearson correlation
  Ordinal or non-normal? --> Spearman rank correlation
  Categorical x Categorical? --> Chi-squared test of independence

Running many tests?
  Apply Bonferroni correction: alpha_adjusted = alpha / number_of_tests
  Or use Holm-Bonferroni (less conservative, still controls family-wise error)
```

## 각 검정을 사용할 때

| 검정 | 데이터 유형 | 가정 | ML 사용 사례 |
|---|---|---|---|
| 짝을 이루는 t-검정 | 연속형, 짝을 이룸 | 차이값의 정규성 | 동일한 k-폴드 분할로 두 모델 비교 |
| 윌콕슨 부호 순위 검정 | 연속형/순서형, 짝을 이룸 | 없음 (비모수) | 두 모델 비교, 작은 k (5-10 폴드) |
| 웰치 t-검정 | 연속형, 독립 | 대략적인 정규성 | 두 개의 별도 데이터셋에서 모델 비교 |
| 맨-휘트니 U 검정 | 연속형/순서형, 독립 | 없음 | 지연(latency) 분포 비교 |
| ANOVA | 연속형, 3개 이상 그룹 | 정규성, 등분산 | 여러 모델 아키텍처 비교 |
| 크루스칼-월리스 검정 | 연속형/순서형, 3개 이상 그룹 | 없음 | 여러 모델 비교, 비정규 메트릭 |
| 카이제곱 검정 | 범주형 빈도 | 기대 빈도 >= 5 | 클래스 분포, 혼동 행렬 비교 |
| 피셔의 정확 검정 | 범주형 빈도 | 작은 표본 | 희귀 이벤트 비교 |
| KS 검정 | 연속형 | 없음 | 예측값이 예상 분포를 따르는지 확인 |
| 부트스트랩 신뢰 구간 | 모든 통계량 | 없음 | AUC, F1, 모든 메트릭에 대한 신뢰 구간 |
| 맥네마 검정 | 짝을 이룬 이진 데이터 | 없음 | 동일한 테스트 세트에서 두 분류기를 비교 |

## 모델 비교 레시피

1. 실험을 실행하기 전에 지표와 유의 수준(alpha = 0.05)을 정의해 보세요.
2. 두 모델을 동일한 k-폴드 교차 검증 분할(k = 5 또는 10)에서 실행해 보세요.
3. 짝을 이룬 점수를 수집해 보세요: (a_1, b_1), (a_2, b_2), ..., (a_k, b_k).
4. 차이를 계산해 보세요: d_i = b_i - a_i.
5. 짝을 이룬 검정을 실행해 보세요 (k <= 10인 경우 윌콕슨 검정, k > 10인 경우 짝을 이룬 t-검정 또는 정규 분포 차이를 사용).
6. p-값, 평균 차이, 95% 신뢰 구간, 효과 크기(Cohen's d)를 보고해 보세요.
7. p < alpha이고 효과 크기가 의미 있는 경우, 차이는 실제이며 조치할 가치가 있습니다.

## 공통 실수

- 데이터가 짝을 이루는데 독립 검정을 사용하는 경우. 두 모델이 동일한 테스트 폴드에서 평가되었다면 짝을 이룬 검정을 사용해야 합니다. 독립 검정은 짝을 버리고 통계적 검정력을 잃습니다.
- 효과 크기 없이 p < 0.05만 보고하는 경우. 통계적으로 유의미한 0.1% 정확도 향상은 배포할 가치가 없습니다. 항상 Cohen's d나 원시 평균 차이를 계산해 보세요.
- 서로 다른 테스트 세트에서 모델을 비교하는 경우. 두 모델의 테스트 세트는 반드시 동일해야 합니다. 서로 다른 테스트 세트는 비교를 무의미하게 만듭니다.
- 20번의 비교를 실행하고 보네르 보정 없이 가장 좋은 결과를 보고하는 경우. alpha = 0.05로 20번의 검정을 수행하면 우연히 1번의 오탐이 발생할 것으로 예상됩니다.
- 불균형 데이터에서 정확도를 사용하는 경우. 99% 다수 클래스인 경우, 단순한 분류기가 99%를 달성합니다. F1, 정밀도-재현율 AUC, 매튜스 상관계수를 사용해 보세요.
- 교차 검증 폴드를 독립 표본으로 취급하는 경우. 이들은 훈련 데이터를 공유하므로 독립성 가정을 위반합니다. 보정된 재샘플링 t-검정은 이를 고려합니다.

## 빠른 참조: 효과 크기 해석

| Cohen's d | 해석 |
|---|---|
| 0.2 | 작은 효과 |
| 0.5 | 중간 효과 |
| 0.8 | 큰 효과 |
| > 1.0 | 매우 큰 효과 |

| 보고할 내용 | 이유 |
|---|---|
| p-값 | 차이가 실제인가요? |
| 신뢰 구간 | 차이가 얼마나 클 수 있는가? |
| 효과 크기 (Cohen's d) | 차이가 의미 있는가? |
| 표본 크기 (n 또는 k 폴드) | 결과를 신뢰할 수 있는가? |
