---
name: skill-statistical-testing
description: ML 모델 비교와 실험 평가에 맞는 통계 검정을 선택한다
version: 1.0.0
phase: 1
lesson: 15
tags: [statistics, hypothesis-testing, model-comparison]
---

# ML을 위한 통계 검정 (Statistical Testing for ML)

모델을 비교하거나, A/B 실험을 돌리거나, 결과를 검증할 때 올바른 검정을 고르는 방법입니다.

## 결정 체크리스트 (Decision Checklist)

1. 무엇을 비교하는가? 평균, 비율, 분포, 상관?
2. 그룹은 몇 개인가? 일표본 vs 기준, 두 그룹, 여러 그룹?
3. 관측이 대응인가(같은 테스트 세트, 같은 폴드) 독립인가?
4. 데이터가 정규분포인가? n < 30이고 명확히 정규가 아니면 비모수를 쓴다.
5. 데이터가 연속, 순서, 범주형 중 무엇인가?
6. 검정을 몇 번 돌리는가? 하나보다 많으면 보정을 적용한다.

## 결정 트리 (Decision tree)

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

## 각 검정을 언제 쓸까 (When to use each test)

| Test | Data type | Assumptions | ML use case |
|---|---|---|---|
| Paired t-test | Continuous, paired | Normal differences | Compare 2 models on same k-fold splits |
| Wilcoxon signed-rank | Continuous/ordinal, paired | None (non-parametric) | Compare 2 models, small k (5-10 folds) |
| Welch's t-test | Continuous, independent | Roughly normal | Compare model on two separate datasets |
| Mann-Whitney U | Continuous/ordinal, independent | None | Compare latency distributions |
| ANOVA | Continuous, 3+ groups | Normal, equal variance | Compare multiple model architectures |
| Kruskal-Wallis | Continuous/ordinal, 3+ groups | None | Compare multiple models, non-normal metrics |
| Chi-squared | Categorical counts | Expected count >= 5 | Compare class distributions, confusion matrices |
| Fisher's exact | Categorical counts | Small samples | Rare event comparison |
| KS test | Continuous | None | Check if predictions follow expected distribution |
| Bootstrap CI | Any statistic | None | Confidence interval for AUC, F1, any metric |
| McNemar's test | Paired binary | None | Compare two classifiers on same test set |

## 모델 비교 레시피 (Model comparison recipe)

1. 실험을 돌리기 전에 지표와 유의수준(alpha = 0.05)을 정의한다.
2. 두 모델을 같은 k-fold 교차검증 분할에서 돌린다 (k = 5 또는 10).
3. 대응 점수를 모은다: (a_1, b_1), (a_2, b_2), ..., (a_k, b_k).
4. 차이를 계산한다: d_i = b_i - a_i.
5. 대응 검정을 돌린다 (k <= 10이면 Wilcoxon, k > 10이거나 차이가 정규면 대응 t-검정).
6. 보고: p-값, 평균 차이, 95% 신뢰구간, 효과 크기(Cohen's d).
7. p < alpha이고 효과 크기가 의미 있으면, 차이는 진짜이며 행동할 가치가 있다.

## 흔한 실수 (Common mistakes)

- 데이터가 대응인데 독립 검정을 쓴다. 두 모델이 같은 테스트 폴드에서 평가되었다면 대응 검정을 써야 한다. 독립 검정은 대응 정보를 버리고 검정력을 잃는다.
- 효과 크기 없이 p < 0.05만 보고한다. 통계적으로 유의한 0.1% 정확도 향상은 배포할 가치가 없다. 항상 Cohen's d 또는 원시 평균 차이를 계산한다.
- 서로 다른 테스트 세트에서 모델을 비교한다. 테스트 세트는 두 모델에 반드시 동일해야 한다. 다른 테스트 세트면 비교가 무의미하다.
- 20개 비교를 돌리고 본페로니 보정 없이 최고만 보고한다. alpha = 0.05에서 20개 검정이면 우연히 거짓 양성이 1개 정도 기대된다.
- 불균형 데이터에서 정확도를 쓴다. 다수 클래스 99%면 단순 분류기도 99%를 달성한다. F1, precision-recall AUC, Matthews 상관계수를 쓴다.
- 교차검증 폴드를 독립 표본처럼 취급한다. 학습 데이터를 공유하므로 독립성 가정을 위반한다. 보정된 재표본 t-검정이 이를 반영한다.

## 빠른 참고: 효과 크기 해석 (Quick reference: effect size interpretation)

| Cohen's d | Interpretation |
|---|---|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| > 1.0 | Very large effect |

| What to report | Why |
|---|---|
| p-value | Is the difference real? |
| Confidence interval | How big could the difference be? |
| Effect size (Cohen's d) | Is the difference meaningful? |
| Sample size (n or k folds) | Can we trust the result? |
