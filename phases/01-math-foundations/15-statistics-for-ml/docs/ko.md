# 머신러닝을 위한 통계 (Statistics for Machine Learning)

> 통계는 모델이 실제로 작동하는지, 아니면 그냥 운이 좋았는지를 알려 주는 방법입니다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## 학습 목표 (Learning Objectives)

- 기술 통계, 피어슨/스피어만 상관, 공분산 행렬을 처음부터 직접 계산합니다
- 가설 검정(t-검정, 카이제곱)을 수행하고 p-값과 신뢰구간을 올바르게 해석합니다
- 부트스트랩 재표본추출로 분포 가정 없이 임의의 지표에 대한 신뢰구간을 구성합니다
- 효과 크기 척도를 사용해 통계적 유의성과 실질적 유의성을 구분합니다

## 문제 상황 (The Problem)

두 모델을 학습했습니다. 모델 A는 테스트 세트에서 0.87, 모델 B는 0.89입니다. 모델 B를 배포합니다. 3주 뒤, 프로덕션 지표가 이전보다 나빠집니다. 무슨 일이 일어난 걸까요?

모델 B는 사실 모델 A보다 뛰어나지 않았습니다. 0.02 차이는 노이즈였습니다. 테스트 세트가 너무 작았거나, 분산이 너무 컸거나, 둘 다였습니다. 개선으로 포장한 무작위성을 그대로 출시한 셈입니다.

이런 일은 끊임없이 일어납니다. Kaggle 리더보드 순위 뒤집힘. 재현에 실패하는 논문. 수백 개 샘플만으로 승자를 선언하는 A/B 테스트. 근본 원인은 항상 같습니다. 누군가 통계를 건너뛴 것입니다.

통계는 신호와 노이즈를 구분하는 도구를 줍니다. 차이가 언제 진짜인지, 얼마나 확신해야 하는지, 결과를 믿기 전에 얼마나 많은 데이터가 필요한지를 알려 줍니다. 모든 ML 파이프라인, 모든 모델 비교, 모든 실험에 통계가 필요합니다. 없으면 추측에 불과합니다.

## 핵심 개념 (The Concept)

### 기술 통계: 데이터 요약하기 (Descriptive Statistics: Summarizing Your Data)

모델을 만들기 전에 데이터가 어떻게 생겼는지 알아야 합니다. 기술 통계는 데이터셋을 그 형태를 담는 몇 개의 숫자로 압축합니다.

**중심 경향 척도(Measures of central tendency)**는 "가운데는 어디인가?"에 답합니다.

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

평균은 균형점입니다. 중앙값은 중간 지점입니다. 둘이 갈라지면 분포가 치우친 것입니다. 소득 분포는 평균 >> 중앙값입니다(억만장자로 인한 오른쪽 치우침). 학습 중 손실 분포는 종종 평균 << 중앙값입니다(쉬운 샘플로 인한 왼쪽 치우침).

**산포 척도(Measures of spread)**는 "데이터가 얼마나 퍼져 있는가?"에 답합니다.

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**백분위수(Percentiles)**는 정렬된 데이터를 100등분합니다. 25번째 백분위수(Q1)는 값의 25%가 이 지점 아래에 있다는 뜻입니다. 50번째 백분위수가 중앙값입니다. 75번째 백분위수가 Q3입니다.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

ML에서는 추론 지연시간, 예측 신뢰도 분포, 오차 분포를 이해할 때 백분위수를 봅니다. 평균 오차는 낮지만 P99 오차가 끔찍한 모델은 안전이 중요한 응용에서는 쓸모없을 수 있습니다.

**표본 vs 모집단 통계.** 표본에서 분산을 계산할 때는 n이 아니라 (n-1)로 나눕니다. 이것이 베셀 보정(Bessel's correction)입니다. 표본 평균이 진짜 모집단 평균이 아니라는 사실을 보정합니다. 분모가 n이면 진짜 분산을 체계적으로 과소추정합니다. (n-1)이면 추정이 비편향입니다.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

실무에서는 n이 크면(수천 샘플) 차이는 무시할 만합니다. n이 작으면(수십 샘플) 중요합니다.

### 상관: 변수들이 함께 움직이는 방식 (Correlation: How Variables Move Together)

상관은 두 변수 사이 선형 관계의 강도와 방향을 측정합니다.

**피어슨 상관계수(Pearson correlation coefficient)**는 선형 연관성을 측정합니다:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

피어슨은 관계가 선형이고 두 변수가 대략 정규분포라고 가정합니다. 이상치에 민감합니다. 극단값 하나면 r을 0.1에서 0.9로 끌어올릴 수 있습니다.

**스피어만 순위 상관(Spearman rank correlation)**은 단조 연관성을 측정합니다:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**각각을 언제 쓸까:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**황금 규칙:** 상관은 인과를 함의하지 않습니다. 아이스크림 판매와 익사 사망은 둘 다 여름에 늘어나서 상관됩니다. 모델 정확도와 파라미터 수는 상관되지만, 파라미터를 늘린다고 정확도가 자동으로 오르지는 않습니다(과적합 참고).

### 공분산 행렬 (Covariance Matrix)

두 변수 사이 공분산은 함께 얼마나 변하는지를 측정합니다:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

d개 특성에 대해 공분산 행렬 C는 d x d 행렬이며 C[i][j] = Cov(feature_i, feature_j)입니다. 대각 원소 C[i][i]는 각 특성의 분산입니다.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**PCA와의 연결.** PCA는 공분산 행렬을 고유분해합니다. 고유벡터가 주성분(분산이 최대인 방향)입니다. 고유값은 각 성분이 얼마나 많은 분산을 담는지를 알려 줍니다. 이것이 Lesson 10에서 다룬 내용이지만, 이제 공분산 행렬을 분해하는 것이 맞는 이유를 봅니다. 데이터의 모든 쌍별 선형 관계를 인코딩하기 때문입니다.

**상관과의 연결.** 상관 행렬은 표준화된 변수(각각을 표준편차로 나눈 것)의 공분산 행렬입니다. 상관은 공분산을 정규화해 모든 값이 [-1, 1]에 들어가게 합니다.

### 가설 검정 (Hypothesis Testing)

가설 검정은 불확실성 아래에서 결정을 내리는 틀입니다. 주장에서 시작해 데이터를 모으고, 그 데이터가 주장과 일치하는지 판단합니다.

**설정:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-값**은 H0가 참이라고 가정할 때, 관측한 만큼 극단적인 데이터를 볼 확률입니다. H0가 참일 확률이 아닙니다. 통계에서 가장 흔한 오해입니다.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**신뢰구간(Confidence intervals)**은 파라미터의 그럴듯한 값의 범위를 줍니다:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

신뢰구간의 폭은 정밀도에 대해 알려 줍니다. 넓은 구간은 높은 불확실성입니다. 좁은 구간은 추정이 정밀하다는 뜻입니다(데이터가 편향되어 있으면 정확하다는 뜻은 아닙니다).

### t-검정 (The t-test)

t-검정은 평균을 비교합니다. 여러 변형이 있습니다.

**일표본 t-검정:** 모집단 평균이 가정한 값과 다른가?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**이표본 t-검정 (독립):** 두 집단 평균이 다른가?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**대응 t-검정(Paired t-test):** 측정이 쌍으로 올 때(같은 모델을 같은 데이터 분할에서 평가):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML에서는 대응 t-검정이 흔합니다. 두 모델을 같은 10개 교차검증 폴드에서 돌리고 점수를 쌍으로 비교합니다.

### 카이제곱 검정 (Chi-squared Test)

카이제곱 검정은 관측 빈도가 기대 빈도와 맞는지를 확인합니다. 범주형 데이터에 유용합니다.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 모델을 위한 A/B 테스트 (A/B Testing for ML Models)

ML의 A/B 테스트는 웹 A/B 테스트와 같지 않습니다. 모델 비교에는 특유의 과제가 있습니다:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**절차:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 통계적 유의성 vs 실질적 유의성 (Statistical Significance vs Practical Significance)

결과가 통계적으로 유의해도 실질적으로는 무의미할 수 있습니다. 데이터가 충분하면 사소한 차이도 통계적으로 유의해집니다.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**효과 크기(Effect size)**는 표본 크기와 무관하게 차이가 얼마나 큰지를 수치화합니다:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

항상 p-값과 효과 크기를 함께 보고하세요. p-값은 차이가 진짜인지 알려 줍니다. 효과 크기는 그게 중요한지 알려 줍니다.

### 다중 비교 문제 (Multiple Comparison Problem)

가설을 많이 검정하면 우연히 "유의한" 결과가 나옵니다. alpha = 0.05에서 20개를 검정하면, 아무 것도 진짜가 아니어도 거짓 양성이 1개 정도 기대됩니다.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**본페로니 보정(Bonferroni correction):** alpha를 검정 수로 나눕니다.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML에서는 여러 지표로 모델을 비교하거나, 많은 하이퍼파라미터 설정을 시험하거나, 여러 데이터셋에서 평가할 때 중요합니다.

### 부트스트랩 방법 (Bootstrap Methods)

부트스트랩은 데이터를 복원 추출로 재표본해 통계량의 표본분포를 추정합니다. 기저 분포에 대한 가정이 필요 없습니다.

**알고리즘:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**부트스트랩 신뢰구간 (백분위수 방법):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**ML에서 부트스트랩이 중요한 이유:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**모델 비교를 위한 부트스트랩:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

분포 가정이 없으므로 대응 t-검정보다 더 견고합니다.

### 모수 vs 비모수 검정 (Parametric vs Non-parametric Tests)

**모수 검정(Parametric tests)**은 특정 분포(보통 정규)를 가정합니다:

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**비모수 검정(Non-parametric tests)**은 분포 가정이 없습니다:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**비모수를 쓸 때:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**모수를 쓸 때:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML 실험에서는 보통 n이 작습니다(교차검증 폴드 5개 또는 10개). 그래서 Wilcoxon signed-rank 같은 비모수 검정이 t-검정보다 더 적절한 경우가 많습니다.

### 중심극한정리: 실무적 함의 (Central Limit Theorem: Practical Implications)

CLT는 표본 평균의 분포가 n이 커질수록, 기저 모집단 분포와 무관하게 정규분포에 가까워진다고 말합니다.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**ML에서 중요한 이유:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT가 하지 않는 것:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 논문의 흔한 통계 실수 (Common Statistical Mistakes in ML Papers)

1. **학습 세트에서 테스트.** 과적합을 보장합니다. 학습 중 모델이 본 적 없는 데이터를 항상 남겨 두세요.

2. **신뢰구간 없음.** 불확실성 없이 정확도 숫자 하나만 보고하면 결과를 재현·검증할 수 없습니다.

3. **다중 비교 무시.** 50개 설정을 시험하고 보정 없이 최고만 보고하면 거짓 양성률이 부풀려집니다.

4. **통계적·실질적 유의성 혼동.** 0.01% 정확도 향상에 p-값 0.001은 의미가 없습니다.

5. **불균형 데이터에서 정확도 사용.** 음성 클래스 99%인 데이터셋에서 99% 정확도는 모델이 아무것도 배우 않았다는 뜻입니다. precision, recall, F1, AUC를 쓰세요.

6. **지표 선별 보고(Cherry-picking).** 모델이 이긴 지표만 보고합니다. 정직한 평가는 관련 지표를 모두 보고합니다.

7. **학습/테스트 분할 사이 정보 누출.** 분할 전에 정규화하거나, 미래 데이터로 과거를 예측합니다.

8. **분산 추정 없는 작은 테스트 세트.** 100개 샘플에서 2% 향상을 주장하는 것은 신호가 아니라 노이즈입니다.

9. **독립이 아닌 데이터에 독립성 가정.** 같은 환자의 의료 영상, 같은 문서의 여러 문장. 그룹 안 관측은 상관되어 있습니다.

10. **P-해킹.** p < 0.05가 나올 때까지 다른 검정, 부분집합, 제외 기준을 시도합니다. 결과는 탐색의 산물입니다.

## 구현하기 (Building It)

다음을 구현합니다:

1. **처음부터 기술 통계** (평균, 중앙값, 최빈값, 표준편차, 백분위수, IQR)
2. **상관 함수** (피어슨과 스피어만, 공분산 행렬 포함)
3. **가설 검정** (일표본 t-검정, 이표본 t-검정, 카이제곱 검정)
4. **부트스트랩 신뢰구간** (임의의 통계량, 가정 불필요)
5. **A/B 테스트 시뮬레이터** (데이터 생성, 검정, Type I·Type II 오류 확인)
6. **통계적 vs 실질적 유의성 데모** (큰 n이 모든 것을 "유의"하게 만듦을 보임)

전부 처음부터, `math`와 `random`만 사용합니다. numpy도 scipy도 없습니다.

```figure
f3-bootstrap-resample
```

## 핵심 용어 (Key Terms)

| Term | Definition |
|---|---|
| Mean | 값을 개수로 나눈 합. 이상치에 민감함. |
| Median | 정렬된 데이터의 가운데 값. 이상치에 강건함. |
| Standard deviation | 분산의 제곱근. 원래 단위로 산포를 측정함. |
| Percentile | 데이터 중 주어진 비율이 그 아래에 있는 값. |
| IQR | 사분위수 범위. Q3 빼기 Q1. 가운데 50%의 산포. |
| Pearson correlation | 두 변수 사이 선형 연관성을 측정. 범위 [-1, 1]. |
| Spearman correlation | 순위를 사용해 단조 연관성을 측정. |
| Covariance matrix | 모든 특성 사이 쌍별 공분산의 행렬. |
| Null hypothesis | 효과나 차이가 없다는 기본 가정. |
| p-value | 귀무가설이 참일 때 이만큼 극단적인 데이터의 확률. |
| Confidence interval | 주어진 신뢰수준에서 파라미터의 그럴듯한 값의 범위. |
| t-test | 평균이 유의하게 다른지 검정. t-분포를 사용. |
| Chi-squared test | 관측 빈도가 기대 빈도와 다른지 검정. |
| Effect size | 표본 크기와 무관한 차이의 크기. Cohen's d가 흔함. |
| Bonferroni correction | 거짓 양성을 통제하기 위해 유의 임계값을 검정 수로 나눔. |
| Bootstrap | 표본분포를 추정하기 위한 복원 재표본추출. |
| Type I error | 거짓 양성. H0가 참인데 기각. |
| Type II error | 거짓 음성. H0가 거짓인데 기각하지 못함. |
| Statistical power | 거짓인 H0를 올바르게 기각할 확률. Power = 1 - Type II 오류율. |
| Central limit theorem | 표본 크기가 커질수록 표본 평균이 정규분포로 수렴. |
| Parametric test | 데이터에 특정 분포(보통 정규)를 가정. |
| Non-parametric test | 분포 가정이 없음. 순위나 부호에 작동. |
