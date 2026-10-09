# 머신러닝을 위한 통계

> 통계는 모델이 실제로 작동하는지, 아니면 단순히 운이 좋았는지 알려주는 방법입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 1단계, 06강 (확률과 분포), 07강 (베이즈 정리)
**시간:** 약 120분

## 학습 목표

- 기초부터 시작하여 기술통계, 피어슨/스피어만 상관계수, 공분산 행렬을 계산해 보세요
- 가설 검정(t-검정, 카이제곱 검정)을 수행하고 p-값과 신뢰 구간을 올바르게 해석해 보세요
- 부트스트랩 리샘플링을 사용하여 분포 가정을 하지 않고도 모든 지표에 대한 신뢰 구간을 구성해 보세요
- 효과 크기 측정치를 사용하여 통계적 유의성과 실무적 유의성을 구분해 보세요

## 문제점

두 개의 모델을 학습했습니다. 모델 A는 테스트 세트에서 0.87의 점수를 받았습니다. 모델 B는 0.89의 점수를 받았습니다. 모델 B를 배포했습니다. 3주 후, 프로덕션 지표가 이전보다 나빠졌습니다. 무슨 일이 일어난 걸까요?

모델 B는 실제로 모델 A보다 성능이 뛰어나지 않았습니다. 0.02의 차이는 잡음(noise)이었습니다. 테스트 세트가 너무 작았거나, 분산이 너무 높았거나, 둘 다 해당했을 수 있습니다. 개선된 것처럼 꾸민 랜덤성을 배포한 셈입니다.

이런 일은 constantly(항상) 발생합니다. Kaggle 리더보드 순위 변동, 재현에 실패하는 논문, 몇백 개의 샘플만으로 승자를 선언하는 A/B 테스트. 근본 원인은 항상 동일합니다: 누군가가 통계를 건너뛰었습니다.

통계는 신호(signal)와 잡음(noise)을 구분하는 도구를 제공합니다. 차이가 실제인지, 얼마나 확신해야 하는지, 결과를 신뢰하려면 얼마나 많은 데이터가 필요한지 알려줍니다. 모든 ML 파이프라인, 모든 모델 비교, 모든 실험에는 통계가 필요합니다. 통계가 없으면 추측만 하고 있는 것입니다.

## 개념

### 기술통계: 데이터 요약하기

무엇이든 모델링하기 전에, 데이터가 어떤 모습인지 알아야 합니다. 기술통계는 데이터셋을 그 형태를 포착하는 몇 개의 숫자로 압축합니다.

**중앙 경향성 측정**은 "중간값이 어디에 있는가?"에 대한 답을 제공합니다.

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

평균은 균형점입니다. 중앙값은 중간 지점입니다. 두 값이 갈라지면 분포가 왜곡된 것입니다. 소득 분포는 평균 >> 중앙값 (억만장자로 인한 오른쪽 왜곡)을 가집니다. 학습 중 손실 분포는 종종 평균 << 중앙값 (쉬운 샘플로 인한 왼쪽 왜곡)을 가집니다.

**산포 측정**은 "데이터가 얼마나 흩어져 있는가?"에 대한 답을 제공합니다.

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

**백분위수**는 정렬된 데이터를 100개의 동일한 부분으로 나눕니다. 25번째 백분위수(Q1)는 값의 25%가 이 점 아래에 위치함을 의미합니다. 50번째 백분위수는 중앙값입니다. 75번째 백분위수는 Q3입니다.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

ML에서는 추론 지연, 예측 신뢰도 분포, 오차 분포를 이해하기 위해 백분위수를 고려합니다. 평균 오차는 낮지만 P99 오차가 심각한 모델은 안전이 중요한 애플리케이션에서는 쓸모없을 수 있습니다.

**표본 대 모집단 통계.** 표본에서 분산을 계산할 때는 n 대신 (n-1)으로 나누세요. 이는 베셀 보정(Bessel's correction)입니다. 이는 표본 평균이 실제 모집단 평균이 아니라는 사실을 보정합니다. 분모에 n을 사용하면 실제 분산을 체계적으로 과소평가합니다. (n-1)을 사용하면 추정치가 편향되지 않습니다.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

실제로는: n이 크다면 (수천 개의 샘플), 차이는 무시할 수 있습니다. n이 작다면 (몇십 개의 샘플), 차이가 중요합니다.

### 상관관계: 변수가 함께 움직이는 방식

상관관계는 두 변수 간의 선형 관계의 강도와 방향을 측정합니다.

**피어슨 상관계수**는 선형 연관성을 측정합니다:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

피어슨은 관계가 선형이며 두 변수가 대략적으로 정규 분포를 따른다고 가정합니다. 이상치에 민감합니다. 하나의 극단적인 점이 r을 0.1에서 0.9로 끌어당길 수 있습니다.

**스피어만 순위 상관계수**는 단조 연관성을 측정합니다:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**각각을 사용할 때:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**황금률:** 상관관계는 인과관계를 의미하지 않습니다. 아이스크림 판매량과 익사 사망률은 여름에 둘 다 증가하기 때문에 상관관계가 있습니다. 모델의 정확도와 매개변수 개수는 상관관계가 있지만, 매개변수를 추가한다고 해서 정확도가 자동으로 개선되지는 않습니다 (참조: 과적합).

### 공분산 행렬

두 변수 간의 공분산은 변수가 함께 변하는 정도를 측정합니다:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

d개의 특성에 대해, 공분산 행렬 C는 d x d 크기의 행렬이며, C[i][j] = Cov(feature_i, feature_j)입니다. 대각선 요소 C[i][i]는 각 특성의 분산입니다.

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

**PCA와의 연결.** PCA는 공분산 행렬의 고유분해를 수행합니다. 고유벡터는 주성분(최대 분산을 가진 방향)이며, 고유값은 각 성분이 포착하는 분산의 양을 나타냅니다. 이는 10강에서 다루었던 내용과 정확히 일치하지만, 이제 공분산 행렬이 왜 분해해야 할 대상인지 이해할 수 있습니다: 공분산 행렬은 데이터 내의 모든 쌍별 선형 관계를 인코딩하기 때문입니다.

**상관관계와의 연결.** 상관 행렬은 표준화된 변수(각 변수를 표준 편차로 나눈 값)의 공분산 행렬입니다. 상관은 공분산을 정규화하여 모든 값이 [-1, 1] 범위에 포함되도록 합니다.

### 가설 검정

가설 검정은 불확실성 하에서 의사결정을 내리기 위한 프레임워크입니다. 주장을 시작점으로 삼아 데이터를 수집하고, 데이터가 그 주장과 일치하는지 판단합니다.

**설정:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-값**은 H0가 참이라고 가정할 때, 관측한 데이터만큼 극단적인 데이터를 볼 확률입니다. H0가 참일 확률이 **아닙니다**. 이는 통계에서 가장 흔한 오해입니다.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**신뢰 구간**은 매개변수에 대해 합리적인 값의 범위를 제공합니다:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

신뢰 구간의 폭은 정밀도를 나타냅니다. 넓은 구간은 높은 불확실성을 의미하며, 좁은 구간은 추정치가 정밀하다는 것을 의미합니다(단, 데이터에 편향이 있다면 정확하지는 않을 수 있습니다).

### t-검정

t-검정은 평균을 비교합니다. 여러 변형이 있습니다.

**단일 표본 t-검정:** 모집단 평균이 가설적으로 설정된 값과 다른가요?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**두 표본 t-검정 (독립):** 두 그룹의 평균이 다른가요?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**짝을 이루는 t-검정:** 측정값이 짝을 이루는 경우(동일한 데이터 분할에 대해 동일한 모델을 평가하는 경우):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML에서는 짝을 이루는 t-검정이 흔합니다. 두 모델을 동일한 10개의 교차 검증 폴드에서 실행하고 점수를 쌍별로 비교합니다.

### 카이스퀘어 검정

카이스퀘어 검정은 관측 빈도가 기대 빈도와 일치하는지 확인합니다. 범주형 데이터에 유용합니다.

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

### ML 모델을 위한 A/B 테스트

ML에서의 A/B 테스트는 웹 A/B 테스트와 다릅니다. 모델 비교에는 고유한 과제가 있습니다:

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

### 통계적 유의성 vs 실용적 유의성

결과는 통계적으로 유의미할 수 있지만, 실용적으로는 의미가 없을 수 있습니다. 충분한 데이터가 있으면, 사소한 차이도 통계적으로 유의미해집니다.

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

**효과 크기**는 표본 크기와 무관하게 차이가 얼마나 큰지 정량화합니다:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

p-값과 효과 크기를 항상 함께 보고하세요. p-값은 차이가 실제인지 알려주며, 효과 크기는 그 차이가 중요한지 알려줍니다.

### 다중 비교 문제

많은 가설을 테스트하면, 일부는 우연히 "유의미한" 결과가 나올 수 있습니다. alpha = 0.05로 20가지를 테스트하면, 아무것도 실제가 아니더라도 1개의 거짓 양성이 기대됩니다.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 보정:** alpha를 테스트 수로 나누세요.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML에서는 모델을 여러 지표로 비교하거나, 많은 하이퍼파라미터 구성을 테스트하거나, 여러 데이터셋에서 평가할 때 이 문제가 중요합니다.

### 부트스트랩 방법

부트스트랩은 데이터를 복원 추출(resampling with replacement)하여 통계량의 표본 분포를 추정합니다. 기본 분포에 대한 가정이 필요하지 않습니다.

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

**부트스트랩 신뢰 구간 (백분율 방법):**

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

이 방법은 분포 가정을 하지 않기 때문에 짝 t-검정보다 더 견고합니다.

### 모수 검정 vs 비모수 검정

**모수 검정**은 특정 분포 (보통 정규 분포)를 가정합니다:

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**비모수 검정**은 분포 가정을 하지 않습니다:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**비모수 검정을 사용할 때:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**모수 검정을 사용할 때:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML 실험에서는 보통 n이 작습니다 (5 또는 10개의 교차 검증 폴드). 따라서 t-검정보다 Wilcoxon 부호 순위 검정 같은 비모수 검정이 더 적절합니다.

### 중심 극한 정리: 실용적 함의

CLT는 표본 평균의 분포가 n이 커짐에 따라 정규 분포에 가까워진다고 설명합니다. 이는 모집단의 분포와 무관합니다.

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

### ML 논문에서 흔한 통계적 실수

1. **학습 세트에서 테스트하기.** 과적합을 보장합니다. 모델이 학습 중에 보지 못한 데이터를 항상 분리해 두세요.

2. **신뢰 구간이 없음.** 불확실성 없이 단일 정확도 수치만 보고하면 결과가 재현 불가능하고 검증할 수 없습니다.

3. **다중 비교를 무시하기.** 50가지 구성을 테스트하고 보정 없이 최상의 결과를 보고하면 오탐율이 높아집니다.

4. **통계적 유의성과 실질적 유의성을 혼동하기.** 0.01%의 정확도 개선에 대해 p-값이 0.001인 것은 의미가 없습니다.

5. **불균형 데이터에서 정확도 사용.** 음성 클래스가 99%인 데이터셋에서 99%의 정확도는 모델이 아무것도 학습하지 않았음을 의미합니다. 정밀도, 재현율, F1, 또는 AUC를 사용하세요.

6. **지표 선택하기(cherry-picking).** 모델이 이기는 지표만 보고하는 것. 정직한 평가는 모든 관련 지표를 보고합니다.

7. **학습/테스트 분할 간 정보 누출.** 분할 전에 정규화하거나, 미래 데이터를 사용하여 과거를 예측하는 것.

8. **분산 추정치가 없는 작은 테스트 세트.** 100개 샘플로 평가하고 2%의 개선을 주장하는 것은 신호가 아닌 잡음입니다.

9. **데이터가 독립적이지 않은데 독립성을 가정하기.** 같은 환자의 의료 이미지, 같은 문서의 여러 문장. 그룹 내 관측치는 상관관계가 있습니다.

10. **P-hacking.** p < 0.05가 나올 때까지 다양한 테스트, 하위 집합, 제외 기준을 시도하는 것. 결과는 탐색의 산물입니다.

## 구현하기

다음 내용을 구현합니다:

1. **기초부터 작성한 기술 통계량** (평균, 중앙값, 최빈값, 표준 편차, 백분위수, IQR)
2. **상관 함수** (피어슨 및 스피어만, 공분산 행렬 포함)
3. **가설 검정** (단일 표본 t-검정, 두 표본 t-검정, 카이제곱 검정)
4. **부트스트랩 신뢰 구간** (임의의 통계량에 대해, 가정 필요 없음)
5. **A/B 테스트 시뮬레이터** (데이터 생성, 테스트, 제1종 및 제2종 오류 확인)
6. **통계적 유의성과 실무적 유의성 데모** (큰 n이 모든 것을 "유의미"하게 만든다는 것을 보여줌)

모두 `math`와 `random`만 사용하여 처음부터 작성합니다. numpy, scipy는 사용하지 않습니다.

```figure
f3-bootstrap-resample
```

## 핵심 용어

| 용어 | 정의 |
|---|---|
| 평균 | 값의 합을 개수로 나눈 값. 이상치에 민감합니다. |
| 중앙값 | 정렬된 데이터의 중간 값. 이상치에 강건합니다. |
| 표준 편차 | 분산의 제곱근. 원 단위에서의 산포를 측정합니다. |
| 백분위수 | 주어진 비율의 데이터가 그 값 아래에 위치하는 값. |
| IQR | 사분위 범위. Q3에서 Q1을 뺀 값. 중앙 50%의 산포를 나타냅니다. |
| 피어슨 상관계수 | 두 변수 간의 선형 연관성을 측정합니다. 범위는 [-1, 1]입니다. |
| 스피어만 상관계수 | 순위를 사용하여 단조 연관성을 측정합니다. |
| 공분산 행렬 | 모든 특징 간의 쌍별 공분산 행렬. |
| 귀무 가설 | 효과가 없거나 차이가 없다는 기본 가정. |
| p-값 | 귀무 가설이 참일 때, 이 정도로 극단적인 데이터가 나올 확률. |
| 신뢰 구간 | 주어진 신뢰 수준에서 매개변수의 타당한 값의 범위. |
| t-검정 | 평균이 유의미하게 다른지 테스트합니다. t-분포를 사용합니다. |
| 카이 제곱 검정 | 관측 빈도가 기대 빈도와 다른지 테스트합니다. |
| 효과 크기 | 표본 크기와 무관한 차이의 크기. Cohen's d가 일반적입니다. |
| Bonferroni 보정 | 오탐(false positive)을 통제하기 위해 유의성 임계값을 테스트 수로 나눕니다. |
| 부트스트랩 | 복원 추출(resampling with replacement)을 통해 표본 분포를 추정합니다. |
| 제1종 오류 | 오탐(False positive). H0가 참일 때 H0를 기각하는 것. |
| 제2종 오류 | 오탐(False negative). H0가 거짓일 때 H0를 기각하지 못하는 것. |
| 통계적 검정력 | 거짓인 H0를 올바르게 기각할 확률. 검정력 = 1 - 제2종 오류율. |
| 중심 극한 정리 | 표본 크기가 커짐에 따라 표본 평균이 정규 분포로 수렴합니다. |
| 모수적 검정 | 데이터에 특정 분포(주로 정규 분포)를 가정합니다. |
| 비모수적 검정 | 분포 가정을 하지 않습니다. 순위나 부호를 기반으로 작동합니다. |
