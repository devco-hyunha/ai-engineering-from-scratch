---
name: skill-probability-reasoning
description: 주어진 ML 문제에 적합한 확률 분포 선택
version: 1.0.0
phase: 1
lesson: 6
tags: [probability, distributions, modeling]
---

# 확률 분포 선택

데이터를 모델링하거나, 손실 함수를 설계하거나, 사전 분포를 설정할 때 올바른 분포를 선택하는 방법입니다.

## 의사결정 체크리스트

1. 결과가 이산적(카테고리, 개수)인가요, 아니면 연속적(측정값, 점수)인가요?
2. 결과가 유한한 범위(예: [0, 1])인가요, 아니면 무한한 범위인가요?
3. 가능한 결과의 개수는 몇 개인가요? 2개? k개? 무한대인가요?
4. 데이터는 대칭인가요, 아니면 비대칭(치우침)인가요?
5. 사건은 독립적인가요, 아니면 상관관계가 있나요?
6. 비율, 개수, 비율(proportion), 측정값 중 무엇을 모델링하고 있나요?

## 분포 의사결정 트리

```
Is the variable discrete?
  Yes --> Only 2 outcomes? --> Bernoulli (p)
     |    k outcomes, one trial? --> Categorical (p1...pk)
     |    k outcomes, n trials? --> Multinomial (n, p1...pk)
     |    Count of successes in n trials? --> Binomial (n, p)
     |    Count of events per interval? --> Poisson (lambda)
     |    Count of trials until first success? --> Geometric (p)
     |    Count of trials until r successes? --> Negative Binomial (r, p)
  No --> Symmetric, bell-shaped? --> Normal (mu, sigma)
     |   Positive values, right-skewed? --> Log-normal or Exponential
     |   Bounded in [0, 1]? --> Beta (alpha, beta)
     |   Positive values, flexible shape? --> Gamma (alpha, beta)
     |   Time between events? --> Exponential (lambda)
     |   Heavy tails needed? --> Student's t (nu) or Cauchy
     |   Multivariate, bell-shaped? --> Multivariate Normal
     |   On a simplex (sums to 1)? --> Dirichlet (alpha)
```

## 실제 ML 시나리오를 분포에 매핑하기

| 시나리오 | 분포 | 매개변수 |
|---|---|---|
| 이진 분류 출력 | 베르누이 | p = sigmoid(logit) |
| 다중 클래스 분류 출력 | 범주형 | p = softmax(logits) |
| 언어 모델의 토큰 예측 | 어휘에 대한 범주형 | p는 softmax에서 계산 |
| 픽셀 강도 (정규화됨) | Beta 또는 Uniform [0, 1] | 이미지 통계에 따라 다름 |
| 문서 내 단어 개수 | 포아송 | lambda = 평균 단어 개수 |
| 사용자 요청 간 시간 | 지수 분포 | lambda = 요청 속도 |
| 측정 오차 | 정규 분포 | mu = 0, sigma는 데이터에서 계산 |
| 가중치 초기화 | 정규 분포 또는 Uniform | Kaiming/Xavier 규칙 |
| VAE 잠재 공간 사전 분포 | 표준 정규 분포 | mu = 0, sigma = 1 |
| 비율에 대한 베이지안 사전 분포 | Beta | alpha, beta는 신뢰도에서 계산 |
| 카테고리 가중치에 대한 베이지안 사전 분포 | Dirichlet | alpha 벡터 |
| 회귀 타겟의 잡음 | 정규 분포 | mu = 0, sigma는 추정 |
| 이상치에 강건한 회귀 | Student's t | 낮은 자유도 |
| 지속 시간/수명 모델링 | Weibull 또는 Gamma | shape와 scale |
| 문서별 토픽 분포 (LDA) | Dirichlet | 희소성을 위해 alpha < 1 |

## 분포가 잘못 사용될 때

- 데이터에 강한 하한이 있을 때(예: 가격, 거리) Normal 분포를 사용하는 경우. Normal 분포는 음수 값에 비영(probability > 0) 확률을 할당합니다. 대신 log-normal 또는 gamma 분포를 사용해 보세요.
- 분산이 평균과 다를 때 Poisson 분포를 사용하는 경우. Poisson 분포는 평균 = 분산을 가정합니다. 분산 > 평균인 경우 negative binomial 분포를 사용해 보세요.
- 다중 클래스 문제에 Bernoulli 분포를 사용하는 경우. Bernoulli 분포는 엄격하게 이진(binary)입니다. k > 2인 경우 categorical 분포를 사용해 보세요.
- 관측치가 상관관계가 있는데 독립성을 가정하는 경우. 시계열, 공간 데이터, 그룹화된 데이터는 독립성을 위반합니다. autoregressive 또는 hierarchical 모델을 사용해 보세요.

## 공통적인 실수

- PDF 값을 확률과 혼동하는 경우. PDF 값은 1을 초과할 수 있습니다. 확률은 PDF를 구간에 대해 적분하여 얻습니다.
- softmax 출력은 categorical 확률이며 독립적인 Bernoulli 확률이 아니라는 점을 잊는 경우. 이들은 구조적으로 합이 1이 됩니다.
- 도메인 지식이 있는데 uniform prior를 사용하는 경우. 잘 선택된 informative prior는 결과를 편향시키지 않으면서 분산을 줄여줍니다.
- log-probabilities를 확률로 취급하는 경우. Log-probs는 항상 음수(또는 0)입니다. 이들은 합이 1이 되지 않습니다.

## 빠른 참조: 분포 특성

| 분포 | 지지집합(Support) | 평균(Mean) | 분산(Variance) | 주요 특성 |
|---|---|---|---|---|
| Bernoulli(p) | {0, 1} | p | p(1-p) | 가장 단순한 이산 분포 |
| Binomial(n, p) | {0..n} | np | np(1-p) | n개의 Bernoulli 합 |
| Poisson(lam) | {0, 1, 2, ...} | lam | lam | 평균 = 분산 |
| Normal(mu, s^2) | (-inf, inf) | mu | s^2 | 주어진 평균/분산에 대해 최대 엔트로피 |
| Exponential(lam) | [0, inf) | 1/lam | 1/lam^2 | 무기억성(Memoryless) |
| Beta(a, b) | [0, 1] | a/(a+b) | ab/((a+b)^2(a+b+1)) | Binomial의 켤레 분포 |
| Gamma(a, b) | (0, inf) | a/b | a/b^2 | Poisson의 켤레 분포 |
| Dirichlet(alpha) | Simplex | alpha_i/sum | (공식 참조) | Categorical의 켤레 분포 |
