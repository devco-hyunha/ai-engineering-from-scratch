---
name: skill-probability-reasoning
description: 주어진 ML 문제에 맞는 확률 분포를 선택한다
version: 1.0.0
phase: 1
lesson: 6
tags: [probability, distributions, modeling]
---

# 확률 분포 선택 (Probability Distribution Selection)

데이터를 모델링하거나, 손실 함수를 설계하거나, prior를 설정할 때 올바른 분포를 고르는 방법입니다.

## 결정 체크리스트 (Decision Checklist)

1. 결과가 이산(범주, 횟수)인가, 연속(측정값, 점수)인가?
2. 결과가 유계(예: [0, 1])인가, 무계인가?
3. 가능한 결과는 몇 개인가? 둘? k개? 무한?
4. 데이터가 대칭인가, 치우쳐 있는가?
5. 사건이 독립인가, 상관되어 있는가?
6. 비율, 횟수, 비율(proportion), 측정값 중 무엇을 모델링하는가?

## 분포 결정 트리 (Distribution decision tree)

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

| 시나리오 | 분포 | 파라미터 |
|---|---|---|
| Binary classification output | Bernoulli | p = sigmoid(logit) |
| Multi-class classification output | Categorical | p = softmax(logits) |
| Token prediction in language models | Categorical over vocab | p from softmax |
| Pixel intensity (normalized) | Beta or Uniform [0, 1] | Depends on image stats |
| Word count in a document | Poisson | lambda = avg word count |
| Time between user requests | Exponential | lambda = request rate |
| Measurement error | Normal | mu = 0, sigma from data |
| Weight initialization | Normal or Uniform | Kaiming/Xavier rules |
| VAE latent space prior | Standard Normal | mu = 0, sigma = 1 |
| Bayesian prior on proportions | Beta | alpha, beta from belief |
| Bayesian prior on category weights | Dirichlet | alpha vector |
| Noise in regression targets | Normal | mu = 0, sigma estimated |
| Outlier-robust regression | Student's t | low degrees of freedom |
| Duration/lifetime modeling | Weibull or Gamma | shape and scale |
| Topic distribution per document (LDA) | Dirichlet | alpha < 1 for sparse |

## 분포가 잘못될 때

- 데이터가 단단한 하한(예: 가격, 거리)을 가질 때 Normal을 쓰는 경우. Normal은 음수에도 0이 아닌 확률을 줍니다. 대신 log-normal 또는 gamma를 쓰세요.
- 분산이 평균과 다를 때 Poisson을 쓰는 경우. Poisson은 mean = variance를 가정합니다. variance > mean이면 negative binomial을 쓰세요.
- 다중 클래스 문제에 Bernoulli를 쓰는 경우. Bernoulli는 엄격히 이진입니다. k > 2이면 categorical을 쓰세요.
- 관측이 상관되어 있는데 독립을 가정하는 경우. 시계열, 공간 데이터, 그룹 데이터는 독립을 위반합니다. autoregressive 또는 hierarchical 모델을 쓰세요.

## 흔한 실수

- PDF 값을 확률과 혼동하기. PDF는 1을 넘을 수 있습니다. 확률은 구간에서 PDF를 적분해 얻습니다.
- Softmax 출력이 독립 Bernoulli 확률이 아니라 categorical 확률이라는 점 잊기. 구성상 합이 1입니다.
- 도메인 지식이 있는데 균등 prior를 쓰기. 잘 고른 정보적 prior는 편향 없이 분산을 줄입니다.
- 로그 확률을 확률처럼 다루기. Log-prob는 항상 음수(또는 0)입니다. 합이 1이 되지 않습니다.

## 빠른 참고: 분포 성질

| Distribution | Support | Mean | Variance | Key property |
|---|---|---|---|---|
| Bernoulli(p) | {0, 1} | p | p(1-p) | Simplest discrete |
| Binomial(n, p) | {0..n} | np | np(1-p) | Sum of n Bernoulli |
| Poisson(lam) | {0, 1, 2, ...} | lam | lam | Mean = variance |
| Normal(mu, s^2) | (-inf, inf) | mu | s^2 | Max entropy for given mean/var |
| Exponential(lam) | [0, inf) | 1/lam | 1/lam^2 | Memoryless |
| Beta(a, b) | [0, 1] | a/(a+b) | ab/((a+b)^2(a+b+1)) | Conjugate to Binomial |
| Gamma(a, b) | (0, inf) | a/b | a/b^2 | Conjugate to Poisson |
| Dirichlet(alpha) | Simplex | alpha_i/sum | (see formula) | Conjugate to Categorical |
