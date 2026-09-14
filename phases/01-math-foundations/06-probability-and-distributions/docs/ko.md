# 확률과 분포 (Probability and Distributions)

> 확률은 AI가 불확실성을 표현하는 언어입니다.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- Bernoulli, categorical, Poisson, uniform, normal 분포에 대해 PMF와 PDF를 처음부터 구현합니다
- 기댓값·분산을 계산하고, 중심극한정리로 가우시안이 왜 어디에나 나타나는지 설명합니다
- 수치 안정성 트릭(최대 로짓 빼기)으로 softmax와 log-softmax를 구현합니다
- 로짓에서 교차 엔트로피 손실을 계산하고 음의 로그 가능도와 연결합니다

## 문제 상황 (The Problem)

분류기가 `[0.03, 0.91, 0.06]`을 출력합니다. 언어 모델은 50,000개 후보에서 다음 단어를 고릅니다. 확산 모델은 학습된 분포에서 샘플링해 이미지를 생성합니다. 이 모든 것이 확률의 작용입니다.

모델이 내리는 모든 예측은 확률 분포입니다. 모든 손실 함수는 예측 분포가 참 분포에서 얼마나 먼지를 측정합니다. 모든 학습 단계는 한 분포가 다른 분포와 더 비슷해지도록 파라미터를 조정합니다. 확률 없이는 ML 논문 한 편도 읽을 수 없고, 모델 하나도 디버깅할 수 없으며, 학습 손실이 왜 NaN인지 이해할 수 없습니다.

## 핵심 개념 (The Concept)

### 사건, 표본 공간, 확률 (Events, Sample Spaces, and Probability)

표본 공간 S는 가능한 모든 결과의 집합입니다. 사건은 표본 공간의 부분집합입니다. 확률은 사건을 0과 1 사이의 수에 대응시킵니다.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 공리가 확률 전체를 정의합니다:
1. 임의의 사건 A에 대해 P(A) >= 0
2. P(S) = 1 (항상 무언가 발생한다)
3. A와 B가 동시에 일어날 수 없을 때 P(A or B) = P(A) + P(B)

그 밖의 모든 것(베이즈 정리, 기댓값, 분포)은 이 세 규칙에서 나옵니다.

### 조건부 확률과 독립 (Conditional Probability and Independence)

P(A|B)는 B가 일어났다는 조건 하에 A의 확률입니다.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

한 사건을 알아도 다른 사건에 대해 아무것도 알려주지 않으면 두 사건은 독립입니다:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

동전 던지기는 독립입니다. 복원 없이 카드를 뽑는 것은 독립이 아닙니다.

### 확률질량함수 vs 확률밀도함수 (PMF vs PDF)

이산 확률변수는 확률질량함수(PMF)를 가집니다. 각 결과는 바로 읽을 수 있는 구체적 확률을 갖습니다.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률변수는 확률밀도함수(PDF)를 가집니다. 한 점에서의 밀도는 확률이 아닙니다. 확률은 구간 위에서 밀도를 적분해 얻습니다.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이 구분은 ML에서 중요합니다. 분류 출력은 PMF(이산 선택)입니다. VAE 잠재 공간은 PDF(연속)를 사용합니다.

### 흔한 분포들 (Common Distributions)

**Bernoulli:** 한 번의 시행, 두 결과. 이진 분류를 모델링합니다.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** 한 번의 시행, k개 결과. 다중 클래스 분류(softmax 출력)를 모델링합니다.

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** 모든 결과가 동등하게 가능. 무작위 초기화에 사용됩니다.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** 종 모양 곡선. 평균(mu)과 분산(sigma^2)으로 매개화됩니다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** 고정 구간에서 드문 사건의 횟수. 사건 발생률을 모델링합니다.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 기댓값과 분산 (Expected Value and Variance)

기댓값은 가중 평균 결과입니다.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

분산은 평균 주변의 퍼짐을 측정합니다.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)로 나타납니다. 분산은 모델 안정성에 대해 알려줍니다. 기울기 분산이 크면 학습이 시끄럽습니다.

### 결합 분포와 주변 분포 (Joint and Marginal Distributions)

결합 분포 P(X, Y)는 두 확률변수를 함께 기술합니다.

결합 PMF 예제 (X = 날씨, Y = 우산):

| | Y=0 (우산 없음) | Y=1 (우산) | 주변 P(X) |
|---|---|---|---|
| X=0 (맑음) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (비) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **주변 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

주변 분포는 다른 변수를 합해서 제거합니다:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

위 표의 행·열 합계가 주변분포입니다.

### 정규분포가 어디에나 나타나는 이유 (Why the Normal Distribution Shows Up Everywhere)

중심극한정리: 많은 독립 확률변수의 합(또는 평균)은 원래 분포와 상관없이 정규분포로 수렴합니다.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

그래서:
- 측정 오차는 대략 정규(많은 작은 독립 원인)
- 신경망 가중치 초기화는 정규분포를 사용
- SGD의 기울기 노이즈는 대략 정규(많은 샘플 기울기의 합)
- 주어진 평균·분산에 대해 정규분포는 최대 엔트로피 분포

### 로그 확률 (Log Probabilities)

원시 확률은 수치 문제를 일으킵니다. 작은 확률을 많이 곱하면 빠르게 0으로 underflow합니다.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

로그 확률이 이를 고칩니다. 곱셈이 덧셈이 됩니다.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0 (0 < P <= 1이므로)
- 더 음수일수록 덜 가능
- 교차 엔트로피 손실은 정답 클래스의 음의 로그 확률

### Softmax를 확률 분포로 (Softmax as a Probability Distribution)

신경망은 원시 점수(로짓)를 출력합니다. Softmax는 이를 유효한 확률 분포로 바꿉니다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 트릭: 지수화 전에 최대 로짓을 빼 overflow를 막습니다.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax는 수치 안정성을 위해 softmax와 log를 결합합니다. PyTorch는 교차 엔트로피 손실 내부에서 이를 사용합니다.

### 샘플링 (Sampling)

샘플링은 분포에서 무작위 값을 뽑는 것입니다. ML에서:
- Dropout은 어떤 뉴런을 0으로 만들지 무작위로 샘플링
- 데이터 증강은 무작위 변환을 샘플링
- 언어 모델은 예측 분포에서 다음 토큰을 샘플링
- 확산 모델은 노이즈를 샘플링하고 점진적으로 디노이즈

임의 분포에서의 샘플링에는 역변환 샘플링, 기각 샘플링, 또는 재매개화 트릭(VAE에서 사용) 같은 기법이 필요합니다.

```figure
gaussian-pdf
```

## 구현하기 (Build It)

### Step 1: 확률 기초 (Probability basics)

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF와 PDF를 처음부터 (PMF and PDF from scratch)

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: 기댓값과 분산 (Expected value and variance)

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: 분포에서 샘플링 (Sampling from distributions)

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax와 로그 확률 (Softmax and log probabilities)

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: 중심극한정리 시연 (Central Limit Theorem demonstration)

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: 시각화 (Visualization)

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

전체 시각화를 포함한 구현은 `code/probability.py`에 있습니다.

## 실용 활용 (Use It)

NumPy와 SciPy를 쓰면 위의 모든 것이 한 줄입니다:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

직접 처음부터 만들었으니, 이제 라이브러리 호출이 무엇을 하는지 압니다.

## 연습 문제 (Exercises)

1. 지수 분포에 대해 역변환 샘플링을 구현하세요. 10,000개 값을 샘플링하고 히스토그램을 참 PDF와 비교해 검증하세요.

2. 편향된 주사위 두 개에 대한 결합 분포 표를 만드세요. 주변 분포를 계산하고 주사위가 독립인지 확인하세요.

3. 로짓 `[2.0, 0.5, -1.0, 3.0, 0.1]`을 출력하는 5-클래스 분류기에서 정답 클래스가 인덱스 3일 때 교차 엔트로피 손실을 계산하세요. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`로 답을 검증하세요.

4. 로그 확률 리스트를 받아 가장 가능성 높은 시퀀스, 총 로그 확률, 이에 해당하는 원시 확률을 반환하는 함수를 작성하세요. 각 단어 확률이 0.01인 50단어 문장으로 테스트하세요.

## 핵심 용어 (Key Terms)

| 용어 (Term) | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Sample space | "가능한 모든 것" | 실험의 가능한 모든 결과의 집합 S |
| PMF | "확률 함수" | 각 이산 결과의 정확한 확률을 주며, 합이 1이 되는 함수 |
| PDF | "확률 곡선" | 연속 변수용 밀도 함수. 구간에서 적분하면 확률 |
| Conditional probability | "무엇이 주어진 확률" | P(A\|B) = P(A and B) / P(B). 베이즈적 사고와 베이즈 정리의 기초 |
| Independence | "서로 영향 없음" | P(A and B) = P(A) * P(B). 한 사건을 알아도 다른 사건에 대해 아무것도 모름 |
| Expected value | "평균" | 모든 결과의 확률 가중 합. 손실 함수는 기댓값 |
| Variance | "얼마나 퍼져 있는지" | 평균으로부터의 제곱 편차의 기댓값. 높은 분산 = 시끄럽고 불안정한 추정 |
| Normal distribution | "종 모양 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). CLT 때문에 어디에나 나타남 |
| Central Limit Theorem | "평균이 정규가 된다" | 많은 독립 샘플의 평균이 출처와 무관하게 정규분포로 수렴 |
| Joint distribution | "두 변수를 함께" | P(X, Y)는 X와 Y 결과의 모든 조합 확률을 기술 |
| Marginal distribution | "다른 변수를 합쳐 제거" | P(X) = sum_y P(X, Y). 결합에서 한 변수의 분포를 복원 |
| Log probability | "확률의 로그" | log P(x). 곱을 합으로 바꿔 긴 시퀀스의 수치 underflow를 방지 |
| Softmax | "점수를 확률로" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수 로짓을 유효 확률 분포로 사상 |
| Cross-entropy | "손실 함수" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정. 낮을수록 좋음 |
| Logits | "원시 모델 출력" | softmax 이전의 비정규화 점수. logistic 함수에서 이름 |
| Sampling | "무작위 값 뽑기" | 확률 분포에 따라 값을 생성. 모델이 출력을 생성하는 방식 |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규가 되는지 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기 내용과 그 이상을 다루는 간결한 참고자료
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치 안정성이 왜 중요한지, 어떻게 달성하는지
