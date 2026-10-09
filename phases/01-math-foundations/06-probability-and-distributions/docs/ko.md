# 확률과 분포

> 확률은 AI가 불확실성을 표현하는 언어입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 1단계, 01-04강
**시간:** 약 75분

## 학습 목표

- 베르누이, 범주형, 포아송, 균일, 정규 분포에 대해 PMF와 PDF를 처음부터 구현해 보세요
- 기댓값과 분산을 계산하고, 가우스 분포가 지배적인 이유를 중앙 극한 정리(Central Limit Theorem)로 설명해 보세요
- 수치적 안정성 트릭(최대 로짓 차감)을 사용하여 소프트맥스 및 로그-소프트맥스 함수를 구축해 보세요
- 로짓에서 교차 엔트로피 손실(Cross-Entropy)을 계산하고, 이를 음의 로그 우도와 연결해 보세요

## 문제점

분류기는 `[0.03, 0.91, 0.06]`을 출력합니다. 언어 모델은 50,000개의 후보 중 다음 단어를 선택합니다. 확산 모델(Diffusion Model)은 학습된 분포에서 샘플링하여 이미지를 생성합니다. 이 모든 것은 확률의 실제 적용입니다.

모델이 하는 모든 예측은 확률 분포입니다. 모든 손실 함수(Loss Function)는 예측된 분포가 실제 분포와 얼마나 멀리 떨어져 있는지를 측정합니다. 모든 학습 단계는 하나의 분포가 다른 분포와 더 유사하게 보이도록 매개변수(Parameter)를 조정합니다. 확률이 없다면 ML 논문을 한 편도 읽을 수 없고, 모델을 한 개도 디버깅할 수 없으며, 학습 손실이 왜 NaN (숫자가 아님)(NaN (Not a Number))인지 이해할 수 없습니다.

## 개념

### 사건, 표본 공간, 확률

표본 공간 S는 모든 가능한 결과의 집합입니다. 사건은 표본 공간의 부분 집합입니다. 확률은 사건을 00강 1 사이의 숫자에 매핑합니다.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 가지 공리가 확률 전체를 정의합니다:
1. 임의의 사건 A에 대해 P(A) >= 0
2. P(S) = 1 (무언가는 항상 발생)
3. A와 B가 동시에 발생할 수 없을 때 P(A 또는 B) = P(A) + P(B)

나머지 모든 것(베이즈 정리, 기댓값, 분포)은 이 세 가지 규칙에서 파생됩니다.

### 조건부 확률과 독립성

P(A|B)는 B가 발생했다는 조건 하에 A가 발생할 확률입니다.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

두 사건은 하나를 알면 다른 하나에 대해 아무것도 알 수 없을 때 독립적입니다:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

동전 던지기는 독립적입니다. 복원하지 않고 카드를 뽑는 것은 독립적이지 않습니다.

### 확률 질량 함수 vs 확률 밀도 함수

이산 확률 변수는 확률 질량 함수(PMF)를 가집니다. 각 결과에는 직접 읽을 수 있는 특정 확률이 있습니다.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률 변수는 확률 밀도 함수(PDF)를 가집니다. 단일 점에서의 밀도는 확률이 아닙니다. 확률은 구간에 대해 밀도를 적분하여 얻습니다.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이 구별은 ML에서 중요합니다. 분류 출력은 PMF(이산 선택)입니다. VAE 잠재 공간은 PDF(연속)를 사용합니다.

### 공통 분포

**베르누이:** 한 번의 시행, 두 가지 결과. 이진 분류를 모델링합니다.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**범주형:** 한 번의 시행, k개의 결과. 다중 클래스 분류(softmax 출력)를 모델링합니다.

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**균등:** 모든 결과가 동일한 확률을 가집니다. 무작위 초기화에 사용됩니다.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**정규 분포(가우시안):** 종 모양 곡선. 평균(mu)과 분산(sigma^2)으로 매개변수화됩니다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**포아송:** 고정된 구간 내 희귀 이벤트의 개수. 이벤트율을 모델링합니다.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 기댓값과 분산

기댓값은 가중 평균 결과입니다.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

분산은 평균 주위의 퍼짐을 측정합니다.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)로 나타납니다. 분산은 모델 안정성에 대해 알려줍니다. 기울기의 높은 분산은 잡음 있는 훈련을 의미합니다.

### 결합 분포와 한계 분포

결합 분포 P(X, Y)는 두 확률 변수를 함께 설명합니다.

결합 PMF 예시 (X = 날씨, Y = 우산):

| | Y=0 (우산 없음) | Y=1 (우산) | 한계 P(X) |
|---|---|---|---|
| X=0 (맑음) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (비) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **한계 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

한계 분포는 다른 변수를 합산하여 제거합니다:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

위 표의 행 및 열 합계는 한계 분포입니다.

### 정규 분포가 모든 곳에서 나타나는 이유

중심 극한 정리: 많은 독립 확률 변수의 합(또는 평균)은 원래 분포에 관계없이 정규 분포로 수렴합니다.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

이것이 바로 그 이유입니다:
- 측정 오차는 대략 정규 분포를 따릅니다 (많은 작은 독립적인 요인들)
- 신경망의 가중치 초기화는 정규 분포를 사용합니다
- SGD의 기울기 잡음은 대략 정규 분포를 따릅니다 (많은 샘플 기울기의 합)
- 주어진 평균과 분산에 대해 정규 분포는 최대 엔트로피 분포입니다

### 로그 확률

원시 확률은 수치적 문제를 일으킵니다. 많은 작은 확률을 서로 곱하면 빠르게 0으로 언더플로우됩니다.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

로그 확률은 이를 해결합니다. 곱셈이 덧셈으로 바뀝니다.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0입니다 (0 < P <= 1이므로)
- 더 음수일수록 가능성이 낮습니다
- 교차 엔트로피 손실은 정답 클래스의 음수 로그 확률입니다

### 확률 분포로서의 Softmax

신경망은 원시 점수(로짓)를 출력합니다. Softmax는 이를 유효한 확률 분포로 변환합니다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax의 트릭: 지수 연산 전에 최대 로짓을 빼서 오버플로우를 방지합니다.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax는 수치적 안정성을 위해 Softmax와 로그를 결합합니다. PyTorch는 교차 엔트로피 손실 내부적으로 이를 사용합니다.

### 샘플링

샘플링은 분포에서 랜덤 값을 뽑는 것을 의미합니다. ML에서는:
- 드롭아웃은 어떤 뉴런을 0으로 만들지 랜덤하게 샘플링합니다
- 데이터 증강은 랜덤 변환을 샘플링합니다
- 언어 모델은 예측된 분포에서 다음 토큰을 샘플링합니다
- 확산 모델은 잡음을 샘플링하고 점진적으로 디노이즈합니다

임의의 분포에서 샘플링하려면 역 변환 샘플링, 거절 샘플링, 재매개변수화 트릭(VAE에서 사용) 같은 기법이 필요합니다.

```figure
gaussian-pdf
```

## 구현하기

### 1단계: 확률 기초

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

### 2단계: PMF와 PDF를 처음부터 구현하기

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

### 3단계: 기대값과 분산

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

### 4단계: 분포에서 샘플링

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

### 5단계: 소프트맥스 및 로그 확률

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

### 6단계: 중심 극한 정리 시연

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 7단계: 시각화

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

모든 시각화를 포함한 전체 구현은 `code/probability.py`에 있습니다.

## 사용하기

NumPy와 SciPy를 사용하면 위 내용은 모두 한 줄 코드로 처리됩니다:

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

이것들을 처음부터 직접 구축했습니다. 이제 라이브러리 함수가 무엇을 하는지 이해하게 되었습니다.

## 연습 문제

1. 지수 분포에 대한 역변환 샘플링을 구현하세요. 10,000개의 값을 샘플링하여 히스토그램을 실제 PDF와 비교하여 검증하세요.

2. 두 개의 편향된 주사위에 대한 결합 분포 표를 구축하세요. 한계 분포를 계산하고 주사위가 독립적인지 확인하세요.

3. 정답 클래스가 인덱스 3일 때 로짓 `[2.0, 0.5, -1.0, 3.0, 0.1]`을 출력하는 5개 클래스 분류기의 교차 엔트로피 손실(Cross-Entropy Loss)을 계산하세요. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`로 답을 검증하세요.

4. 로그 확률 목록을 받아 가장 가능성 높은 시퀀스, 총 로그 확률, 그리고 이에 해당하는 원시(raw) 확률을 반환하는 함수를 작성하세요. 각 단어의 확률이 0.01인 50단어 문장으로 테스트하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 표본 공간(Sample Space) | "모든 가능성" | 실험의 모든 가능한 결과의 집합 S |
| PMF | "확률 함수" | 각 이산 결과의 정확한 확률을 주며 합이 1이 되는 함수 |
| PDF | "확률 곡선" | 연속 변수에 대한 밀도 함수. 구간에 대해 적분하면 확률을 얻습니다 |
| 조건부 확률 | "무언가가 주어졌을 때의 확률" | P(A\|B) = P(A and B) / P(B). 베이지안 사고와 베이즈 정리의 기초 |
| 독립성 | "서로 영향을 주지 않음" | P(A and B) = P(A) * P(B). 한 사건을 알면 다른 사건에 대해 아무것도 알 수 없음 |
| 기댓값 | "평균" | 모든 결과의 확률 가중 합. 손실 함수는 기댓값입니다 |
| 분산 | "얼마나 퍼져 있는지" | 평균으로부터의 기대 제곱 편차. 높은 분산 = 잡음이 많고 불안정한 추정 |
| 정규 분포 | "종 모양 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). 중심 극한 정리(CLT) 때문에 모든 곳에서 나타남 |
| 중심 극한 정리 | "평균이 정규 분포가 됨" | 많은 독립 표본의 평균은 원천 분포와 관계없이 정규 분포로 수렴함 |
| 결합 분포 | "두 변수가 함께" | P(X, Y)는 X와 Y 결과의 모든 조합에 대한 확률을 설명함 |
| 한계 분포 | "다른 변수를 합산하여 제거" | P(X) = sum_y P(X, Y). 결합 분포에서 하나의 변수의 분포를 복원함 |
| 로그 확률 | "확률의 로그" | log P(x). 곱을 합으로 변환하여 긴 시퀀스에서 수치적 언더플로우를 방지함 |
| Softmax | "점수를 확률로 변환" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수 값 로짓을 유효한 확률 분포로 매핑함 |
| 교차 엔트로피(Cross-Entropy) | "손실 함수(Loss Function)" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정함. 낮을수록 좋음 |
| 로짓(Logits) | "모델의 원시 출력" | Softmax 이전의 정규화되지 않은 점수. 로지스틱 함수의 이름을 따서 명명됨 |
| 샘플링(Sampling) | "무작위 값 추출" | 확률 분포에 따라 값을 생성하는 것. 모델이 출력을 생성하는 방식 |

## 추가 읽기

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규 분포가 되는지에 대한 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기서 다루는 모든 내용과 그 이상을 간결하게 다루는 참고 자료
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치적 안정성이 중요한 이유와 달성 방법
