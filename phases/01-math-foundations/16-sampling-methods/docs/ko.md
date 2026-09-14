# 샘플링 방법 (Sampling Methods)

> 샘플링은 AI가 가능성의 공간을 탐색하는 방식입니다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## 학습 목표 (Learning Objectives)

- 균등 난수만으로 역 CDF, 기각 샘플링, 중요도 샘플링을 처음부터 구현합니다
- 언어 모델 토큰 생성을 위한 temperature, top-k, top-p(nucleus) 샘플링을 만듭니다
- 재매개변수화 트릭과 그것이 VAE에서 샘플링을 통한 역전파를 가능하게 하는 이유를 설명합니다
- 비정규화 목표 분포에서 샘플을 뽑기 위해 Metropolis-Hastings MCMC를 실행합니다

## 문제 상황 (The Problem)

언어 모델이 프롬프트 처리를 마치고 50,000개 로짓의 벡터를 만듭니다. 어휘의 토큰마다 하나씩. 이제 하나를 골라야 합니다. 어떻게?

항상 확률이 가장 높은 토큰만 고르면 모든 응답이 동일합니다. 결정론적입니다. 지루합니다. 균등하게 무작위로 고르면 출력은 횡설수설입니다. 답은 이 극단 사이에 있고, 그 위치는 샘플링이 제어합니다.

샘플링은 텍스트 생성에만 국한되지 않습니다. 강화 학습은 궤적을 샘플링해 정책 경사를 추정합니다. VAE는 학습된 분포에서 샘플을 뽑고 무작위성을 통해 역전파해 잠재 표현을 학습합니다. 확산 모델은 노이즈를 샘플링하고 반복적으로 디노이징해 이미지를 생성합니다. 몬테카를로 방법은 닫힌 형식 해가 없는 적분을 추정합니다. MCMC 알고리즘은 열거가 불가능한 고차원 사후분포를 탐색합니다.

모든 생성 AI 시스템은 샘플링 시스템입니다. 샘플링 전략이 출력의 품질, 다양성, 제어 가능성을 결정합니다. 이 레슨은 균등 난수에서 시작해 현대 LLM과 생성 모델을 구동하는 기법까지, 주요 샘플링 방법을 처음부터 만듭니다.

## 핵심 개념 (The Concept)

### 샘플링이 중요한 이유 (Why Sampling Matters)

샘플링은 AI와 머신러닝에서 네 가지 근본 역할로 나타납니다:

**생성(Generation).** 언어 모델, 확산 모델, GAN은 모두 샘플링으로 출력을 만듭니다. 샘플링 알고리즘이 창의성, 일관성, 다양성을 직접 제어합니다. Temperature, top-k, nucleus 샘플링은 엔지니어가 매일 돌리는 다이얼입니다.

**학습(Training).** 확률적 경사 하강은 미니배치를 샘플링합니다. Dropout은 비활성화할 뉴런을 샘플링합니다. 데이터 증강은 무작위 변환을 샘플링합니다. 중요도 샘플링은 강화 학습(PPO, TRPO)에서 경사 분산을 줄이기 위해 샘플에 가중치를 다시 줍니다.

**추정(Estimation).** ML의 많은 양은 닫힌 형식 해가 없습니다. 데이터 분포 위 기댓값 손실, 에너지 기반 모델의 파티션 함수, 베이즈 추론의 evidence. 몬테카를로 추정은 샘플 평균으로 이 모두를 근사합니다.

**탐색(Exploration).** MCMC 알고리즘은 베이즈 추론에서 사후분포를 탐색합니다. 진화 전략은 파라미터 섭동을 샘플링합니다. Thompson sampling은 밴딧에서 탐색과 활용의 균형을 맞춥니다.

핵심 과제: 직접 샘플링할 수 있는 것은 단순한 분포(균등, 정규)뿐입니다. 그 외에는 단순 샘플을 목표 분포의 샘플로 바꾸는 방법이 필요합니다.

### 균등 무작위 샘플링 (Uniform Random Sampling)

모든 샘플링 방법은 여기서 시작합니다. 균등 난수 생성기는 [0, 1) 값을 만들며, 길이가 같은 모든 부분 구간의 확률이 같습니다.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

이산 집합 n개에서 균등 샘플링하려면 U를 만들고 floor(n * U)를 반환합니다. 연속 구간 [a, b]에서는 a + (b - a) * U를 계산합니다.

핵심 통찰: 균등 난수 하나가 어떤 분포에서든 샘플 하나를 만들기에 딱 맞는 무작위성을 담고 있습니다. 요령은 올바른 변환을 찾는 것입니다.

### 역 CDF 방법 (Inverse Transform Sampling)

누적분포함수(CDF)는 값을 확률로 매핑합니다:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

역 CDF는 확률을 다시 값으로 매핑합니다. U ~ Uniform(0, 1)이면 X = F_inverse(U)는 목표 분포를 따릅니다.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**지수분포 예:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

F_inverse를 닫힌 형식으로 쓸 수 있을 때 완벽하게 작동합니다. 정규분포에는 닫힌 형식 역 CDF가 없어 다른 방법(Box-Muller, 또는 수치 근사)을 씁니다.

**이산 버전:** 이산 분포에서는 CDF를 누적합으로 만들고, U를 생성한 뒤 누적합이 U를 넘는 첫 인덱스를 찾습니다. Lesson 06의 `sample_categorical`이 이렇게 동작합니다.

### 기각 샘플링 (Rejection Sampling)

CDF를 역으로 구할 수 없지만 목표 PDF를 상수 배까지 평가할 수 있을 때 기각 샘플링이 작동합니다.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

경계 M이 타이트할수록 수락률이 높습니다. 저차원(1–3)에서는 기각 샘플링이 잘 됩니다. 고차원에서는 proposal 부피의 대부분이 기각되어 수락률이 지수적으로 떨어집니다. 이것이 기각 샘플링의 차원의 저주입니다.

**예: 절단 정규분포에서 샘플링.** 절단 구간에 균등 proposal을 씁니다. 포락선 M은 그 구간에서 정규 PDF의 최댓값입니다.

**예: 반원에서 샘플링.** 경계 사각형에서 균등하게 제안합니다. 점이 반원 안에 있으면 수락합니다. 몬테카를로가 pi를 계산하는 방식이 이것입니다. 수락률이 면적비 pi/4와 같습니다.

### 중요도 샘플링 (Importance Sampling)

때로는 목표 분포 p(x)의 샘플이 필요하지 않습니다. p(x) 아래 기댓값을 추정해야 하고, 다른 분포 q(x)의 샘플이 있을 뿐입니다.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

강화 학습에서 핵심입니다. PPO(Proximal Policy Optimization)에서는 옛 정책 pi_old 아래 궤적을 모으지만 새 정책 pi_new를 최적화하고 싶습니다. 중요도 가중치는 pi_new(a|s) / pi_old(a|s)입니다. PPO는 새 정책이 옛 정책에서 너무 멀어지지 않도록 이 가중치를 클리핑합니다.

중요도 샘플링 추정량의 분산은 q가 p와 얼마나 비슷한지에 달립니다. q가 p와 매우 다르면 소수의 샘플이 거대한 가중치를 받아 추정을 지배합니다. 자기정규화 중요도 샘플링은 가중치 합으로 나눠 이 문제를 줄입니다:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 몬테카를로 추정 (Monte Carlo Estimation)

몬테카를로 추정은 무작위 샘플을 평균내어 적분을 근사합니다. 큰 수의 법칙이 수렴을 보장합니다.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

오차율이 차원과 무관합니다. 그리드 기반 적분이 불가능한 고차원에서 몬테카를로가 지배하는 이유입니다.

**pi 추정:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**기댓값 추정:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 마르코프 연쇄 몬테카를로 (MCMC): Metropolis-Hastings

MCMC는 정상분포가 목표 분포 p(x)인 마르코프 연쇄를 구성합니다. 충분한 단계 뒤, 연쇄의 샘플은 (대략) p(x)의 샘플입니다.

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

대칭 proposal(q(x'|x) = q(x|x'))이면 비가 p(x')/p(x)로 단순해집니다. 이것이 원래 Metropolis 알고리즘입니다.

**왜 작동하는가.** 수락 규칙은 상세평형(detailed balance)을 보장합니다. x에 있다가 x'로 갈 확률과 x'에 있다가 x로 갈 확률이 같습니다. 상세평형은 p(x)가 연쇄의 정상분포임을 함의합니다.

**실무 고려사항:**
- Burn-in: 연쇄가 평형에 도달하기 전 초기 샘플을 버림
- Thinning: 자기상관을 줄이기 위해 k번째마다 샘플을 유지
- Proposal 스케일: 너무 작으면 연쇄가 느리게 움직임(높은 수락, 느린 탐색); 너무 크면 대부분의 proposal이 기각됨(낮은 수락, 제자리)
- 고차원에서 가우시안 proposal의 최적 수락률은 대략 0.234

### 깁스 샘플링 (Gibbs Sampling)

깁스 샘플링은 다변량 분포를 위한 MCMC의 특수 경우입니다. 모든 차원을 한 번에 제안하는 대신, 조건부 분포에서 변수를 하나씩 갱신합니다.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

깁스 샘플링은 각 조건부 분포 p(x_i | x_{-i})에서 샘플링할 수 있어야 합니다. 많은 모델에서 직관적입니다:
- 베이즈 네트워크: 조건부가 그래프 구조에서 따라옴
- 가우시안 혼합: 조건부가 가우시안
- 이징 모델: 각 스핀의 조건부는 이웃에만 의존

정확한 조건부에서 샘플링하면 상세평형을 자동으로 만족하므로 수락률은 항상 1입니다(모든 proposal이 수락됨).

**한계.** 변수가 강하게 상관되면 한 변수씩 갱신해서는 분포를 가로지르는 큰 대각 이동을 할 수 없어 깁스 샘플링이 느리게 혼합됩니다.

### Temperature 샘플링 (LLM에서 사용)

언어 모델은 어휘의 각 토큰에 대해 로짓 z_1, ..., z_V를 출력합니다. Softmax가 이를 확률로 바꿉니다. Temperature는 softmax 전에 로짓을 다시 스케일합니다:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**왜 작동하는가.** 로짓을 T < 1로 나누면 로짓 사이 차이가 증폭됩니다. z_1 = 2, z_2 = 1이면 T = 0.5로 나눠 z_1/T = 4, z_2/T = 2가 되어 격차가 커집니다. Softmax 후 가장 높은 로짓 토큰이 훨씬 큰 몫을 받습니다.

**실무에서:**
- T = 0.0: 탐욕 디코딩, 사실 Q&A에 최적
- T = 0.3-0.7: 약간 창의적, 코드 생성에 좋음
- T = 0.7-1.0: 균형, 일반 대화에 좋음
- T = 1.0-1.5: 창작, 브레인스토밍
- T > 1.5: 점점 무작위, 거의 쓸모없음

Temperature는 어떤 토큰이 가능한지를 바꾸지 않습니다. 각 토큰에 할당된 확률 질량을 바꿉니다.

### Top-k 샘플링

Top-k 샘플링은 후보 집합을 확률이 가장 높은 k개 토큰으로 제한한 뒤, 재정규화하고 그 제한된 집합에서 샘플링합니다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k는 어휘 분포의 긴 꼬리에 있는 극히 가능성 낮은 토큰(오타, 무의미)을 모델이 고르지 못하게 합니다. 문제: 맥락과 무관하게 k가 고정입니다. 모델이 자신 있을 때(한 토큰이 95% 확률) k = 40은 여전히 39개 대안을 허용합니다. 모델이 불확실할 때(확률이 1000개 토큰에 퍼짐) k = 40은 그럴듯한 선택지를 잘라 냅니다.

### Top-p (Nucleus) 샘플링

Top-p 샘플링은 후보 집합 크기를 동적으로 조정합니다. 고정 개수 대신, 누적 확률이 p를 넘는 가장 토큰 집합을 유지합니다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

모델이 자신 있으면 nucleus 샘플링은 토큰을 적게 유지합니다(어쩌면 2–3개). 불확실하면 많이 유지합니다(어쩌면 200개). 이 적응 행동 때문에 nucleus 샘플링이 일반적으로 top-k보다 더 나은 텍스트를 만듭니다.

**흔한 조합:**
- Temperature 0.7 + top-p 0.9: 좋은 범용 설정
- Temperature 0.0 (탐욕): 결정론적 작업에 최적
- Temperature 1.0 + top-k 50: Fan et al. (2018) 원 논문 설정

Top-k와 top-p는 결합할 수 있습니다. 먼저 top-k를 적용한 뒤, 남은 집합에 top-p를 적용합니다.

### 재매개변수화 트릭 (VAE에서 사용)

변분 오토인코더(VAE)는 입력을 잠재 공간의 분포로 인코딩하고, 그 분포에서 샘플을 뽑아, 샘플을 다시 디코딩하며 학습합니다. 문제: 샘플링 연산을 통해 역전파할 수 없습니다.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

재매개변수화 트릭은 무작위성과 파라미터를 분리합니다:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

N(mu, sigma^2)가 mu + sigma * N(0, 1)과 같은 분포이기 때문에 작동합니다. 핵심 통찰: 무작위성을 파라미터 없는 원천(epsilon)으로 옮긴 뒤, 샘플을 파라미터의 미분 가능 변환으로 표현합니다.

**VAE 학습 루프에서:**
1. 인코더가 각 입력에 대해 mu와 log(sigma^2)를 출력
2. epsilon ~ N(0, 1) 샘플
3. z = mu + sigma * epsilon 계산
4. z를 디코딩해 입력 재구성
5. 단계 4, 3, 2, 1을 통해 역전파 (단계 3이 미분 가능하므로 가능)

재매개변수화 트릭 없이는 표준 역전파로 VAE를 학습할 수 없습니다. 이 한 가지 통찰이 VAE를 실용적으로 만들었습니다.

### Gumbel-Softmax (미분 가능한 범주형 샘플링)

재매개변수화 트릭은 연속 분포(가우시안)에 작동합니다. 이산 범주형 분포에는 다른 접근이 필요합니다. Gumbel-Softmax는 범주형 샘플링의 미분 가능 근사를 제공합니다.

**Gumbel-Max 트릭 (미분 불가능):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (미분 가능 근사):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax는 이산 샘플의 연속 완화를 만듭니다. 출력은 hard one-hot 대신 확률 벡터(soft one-hot)입니다. Softmax를 통해 경사가 흐릅니다. 학습의 순전파에서는 "straight-through" 추정량을 쓸 수 있습니다. 순전파에는 hard argmax를, 역전파에는 soft Gumbel-Softmax 경사를 씁니다.

**응용:**
- VAE의 이산 잠재 변수
- 신경망 아키텍처 탐색 (이산 연산 선택)
- Hard 어텐션 메커니즘
- 이산 행동이 있는 강화 학습

### 층화 샘플링 (Stratified Sampling)

표준 몬테카를로 샘플링은 우연히 표본 공간에 빈틈을 남길 수 있습니다. 층화 샘플링은 공간을 층으로 나누고 각 층에서 샘플링해 균등한 커버리지를 강제합니다.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

층화 샘플링은 표준 몬테카를로보다 항상 분산이 작거나 같습니다:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**응용:**
- 수치 적분 (준몬테카를로)
- 학습 데이터 분할 (각 폴드에서 클래스 균형 보장)
- 층화를 곁들인 중요도 샘플링 (두 기법 결합)
- NeRF(Neural Radiance Fields)는 카메라 광선을 따라 층화 샘플링을 사용

### 확산 모델과의 연결 (Connection to Diffusion Models)

확산 모델은 샘플링 과정으로 이미지를 생성합니다. 순방향 과정은 T 단계에 걸쳐 이미지에 가우시안 노이즈를 더해 순수 노이즈가 될 때까지 갑니다. 역과정은 디노이징을 학습해 원본 이미지를 단계적으로 복원합니다.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

이 레슨의 방법과의 연결:
- 각 디노이징 단계는 재매개변수화 트릭을 사용 (노이즈 샘플, 결정론적 변환 적용)
- 노이즈 스케줄 {alpha_t}는 일종의 temperature annealing을 제어
- 학습은 몬테카를로 추정으로 ELBO(evidence lower bound)를 근사
- 확산 모델의 ancestral sampling은 마르코프 연쇄 (각 단계는 현재 상태에만 의존)

전체 이미지 생성 과정은 반복 샘플링입니다. 노이즈에서 시작해, 매 단계마다 학습된 디노이징 모델에 조건화된 약간 덜 노이즈한 버전을 샘플링합니다.

```figure
monte-carlo-pi
```

## 구현하기 (Build It)

### Step 1: 균등과 역 CDF 샘플링

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

지수분포 샘플 10,000개를 만들고 평균이 1/lambda인지 검증하세요.

### Step 2: 기각 샘플링

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

기각 샘플링으로 절단 정규분포에서 뽑으세요. 샘플을 히스토그램으로 그려 형태를 검증하세요.

### Step 3: 중요도 샘플링

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

균등 proposal로 정규분포 아래 E[X^2]를 추정하세요. 알려진 답(mu^2 + sigma^2)과 비교하세요.

### Step 4: pi의 몬테카를로 추정

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

이봉 분포(두 가우시안의 혼합)에서 샘플링하세요. 연쇄의 궤적을 시각화하세요.

### Step 6: 깁스 샘플링

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature 샘플링

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

토큰 로짓 집합에서 temperature가 출력 분포를 어떻게 바꾸는지 보이세요.

### Step 8: Top-k와 top-p 샘플링

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: 재매개변수화 트릭

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

경사가 재매개변수화된 샘플을 통해는 흐르지만 직접 샘플링을 통해서는 흐르지 않음을 보이세요.

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

temperature를 낮추면 출력이 one-hot 벡터에 가까워짐을 보이세요.

모든 시각화를 포함한 전체 구현은 `code/sampling.py`에 있습니다.

## 실용 활용 (Use It)

NumPy와 SciPy를 쓰면 프로덕션 버전은 다음과 같습니다:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

규모 있는 MCMC에는 전용 라이브러리를 쓰세요:
- PyMC: NUTS(적응형 HMC)를 쓰는 완전한 베이즈 모델링
- emcee: 앙상블 MCMC 샘플러
- NumPyro/JAX: GPU 가속 MCMC

처음부터 직접 만들었습니다. 이제 라이브러리 호출이 무엇을 하는지 압니다.

## 연습 문제 (Exercises)

1. Cauchy 분포에 대해 역 CDF 샘플링을 구현하세요. CDF는 F(x) = 0.5 + arctan(x)/pi입니다. 샘플 10,000개를 만들고 진짜 PDF에 대해 히스토그램을 그리세요. 무거운 꼬리(중심에서 먼 극단값)를 확인하세요.

2. Uniform(0, 1) proposal로 Beta(2, 5) 분포에서 샘플을 만드는 기각 샘플링을 쓰세요. 수락된 샘플을 진짜 Beta PDF에 대해 그리세요. 이론적 수락률은?

3. 0에서 pi까지 sin(x)의 적분을 몬테카를로로 1,000, 10,000, 100,000 샘플에서 추정하세요. 각 수준의 오차를 비교하세요. 오차가 O(1/sqrt(N))로 스케일됨을 검증하세요.

4. p(x, y)가 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)에 비례하는 2D 분포에서 샘플링하도록 Metropolis-Hastings를 구현하세요. 샘플과 연쇄 궤적을 그리세요. 서로 다른 proposal 표준편차로 실험하세요.

5. 완전한 텍스트 생성 데모를 만드세요. 로짓이 있는 단어 10개 어휘가 주어지면 (a) 탐욕, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9로 토큰 20개 시퀀스를 생성하세요. 5회 실행에 걸쳐 출력 다양성을 비교하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "무작위 값을 뽑기" | 확률 분포에 따라 값을 생성. 모든 생성 AI 뒤의 메커니즘 |
| Uniform distribution | "모두 똑같이 가능" | [a, b]의 모든 값이 같은 확률밀도 1/(b-a). 모든 샘플링 방법의 출발점 |
| Inverse CDF | "확률 변환" | F_inverse(U)가 알려진 CDF가 있는 어떤 분포의 샘플로 균등 샘플을 바꿈. 정확하고 효율적 |
| Rejection sampling | "제안하고 수락/기각" | 단순 proposal에서 생성하고, target/proposal 비에 비례하는 확률로 수락. 정확하지만 샘플을 낭비 |
| Importance sampling | "샘플에 가중치 다시 주기" | q(x) 샘플에 p(x)/q(x) 가중치를 줘 p(x) 아래 기댓값을 추정. RL의 PPO 핵심 |
| Monte Carlo | "무작위 샘플 평균" | 적분을 샘플 평균으로 근사. 오차 O(1/sqrt(N)), 차원과 무관 |
| MCMC | "수렴하는 무작위 보행" | 정상분포가 목표인 마르코프 연쇄를 구성. Metropolis-Hastings가 기초 알고리즘 |
| Metropolis-Hastings | "오르막은 수락, 내리막은 가끔" | 이동을 제안하고 밀도비로 수락. 상세평형이 목표 분포로의 수렴을 보장 |
| Gibbs sampling | "한 변수씩" | 나머지를 고정한 채 각 변수를 조건부 분포에서 갱신. 수락률 100% |
| Temperature | "자신감 다이얼" | Softmax 전에 로짓을 T로 나눔. T<1이면 날카로워짐(더 자신), T>1이면 평평해짐(더 다양) |
| Top-k sampling | "상위 k개만 유지" | 확률이 가장 높은 k개 외를 0으로 만들고 재정규화 후 샘플. 고정 후보 집합 크기 |
| Nucleus sampling (top-p) | "그럴듯한 것만 유지" | 누적 확률이 p를 넘는 최소 토큰 집합을 유지. 적응형 후보 집합 크기 |
| Reparameterization trick | "무작위성을 밖으로" | z = mu + sigma * epsilon (epsilon ~ N(0,1)). 샘플링을 미분 가능하게. VAE 학습에 필수 |
| Gumbel-Softmax | "부드러운 범주형 샘플링" | Gumbel 노이즈 + temperature softmax로 범주형 샘플링의 미분 가능 근사 |
| Stratified sampling | "강제 커버리지" | 표본 공간을 층으로 나누고 각 층에서 샘플. 순진한 몬테카를로보다 항상 낮은 분산 |
| Burn-in | "워밍업 기간" | 연쇄가 정상분포에 도달하기 전 버리는 초기 MCMC 샘플 |
| Detailed balance | "가역성 조건" | p(x) * T(x->y) = p(y) * T(y->x). p가 마르코프 연쇄의 정상분포가 되기 위한 충분 조건 |
| Diffusion sampling | "반복 디노이징" | 노이즈에서 시작해 학습된 디노이징 단계를 적용해 데이터 생성. 각 단계는 조건부 샘플링 |

## 더 읽을거리 (Further Reading)

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 기초에 대한 상세 튜토리얼
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 원 Gumbel-Softmax 논문
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) 샘플링 논문
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 재매개변수화 트릭을 도입한 VAE 논문
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM이 샘플링을 이미지 생성에 연결
