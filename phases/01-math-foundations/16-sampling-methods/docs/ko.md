# 샘플링 방법

> 샘플링은 AI가 가능성의 공간을 탐색하는 방식입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 1단계, 06-07강 (확률, 베이즈 정리)
**시간:** 약 120분

## 학습 목표

- 균일 난수만 사용하여 역 CDF, 기각 샘플링, 중요도 샘플링을 처음부터 구현해 보세요
- 언어 모델 토큰 생성을 위해 온도, top-k, top-p (핵) 샘플링을 구축해 보세요
- 재파라미터화 트릭(reparameterization trick)을 설명하고, VAE에서 샘플링을 통해 역전파가 가능해지는 이유를 이해해 보세요
- 정규화되지 않은 목표 분포에서 샘플링하기 위해 Metropolis-Hastings MCMC를 실행해 보세요

## 문제점

언어 모델이 프롬프트 처리를 완료하고 어휘의 모든 토큰에 대한 50,000개의 로짓 벡터를 생성합니다. 이제 하나를 선택해야 합니다. 어떻게 선택할까요?

항상 확률이 가장 높은 토큰을 선택하면 모든 응답이 동일해집니다. 결정론적이고 지루합니다. 균일하게 무작위로 선택하면 출력은 의미 없는 글자가 됩니다. 정답은 이 두 극단 사이의 어딘가에 있으며, 그 어딘가는 샘플링이 제어합니다.

샘플링은 텍스트 생성에만 국한되지 않습니다. 강화 학습은 궤적 샘플링으로 정책 기울기를 추정합니다. VAE는 학습된 분포에서 샘플링하고 무작위성을 통해 역전파하여 잠재 표현을 학습합니다. 확산 모델은 잡음을 샘플링하고 반복적으로 잡음을 제거하여 이미지를 생성합니다. 몬테 카를로 방법은 닫힌 형식의 해가 없는 적분을 추정합니다. MCMC 알고리즘은 열거가 불가능한 고차원 사후 분포를 탐색합니다.

모든 생성형 AI 시스템은 샘플링 시스템입니다. 샘플링 전략은 출력의 품질, 다양성, 제어 가능성을 결정합니다. 이 강의는 균일 난수부터 시작하여 현대 LLM 및 생성형 모델을 구동하는 기술까지 모든 주요 샘플링 방법을 처음부터 구축합니다.

## 개념

### 샘플링이 중요한 이유

샘플링은 AI 및 머신러닝 전반에서 네 가지 기본 역할로 나타납니다:

**생성.** 언어 모델, 확산 모델(Diffusion Model), GAN (생성적 적대 신경망)(GAN (Generative Adversarial Network))은 모두 샘플링을 통해 출력물을 생성합니다. 샘플링 알고리즘은 창의성, 일관성, 다양성을 직접적으로 제어합니다. 온도(Temperature), Top-k, 핵 샘플링 (Top-p)(Nucleus Sampling (Top-p))는 엔지니어가 매일 조절하는 주요 제어 수단입니다.

**학습.** 확률적 경사 하강법 (SGD)(Stochastic Gradient Descent (SGD))는 미니배치를 샘플링합니다. 드롭아웃(Dropout)은 뉴런을 샘플링하여 비활성화합니다. 데이터 증강(Data Augmentation)은 랜덤 변환을 샘플링합니다. 중요도 샘플링은 샘플의 가중치를 재조정하여 강화 학습(PPO, TRPO)에서 기울기 분산을 줄입니다.

**추정.** ML의 많은 양은 폐형(closed-form) 해가 없습니다. 데이터 분포에 대한 기대 손실, 에너지 기반 모델의 분할 함수, 베이지안 추론의 증거(evidence) 등이 이에 해당합니다. 몬테카를로(Monte Carlo) 추정은 샘플의 평균을 통해 이러한 모든 값을 근사합니다.

**탐색.** MCMC 알고리즘은 베이지안 추론에서 사후 분포를 탐색합니다. 진화 전략은 매개변수 Perturbation을 샘플링합니다. 톰슨 샘플링은 밴디트 문제에서 탐색과 활용의 균형을 맞춥니다.

핵심 과제는 다음과 같습니다. 단순한 분포(균일 분포, 정규 분포)에서만 직접 샘플링할 수 있습니다. 그 외의 모든 경우, 단순한 샘플을 목표 분포의 샘플로 변환하는 방법이 필요합니다.

### 균일 랜덤 샘플링

모든 샘플링 방법은 여기서 시작합니다. 균일 랜덤 넘버 생성기는 [0, 1) 범위의 값을 생성하며, 같은 길이의 모든 하위 구간은 동일한 확률을 가집니다.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

n개의 항목으로 구성된 이산 집합에서 균일하게 샘플링하려면 U를 생성하고 floor(n * U)를 반환합니다. 연속 범위 [a, b]에서 샘플링하려면 a + (b - a) * U를 계산합니다.

핵심 통찰은 다음과 같습니다. 하나의 균일 랜덤 넘버는 임의의 분포에서 하나의 샘플을 생성하는 데 필요한 정확한 양의 랜덤성을 포함하고 있습니다. 요령은 올바른 변환을 찾는 것입니다.

### 역 CDF 방법 (역 변환 샘플링)

누적 분포 함수(CDF)는 값을 확률로 매핑합니다:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

역 CDF는 확률을 값으로 다시 매핑합니다. U ~ Uniform(0, 1)이라면, X = F_inverse(U)는 목표 분포를 따릅니다.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**지수 분포 예시:**

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

F_inverse를 닫힌 형태로 표현할 수 있을 때 이 방법은 완벽하게 작동합니다. 정규 분포의 경우, 닫힌 형태의 역 CDF가 없으므로 다른 방법(Box-Muller 또는 수치적 근사)을 사용합니다.

**이산 버전:** 이산 분포의 경우, CDF를 누적 합으로 구성하고 U를 생성한 후, 누적 합이 U를 초과하는 첫 번째 인덱스를 찾습니다. 이는 06강에서 `sample_categorical`가 작동하는 방식입니다.

### 거절 샘플링

CDF를 역산할 수 없지만 상수까지 목표 PDF를 평가할 수 있다면, 거절 샘플링이 작동합니다.

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

경계 M이 더 타이트할수록 수용률이 높아집니다. 낮은 차원(1-3)에서는 거절 샘플링이 잘 작동합니다. 높은 차원에서는 제안 볼륨의 대부분이 거절되므로 수용률이 지수적으로 감소합니다. 이는 거절 샘플링에서의 차원의 저주입니다.

**예시: 잘린 정규 분포에서 샘플링하기.** 잘린 범위에 대한 균일 제안 분포를 사용합니다. 외피 M은 해당 범위에서 정규 PDF의 최댓값입니다.

**예시: 반원형에서 샘플링하기.** 경계 사각형 내에서 균일하게 제안합니다. 점이 반원 내부에 떨어지면 수용합니다. 이는 몬테 카를로가 pi를 계산하는 방식입니다: 수용률이 면적 비율 pi/4와 같습니다.

### 중요도 샘플링

때로는 목표 분포 p(x)에서 샘플이 필요하지 않습니다. p(x) 하의 기대값을 추정해야 하며, 다른 분포 q(x)에서 샘플을 가지고 있습니다.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

이는 강화 학습에서 매우 중요합니다. PPO (근접 정책 최적화)(Proximal Policy Optimization)에서는 이전 정책 pi_old 하에서 궤적을 수집하지만 새로운 정책 pi_new를 최적화해야 합니다. 중요도 가중치는 pi_new(a|s) / pi_old(a|s)입니다. PPO는 새로운 정책이 이전 정책에서 너무 멀리 벗어나지 않도록 이러한 가중치를 클리핑합니다.

중요도 샘플링 추정량의 분산은 q가 p와 얼마나 유사한지에 따라 달라집니다. q가 p와 매우 다르다면, 몇몇 샘플이 거대한 가중치를 받아 추정치를 지배하게 됩니다. 자기 정규화 중요도 샘플링은 가중치의 합으로 나누어 이 문제를 줄입니다:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 몬테 카를로 추정

몬테 카를로 추정은 랜덤 샘플의 평균으로 적분을 근사합니다. 대수의 법칙이 수렴을 보장합니다.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

오류율은 차원과 무관합니다. 격자 기반 적분이 불가능한 고차원에서는 몬테카를로 방법이 지배적인 이유입니다.

**pi 추정:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**기대값 추정:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 마르코프 체인 몬테카를로 (MCMC): 메트로폴리스-헤이스팅스

MCMC는 정상 분포가 목표 분포 p(x)인 마르코프 체인을 구성합니다. 충분한 단계가 지나면 체인에서 샘플링한 값은 (대략적으로) p(x)에서 샘플링한 값이 됩니다.

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

대칭 제안(q(x'|x) = q(x|x'))의 경우, 비율은 p(x')/p(x)로 단순화됩니다. 이것이 원래의 메트로폴리스 알고리즘입니다.

**작동 원리.** 수용 규칙은 상세 균형(detailed balance)을 보장합니다: x에 있고 x'로 이동할 확률은 x'에 있고 x로 이동할 확률과 같습니다. 상세 균형은 p(x)가 체인의 정상 분포임을 의미합니다.

**실용적 고려 사항:**
- 번인(Burn-in): 체인이 평형 상태에 도달하기 전의 초기 샘플은 버립니다
- 간격 샘플링(Thinning): 자기상관을 줄이기 위해 k번째 샘플마다 하나만 유지합니다
- 제안 스케일: 너무 작으면 체인이 느리게 이동합니다(높은 수용률, 느린 탐색); 너무 크면 대부분의 제안이 거부됩니다(낮은 수용률, 제자리걸음)
- 고차원에서 가우시안 제안의 최적 수용률은 약 0.234입니다

### 깁스 샘플링

깁스 샘플링은 다변량 분포를 위한 MCMC의 특수한 경우입니다. 모든 차원에서 한 번에 이동을 제안하는 대신, 조건부 분포에서 한 변수씩 업데이트합니다.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

깁스 샘플링은 각 조건부 분포 p(x_i | x_{-i})에서 샘플링할 수 있어야 합니다. 많은 모델에서는 이것이 간단합니다:
- 베이즈 네트워크: 조건부는 그래프 구조에서 유래합니다
- 가우시안 혼합: 조건부는 가우시안입니다
- 이징 모델: 각 스핀의 조건부는 이웃에만 의존합니다

정확한 조건부에서 샘플링하면 상세 균형이 자동으로 만족되므로, 수용률은 항상 1입니다(모든 제안이 수용됨).

**한계.** 변수가 서로 강하게 상관되어 있을 때, 깁스 샘플링은 분포를 따라 대각선 방향으로 큰 이동이 불가능하므로 수렴이 느려집니다.

### 온도 샘플링 (LLM에서 사용)

언어 모델은 어휘의 각 토큰에 대해 로짓 z_1, ..., z_V를 출력합니다. 소프트맥스는 이를 확률로 변환합니다. 온도는 소프트맥스 전에 로짓을 재조정합니다:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**작동 원리.** 로짓을 T < 1로 나누면 로짓 간의 차이가 증폭됩니다. z_1 = 2이고 z_2 = 1일 때, T = 0.5로 나누면 z_1/T = 4, z_2/T = 2가 되어 격차가 커집니다. 소프트맥스 이후, 가장 높은 로짓을 가진 토큰이 훨씬 더 큰 확률 비중을 차지합니다.

**실제 사용 시:**
- T = 0.0: 탐욕 디코딩, 사실 기반 Q&A에 최적
- T = 0.3-0.7: 약간의 창의성, 코드 생성에 적합
- T = 0.7-1.0: 균형 잡힌 설정, 일반적인 대화에 적합
- T = 1.0-1.5: 창의적 글쓰기, 브레인스토밍
- T > 1.5: 점점 더 무작위적, 거의 유용하지 않음

온도는 가능한 토큰의 종류를 바꾸지 않습니다. 각 토큰에 할당되는 확률 질량을 변경합니다.

### Top-k 샘플링

Top-k 샘플링은 후보 집합을 확률이 가장 높은 k개의 토큰으로 제한한 후, 정규화하여 그 제한된 집합에서 샘플링합니다.

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

Top-k는 어휘 분포의 긴 꼬리에 존재하는 극도로 가능성이 낮은 토큰(오타, 무의미한 텍스트)을 모델이 선택하는 것을 방지합니다. 문제는 k가 컨텍스트와 무관하게 고정된다는 점입니다. 모델이 확신할 때(한 토큰가 95% 확률), k = 40은 여전히 39개의 대안을 허용합니다. 모델이 불확신할 때(확률이 1000개 토큰에 분산), k = 40은 타당한 옵션을 잘라냅니다.

### Top-p (핵) 샘플링

Top-p 샘플링은 후보 집합의 크기를 동적으로 조정합니다. 고정된 수의 토큰을 유지하는 대신, 누적 확률이 p를 초과하는 가장 작은 토큰 집합을 유지합니다.

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

모델이 확신할 때, 핵 샘플링은 몇 개의 토큰만 유지합니다(아마도 2-3개). 모델이 불확신할 때는 많은 토큰을 유지합니다(아마도 200개). 이러한 적응적 행동 때문에 핵 샘플링은 일반적으로 Top-k보다 더 나은 텍스트를 생성합니다.

**일반적인 조합:**
- 온도(Temperature) 0.7 + top-p 0.9: 범용 설정으로 적합합니다
- 온도(Temperature) 0.0 (탐욕적 디코딩): 결정론적 작업에 가장 좋습니다
- 온도(Temperature) 1.0 + top-k 50: Fan et al. (2018) 원 논문 설정입니다

top-k와 top-p는 결합할 수 있습니다. top-k를 먼저 적용한 후, 남은 집합에 top-p를 적용해 보세요.

### 재파라미터화 트릭 (Reparameterization Trick) (VAE에서 사용)

변분 오토인코더 (VAE)(Variational Autoencoder (VAE))는 입력을 잠재 공간(Latent Space)의 분포로 인코딩하고, 그 분포에서 샘플링하며, 샘플을 디코딩하여 입력을 복원하는 방식으로 학습합니다. 문제점: 샘플링 연산에 대해 역전파(Backpropagation)할 수 없습니다.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

재파라미터화 트릭은 무작위성을 매개변수(Parameter)와 분리합니다:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

N(mu, sigma^2)는 mu + sigma * N(0, 1)과 동일한 분포를 가지므로 이 방법이 작동합니다. 핵심 통찰: 무작위성을 매개변수가 없는 소스(epsilon)로 이동한 후, 샘플을 매개변수에 대한 미분 가능한 변환으로 표현합니다.

**VAE 학습 루프에서:**
1. 인코더(Encoder)는 각 입력에 대해 mu와 log(sigma^2)를 출력합니다
2. epsilon ~ N(0, 1)을 샘플링합니다
3. z = mu + sigma * epsilon을 계산합니다
4. z를 디코딩하여 입력을 복원합니다
5. 단계 4, 3, 2, 1을 통해 역전파합니다 (단계 3이 미분 가능하므로 가능합니다)

재파라미터화 트릭 없이는 VAE를 표준 역전파로 학습할 수 없습니다. 이 하나의 통찰이 VAE를 실용적으로 만들었습니다.

### Gumbel-Softmax (미분 가능한 범주형 샘플링)

재파라미터화 트릭은 연속 분포(가우시안)에 대해 작동합니다. 이산 범주형 분포의 경우 다른 접근법이 필요합니다. Gumbel-Softmax는 범주형 샘플링에 대한 미분 가능한 근사를 제공합니다.

**Gumbel-Max 트릭 (미분 불가능):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (미분 가능한 근사):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax는 이산 샘플의 연속적 완화(relaxation)를 생성합니다. 출력은 하드 원-핫(hard one-hot)이 아닌 확률 벡터(소프트 원--hot)입니다. 기울기(Gradient)는 소프트맥스(Softmax)를 통해 흐릅니다. 학습 중 순방향 전달(forward pass)에서는 "straight-through" 추정기를 사용할 수 있습니다: 순방향 전달에는 하드 argmax를 사용하지만, 역방향 전달(backward pass)에는 소프트 Gumbel-Softmax 기울기를 사용합니다.

**응용 분야:**
- VAE의 이산 잠재 변수
- 신경망 아키텍처 검색 (이산 연산 선택)
- 하드 어텐션 메커니즘
- 이산 행동을 사용하는 강화 학습

### 층화 샘플링

표준 몬테카를로 샘플링은 우연히 샘플 공간에 공백을 남길 수 있습니다. 층화 샘플링은 공간을 층(strata)으로 나누고 각 층에서 샘플링하여 균일한 커버리지를 강제합니다.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

층화 샘플링은 표준 몬테카를로 샘플링보다 항상 분산이 낮거나 같습니다:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**응용 분야:**
- 수치 적분 (준 몬테카를로)
- 학습 데이터 분할 (각 폴드에서 클래스 균형 보장)
- 층화 결합 중요도 샘플링 (두 기법 결합)
- NeRF (신경 방사선 필드)는 카메라 광선을 따라 층화 샘플링을 사용합니다

### 확산 모델과의 연결

확산 모델은 샘플링 과정을 통해 이미지를 생성합니다. 순방향 과정은 T 단계에 걸쳐 이미지에 가우시안 노이즈를 추가하여 순수한 노이즈가 될 때까지 진행합니다. 역방향 과정은 노이즈를 제거하는 것을 학습하여 단계별로 원본 이미지를 복원합니다.

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

이 강의의 기법과의 연결:
- 각 노이즈 제거 단계는 재파라미터화 트릭을 사용합니다 (노이즈 샘플링, 결정적 변환 적용)
- 노이즈 스케줄 {alpha_t}는 온도 어닐링의 일종입니다
- 학습은 ELBO (증거 하한)를 근사하기 위해 몬테카를로 추정법을 사용합니다
- 확산 모델의 조상 샘플링은 마르코프 체인입니다 (각 단계는 현재 상태에만 의존)

이미지 생성 과정 전체는 반복적 샘플링입니다: 노이즈에서 시작하여, 각 단계에서 학습된 노이즈 제거 모델에 조건을 걸어 조금 더 노이즈가 적은 버전을 샘플링합니다.

```figure
monte-carlo-pi
```

## 구현하기

### 1단계: 균일 분포 및 역 CDF 샘플링

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000개의 지수 분포 샘플을 생성하고 평균이 1/lambda인지 확인해 보세요.

### 2단계: 거절 샘플링

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

거절 샘플링을 사용하여 절단된 정규 분포에서 샘플을 추출하세요. 샘플을 히스토그램으로 그려 형태를 확인해 보세요.

### 3단계: 중요도 샘플링

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

정규 분포에서 균일 제안 분포를 사용하여 E[X^2]를 추정해 보세요. 알려진 정답(mu^2 + sigma^2)과 비교해 보세요.

### 4단계: 파이(pi)의 몬테카를로 추정

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

### 5단계: Metropolis-Hastings MCMC

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

이봉(bimodal) 분포(두 가우시안의 혼합)에서 샘플링하고 체인의 궤적을 시각화해 보세요.

### 6단계: 깁스 샘플링

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

### 7단계: 온도 샘플링

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

온도가 토큰 로짓(logits) 집합의 출력 분포를 어떻게 변화시키는지 보여주세요.

### 8단계: Top-k 및 Top-p 샘플링

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

### 9단계: 재파라미터화 트릭

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

기울기가 재파라미터화된 샘플을 통해 흐르지만, 직접 샘플링을 통해서는 흐르지 않는다는 것을 시연해 보세요.

### 10단계: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

온도를 낮추면 출력이 원-핫(one-hot) 벡터에 가까워지는 과정을 보여주세요.

모든 시각화를 포함한 전체 구현은 `code/sampling.py`에 있습니다.

## 사용하기

NumPy와 SciPy를 사용한 프로덕션 버전은 다음과 같습니다:

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

대규모 MCMC를 위해 전용 라이브러리를 사용하세요:
- PyMC: NUTS(적응형 HMC)를 사용한 완전한 베이지안 모델링
- emcee: 앙상블 MCMC 샘플러
- NumPyro/JAX: GPU 가속 MCMC

이들을 처음부터 구축했습니다. 이제 라이브러리 호출이 무엇을 수행하는지 알 수 있습니다.

## 연습 문제

1. 코시(Cauchy) 분포에 대한 역 CDF 샘플링을 구현하세요. CDF는 F(x) = 0.5 + arctan(x)/pi입니다. 10,000개의 샘플을 생성하고 히스토그램을 실제 PDF와 비교해 보세요. 무거운 꼬리(중심에서 멀리 떨어진 극단값)를 확인하세요.

2. Uniform(0, 1) 제안 분포를 사용하여 Beta(2, 5) 분포에서 샘플을 생성하기 위해 기각 샘플링을 사용하세요. 승인된 샘플을 실제 Beta PDF와 비교하여 플롯하세요. 이론적인 승인율은 얼마인가요?

3. 1,000, 10,000, 100,000개의 샘플을 사용하여 몬테카를로 방식으로 sin(x)의 0에서 pi까지의 적분을 추정하세요. 각 수준에서의 오차를 비교하세요. 오차가 O(1/sqrt(N))으로 스케일링됨을 검증하세요.

4. exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)에 비례하는 2D 분포 p(x, y)에서 샘플링하기 위해 Metropolis-Hastings를 구현하세요. 샘플과 체인 궤적을 플롯하세요. 다양한 제안 표준 편차를 실험해 보세요.

5. 완전한 텍스트 생성 데모를 구축해 보세요: 10개의 단어 어휘와 로짓(Logits)이 주어졌을 때, (a) 탐욕(greedy), (b) 온도(Temperature)=0.7, (c) Top-k=3, (d) Top-p=0.9를 사용하여 20개 토큰의 시퀀스를 생성합니다. 5번의 실행에 걸쳐 출력의 다양성을 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 샘플링(Sampling) | "무작위 값 그리기" | 확률 분포에 따라 값을 생성하는 것. 모든 생성형 AI의 메커니즘 |
| 균등 분포(Uniform distribution) | "모두 동일한 가능성" | [a, b]의 모든 값이 동일한 확률 밀도 1/(b-a)를 가짐. 모든 샘플링 방법의 시작점 |
| 역 CDF(Inverse CDF) | "확률 변환" | F_inverse(U)는 균등 샘플을 알려진 CDF를 가진 임의의 분포 샘플로 변환합니다. 정확하고 효율적 |
| 기각 샘플링(Rejection sampling) | "제안하고 수용/기각하기" | 간단한 제안 분포에서 생성하고, 목표/제안 비율에 비례하는 확률로 수용합니다. 정확하지만 샘플을 낭비함 |
| 중요도 샘플링(Importance sampling) | "샘플 가중치 재조정" | q(x)에서 샘플링하여 각 샘플에 p(x)/q(x) 가중치를 적용해 p(x) 하의 기대값을 추정합니다. RL에서 PPO의 핵심 |
| 몬테 카를로(Monte Carlo) | "무작위 샘플 평균 내기" | 적분을 샘플 평균으로 근사합니다. 차원과 무관하게 오차는 O(1/sqrt(N)) |
| MCMC | "수렴하는 랜덤 워크" | 정상 분포가 목표 분포인 마르코프 체인을 구성합니다. Metropolis-Hastings는 기초 알고리즘 |
| Metropolis-Hastings | "상승은 수용, 하강은 때때로 수용" | 이동 제안, 밀도 비율에 기반해 수용합니다. 상세 균형(Detailed balance)이 목표 분포로의 수렴을 보장 |
| 깁스 샘플링(Gibbs sampling) | "한 번에 하나의 변수" | 다른 변수는 고정하고 각 변수를 조건부 분포에서 업데이트합니다. 수용률 100% |
| 온도(Temperature) | "확신 조절 나사" | Softmax 전에 로짓(Logits)을 T로 나눕니다. T<1은 날카롭게(더 확신), T>1은 평평하게(더 다양) 만듭니다 |
| Top-k 샘플링(Top-k sampling) | "k개의 최상위 유지" | 가장 확률이 높은 k개 토큰을 제외한 나머지를 0으로 만들고, 정규화한 후 샘플링합니다. 후보 집합 크기가 고정 |
| 핵 샘플링 (Top-p)(Nucleus sampling (top-p)) | "확률 있는 것 유지" | 누적 확률이 p를 초과하는 가장 작은 토큰 집합을 유지합니다. 후보 집합 크기가 적응적 |
| 재파라미터화 트릭(reparameterization trick) | "무작위성을 외부로 이동" | z = mu + sigma * epsilon (epsilon ~ N(0,1))로 작성합니다. 샘플링을 미분 가능하게 만듭니다. VAE 학습에 필수적입니다 |
| Gumbel-Softmax | "소프트 범주형 샘플링" | Gumbel 노이즈와 온도 매개변수를 사용한 softmax를 통해 범주형 샘플링을 미분 가능하게 근사합니다 |
| 층화 샘플링(stratified sampling) | "강제 커버리지" | 샘플 공간을 층(strata)으로 나누고 각 층에서 샘플링합니다. 항상 단순 몬테카를로보다 분산이 낮습니다 |
| 번인(burn-in) | "워밍업 기간" | 체인이 정상 분포(stationary distribution)에 도달하기 전의 초기 MCMC 샘플을 버립니다 |
| 상세 균형(detailed balance) | "가역성 조건" | p(x) * T(x->y) = p(y) * T(y->x). 마르코프 체인의 정상 분포가 p가 되기 위한 충분 조건입니다 |
| 확산 샘플링(diffusion sampling) | "반복적 디노이징" | 노이즈에서 시작하여 학습된 디노이징 단계를 적용하여 데이터를 생성합니다. 각 단계는 조건부 샘플링 연산입니다 |

## 추가 읽기

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 기초에 대한 상세 튜토리얼
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 원 논문
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 핵 샘플링(top-p) 논문
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 재파라미터화 트릭을 도입한 VAE 논문
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM은 샘플링을 이미지 생성과 연결합니다
