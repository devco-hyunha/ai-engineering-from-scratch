# 머신러닝을 위한 미적분 (Calculus for Machine Learning)

> 도함수는 어느 쪽이 내리막인지 알려 줍니다. 신경망이 학습하는 데 필요한 전부입니다.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- 흔한 ML 함수(x^2, sigmoid, cross-entropy)에 대해 수치·해석 도함수를 계산합니다
- 손실 함수를 최소화하기 위해 1D·2D에서 처음부터 경사하강법을 구현합니다
- 선형 회귀 모델의 기울기를 유도하고 수동 가중치 갱신으로 학습합니다
- 헤세 행렬, 테일러 급수 근사, 그리고 최적화 방법과의 연결을 설명합니다

## 문제 상황 (The Problem)

수백만 가중치를 가진 신경망이 있습니다. 각 가중치는 노브입니다. 모델을 조금 덜 틀리게 만들려면 모든 노브를 어느 방향으로 돌려야 하는지 알아내야 합니다. 미적분이 그 방향을 줍니다.

미적분 없이는 신경망 학습이 무작위 변화를 시도하며 최선이기를 바라는 일이 됩니다. 도함수가 있으면 각 가중치가 오차에 어떻게 영향을 주는지 정확히 압니다. 매번 모든 노브를 올바른 방향으로 돌립니다.

## 핵심 개념 (The Concept)

### 도함수란 무엇인가?

도함수는 변화율을 측정합니다. 함수 y = f(x)에서 도함수 f'(x)는 이렇게 말합니다: x를 아주 조금 밀면 y는 얼마나 바뀌는가?

기하적으로 도함수는 한 점에서의 접선 기울기입니다.

**f(x) = x^2:**

| x | f(x) | f'(x) (기울기) |
|---|------|---------------|
| 0 | 0    | 0 (평평, 바닥) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (이 점에서의 접선 기울기) |
| 3 | 9    | 6 |

x=2에서 기울기는 4입니다. x를 오른쪽으로 아주 조금 옮기면 y는 대략 그 양의 4배만큼 증가합니다. x=0에서 기울기는 0입니다. 그릇의 바닥에 있습니다.

형식적 정의:

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

코드에서는 극한을 건너뛰고 아주 작은 h를 씁니다. 그것이 수치 도함수입니다.

### 편도함수: 한 번에 한 변수

실제 함수는 입력이 많습니다. 신경망 손실은 수천 개의 가중치에 의존합니다. 편도함수는 하나를 제외한 모든 변수를 상수로 두고, 그 하나에 대해 미분합니다.

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

각 편도함수는 이렇게 답합니다: 이 가중치만 조금 밀면 손실은 어떻게 바뀌는가?

### 기울기: 모든 편도함수의 벡터

기울기(gradient)는 모든 편도함수를 하나의 벡터로 모읍니다. 함수 f(x, y, z)에서 기울기는:

```
grad f = [ df/dx, df/dy, df/dz ]
```

기울기는 가장 가파른 상승 방향을 가리킵니다. 함수를 최소화하려면 반대 방향으로 갑니다.

**f(x,y) = x^2 + y^2의 등고선:**

함수는 등고선이 동심원인 그릇 모양입니다. 최솟값은 (0, 0)에 있습니다.

| 점 | grad f | -grad f (하강 방향) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (최솟값에서 멀어지는 오르막) | [-2, -2] (최솟값을 향한 내리막) |
| (0, 0) | [0, 0] (평평, 최솟값) | [0, 0] |

이것이 그림으로 본 경사하강법입니다. 기울기를 계산하고, 부호를 바꾸고, 한 걸음 내딛습니다.

### 최적화와의 연결

신경망 학습은 최적화입니다. 모델이 얼마나 틀렸는지 측정하는 손실 함수 L(w1, w2, ..., wn)가 있고, 이를 최소화하려 합니다.

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

학습률은 걸음 크기를 조절합니다. 너무 크면 오버슈트합니다. 너무 작으면 기어갑니다.

**손실 지형 (1D 단면):**

손실 함수 L(w)는 가중치 w가 변함에 따라 봉우리와 골짜기가 있는 곡선을 만듭니다.

| 특징 | 설명 |
|---------|-------------|
| 전역 최솟값 | 전체 곡선에서 가장 낮은 점 — 최선의 해 |
| 국소 최솟값 | 이웃보다 낮지만 전체에서 가장 낮지는 않은 골짜기 |
| 기울기 | 경사하강법은 어떤 시작점에서든 내리막을 따름 |

경사하강법은 내리막을 따릅니다. 국소 최솟값에 갇힐 수 있지만, 고차원 공간(수백만 가중치)에서는 실무상 거의 문제가 되지 않습니다.

### 수치 도함수 vs 해석 도함수

도함수를 계산하는 방법은 두 가지입니다.

해석적: 미적분 규칙을 손으로 적용합니다. f(x) = x^2이면 도함수는 f'(x) = 2x. 정확하고 빠릅니다.

수치적: 정의를 사용해 근사합니다. 아주 작은 h에 대해 f(x+h)와 f(x-h)를 계산한 뒤 차이를 씁니다.

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

수치 도함수는 느리지만 어떤 함수에도 동작합니다. 해석 도함수는 빠르지만 공식을 유도해야 합니다. 신경망 프레임워크는 세 번째 접근을 씁니다: 자동미분. 정확한 도함수를 기계적으로 계산합니다. Phase 3에서 봅니다.

### 간단한 함수의 손으로 구한 도함수

ML에서 반복해서 보는 도함수들입니다.

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

f(x) = x^2에 대해:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

f(w) = wx + b (x=3, b=1)에 대해:

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 연쇄법칙 (The chain rule)

함수가 합성되면 연쇄법칙이 미분 방법을 알려 줍니다.

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

신경망은 함수의 사슬입니다: input -> linear -> activation -> linear -> activation -> loss. 역전파는 출력에서 입력으로 연쇄법칙을 반복 적용한 것입니다. 그게 알고리즘의 전부입니다.

### 헤세 행렬 (The Hessian Matrix)

기울기는 경사를 알려 줍니다. 헤세는 곡률을 알려 줍니다.

헤세는 2차 편도함수의 행렬입니다. 함수 f(x1, x2, ..., xn)에서 헤세의 (i, j) 항목은:

```
H[i][j] = d^2f / (dx_i * dx_j)
```

2변수 함수 f(x, y)에 대해:

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**임계점(기울기 = 0)에서 헤세가 알려 주는 것:**

| 헤세 성질 | 의미 | 예제 곡면 |
|-----------------|---------|-----------------|
| 양의 정부호 (모든 고유값 > 0) | 국소 최솟값 | 위로 열린 그릇 |
| 음의 정부호 (모든 고유값 < 0) | 국소 최댓값 | 아래로 열린 그릇 |
| 부정부호 (혼합 고유값) | 안장점 | 말안장 모양 |

**예:** f(x, y) = x^2 - y^2 (안장 함수)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

f(x, y) = x^2 + y^2 (그릇)과 비교:

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**ML에서 헤세가 중요한 이유:**

뉴턴 방법은 헤세를 사용해 경사하강법보다 나은 최적화 걸음을 합니다. 경사만 따르는 대신 곡률을 반영합니다:

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

뉴턴 방법이 더 빨리 수렴하는 이유는 헤세가 기울기를 "재스케일"하기 때문입니다 — 가파른 방향은 작은 걸음, 평평한 방향은 큰 걸음.

문제는: 파라미터가 N개인 신경망에서 헤세는 N x N입니다. 파라미터가 100만 개면 1조 개의 항목이 필요합니다. 그래서 근사를 씁니다.

| 방법 | 사용하는 것 | 비용 | 수렴 |
|--------|-------------|------|-------------|
| 경사하강법 | 1차 도함수만 | 걸음당 O(N) | 느림 (선형) |
| 뉴턴 방법 | 전체 헤세 | 걸음당 O(N^3) | 빠름 (이차) |
| L-BFGS | 기울기 이력으로 헤세 근사 | 걸음당 O(N) | 중간 (초선형) |
| Adam | 파라미터별 적응형 비율 (대각 헤세 근사) | 걸음당 O(N) | 중간 |
| Natural gradient | Fisher 정보 행렬 (통계적 헤세) | 걸음당 O(N^2) | 빠름 |

실무에서 Adam이 딥러닝의 기본 옵티마이저입니다. 파라미터별 기울기의 이동 평균과 분산을 추적해 2차 정보를 싸게 근사합니다.

### 테일러 급수 근사

임의의 매끄러운 함수는 국소적으로 다항식으로 근사할 수 있습니다:

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

항을 많이 넣을수록 근사는 좋아집니다 — 다만 점 x 근처에서만.

**ML에서 테일러 급수가 중요한 이유:**

- **1차 테일러 = 경사하강법.** f(x + h) ~ f(x) + f'(x)*h를 쓰면 선형 근사입니다. 경사하강법은 이 선형 모델을 최소화해 h = -lr * f'(x)를 고릅니다.

- **2차 테일러 = 뉴턴 방법.** f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2를 쓰면 이차 모델입니다. 최소화하면 h = -f'(x)/f''(x) — 뉴턴 걸음입니다.

- **손실 함수 설계.** MSE와 cross-entropy는 매끄러워서 테일러 전개가 잘 동작합니다. 우연이 아닙니다. 매끄러운 손실은 최적화를 예측 가능하게 만듭니다.

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

핵심 통찰: 모든 기울기 기반 최적화는 사실 손실 함수를 국소적으로 근사하고, 그 근사의 최솟값으로 한 걸음 내딛는 일입니다.

### ML에서의 적분

도함수는 변화율을 알려 줍니다. 적분은 누적 — 곡선 아래 면적 — 을 계산합니다.

ML에서 적분을 손으로 계산하는 일은 드물지만, 개념은 어디에나 있습니다:

**확률.** 밀도 p(x)를 가진 연속 확률변수에 대해:
```
P(a < X < b) = integral from a to b of p(x) dx
```
a와 b 사이 확률밀도 곡선 아래 면적이 그 구간에 떨어질 확률입니다.

**기댓값.** 확률로 가중된 평균 결과:
```
E[f(X)] = integral of f(x) * p(x) dx
```
데이터 분포에 대한 기댓값 손실은 적분입니다. 학습은 이에 대한 경험적 근사를 최소화합니다.

**KL 발산.** 두 분포가 얼마나 다른지 측정합니다:
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
VAE, knowledge distillation, 베이지안 추론에 쓰입니다.

**정규화 상수.** 베이지안 추론에서:
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
분모는 가능한 모든 파라미터 값에 대한 적분입니다. 종종 다루기 어려워 MCMC와 변분 추론 같은 근사를 씁니다.

| 적분 개념 | ML에서 등장하는 곳 |
|-----------------|----------------------|
| 곡선 아래 면적 | 밀도 함수로부터의 확률 |
| 기댓값 | 손실 함수, 위험 최소화 |
| KL 발산 | VAE, 정책 최적화, distillation |
| 정규화 | 베이지안 사후분포, softmax 분모 |
| 주변 가능도 | 모델 비교, evidence lower bound (ELBO) |

### 계산 그래프에서의 다변수 연쇄법칙

연쇄법칙은 스칼라 함수의 직선에만 적용되지 않습니다. 신경망에서는 변수가 갈라지고 합쳐집니다. 간단한 순전파에서 도함수가 흐르는 방식입니다:

```mermaid
graph LR
    x["x (입력)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"손실 함수"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

역전파는 오른쪽에서 왼쪽으로 기울기를 계산합니다:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

각 화살표는 국소 도함수를 곱합니다. 어떤 파라미터의 기울기도 손실에서 그 파라미터까지 경로를 따른 모든 국소 도함수의 곱입니다. 경로가 갈라지고 합쳐지면 기여를 더합니다(다변수 연쇄법칙).

역전파의 전부는 이것입니다: 계산 그래프를 통해 출력에서 입력으로 체계적으로 적용된 연쇄법칙.

### 야코비 행렬 (The Jacobian matrix)

함수가 벡터를 벡터로 매핑할 때(신경망 레이어처럼) 도함수는 행렬입니다. 야코비는 모든 출력의 모든 입력에 대한 편도함수를 담습니다.

f: R^n -> R^m에 대해 야코비 J는 m x n 행렬입니다:

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

신경망에서 야코비를 손으로 계산하지는 않습니다. PyTorch가 처리합니다. 하지만 존재한다는 사실을 알면 역전파의 shape를 이해하는 데 도움이 됩니다: 레이어가 R^n을 R^m으로 매핑하면 야코비는 m x n입니다. 기울기는 이 행렬의 전치를 통해 뒤로 흐릅니다.

### 신경망에서 왜 중요한가

신경망의 모든 가중치에 기울기가 붙습니다. 기울기는 손실을 줄이기 위해 그 가중치를 어떻게 조정할지 알려 줍니다.

```mermaid
graph LR
    subgraph Forward["순전파"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["역전파"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

각 가중치 갱신:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

순전파는 예측과 손실을 계산합니다. 역전파는 모든 가중치에 대한 손실의 기울기를 계산합니다. 그다음 모든 가중치가 내리막으로 작은 걸음을 내딛습니다. 수백만 걸음 반복합니다. 그것이 딥러닝입니다.

```figure
derivative-tangent
```

## 구현하기 (Build It)

### Step 1: 처음부터 수치 도함수

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

수치 도함수가 해석 도함수와 소수점 여러 자리까지 일치합니다.

### Step 2: 편도함수와 기울기

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### Step 3: f(x) = x^2의 최솟값을 찾는 경사하강법

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

x=5에서 시작해 매 걸음마다 최솟값 x=0에 더 가까워집니다.

### Step 4: 2D 함수에서의 경사하강법

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### Step 5: 수치 도함수와 해석 도함수 비교

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### Step 6: 헤세를 수치적으로 계산하기

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

안장 함수의 헤세는 고유값 2와 -2(혼합 부호, 안장점 확인)를 가집니다. 그릇은 고유값 2와 2(둘 다 양수, 최솟값 확인)를 가집니다.

### Step 7: 동작하는 테일러 근사

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

x0=0 근처에서 sin(x) ~ x (1차 테일러)입니다. 작은 h에서는 근사가 훌륭하지만 큰 h에서는 무너집니다. 그래서 경사하강법은 작은 학습률에서 가장 잘 동작합니다 — 매 걸음이 선형 근사가 정확하다고 가정합니다.

### Step 8: 신경망에서 왜 중요한가

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

모든 기울기 기반 학습 루프가 이 패턴을 따릅니다: 예측, 손실 계산, 기울기 계산, 가중치 갱신.

## 실용 활용 (Use It)

NumPy를 쓰면 같은 연산이 더 빠르고 간결합니다:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

방금 경사하강법을 처음부터 만들었습니다. PyTorch는 기울기 계산을 자동화하지만, 갱신 루프는 동일합니다.

## 연습 문제 (Exercises)

1. `numerical_derivative`를 두 번 호출해 `numerical_second_derivative(f, x)`를 구현하세요. x=2에서 x^3의 2차 도함수가 12인지 검증하세요.
2. 경사하강법으로 f(x, y) = (x - 3)^2 + (y + 1)^2의 최솟값을 찾으세요. (0, 0)에서 시작하세요. (3, -1)로 수렴해야 합니다.
3. 경사하강법 루프에 모멘텀을 추가하세요: 과거 기울기를 누적하는 속도 벡터를 유지합니다. f(x) = x^4 - 3x^2에서 모멘텀 유무의 수렴 속도를 비교하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 도함수 (Derivative) | "기울기" | 한 점에서 함수의 변화율. 입력이 한 단위 바뀔 때 출력이 얼마나 바뀌는지. |
| 편도함수 | "한 변수의 도함수" | 나머지를 상수로 두고 한 변수에 대해 취한 도함수. |
| 기울기 (Gradient) | "가장 가파른 상승 방향" | 모든 편도함수의 벡터. 함수를 가장 빨리 키우는 방향을 가리킴. |
| 경사하강법 | "내리막으로 가라" | 파라미터에서 기울기(곱하기 학습률)를 빼 손실을 줄임. 신경망 학습의 핵심. |
| 학습률 | "걸음 크기" | 경사하강법 걸음이 얼마나 큰지 조절하는 스칼라. 너무 크면 발산, 너무 작으면 느리게 수렴. |
| 연쇄법칙 | "도함수를 곱하라" | 합성 함수 미분 규칙: df/dx = df/dg * dg/dx. 역전파의 수학적 기초. |
| 야코비 (Jacobian) | "도함수의 행렬" | 벡터를 벡터로 매핑하는 함수에서, 출력의 입력에 대한 모든 편도함수 행렬. |
| 수치 도함수 | "유한 차분" | 근처 두 점에서 함수를 평가해 그 사이 기울기로 도함수를 근사. |
| 역전파 | "역모드 자동미분" | 연쇄법칙으로 출력에서 입력까지 레이어별로 기울기를 계산. 신경망이 학습하는 방식. |
| 헤세 (Hessian) | "2차 도함수의 행렬" | 모든 2차 편도함수의 행렬. 함수의 곡률을 기술. 임계점에서 양의 정부호면 국소 최솟값. |
| 테일러 급수 | "다항식 근사" | 도함수로 점 근처 함수를 근사: f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... 경사하강법과 뉴턴 방법이 왜 동작하는지 이해하는 기초. |
| 적분 | "곡선 아래 면적" | 구간에 걸친 양의 누적. ML에서 적분은 확률, 기댓값, KL 발산을 정의. |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - 도함수, 적분, 연쇄법칙에 대한 시각적 직관
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - 기울기가 신경망 레이어를 통해 어떻게 흐르는지
