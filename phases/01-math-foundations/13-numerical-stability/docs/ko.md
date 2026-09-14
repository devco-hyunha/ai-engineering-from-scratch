# 수치 안정성 (Numerical Stability)

> 부동소수점은 새는 추상화입니다. 학습 중에 당신을 물어뜯을 것이고, 다가오는 것도 보지 못할 것입니다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## 학습 목표 (Learning Objectives)

- max 빼기(max-subtraction) 트릭으로 수치적으로 안정적인 softmax와 log-sum-exp를 구현합니다
- 부동소수점 계산에서 오버플로(overflow), 언더플로(underflow), 파국적 소거(catastrophic cancellation)를 식별합니다
- 중심 유한차분(centered finite differences)으로 해석적 그래디언트를 수치 그래디언트와 대조해 검증합니다
- 학습에서 float16보다 bfloat16이 선호되는 이유와, 손실 스케일링(loss scaling)이 그래디언트 언더플로를 막는 방식을 설명합니다

## 문제 상황 (The Problem)

모델이 세 시간 동안 학습하다가 손실이 NaN이 됩니다. print를 넣습니다. 스텝 9,000에서는 logits가 정상입니다. 스텝 9,001에서는 `inf`입니다. 스텝 9,002에서는 모든 그래디언트가 `nan`이 되고 학습은 끝납니다.

또는: 모델은 끝까지 학습되지만 정확도가 논문보다 2% 낮습니다. 전부 점검합니다. 아키텍처 일치. 하이퍼파라미터 일치. 데이터 일치. 문제는 논문이 float32를 썼고 당신은 올바른 스케일링 없이 float16을 썼다는 점입니다. 쌓인 반올림 오차 32비트가 조용히 정확도를 갉아먹었습니다.

또는: 교차 엔트로피 손실을 처음부터 구현합니다. 작은 logits에서는 동작합니다. logits가 100을 넘으면 `inf`를 반환합니다. softmax가 오버플로한 것입니다. `exp(100)`은 float32가 표현할 수 있는 범위를 넘습니다. 모든 ML 프레임워크는 두 줄짜리 트릭으로 이를 처리합니다. 당신은 그 트릭이 있는지조차 몰랐습니다.

수치 안정성은 이론적 관심이 아닙니다. 성공하는 학습 실행과 조용히 실패하는 실행의 차이입니다. 결국 디버깅하게 될 심각한 ML 버그는 거의 모두 부동소수점으로 귀결됩니다.

## 핵심 개념 (The Concept)

### IEEE 754: 컴퓨터가 실수를 저장하는 방식

컴퓨터는 IEEE 754 표준에 따라 실수를 부동소수점 값으로 저장합니다. float에는 부호 비트, 지수(exponent), 가수(mantissa, significand) 세 부분이 있습니다.

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

가수는 정밀도(유효 자릿수)를, 지수는 범위(얼마나 크거나 작을 수 있는지)를 결정합니다.

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32는 약 7자리 십진 정밀도를 줍니다. 즉 1.0000001과 1.0000002는 구분하지만 1.00000001과 1.00000002는 구분하지 못합니다. 7자리 이후는 전부 반올림 잡음입니다.

float16은 약 3자리입니다. 표현 가능한 최댓값은 65,504입니다. logits, 그래디언트, 활성화가 일상적으로 이 값을 넘는 ML에서는 불안할 정도로 작습니다.

bfloat16은 float16의 범위 문제에 대한 Google의 답입니다. float32와 같은 8비트 지수(같은 범위, 최대 3.4e38)를 갖지만 가수 비트는 7개뿐입니다(float16보다 정밀도는 낮음). 신경망 학습에서는 정밀도보다 범위가 더 중요하므로 보통 bfloat16이 이깁니다.

### 왜 0.1 + 0.2 != 0.3인가

숫자 0.1은 이진 부동소수점으로 정확히 표현할 수 없습니다. 기수 2에서는 순환 분수입니다:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32는 이를 가수 23비트로 자릅니다. 저장된 값은 대략 0.100000001490116입니다. 마찬가지로 0.2는 대략 0.200000002980232로 저장됩니다. 합은 0.300000004470348이지 0.3이 아닙니다.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

이것이 ML에서 중요한 이유:

1. `if loss < threshold` 같은 손실 비교가 잘못된 답을 줄 수 있습니다
2. 많은 작은 값의 누적(수천 스텝에 걸친 그래디언트 업데이트)이 참 합에서 벗어납니다
3. `==`로 float를 비교하면 체크섬과 재현성 테스트가 실패합니다

해결: float를 `==`로 비교하지 마세요. `abs(a - b) < epsilon` 또는 `math.isclose()`를 사용하세요.

### 파국적 소거 (Catastrophic Cancellation)

거의 같은 두 부동소수점을 빼면 유효 자릿수가 상쇄되고, 반올림 잡음이 앞자리로 올라옵니다.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

한 번의 뺄셈으로 상대 오차 19%입니다. ML에서는 다음 경우에 발생합니다:

- 평균이 큰 데이터의 분산 계산: E[x]가 클 때 `E[x^2] - E[x]^2`
- 거의 같은 로그 확률끼리의 뺄셈
- epsilon이 너무 작은 유한차분 그래디언트

해결: 크고 거의 같은 수를 빼지 않도록 식을 재배열하세요. 분산에는 Welford 알고리즘을 쓰거나 먼저 데이터를 중심화하세요. 로그 확률은 끝까지 로그 공간에서 다루세요.

### 오버플로와 언더플로

오버플로는 결과가 표현하기에 너무 클 때 발생합니다. 언더플로는 너무 작을 때(표현 가능한 최소 양수보다 0에 가까울 때) 발생합니다.

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()`는 ML에서 오버플로의 주요 원인입니다:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()`는 반대 방향입니다:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

ML에서 `exp()`는 softmax, sigmoid, 확률 계산에 등장합니다. `log()`는 교차 엔트로피, 로그 가능도, KL 발산에 등장합니다. `log(exp(x))` 조합은 올바른 트릭 없이는 지뢰밭입니다.

### Log-Sum-Exp 트릭

`log(sum(exp(x_i)))`를 직접 계산하는 것은 수치적으로 위험합니다. 어떤 `x_i`가 크면 `exp(x_i)`가 오버플로합니다. 모든 `x_i`가 매우 음수이면 모든 `exp(x_i)`가 0으로 언더플로하고 `log(0)`은 `-inf`입니다.

트릭: 지수화하기 전에 최댓값을 뺍니다.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

이유: `max(x)`를 뺀 뒤 가장 큰 지수는 `exp(0) = 1`입니다. 오버플로가 불가능합니다. 합의 항 중 적어도 하나는 1이므로 합은 최소 1이고, `log(1) = 0`입니다. `-inf`로의 언더플로도 불가능합니다.

증명:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`로 두면 오버플로가 제거됩니다.

이 트릭은 ML 어디에나 등장합니다:
- Softmax 정규화
- 교차 엔트로피 손실 계산
- 시퀀스 모델의 로그 확률 합산
- 가우시안 혼합
- 변분 추론

### Softmax에 Max 빼기 트릭이 필요한 이유

Softmax는 logits를 확률로 바꿉니다:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

트릭 없이 [100, 101, 102] logits는 오버플로를 일으킵니다:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

트릭을 쓰면 max(x) = 102를 뺍니다:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

확률은 동일합니다. 계산은 안전합니다. 이것은 최적화가 아닙니다. 정확성을 위한 필수 조건입니다.

### NaN과 Inf: 탐지와 예방

`nan`(Not a Number)과 `inf`(infinity)는 계산을 통해 바이러스처럼 전파됩니다. 그래디언트 업데이트에 `nan`이 하나 있으면 가중치가 `nan`이 되고, 이후 모든 출력이 `nan`이 됩니다. 한 스텝 만에 학습이 끝납니다.

`inf`가 나타나는 방식:
- 큰 양수에 대한 `exp()`
- 0으로 나누기: `1.0 / 0.0`
- 누적에서의 `float32` 오버플로

`nan`이 나타나는 방식:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수에 대한 `sqrt()`
- 음수에 대한 `log()`
- 이미 있는 `nan`이 관련된 모든 산술

탐지:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

예방 전략:

1. `exp()` 입력을 클램프: `exp(clamp(x, -80, 80))`
2. 분모에 epsilon 추가: `x / (y + 1e-8)`
3. `log()` 안에 epsilon 추가: `log(x + 1e-8)`
4. 안정적 구현 사용 (log-sum-exp, stable softmax)
5. 가중치 폭주를 막는 그래디언트 클리핑
6. 디버깅 중 매 forward 패스 후 `nan`/`inf` 검사

### 수치 그래디언트 검사

해석적 그래디언트(역전파)에는 버그가 있을 수 있습니다. 수치 그래디언트 검사는 유한차분으로 그래디언트를 계산해 이를 검증합니다.

중심차분 공식:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

이것은 O(h^2) 정확도이며, O(h)뿐인 전방차분 `(f(x+h) - f(x)) / h`보다 훨씬 낫습니다.

h 선택: 너무 크면 근사가 틀리고, 너무 작으면 파국적 소거가 답을 망칩니다. 보통 `h = 1e-5` ~ `1e-7`입니다.

검사: 해석적·수치 그래디언트 사이의 상대 오차를 계산합니다.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

경험 규칙:
- relative_error < 1e-7: 완벽, 그래디언트가 맞음
- relative_error < 1e-5: 허용 가능, 대체로 맞음
- relative_error > 1e-3: 무언가 잘못됨
- relative_error > 1: 그래디언트가 완전히 틀림

새 레이어나 손실 함수를 구현할 때마다 반드시 그래디언트를 검사하세요. PyTorch는 이를 위해 `torch.autograd.gradcheck()`를 제공합니다.

### 혼합 정밀도 학습 (Mixed Precision Training)

현대 GPU에는 float16 행렬 곱을 float32보다 2–8배 빠르게 계산하는 전용 하드웨어(Tensor Cores)가 있습니다. 혼합 정밀도 학습은 이를 활용합니다:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

순수 float16 학습의 문제: 그래디언트가 종종 매우 작습니다(1e-8 이하). Float16은 ~6e-8 미만을 0으로 언더플로합니다. 모든 그래디언트 업데이트가 0이 되어 모델이 학습을 멈춥니다.

해결은 손실 스케일링입니다:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

동적 손실 스케일링은 스케일 인자를 자동으로 조정합니다. 큰 값(65536)으로 시작합니다. 그래디언트가 `inf`로 오버플로하면 반으로 줄입니다. N 스텝 동안 오버플로가 없으면 두 배로 늘립니다.

### bfloat16 vs float16: 학습에서 bfloat16이 이기는 이유

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16은 정밀도가 더 높지만(가수 10비트 vs 7비트) 범위가 제한됩니다(최대 ~65,504). bfloat16은 정밀도는 낮지만 float32와 같은 범위입니다(최대 ~3.4e38).

신경망 학습에서:

- 활성화와 logits는 학습 스파이크 중에 65,504를 자주 넘습니다. float16은 오버플로하고, bfloat16은 처리합니다.
- float16에는 손실 스케일링이 필요하지만, bfloat16은 범위가 그래디언트 크기 스펙트럼을 덮으므로 보통 불필요합니다.
- bfloat16은 float32의 단순 절단입니다: 가수 하위 16비트를 버립니다. 변환은 간단하고 지수에서는 무손실입니다.

float16은 값이 유계이고 정밀도가 더 중요한 추론에 선호됩니다. bfloat16은 범위가 더 중요한 학습에 선호됩니다. 그래서 TPU와 현대 NVIDIA GPU(A100, H100)가 네이티브 bfloat16을 지원합니다.

### 그래디언트 클리핑

폭발 그래디언트는 여러 레이어를 지나며 그래디언트가 지수적으로 커질 때 발생합니다(RNN, 깊은 네트워크, 트랜스포머에서 흔함). 하나의 큰 그래디언트가 한 스텝에 모든 가중치를 망가뜨릴 수 있습니다.

클리핑 두 종류:

**값으로 클리핑:** 각 그래디언트 요소를 독립적으로 클램프합니다.

```
grad = clamp(grad, -max_val, max_val)
```

단순하지만 그래디언트 벡터의 방향을 바꿀 수 있습니다.

**노름으로 클리핑:** 전체 그래디언트 벡터를 스케일해 노름이 임계값을 넘지 않게 합니다.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

그래디언트 방향을 보존합니다. `torch.nn.utils.clip_grad_norm_()`가 하는 일이며, 표준 선택입니다.

전형적 값: 트랜스포머는 `max_norm=1.0`, RL은 `max_norm=0.5`, 단순한 네트워크는 `max_norm=5.0`.

그래디언트 클리핑은 임시방편이 아닙니다. 안전 장치입니다. 없으면 이상치 배치 하나가 수주간의 학습을 망칠 만큼 큰 그래디언트를 만들 수 있습니다.

### 수치 안정화 장치로서의 정규화 레이어

배치 정규화, 레이어 정규화, RMS 정규화는 보통 수렴을 돕는 정규화기로 소개됩니다. 동시에 수치 안정화 장치이기도 합니다.

정규화 없이 활성화는 레이어를 지나며 지수적으로 커지거나 작아질 수 있습니다:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

정규화는 매 레이어에서 활성화를 다시 중심화하고 스케일합니다:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`(보통 1e-5)은 모든 활성화가 동일할 때 0으로 나누기를 막습니다. 학습된 파라미터 `gamma`와 `beta`는 네트워크가 필요한 스케일을 복원하게 합니다.

이로써 값이 네트워크 전체에서 수치적으로 안전한 범위에 머물며, forward의 오버플로와 backward의 그래디언트 폭주를 모두 막습니다.

### 흔한 ML 수치 버그

**버그: 몇 에폭 후 손실이 NaN.**
원인: logits가 너무 커져 softmax가 오버플로. 또는 학습률이 너무 높아 가중치가 발산.
수정: 안정적 softmax(max 빼기), 학습률 감소, 그래디언트 클리핑 추가.

**버그: 손실이 log(num_classes)에 고착.**
원인: 모델 출력이 거의 균일 확률. 종종 그래디언트 소실 또는 학습이 전혀 안 됨을 의미.
수정: 데이터 라벨 확인, 손실 함수 검증, dead ReLU 점검.

**버그: 검증 정확도가 기대보다 1–3% 낮음.**
원인: 적절한 손실 스케일링 없는 혼합 정밀도. 그래디언트 언더플로가 작은 업데이트를 조용히 0으로 만듦.
수정: 동적 손실 스케일링 활성화, 또는 bfloat16으로 전환.

**버그: 일부 레이어의 그래디언트 노름이 0.0.**
원인: dead ReLU 뉴런(입력이 전부 음수), 또는 float16 언더플로.
수정: LeakyReLU 또는 GELU 사용, 그래디언트 스케일링, 가중치 초기화 점검.

**버그: 한 GPU에서는 동작하지만 다른 GPU에서는 결과가 다름.**
원인: 비결정적 부동소수점 누적 순서. GPU 병렬 환원이 하드웨어마다 다른 순서로 합산하고, 부동소수점 덧셈은 결합법칙이 성립하지 않음.
수정: 작은 차이(1e-6)를 받아들이거나, `torch.use_deterministic_algorithms(True)`를 설정하고 속도 저하를 감수.

**버그: 손실 계산에서 `exp()`가 `inf`를 반환.**
원인: max 빼기 트릭 없이 raw logits를 `exp()`에 전달.
수정: 내부적으로 log-sum-exp를 구현한 `torch.nn.functional.log_softmax()` 사용.

**버그: float32에서 float16으로 바꾼 뒤 학습이 발산.**
원인: float16은 6e-8 미만 그래디언트나 65,504 초과 활성화를 표현하지 못함.
수정: 손실 스케일링이 있는 혼합 정밀도(AMP), 또는 bfloat16 사용.

```figure
logsumexp-stability
```

## 구현하기 (Build It)

### Step 1: 부동소수점 정밀도 한계 시연

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: 나이브 vs 안정적 softmax 구현

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: 안정적 log-sum-exp 구현

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: 안정적 교차 엔트로피 구현

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: 그래디언트 검사

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 실용 활용 (Use It)

### 혼합 정밀도 시뮬레이션

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 그래디언트 클리핑

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 탐지

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

모든 엣지 케이스를 시연하는 완전한 구현은 `code/numerical.py`를 보세요.

## 배포할 산출물 (Ship It)

이 레슨이 만드는 것:
- 안정적 softmax, log-sum-exp, 교차 엔트로피, 그래디언트 검사, 혼합 정밀도 시뮬레이션이 있는 `code/numerical.py`
- 학습에서 NaN/Inf와 수치 이슈를 진단하는 `outputs/prompt-numerical-debugger.md`

이 안정적 구현들은 Phase 3에서 학습 루프를 만들 때, Phase 4에서 어텐션 메커니즘을 구현할 때 다시 등장합니다.

## 연습 문제 (Exercises)

1. **파국적 소거.** float32에서 나이브 공식 `E[x^2] - E[x]^2`로 [1000000.0, 1000001.0, 1000002.0]의 분산을 계산하세요. 그다음 Welford 온라인 알고리즘으로 계산하세요. 참 분산(0.6667)에 대한 오차를 비교하세요.

2. **정밀도 사냥.** Python에서 `1.0 + x == 1.0`이 되는 가장 작은 양수 float32 값 `x`를 찾으세요. 이것이 기계 엡실론(machine epsilon)입니다. `numpy.finfo(numpy.float32).eps`와 일치하는지 확인하세요.

3. **Log-sum-exp 엣지 케이스.** `logsumexp_stable`을 다음으로 테스트하세요: (a) 모든 값이 같음, (b) 하나가 나머지보다 훨씬 큼, (c) 모두 매우 음수(-1000). 나이브 버전이 실패하는 곳에서 올바른 결과가 나오는지 검증하세요.

4. **신경망 레이어 그래디언트 검사.** 단일 선형 레이어 `y = Wx + b`와 해석적 backward를 구현하세요. 3x2 가중치 행렬에 대해 `numerical_gradient`로 정확성을 검증하세요.

5. **손실 스케일링 실험.** float16 학습을 시뮬레이션하세요: [1e-9, 1e-3] 범위의 랜덤 그래디언트를 만들고 float16으로 변환한 뒤 0이 되는 비율을 측정하세요. 그다음 손실 스케일링(1024배)을 적용하고, float16으로 변환한 뒤 다시 스케일을 되돌린 다음 0 비율을 다시 측정하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "플로트 표준" | 이진 부동소수점 형식, 반올림 규칙, 특수 값(inf, nan)을 정의하는 국제 표준. 모든 현대 CPU와 GPU가 구현합니다. |
| Machine epsilon | "정밀도 한계" | 주어진 float 형식에서 1.0 + e != 1.0이 되는 가장 작은 값 e. float32에서는 약 1.19e-7. |
| Catastrophic cancellation | "뺄셈으로 인한 정밀도 손실" | 거의 같은 부동소수점을 빼면 유효 자릿수가 상쇄되고 반올림 잡음이 결과를 지배합니다. |
| Overflow | "너무 큰 수" | 결과가 표현 가능한 최댓값을 넘어 inf가 됩니다. exp(89)는 float32에서 오버플로합니다. |
| Underflow | "너무 작은 수" | 결과가 표현 가능한 최소 양수보다 0에 가까워 0.0이 됩니다. exp(-104)는 float32에서 언더플로합니다. |
| Log-sum-exp trick | "먼저 max를 빼라" | exp(max(x))를 묶어내어 log(sum(exp(x)))를 계산해 오버플로·언더플로를 막습니다. softmax, 교차 엔트로피, 로그 확률 수학에 사용됩니다. |
| Stable softmax | "폭발하지 않는 Softmax" | 지수화하기 전에 max(logits)를 뺍니다. 수치적으로 동일한 결과, 오버플로 불가능. |
| Gradient checking | "역전파 검증" | 역전파의 해석적 그래디언트를 유한차분의 수치 그래디언트와 비교해 구현 버그를 잡습니다. |
| Mixed precision | "Float16 forward, float32 backward" | 속도가 중요한 연산에는 낮은 정밀도, 수치에 민감한 연산에는 높은 정밀도 float를 사용합니다. 전형적 속도 향상은 2–3배입니다. |
| Loss scaling | "그래디언트 언더플로 방지" | 역전파 전에 손실에 큰 상수를 곱해 그래디언트가 float16 표현 범위에 머물게 한 뒤, 가중치 업데이트 전에 같은 상수로 나눕니다. |
| bfloat16 | "Brain floating point" | Google의 16비트 형식. 지수 8비트(float32와 같은 범위), 가수 7비트(float16보다 낮은 정밀도). 학습에 선호됩니다. |
| Gradient clipping | "그래디언트 노름 상한" | 그래디언트 벡터를 스케일해 노름이 임계값을 넘지 않게 합니다. 폭발 그래디언트가 가중치를 망치는 것을 막습니다. |
| NaN | "Not a Number" | 정의되지 않은 연산(0/0, inf-inf, sqrt(-1))에서 나오는 특수 float 값. 이후 모든 산술에 전파됩니다. |
| Inf | "Infinity" | 오버플로 또는 0으로 나누기에서 나오는 특수 float 값. 결합하면 NaN이 될 수 있습니다(inf - inf, inf * 0). |
| Numerical gradient | "무식한 미분" | f(x+h)와 f(x-h)를 평가해 2h로 나누어 도함수를 근사합니다. 느리지만 검증에는 신뢰할 수 있습니다. |

## 더 읽을거리 (Further Reading)

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 결정적 참고 문헌. 밀도 높지만 완전함
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 학습용 손실 스케일링을 소개한 NVIDIA 논문
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 혼합 정밀도 실전 가이드
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google이 TPU에 이 형식을 선택한 이유
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 부동소수점 합의 반올림 오차를 줄이는 알고리즘
