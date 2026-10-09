# 수치적 안정성

> 부동 소수점은 누수가 있는 추상화입니다. 학습 중에 문제가 발생하며, 예상치 못한 곳에서 나타납니다.

**유형:** Build
**언어:** Python
**선수 요건:** 1단계, 01-04강
**시간:** 약 120분

## 학습 목표

- 최댓값 차감 트릭(max-subtraction trick)을 사용하여 수치적으로 안정적인 softmax와 log-sum-exp를 구현해 보세요
- 부동 소수점 연산에서 오버플로, 언더플로, 치명적 상쇄(catastrophic cancellation)를 식별해 보세요
- 중앙 유한 차분(centered finite differences)을 사용하여 해석적 기울기를 수치적 기울기와 비교 검증해 보세요
- 학습 시 float16보다 bfloat16이 선호되는 이유와 손실 스케일링(loss scaling)이 기울기 언더플로를 방지하는 방식을 설명해 보세요

## 문제점

모델이 3시간 동안 학습한 후 손실이 NaN이 됩니다. print 문을 추가해 보세요. 9,000 스텝에서는 로짓(logits)이 정상입니다. 9,001 스텝에서는 `inf`가 됩니다. 9,002 스텝이 되면 모든 기울기가 `nan`가 되고 학습이 중단됩니다.

또는: 모델이 완결까지 학습되었지만 정확도가 논문이 주장한 것보다 2% 낮습니다. 모든 것을 점검해 보세요. 아키텍처는 일치합니다. 하이퍼파라미터도 일치합니다. 데이터도 일치합니다. 문제는 논문이 float32를 사용했고, 당신은 올바른 스케일링 없이 float16을 사용했다는 점입니다. 32비트의 누적된 반올림 오차가 조용히 정확도를 갉아먹었습니다.

또는: 교차 엔트로피 손실(cross-entropy loss)을 처음부터 구현했습니다. 작은 로짓에서는 잘 작동합니다. 로짓이 100을 초과하면 `inf`를 반환합니다. `exp(100)`가 float32가 표현할 수 있는 값보다 크기 때문에 softmax가 오버플로되었습니다. 모든 ML 프레임워크는 2줄 트릭으로 이 문제를 처리합니다. 당신은 이 트릭이 존재한다는 것을 몰랐습니다.

수치적 안정성은 이론적 우려가 아닙니다. 성공하는 학습 실행과 조용히 실패하는 학습 실행의 차이입니다. 디버깅하는 모든 심각한 ML 버그는 결국 부동 소수점 문제로 귀결됩니다.

## 개념

### IEEE 754: 컴퓨터가 실수를 저장하는 방식

컴퓨터는 IEEE 754 표준에 따라 부동 소수점 값으로 실수를 저장합니다. float는 세 부분으로 구성됩니다: 부호 비트, 지수, 그리고 가수(mantissa, significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

가수(mantissa)는 정밀도(유효 숫자의 개수)를 결정합니다. 지수(exponent)는 범위(숫자가 얼마나 크거나 작을 수 있는지)를 결정합니다.

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32는 약 7자리 십진수의 정밀도를 제공합니다. 즉, 1.01강 1.0000002는 구별할 수 있지만, 1.01강 1.00000002는 구별할 수 없습니다. 7자리 이후는 모두 반올림 잡음입니다.

float16은 약 3자리 정밀도를 제공합니다. 표현할 수 있는 가장 큰 숫자는 65,504입니다. ML에서는 로짓(logits), 기울기(gradients), 활성화 값(activations)이 이 값을 일상적으로 초과하므로, 이는 우려할 정도로 작은 값입니다.

bfloat16은 float16의 범위 문제를 해결하기 위한 Google의 답변입니다. float32와 동일한 8비트 지수를 가지므로 범위(최대 3.4e38)는 동일하지만, 가수(mantissa) 비트는 7비트뿐이어서 float16보다 정밀도가 낮습니다. 신경망 학습에서는 정밀도보다 범위가 더 중요하므로, bfloat16이 일반적으로 더 유리합니다.

### 왜 0.1 + 0.2 != 0.3일까요?

숫자 0.1은 이진 부동 소수점(binary floating point)으로 정확히 표현할 수 없습니다. 2진법에서는 순환 소수(repeating fraction)가 됩니다:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32는 이를 23비트 가수로 잘라냅니다. 저장된 값은 약 0.100000001490116입니다. 마찬가지로, 0.2는 약 0.200000002980232로 저장됩니다. 두 값의 합은 0.3이 아니라 0.300000004470348입니다.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

ML에서는 이것이 중요합니다. 왜냐하면:

1. `if loss < threshold`와 같은 손실 비교가 잘못된 결과를 줄 수 있습니다
2. 많은 작은 값(수천 단계에 걸친 기울기 업데이트)을 누적하면 실제 합계에서 벗어납니다
3. 부동 소수점을 `==`로 비교하면 체크섬(checksums) 및 재현성 테스트가 실패합니다

해결 방법: 부동 소수점을 `==`로 비교하지 마세요. `abs(a - b) < epsilon` 또는 `math.isclose()`를 사용하세요.

### 치명적 소거(Catastrophic Cancellation)

거의 동일한 두 부동 소수점 숫자를 빼면 유효 숫자가 소거되고, 반올림 잡음이 가장 앞자리 숫자로 승격됩니다.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

단 한 번의 뺄셈으로 상대 오차가 19%가 됩니다. ML에서는 다음 상황에서 이런 일이 발생합니다:

- 평균(mean)이 큰 데이터의 분산(variance)을 계산할 때: E[x]가 크면 `E[x^2] - E[x]^2`
- 거의 동일한 로그 확률(log-probabilities)을 뺄 때
- 너무 작은 epsilon으로 유한 차분(finite-difference) 기울기를 계산할 때

해결 방법: 크기가 크고 거의 같은 수를 빼는 연산을 피하도록 공식을 재배열하세요. 분산 계산에는 Welford 알고리즘을 사용하거나 데이터를 먼저 중심화하세요. 로그 확률은 전체적으로 로그 공간에서 연산하세요.

### 오버플로와 언더플로

오버플로(Overflow)는 결과가 표현 가능한 범위를 초과할 때 발생하고, 언더플로(Underflow)는 표현 가능한 최소 양수보다 더 작아질 때(0에 가까워질 때) 발생합니다.

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 함수는 ML에서 오버플로의 주요 원인입니다:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 함수는 반대 방향(언더플로)으로 작용합니다:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

ML에서 `exp()`는 softmax, sigmoid 및 확률 계산에 등장합니다. `log()`는 교차 엔트로피(Cross-Entropy), 로그 우도(log-likelihood), KL 발산(KL divergence)에 등장합니다. `log(exp(x))`의 조합은 적절한 트릭 없이는 지뢰밭과 같습니다.

### 로그 합 지수(Log-Sum-Exp) 트릭

`log(sum(exp(x_i)))`를 직접 계산하는 것은 수치적으로 위험합니다. `x_i` 중 하나가 크면 `exp(x_i)`가 오버플로됩니다. 모든 `x_i`가 매우 음수이면 모든 `exp(x_i)`가 0으로 언더플로되어 `log(0)`가 `-inf`이 됩니다.

트릭: 지수화(exponentiating)하기 전에 최대값을 빼세요.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

이 트릭이 작동하는 이유: `max(x)`를 뺀 후 가장 큰 지수는 `exp(0) = 1`입니다. 오버플로는 불가능합니다. 합산의 최소한 한 항은 1이므로 합은 최소 1이며 `log(1) = 0`입니다. `-inf`로의 언더플로는 불가능합니다.

증명:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`를 설정하면 오버플로가 제거됩니다.

이 트릭은 ML의 모든 곳에서 나타납니다:
- Softmax 정규화
- 교차 엔트로피 손실 계산
- 시퀀스 모델에서의 로그 확률 합산
- 가우시안 혼합(Mixture of Gaussians)
- 변분 추론(Variational inference)

### Softmax가 최대값 빼기 트릭을 필요로 하는 이유

Softmax는 로짓(Logits)을 확률로 변환합니다:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

트릭이 없으면 [100, 101, 102]의 로짓은 오버플로를 일으킵니다:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

트릭을 적용하여 max(x) = 102를 빼면:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

확률은 동일합니다. 연산은 안전합니다. 이는 최적화가 아니라 정확성을 위한 필수 조건입니다.

### NaN과 Inf: 탐지 및 방지

`nan` (숫자가 아님)과 `inf` (무한대)는 연산 과정에서 전염성처럼 퍼져나갑니다. 기울기 업데이트에서 `nan`가 하나 발생하면 가중치가 `nan`이 되고, 이후 모든 출력은 `nan`이 됩니다. 한 단계 만에 학습이 중단됩니다.

`inf`가 나타나는 방식:
- 큰 양수의 `exp()`
- 0으로 나누기: `1.0 / 0.0`
- 누적 연산에서의 `float32` 오버플로

`nan`가 나타나는 방식:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수의 `sqrt()`
- 음수의 `log()`
- 이미 존재하는 `nan`를 포함하는 모든 산술 연산

탐지:

```python
import math

math.isnan(x)       # x가 nan이면 True
math.isinf(x)       # x가 +inf 또는 -inf이면 True
math.isfinite(x)    # x가 nan도 inf도 아닌 경우 True
```

예방 전략:

1. 입력을 `exp()`로 클램프: `exp(clamp(x, -80, 80))`
2. 분모에 epsilon 추가: `x / (y + 1e-8)`
3. `log()` 내부에 epsilon 추가: `log(x + 1e-8)`
4. 안정적인 구현 사용 (log-sum-exp, 안정적 softmax)
5. 가중치 폭발을 방지하기 위한 기울기 클리핑
6. 디버깅 중 모든 순전파 후 `nan`/`inf` 확인

### 수치적 기울기 검사

분석적 기울기(역전파로부터)는 버그가 있을 수 있습니다. 수치적 기울기 검사는 유한 차분으로 기울기를 계산하여 이를 검증합니다.

중심 차분 공식:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

이 공식은 O(h^2) 정확도를 가지며, O(h) 정확도만 가지는 전방 차분 `(f(x+h) - f(x)) / h`보다 훨씬 더 정확합니다.

h 선택: 너무 크면 근사값이 부정확해집니다. 너무 작으면 치명적인 상쇄(catastrophic cancellation)로 인해 답이 파괴됩니다. 일반적으로 `h = 1e-5`부터 `1e-7`를 사용합니다.

검사: 분석적 기울기와 수치적 기울기 간의 상대적 차이를 계산합니다.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则:
- relative_error < 1e-7: 완벽함, 기울기가 정확합니다
- relative_error < 1e-5: 허용 가능, 아마도 정확합니다
- relative_error > 1e-3: 문제가 있습니다
- relative_error > 1: 기울기가 완전히 잘못되었습니다

새로운 레이어나 손실 함수를 구현할 때는 항상 기울기를 확인해 보세요. PyTorch는 이를 위해 `torch.autograd.gradcheck()`을 제공합니다.

### 혼합 정밀도 훈련

최신 GPU는 float32보다 2-8배 빠른 float16 행렬 곱셈을 계산하는 전용 하드웨어(Tensor Cores)를 갖추고 있습니다. 혼합 정밀도 훈련은 이를 활용합니다:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

순수 float16 훈련의 문제점: 기울기는 종종 매우 작습니다(1e-8 이하). Float16은 ~6e-8 이하의 값을 0으로 언더플로우 처리합니다. 모든 기울기 업데이트가 0이 되어 모델이 학습을 멈추게 됩니다.

해결 방법은 손실 스케일링입니다:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

동적 손실 스케일링은 스케일 팩터를 자동으로 조정합니다. 큰 값(65536)으로 시작하세요. 기울기가 `inf`으로 오버플로우되면 절반으로 줄이세요. N 단계 동안 오버플로우가 없으면 두 배로 늘리세요.

### bfloat16 vs float16: 훈련에 bfloat16이 더 좋은 이유

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16은 더 높은 정밀도(10비트 vs 7비트)를 가지지만 범위가 제한적입니다(최대 ~65,504). bfloat16은 정밀도가 낮지만 float32와 동일한 범위를 가집니다(최대 ~3.4e38).

신경망 훈련의 경우:

- 훈련 중 스파이크 시 활성화 값과 로짓은 65,504를 초과하는 경우가 많습니다. float16은 오버플로우되지만 bfloat16은 이를 처리합니다.
- float16은 손실 스케일링이 필요하지만, bfloat16은 범위가 기울기 크기 스펙트럼을 커버하므로 일반적으로 손실 스케일링이 필요하지 않습니다.
- bfloat16은 float32의 단순한 절단입니다: 가수의 하위 16비트를 버립니다. 변환은 지수 부분에서 손실 없이 간단합니다.

값이 제한되고 정밀도가 더 중요한 추론에서는 float16이 선호됩니다. 범위가 더 중요한 훈련에서는 bfloat16이 선호됩니다. 이것이 TPU와 최신 NVIDIA GPU(A100, H100)가 네이티브 bfloat16을 지원하는 이유입니다.

### 기울기 클리핑

기울기 폭발은 기울기가 많은 레이어를 통해 지수적으로 증가할 때 발생합니다(RNN, 깊은 네트워크, 트랜스포머에서 흔함). 하나의 큰 기울기가 한 단계에서 모든 가중치를 손상시킬 수 있습니다.

두 가지 클리핑 유형:

**값으로 클리핑:** 각 기울기 요소를 독립적으로 클램프합니다.

```
grad = clamp(grad, -max_val, max_val)
```

단순하지만 기울기 벡터의 방향을 변경할 수 있습니다.

**노름으로 클리핑:** 전체 기울기 벡터의 크기가 임계값을 초과하지 않도록 스케일링합니다.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

기울기의 방향을 보존합니다. `torch.nn.utils.clip_grad_norm_()`이 수행하는 작업입니다. 표준적인 선택입니다.

일반적인 값: 트랜스포머에는 `max_norm=1.0`, RL에는 `max_norm=0.5`, 더 단순한 네트워크에는 `max_norm=5.0`를 사용합니다.

기울기 클리핑은 해킹이 아닙니다. 안전 메커니즘입니다. 이것이 없으면 단일 이상치 배치로 인해 몇 주간의 학습을 망칠 만큼 큰 기울기가 생성될 수 있습니다.

### 수치적 안정화 장치로서의 정규화 레이어

배치 정규화, 레이어 정규화, RMS 정규화는 일반적으로 학습 수렴을 돕는 정규화 기법으로 소개됩니다. 이들은 또한 수치적 안정화 장치이기도 합니다.

정규화가 없으면 활성화 값이 레이어를 거치며 지수적으로 커지거나 작아질 수 있습니다:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

정규화는 모든 레이어에서 활성화 값을 재중심화하고 재스케일링합니다:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (일반적으로 1e-5)는 모든 활성화 값이 동일할 때 0으로 나누는 것을 방지합니다. 학습된 매개변수 `gamma`과 `beta`는 네트워크가 필요한 스케일을 복원할 수 있게 해줍니다.

이로써 네트워크 전체에서 값이 수치적으로 안전한 범위에 유지되어, 순방향 전파에서의 오버플로와 역방향 전파에서의 기울기 폭발을 모두 방지합니다.

### 공통적인 ML 수치적 버그

**버그: 몇 에포크 후 손실이 NaN이 됩니다.**
원인: 로짓이 너무 커져서 소프트맥스가 오버플로되었습니다. 또는 학습률이 너무 높아 가중치가 발산했습니다.
해결: 안정화된 소프트맥스(최댓값 차감)를 사용하거나, 학습률을 낮추거나, 기울기 클리핑을 추가하세요.

**버그: 손실이 log(num_classes)에 고착됩니다.**
원인: 모델 출력이 거의 균일한 확률입니다. 이는 보통 기울기가 소실되거나 모델이 전혀 학습되지 않는다는 의미입니다.
해결: 데이터 레이블이 올바른지 확인하고, 손실 함수를 검증하며, 죽은 ReLU가 있는지 확인하세요.

**버그: 검증 정확도가 예상보다 1-3% 낮습니다.**
원인: 적절한 손실 스케일링 없이 혼합 정밀도를 사용했습니다. 기울기 언더플로로 인해 작은 업데이트가 조용히 0으로 처리됩니다.
해결: 동적 손실 스케일링을 활성화하거나, bfloat16으로 전환하세요.

**버그: 일부 레이어의 기울기 노름이 0.0입니다.**
원인: 죽은 ReLU 뉴런(모든 입력이 음수) 또는 float16 언더플로입니다.
해결: LeakyReLU나 GELU를 사용하거나, 기울기 스케일링을 사용하며, 가중치 초기화를 확인하세요.

**버그: 모델은 한 GPU에서 작동하지만 다른 GPU에서는 다른 결과를 반환합니다.**
원인: 비결정적인 부동 소수점 누적 순서. GPU 병렬 감소는 하드웨어에 따라 다른 순서로 합산하며, 부동 소수점 덧셈은 결합 법칙이 성립하지 않습니다.
해결: 작은 차이(1e-6)를 허용하거나, `torch.use_deterministic_algorithms(True)`를 설정하고 속도 페널티를 감수하세요.

**버그: `exp()`가 손실 계산에서 `inf`를 반환합니다.**
원인: 최대값 빼기 트릭(max-subtraction trick) 없이 로짓(Logits)을 `exp()`에 직접 전달했습니다.
해결: 내부적으로 로그 합 지수(log-sum-exp)를 구현하는 `torch.nn.functional.log_softmax()`를 사용하세요.

**버그: float32에서 float16으로 전환한 후 학습이 발산합니다.**
원인: float16은 6e-8 미만의 기울기 크기나 65,504 이상의 활성값을 표현할 수 없습니다.
해결: 손실 스케일링(loss scaling)이 포함된 혼합 정밀도(Mixed Precision)(AMP)를 사용하거나, bfloat16을 대신 사용하세요.

```figure
logsumexp-stability
```

## 구현하기

### 1단계: 부동 소수점 정밀도 한계를 시연하세요

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 2단계: 단순한(naive) softmax와 안정적인 softmax를 구현하세요

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
# softmax_naive(dangerous_logits)는 [nan, nan, nan]을 반환합니다
```

### 3단계: 안정적인 로그 합 지수(log-sum-exp)를 구현하세요

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
# logsumexp_naive(large)는 inf를 반환합니다
```

### 4단계: 안정적인 교차 엔트로피(Cross-Entropy)를 구현하세요

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

### 5단계: 기울기 검사(Gradient checking)를 수행하세요

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

## 사용하기

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

### 기울기 클리핑(Gradient Clipping)

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

### NaN/Inf 감지

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

모든 엣지 케이스가 시연된 완전한 구현은 `code/numerical.py`를 참고하세요.

## 출시하기

이 강은 다음을 생성합니다:
- 안정적인 softmax, 로그 합 지수(log-sum-exp), 교차 엔트로피(Cross-Entropy), 기울기 검사(Gradient checking), 혼합 정밀도 시뮬레이션이 포함된 `code/numerical.py`
- 학습 중 NaN/Inf 및 수치적 문제를 진단하기 위한 `outputs/prompt-numerical-debugger.md`

이러한 안정적인 구현은 3단계에서 학습 루프를 구축할 때와 4단계에서 어텐션 메커니즘을 구현할 때 다시 등장합니다.

## 연습 문제

1. **치명적 소거(Catastrophic cancellation).** float32에서 단순 공식 `E[x^2] - E[x]^2`를 사용하여 [1000000.0, 1000001.0, 1000002.0]의 분산을 계산하세요. 그 후 Welford의 온라인 알고리즘을 사용하여 계산하세요. 참 분산(0.6667)과 비교하여 오차를 확인하세요.

2. **정밀도 탐색.** Python에서 `1.0 + x == 1.0`가 성립하는 가장 작은 양의 float32 값 `x`을 찾아보세요. 이것이 기계적 epsilon입니다. `numpy.finfo(numpy.float32).eps`과 일치하는지 확인하세요.

3. **Log-sum-exp 경계 사례.** `logsumexp_stable` 함수를 다음 조건으로 테스트해 보세요: (a) 모든 값이 동일한 경우, (b) 하나의 값이 나머지보다 훨씬 큰 경우, (c) 모든 값이 매우 음수인 경우(-1000). 단순한(naive) 버전이 실패하는 곳에서 올바른 결과를 반환하는지 확인하세요.

4. **신경망 레이어의 기울기 검증.** 단일 선형 레이어 `y = Wx + b`과 그 해석적 역전파를 구현하세요. `numerical_gradient`을 사용하여 3x2 가중치 행렬에 대해 정확성을 검증하세요.

5. **손실 스케일링 실험.** float16으로 훈련을 시뮬레이션하세요: [1e-9, 1e-3] 범위의 랜덤 기울기를 생성하고 float16으로 변환한 후, 0이 되는 비율을 측정하세요. 그 다음 손실 스케일링(1024 곱하기)을 적용하고, float16으로 변환한 후 다시 스케일링을 되돌려 0이 되는 비율을 다시 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| IEEE 754 | "float 표준" | 바이너리 부동 소수점 형식, 반올림 규칙, 특수 값(inf, nan)을 정의하는 국제 표준입니다. 모든 현대 CPU와 GPU가 이를 구현합니다. |
| 기계적 epsilon | "정밀도 한계" | 주어진 float 형식에서 1.0 + e != 1.0이 되는 가장 작은 값 e입니다. float32의 경우 약 1.19e-7입니다. |
| 파국적 소거(Catastrophic cancellation) | "뺄셈으로 인한 정밀도 손실" | 거의 동일한 부동 소수점 숫자를 뺄 때, 유효 숫자가 소거되고 반올림 잡음이 결과에 지배적으로 영향을 미칩니다. |
| 오버플로(Overflow) | "숫자가 너무 큼" | 결과가 표현 가능한 최대 값을 초과하여 inf가 됩니다. exp(89)는 float32에서 오버플로가 발생합니다. |
| 언더플로(Underflow) | "숫자가 너무 작음" | 결과가 표현 가능한 가장 작은 양수보다 0에 가까워져 0.0이 됩니다. exp(-104)는 float32에서 언더플로가 발생합니다. |
| Log-sum-exp 트릭 | "먼저 최대값을 빼기" | exp(max(x))를 인출하여 오버플로와 언더플로를 방지하는 방식으로 log(sum(exp(x)))를 계산하는 기법입니다. Softmax, 교차 엔트로피, 로그 확률 계산에 사용됩니다. |
| 안정적 Softmax | "폭발하지 않는 Softmax" | 지수화하기 전에 max(logits)를 빼는 방식입니다. 수치적으로 동일한 결과를 제공하며, 오버플로가 발생하지 않습니다. |
| 기울기 검사 | "역전파 검증" | 역전파로 계산한 분석적 기울기를 유한 차분으로 계산한 수치적 기울기와 비교하여 구현 버그를 잡습니다. |
| 혼합 정밀도 | "Float16 순방향, Float32 역방향" | 속도 중심 연산에는 낮은 정밀도 부동소수점을, 수치적으로 민감한 연산에는 높은 정밀도 부동소수점을 사용합니다. 일반적인 속도 향상은 2~3배입니다. |
| 손실 스케일링 | "기울기 언플로우 방지" | 역전파 전에 손실에 큰 상수를 곱하여 기울기가 float16의 표현 가능한 범위에 유지되도록 하고, 가중치 업데이트 전에 같은 상수로 나눕니다. |
| bfloat16 | "브레인 부동소수점" | Google의 16비트 형식으로, 8비트 지수(float32와 동일한 범위)와 7비트 가수(float16보다 낮은 정밀도)를 사용합니다. 학습에 선호됩니다. |
| 기울기 클리핑 | "기울기 노름 상한 설정" | 기울기 벡터의 노름이 임계값을 초과하지 않도록 스케일링합니다. 폭발하는 기울기가 가중치를 망치는 것을 방지합니다. |
| NaN | "숫자가 아님" | 정의되지 않은 연산(0/0, inf-inf, sqrt(-1))에서 발생하는 특수한 부동소수점 값입니다. 이후 모든 산술 연산에 전파됩니다. |
| Inf | "무한대" | 오버플로우나 0으로 나누는 연산에서 발생하는 특수한 부동소수점 값입니다. 결합되어 NaN을 생성할 수 있습니다(inf - inf, inf * 0). |
| 수치적 기울기 | "총알파 방식의 미분" | f(x+h)와 f(x-h)를 평가하고 2h로 나누어 미분값을 근사합니다. 느리지만 검증에는 신뢰할 수 있습니다. |

## 추가 읽기

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 결정적인 참고 자료, 밀도 높지만 완전합니다
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 학습을 위한 손실 스케일링을 도입한 NVIDIA 논문
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch에서 혼합 정밀도를 사용하는 실용 가이드
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google이 TPU에 이 형식을 선택한 이유
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 부동소수점 합산에서 반올림 오차를 줄이는 알고리즘
