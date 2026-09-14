---
name: skill-gradient-computation
description: 흔한 ML 손실 함수의 기울기를 계산하고 적절한 미분 접근을 선택함
version: 1.0.0
phase: 1
lesson: 4
tags: [calculus, gradients, backpropagation]
---

# ML을 위한 기울기 계산 (Gradient Computation for ML)

신경망에 쓰이는 손실 함수, 활성화 함수, 레이어 연산의 기울기를 계산하기 위한 실무 참고서입니다.

## 결정 체크리스트

1. 함수가 간단한 원시 연산(거듭제곱, exp, log, 삼각)으로 구성되는가? 해석 도함수와 연쇄법칙을 사용하세요.
2. 함수가 커스텀이거나 블랙박스인가? 수치 미분을 사용하세요: `(f(x+h) - f(x-h)) / (2h)`, h = 1e-7.
3. 함수가 PyTorch/JAX의 텐서 연산으로 구성되는가? 자동미분에 맡기세요. 수치 체크로 검증하세요.
4. 스칼라 손실의 가중치 행렬에 대한 기울기가 필요한가? 계산 그래프를 따라 한 노드씩 연쇄법칙을 적용하세요.
5. 미분 불가능한 연산(argmax, 반올림, 샘플링)이 있는가? straight-through estimator 또는 재매개변수화 트릭을 사용하세요.

## 각 접근을 언제 쓸까

| 접근 | 언제 쓸까 | 비용 |
|---|---|---|
| 해석적 (손으로 유도) | 간단한 함수, 자동미분 출력 검증 | 런타임 비용 없음 |
| 수치적 (유한 차분) | 디버깅, 기울기 검사, 블랙박스 함수 | n개 파라미터에 대해 2n번 순전파 |
| 자동미분 | 임의의 미분 가능 계산 그래프 (기본값) | 한 번의 역전파 |
| 기호적 (SymPy, Mathematica) | 논문용 닫힌 형태 기울기 유도 | 컴파일 타임만 |

## 빠른 참고: 흔한 도함수

| 함수 | f(x) | f'(x) | ML 맥락 |
|---|---|---|---|
| MSE loss | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | 회귀 |
| Cross-entropy (binary) | -(y log(p) + (1-y) log(1-p)) | p - y (sigmoid 이후) | 이진 분류 |
| Cross-entropy (multi) | -log(p_true_class) | p - one_hot(y) (softmax 이후) | 다중 클래스 분류 |
| Sigmoid | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | 출력 게이트, 이진 출력 |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | 은닉 활성화 (레거시) |
| ReLU | max(0, x) | 1 if x > 0, 0 if x < 0 | 기본 은닉 활성화 |
| Leaky ReLU | max(0.01x, x) | 1 if x > 0, 0.01 if x < 0 | dead neuron 회피 |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | 트랜스포머 |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | s_i(1 - s_i) for i=j, -s_i*s_j for i!=j | 출력 레이어 (야코비) |
| Log-softmax | x_i - log(sum(e^(x_j))) | 1 - softmax(x_i) for the i-th entry | 수치적으로 안정한 CE |
| Linear layer | y = Wx + b | dL/dW = dL/dy * x^T, dL/db = dL/dy | 모든 레이어 |
| L2 regularization | lambda * sum(w^2) | 2 * lambda * w | Weight decay |
| L1 regularization | lambda * sum(\|w\|) | lambda * sign(w) | 희소성 |

## 흔한 실수

- 배치 평균 손실(MSE, cross-entropy)에서 1/n 인자를 잊음. 기울기가 배치 크기로 스케일됩니다.
- Softmax 기울기를 벡터로 계산하지만 실제로는 야코비 행렬입니다. Cross-entropy + softmax 결합에서는 기울기가 (p - y)로 단순화되어 전체 야코비를 피할 수 있습니다.
- 연쇄법칙 순서를 거꾸로 적용함. 손실에서 뒤로 작업하세요: dL/dW = dL/dy * dy/dW.
- 수치 도함수에 너무 큰 h(h = 0.1)나 너무 작은 h(h = 1e-15)를 씀. float64에서는 h = 1e-7을 유지하세요.
- ReLU가 정확히 x = 0에서 기울기가 정의되지 않음을 잊음. 실무에서는 0 또는 0.5로 둡니다.

## 기울기 검사 레시피

```
For each parameter w:
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = backward pass value
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5
```

상대 오차가 1e-3을 넘으면 무언가 잘못된 것입니다. 1e-5와 1e-3 사이면 조사하세요.
