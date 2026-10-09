---
name: skill-gradient-computation
description: Compute gradients of common ML loss functions and choose the right derivative approach
version: 1.0.0
phase: 1
lesson: 4
tags: [calculus, gradients, backpropagation]
---

# ML을 위한 기울기 계산

신경망에서 사용되는 손실 함수, 활성화 함수, 레이어 연산의 기울기를 계산하기 위한 실용적 참고 자료입니다.

## 의사 결정 체크리스트

1. 함수가 단순한 기본 연산(멱, 지수, 로그, 삼각)으로 구성되어 있나요? 해석적 미분과 연쇄 법칙을 사용하세요.
2. 함수가 사용자 정의 또는 블랙박스 연산인가요? 수치 미분을 사용하세요: `(f(x+h) - f(x-h)) / (2h)`, h = 1e-7.
3. 함수가 PyTorch/JAX의 텐서 연산으로 만들어졌나요? 오토그라드(Autograd)가 처리하도록 하세요. 수치적 검증을 통해 확인하세요.
4. 가중치 행렬에 대한 스칼라 손실의 기울기가 필요하나요? 계산 그래프를 통해 연쇄 법칙을 적용하여, 한 노드씩 처리하세요.
5. 미분 불가능한 연산(argmax, 반올림, 샘플링)이 있나요? 스트레이트 스루 추정기(straight-through estimator)나 재파라미터화 트릭(reparameterization trick)을 사용하세요.

## 각 접근법을 사용할 때

| 접근법 | 사용 시점 | 비용 |
|---|---|---|
| 해석적 (수식 유도) | 단순한 함수, 오토그라드 출력 검증 | 런타임 비용 없음 |
| 수치적 (유한 차분) | 디버깅, 기울기 검사, 블랙박스 함수 | n개 매개변수에 대해 2n번의 순방향 전파 |
| 자동 미분 | 모든 미분 가능한 계산 그래프 (기본값) | 한 번의 역방향 전파 |
| 기호적 (SymPy, Mathematica) | 논문을 위한 폐형(closed-form) 기울기 유도 | 컴파일 시간만 소모 |

## 빠른 참고: 일반적인 미분

| 함수 | f(x) | f'(x) | ML 맥락 |
|---|---|---|---|
| MSE 손실 | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | 회귀 |
| 교차 엔트로피 (이진) | -(y log(p) + (1-y) log(1-p)) | p - y (시그모이드 이후) | 이진 분류 |
| 교차 엔트로피 (다중) | -log(p_true_class) | p - one_hot(y) (소프트맥스 이후) | 다중 클래스 분류 |
| 시그모이드 | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | 출력 게이트, 이진 출력 |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | 은닉층 활성화 함수 (레거시) |
| ReLU | max(0, x) | x > 0이면 1, x < 0이면 0 | 기본 은닉층 활성화 함수 |
| Leaky ReLU | max(0.01x, x) | x > 0이면 1, x < 0이면 0.01 | 죽은 뉴런(dead neuron) 방지 |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | 트랜스포머 |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | i=j일 때 s_i(1 - s_i), i!=j일 때 -s_i*s_j | 출력층 (자코비안) |
| Log-softmax | x_i - log(sum(e^(x_j))) | i번째 항목에 대해 1 - softmax(x_i) | 수치적으로 안정적인 CE |
| 선형 레이어 | y = Wx + b | dL/dW = dL/dy * x^T, dL/db = dL/dy | 모든 레이어 |
| L2 정규화 | lambda * sum(w^2) | 2 * lambda * w | 가중치 감쇠 |
| L1 정규화 | lambda * sum(\|w\|) | lambda * sign(w) | 희소성 |

## 공통 실수

- 배치 평균 손실(MSE, 교차 엔트로피)에서 1/n 계수를 잊는 경우입니다. 기울기는 배치 크기에 의해 스케일링됩니다.
- softmax 기울기를 벡터로 계산하는 경우입니다. 실제로는 자코비안 행렬입니다. 교차 엔트로피 + softmax가 결합된 경우, 기울기는 (p - y)로 단순화되어 전체 자코비안을 피할 수 있습니다.
- 연쇄 법칙(chain rule)을 잘못된 순서로 적용하는 경우입니다. 손실로부터 역순으로 작업하세요: dL/dW = dL/dy * dy/dW.
- 수치 미분에서 h가 너무 크거나(h = 0.1) 너무 작은(h = 1e-15) 경우입니다. float64의 경우 h = 1e-7을 사용하세요.
- ReLU가 정확히 x = 0에서 기울기가 정의되지 않는다는 것을 잊는 경우입니다. 실제로는 0 또는 0.5로 설정하세요.

## 기울기 검사 레시피

```
For each parameter w:
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = backward pass value
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5
```

상대 오차가 1e-3 이상이면 문제가 있습니다. 1e-5와 1e-3 사이인 경우 조사해 보세요.
