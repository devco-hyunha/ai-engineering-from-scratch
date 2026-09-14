---
name: skill-convexity-checker
description: 최적화 문제가 볼록인지 판정하고 올바른 해법을 선택
version: 1.0.0
phase: 1
lesson: 18
tags: [optimization, convexity, solvers]
---

# 볼록성 검사기 (Convexity Checker)

최적화 문제가 볼록인지 확인하고, 그 답으로 무엇을 할지를 위한 가이드입니다.

## 결정 체크리스트

1. 목적 함수가 볼록인가? (헤세 양의 준정부호 또는 합성 규칙 확인)
2. 모든 부등식 제약이 g_i(x) <= 0 형태이고 각 g_i가 볼록인가?
3. 모든 등식 제약이 아핀(선형)인가?
4. 셋 모두 예이면 문제는 볼록입니다. 수렴 보장이 있는 볼록 해법을 쓰세요.
5. 하나라도 아니면 문제는 비볼록입니다. SGD/Adam을 쓰고 국소 최적해를 받아들이세요.

## 함수 볼록성 검사 방법

| Test | Applies to | Method |
|---|---|---|
| Second derivative >= 0 | Scalar functions f(x) | f''(x)를 계산. 모든 x에서 f''(x) >= 0이면 볼록. |
| Hessian is PSD | Multivariate functions f(x) | H(x)를 계산. 모든 곳에서 고유값 >= 0이면 볼록. |
| Definition test | Any function | 샘플 x, y, t에 대해 f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y) 확인. |
| Composition rules | Composed functions | 아래 합성 표 참고. |
| Restriction to a line | Multivariate f | 모든 x, v에 대해 g(t) = f(x + tv)가 t에 대해 볼록이면 f가 볼록. |

## 합성 규칙 (볼록성 보존)

| Operation | Result |
|---|---|
| f + g (both convex) | Convex |
| c * f (c > 0, f convex) | Convex |
| max(f, g) (both convex) | Convex |
| f(Ax + b) where f is convex | Convex |
| g(f(x)) where g is convex non-decreasing and f is convex | Convex |
| g(f(x)) where g is convex non-increasing and f is concave | Convex |
| sum of convex functions | Convex |
| pointwise supremum of convex functions | Convex |

## 흔한 ML 목적함수: 볼록인가?

| Objective | Convex? | Reason |
|---|---|---|
| MSE: (1/n) sum(y - Xw)^2 | Yes | w에 대해 이차, Hessian = (2/n) X^T X가 PSD |
| Logistic loss: sum(log(1 + exp(-y_i * w^T x_i))) | Yes | 볼록 함수의 합 (log-sum-exp 족) |
| Hinge loss: sum(max(0, 1 - y_i * w^T x_i)) | Yes | 볼록(선형) 함수의 최댓값 |
| L2 regularization: lambda * \|\|w\|\|^2 | Yes | 이차, Hessian = 2*lambda*I |
| L1 regularization: lambda * \|\|w\|\|_1 | Yes | 절댓값의 합 (볼록이지만 미분 불가능) |
| Ridge regression: MSE + L2 | Yes | 두 볼록 함수의 합 |
| LASSO: MSE + L1 | Yes | 두 볼록 함수의 합 |
| Elastic net: MSE + L1 + L2 | Yes | 볼록 함수의 합 |
| SVM (primal): hinge + L2 | Yes | 볼록 함수의 합 |
| Cross-entropy with softmax | Yes (in logits) | Log-sum-exp가 볼록 |
| Neural network (any loss) | No | 비선형 활성화가 비볼록 합성을 만듦 |
| k-means objective | No | 이산 할당 단계 |
| Matrix factorization: \|\|X - UV^T\|\|^2 | No | U와 V에 대해 쌍선형 |
| GAN loss | No | Minimax, 생성기에 대해 비볼록 |
| Contrastive loss (InfoNCE) | No | 음성 샘플이 있는 지수비의 로그 |

## 볼록성에 따른 해법 선택

| Problem type | Solver | Convergence guarantee |
|---|---|---|
| Convex, smooth, unconstrained | Gradient descent | O(1/k) to global minimum |
| Convex, smooth, unconstrained | L-BFGS | Superlinear to global minimum |
| Convex, smooth, unconstrained | Newton's method | Quadratic near minimum (if Hessian tractable) |
| Convex, smooth, constrained | Interior point method | Polynomial time |
| Convex, non-smooth (L1) | Proximal gradient / ISTA | O(1/k) to global minimum |
| Convex, non-smooth (L1) | ADMM | Flexible, handles constraints |
| Convex, quadratic | Conjugate gradient | Exact in n steps |
| Non-convex, smooth | SGD / Adam | Converges to local minimum |
| Non-convex, smooth | SGD + restarts | Better local minimum on average |
| Non-convex, smooth | Overparameterize + SGD | Flat minima, good generalization |

## 흔한 실수

- 손실 함수가 볼록이라고 문제가 볼록이라고 가정하기. 손실은 최적화하는 파라미터에 대해 볼록해야 합니다. 교차 엔트로피는 로짓에 대해 볼록하지만, 입력에서 로짓으로의 전체 신경망 매핑은 비볼록입니다.
- 비볼록 문제에 뉴턴법 쓰기. 헤세에 음의 고유값이 있을 수 있어, 뉴턴이 최솟값 대신 안장점이나 최댓값 쪽으로 움직일 수 있습니다.
- L1 정규화가 0에서 목적함수를 미분 불가능하게 만든다는 점을 잊기. 표준 경사하강법은 잘 동작하지 않습니다. 근위 경사하강법이나 부분기울기 방법을 쓰세요.
- A^T A를 형성해 조건수를 제곱하기. 최소제곱을 풀어야 하고 A가 불량조건이면 정규방정식 대신 QR이나 SVD를 쓰세요.
- 확인 없이 비볼록이라고 선언하기. 많은 ML 문제(선형 모델, SVM, 로지스틱 회귀)는 볼록이며 더 강한 해법의 이점을 받습니다.

## 빠른 검사: 내 문제는 볼록인가?

```
1. Write out the objective: minimize f(w) subject to constraints
2. For each term in f(w):
   - Is it quadratic with PSD matrix? -> Convex
   - Is it a norm? -> Convex
   - Is it log-sum-exp? -> Convex
   - Does it involve w nonlinearly (sigmoid(w), w1*w2)? -> Likely non-convex
3. Are all constraints linear or convex inequalities?
4. If ALL terms are convex and constraints are convex/linear -> problem is convex
5. If ANY term is non-convex -> problem is non-convex
```
