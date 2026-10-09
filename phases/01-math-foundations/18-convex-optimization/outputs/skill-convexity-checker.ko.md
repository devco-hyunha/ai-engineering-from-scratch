---
name: skill-convexity-checker
description: 최적화 문제가 볼록한지 판단하고 올바른 솔버를 선택합니다
version: 1.0.0
phase: 1단계
lesson: 18강
tags: [optimization, convexity, solvers]
---

# 볼록성 검사기

최적화 문제가 볼록한지 확인하는 방법과 그 결과에 대한 대응 방법을 알아보세요.

## 의사 결정 체크리스트

1. 목적 함수가 볼록한가요? (헤시안(Hessian)의 양의 준반음정성(positive semi-definiteness)을 확인하거나 합성 규칙을 사용하세요.)
2. 모든 부등식 제약 조건이 g_i(x) <= 0 형태이며, 각 g_i가 볼록 함수인가요?
3. 모든 등식 제약 조건이 아핀(선형)인가요?
4. 세 조건 모두 '예'라면, 문제는 볼록합니다. 수렴 보장이 있는 볼록 솔버를 사용하세요.
5. 하나라도 '아니오'라면, 문제는 비볼록(non-convex)합니다. SGD/Adam을 사용하고 국소 최적해를 수용하세요.

## 함수의 볼록성을 테스트하는 방법

| 테스트 | 적용 대상 | 방법 |
|---|---|---|
| 이계 도함수 >= 0 | 스칼라 함수 f(x) | f''(x)를 계산하세요. 모든 x에 대해 f''(x) >= 0이면 볼록합니다. |
| 헤시안(Hessian)이 PSD | 다변량 함수 f(x) | H(x)를 계산하세요. 모든 고유값이 모든 곳에서 >= 0이면 볼록합니다. |
| 정의 테스트 | 모든 함수 | 샘플링된 x, y, t에 대해 f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)를 확인하세요. |
| 합성 규칙 | 합성된 함수 | 아래 합성 표를 참조하세요. |
| 선(line)으로의 제한 | 다변량 f | f는 모든 x, v에 대해 g(t) = f(x + tv)가 t에 대해 볼록일 때만 볼록입니다. |

## 합성 규칙 (볼록성 보존)

| 연산 | 결과 |
|---|---|
| f + g (둘 다 볼록) | 볼록 |
| c * f (c > 0, f 볼록) | 볼록 |
| max(f, g) (둘 다 볼록) | 볼록 |
| f(Ax + b) (f가 볼록) | 볼록 |
| g(f(x)) (g가 볼록이고 비감소, f가 볼록) | 볼록 |
| g(f(x)) (g가 볼록이고 비증가, f가 오목) | 볼록 |
| 볼록 함수들의 합 | 볼록 |
| 볼록 함수들의 점별 상한(pointwise supremum) | 볼록 |

## 일반적인 ML 목적 함수: 볼록한가요?

| 목적 함수 | 볼록 여부 | 이유 |
|---|---|---|
| MSE: (1/n) sum(y - Xw)^2 | 예 | w에 대해 2차, 헤시안 = (2/n) X^T X는 PSD |
| 로지스틱 손실: sum(log(1 + exp(-y_i * w^T x_i))) | 예 | 볼록 함수의 합 (log-sum-exp 계열) |
| 힌지 손실: sum(max(0, 1 - y_i * w^T x_i)) | 예 | 볼록 (선형) 함수의 최대값 |
| L2 정규화: lambda * \|\|w\|\|^2 | 예 | 2차, 헤시안 = 2*lambda*I |
| L1 정규화: lambda * \|\|w\|\|_1 | 예 | 절댓값의 합 (볼록하지만 미분 불가능) |
| 릿지 회귀: MSE + L2 | 예 | 두 볼록 함수의 합 |
| LASSO: MSE + L1 | 예 | 두 볼록 함수의 합 |
| Elastic net: MSE + L1 + L2 | 예 | 볼록 함수의 합 |
| SVM (원형 문제): 힌지 + L2 | 예 | 볼록 함수의 합 |
| Softmax를 사용한 교차 엔트로피 | 예 (로짓에서) | Log-sum-exp는 볼록 |
| 신경망 (임의의 손실) | 아니오 | 비선형 활성화가 비볼록 합성을 생성 |
| k-means 목적 함수 | 아니오 | 이산 할당 단계 |
| 행렬 분해: \|\|X - UV^T\|\|^2 | 아니오 | U와 V에 대해 쌍선형 |
| GAN 손실 | 아니오 | Minimax, 생성자에 대해 비볼록 |
| 대조 손실 (InfoNCE) | 아니오 | 음수 샘플을 포함한 지수 함수 비율의 로그 |

## 볼록성에 기반한 솔버 선택

| 문제 유형 | 솔버 | 수렴 보장 |
|---|---|---|
| 볼록, 매끄러운, 무제약 | 경사 하강법 | 전역 최소값까지 O(1/k) |
| 볼록, 매끄러운, 무제약 | L-BFGS | 전역 최소값까지 초선형 |
| 볼록, 매끄러운, 무제약 | 뉴턴 방법 | 최소값 근처에서 2차 수렴 (헤시안이 계산 가능한 경우) |
| 볼록, 매끄러운, 제약 있음 | 내점법 | 다항 시간 |
| 볼록, 비매끄러운 (L1) | 근접 경사 / ISTA | 전역 최소값까지 O(1/k) |
| 볼록, 비매끄러운 (L1) | ADMM | 유연하며, 제약 처리 가능 |
| 볼록, 2차 | 켤레 기울기 | n 단계에서 정확 |
| 비볼록, 매끄러운 | SGD / Adam | 지역 최소값으로 수렴 |
| 비볼록, 매끄러운 | SGD + 재시작 | 평균적으로 더 나은 국소 최소값 |
| 비볼록, 매끄러운 | 과매개변수화 + SGD | 평평한 최소값, 좋은 일반화 |

## 공통 실수

- 손실 함수가 볼록하다는 이유로 문제가 볼록하다고 가정하는 것. 손실은 최적화하는 매개변수에 대해 볼록해야 합니다. 교차 엔트로피(Cross-Entropy)는 로짓(Logits)에 대해 볼록하지만, 입력에서 로짓으로 가는 전체 신경망 매핑은 비볼록입니다.
- 비볼록 문제에 뉴턴 방법(Newton's method)을 사용하는 것. 헤시안(Hessian)이 음의 고유값(Eigenvalue)을 가질 수 있으며, 이는 뉴턴 방법이 최소값이 아닌 안장점(saddle points)이나 최대값으로 이동하게 만듭니다.
- L1 정규화(Normalization)가 목적 함수를 0에서 미분 불가능하게 만든다는 것을 잊는 것. 표준 경사 하강법(Gradient Descent)은 잘 작동하지 않습니다. 근사 경사 하강법(proximal gradient descent)이나 부분 기울기(subgradient) 방법을 사용하세요.
- A^T A를 형성하여 조건수(condition number)를 제곱하는 것. 최소 제곱 문제를 풀어야 하고 A가 조건이 나쁜(ill-conditioned) 경우, 정규 방정식(normal equations) 대신 QR 또는 SVD를 사용하세요.
- 확인하지 않고 문제가 비볼록하다고 선언하는 것. 많은 ML 문제(선형 모델, SVM, 로지스틱 회귀)는 볼록하며 더 강력한 솔버(solver)의 이점을 누립니다.

## 빠른 테스트: 내 문제는 볼록한가요?

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
