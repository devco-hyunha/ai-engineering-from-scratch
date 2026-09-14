---
name: skill-sampling-strategy
description: 생성, 추정, 추론에 맞는 샘플링 방법을 선택한다
version: 1.0.0
phase: 1
lesson: 16
tags: [sampling, mcmc, generation]
---

# 샘플링 전략 선택 (Sampling Strategy Selection)

텍스트 생성, 베이즈 추론, 몬테카를로 추정, 학습에 맞는 샘플링 방법을 고르는 방법입니다.

## 결정 체크리스트 (Decision Checklist)

1. 출력(텍스트, 이미지)을 생성하는가, 양(적분, 기댓값)을 추정하는가?
2. 목표 분포에서 직접 샘플링할 수 있는가, 아니면 밀도만 평가할 수 있는가?
3. 목표 분포가 이산인가 연속인가?
4. 표본 공간의 차원은? 저(< 5), 중(5-100), 고(> 100)?
5. 정확한 샘플이 필요한가, 근사로 충분한가?
6. 샘플링 연산을 통한 경사가 필요한가?

## 각 방법을 언제 쓸까 (When to use each method)

| Method | When to use | Complexity | Exact? |
|---|---|---|---|
| Direct sampling | You have the CDF or can use a library function | O(1) per sample | Yes |
| Inverse CDF | Known closed-form CDF inverse (exponential, Cauchy) | O(1) per sample | Yes |
| Box-Muller | Need normal samples without a library | O(1) per sample | Yes |
| Rejection sampling | Can evaluate target PDF, low dimension (1-3) | O(1/acceptance) per sample | Yes |
| Importance sampling | Need expectations, not individual samples | O(n) for n samples | Approximate |
| Stratified sampling | Monte Carlo estimation, want lower variance | O(n) for n samples | Approximate |
| Metropolis-Hastings | High-dimensional, can evaluate unnormalized density | O(1) per step + burn-in | Asymptotically |
| Gibbs sampling | Can sample from each conditional distribution | O(d) per full sweep | Asymptotically |
| HMC/NUTS | High-dimensional continuous, smooth density | O(L * d) per step | Asymptotically |
| Temperature sampling | LLM text generation, control creativity | O(V) for vocab size V | N/A |
| Top-k sampling | LLM generation, remove unlikely tokens | O(V log k) | N/A |
| Top-p (nucleus) | LLM generation, adaptive candidate set | O(V log V) | N/A |
| Reparameterization | Need gradients through Gaussian sampling (VAEs) | O(d) | Yes |
| Gumbel-Softmax | Need gradients through categorical sampling | O(k) for k classes | Approximate |

## LLM 생성 설정 (LLM generation settings)

| Use case | Temperature | Top-p | Top-k | Notes |
|---|---|---|---|---|
| Factual Q&A | 0.0 (greedy) | -- | -- | Deterministic, no randomness |
| Code generation | 0.2-0.5 | 0.9 | -- | Low creativity, high coherence |
| General chat | 0.7 | 0.9 | -- | Balanced |
| Creative writing | 0.9-1.2 | 0.95 | -- | Higher diversity |
| Brainstorming | 1.0-1.5 | 0.95 | -- | Maximum diversity, may lose coherence |

Temperature와 top-p는 결합할 수 있습니다. 먼저 temperature를 적용(로짓 스케일)한 뒤 top-p 필터링을 적용합니다.

## MCMC 방법 선택 (MCMC method selection)

| Property | Metropolis-Hastings | Gibbs | HMC/NUTS |
|---|---|---|---|
| Dimension | Any | Any (best < 100) | High (100+) |
| Requires conditionals | No | Yes | No |
| Requires gradient | No | No | Yes |
| Acceptance rate | Tune to ~23% | Always 100% | Tune to ~65% |
| Correlation | High (random walk) | Moderate | Low |
| Burn-in | Long | Moderate | Short |
| Best for | Exploration, simple models | Conjugate models, Bayesian networks | Continuous posteriors, deep probabilistic models |

## 흔한 실수 (Common mistakes)

- 고차원에서 기각 샘플링을 쓴다. 수락률이 차원에 따라 지수적으로 떨어진다. 5차원 이상이면 MCMC로 전환한다.
- MCMC proposal 분산을 너무 크거나 작게 설정한다. 너무 크면: 대부분 기각, 연쇄 정체. 너무 작으면: 전부 수락, 연쇄가 느리게 움직임. 무작위 보행 MH는 수락률 ~23%를 목표로 한다.
- Burn-in을 잊는다. MCMC의 처음 N개 샘플은 시작점에 편향된다. 최소 1000단계(복잡한 분포면 더)를 버린다.
- 목표와 매우 다른 proposal로 중요도 샘플링을 쓴다. 소수 샘플이 거대한 가중치를 받아 추정이 불안정해진다. 유효 표본 크기를 모니터링한다: ESS = (sum w_i)^2 / sum(w_i^2).
- 결정론적 출력이 필요한 작업(예: 분류, 구조화 추출)에 temperature > 0을 쓴다. 대신 탐욕(T=0) 또는 beam search를 쓴다.
- Temperature만 쓰고 top-p와 결합하지 않는다. Temperature만으로는 긴 꼬리의 쓰레기 토큰을 제거하지 못한다. Top-p가 한다.
- 표준 샘플링 연산을 통해 역전파한다. 연속(가우시안)에는 재매개변수화 트릭, 이산(범주형)에는 Gumbel-Softmax를 쓴다.

## 빠른 참고: 분산 감소 기법 (Quick reference: variance reduction techniques)

| Technique | How it works | Variance reduction |
|---|---|---|
| Stratified sampling | Divide space into strata, sample each | Always <= standard MC |
| Antithetic variates | Use both U and 1-U | Works for monotone functions |
| Control variates | Subtract a known-mean variable | Proportional to correlation |
| Importance sampling | Reweight samples from a better proposal | Depends on proposal quality |
| Latin hypercube | Stratify each dimension independently | Better than stratified in high-d |
