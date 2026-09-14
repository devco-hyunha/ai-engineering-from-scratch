---
name: skill-information-theory
description: 정보 이론 개념을 ML 손실 함수, 모델 평가, 특성 선택에 적용한다
version: 1.0.0
phase: 1
lesson: 9
tags: [information-theory, entropy, loss-functions]
---

# ML을 위한 정보 이론 (Information Theory for ML)

머신러닝 시스템에서 엔트로피, 교차 엔트로피, KL 발산, 상호정보를 언제 쓸지.

## 결정 체크리스트 (Decision Checklist)

1. 단일 분포의 불확실성을 측정? **엔트로피**를 쓰세요.
2. 모델이 참 라벨을 얼마나 잘 근사하는지 측정? **교차 엔트로피**를 쓰세요(이것이 분류 손실입니다).
3. 두 분포 사이의 거리 측정? **KL 발산**을 쓰세요.
4. 두 변수가 관련있는지 확인? **상호정보**를 쓰세요.
5. 언어 모델 품질 보고? **Perplexity**(교차 엔트로피의 지수)를 쓰세요.
6. 한 모델을 다른 모델로 증류? Teacher에서 student로의 **KL 발산**을 최소화하세요.

## 각 측도를 언제 쓸지

| Measure | Formula | Use case | ML application |
|---|---|---|---|
| Entropy H(P) | -sum(p log p) | How uncertain is this distribution? | Data complexity, maximum entropy models |
| Cross-entropy H(P,Q) | -sum(p log q) | How good is model Q at predicting true P? | Classification loss, language model loss |
| KL divergence D(P\|\|Q) | sum(p log(p/q)) | How different are P and Q? | VAE loss (ELBO), knowledge distillation, RLHF |
| Mutual information I(X;Y) | H(X) - H(X\|Y) | How much does Y tell us about X? | Feature selection, representation learning |
| Perplexity | exp(H(P,Q)) or 2^H | How confused is the model? | Language model evaluation |
| Conditional entropy H(X\|Y) | -sum(p(x,y) log p(x\|y)) | Remaining uncertainty in X after knowing Y | Feature informativeness |

## 핵심 관계 (Key relationships)

```
Cross-entropy  = Entropy + KL divergence
H(P, Q)        = H(P)   + D_KL(P || Q)

Since H(P) is constant during training:
  Minimizing cross-entropy = Minimizing KL divergence

Mutual information = Entropy - Conditional entropy
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

Perplexity = exp(cross-entropy in nats)
           = 2^(cross-entropy in bits)
```

## 빠른 참고: 공식과 단위

| Formula | Bits (log base 2) | Nats (log base e) |
|---|---|---|
| Information: -log(p) | -log2(p) | -ln(p) |
| Entropy: -sum(p log p) | bits | nats |
| 1 nat = | 1.4427 bits | 1 nat |
| PyTorch default | -- | nats |
| Information theory papers | bits | -- |

## 값 해석하기

| 엔트로피 값 | 의미 |
|---|---|
| 0 | 결정적. 한 결과가 확률 1. |
| log(n) | 최대 불확실성. n개 결과에 대한 균등 분포. |
| Low | 분포가 뾰족함. 모델이 확신. |
| High | 분포가 평평함. 모델이 불확실. |

| Perplexity 값 | 언어 모델 품질 |
|---|---|
| 1 | 완벽한 예측(실무에서는 거의 없음) |
| 10 | 평균적으로 ~10개 동등 가능 토큰 중에서 선택 |
| 50 | 표준 벤치마크에서 GPT-2 수준 |
| < 10 | 잘 표현된 도메인의 최신 수준 |

## 흔한 실수

- KL 발산을 계산하고 대칭으로 취급하기. D_KL(P||Q) != D_KL(Q||P). 대칭 측도가 필요하면 Jensen-Shannon 발산을 쓰세요: JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q).
- One-hot 라벨에서 교차 엔트로피가 -log(p_true_class)로 단순화됨을 잊기. 참 분포가 one-hot이면 모든 클래스에 대해 합할 필요가 없습니다.
- 코드에서는 로그 밑 2를 쓰고 nats로 보고하기(또는 그 반대). PyTorch는 기본적으로 자연로그를 씁니다. nats를 bits로 바꾸려면 log2(e) = 1.4427을 곱하세요.
- 빈 사건 또는 확률 0인 사건의 엔트로피 계산. 관례: 0 * log(0) = 0, lim(p->0) p*log(p) = 0이므로.
- 서로 다른 어휘에 걸쳐 perplexity 비교. 어휘 크기 50k·perplexity 30인 모델은 어휘 크기 10k·perplexity 30인 모델과 직접 비교할 수 없습니다.

## 프로덕션 ML에서 각 개념이 나타나는 곳

| Concept | Where you see it |
|---|---|
| Cross-entropy loss | Every classification model (nn.CrossEntropyLoss) |
| KL divergence | VAE ELBO, PPO clipping, knowledge distillation |
| Entropy regularization | Exploration bonus in RL (higher entropy = more exploration) |
| Mutual information | Feature selection, InfoNCE loss (contrastive learning) |
| Perplexity | Language model benchmarks (lower = better) |
| Label smoothing | Replaces one-hot with soft targets, reduces cross-entropy overconfidence |
| Temperature scaling | Divides logits by T before softmax, controls entropy of output |
