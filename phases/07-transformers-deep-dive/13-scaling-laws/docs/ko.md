# 스케일링 법칙 (Scaling Laws)

> 2020년 Kaplan의 논문은 모델이 커질수록 손실(loss)이 낮아진다고 설명했습니다. 2022년 Hoffmann의 논문은 모델이 충분히 학습되지 않았다(under-training)고 지적했습니다. 연산량(Compute)은 파라미터(parameters)와 토큰(tokens)이라는 두 가지 범주로 나뉘며, 그 배분 방식은 단순하지 않습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 문제점 (The Problem)

$C$만큼의 학습 연산량(FLOPs)을 보유한 상태에서 최적의 모델을 만들고자 할 때, 두 가지 조절 가능한 요소(knobs)에 직면하게 됩니다.

1. **파라미터 수($N$)는 얼마인가?** 모델이 커질수록 용량(capacity)이 높아집니다.
2. **학습 토큰 수($D$)는 얼마인가?** 데이터가 많을수록 용량을 더 잘 활용할 수 있습니다.

FLOPs는 대략 `6 × N × D`에 비례하여 확장됩니다. $N$을 높이고 $D$를 낮출 수도 있고, 반대로 $D$를 높이고 $N$을 낮출 수도 있습니다. 어떤 방식이 더 나을까요?

2022년 이전에는 "$N$을 최대한 높여라"가 정답이었습니다. GPT-3(2020)는 약 300B 개의 토큰으로 학습된 175B 개의 파라미터를 가졌습니다. 파라미터당 약 1.7개의 토큰 비율입니다. Kaplan 스케일링 법칙(Kaplan scaling laws)이 이를 뒷받침했습니다.

하지만 Chinchilla라고 불리는 소규모 모델군을 학습시킨 Hoffmann et al. (2022)은 다른 결과를 발견했습니다. 최적의 비율은 **파라미터당 20개의 토큰**에 더 가깝다는 것입니다. 즉, GPT-3는 학습이 10배 부족(under-trained)했습니다. Chinchilla(70B 파라미터, 1.4T 토큰)는 GPT-3(175B, 300B 토큰)보다 추론 비용(inference cost)은 2.5배 낮으면서도 모든 벤치마크에서 더 뛰어난 성능을 보였습니다.

2026년은 Chinchilla의 시대이지만, 한 가지 중요한 반전이 있습니다. Llama 3 8B는 15조 개의 토큰으로 학습되었으며, 이는 파라미터당 1,875개의 토큰 비율입니다. Chinchilla 최적값보다 94배나 높은 수치입니다. 대규모로 사용될 모델의 경우 학습 비용보다 추론 비용이 더 중요하기 때문에, 배포 가능한 작은 크기(footprint)를 확보하고자 (Chinchilla 기준을 넘어) 과잉 학습(over-training)하는 것이 2026년의 기본 전략입니다.

## 개념 (The Concept)

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### 호프만 법칙 (The Hoffmann law)

Chinchilla 논문에 따르면, 손실(loss)은 다음과 같은 식을 따릅니다:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 파라미터 수 (임베딩 제외).
- `D` = 학습 토큰 수.
- `α ≈ 0.34`, `β ≈ 0.28` (대략적으로 대칭적임).
- `E ≈ 1.69`, 줄일 수 없는 손실 상한선(irreducible loss ceiling).
- `A ≈ 406`, `B ≈ 411`.

규모를 확장함에 따라 두 항은 서로 트레이드오프(trade-off) 관계를 가집니다. 고정된 연산량($C = 6ND$) 조건에서 `N`에 대해 미분하여 풀면 다음과 같습니다:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

연산 최적화(Compute-optimal): 파라미터당 20개의 토큰.

### 왜 과잉 학습(Over-training)을 하는가

Chinchilla-optimal(친칠라 최적화)은 학습 FLOP당 학습 손실(training loss)을 최소화하는 방식입니다. 하지만 학습 비용은 한 번만 지불하면 되지만, 추론(inference) 비용은 영구적으로 발생합니다.

매달 1조 개의 토큰을 처리하는 챗봇의 경우, 전체 비용에서 추론 비용이 압도적인 비중을 차지합니다. Llama의 접근 방식은 '더 작게, 더 오래(train smaller, longer)' 학습하는 것입니다. 15T 토큰으로 학습된 8B 모델은 추론에 매우 최적화되어 있습니다:

- 소비자용 GPU에 탑재 가능합니다.
- 지연 시간(latency)이 Chinchilla-optimal 방식의 70B 모델보다 훨씬 짧습니다.
- 대부분의 작업에서 충분히 근접한 품질을 보여줍니다.

DeepMind의 2024년 논문("Over-training is the new optimal")은 이를 공식화했습니다. 추론 중심의 워크로드(workloads)의 경우, 서비스 규모에 따라 매개변수(parameter)당 적정 비율은 100~500 토큰에 더 가깝습니다.

### 창발성 vs 매끄러움 (Emergence vs smoothness)

**주장:** 특정 능력(산술, 다단계 추론, 사고의 사슬(chain-of-thought) 수행 등)은 특정 규모(scale)에서 갑자기 "창발(emerge)"한다.

Schaeffer et al. (2023)은 이것이 측정상의 인위적 결과물(measurement artifact)이라고 주장했습니다. 창발적 지표들은 불연속적인 점수 산정 방식(정확히 일치 여부, 임계값 기준 정확도 등)을 사용하기 때문에, 기저에 있는 로짓(logits)의 매끄러운 개선 과정을 가리게 됩니다. 반면, 연속적인 지표(cross-entropy)를 사용하면 매끄러운 곡선이 나타납니다.

2026년 현재의 합의된 견해는 다음과 같습니다: 연속적인 손실(continuous loss)을 통한 예측은 신뢰할 수 있습니다. 벤치마크 점수의 급격한 도약은 종종 채점 방식에 의한 인위적 결과물입니다. 계획 예산(plan budgets)을 세울 때는 연속적인 지표를 기준으로 삼으세요.

### 2026년의 전망 (The 2026 picture)

스케일링 법칙(scaling laws)은 여전히 유효하지만, 다음과 같은 변화가 있습니다:

| 요소 (Factor) | 변화 양상 (Changed how) |
|--------|-------------|
| 데이터 품질 (Data quality) | "좋은" 토큰을 선별(Phi 스타일)하는 것이 유효 연산량(effective compute)을 2배 이상 변화시킴 |
| MoE | 전체 파라미터 수가 활성 FLOPs(active FLOPs)와 분리됨; 스케일링 법칙은 활성 FLOPs를 기준으로 작동함 |
| 사후 학습 (Post-training) | 특정 능력(지시 이행, 코드)은 사전 학습(pretraining)보다 SFT+RLHF를 통해 더 크게 변화함 |
| 멀티모달리티 (Multimodality) | 이미지 + 텍스트 토큰이 함께 스케일링됨; 모달리티별로 별도의 곡선이 존재함 |
| 합성 데이터 (Synthetic data) | 모델이 학습 데이터를 생성함; 유효 연산량이 복리로 증가할 수 있음 |

Muon 옵티마이저(Kimi Moonlight, 2024)는 동일한 데이터 조건에서 AdamW 대비 약 2배의 유효 연산량(effective compute) 이득을 보여주었습니다. 일부 2026년 학습 실행에서는 Muon을 기본값으로 사용합니다. 이는 스케일링 법칙의 형태(shape)가 아닌, 법칙 내의 절대 상수(absolute constant)를 변화시킵니다.

```figure
scaling-laws
```

## 구현하기 (Build It)

`code/main.py`를 확인해 보세요. Chinchilla 손실 방정식을 구현하고, 여러 컴퓨팅 예산(compute budgets)에 대해 계산 최적화된 `(N, D)` 값을 도출합니다.

### 1단계: Chinchilla 손실(Chinchilla loss)

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

고정된 `C = 6ND` 값에 대해 `(N, D)` 평면 위에서 `L`을 등고선(contour)으로 시각화해 보세요. 그리고 최솟값을 찾아보세요.

### 2단계: compute-optimal frontier (연산 최적 프런티어)

`1e17`에서 `1e25` FLOPs 사이의 연산 예산(compute budgets)에 대하여, `6ND = C`라는 제약 조건 하에 손실(loss)을 최소화하는 `(N, D)`를 찾습니다. 이때 `D/N ≈ 20`인 비율을 확인해 보세요.

### 3단계: 과잉 학습 비용 (over-training cost)

최적의 `N`보다 10배 작은 모델(최적 `N`의 1/10, 최적 `D`의 10배)을 학습시키기 위해 지불해야 하는 추가 손실(extra loss)을 계산합니다. 그 대가로 얻는 추론 FLOP 절감량(N에 비례)을 보고합니다.

### 4단계: 실제 모델과 비교하기 (Compare to real models)

GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (활성 파라미터)의 알려진 `(N, D)` 쌍을 입력하여, 예측된 손실(predicted loss)과 보고된 손실(reported loss)을 비교해 보세요.

## 활용하기 (Use It)

직접 프런티어 모델(frontier model)을 훈련할 가능성은 낮습니다. 하지만 스케일링 법칙(scaling laws)은 다음을 알려줍니다:

1. **파인튜닝(fine-tune)에 데이터가 충분한지 여부.** 만약 작업 특화 데이터가 베이스 모델 파라미터당 20 토큰 미만이라면, 특정 손실 하한선(loss floor)에서 포화 상태에 도달할 것을 예상해야 합니다.
2. **더 큰 베이스 모델을 선택할지 여부.** 예산의 전부를 추론(inference)에 사용하고 있다면, 크기는 더 작지만 더 오래 학습된 모델을 선호하세요.
3. **수익 체감 지점(where the returns diminish).** Chinchilla 최적값(Chinchilla-optimal)의 1000배를 넘어서면, 로그 손실(log-loss)의 변화는 노이즈 수준이 됩니다.

**2026년의 연구 궤적(Research trajectory):**

- **데이터 제약 체계(Data-constrained regime).** 웹에는 유한한 수의 고품질 토큰이 존재합니다(필터링 후 영어 기준 약 5~10조 개). 프런티어 모델의 사전 학습(pretraining)은 이 한계치에 도달하고 있습니다. 합성 데이터(synthetic data), 다국어, 멀티모달, 그리고 RLHF 규모의 파인튜닝이 다음 동력이 될 것입니다.
- **연산 승수 트릭(Compute-multiplier tricks).** Muon 옵티마이저, MoE, 더 나은 데이터 큐레이션 — 이들은 각각 점근선(asymptote)이 아닌 절대적인 상수 값을 변화시킵니다.
- **RL을 위한 스케일링 법칙(Scaling laws for RL).** 미해결 과제입니다. 초기 증거들은 RL 샘플에서도 멱법칙(power-law)이 나타나지만, 사전 학습과는 매우 다른 지수(exponent)를 가진다는 점을 시사합니다.

## Ship It (실행해 보기)

`outputs/skill-training-budget-estimator.md`를 참조하세요. 이 기술(skill)은 계산 예산(compute budget), 배포 제약 조건(deployment constraints), 그리고 목표 손실(target loss)이 주어졌을 때 새로운 학습 실행을 위한 `(N, D, hours, GPU)`를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 연산 예산(compute budgets)이 `1e20`, `1e22`, `1e24`일 때 Chinchilla 최적 `(N, D)` 값을 출력하고, 실제 모델 표와 비교해 보세요.
2. **중간 (Medium).** 연산량에 따른 Hoffmann 손실 함수 곡선(Hoffmann loss-as-function-of-compute curve)을 구현해 보세요. 연산 최적 프런티어(compute-optimal frontier)에 대해 `log10(C)` 대비 손실(loss)을 그래프로 그려 보세요. 교차 엔트로피(cross-entropy)를 0.1만큼 추가로 낮추기 위해 다음 단계에서 `>10^28` FLOPs가 필요하다고 예측되는 시점을 찾아보세요.
3. **어려움 (Hard).** 동일한 데이터셋으로 학습된 5개의 아주 작은 모델(파라미터 수 100K ~ 10M)을 사용하여 자신만의 스케일링 법칙(scaling law)을 피팅(fit)해 보세요. `α`와 `E`를 추정해 보세요. 여러분이 구한 지수(exponents)가 발표된 수치와 얼마나 잘 일치하나요?

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 파라미터 (Parameters, N) | "모델 크기" | 임베딩을 제외한 가중치 수; 모델의 용량을 결정합니다. |
| 토큰 (Tokens, D) | "학습 데이터" | 학습에 사용된 토큰의 수; 파라미터가 얼마나 잘 활용되는지를 결정합니다. |
| 연산량 (Compute, C) | "소모된 FLOPs" | 표준 트랜스포머의 경우 약 `6 × N × D`입니다. |
| 친칠라 최적 (Chinchilla-optimal) | "D/N ≈ 20" | 사전 학습(pretraining) 시 FLOP당 손실(loss)을 최소화하는 비율입니다. |
| 과잉 학습 (Over-training) | "Past Chinchilla" | 추론(inference) 시의 FLOP를 절약하기 위해 추가적인 학습 FLOP를 투입하는 것; D/N >> 20. |
| 줄일 수 없는 손실 (Irreducible loss) | "바닥(The floor)" | 스케일링 법칙(scaling law)의 `E` 항; 데이터 자체의 엔트로피를 의미합니다. |
| 창발적 능력 (Emergent capability) | "규모에 따른 급격한 도약" | 종종 평가 지표의 산물(artifact)인 경우가 많으며, 연속적인 손실(loss)은 매끄럽게 나타납니다. |
| 유효 연산량 (Effective compute) | "학습 효율성 승수" | 더 나은 데이터 / 옵티마이저 / 아키텍처는 하나의 FLOP가 도달할 수 있는 범위를 배가시킵니다. |

## 추가 읽을거리 (Further Reading)

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 최초의 스케일링 법칙(scaling law) 논문; 과소 학습(under-training) 상태를 다룹니다.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla 논문입니다.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 창발성(emergence)이 측정 방식에 따른 인위적 결과물(measurement artifact)일 수 있음을 다룹니다.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — Llama의 과잉 학습(over-training)이 왜 해당 워크로드에 적합한지 설명합니다.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 연산 효율을 2배로 높여주는 최적화 도구(optimizer)입니다.
