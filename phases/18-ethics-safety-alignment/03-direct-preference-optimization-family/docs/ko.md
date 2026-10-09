# 직접 선호 최적화(DPO) 계열

> Rafailov et al. (2023)은 RLHF의 최적해가 선호 데이터에 대해 닫힌 형태(closed form)를 가진다는 것을 보여 주었습니다. 따라서 명시적인 보상 모델을 생략하고 정책을 직접 최적화할 수 있습니다. 이 통찰은 IPO, KTO, SimPO, ORPO, BPO라는 계열을 탄생시켰으며, 각각은 DPO의 특정 실패 모드를 해결합니다. 2026년 현재, 직접 정렬 알고리즘(DAA)은 PPO보다 더 많은 프론티어 후처리(post-training) 실행에 사용되고 있습니다. 하지만 02강에서 언급된 과최적화 곡선은 여전히 적용됩니다. DAA는 Goodhart의 법칙을 피하지 못하며, 단지 그 영향이 나타나는 위치를 이동시킬 뿐입니다.

**유형:** Learn
**언어:** Python (stdlib, six-variant preference-loss comparator)
**선수 요건:** 18단계 · 01강 (InstructGPT), 18단계 · 02강 (보상 해킹), 10단계 · 08강 (DPO 기초)
**시간:** 약 75분

## 학습 목표

- KL 제약이 있는 RLHF 최적해로부터 DPO의 닫힌 형태를 유도해 보세요.
- IPO, KTO, SimPO, ORPO, BPO가 각각 DPO의 어떤 실패 모드를 해결하는지 설명해 보세요.
- "암시적 보상 격차"와 "선호 강도"를 구분하고, IPO의 항등 매핑(identity mapping)이 중요한 이유를 설명해 보세요.
- 명시적인 보상 모델(RM)이 없음에도 불구하고 Rafailov et al. (NeurIPS 2024)이 DAA가 과최적화됨을 증명하는 이유를 설명해 보세요.

## 문제점

RLHF 목적 함수(01강):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

는 알려진 최적해를 가집니다:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

따라서 보상은 최적 정책과 참조 정책의 비율로 암시적으로 정의됩니다:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

이를 Bradley-Terry 선호 가능성 likelihood에 대입하면, 분할 함수 `Z(x)`는 `x`에만 의존하므로 상쇄됩니다. 남는 것은 정책 매개변수만으로 이루어진 손실 함수이며, 보상 모델은 필요하지 않습니다. 이것이 DPO입니다.

주의할 점: 이 유도 과정은 최적해가 도달 가능하고, 선호 데이터가 분포 내(in-distribution)에 있으며, 참조 정책이 true mode anchor라고 가정합니다. 이 가정들은 정확히 성립하지 않습니다. 계열의 각 구성원은 서로 다른 위반된 가정을 해결합니다.

## 개념

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

잘못될 수 있는 점:

- 암시적 보상 격차 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`는 상한이 없습니다. 작은 선호가 임의로 큰 격차를 생성할 수 있습니다.
- 손실은 선택된 응답과 거부된 응답의 로그 확률을 반대 방향으로 유도합니다. 거부된 응답의 로그 확률이 더 빠르게 떨어지는 한, 선택된 응답의 절대 로그 확률을 낮출 수 있습니다. 이것이 Degraded Chosen Response 현상입니다.
- 분포 밖의 선호(OOD preference)는 희소한 쌍과 희소한 쌍을 비교하며 임의의 암시적 보상(implicit reward)을 생성합니다.

### IPO (Azar et al., 2024)

Identity Preference Optimization은 선호 확률에 대한 로그 시그모이드(log-sigmoid)를 항등 매핑(identity mapping)으로 대체합니다. 손실은 유한한 타겟에 대한 제곱 오차(squared-error)가 됩니다:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

마진(margin)은 `1/(2 beta)`로 제한됩니다. 선호 강도와 암시적 보상 격차는 비례합니다. 폭발(blow-up)이 없습니다.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization은 쌍 구조(pairwise structure)를 완전히 제거합니다. 단일 레이블이 지정된 출력과 "원하는(desirable)" 또는 "원하지 않는(undesirable)" 이진 신호가 주어지면, 전망 이론(prospect theory)의 효용 함수로 매핑합니다:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

이득과 손실에 대한 가중치가 다릅니다(손실 회피(loss aversion)). 장점: 쌍이 없는(unpaired) 데이터를 사용할 수 있으며, 이는 훨씬 더 풍부합니다.

### SimPO (Meng et al., 2024)

Simple Preference Optimization은 학습 신호를 생성(generation)과 정렬합니다. 참조 정책(reference policy)을 완전히 제거하고 로그 우도(log-likelihood)를 길이로 정규화합니다:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

안정성을 위해 마진 `gamma`을 사용합니다. 길이 정규화는 DPO의 길이 편향(length-bias) 실패 모드(구성상 더 긴 `y_w`가 더 큰 로그 확률 격차를 생성)를 악용하는 유인을 제거합니다.

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization은 표준 SFT 음의 로그 우도(negative log-likelihood)에 선호 항을 추가합니다:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

참조 정책이 없습니다. SFT 항이 정규화자(regularizer) 역할을 합니다. 기본 모델에서 정렬된 모델까지 단일 단계로 학습합니다. 별도의 SFT 체크포인트가 필요 없습니다.

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

Degraded Chosen Responses 문제를 식별합니다. DPO는 `y_w > y_l`의 순위를 보존하지만, `y_w`의 절대 로그 확률이 떨어질 수 있습니다. BPO는 선택된 응답의 하향 이동(downward moves)에 페널티를 부여하는 한 줄의 보정을 추가합니다. Llama-3.1-8B-Instruct의 수학 추론에서 DPO 대비 +10.1%의 정확도 향상이 보고되었습니다.

### 보편적인 결과: DAAs는 여전히 과최적화(over-optimize)합니다

Rafailov et al.의 "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024)는 여러 데이터셋에서 KL 예산을 변경하며 DPO, IPO, SLiC로 정책을 학습했습니다. 황금 보상 대비 KL 곡선은 Gao et al.의 피크 및 붕괴 형태와 동일합니다. 암시적 보상은 학습 중 분포 밖(out-of-distribution) 샘플을 쿼리합니다. KL 정규화는 이를 안정화하지 못합니다.

DAA는 Goodhart의 함정을 피하지 못합니다. 문제가 발생하는 표면이 "보상 모델 과최적화"에서 "참조 정책 비율 과최적화"로 바뀔 뿐입니다. 보편적인 해결책인 더 나은 데이터, 앙상블, 조기 종료는 두 경우 모두에 적용됩니다.

### 이 중 선택하기 (2026)

- 대규모 쌍 선호 데이터가 있다면: 보수적인 beta를 사용한 DPO, 길이 편향이 명확하다면 SimPO를 사용하세요.
- 비쌍(binary) 피드백이 있다면: KTO를 사용하세요.
- 기본 모델에서 단일 단계 파이프라인을 원한다면: ORPO를 사용하세요.
- DPO 로그에서 선택된(chosen) log-probs가 저하되는 것을 본다면: BPO를 사용하세요.
- 선호 강도가 크게 다양하고 DPO가 포화(saturating) 상태라면: IPO를 사용하세요.

모든 연구실은 배터리 테스트로 다섯 가지 모두를 실행하고 작업별로 최적의 방법을 선택합니다. 수학 추론과 안전성에서 최적점이 동일할 이유는 없습니다.

```figure
dpo-margin
```

## 사용하기

`code/main.py`는 선호 강도가 쌍마다 다른 장난감 선호 데이터셋에서 여섯 가지 손실(DPO, IPO, KTO, SimPO, ORPO, BPO)을 비교합니다. 각 손실은 작은 softmax 정책을 사용하여 동일한 500개 쌍 샘플에 대해 최적화됩니다. 방법별 최종 승률, 선택된 log-prob 드리프트, 암시적 보상 분산을 플롯합니다.

## 출시하기

이 강의는 `outputs/skill-preference-loss-selector.md`을 생성합니다. 데이터셋 통계(쌍 vs 비쌍, 선호 강도 가변 vs 균일, 길이 분포)와 목표(단일 단계 vs SFT 후 선호 학습)가 주어지면, 선호 손실을 추천하고 보호하는 실패 모드를 보고하세요.

## 연습 문제

1. `code/main.py`을 실행하세요. DPO와 BPO의 최종 선택된 log-prob 감소량을 보고하세요. BPO는 더 높은 선택된 절대 확률을 유지해야 합니다. 이를 검증하세요.

2. 모든 쌍의 선호 강도가 동일하도록 선호 데이터를 수정하세요. 여섯 가지 방법 중 가장 견고한 것은 무엇이며, 어떤 것이 저하됩니까? 여기서 IPO의 장점을 설명하세요.

3. 거절된 응답이 선택된 응답보다 평균 2배 길도록 만드세요. 다른 것은 변경하지 않고, DPO의 길이 활용을 수치적으로 보여주고 SimPO의 수정을 확인하세요.

4. Rafailov et al. (NeurIPS 2024)은 DAA가 과최적화한다고 주장합니다. 단일 지점 버전을 재현하세요: 선택된 응답과 거절된 응답의 KL 발산을 플롯하고, 큰 beta에서 DPO의 과최적화를 관찰하세요.

5. BPO 논문 초록 (OpenReview b97EwMUWu7)을 읽으세요. BPO가 DPO에 추가하는 한 줄 수정을 적어보세요. `code/main.py`의 구현과 대조하여 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| DPO | "보상 모델 없는 RLHF" | 폐형 RLHF 최적해에서 유도된 손실; 정책 매개변수만 사용 |
| 암시적 보상 | "로그 비율" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO가 암시하는 보상 |
| IPO | "경계 DPO" | 로그 시그모이드를 항등 함수로 대체; 암시적 보상 간격이 `1/(2 beta)`로 제한됨 |
| KTO | "비쌍 DPO" | 손실 회피를 포함하여 단일 레이블에 대한 전망 이론 유틸리티 |
| SimPO | "참조 없는 DPO" | 길이 정규화된 로그 우도 + 마진; 참조 정책 없음 |
| ORPO | "단일 단계 DPO" | NLL + 오dds 비율 선호 항; 기본 모델에서 한 번의 패스로 학습 |
| BPO | "선택 보존 DPO" | 선택된 응답의 절대 로그 확률을 감소시키는 것에 대한 페널티를 추가한 DPO |
| 저하된 선택 | "선택이 내려감" | DPO는 거절된 응답이 더 빠르게 떨어지는 한 선택된 응답의 로그 확률을 감소시킴 |
| DAA | "직접 정렬 알고리즘" | 명시적 RM을 건너뛰는 모든 선호 손실 방법 |

## 추가 읽기

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
