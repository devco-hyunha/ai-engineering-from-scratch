# Flow Matching & Rectified Flows

> 확산 모델(Diffusion Model)은 잡음에서 데이터까지 곡선 경로를 따르기 때문에 20~50개의 샘플링 단계가 필요합니다. Flow Matching (Lipman et al., 2023)과 Rectified Flow (Liu et al., 2022)는 직선 경로를 학습합니다. 더 직선적인 경로는 더 적은 단계, 즉 더 빠른 추론을 의미합니다. Stable Diffusion 3, Flux.1, AudioCraft 2는 모두 2024년에 Flow Matching으로 전환했습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 06강 (DDPM), 1단계 · 미적분
**시간:** 약 45분

## 문제점

DDPM의 역과정은 `N(0, I)`에서 데이터 분포로 돌아가는 1000단계 확률적 이동입니다. DDIM은 이를 20~50개의 결정론적 단계로 축약했습니다. 더 적은 단계, 이상적으로는 단 한 단계가 필요합니다. 장애물은 역과정을 푸는 ODE가 강성(stiff)이며 경로가 곡선이라는 점입니다.

잡음에서 데이터까지의 경로를 *직선*으로 학습할 수 있다면, `t=1`에서 `t=0`까지 단일 오일러(Euler) 단계로 작동할 수 있습니다. Flow Matching은 이를 직접 구축합니다: `x_1 ∼ N(0, I)`에서 `x_0 ∼ data`까지의 직선 보간을 정의하고, 그 시간 미분값을 일치하도록 벡터장 `v_θ(x, t)`를 학습한 뒤, 추론 시 적분합니다.

Rectified Flow (Liu 2022)는 한 발 더 나아갑니다: reflow 절차를 통해 경로를 반복적으로 직선화하여 점점 더 선형에 가까운 ODE를 생성합니다. 두 번의 reflow 반복 후, 2단계 샘플러가 50단계 DDPM의 품질과 일치합니다.

## 개념

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 직선 흐름

정의:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

여기서 `x_0 ~ data`이고 `x_1 ~ N(0, I)`입니다. 이 직선을 따른 시간 미분값은 상수입니다:

```
dx_t / dt = x_1 - x_0
```

신경망 벡터장 `v_θ(x_t, t)`를 정의하고 이 미분값을 일치하도록 학습합니다:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

이것이 **조건부 Flow Matching** 손실(Lipman 2023)입니다. 학습은 시뮬레이션이 필요 없습니다: ODE를 전개하지 않습니다. `(x_0, x_1, t)`를 샘플링하고 회귀(regress)만 수행합니다.

### 샘플링

추론 시, 학습된 벡터장을 시간 *역방향*으로 적분합니다:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

`x_1 ~ N(0, I)`에서 시작하여 `t=0`까지 오일러 단계로 내려갑니다.

### Rectified Flow (Liu 2022)

직선 흐름은 작동하지만 학습된 경로가 *실제로는 직선이 아닙니다* — 많은 `x_0`가 동일한 `x_1`로 매핑될 수 있기 때문에 경로가 곡선으로 휘어집니다. Rectified flow의 reflow 단계는 다음과 같습니다:

1. 무작위 쌍으로 flow 모델 v_1을 학습합니다.
2. v_1을 `x_1`에서 그 착지점 `x_0`까지 적분하여 N개의 쌍 `(x_1, x_0)`을 샘플링합니다.
3. 이 쌍으로 v_2를 학습합니다. 쌍이 이제 "ODE 매칭" 상태이므로, 두 점 사이의 직선 보간은 실제로 더 평평합니다.
4. 반복합니다.

실제로는 2번의 reflow 반복으로 거의 선형적인 상태에 도달하여 2-4단계 추론이 가능해집니다. SDXL-Turbo, SD3-Turbo, LCM은 모두 flow matching 모델에서 증류(distilled)된 모델입니다.

### 2024년 이미지 생성에서 이 방식이 승리한 이유

세 가지 이유:

1. **시뮬레이션 없는 학습** — 학습 중 ODE 언로딩(unrolling)이 필요 없으며, 구현이 간단합니다.
2. **더 나은 손실 기하학** — 직선 경로는 일관된 신호 대 잡음비를 가지며, DDPM ε-loss는 스케줄의 끝부분에서 SNR이 좋지 않습니다.
3. **더 빠른 추론** — SDXL-Turbo 품질로 4-8단계 추론이 가능하며, consistency distillation을 사용하면 1단계 추론이 가능합니다.

## Flow matching vs DDPM — 정확한 연결 고리

가우시안 조건부 경로를 사용하는 flow matching은 *특정 노이즈 스케줄을 가진* diffusion입니다. `x_t = α(t) x_0 + σ(t) x_1` 스케줄을 선택하면 flow matching은 `v = α'·x_0 - σ'·x_1`을 사용하여 Stratonovich로 재공식화된 diffusion을 복원합니다. 가우시안 경로에 대해 두 방법은 대수적으로 동등합니다.

Flow matching이 추가한 것: 목표(단순한 속도)의 *명확성*, 더 깔끔한 손실 함수, 그리고 비가우시안 보간법을 실험할 수 있는 자유로움입니다.

```figure
normalizing-flow
```

## 구현하기

`code/main.py`은 2-모드 가우시안 혼합에 대한 1-D flow matching을 구현합니다. 벡터 필드 `v_θ(x, t)`은 직선 목표(target)로 학습된 작은 MLP입니다. 추론 시 1, 2, 4, 20단계의 Euler 적분 단계를 수행하고 샘플 품질을 비교합니다.

### 1단계: 학습 손실

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 역전파 + 업데이트
```

### 2단계: 다단계 추론

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 3단계: 단계 수 비교

4단계 샘플러가 이미 20단계 품질과 일치할 것으로 예상됩니다 — 이는 지연(latency)에 큰 영향을 미칩니다.

## 문제점

- **시간 매개변수화.** 플로우 매칭은 `t ∈ [0, 1]`을 사용하며, `t=0`은 데이터에서, `t=1`은 잡음에서 적용됩니다. DDPM은 `t ∈ [0, T]`을 사용하며, `t=0`은 데이터에서, `t=T`은 잡음에서 적용됩니다. 방향은 같지만 스케일이 다릅니다. 논문들이 이 부분을 자주 잘못 설명합니다.
- **스케줄 선택.** Rectified flow의 직선은 "the" flow-matching 스케줄이지만, 더 나은 스케일 커버리지를 위해 cosine 또는 logit-normal t-샘플링(SD3가 이를 사용함)을 사용할 수 있습니다.
- **Reflow 비용.** Reflow를 위한 쌍 데이터셋 생성은 샘플마다 전체 추론 패스가 필요합니다. 1-2 스텝 추론이 정말로 필요한 경우에만 reflow를 수행하세요.
- **Classifier-free guidance는 여전히 적용됩니다.** 선형 결합에서 ε를 v로 교체하기만 하면 됩니다: `v_cfg = (1+w) v_cond - w v_uncond`.

## 사용하기

| 사용 사례 | 2026 스택 |
|----------|-----------|
| 텍스트-이미지, 최고 품질 | Flow matching: SD3, Flux.1-dev |
| 텍스트-이미지, 1-4 스텝 | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 실시간 추론 | Flow-matched 기반의 Consistency distillation (LCM, PCM) |
| 오디오 생성 | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| 비디오 생성 | Flow matching과 diffusion의 혼합 (Sora, Veo, Stable Video) |
| 과학 / 물리 (입자 궤적, 분자) | Flow matching + equivariant vector field |

2025-2026년 논문에서 "diffusion보다 빠르다"라고 언급하는 경우, 거의 항상 flow matching + distillation입니다.

## 출시하기

`outputs/skill-fm-tuner.md`을 저장하세요. 스킬은 diffusion 스타일 모델 사양을 flow-matching 학습 구성으로 변환합니다: 스케줄 선택, 시간 샘플링 분포 (uniform / logit-normal), 옵티마이저, reflow 계획, 목표 스텝 수, 평가 프로토콜.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하고 1-스텝 vs 20-스텝 MSE를 실제 데이터 분포와 비교하세요.
2. **중간.** uniform `t` 샘플링을 logit-normal로 전환하세요 (샘플링을 mid-t에 집중시킴). 모델 품질이 개선됩니까?
3. **어려움.** reflow 반복 한 번을 구현하세요: 첫 번째 모델을 적분하여 쌍 (x_0, x_1)을 생성하고, 쌍으로 두 번째 모델을 학습한 후 1-스텝 샘플 품질을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 플로우 매칭 | "직선 확산" | `v_θ(x, t)`을 학습하여 `x_1 - x_0`을 보간 경로(interpolant)를 따라 매칭합니다. |
| 정류 플로우 | "Reflow" | 학습된 플로우를 직선화하는 반복 절차입니다. |
| 속도 필드 | "v_θ" | 모델의 출력 — `x_t`을 이동해야 하는 방향입니다. |
| 직선 보간 경로 | "The path" | `x_t = (1-t)·x_0 + t·x_1`; 자명한(target) 미분값입니다. |
| 오일러 샘플러 | "1차 ODE 솔버" | 가장 단순한 적분기; 경로가 직선일 때 잘 작동합니다. |
| 로짓 정규 t | "SD3 샘플링" | `t` 샘플링을 기울기가 가장 강한 중간 값에 집중합니다. |
| 일관성 증류 | "1-스텝 샘플러" | 학생(student) 모델을 학습하여 임의의 `x_t`을 `x_0`에 직접 매핑합니다. |
| 속도 기반 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; 동일한 트릭, 새로운 변수입니다. |

## 프로덕션 노트: Flux.1-schnell은 플로우 매칭의 가장 빠른 형태입니다

플로우 매칭의 프로덕션 성과는 Flux.1-schnell입니다. Flux-dev급 품질을 유지하면서 추론 스텝을 1-4 스텝으로 증류(distilled)한 플로우 매칭 DiT입니다. Niels의 "8GB 머신에서 Flux 실행하기" 노트북은 참고 배포 레시피입니다: T5 + CLIP 인코딩, 양자화된 MMDiT 디노이즈 (schnell은 4 스텝, dev는 50 스텝), VAE 디코딩. 비용 계산은 다음과 같습니다:

| 변형 | 스텝 | L4에서 1024² 지연 시간 | 총 FLOPs (상대적) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (원본) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12배 빠름) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-스텝 | 2 | ~0.3 s | 0.03× |

프로덕션 규칙: **플로우 매칭 기반 모델 + 증류 = 2026년 빠른 텍스트-투-이미지의 기본값입니다.** 모든 주요 벤더가 이 조합을 제공합니다: SD3-Turbo (SD3 + 플로우 + 증류), Flux-schnell (Flux-dev + 정류 플로우 직선화), CogView-4-Flash. 순수 확산 기반 모델은 레거시 체크포인트에만 존재합니다.

## 추가 읽기

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 정류 플로우(rectified flow).
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 플로우 매칭(flow matching).
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — 대규모 정류 플로우, SD3.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — FM + 확산을 포괄하는 일반 프레임워크.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 확산/유량의 1단계 증류입니다.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — 터보 변형입니다.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 프로덕션 환경에서의 유량 매칭입니다.
