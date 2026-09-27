# Flow Matching & Rectified Flows (플로우 매칭 및 정류 흐름)

> 확산 모델(Diffusion models)은 노이즈에서 데이터로 가는 곡선 경로를 따라 이동하기 때문에 20~50단계의 샘플링 단계가 필요합니다. Flow matching (Lipman et al., 2023)과 rectified flow (Liu et al., 2022)는 직선 경로를 학습합니다. 경로가 직선에 가까울수록 필요한 단계가 줄어들고 추론 속도가 빨라집니다. Stable Diffusion 3, Flux.1, AudioCraft 2는 모두 2024년에 flow matching 방식으로 전환했습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## 문제점 (The Problem)

DDPM의 역과정(reverse process)은 `N(0, I)`에서 데이터 분포로 되돌아가는 1000단계의 확률적 보행(stochastic walk)입니다. DDIM은 이를 20~50단계의 결정론적(deterministic) 단계로 압축했습니다. 여러분은 더 적은 단계, 이상적으로는 단 한 단계를 원합니다. 여기서 걸림돌은 역과정을 해결하는 ODE(상미분 방정식)가 경직(stiff)되어 있다는 점, 즉 경로가 곡선이라는 점입니다.

만약 노이즈에서 데이터로 가는 경로가 *직선(straight line)*이 되도록 모델을 학습시킬 수 있다면, `t=1`에서 `t=0`까지 단 한 번의 오일러 단계(Euler step)만으로도 작동할 것입니다. 플로우 매칭(Flow matching)은 이를 직접적으로 구축합니다. 즉, `x_1 ∼ N(0, I)`에서 `x_0 ∼ data`로 가는 직선 보간(straight-line interpolation)을 정의하고, 벡터장 `v_θ(x, t)`가 그 시간 미분값과 일치하도록 학습한 뒤, 추론 시에 이를 적분합니다.

정류 흐름(Rectified flow, Liu 2022)은 여기서 더 나아갑니다. 점진적으로 선형에 가까운 ODE를 생성하는 리플로우(reflow) 절차를 통해 경로를 반복적으로 직선화합니다. 두 번의 리플로우 반복을 거치면, 2단계 샘플러가 50단계 DDPM과 동일한 품질을 구현할 수 있습니다.

## 개념 (The Concept)

![Flow matching: 노이즈와 데이터 사이의 직선 보간(straight-line interpolation)](../assets/flow-matching.svg)

### 직선 흐름 (Straight-line flow)

정의:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

여기서 `x_0 ~ data`이고 `x_1 ~ N(0, I)`입니다. 이 직선을 따른 시간 미분값은 일정합니다:

```
dx_t / dt = x_1 - x_0
```

신경망 벡터장(neural vector field) `v_θ(x_t, t)`를 정의하고, 이 미분값과 일치하도록 학습시킵니다:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

이것이 **조건부 흐름 매칭(conditional flow matching)** 손실 함수입니다 (Lipman 2023). 학습 과정은 시뮬레이션이 필요 없습니다(simulation-free). 즉, ODE를 전개(unroll)할 필요 없이, 단순히 `(x_0, x_1, t)`를 샘플링하여 회귀(regress)하면 됩니다.

### 샘플링 (Sampling)

추론(Inference) 시에는 학습된 벡터장(vector field)을 시간의 *역방향*으로 통합합니다:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

`x_1 ~ N(0, I)`에서 시작하여, 오일러 방법(Euler-step)을 통해 `t=0`까지 단계적으로 내려가 보세요.

### Rectified flow (Liu 2022)

직선 흐름(Straight-line flow) 방식은 작동하지만, 학습된 경로가 *실제로 직선은 아닙니다*. 많은 `x_0`가 동일한 `x_1`로 매핑될 수 있기 때문에 경로가 휘어지게 됩니다. Rectified flow의 리플로우(reflow) 단계는 다음과 같습니다:

1. 무작위 쌍(random pairings)을 사용하여 흐름 모델 `v_1`을 학습시킵니다.
2. `x_1`에서 해당 도착 지점인 `x_0`까지 `v_1`을 적분하여 $N$개의 쌍 `(x_1, x_0)`을 샘플링합니다.
3. 해당 쌍들을 사용하여 `v_2`를 학습시킵니다. 이제 쌍들이 "ODE-매칭(ODE-matched)"되었기 때문에, 그 사이의 직선 보간(straight-line interpolant)이 실제로 더 평탄해집니다.
4. 이 과정을 반복합니다.

실제로 2번의 리플로우 반복을 거치면 거의 선형에 가까운 상태가 되어, 2~4단계의 추론(inference)이 가능해집니다. SDXL-Turbo, SD3-Turbo, LCM은 모두 흐름 매칭(flow-matching)으로부터 증류(distilled)된 모델들입니다.

### 2024년 이미지 분야에서 이것이 승리한 이유 (Why this won for images in 2024)

세 가지 이유:

1. **시뮬레이션이 필요 없는 학습 (Simulation-free training)** — 학습 중 ODE 언롤링(unrolling)이 필요 없으며, 구현이 매우 간단합니다.
2. **더 나은 손실 기하학 (Better loss geometry)** — 직선 경로는 일관된 신호 대 잡음비(SNR)를 갖는 반면, DDPM의 `ε-loss`는 스케줄의 경계 부분에서 SNR이 좋지 않습니다.
3. **더 빠른 추론 (Faster inference)** — SDXL-Turbo 수준의 품질을 4~8단계 내에 달성하며, 일관성 증류(consistency distillation)를 사용하면 1단계로 가능합니다.

## Flow matching vs DDPM — 정확한 연결 고리 (The exact connection)

가우시안 조건부 경로(Gaussian-conditional path)를 사용하는 Flow matching은 *특정한 노이즈 스케줄(noise schedule)을 가진* 확산(diffusion) 모델입니다. `x_t = α(t) x_0 + σ(t) x_1` 스케줄을 선택하면, Flow matching은 `v = α'·x_0 - σ'·x_1`인 Stratonovich 재구성 확산(Stratonovich-reformulated diffusion)을 복원해냅니다. 가우시안 경로에 대해서는 이 두 방식이 대수적으로 동일합니다.

Flow matching이 추가한 점: 타겟의 *명확성*(단순한 속도), 더 깔끔한 손실 함수(loss), 그리고 비가우시안 보간법(non-Gaussian interpolants)을 실험할 수 있는 자유입니다.

```figure
normalizing-flow
```

## 구현하기 (Build It)

`code/main.py`는 두 개의 모드를 가진 가우시안 혼합 모델(Gaussian mixture)에 대한 1차원 플로우 매칭(1-D flow matching)을 구현합니다. 벡터장 `v_θ(x, t)`는 직선 타겟(straight-line target)을 사용하여 학습된 아주 작은 MLP입니다. 추론(inference) 단계에서는 1, 2, 4, 20회의 오일러 단계(Euler steps)를 각각 적분하여 샘플 품질을 비교해 보세요.

### 1단계: 학습 손실 (training loss)

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

### 2단계: 다단계 추론 (multi-step inference)

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 3단계: 스텝 수 비교 (compare step counts)

4-스텝 샘플러(4-step sampler)가 이미 20-스텝의 품질과 일치할 것으로 기대합니다. 이는 지연 시간(latency) 측면에서 매우 중요한 요소입니다.

## 주의 사항 (Pitfalls)

- **시간 매개변수화 (Time parameterization).** Flow matching은 `t ∈ [0, 1]`을 사용하며 `t=0`은 데이터, `t=1`은 노이즈를 나타냅니다. 반면 DDPM은 `t ∈ [0, T]`를 사용하며 `t=0`은 데이터, `t=T`는 노이즈를 나타냅니다. 방향은 같지만 스케일이 다릅니다. 논문들에서 이 부분을 지속적으로 혼동하곤 합니다.
- **스케줄 선택 (Schedule choice).** Rectified flow의 직선(straight line)이 "표준적인" flow-matching 스케줄이지만, 더 나은 스케일 커버리지를 위해 코사인(cosine) 또는 로짓-노멀(logit-normal) `t`-샘플링(SD3가 이 방식을 사용함)을 사용할 수 있습니다.
- **Reflow 비용 (Reflow cost).** Reflow를 위한 쌍을 이룬 데이터셋(paired dataset)을 생성하는 것은 샘플당 전체 추론 과정을 한 번 거쳐야 함을 의미합니다. 1~2단계 추론(1-2 step inference)이 정말로 필요한 경우가 아니라면 reflow를 수행하지 마세요.
- **Classifier-free guidance의 적용.** 선형 결합에서 `ε`를 `v`로 바꾸기만 하면 됩니다: `v_cfg = (1+w) v_cond - w v_uncond`.

## 활용 사례 (Use It)

| 활용 사례 (Use case) | 2026년 스택 (2026 stack) |
|----------|-----------|
| 텍스트-이미지 생성, 최고 품질 (Text-to-image, best quality) | Flow matching: SD3, Flux.1-dev |
| 텍스트-이미지 생성, 1-4 단계 (Text-to-image, 1-4 steps) | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 실시간 추론 (Real-time inference) | Flow-matched 베이스 모델로부터의 Consistency distillation (LCM, PCM) |
| 오디오 생성 (Audio generation) | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| 비디오 생성 (Video generation) | Flow matching과 Diffusion의 혼합 (Sora, Veo, Stable Video) |
| 과학 / 물리학 (입자 궤적, 분자) (Science / physics) | Flow matching + equivariant vector field |

2025-2026년 사이에 논문에서 "diffusion보다 빠르다(faster than diffusion)"라고 언급한다면, 그것은 거의 항상 flow matching + distillation 기술을 의미합니다.

## Ship It (실행하기)

`outputs/skill-fm-tuner.md`를 저장하세요. 이 스킬은 확산 방식(diffusion-style)의 모델 사양을 입력받아 다음과 같은 요소들을 포함한 플로우 매칭(flow-matching) 학습 설정으로 변환합니다: 스케줄 선택, 시간 샘플링 분포(uniform / logit-normal), 옵티마이저, 리플로우(reflow) 계획, 목표 단계 수, 평가 프로토콜.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하고, 1단계(1-step)와 20단계(20-step) MSE를 실제 데이터 분포(true data distribution)와 비교해 보세요.
2. **중간 (Medium).** 균등한(uniform) `t` 샘플링에서 로짓-노멀(logit-normal) 샘플링(중간 `t` 값에 샘플링을 집중시킴)으로 전환해 보세요. 모델의 품질이 향상되나요?
3. **어려움 (Hard).** 한 번의 리플로우(reflow) 반복을 구현해 보세요: 첫 번째 모델을 적분하여 쌍을 이룬 `(x_0, x_1)`을 생성하고, 이 쌍들을 사용하여 두 번째 모델을 학습시킨 뒤, 1단계 샘플 품질을 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Flow matching | "직선 확산(Straight-line diffusion)" | 보간법(interpolant)을 따라 `x_1 - x_0`와 일치하도록 `v_θ(x, t)`를 학습시키는 것. |
| Rectified flow | "Reflow" | 학습된 흐름(flow)을 직선화하는 반복적인 절차. |
| Velocity field | "v_θ" | 모델의 출력값 — `x_t`가 이동해야 할 방향. |
| Straight-line interpolant | "경로(The path)" | `x_t = (1-t)·x_0 + t·x_1`; 타겟 미분값이 단순함. |
| Euler sampler | "1차 ODE 솔버(1st order ODE solver)" | 가장 단순한 적분기; 경로가 직선일 때 잘 작동함. |
| Logit-normal t | "SD3 샘플링(SD3 sampling)" | 기울기(gradient)가 가장 강한 중간값 쪽으로 `t` 샘플링을 집중시키는 것. |
| Consistency distillation | "1단계 샘플러(1-step sampler)" | 어떤 `x_t`라도 `x_0`로 직접 매핑하도록 학생 모델을 학습시키는 것. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; 동일한 기법을 새로운 변수에 적용함. |

## 프로덕션 노트: Flux.1-schnell은 가장 빠른 flow matching 모델입니다

Flow matching의 프로덕션 측면에서의 승리는 Flux.1-schnell입니다. 이는 Flux-dev 수준의 품질을 유지하면서 1~4회의 추론 단계(inference steps)로 증류(distilled)된 flow-matched DiT입니다. Niels의 "8GB 머신에서 Flux 실행하기" 노트북은 표준 배포 레시피 역할을 합니다: T5 + CLIP 인코딩, 양자화된 MMDiT 디노이징(`schnell`은 4단계, `dev`는 50단계), VAE 디코딩 순으로 진행됩니다. 비용 산출 결과는 다음과 같습니다:

| 변형(Variant) | 단계(Steps) | L4에서의 1024² 지연 시간(Latency) | 총 FLOPs (상대값) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12배 더 빠름) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

프로덕션 규칙: **flow-matched base + distillation(증류) = 빠른 text-to-image를 위한 2026년의 기본 표준**입니다. 모든 주요 벤더가 이 조합을 출시하고 있습니다: SD3-Turbo (SD3 + flow + distillation), Flux-schnell (Flux-dev + rectified-flow straightening), CogView-4-Flash. 순수 확산(pure diffusion) 기반 모델은 레거시 체크포인트로만 존재합니다.

## 추가 읽을거리 (Further Reading)

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow(정류 흐름).
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching(흐름 매칭).
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, 대규모 rectified flow.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — FM + diffusion을 아우르는 일반적인 프레임워크.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow의 1단계 증류(1-step distillation).
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 변형 모델.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 실제 서비스에 적용된 flow matching.
