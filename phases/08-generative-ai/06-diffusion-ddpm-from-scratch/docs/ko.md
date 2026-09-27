# 확산 모델(Diffusion Models) — DDPM 밑바닥부터 구현하기

> Ho, Jain, Abbeel (2020)은 이 분야의 핵심 레시피를 제공했습니다. 천 번의 작은 단계를 거쳐 데이터를 노이즈로 파괴하세요. 그리고 노이즈를 예측하도록 하나의 신경망을 학습시킵니다. 추론 시에는 이 과정을 역으로 수행합니다. 오늘날 모든 주류 이미지, 비디오, 3D 및 음악 모델은 이 루프를 기반으로 작동하며, 그 위에 flow matching이나 consistency 기법이 추가되어 있을 수 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 문제점 (The Problem)

당신은 `p_data(x)`를 위한 샘플러를 원합니다. GAN은 종종 발산하는 미니맥스 게임(minimax game)을 수행합니다. VAE는 가우시안 디코더(Gaussian decoder)를 사용하여 흐릿한 샘플을 생성합니다. 당신이 진정으로 원하는 것은 (a) 단일하고 안정적인 손실 함수(안장점이나 미니맥스가 없는), (b) `log p(x)`의 하한선(likelihood를 가질 수 있도록), (c) SOTA(최첨단) 품질에 부합하는 샘플을 생성하는 학습 목표입니다.

Sohl-Dickstein 등(2015)은 이론적인 해답을 제시했습니다: 가우시안 노이즈를 점진적으로 추가하는 마르코프 체인(Markov chain) `q(x_t | x_{t-1})`을 정의하고, 이를 역으로 노이즈를 제거하는 역방향 체인 `p_θ(x_{t-1} | x_t)`를 학습시키는 것입니다. Ho, Jain, Abbeel(2020)은 손실 함수를 '노이즈 예측'이라는 한 줄로 단순화할 수 있음을 보여주었으며 수학적 구조를 정립했습니다. 2020년에 이것은 하나의 호기심에 불과했습니다. 2021년에는 SOTA급 샘플을 만들어냈습니다. 2022년에는 Stable Diffusion이 되었습니다. 그리고 2026년, 이것은 모든 생성 모델의 기반(substrate)이 되었습니다.

## 개념 (The Concept)

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**순방향 과정 (Forward process) `q`.** `T`번의 작은 단계에 걸쳐 가우시안 노이즈를 추가합니다. 수학적으로 다루기 쉬운 이유는 누적 단계 또한 가우시안 분포를 따른다는 폐쇄형(closed form) 공식이 존재하기 때문입니다:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

여기서 `α̅_t = ∏_{s=1..t} (1 - β_s)`이며, `β_t`는 스케줄에 따라 결정됩니다. `T=1000` 단계 동안 `β_t`를 1e-4에서 0.02까지 선형적으로 선택하면, `x_T`는 대략 `N(0, I)`가 됩니다.

**역방향 과정 (Reverse process) `p_θ`.** 추가된 노이즈를 예측하는 신경망 `ε_θ(x_t, t)`를 학습합니다. `x_t`가 주어졌을 때, 다음 식을 통해 노이즈를 제거(denoise)합니다:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

여기서 `σ_t`는 `sqrt(β_t)`이거나 학습된 분산입니다. 식은 복잡해 보이지만 단순한 대수학적 계산입니다. 사후 확률(posterior) `q(x_{t-1} | x_t, x_0)`를 바탕으로 `x_{t-1}`에 대해 풀고, `x_0`를 노이즈 예측값으로 대체한 결과입니다.

**학습 손실 (Training loss).**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

데이터에서 `x_0`를 샘플링하고, 무작위로 `t`를 선택하며, `ε ~ N(0, I)`를 샘플링합니다. 그 후 폐쇄형 공식을 통해 노이즈가 섞인 `x_t`를 한 번에 계산하고, 노이즈에 대해 회귀(regress)합니다. 단일 손실 함수를 사용하며, minimax, KL, 또는 재매개변수화 트릭(reparameterization tricks)이 필요하지 않습니다.

**샘플링 (Sampling).** `x_T ~ N(0, I)`에서 시작합니다. `t = T`부터 `1`까지 역방향 단계를 반복합니다. 완료되었습니다.

## 작동 원리 (Why it works)

세 가지 직관:

1. **노이즈 제거(Denoising)는 쉽고, 생성(Generating)은 어렵습니다.** `t=T`일 때 데이터는 순수한 노이즈 상태이며, 네트워크는 사소한 문제를 해결하면 됩니다. `t=0`일 때 네트워크는 단지 몇 개의 픽셀만 정리하면 됩니다. 중간 단계인 `t`에서는 문제가 어렵지만, 네트워크는 모든 노이즈 레벨로부터 동일한 가중치로 흐르는 수많은 그래디언트(gradient)를 가집니다.

2. **변형된 형태의 스코어 매칭(Score matching in disguise).** Vincent (2011)는 노이즈를 예측하는 것이 스코어(score)인 `∇_x log q(x_t | x_0)`를 추정하는 것과 동일함을 증명했습니다. 역방향 SDE(Reverse SDE)는 이 스코어를 사용하여 밀도 그래디언트(density gradient)를 따라 올라갑니다. 즉, 고확률 영역을 향해 안내된 무작위 보행(guided random walk)을 수행하는 것입니다.

3. **ELBO가 단순한 MSE로 축소됩니다.** 전체 변분 하한(variational lower bound)은 타임스텝당 하나의 KL 항을 가집니다. DDPM의 파라미터화(parameterization)를 사용하면 이러한 KL 항들은 특정 계수가 적용된 노이즈 예측에 대한 MSE로 단순화됩니다. Ho는 이 계수들을 생략하고 이를 "단순한(simple)" 손실 함수라고 불렀으며, 오히려 품질이 *향상*되었습니다.

```figure
diffusion-denoise
```

## 구축하기 (Build It)

`code/main.py`는 1-D DDPM을 구현합니다. 데이터는 두 개의 모드를 가진 혼합 분포(two-mode mixture)입니다. "net"은 `(x_t, t)`를 입력받아 예측된 노이즈를 출력하는 아주 작은 MLP입니다. 학습은 한 줄의 손실 함수(loss)로 이루어집니다. 샘플링은 역방향 체인(reverse chain)을 반복합니다.

### 1단계: 순방향 스케줄 (Forward Schedule, closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 2단계: 한 번에 `x_t` 샘플링하기 (sample `x_t` in one shot)

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 3단계: 한 번의 학습 단계 (one training step)

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 4단계: 역샘플링 (reverse sampling)

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

40개의 타임스텝(timesteps)과 24개의 유닛을 가진 MLP를 사용하는 1차원 문제의 경우, 약 200 에포크(epochs) 내에 두 개의 모드가 섞인 혼합 분포(two-mode mixture)를 학습합니다.

## 시간 조건화 (Time conditioning)

네트워크는 현재 어떤 타임스텝(timestep)을 디노이징(denoising)하고 있는지 알아야 합니다. 두 가지 표준적인 옵션이 있습니다:

- **사인파 임베딩 (Sinusoidal embedding).** Transformer의 위치 인코딩(positional encoding)과 유사합니다. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. 이를 MLP에 통과시킨 후 네트워크 전체에 브로드캐스트(broadcast)합니다.
- **FiLM / 그룹 정규화 조건화 (FiLM / group-norm conditioning).** 각 블록에서 임베딩을 채널별 스케일/편향(scale/bias)으로 투영(FiLM)합니다.

우리의 예제 코드는 사인파 → 결합(concat) 방식을 사용합니다. 실제 프로덕션용 U-Net은 FiLM을 사용합니다.

## 주의 사항 (Pitfalls)

- **스케줄(Schedule)이 매우 중요합니다.** 선형 `β` (Linear `β`)는 DDPM의 기본값이지만, 코사인 스케줄(cosine schedule; Nichol & Dhariwal, 2021)이 동일한 연산량 대비 더 나은 FID를 제공합니다. 품질이 정체된다면 스케줄을 변경해 보세요.
- **타임스텝 임베딩(Timestep embedding)은 취약합니다.** 단순한 1차원 예제에서는 `t`를 실수(float) 형태로 직접 전달해도 작동하지만, 이미지에서는 실패합니다. 항상 적절한 임베딩을 사용하세요.
- **V-prediction vs ε-prediction.** 좁은 영역(매우 작거나 매우 큰 `t`)에서 `ε`는 신호 대 잡음비(signal-to-noise)가 낮습니다. V-prediction (`v = α·ε - σ·x`)이 더 안정적이며, SDXL, SD3, Flux에서 이를 사용합니다.
- **Classifier-free guidance.** 추론 시, 조건부(conditional) `ε`와 비조건부(unconditional) `ε`를 모두 계산한 다음, `w ≈ 3-7` 범위에서 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`를 적용합니다. 이 내용은 08강에서 다룹니다.
- **1000단계(steps)는 너무 많습니다.** 실제 서비스 환경에서는 DDIM (20-50단계), DPM-Solver (10-20단계) 또는 증류(distillation, 1-4단계) 기법을 사용합니다. 12강을 참고하세요.

## 활용 사례 (Use It)

| 역할 (Role) | 2026년 전형적인 스택 (Typical stack in 2026) |
|------|-----------------------|
| 이미지 픽셀 공간 확산 (Image pixel-space diffusion, 소규모/토이 모델) | DDPM + U-Net |
| 이미지 잠재 확산 (Image latent diffusion) | VAE encoder + U-Net 또는 DiT (Lesson 07) |
| 비디오 잠재 확산 (Video latent diffusion) | 시공간 DiT (Spatiotemporal DiT; Sora, Veo, WAN) |
| 오디오 잠재 확산 (Audio latent diffusion) | Encodec + diffusion transformer |
| 과학 (Science; 분자, 단백질, 물리학) | 등변 확산 (Equivariant diffusion; EDM, RFdiffusion, AlphaFold3) |

확산(Diffusion)은 보편적인 생성 백본(generative backbone)입니다. 플로우 매칭(Flow matching, Lesson 13)은 2024-2026년의 경쟁 기술로, 동일한 품질에서 일반적으로 추론 속도(inference speed) 측면에서 우위를 점합니다.

## Ship It

`outputs/skill-diffusion-trainer.md`를 저장하세요. Skill은 데이터셋과 연산 예산(compute budget)을 입력받아 다음 항목들을 출력합니다: 스케줄(schedule; linear/cosine/sigmoid), 예측 대상(prediction target; ε/v/x), 단계 수(number of steps), 가이던스 스케일(guidance scale), 샘플러 계열(sampler family), 그리고 평가 프로토콜(eval protocol)입니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에서 `T`를 40에서 10으로 변경해 보세요. 샘플 품질(출력의 시각적 히스토그램)이 어떻게 저하되나요? 어느 정도의 `T`에서 이봉 분포(two-mode) 구조가 붕괴되나요?
2. **중간 (Medium).** $\epsilon$-prediction에서 $v$-prediction으로 전환해 보세요. 역단계(reverse step)를 다시 유도해 보세요. 최종 샘플 품질을 비교해 보세요.
3. **어려움 (Hard).** 분류기 없는 가이드(classifier-free guidance)를 추가해 보세요. 클래스 레이블 `c ∈ {0, 1}`을 조건으로 설정하고, 학습 중 10%의 확률로 이를 누락(drop)시키세요. 샘플링 시에는 `ε = (1+w)·ε_cond - w·ε_uncond`를 사용합니다. `w = 0, 1, 3, 7`일 때 조건부 모드 적중률(conditional-mode-hit rate)을 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Forward process (순방향 과정) | "노이즈 추가" | 데이터를 파괴하는 고정된 마르코프 체인(Markov chain) `q(x_t \| x_{t-1})`. |
| Reverse process (역방향 과정) | "디노이징(Denoising)" | 데이터를 재구성하는 학습된 체인 `p_θ(x_{t-1} \| x_t)`. |
| β schedule (베타 스케줄) | "노이즈 사다리" | 단계별 분산(variance); 선형(linear), 코사인(cosine), 또는 시그모이드(sigmoid) 방식. |
| α̅ (알파 바) | "Alpha bar" | 누적 곱 `∏(1 - β)`; `x_0`로부터 `x_t`를 구하는 폐쇄형(closed-form) 식을 제공함. |
| Simple loss (단순 손실 함수) | "노이즈에 대한 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`; 모든 변분 유도(variational derivations)가 이 식으로 수렴함. |
| ε-prediction (엡실론 예측) | "노이즈 예측" | 출력값이 추가된 노이즈임; 표준 DDPM 방식. |
| V-prediction (V-예측) | "속도(velocity) 예측" | 출력값이 `α·ε - σ·x`임; $t$ 전반에 걸쳐 더 나은 조건화(conditioning)를 제공함. |
| DDPM | "그 논문" | Ho et al. 2020; 선형 $\beta$, 1000단계, U-Net 사용. |
| DDIM | "결정론적 샘플러" | 비-마르코프(Non-Markov) 샘플러, 20-50단계, 동일한 학습 목표 사용. |
| Classifier-free guidance (CFG) | "CFG" | 조건부(conditional) 및 무조건부(unconditional) 노이즈 예측을 혼합하여 조건화를 증폭함. |

## 프로덕션 노트: 확산 추론(diffusion inference)은 단계 수(step-count)의 문제입니다

DDPM 논문은 $T=1000$번의 역과정(reverse steps)을 수행합니다. 하지만 실제 프로덕션 환경에서 이를 그대로 사용하는 경우는 없습니다. 모든 실제 추론 스택은 다음 세 가지 전략 중 하나를 선택하며, 각 전략은 "지연 시간(latency)이 어디에서 발생하는가"라는 프로덕션 관점의 프레임워크와 명확하게 매칭됩니다.

1. **동일 모델, 더 빠른 샘플러(Faster sampler, same model).** DDIM (20-50단계), DPM-Solver++ (10-20단계), UniPC (8-16단계). 역과정 루프를 교체하는 방식이며, 학습된 `ε_θ` 가중치는 그대로 유지됩니다. 지연 시간을 20~50배 단축합니다.
2. **증류(Distillation).** 학생 모델(student)이 교사 모델(teacher)을 더 적은 단계로 따라하도록 학습시킵니다: Progressive Distillation (2 → 1단계), Consistency Models (임의 단계 → 1-4단계), LCM, SDXL-Turbo, SD3-Turbo. 지연 시간을 추가로 5~10배 단축하지만, 재학습이 필요합니다.
3. **캐싱 및 컴파일(Caching and compilation).** `torch.compile(unet, mode="reduce-overhead")`, TensorRT-LLM의 확산 백엔드, `xformers`/SDPA 어텐션, bf16 가중치 사용. 단계당 지연 시간을 약 2배 단축합니다. (1)번 및 (2)번 전략과 병행하여 사용할 수 있습니다.

프로덕션 확산 서버의 예산(budget) 논의는 프로덕션 문헌에서 LLM을 설명하는 방식과 동일합니다. 지연 시간(latency)은 `num_steps × step_cost + VAE_decode`이며, 처리량(throughput)은 `batch_size × (num_steps × step_cost)^-1`입니다. TTFT(첫 토큰 생성 시간)는 매우 작지만(한 단계), 사용자 관점에서는 이미지 생성이 "한 번에(all-at-once)" 이루어지므로 TPOT(토큰당 생성 시간)에 상응하는 값은 전체 응답 시간이 됩니다.

## 추가 읽을거리 (Further Reading)

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 시대를 앞서간 확산(diffusion) 논문입니다.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM입니다.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — 더 적은 단계로 작동하는 DDIM입니다.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 코사인 스케줄(cosine schedule)과 학습된 분산(learned variance)을 다룹니다.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 분류기 가이드(classifier guidance)를 소개합니다.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG입니다.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 통합된 표기법과 가장 깔끔한 레시피를 제공합니다.
