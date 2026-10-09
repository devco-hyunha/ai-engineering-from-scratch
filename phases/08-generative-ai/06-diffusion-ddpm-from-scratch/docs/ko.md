# 확산 모델 — DDPM从零부터 구현하기

> Ho, Jain, Abbeel (2020)은 이 분야가 놓칠 수 없는 레시피를 제시했습니다. 천 개의 작은 단계로 데이터를 노이즈로 파괴합니다. 하나의 신경망을 훈련하여 노이즈를 예측합니다. 추론 시 이 과정을 역전합니다. 오늘날 모든 주류 이미지, 비디오, 3D 및 음악 모델은 이 루프를 기반으로 작동하며, 그 위에 플로우 매칭이나 일관성 트릭이 추가될 수 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 02강 (역전파), 8단계 · 02강 (변분 오토인코더 (VAE))
**시간:** 약 75분

## 문제점

`p_data(x)`에 대한 샘플러가 필요합니다. GAN은 종종 발산하는 미니맥스 게임을 수행합니다. VAE는 가우시안 디코더에서 흐릿한 샘플을 생성합니다. 실제로 원하는 것은 (a) 단일 안정적 손실 (안장점 없음, 미니맥스 없음), (b) `log p(x)`에 대한 하한 (따라서 우도 값을 가짐), (c) SOTA 품질과 일치하는 샘플을 제공하는 훈련 목표입니다.

Sohl-Dickstein et al. (2015)은 이론적 답변을 제시했습니다. 가우시안 노이즈를 점진적으로 추가하는 마르코프 체인 `q(x_t | x_{t-1})`를 정의하고, 노이즈를 제거하는 역 체인 `p_θ(x_{t-1} | x_t)`을 훈련하는 것입니다. Ho, Jain, Abbeel (2020)는 손실을 한 줄로 단순화할 수 있음을 보여줬습니다 — 노이즈를 예측하는 것 — 그리고 수식을 정리했습니다. 2020년에는 호기심에 불과했습니다. 2021년에는 최신(state-of-the-art) 샘플을 생성했습니다. 2022년에는 Stable Diffusion이 되었습니다. 2026년에는 기본 구성 요소가 되었습니다.

## 개념

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**순방향 과정 `q`.** `T`의 작은 단계로 가우시안 노이즈를 추가합니다. 폐형(closed form) — 수학이 tractable한 이유 — 누적 단계도 가우시안이라는 것입니다:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

여기서 `α̅_t = ∏_{s=1..t} (1 - β_s)`는 `β_t` 스케줄에 대한 값입니다. T=1000 단계에 걸쳐 1e-4에서 0.02까지 선형적으로 `β_t`를 선택하면 `x_T`는 대략 `N(0, I)`가 됩니다.

**역방향 과정 `p_θ`.** 추가된 노이즈를 예측하는 신경망 `ε_θ(x_t, t)`를 학습합니다. `x_t`가 주어지면, 다음으로 노이즈를 제거합니다:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

여기서 `σ_t`는 `sqrt(β_t)` 또는 학습된 분산입니다. 식은 복잡해 보이지만 단순한 대수입니다 — 사후 분포 `q(x_{t-1} | x_t, x_0)`가 주어졌을 때 `x_{t-1}`를 구하고, `x_0`를 노이즈 예측 추정치로 대체하는 것입니다.

**훈련 손실.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

데이터에서 `x_0`을 샘플링하고, 랜덤 `t`을 선택한 후 `ε ~ N(0, I)`을 샘플링하여, 폐형(closed form)으로 한 번에 노이즈가 포함된 `x_t`을 계산하고 노이즈에 대해 회귀합니다. 하나의 손실 함수만 사용하며, 미니맥스(minimax), KL, 재파라미터화(trick)는 필요하지 않습니다.

**샘플링.** `x_T ~ N(0, I)`에서 시작합니다. `t = T`부터 `1`까지 역방향 스텝을 반복합니다. 완료되었습니다.

## 왜 효과가 있는가

세 가지 직관:

1. **디노이징은 쉽지만 생성은 어렵습니다.** `t=T`에서는 데이터가 순수한 잡음이며, 네트워크는 자명한(trivial) 문제를 해결해야 합니다. `t=0`에서는 네트워크가 몇 개의 픽셀만 정리하면 됩니다. 중간 `t`에서는 문제가 어렵지만, 모든 잡음 수준에서 동일한 가중치를 통해 많은 기울기가 흐릅니다.

2. **변장된 점수 매칭(Score Matching).** Vincent (2011)는 잡음을 예측하는 것이 `∇_x log q(x_t | x_0)`, 즉 *점수(score)*를 추정하는 것과 동등하다는 것을 증명했습니다. 역방향 SDE는 이 점수를 사용하여 밀도 기울기를 따라 이동하며, 이는 고확률 영역으로의 가이드된 랜덤 워크입니다.

3. **ELBO는 단순한 MSE로 축소됩니다.** 전체 변분 하한(variational lower bound)에는 시간 스텝마다 KL 항이 있습니다. DDPM의 파라미터화에서는 이러한 KL 항이 특정 계수를 가진 잡음 예측에 대한 MSE로 단순화됩니다. Ho는 계수를 제거하고("simple" loss라 명명) 품질이 *향상*되었습니다.

```figure
diffusion-denoise
```

## 구현하기

`code/main.py`은 1-D DDPM을 구현합니다. 데이터는 두 모드(two-mode) 혼합입니다. "네트워크"는 `(x_t, t)`을 입력으로 받아 예측된 잡음을 출력하는 작은 MLP입니다. 학습은 한 줄의 손실 함수로 이루어집니다. 샘플링은 역방향 체인을 반복합니다.

### 1단계: 순방향 스케줄 (폐형)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 2단계: 한 번에 `x_t`을 샘플링

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 3단계: 한 번의 학습 스텝

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 4단계: 역방향 샘플링

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

40개의 시간 스텝과 24유닛 MLP를 가진 1-D 문제에서는 약 200 에포크(epoch) 안에 두 모드 혼합을 학습합니다.

## 시간 조건부(Time conditioning)

네트워크는 어떤 시간 스텝을 디노이징하는지 알아야 합니다. 두 가지 표준 옵션이 있습니다:

- **사인(sinusoidal) 임베딩.** 트랜스포머 위치 인코딩과 유사합니다. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. MLP를 통과하여 네트워크에 브로드캐스트합니다.
- **FiLM / 그룹 정규화 조건부.** 각 블록에서 임베딩을 채널별 스케일/바이어스(FiLM)로 투영합니다.

우리의 토이(toy) 코드는 사인 임베딩 → 연결(concat)을 사용합니다. 프로덕션 U-Net은 FiLM을 사용합니다.

## 함정

- **스케줄이 매우 중요합니다.** 선형 `β`은 DDPM의 기본값이지만, 코사인 스케줄(Nichol & Dhariwal, 2021)은 동일한 연산량에서 더 나은 FID를 제공합니다. 품질이 정체되면 스케줄을 변경해 보세요.
- **타임스텝 임베딩은 취약합니다.** `t`을 단순한 float로 전달하는 방식은 장난감 1-D 데이터에는 작동하지만 이미지에서는 실패합니다. 항상 적절한 임베딩을 사용하세요.
- **V-예측 vs ε-예측.** 좁은 영역(매우 작은 t 또는 매우 큰 t)에서는 `ε`의 신호 대 잡음비가 낮습니다. V-예측(`v = α·ε - σ·x`)은 더 안정적이며, SDXL, SD3, Flux가 이를 사용합니다.
- **분류기 없는 가이드(Classifier-free guidance).** 추론 시 조건부 `ε`과 비조건부 을 모두 계산한 후, `w ≈ 3-7`를 사용하여 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`을 수행합니다. 08강에서 다루었습니다.
- **1000 스텝은 많습니다.** 프로덕션에서는 DDIM(20-50 스텝), DPM-Solver(10-20 스텝), 또는 증류(1-4 스텝)를 사용합니다. 12강을 참고하세요.

## 사용하기

| 역할 | 2026년 전형적인 스택 |
|------|-----------------------|
| 이미지 픽셀 공간 확산 (소형, 장난감) | DDPM + U-Net |
| 이미지 잠재(latent) 확산 | VAE 인코더 + U-Net 또는 DiT (07강) |
| 비디오 잠재(latent) 확산 | 시공간 DiT (Sora, Veo, WAN) |
| 오디오 잠재(latent) 확산 | Encodec + 확산 트랜스포머 |
| 과학 (분자, 단백질, 물리) | 등변 확산(Equivariant diffusion) (EDM, RFdiffusion, AlphaFold3) |

확산(Diffusion)은 범용 생성 백본입니다. 흐름 매칭(Flow matching) (13강)은 2024-2026년 경쟁자로, 동일한 품질에서 추론 속도가 보통 더 빠릅니다.

## 출시하기

`outputs/skill-diffusion-trainer.md`을 저장하세요. 스킬은 데이터셋과 연산 예산을 입력으로 받아 다음을 출력합니다: 스케줄(linear/cosine/sigmoid), 예측 대상(ε/v/x), 스텝 수, 가이드 스케일, 샘플러 계열, 평가 프로토콜.

## 연습 문제

1. **쉬움.** `code/main.py`에서 T를 40에서 10으로 변경하세요. 샘플 품질(출력의 시각적 히스토그램)은 어떻게 저하됩니까? T가 몇일 때 두 모드(two-mode) 구조가 붕괴됩니까?
2. **중간.** ε-예측에서 v-예측으로 전환하세요. 역방향 스텝을 다시 유도하고 최종 샘플 품질을 비교하세요.
3. **난이도: 상.** 분류기 없는 가이드(classifier-free guidance)를 추가하세요. 클래스 레이블 `c ∈ {0, 1}`에 조건을 걸고, 학습 중 10%의 확률로 이를 드롭(drop)하며, 샘플링 시 `ε = (1+w)·ε_cond - w·ε_uncond`을 사용하세요. `w = 0, 1, 3, 7`에서 조건부 모드 적중률(conditional-mode-hit rate)을 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 순방향 과정 | "노이즈 추가" | 데이터를 파괴하는 고정 마르코프 체인 `q(x_t \| x_{t-1})`. |
| 역방향 과정 | "디노이징(Denoising)" | 데이터를 재구성하는 학습된 체인 `p_θ(x_{t-1} \| x_t)`. |
| β 스케줄 | "노이즈 사다리" | 단계별 분산; 선형, 코사인, 또는 시그모이드. |
| α̅ | "알파 바(Alpha bar)" | 누적 곱 `∏(1 - β)`; `x_0`로부터 `x_t`의 폐형(closed-form)을 제공합니다. |
| 단순 손실 | "노이즈에 대한 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`; 모든 변분 유도(variational derivations)는 이것으로 수렴합니다. |
| ε-예측 | "노이즈 예측" | 출력은 추가된 노이즈입니다; 표준 DDPM. |
| V-예측 | "속도 예측" | 출력은 `α·ε - σ·x`입니다; t에 걸쳐 더 나은 조건부 처리(Conditioning)를 제공합니다. |
| DDPM | "논문" | Ho et al. 2020; 선형 β, 1000 단계, U-Net. |
| DDIM | "결정적 샘플러" | 비마르코프(non-Markov) 샘플러, 20-50 단계, 동일한 학습 목표. |
| 분류기 없는 가이드(CFG) | "CFG" | 조건부 및 무조건부 노이즈 예측을 혼합하여 조건부 처리를 증폭합니다. |

## 프로덕션 노트: 확산(Diffusion) 추론은 단계 수 문제입니다

DDPM 논문은 T=1000의 역방향 단계를 실행합니다. 프로덕션에서 이를 배포하는 사람은 없습니다. 모든 실제 추론 스택은 세 가지 전략 중 하나를 선택합니다. 각 전략은 "레이턴시(latency)가 어디서 발생하고 있는가"라는 프로덕션 관점과 명확하게 매핑됩니다:

1. **더 빠른 샘플러, 동일한 모델.** DDIM (20-50 단계), DPM-Solver++ (10-20), UniPC (8-16). 역방향 루프의 드롭인(drop-in) 대체품입니다; 학습된 `ε_θ` 가중치는 변경되지 않습니다. 레이턴시를 20-50배 줄입니다.
2. **증류(Distillation).** 학생(student) 모델이 더 적은 단계로 교사(teacher) 모델과 일치하도록 학습합니다: 점진적 증류(Progressive Distillation) (2 → 1), 일관성 모델(Consistency Models) (임의 → 1-4), LCM, SDXL-Turbo, SD3-Turbo. 레이턴시를 추가로 5-10배 줄이며, 재학습이 필요합니다.
3. **캐싱 및 컴파일.** `torch.compile(unet, mode="reduce-overhead")`, TensorRT-LLM의 확산 백엔드, `xformers`/SDPA 어텐션, bf16 가중치. 단계별 레이턴시를 약 2배 줄입니다. (1) 및 (2)과 결합할 수 있습니다.

프로덕션 확산(diffusion) 서버의 예산 대화는 프로덕션 문헌이 LLM에 대해 설명하는 것과 동일합니다: 지연(latency)은 `num_steps × step_cost + VAE_decode`, 처리량(throughput)은 `batch_size × (num_steps × step_cost)^-1`입니다. TTFT는 작습니다(한 단계); TPOT에 해당하는 값은 전체 응답 시간입니다. 왜냐하면 이미지 생성은 사용자의 관점에서 "한 번에 전체" 생성되기 때문입니다.

## 추가 읽기

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 시대를 앞선 확산(diffusion) 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, 더 적은 단계.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 코사인 스케줄, 학습된 분산.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 분류기 가이드(guidance).
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 통합 표기법, 가장 깔끔한 레시피.
