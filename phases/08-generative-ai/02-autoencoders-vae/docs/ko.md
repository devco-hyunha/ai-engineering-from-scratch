# 오토인코더 및 변분 오토인코더 (VAE)

> 일반 오토인코더는 압축한 후 재구성합니다. 이는 기억하는 모델이며, 생성하지는 않습니다. 하나의 트릭을 추가해 보세요. 코드가 가우시안처럼 보이도록 강제하는 것입니다. 그러면 샘플러가 됩니다. `z = μ + σ·ε`의 재파라미터화라는 이 단일 트릭 때문에, 2026년에 사용하는 모든 잠재 확산(latent-diffusion) 및 흐름 매칭(flow-matching) 이미지 모델의 입력에 VAE가 존재합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 02강 (역전파), 3단계 · 07강 (CNN), 8단계 · 01강 (분류 체계)
**시간:** 약 75분

## 문제점

784픽셀 MNIST 숫자를 16개의 숫자로 구성된 코드로 압축한 후 재구성해 보세요. 일반 오토인코더는 재구성 MSE를 잘 수행하지만, 코드 공간은 울퉁불퉁한 상태입니다. 코드 공간에서 랜덤한 점을 선택해 디코딩하면 잡음이 생성됩니다. 샘플러가 없습니다. 이는 압축 모델에 불과합니다.

실제로 원하는 것은 다음과 같습니다: (a) 코드 공간이 샘플링할 수 있는 깨끗하고 매끄러운 분포, 예를 들어 등방성 가우시안 `N(0, I)`, (b) 임의의 샘플을 디코딩하면 그럴듯한 숫자가 생성됨, (c) 인코더와 디코더가 여전히 잘 압축함. 세 가지 목표, 하나의 아키텍처, 하나의 손실 함수입니다.

Kingma의 2013년 VAE는 인코더가 *분포* `q(z|x) = N(μ(x), σ(x)²)`를 출력하도록 학습하고, KL 페널티를 통해 해당 분포를 사전 분포 `N(0, I)` 쪽으로 끌어당기며, 디코딩 전에 `q(z|x)`에서 `z`를 샘플링하는 방식으로 이를 해결합니다. 추론 시에는 인코더를 제거하고 `z ~ N(0, I)`를 샘플링한 후 디코딩합니다. KL 페널티가 코드 공간을 구조화하도록 강제합니다.

2026년에는 VAE가 단독으로 출시되는 경우가 거의 없습니다. 원시 이미지 품질 측면에서는 확산 모델에 밀렸기 때문입니다. 하지만 모든 잠재 확산 모델(SD 1/2/XL/3, Flux, AudioCraft)의 인코더로 선택됩니다. VAE를 학습하면 사용하는 모든 이미지 파이프라인의 보이지 않는 첫 번째 레이어를 배우게 됩니다.

## 개념

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**오토인코더.** `z = encoder(x)`, `x̂ = decoder(z)`, 손실 = `||x - x̂||²`. 코드 공간은 구조화되지 않았습니다.

**VAE 인코더.** 두 개의 벡터를 출력합니다: `μ(x)` 및 `log σ²(x)`. 이 벡터들이 `q(z|x) = N(μ, diag(σ²))`를 정의합니다.

**재파라미터화 트릭(reparameterization trick).** `q(z|x)`에서 샘플링하는 것은 미분 불가능합니다. 샘플을 `z = μ + σ·ε`로 다시 쓰세요. 여기서 `ε ~ N(0, I)`입니다. 이제 `z`은 `(μ, σ)`와 매개변수가 아닌 잡음의 결정론적 함수입니다. 기울기는 `μ`와 `σ`를 통해 흐릅니다.

**손실(Loss).** 증거 하한(Evidence Lower BOund, ELBO), 두 항:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

재구성(reconstruction)은 `x̂`을 `x` 쪽으로 밀어냅니다. KL은 `q(z|x)`을 사전 분포(prior) 쪽으로 밀어냅니다. 두 항은 상충(trade-off)합니다. 작은 β (<1)는 더 선명한 샘플을 만들고, 코드 공간은 가우시안 분포에서 더 멀어집니다. 큰 β (>1)는 더 깨끗한 코드 공간을 만들고, 샘플은 더 흐려집니다. β-VAE (Higgins 2017)는 이 조절 변수(knob)를 유명하게 만들었고, 분해(disentanglement) 연구를 촉발했습니다.

**샘플링(Sampling).** 추론 시: `z ~ N(0, I)`을 뽑고, 디코더를 통해 순전파(forward)합니다. 한 번의 순전파로 끝납니다. 확산 모델(diffusion)처럼 반복적인 샘플링이 필요하지 않습니다.

```figure
vae-latent-grid
```

## 구현하기

`code/main.py`은 numpy나 torch 없이 작은 VAE를 구현합니다. 입력은 8-D 공간의 2개 성분 가우시안 혼합에서 뽑은 8차원 합성 데이터입니다. 인코더와 디코더는 단일 은닉층 MLP입니다. tanh 활성화 함수, 순전파, 손실, 그리고 수동으로 작성한 역전파를 구현합니다. 프로덕션용이 아니라 교육용입니다.

### 1단계: 인코더 순전파

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`σ` 대신 `log σ²`를 사용하세요. 이렇게 하면 네트워크 출력이 제약되지 않습니다. (σ의 softplus는 함정입니다. σ ≈ 0에서 기울기가 사라집니다.)

### 2단계: 재파라미터화 및 디코딩

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 3단계: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

두 분포 모두 가우시안 분포이므로 KL은 정확한 폐형(closed-form)으로 계산됩니다. 수치적으로 적분하지 마세요. 2026년에도 Monte-Carlo KL 추정치를 쓰는 코드를 배포하는 사람들이 있습니다. 이유 없이 3배 더 느려집니다.

### 4단계: 생성

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

이것이 생성 모델입니다. 5줄입니다.

## 문제점

- **사후 분포 붕괴(Posterior collapse).** KL 항이 `q(z|x) → N(0, I)`을 너무 강하게 밀어내서 `z`이 `x`에 대한 정보를 담지 못합니다. 해결책: β-어닐링(β=0에서 시작해 1로 올림), free bits, 또는 비활성 차원에서는 KL을 건너뛰는 것.
- **흐린 샘플(Blurry samples).** 가우시안 디코더의 우도는 MSE 재구성을 의미하며, 이는 L2에 대해 Bayes-최적(평균)입니다. 일련의 가능한 숫자의 평균은 흐릿한 숫자입니다. 해결책: 이산 디코더(VQ-VAE, NVAE)를 사용하거나, VAE를 인코더로만 쓰고 잠재 공간(latents) 위에 확산 모델을 쌓는 것(Stable Diffusion이 하는 방식).
- **β가 너무 크고, 너무 일찍 적용.** 사후 붕괴(posterior collapse)를 참고하세요. β≈0.01에서 시작하여 점진적으로 높여 보세요.
- **잠재 차원(latent dim)이 너무 작음.** MNIST에는 16-D가 작동하고, ImageNet 256²에는 256-D, ImageNet 1024²에는 2048-D가 작동합니다. Stable Diffusion의 VAE는 512×512×3 → 64×64×4로 압축합니다 (공간 영역에서 32배 다운샘플링, 채널에서 32배).

## 사용하기

2026년 VAE 스택:

| 상황 | 선택 |
|-----------|------|
| 확산(diffusion)용 이미지 잠재 인코더 | Stable Diffusion VAE (`sd-vae-ft-ema`) 또는 Flux VAE |
| 오디오 잠재 인코더 | Encodec (Meta), SoundStream, 또는 DAC (Descript) |
| 비디오 잠재(latents) | Sora의 시공간 패치, Latte VAE, WAN VAE |
| 분리(disentangled) 표현 학습 | β-VAE, FactorVAE, TCVAE |
| 이산(latent) 잠재 (트랜스포머 모델링용) | VQ-VAE, RVQ (ResidualVQ) |
| 생성용 연속 잠재 | Plain VAE, 그 후 해당 잠재 공간에서 flow/diffusion 모델에 조건을 부여 |

잠재 확산(latent-diffusion) 모델은 인코더와 디코더 사이에 확산 모델이 존재하는 VAE입니다. VAE는 거친 압축을 수행하고, 확산 모델이 주요 작업을 수행합니다. 비디오(VAE + 비디오 확산 DiT)와 오디오(Encodec + MusicGen 트랜스포머)에서도 동일한 패턴이 적용됩니다.

## 출시하기

`outputs/skill-vae-trainer.md`을 저장하세요.

스킬은 데이터셋 프로필 + 잠재 차원 목표 + 다운스트림 사용 목적(재구성, 샘플링, 또는 잠재 확산 입력)을 받아 아키텍처 선택(plain/β/VQ/RVQ), β 스케줄, 잠재 차원, 디코더 가능성(Gaussian vs categorical), 그리고 평가 계획(재구성 MSE, 차원별 KL, `q(z|x)`과 `N(0, I)` 간의 Fréchet 거리)을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 `β`을 `0.01`, `0.1`, `1.0`, `5.0`로 변경하세요. 최종 재구성 MSE와 KL을 기록하세요. 합성 데이터에 대해 어떤 β가 파레토 최적(Pareto-best)인가요?
2. **중간.** Gaussian 디코더 가능성을 Bernoulli 가능성(교차 엔트로피 손실)으로 교체하세요. 동일한 합성 데이터의 이진화(binarized) 버전에서 샘플 품질을 비교하세요.
3. **난이도: 상.** `code/main.py`을 미니 VQ-VAE로 확장하세요: 연속 `z`를 K=32 항목의 코드북에서 최근접 이웃 조회로 교체하세요. 재구성 MSE를 비교하고 코드북 항목이 몇 개나 사용되었는지 보고하세요 (코드북 붕괴는 실제로 발생합니다).

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 오토인코더 | 인코딩-디코딩 네트워크 | `x → z → x̂`, MSE를 학습하세요. 생성형이 아닙니다. |
| VAE | 샘플러가 있는 AE | 인코더가 분포를 출력하며, KL 페널티가 코드 공간을 형성합니다. |
| ELBO | 증거 하한 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; `q = p(z\|x)`일 때 밀접합니다. |
| 재파라미터화 | `z = μ + σ·ε` | 확률적 노드를 결정론적 + 순수 잡음으로 재작성합니다. 샘플링을 통한 역전파를 가능하게 합니다. |
| 사전 분포 | `p(z)` | 잠재 변수의 목표 분포로, 일반적으로 `N(0, I)`입니다. |
| 사후 붕괴 | "KL 항이 이김" | 인코더가 `x`를 무시하고 사전 분포를 출력합니다; 디코더가 환각을 생성해야 합니다. |
| β-VAE | 조정 가능한 KL 가중치 | `loss = recon + β·KL`. 더 높은 β = 더 분해되지만 더 흐려집니다. |
| VQ-VAE | 이산 잠재 변수 | 연속 `z`를 최근접 코드북 벡터로 교체합니다; 트랜스포머 모델링을 가능하게 합니다. |

## 프로덕션 노트: VAE는 확산 서버에서 가장 뜨거운 경로입니다

Stable Diffusion / Flux / SD3 파이프라인에서 VAE는 요청당 두 번 호출됩니다 — img2img / 인페인팅을 수행하는 경우 인코딩 한 번과 디코딩 한 번. 1024²에서 디코더 패스는 전체 파이프라인에서 단일 최대 활성화 메모리 피크인 경우가 많으며, `128×128×16` 잠재 변수를 `1024×1024×3`로 업샘플링하기 때문입니다. 두 가지 실질적인 결과:

- **디코딩을 슬라이스하거나 타일링하세요.** `diffusers`은 `pipe.vae.enable_slicing()`과 `pipe.vae.enable_tiling()`을 노출합니다. 타일링은 작은 이음새 아티팩트를 `O(H·W)` 대신 `O(tile²)` 메모리로 교환합니다. 소비자 GPU에서 1024²+에 필수적입니다.
- **디코더는 bf16, 최종 리사이즈는 fp32 수치 연산.** SD 1.x VAE는 fp32로 릴리스되었으며, 1024²+에서 fp16으로 캐스트할 때 *조용히 NaN을 생성합니다*. SDXL은 `madebyollin/sdxl-vae-fp16-fix`을 제공합니다 — 항상 fp16 수정 변형을 선호하거나 bf16을 사용하세요.

## 추가 읽기

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 분해된 β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 최신 이미지 VAE입니다.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; 인코더로 VAE를 사용합니다.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 오디오 VAE 표준인 Encodec입니다.
