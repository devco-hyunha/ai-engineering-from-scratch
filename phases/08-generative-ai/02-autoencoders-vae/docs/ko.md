# 오토인코더 및 변이형 오토인코더 (Autoencoders & Variational Autoencoders (VAE))

> 일반적인 오토인코더는 데이터를 압축한 뒤 재구성합니다. 즉, 데이터를 암기할 뿐 생성하지는 못합니다. 여기에 한 가지 트릭, 즉 코드가 가우시안(Gaussian) 분포를 따르도록 강제하는 기법을 추가하면 샘플러(sampler)가 됩니다. `z = μ + σ·ε`라는 재매개변수화(reparameterization)라는 단 하나의 트릭 덕분에, 여러분이 2026년에 사용하는 모든 잠재 확산(latent-diffusion) 및 플로우 매칭(flow-matching) 이미지 모델의 입력단에는 VAE가 포함되어 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## 문제 (The Problem)

784픽셀의 MNIST 숫자를 16개의 숫자로 이루어진 코드로 압축한 뒤, 이를 다시 복원해 보세요. 일반적인 오토인코더(Autoencoder)는 복원 MSE(평균 제곱 오차) 측면에서는 뛰어난 성능을 보이겠지만, 코드 공간(code space)은 울퉁불퉁하고 무질서한 상태가 됩니다. 코드 공간에서 임의의 점을 선택해 디코딩하면 노이즈가 생성될 뿐입니다. 즉, 샘플러(sampler)가 없는 상태이며, 이는 그저 압축 모델을 겉모습만 바꾼 것에 불과합니다.

여러분이 실제로 원하는 것은 다음과 같습니다: (a) 코드 공간이 샘플링이 가능한 깨끗하고 매끄러운 분포(예: 등방성 가우시안 `N(0, I)`)를 갖는 것, (b) 어떤 샘플을 디코딩하더라도 그럴듯한 숫자가 생성되는 것, (c) 인코더와 디코더가 여전히 압축을 잘 수행하는 것입니다. 세 가지 목표를 하나의 아키텍처와 하나의 손실 함수로 달성해야 합니다.

Kingma의 2013년 VAE는 인코더가 *분포* `q(z|x) = N(μ(x), σ(x)²)`를 출력하도록 학습시키고, KL 페널티(KL penalty)를 통해 해당 분포를 사전 분포(prior)인 `N(0, I)` 쪽으로 끌어당기며, 디코딩하기 전에 `q(z|x)`로부터 `z`를 샘플링함으로써 이 문제를 해결합니다. 추론(inference) 시에는 인코더를 제외하고, `z ~ N(0, I)`에서 샘플링한 뒤 디코딩하면 됩니다. 코드 공간에 구조를 부여하는 핵심은 바로 KL 페널티입니다.

2026년 현재, VAE는 순수 이미지 품질 면에서 디퓨전(diffusion) 모델에 밀려 단독으로 쓰이는 경우는 드뭅니다. 하지만 모든 잠재 디퓨전 모델(Latent-Diffusion Model; SD 1/2/XL/3, Flux, AudioCraft 등)에서 인코더로 선택되는 핵심 기술입니다. VAE를 배우는 것은 여러분이 사용하는 모든 이미지 파이프라인의 보이지 않는 첫 번째 레이어를 배우는 것과 같습니다.

## 개념 (The Concept)

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**오토인코더(Autoencoder).** `z = encoder(x)`, `x̂ = decoder(z)`, 손실(loss) = `||x - x̂||²`. 코드 공간(code space)이 구조화되어 있지 않습니다.

**VAE 인코더(VAE encoder).** 두 개의 벡터인 `μ(x)`와 `log σ²(x)`를 출력합니다. 이들은 `q(z|x) = N(μ, diag(σ²))`를 정의합니다.

**재매개변수화 트릭(Reparameterization trick).** `q(z|x)`로부터 샘플링하는 것은 미분 가능하지 않습니다. 샘플을 `z = μ + σ·ε` (단, `ε ~ N(0, I)`)로 다시 작성합니다. 이제 `z`는 `(μ, σ)`의 결정론적 함수에 매개변수가 없는 노이즈가 더해진 형태가 되어, `μ`와 `σ`를 통해 그래디언트(gradient)가 흐를 수 있습니다.

**손실 함수(Loss).** 증거 하한(Evidence Lower Bound, ELBO)은 다음 두 항으로 구성됩니다:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

재구성(Reconstruction) 항은 `x̂`가 `x`에 가까워지도록 유도합니다. KL 항은 `q(z|x)`를 사전 분포(prior)에 가깝게 밀어붙입니다. 이 둘은 트레이드오프 관계에 있습니다. 작은 $\beta$ (<1)는 더 선명한 샘플을 생성하지만 코드 공간이 덜 가우시안(Gaussian) 형태를 띱니다. 큰 $\beta$ (>1)는 더 깨끗한 코드 공간을 만들지만 샘플이 흐릿해집니다. $\beta$-VAE (Higgins 2017)는 이 조절 노브(knob)를 유명하게 만들었으며, 얽힘 해제(disentanglement) 연구의 시발점이 되었습니다.

**샘플링(Sampling).** 추론(inference) 시에는 `z ~ N(0, I)`를 추출한 뒤 디코더를 통해 순전파(forward)합니다. 디퓨전(diffusion)과 같은 반복적인 샘플링 없이 단 한 번의 순전파로 완료됩니다.

```figure
vae-latent-grid
```

## 구현하기 (Build It)

`code/main.py`는 `numpy`나 `torch` 없이 구현된 아주 작은 VAE(Variational Autoencoder)를 포함하고 있습니다. 입력 데이터는 8차원 가우시안 혼합 모델(Gaussian mixture)에서 추출된 8차원 합성 데이터입니다. 인코더(Encoder)와 디코더(Decoder)는 단일 은닉층을 가진 MLP(Multi-Layer Perceptron)로 구성됩니다. `tanh` 활성화 함수, 순전파(forward pass), 손실 함수(loss), 그리고 직접 작성한 역전파(backward pass)를 구현했습니다. 실제 서비스용이 아닌 교육용 목적입니다.

### 1단계: 인코더 순전파 (Encoder forward)

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`σ` 대신 `log σ²`를 사용하는 이유는 네트워크 출력이 제약되지 않도록 하기 위함입니다 (`σ`에 `softplus`를 적용하는 것은 함정입니다. `σ ≈ 0`에서 그래디언트가 소실됩니다).

### 2단계: 재매개변수화 및 디코딩 (Reparameterize and decode)

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 3단계: ELBO (Evidence Lower Bound)

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

두 분포가 모두 가우시안(Gaussian)이므로 정확한 폐쇄형(closed-form) KL을 사용합니다. 수치적으로 적분하지 마세요. 2026년에도 여전히 몬테카를로(Monte-Carlo) KL 추정치를 사용하여 코드를 배포하는 사람들이 있는데, 이는 아무런 이유 없이 3배 더 느립니다.

### 4단계: 생성 (Generate)

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

이것이 생성 모델(generative model)입니다. 단 다섯 줄입니다.

## 주의 사항 (Pitfalls)

- **사후 확률 붕괴 (Posterior collapse).** KL 항이 `q(z|x) → N(0, I)`가 되도록 너무 공격적으로 유도하여, `z`가 `x`에 대한 정보를 전혀 담지 못하게 되는 현상입니다. 해결책: $\beta$-어닐링($\beta=0$에서 시작하여 1까지 점진적으로 증가), 프리 비트(free bits), 또는 비활성 차원의 KL 계산 생략 등을 고려해 보세요.
- **흐릿한 샘플 (Blurry samples).** 가우시안 디코더 가능도(likelihood)는 MSE 재구성을 의미하며, 이는 L2(평균)에 대해 베이즈 최적(Bayes-optimal)입니다. 즉, 가능한 여러 숫자들의 평균값은 흐릿한 숫자가 됩니다. 해결책: 이산형 디코더(VQ-VAE, NVAE)를 사용하거나, VAE를 인코더로만 사용하고 잠재 공간(latents) 위에 디퓨전(diffusion)을 쌓아 보세요(이것이 Stable Diffusion이 작동하는 방식입니다).
- **$\beta$ 값이 너무 크거나 너무 일찍 적용됨.** 사후 확률 붕괴 항목을 참조하세요. $\beta \approx 0.01$에서 시작하여 점진적으로 높여 보세요.
- **잠재 차원(Latent dim)이 너무 작음.** MNIST에는 16-D, ImageNet 256²에는 256-D, ImageNet 1024²에는 2048-D가 적당합니다. Stable Diffusion의 VAE는 512×512×3을 64×64×4로 압축합니다(공간 면적에서 32배, 채널에서 32배 다운샘플링).

## 활용하기 (Use It)

2026년 VAE 스택:

| 상황 | 선택 (Pick) |
|-----------|------|
| 확산 모델용 이미지 잠재 인코더 (Image-latent encoder for diffusion) | Stable Diffusion VAE (`sd-vae-ft-ema`) 또는 Flux VAE |
| 오디오 잠재 인코더 (Audio-latent encoder) | Encodec (Meta), SoundStream, 또는 DAC (Descript) |
| 비디오 잠재 변수 (Video latents) | Sora의 시공간 패치 (spatiotemporal patches), Latte VAE, WAN VAE |
| 얽힘 해제 표현 학습 (Disentangled representation learning) | $\beta$-VAE, FactorVAE, TCVAE |
| 이산 잠재 변수 (Discrete latents, 트랜스포머 모델링용) | VQ-VAE, RVQ (ResidualVQ) |
| 생성을 위한 연속 잠재 변수 (Continuous latents for generation) | 일반 VAE를 사용한 후, 해당 잠재 공간 내에서 flow/diffusion 모델에 조건 부여 |

잠재 확산 모델(Latent-diffusion model)은 인코더와 디코더 사이에 확산 모델이 존재하는 VAE입니다. VAE가 거친 압축(coarse compression)을 수행하면, 확산 모델이 핵심적인 작업(heavy lifting)을 수행합니다. 이는 비디오(VAE + 비디오 확산 DiT)와 오디오(Encodec + MusicGen transformer)에서도 동일한 패턴으로 적용됩니다.

## Ship It (실행하기)

`outputs/skill-vae-trainer.md`를 저장하세요.

이 기술(Skill)은 데이터셋 프로필(dataset profile) + 잠재 차원 목표(latent-dim target) + 다운스트림 용도(downstream use: 재구성, 샘플링 또는 잠재 확산 모델 입력)를 입력받아 다음을 출력합니다: 아키텍처 선택(plain/β/VQ/RVQ), $\beta$ 스케줄, 잠재 차원, 디코더 가능도(Gaussian vs categorical), 그리고 평가 계획(재구성 MSE, 차원당 KL, `q(z|x)`와 `N(0, I)` 사이의 Fréchet distance).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 있는 `β` 값을 `0.01`, `0.1`, `1.0`, `5.0`으로 변경해 보세요. 최종 재구성 MSE(reconstruction MSE)와 KL 값을 기록하세요. 여러분의 합성 데이터(synthetic data)에 대해 어떤 `β` 값이 파레토 최적(Pareto-best)인가요?
2. **중간 (Medium).** 가우시안 디코더 가능도(Gaussian decoder likelihood)를 베르누이 가능도(Bernoulli likelihood, 교차 엔트로피 손실)로 교체해 보세요. 동일한 합성 데이터를 이진화(binarized)한 버전에서 샘플 품질을 비교해 보세요.
3. **어려움 (Hard).** `code/main.py`를 미니 VQ-VAE로 확장해 보세요. 연속적인 `z`를 $K=32$개의 항목을 가진 코드북(codebook) 내의 최근접 이웃 조회(nearest-neighbour lookup)로 교체합니다. 재구성 MSE를 비교하고, 얼마나 많은 코드북 항목이 사용되는지 보고하세요 (코드북 붕괴(codebook collapse) 현상이 실제로 발생할 수 있습니다).

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Autoencoder | 인코더-디코더 네트워크 | `x → z → x̂`, MSE를 학습함. 생성 모델이 아님. |
| VAE | 샘플러가 있는 AE | 인코더가 분포를 출력하며, KL 페널티가 코드 공간을 형성함. |
| ELBO | 증거 하한 (Evidence lower bound) | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; `q = p(z\|x)`일 때 값이 조밀해짐. |
| Reparameterization | `z = μ + σ·ε` | 확률적 노드를 결정론적 요소 + 순수 노이즈로 재작성함. 샘플링을 통한 역전파를 가능하게 함. |
| Prior | `p(z)` | 잠재 변수(latent)를 위한 목표 분포로, 일반적으로 `N(0, I)`임. |
| Posterior collapse | "KL 항이 이김" | 인코더가 `x`를 무시하고 사전 분포(prior)를 출력함; 디코더는 환각(hallucinate)을 일으켜야 함. |
| $\beta$-VAE | 조절 가능한 KL 가중치 | `loss = recon + β·KL`. $\beta$가 높을수록 더 잘 분리(disentangled)되지만 이미지는 흐릿해짐. |
| VQ-VAE | 이산 잠재 변수 (Discrete latent) | 연속적인 `z`를 가장 가까운 코드북 벡터로 대체함; 트랜스포머 모델링을 가능하게 함. |

## 프로덕션 노트: VAE는 디퓨전 서버에서 가장 부하가 큰 경로입니다 (The VAE is the hottest path in a diffusion server)

Stable Diffusion / Flux / SD3 파이프라인에서 VAE는 요청당 두 번 호출됩니다. 한 번은 인코딩을 위해(img2img / inpainting 수행 시), 다른 한 번은 디코딩을 위해 호출됩니다. 1024² 해상도에서 디코더 패스는 `128×128×16` 크기의 잠재 변수(latents)를 `1024×1024×3`으로 업샘플링하기 때문에, 전체 파이프라인에서 단일 활성화 메모리(activation-memory) 피크가 발생하는 가장 큰 지점인 경우가 많습니다. 이로 인해 발생하는 두 가지 실질적인 결과는 다음과 같습니다:

- **디코딩 시 슬라이스(Slice) 또는 타일(Tile) 처리.** `diffusers`는 `pipe.vae.enable_slicing()`과 `pipe.vae.enable_tiling()`을 제공합니다. 타일링(Tiling)은 미세한 경계선 아티팩트(seam artifact)를 대가로, 메모리 사용량을 `O(H·W)`에서 `O(tile²)`로 줄여줍니다. 소비자용 GPU에서 1024²+ 해상도를 다룰 때 필수적입니다.
- **bf16 디코더 및 최종 리사이즈를 위한 fp32 수치 연산.** SD 1.x VAE는 fp32로 출시되었으며, 1024²+ 해상도에서 fp16으로 캐스팅할 경우 *조용히 NaN(Not a Number)을 생성*합니다. SDXL은 `madebyollin/sdxl-vae-fp16-fix`를 포함하고 있습니다. 항상 fp16-fix 변형을 선호하거나 `bf16`을 사용하세요.

## 추가 읽을거리 (Further Reading)

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문입니다.
- [Higgins et al. (2017). $\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 얽힘 해제(disentangled) $\beta$-VAE에 관한 연구입니다.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE 논문입니다.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 최첨단(SOTA) 이미지 VAE 연구입니다.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; 인코더로서의 VAE를 다룹니다.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 오디오 VAE의 표준인 Encodec에 관한 연구입니다.
