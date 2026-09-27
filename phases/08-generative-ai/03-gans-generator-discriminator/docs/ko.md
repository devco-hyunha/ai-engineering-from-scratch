# GANs — 생성자 vs 판별자 (Generator vs Discriminator)

> 2014년 Goodfellow의 기발한 전략은 밀도(density) 추정을 완전히 건너뛴 것이었습니다. 두 개의 네트워크를 사용합니다. 하나는 가짜를 만들고, 다른 하나는 이를 잡아냅니다. 이들은 가짜가 진짜와 구별할 수 없을 때까지 서로 싸웁니다. 이론적으로는 작동하지 않아야 하며, 실제로도 그렇지 않은 경우가 많습니다. 하지만 성공했을 때, 특정 좁은 도메인에서 생성된 샘플은 기존 문헌 중 가장 선명한 품질을 보여줍니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 문제점 (The Problem)

VAE는 MSE 디코더 손실(loss)이 *평균(mean)* 이미지에 대해 베이즈 최적(Bayes-optimal)이기 때문에 흐릿한 샘플을 생성합니다. 여러 가능한 숫자들의 평균은 흐릿한 숫자가 될 수밖에 없습니다. 여러분은 특정 타겟과의 픽셀 단위 근접성이 아니라, *그럴듯함(plausibility)*에 보상을 주는 손실 함수를 원합니다. 하지만 그럴듯함에 대한 폐쇄형(closed-form) 공식은 존재하지 않습니다. 이를 직접 학습해야 합니다.

Goodfellow의 아이디어는 다음과 같습니다: 실제 이미지와 가짜 이미지를 구분하도록 분류기 `D(x)`를 학습시킵니다. 그리고 `D`를 속이도록 생성자 `G(z)`를 학습시킵니다. `G`를 위한 손실 신호는 현재 `D`가 무엇을 진짜처럼 보이게 만든다고 판단하는지에 따라 결정됩니다. 이 신호는 `G`가 개선됨에 따라 계속 변하며, 움직이는 타겟을 쫓게 됩니다. 두 네트워크가 모두 수렴하면, `G`는 `log p(x)`를 명시적으로 기록하지 않고도 데이터 분포를 학습하게 됩니다.

이것이 적대적 학습(adversarial training)입니다. 수학적으로는 다음과 같은 미니맥스 게임(minimax game)입니다:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026년 현재, GAN은 더 이상 SOTA(최첨단) 생성 모델은 아닙니다(그 왕좌는 확산 모델(diffusion)과 플로우 매칭(flow matching)이 차지했습니다). 하지만 StyleGAN 2/3는 여전히 지금까지 출시된 모델 중 가장 선명한 얼굴 모델로 남아 있으며, GAN 판별자(discriminator)는 확산 모델 학습 시 *지각 손실(perceptual losses)*로 사용됩니다. 또한 적대적 학습은 실시간 확산 모델을 구현할 수 있게 해주는 빠른 1단계 증류(fast 1-step distillations, 예: SDXL-Turbo, SD3-Turbo, LCM)의 핵심 동력입니다.

## 개념 (The Concept)

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**생성자 (Generator) `G(z)`.** 노이즈 벡터 `z ~ N(0, I)`를 샘플 `x̂`로 매핑합니다. 디코더 형태의 네트워크(dense 또는 transposed conv)를 사용합니다.

**판별자 (Discriminator) `D(x)`.** 샘플을 스칼라 확률(또는 점수)로 매핑합니다. 실제 데이터(Real) → 1, 가짜 데이터(Fake) → 0으로 분류합니다.

**손실 함수 (Loss).** 두 가지 업데이트가 번갈아 가며 수행됩니다:

- **`D` 학습:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. 실제=1, 가짜=0에 대한 이진 교차 엔트로피(Binary cross-entropy)를 사용합니다.
- **`G` 학습:** `loss_G = -log D(G(z))`. 이는 Goodfellow가 사용한 *비포화(non-saturating)* 형태입니다 (원래의 `log(1 - D(G(z)))`는 `D`가 확신을 가질 때 포화되어 그래디언트를 소멸시킵니다).

**학습 루프 (Training loop).** `D`를 한 단계 학습시킨 후, `G`를 한 단계 학습시킵니다. 이를 반복합니다.

**작동 원리 (Why it works).** 만약 `G`가 `p_data`와 완벽하게 일치한다면, `D`는 확률적 추측보다 더 나은 성능을 낼 수 없으므로 모든 곳에서 0.5를 출력하게 됩니다. 이때 `G`는 더 이상 그래디언트를 얻지 못하며, 평형(Equilibrium) 상태에 도달합니다.

**실패 원인 (Why it breaks).** 모드 붕괴(Mode collapse, `G`가 `D`가 분류할 수 없는 하나의 모드를 찾아내어 그것만 계속 생성하는 현상), 그래디언트 소실(Vanishing gradient, `D`가 너무 빨리 학습되어 `log D`가 포화되는 현상), 학습 불안정성(Training instability, 학습률, 배치 크기 등 다양한 요인에 의해 발생) 등이 있습니다.

## GAN을 작동하게 만든 변형 모델들 (Variants that made GANs work)

| 연도 | 혁신 기술 (Innovation) | 해결책 (Fix) |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — 최초의 안정적인 아키텍처. |
| 2017 | WGAN, WGAN-GP | BCE를 Wasserstein distance + gradient penalty로 교체. 기울기 소실(vanishing gradient) 문제 해결. |
| 2017 | Spectral normalization | 판별자(discriminator)에 Lipschitz-bound 적용. 2026년 판별자에서도 여전히 사용됨. |
| 2018 | Progressive GAN | 저해상도부터 먼저 학습한 후 레이어를 추가. 최초의 메가픽셀 결과물 도출. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. 고정된 도메인의 실사 이미지 생성 분야에서 SOTA(State of the art) 달성. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant — 2026년에도 여전히 얼굴 생성의 골드 표준. |
| 2022 | StyleGAN-XL | Conditional, class-aware, 더 큰 규모(larger scale). |
| 2024 | R3GAN | 더 강력한 정규화(regularization)로 재탄생; 별도의 트릭 없이 1024² 해상도 구현. |

```figure
gan-minimax
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 두 개의 가우시안 혼합(mixture of two Gaussians)인 1차원 데이터를 사용하여 아주 작은 GAN을 학습시킵니다. 생성자(Generator)와 판별자(Discriminator)는 단일 은닉층을 가진 MLP(다층 퍼셉트론)로 구성됩니다. 우리는 순전파(forward), 역전파(backward), 그리고 미니맥스(minimax) 루프를 직접 구현합니다. 이 과정의 목표는 두 가지 주요 실패 모드(모드 붕괴(mode collapse) + 기울기 소실(vanishing gradient))가 발생하는 과정을 직접 확인하는 것입니다.

### 1단계: 비포화 손실 (non-saturating loss)

기본적인 Goodfellow 손실 함수인 `log(1 - D(G(z)))`는 D가 G의 가짜 데이터를 높은 확신도로 가짜라고 분류할 때 0으로 수렴합니다. 이 시점에서 G를 위한 그래디언트(gradient)는 사실상 0이 되며, G는 더 이상 개선될 수 없습니다. 비포화 형태인 `-log D(G(z))`는 이와 반대되는 점근선을 가집니다. 즉, D가 확신을 가질 때 값이 급격히 커지며 G에게 강력한 신호를 제공합니다.

```python
def g_loss(d_fake):
    # log D(G(z))를 최대화 <=> -log D(G(z))를 최소화
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 2단계: 생성자(Generator) 단계당 1회의 판별자(Discriminator) 단계 수행

```python
for step in range(steps):
    # D 학습
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # G 학습
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 새로운 가짜 데이터 생성
    update_G(fake_batch)
```

G를 위해 새로운 가짜 데이터(Fresh fakes)를 생성해야 합니다. 그렇지 않으면 그래디언트가 오래된 값(stale)이 됩니다.

### 3단계: 모드 붕괴(mode collapse) 관찰하기

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

전형적인 증상은 두 개의 실제 모드 중 하나가 더 이상 생성되지 않는 것입니다. 판별자(discriminator)는 해당 모드가 가짜(fake)로 분류된 적이 없기 때문에 이를 교정하는 것을 중단하게 됩니다.

## 주의 사항 (Pitfalls)

- **판별자(Discriminator)가 너무 강력함.** 판별자의 학습률(learning rate)을 2~5배 낮추거나, 인스턴스/레이어 노이즈(instance/layer noise)를 추가해 보세요. 판별자의 정확도가 95%를 넘어가면 생성자(Generator)는 학습 동력을 잃습니다.
- **생성자가 하나의 모드(mode)를 암기함.** 판별자 입력에 노이즈를 추가하거나, 미니배치 판별자 레이어(minibatch-discriminator layer)를 사용하거나, WGAN-GP로 전환해 보세요.
- **배치 정규화(Batch norm)의 통계치 유출.** 실제 배치와 가짜 배치가 동일한 BN 레이어를 통과하면 통계치가 섞이게 됩니다. 대신 인스턴스 정규화(instance norm)나 스펙트럴 정규화(spectral norm)를 사용해 보세요.
- **인셉션 점수(Inception-score) 조작.** FID와 IS는 샘플 수가 적을 때 노이즈가 심합니다. 평가 시에는 10,000개 이상의 샘플을 사용해 보세요.
- **조건부 작업(conditional tasks)에서 원샷 샘플링(One-shot sampling)은 불가능에 가깝습니다.** 사용 가능한 결과물을 얻으려면 여전히 CFG 스케일, 트렁케이션 트릭(truncation tricks), 재샘플링(re-sampling)이 필요합니다.

## 활용하기 (Use It)

2026년 GAN 스택:

| 상황 | 선택 (Pick) |
|-----------|------|
| 실사 같은 사람 얼굴, 고정된 포즈 | StyleGAN3 (가장 선명하고 크기가 작음) |
| 애니메이션 / 스타일화된 얼굴 | StyleGAN-XL 또는 Stable Diffusion LoRA |
| 이미지 대 이미지 변환 (Image-to-image translation) | Pix2Pix / CycleGAN (Phase 8 · 04) 또는 ControlNet (Phase 8 · 08) |
| 빠른 1단계 텍스트-이미지 생성 (Fast 1-step text-to-image) | 확산 모델의 적대적 증류 (Adversarial distillation of diffusion) (SDXL-Turbo, SD3-Turbo) |
| 확산 모델 학습 중 지각 손실 (Perceptual loss) 적용 | 이미지 크롭(crop)에 대한 작은 GAN 판별자(discriminator) 사용 |
| 멀티모달, 개방형(open-ended) 모든 작업 | 사용하지 마세요 — 확산 모델(diffusion) 또는 플로우 매칭(flow matching)을 사용하세요 |

GAN은 선명하지만 범위가 좁습니다. 사진, 임의의 텍스트 프롬프트, 비디오와 같이 도메인이 확장되면 확산 모델(diffusion)로 전환하세요. 적대적 기법(adversarial trick)은 독립적인 생성자가 아닌, 하나의 구성 요소(지각 손실, 증류 등)로서 계속 활용됩니다.

## Ship It (실행하기)

`outputs/skill-gan-debugger.md`를 저장하세요. 이 스킬은 실패한 GAN 실행 결과(손실 곡선, 샘플 그리드, 데이터셋 크기)를 입력받아, 발생 가능한 원인에 대한 순위 목록, 한 줄 해결책, 그리고 재실행 프로토콜을 출력합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 기본 설정으로 `code/main.py`를 실행해 보세요. 그 다음 `D_LR = 5 * G_LR`로 설정하고 다시 실행해 보세요. G의 손실(loss)이 얼마나 빨리 상수로 수렴(collapse)하나요?
2. **중간 (Medium).** Goodfellow의 BCE 손실을 WGAN 손실로 교체해 보세요: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`로 설정하고, D의 가중치를 `[-0.01, 0.01]` 범위로 클리핑(clip)하세요. 학습이 더 안정적인가요? 실제 소요 시간(wall-clock) 기준 수렴 속도를 비교해 보세요.
3. **어려움 (Hard).** 1-D 예제를 2-D 데이터(링 형태의 8개 가우시안 혼합 모델)로 확장해 보세요. 1k, 5k, 10k 단계에서 생성자(generator)가 8개의 모드(mode) 중 몇 개를 포착하는지 추적해 보세요. 미니배치 판별(minibatch discrimination)을 구현하고 다시 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Generator (생성자) | "G" | 노이즈를 샘플로 변환하는 네트워크, `G: z → x̂`. |
| Discriminator (판별자) | "D" | 진짜와 가짜를 구분하는 분류기 `D: x → [0, 1]`. |
| Minimax (미니맥스) | "게임(The game)" | 공동 목적 함수에 대한 `min_G max_D`. |
| Non-saturating loss (비포화 손실) | "해결책(The fix)" | G를 위해 `log(1 - D(G(z)))` 대신 `-log D(G(z))`를 사용함. |
| Mode collapse (모드 붕괴) | "G가 한 가지만 외웠어" | 데이터가 다양함에도 불구하고 생성자가 소수의 차별화되지 않은 출력물만 생성함. |
| WGAN | "Wasserstein" | BCE를 Earth-Mover 거리 + gradient penalty로 교체; 더 부드러운 그래디언트 제공. |
| Spectral norm (스펙트럼 정규화) | "Lipschitz 트릭" | D의 기울기를 제한하기 위해 가중치 노름(weight norms)을 제약함; 학습을 안정화함. |
| StyleGAN | "잘 작동하는 모델" | Mapping network + AdaIN; 2026년 현재까지도 얼굴 생성 분야에서 최고 수준임. |

## 프로덕션 노트: 원샷 추론(one-shot inference)은 GAN의 지속적인 강점입니다

GAN은 더 이상 오픈 도메인 생성의 샘플 품질 측면에서 승리하지 못하지만, 추론 비용(inference cost) 측면에서는 여전히 우위를 점하고 있습니다. 프로덕션 추론 문헌의 관점에서 GAN은 다음과 같은 특징을 가집니다:

- **Pre-fill 및 decode 단계가 없음.** 단 한 번의 `G(z)` 순전파(forward pass)로 완료됩니다. TTFT(첫 토큰 생성 시간) ≈ 전체 지연 시간(total latency)입니다.
- **KV-cache 압박이 없음.** 유일한 상태(state)는 가중치(weights)뿐입니다. 배치 크기(batch size)는 캐시가 아닌 활성화 메모리(activation memory)에 의해 제한됩니다.
- **사소한 수준의 연속 배치(continuous batching).** 모든 요청이 동일한 고정 FLOPs를 소모하므로, 서버의 목표 점유율에 맞춘 정적 배치(static batch)가 대개 최적입니다. 별도의 인플라이트 스케줄러(in-flight scheduler)가 필요하지 않습니다.

이것이 2026년 빠른 텍스트-이미지 생성을 위한 지배적인 기술이 GAN 증류(GAN distillation; SDXL-Turbo, SD3-Turbo, ADD, LCM)인 이유입니다. 이 기술은 확산 모델(diffusion) 기반의 분포를 유지하면서, 20~50단계의 확산 파이프라인을 1~4단계의 GAN 스타일 순전파로 압축합니다. 적대적 손실(adversarial loss)은 느린 생성자를 빠른 생성자로 전환하기 위한 학습 단계의 조절 장치(knob)로서 여전히 유효합니다.

## 추가 읽을거리 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN의 원본 논문입니다.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 최초의 안정적인 아키텍처입니다.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN입니다.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN입니다.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2입니다.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3입니다.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo입니다.
