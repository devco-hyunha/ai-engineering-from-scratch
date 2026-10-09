# GAN — 생성자 vs 판별자

> 2014년 Goodfellow의 비법은 밀도를 완전히 생략하는 것이었습니다. 두 개의 네트워크. 하나는 가짜를 만들고, 하나는 가짜를 잡아냅니다. 가짜가 진짜와 구별할 수 없을 때까지 싸웁니다. 이론적으로는 작동하지 않아야 합니다. 실제로도 자주 작동하지 않습니다. 작동할 때, 생성된 샘플은 좁은 영역에서 문헌상 가장 선명한 품질을 유지합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 02강 (역전파), 3단계 · 08강 (옵티마이저), 8단계 · 02강 (변분 오토인코더 (VAE))
**시간:** 약 75분

## 문제점

VAE는 MSE 디코더 손실이 *평균* 이미지에 대해 베이즈 최적이기 때문에 흐릿한 샘플을 생성합니다. 많은 가능한 숫자의 평균은 흐릿한 숫자입니다. 특정 타겟에 대한 픽셀 단위 근접성이 아닌 *타당성(plausibility)*을 보상하는 손실이 필요합니다. 타당성에 대한 폐형(closed-form)은 없습니다. 이를 학습해야 합니다.

Goodfellow의 아이디어: `D(x)`를 훈련하여 진짜 이미지와 가짜를 구별합니다. 생성자 `G(z)`를 훈련하여 `D`를 속입니다. `G`에 대한 손실 신호는 `D`가 현재 '진짜처럼 보이게 만드는 것'이라고 생각하는 것입니다. 이 신호는 `G`가 개선됨에 따라 업데이트되며, 움직이는 타겟을 쫓습니다. 두 네트워크가 수렴하면, `G`는 `log p(x)`을 명시적으로 작성하지 않고도 데이터 분포를 학습하게 됩니다.

이것은 적대적 학습(adversarial training)입니다. 수식은 미니맥스(minimax) 게임입니다:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026년 현재 GAN은 더 이상 SOTA 생성기가 아닙니다 (확산 모델과 흐름 매칭(flow matching)이 그 자리를 차지했습니다). 하지만 StyleGAN 2/3는 지금까지 출시된 가장 선명한 얼굴 모델로 남아 있으며, GAN 판별자는 확산 모델 훈련에서 *지각 손실(perceptual losses)*로 사용되고, 적대적 학습은 SDXL-Turbo, SD3-Turbo, LCM 같은 빠른 1단계 증류(distillations)를 가능하게 하여 실시간 확산 모델을 출시할 수 있게 합니다.

## 개념

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**생성자 `G(z)`.** 노이즈 벡터 `z ~ N(0, I)`를 샘플 `x̂`로 매핑합니다. 디코더 형태의 네트워크 (밀집(Dense) 또는 전치 합성곱(transposed conv))입니다.

**판별자 `D(x)`.** 샘플을 스칼라 확률 (또는 점수)로 매핑합니다. 진짜 → 1, 가짜 → 0.

**손실.** 두 개의 교대 업데이트:

- **`D` 훈련:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. 진짜=1, 가짜=0에 대한 이진 교차 엔트로피(Binary cross-entropy).
- **학습 `G`:** `loss_G = -log D(G(z))`. Goodfellow가 사용한 *비포화(non-saturating)* 형태입니다 (원래 `log(1 - D(G(z)))`는 `D`이 확신할 때 포화되어 기울기를 소멸시킵니다).

**학습 루프.** `D` 한 단계, `G` 한 단계. 반복합니다.

**작동하는 이유.** `G`가 `p_data`을 완벽하게 일치시키면, `D`는 우연보다 잘할 수 없고 모든 곳에서 0.5를 출력합니다; `G`은 더 이상 기울기를 받지 못합니다. 평형 상태입니다.

**작동하지 않는 이유.** 모드 붕괴 (`G`가 `D`이 분류할 수 없는 하나의 모드를 찾아 영원히 생성함), 기울기 소멸 (`D`가 너무 빨리 학습되어 `log D`이 포화됨), 학습 불안정성 (학습률, 배치 크기 등).

## GAN을 작동하게 만든 변형들

| 연도 | 혁신 | 해결 |
|------|------------|-----|
| 2015 | DCGAN | 합성곱/역합성곱, 배치 정규화, LeakyReLU — 최초의 안정적 아키텍처. |
| 2017 | WGAN, WGAN-GP | BCE를 Wasserstein 거리 + 기울기 페널티로 대체. 기울기 소멸을 해결합니다. |
| 2017 | 스펙트럴 정규화 | 판별자에 Lipschitz 경계를 설정합니다. 2026년 판별자에서도 여전히 사용됩니다. |
| 2018 | Progressive GAN | 저해상도로 먼저 학습하고, 레이어를 추가합니다. 최초의 메가픽셀 결과. |
| 2019 | StyleGAN / StyleGAN2 | 매핑 네트워크 + 적응형 인스턴스 정규화. 고정된 도메인의 사진적 사실주의에서 최신 기술입니다. |
| 2021 | StyleGAN3 | 에이리어스 프리, 번역 등변성 — 2026년에도 얼굴의 표준입니다. |
| 2022 | StyleGAN-XL | 조건부, 클래스 인식, 더 큰 규모. |
| 2024 | R3GAN | 더 강한 정규화로 리브랜딩; 트릭 없이 1024²에서 작동합니다. |

```figure
gan-minimax
```

## 구현하기

`code/main.py`는 1-D 데이터인 두 가우시안 혼합물에 작은 GAN을 학습합니다. 생성자와 판별자는 단일 은닉층 MLP입니다. 순전파, 역전파, 미니맥스 루프를 직접 구현합니다. 목표는 두 가지 주요 실패 모드 (모드 붕괴 + 기울기 소멸)가 발생하는 것을 보는 것입니다.

### 1단계: 비포화 손실

바닐라 Goodfellow 손실 `log(1 - D(G(z)))`은 판별자(D)가 생성자(G)의 가짜 이미지를 가짜로 높은 확신으로 분류할 때 0으로 수렴합니다. 이때 생성자(G)의 기울기는 사실상 0이 되며, G는 개선할 수 없습니다. 비포화(non-saturating) 형태 `-log D(G(z))`은 반대되는 점근 거동을 보입니다. 판별자(D)가 확신할 때 값이 급증하여 G에 강한 신호를 제공합니다.

```python
def g_loss(d_fake):
    # log D(G(z)) 최대화 <=> -log D(G(z)) 최소화
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 2단계: 생성자 단계마다 판별자 단계 한 번 수행

```python
for step in range(steps):
    # 판별자(D) 학습
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 생성자(G) 학습
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 새로운 가짜 이미지
    update_G(fake_batch)
```

생성자(G)를 위해 새로운 가짜 이미지를 사용해야 합니다. 그렇지 않으면 기울기가 낡은 상태가 됩니다.

### 3단계: 모드 붕괴(mode collapse)를 주의하세요

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

전형적인 증상: 두 개의 실제 모드 중 하나가 더 이상 생성되지 않습니다. 판별자(D)는 이를 가짜로 본 적이 없으므로 수정하지 않습니다.

## 함정

- **판별자(D)가 너무 강함.** 판별자(D)의 학습률을 2~5배 낮추거나, 인스턴스/레이어 노이즈를 추가하세요. 판별자(D)의 정확도가 95%를 넘으면 생성자(G)는 죽은 상태입니다.
- **생성자(G)가 특정 모드만 기억함.** 판별자(D) 입력에 노이즈를 추가하거나, 미니배치 판별자 레이어를 사용하거나, WGAN-GP로 전환하세요.
- **배치 정규화(batch norm) 통계 누수.** 실제 배치와 가짜 배치가 같은 BN 레이어를 통과하면 통계가 섞입니다. 대신 인스턴스 정규화(instance norm)나 스펙트럼 정규화(spectral norm)를 사용하세요.
- **Inception 점수 조작.** FID와 IS는 샘플 수가 적을 때 노이즈가 큽니다. 평가 시에는 1만 개 이상의 샘플을 사용하세요.
- **조건부 작업에서 원샷(one-shot) 샘플링은 거짓말입니다.** 사용 가능한 출력을 얻으려면 CFG 스케일, 절단(truncation) 트릭, 재샘플링이 필요합니다.

## 사용하기

2026년 GAN 스택:

| 상황 | 선택 |
|-----------|------|
| 고정된 포즈의 사실적인 인간 얼굴 | StyleGAN3 (가장 선명하고 작음) |
| 애니메이션 / 스타일화된 얼굴 | StyleGAN-XL 또는 Stable Diffusion LoRA |
| 이미지 간 번역(image-to-image translation) | Pix2Pix / CycleGAN (8단계 · 04) 또는 ControlNet (8단계 · 08) |
| 빠른 1단계 텍스트-이미지 생성 | 확산 모델의 적대적 증류(Adversarial distillation of diffusion) (SDXL-Turbo, SD3-Turbo) |
| 확산 모델 학습기 내부의 지각 손실(perceptual loss) | 이미지 크롭에 대한 작은 GAN 판별자 |
| 멀티모달, 개방형 작업 | 사용하지 마세요 — 확산 모델이나 흐름 매칭(flow matching)을 사용하세요 |

GAN은 선명하지만 범위가 좁습니다. 사진, 임의의 텍스트 프롬프트, 비디오 등 도메인이 확장되면 확산 모델로 전환하세요. 적대적 기법은 독립적인 생성기가 아니라 구성 요소(지각 손실, 증류)로서 계속 활용됩니다.

## 출시하기

`outputs/skill-gan-debugger.md`을 저장하세요. 스킬은 실패한 GAN 실행(손실 곡선, 샘플 그리드, 데이터셋 크기)을 받아 가능한 원인, 한 줄 수정안, 재실행 프로토콜을 순위 매겨 출력합니다.

## 연습 문제

1. **쉬움.** 기본 설정으로 `code/main.py`을 실행하세요. 그 후 `D_LR = 5 * G_LR`을 설정하고 재실행하세요. G의 손실이 얼마나 빠르게 상수로 붕괴하나요?
2. **중간.** Goodfellow BCE 손실을 WGAN 손실로 교체하세요: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, 그리고 D의 가중치를 `[-0.01, 0.01]`으로 클리핑하세요. 학습이 더 안정적인가요? 벽시계 수렴 시간을 비교하세요.
3. **어려움.** 1-D 예제를 2-D 데이터(링 위의 8개 가우시안 혼합)로 확장하세요. 1k, 5k, 10k 단계에서 생성기가 8개 모드 중 몇 개를 포착하는지 추적하세요. 미니배치 판별을 구현하고 다시 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 생성기 | "G" | 잡음에서 샘플로 변환하는 네트워크, `G: z → x̂`. |
| 판별기 | "D" | 분류기 `D: x → [0, 1]`, 진짜와 가짜를 구분. |
| 미니맥스 | "게임" | 결합 목적 함수의 `min_G max_D`. |
| 비포화 손실 | "수정안" | G에 `log(1 - D(G(z)))` 대신 `-log D(G(z))`을 사용하세요. |
| 모드 붕괴 | "G가 한 가지만 기억함" | 다양한 데이터에도 생성기가 몇 개의 서로 다른 출력만 생성함. |
| WGAN | "Wasserstein" | BCE를 Earth-Mover 거리 + 기울기 페널티로 교체; 더 매끄러운 기울기. |
| 스펙트럴 노름 | "Lipschitz 트릭" | D의 가중치 노름을 제한하여 기울기를 제한; 학습을 안정화. |
| StyleGAN | "작동하는 것" | 매핑 네트워크 + AdaIN; 얼굴 생성에서 최고 수준, 2026년에도 여전히 유효. |

## 프로덕션 노트: 일회성 추론은 GAN의 지속적 강점

GAN은 개방 도메인 생성에서 샘플 품질로 더 이상 승리하지 못하지만, 추론 비용에서는 여전히 승리합니다. 프로덕션 추론 문헌의 용어에서 GAN은 다음을 가집니다:

- **프리필, 디코딩 단계 없음.** 단일 `G(z)` 순방향 패스. TTFT ≈ 총 지연 시간.
- **KV 캐시 압력 없음.** 유일한 상태는 가중치입니다. 배치 크기는 캐시가 아닌 활성화 메모리에 의해 제한됩니다.
- **간단한 연속 배치.** 모든 요청이 동일한 고정 FLOPs를 사용하므로, 서버의 목표 점유율에서의 정적 배치가 일반적으로 최적입니다. 진행 중 스케줄러가 필요하지 않습니다.

이것이 2026년 빠른 텍스트-이미지 생성에서 GAN 증류(SDXL-Turbo, SD3-Turbo, ADD, LCM)가 지배적인 기법으로 자리 잡은 이유입니다. 이 기법은 20-50단계 확산 파이프라인을 1-4번의 GAN 스타일 순방향 패스로 압축하면서도 확산 기반의 분포를 유지합니다. 적대적 손실은 느린 생성기를 빠른 생성기로 변환하기 위한 학습 시간 조정 변수로 남습니다.

## 추가 읽기

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 원본 GAN 논문.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 최초의 안정적 아키텍처.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
