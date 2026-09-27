# StyleGAN

> 대부분의 생성기(Generator)는 모든 레이어에 동시에 `z`를 주입합니다. StyleGAN은 이를 분리했습니다. 먼저 `z`를 중간 단계인 `w`로 매핑한 다음, AdaIN을 통해 모든 해상도 레벨에서 `w`를 *주입(inject)*합니다. 이 단 한 번의 변화가 잠재 공간(latent space)의 얽힘을 풀어냈으며, 7년 동안 실사 같은 얼굴 생성 문제를 해결된 과제로 만들었습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## 문제점 (The Problem)

DCGAN은 전치 컨볼루션(transposed convolutions) 스택을 통해 `z`를 이미지로 매핑합니다. 문제는 `z`가 포즈, 조명, 정체성, 배경 등 모든 요소를 얽힌(entangled) 상태로 제어한다는 점입니다. `z`의 한 축을 따라 이동하면 이 네 가지 요소가 모두 함께 변합니다. 표현 방식이 분리되어 있지 않기 때문에 모델에게 "같은 사람이지만 다른 포즈"를 요청할 수 없습니다.

Karras 등(2019, NVIDIA)은 다음과 같은 방식을 제안했습니다: `z`를 컨볼루션 레이어에 직접 입력하는 것을 중단하십시오. 대신 상수 `4×4×512` 텐서를 네트워크 입력으로 사용합니다. `z ∈ Z → w ∈ W`로 매핑하는 8개 레이어의 MLP를 학습시킵니다. 그 후 *적응형 인스턴스 정규화*(adaptive instance normalization, AdaIN)를 통해 모든 해상도 단계에서 `w`를 주입합니다. 즉, 각 컨볼루션 피처 맵을 정규화한 다음, `w`의 아핀 투영(affine projections)을 사용하여 스케일과 시프트를 적용합니다. 또한 확률적인 세부 사항(모공, 머리카락 가닥 등)을 위해 레이어마다 노이즈를 추가합니다.

그 결과: `W`는 "고수준 스타일"(포즈, 정체성)과 "저수준 스타일"(조명, 색상)에 대해 대략적으로 직교하는(orthogonal) 축을 갖게 됩니다. 저해상도 레벨에는 이미지 A의 `w`를 사용하고, 고해상도 레벨에는 이미지 B의 `w`를 사용하여 두 이미지 간의 스타일을 교체할 수 있습니다. 이는 편집(editing), 교차 도메인 스타일화(cross-domain stylization), 그리고 "StyleGAN-inversion"이라는 전체 연구 분야를 열어주었습니다.

## 개념 (The Concept)

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**매핑 네트워크 (Mapping network).** `f: Z → W` 형태의 8개 레이어로 구성된 MLP입니다. `Z = N(0, I)^512`이며, `W`는 가우시안 분포를 따르도록 강제되지 않습니다. 대신 데이터에 적합한 형태를 학습합니다.

**합성 네트워크 (Synthesis network).** 학습된 상수 `4×4×512`에서 시작합니다. 각 해상도 블록은 `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise` 과정을 거칩니다. 해상도는 4, 8, 16, 32, 64, 128, 256, 512, 1024로 두 배씩 증가합니다.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

여기서 `y_scale`과 `y_bias`는 `w`의 아핀 투영(affine projections)으로부터 얻어집니다. 각 피처 맵(feature map)별로 정규화한 후 스타일을 재구성(restyle)합니다. 여기서 "스타일(Style)"은 피처 맵의 1차 및 2차 통계량을 의미합니다.

**레이어별 노이즈 (Per-layer noise).** 각 피처 맵에 단일 채널 가우시안 노이즈를 추가하며, 이는 학습된 채널별 계수에 의해 스케일링됩니다. 이는 전체적인 구조에 영향을 주지 않으면서 확률적인 세부 사항(stochastic detail)을 제어합니다.

**절단 트릭 (Truncation trick).** 추론 시, `z`를 샘플링하고 `w = mapping(z)`를 계산한 뒤, `w' = ŵ + ψ·(w - ŵ)`를 구합니다. 여기서 `ŵ`는 많은 샘플에 대한 `w`의 평균입니다. `ψ < 1` 값을 조절하여 다양성(diversity)과 품질(quality) 사이의 균형을 맞춥니다. 거의 모든 StyleGAN 데모는 `ψ ≈ 0.7`을 사용합니다.

## StyleGAN 1 → 2 → 3

| 버전 (Version) | 연도 (Year) | 혁신 요소 (Innovation) |
|---------|------|------------|
| StyleGAN | 2019 | 매핑 네트워크(Mapping network) + AdaIN + 노이즈 + 점진적 성장(progressive growing). |
| StyleGAN2 | 2020 | 가중치 디모듈레이션(Weight demodulation)이 AdaIN을 대체(물방울 아티팩트 해결); 스킵/잔차(skip/residual) 구조; 경로 길이 정규화(path-length regularization). |
| StyleGAN3 | 2021 | 에일리어스 프리 컨볼루션(Alias-free convolution) + 등변 커널(equivariant kernels); 픽셀 그리드에 질감이 고정되는 현상 제거. |
| StyleGAN-XL | 2022 | 클래스 조건부(Class-conditional), 1024², ImageNet. |
| R3GAN | 2024 | 더 강력한 정규화로 재탄생; FFHQ-1024에서 20배 적은 파라미터로 디퓨전(diffusion)과의 격차 해소. |

2026년 기준으로 StyleGAN3는 다음의 경우에 기본적으로 사용됩니다: (a) 높은 FPS를 유지하는 좁은 도메인의 실사성(photorealism), (b) 퓨샷 도메인 적응(few-shot domain adaptation) (100장의 이미지로 새로운 데이터셋 학습, 매핑 네트워크 고정), (c) 인버전 기반 편집(inversion-based editing) (실제 사진을 재구성하는 `w`를 찾은 뒤 해당 `w`를 편집). 오픈 도메인 텍스트-투-이미지(text-to-image)의 경우, StyleGAN3가 아닌 디퓨전(diffusion)이 적합한 도구입니다.

```figure
gx-stylegan-mapping
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 1차원 형태의 간단한 "style-GAN lite"를 구현합니다. 여기에는 매핑 MLP(mapping MLP), 학습된 상수 벡터를 입력받아 `w`에서 유도된 스케일/바이어스(scale/bias)로 이를 변조(modulate)하는 합성 함수(synthesis function), 그리고 레이어별 노이즈(per-layer noise)가 포함됩니다. 이 구현을 통해 `w`를 아핀 변조(affine-modulation) 방식으로 주입하는 것이 생성기(generator)의 입력에 `z`를 결합(concatenating)하는 방식과 대등하거나 오히려 더 나은 성능을 보임을 확인할 수 있습니다.

### 1단계: 매핑 네트워크 (mapping network)

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 2단계: 적응형 인스턴스 정규화 (Adaptive Instance Normalization)

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

특징 맵(feature-map)별 스케일(scale)과 편향(bias)은 선형 투영(linear projection)을 통해 `w`로부터 도출됩니다.

### 3단계: 레이어별 노이즈 (per-layer noise)

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

채널당 `sigma`는 학습 가능합니다.

## 주의 사항 (Pitfalls)

- **물방울 아티팩트 (Droplet artifacts).** StyleGAN 1은 AdaIN이 평균을 0으로 만들었기 때문에 피처 맵(feature maps)에 덩어리진 물방울 형태의 아티팩트를 생성했습니다. StyleGAN 2의 가중치 디모듈레이션(weight demodulation)은 컨볼루션 가중치를 스케일링하는 방식으로 이 문제를 해결합니다.
- **텍스처 고착 (Texture sticking).** StyleGAN 1과 2의 텍스처는 객체 좌표가 아닌 픽셀 좌표를 따랐습니다(보간 시 눈에 띔). StyleGAN 3의 에일리어스 프리 컨볼루션(alias-free convolutions)은 윈도우드 싱크 필터(windowed sinc filters)를 사용하여 이 문제를 해결합니다.
- **모드 커버리지 (Mode coverage).** 절단(Truncation) `ψ < 0.7`은 결과물이 깔끔해 보이지만 좁은 원뿔 영역에서만 샘플링합니다. 다양성이 필요하다면 `ψ = 1.0`을 사용해 보세요.
- **인버전은 손실이 발생합니다 (Inversion is lossy).** 실제 사진을 `W` 공간으로 인버전하는 작업은 대개 최적화(optimization)나 인코더(`e4e`, `ReStyle`, `HyperStyle`)를 통해 수행됩니다. 결과물은 많은 반복(iteration)을 거치면서 드리프트(drift) 현상이 발생할 수 있습니다.

## 활용 방법 (Use It)

| 유스케이스 (Use case) | 접근 방식 (Approach) |
|----------|----------|
| 실사 인물 얼굴 (애니메이션, 제품, 협소한 범위) | StyleGAN3 FFHQ / 커스텀 파인튜닝(custom fine-tune) |
| 사진을 이용한 얼굴 편집 | e4e inversion + StyleSpace / InterFaceGAN 방향성(directions) |
| 얼굴 교체 / 재현 (Face swap / reenactment) | StyleGAN + 인코더(encoder) + 블렌딩(blending) |
| 아바타 파이프라인 | 데이터가 적은 경우 ADA를 활용한 StyleGAN3 파인튜닝 |
| 소량의 이미지를 이용한 도메인 적응 (Domain adaptation) | 매핑 네트워크(mapping network)는 고정하고, 합성(synthesis) 네트워크를 파인튜닝 |
| 멀티모달 또는 텍스트 조건부 생성 | 사용하지 마세요 — 확산 모델(diffusion)을 사용하세요 |

정답이 "사람 얼굴 사진"인 제품급 데모의 경우, StyleGAN은 추론 비용(단일 순전파, 4090 기준 10ms 미만)과 동일한 품질 기준에서의 선명도 측면에서 확산 모델(diffusion)보다 우수합니다.

## Ship It (실행하기)

`outputs/skill-stylegan-inversion.md`를 저장하세요. 이 스킬은 실제 사진을 입력받아 다음 항목들을 출력합니다: 인버전 방법(inversion method: `e4e` / `ReStyle` / `HyperStyle`), 예상 잠재 손실(expected latent loss), 편집 예산(editing budget: 아티팩트가 발생하기 전 `W` 공간에서 이동할 수 있는 거리), 그리고 검증된 편집 방향 리스트(known-good editing directions: 나이, 표정, 포즈).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `adain_on=True`와 `adain_on=False` 설정으로 `code/main.py`를 실행해 보세요. 고정된 잠재 변수(fixed latent)와 섭동된 잠재 변수(perturbed latent)에 대해 출력값의 확산(spread) 정도를 비교해 보세요.
2. **중간 (Medium).** 믹싱 정규화(mixing regularization)를 구현해 보세요. 학습 배치에 대해 `w_a`와 `w_b`를 계산한 뒤, 합성(synthesis)의 전반부에는 `w_a`를, 후반부에는 `w_b`를 적용합니다. 디코더가 얽히지 않은 스타일(disentangled styles)을 학습하나요?
3. **어려움 (Hard).** 사전 학습된 StyleGAN3 FFHQ 모델(`ffhq-1024.pkl`)을 사용하세요. 라벨링된 샘플로 SVM을 학습시켜 "미소(smile)"를 조절하는 `w` 방향을 찾아보세요. 정체성(identity)이 변하기 전까지 얼마나 멀리 밀어붙일 수 있는지 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 매핑 네트워크 (Mapping network) | "MLP" | `f: Z → W`, 8개 레이어로 구성되며, 잠재 기하학(latent geometry)을 데이터 통계로부터 분리합니다. |
| W 공간 (W space) | "스타일 공간 (style space)" | 매핑 네트워크의 출력값이며, 대략적으로 얽힘이 해제(disentangled)된 상태입니다. |
| AdaIN | "적응형 인스턴스 정규화 (Adaptive instance norm)" | 특징 맵(feature map)을 정규화한 후, `w` 투영값에 따라 스케일(scale) 및 시프트(shift)를 적용합니다. |
| 절단 트릭 (Truncation trick) | "Psi (ψ)" | `w = mean + ψ·(w - mean)`, ψ<1을 통해 다양성을 희생하는 대신 품질을 높입니다. |
| 경로 길이 정규화 (Path-length regularization) | "PL reg" | `w`의 단위 변화당 이미지의 큰 변화에 페널티를 부여하여, `W`를 더 부드럽게 만듭니다. |
| 가중치 디모듈레이션 (Weight demodulation) | "StyleGAN2의 해결책" | 활성화 함수(activation) 대신 컨볼루션 가중치를 정규화하여, 물방울 아티팩트(droplet artifacts)를 제거합니다. |
| 에일리어싱 프리 (Alias-free) | "StyleGAN3의 트릭" | 윈도우 싱크 필터(Windowed sinc filters)를 사용하여, 텍스처가 픽셀 그리드에 고정되는 현상을 제거합니다. |
| 인버전 (Inversion) | "실제 이미지에 대한 w 찾기" | `G(w) ≈ x`가 되도록 `x → w`를 최적화하거나 인코딩합니다. |

## 프로덕션 노트: 2026년에도 StyleGAN이 여전히 사용되는 이유

4090 GPU에서 StyleGAN3는 `num_steps = 1`, VAE 디코딩 없음, cross-attention 패스 없음 조건으로 1024² FFHQ 얼굴을 10ms 이내에 생성합니다. 프로덕션 관점에서 이는 모든 이미지 생성기의 최저 지연 시간(floor latency)입니다. 동일한 해상도에서 50단계의 SDXL + VAE-decode 파이프라인은 약 3초가 소요됩니다. 이는 **300배의 격차**이며, 특정 도메인 제품(아바타 서비스, 신분증 문서 파이프라인, 스톡 얼굴 생성)의 경우 총 소유 비용(TCO) 측면에서 압도적인 우위를 점합니다.

두 가지 운영상의 결과는 다음과 같습니다:

- **스케줄러와 배치 처리기(batcher)가 필요 없습니다.** 목표 점유율(occupancy)에 맞춘 정적 배치(Static batch)가 최적입니다. 모든 요청이 동일한 FLOPs를 소모하기 때문에, (LLM과 확산 모델에 필수적인) 연속 배치(Continuous batching)는 아무런 이득을 주지 못합니다.
- **절단(Truncation) `ψ`는 안전 조절 노브(safety knob)입니다.** `ψ < 0.7`은 매핑 네트워크 범위의 좁은 원뿔(cone)에서 샘플링합니다. 이는 서빙 레이어가 샘플 분산(sample variance)을 제어할 수 있는 유일한 레버입니다. 피크 부하 시에는 `ψ`를 낮추고, 프리미엄 사용자에게는 높여서 제공하세요.

## 추가 읽을거리 (Further Reading)

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 현대적인 최소 GAN 레시피(modern minimal GAN recipe).
