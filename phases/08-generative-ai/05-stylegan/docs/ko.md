# StyleGAN

> 대부분의 생성기는 `z`을 모든 레이어에 동시에 주입합니다. StyleGAN은 이를 분리했습니다. 먼저 `z`을 중간 `w`로 매핑한 후, AdaIN을 통해 모든 해상도 수준에서 `w`을 *주입*합니다. 이 단일 변경으로 잠재 공간(latent space)이 분리되어, 7년 동안 사실적인 얼굴 생성이 해결된 문제가 되었습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 03 (GAN), 4단계 · 08 (정규화), 3단계 · 07 (CNN)
**시간:** 약 45분

## 문제점

DCGAN은 `z`을 전치 합성곱(convolution) 스택을 통해 이미지로 매핑합니다. 문제는 `z`이 포즈, 조명, 정체성, 배경 등 모든 것을 함께 제어한다는 점입니다. `z`의 한 축을 따라 이동하면 네 가지가 모두 변합니다. 표현이 그렇게 분해되지 않기 때문에 모델에게 "같은 사람, 다른 포즈"를 요청할 수 없습니다.

Karras et al. (2019, NVIDIA)은 conv 레이어에 `z`을 직접 공급하는 것을 중단할 것을 제안했습니다. 네트워크 입력으로 상수 `4×4×512` 텐서를 공급합니다. `z ∈ Z → w ∈ W`을 매핑하는 8층 MLP를 학습합니다. *적응형 인스턴스 정규화*(AdaIN)를 통해 모든 해상도에서 `w`을 주입합니다: 각 conv 특징 맵을 정규화한 후, `w`의 아핀 투영으로 스케일 및 시프트합니다. 확률적 세부 사항(피부 모공, 머리카락 가닥)을 위해 레이어별 노이즈를 추가합니다.

결과: `W`는 "고수준 스타일"(포즈, 정체성)과 "세부 스타일"(조명, 색상)에 대해 대략적으로 직교하는 축을 가집니다. 저해상도 수준에서는 이미지 A의 `w`을, 고해상도에서는 이미지 B의 `w`을 사용하여 두 이미지 간에 스타일을 교환할 수 있습니다. 이는 편집, 크로스 도메인 스타일화 및 전체 "StyleGAN-inversion" 연구 라인을 가능하게 했습니다.

## 개념

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**매핑 네트워크.** `f: Z → W`, 8층 MLP. `Z = N(0, I)^512`. `W`는 가우시안 분포로 강제되지 않으며, 데이터에 적응된 형태를 학습합니다.

**합성 네트워크.** 학습된 상수 `4×4×512`에서 시작합니다. 각 해상도 블록: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. 해상도는 4, 8, 16, 32, 64, 128, 256, 512, 1024로 두 배씩 증가합니다.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

여기서 `y_scale`와 `y_bias`는 `w`의 아핀 투영에서 나옵니다. 각 특징 맵(feature map)에 대해 정규화한 후 스타일링을 적용하세요. 여기서 "스타일"은 특징 맵의 1차 및 2차 통계입니다.

**레이어별 노이즈.** 각 특징 맵에 단일 채널 가우시안 노이즈를 추가하고, 학습된 채널별 계수로 스케일링합니다. 전역 구조에 영향을 주지 않으면서 확률적 세부 사항을 제어합니다.

**절단(truncation) 트릭.** 추론 시 `z`를 샘플링하고, `w = mapping(z)`를 계산한 후 `w' = ŵ + ψ·(w - ŵ)`를 계산합니다. 여기서 `ŵ`는 많은 샘플에 대한 평균 `w`입니다. `ψ < 1`는 다양성을 희생하여 품질을 높입니다. 거의 모든 StyleGAN 데모가 `ψ ≈ 0.7`를 사용합니다.

## StyleGAN 1 → 2 → 3

| 버전 | 연도 | 혁신 |
|---------|------|------------|
| StyleGAN | 2019 | 매핑 네트워크 + AdaIN + 노이즈 + 점진적 성장. |
| StyleGAN2 | 2020 | 가중치 변조(weight demodulation)가 AdaIN을 대체(드롭릿 아티팩트 수정); 스킵/잔여(residual) 아키텍처; 경로 길이 정규화. |
| StyleGAN3 | 2021 | 에이리어스 프리(alias-free) 컨볼루션 + 등변 커널; 픽셀 그리드에 텍스처가 붙는 현상 제거. |
| StyleGAN-XL | 2022 | 클래스 조건부, 1024², ImageNet. |
| R3GAN | 2024 | 더 강력한 정규화로 리브랜딩; FFHQ-1024에서 파라미터를 20배 적게 사용하며 확산 모델과의 격차 해소. |

2026년에도 StyleGAN3는 (a) 높은 FPS에서의 좁은 영역 사진 리얼리즘, (b) 소수 예시(few-shot) 도메인 적응(100장의 이미지로 새 데이터셋에 학습하고 매핑을 고정), (c) 역전(inversion) 기반 편집(실제 사진을 재구성하는 `w`를 찾은 후 해당 `w`를 편집)의 기본 도구로 남아 있습니다. 개방형 도메인 텍스트-투-이미지에는 이 도구가 적합하지 않으며, 확산 모델이 적합합니다.

```figure
gx-stylegan-mapping
```

## 구현하기

`code/main.py`는 1-D에서 장난감 "style-GAN lite"를 구현합니다. 매핑 MLP, 학습된 상수 벡터를 받아 `w`에서 파생된 스케일/바이어스로 변조하는 합성 함수, 그리고 레이어별 노이즈를 포함합니다. `w`를 아핀 변조로 주입하는 것이 생성기의 입력에 `z`를 연결(concatenating)하는 것보다 성능이 같거나 더 좋다는 것을 보여줍니다.

### 1단계: 매핑 네트워크

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 2단계: 적응형 인스턴스 정규화

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

각 특징 맵의 스케일과 바이어스는 `w`에서 선형 투영을 통해 나옵니다.

### 3단계: 레이어별 노이즈

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

채널별 시그마(sigma)는 학습 가능합니다.

## 주의할 점

- **드롭렛 아티팩트.** StyleGAN 1은 AdaIN이 평균을 0으로 만들었기 때문에 특징 맵에서 뭉친 드롭렛을 생성했습니다. StyleGAN 2의 가중치 변조(weight demodulation)는 합성곱 가중치를 스케일링하여 이 문제를 해결합니다.
- **텍스처 고정.** StyleGAN 01강 2의 텍스처는 객체 좌표가 아닌 픽셀 좌표를 따랐습니다 (보간 시 확인 가능). StyleGAN 3의 에이리어스 프리(alias-free) 합성곱은 윈도우 sinc 필터를 사용하여 이 문제를 해결합니다.
- **모드 커버리지.** 절단(truncation) `ψ < 0.7`은 깔끔해 보이지만 좁은 원뿔에서 샘플링합니다. 다양성이 필요하다면 `ψ = 1.0`을 사용해 보세요.
- **역변환은 손실적입니다.** 실제 사진을 `W`로 역변환하는 것은 보통 최적화나 인코더(e4e, ReStyle, HyperStyle)를 통해 수행됩니다. 많은 반복을 거치면 결과가 변합니다.

## 사용하기

| 사용 사례 | 접근법 |
|----------|----------|
| 사실적인 인간 얼굴 (애니메이션, 제품, 좁은 범위) | StyleGAN3 FFHQ / 커스텀 미세 조정 |
| 사진 기반 얼굴 편집 | e4e 역변환 + StyleSpace / InterFaceGAN 방향 |
| 얼굴 스왑 / 재현 | StyleGAN + 인코더 + 블렌딩 |
| 아바타 파이프라인 | StyleGAN3 + ADA를 활용한 저데이터 미세 조정 |
| 몇 장의 이미지로부터의 도메인 적응 | 매핑 네트워크 동결, 합성 네트워크 미세 조정 |
| 멀티모달 또는 텍스트 조건부 생성 | 하지 마세요 — 확산 모델을 사용하세요 |

"사람 얼굴 사진"이 답인 제품급 데모에서는 StyleGAN이 확산 모델보다 추론 비용(단일 순방향 패스, 4090에서 <10ms)과 동일한 품질 기준에서의 선명도 측면에서 우월합니다.

## 출시하기

`outputs/skill-stylegan-inversion.md`을 저장하세요. 스킬은 실제 사진을 입력받아 다음을 출력합니다: 역변환 방법(e4e / ReStyle / HyperStyle), 예상 잠재(latent) 손실, 편집 예산(`W`에서 아티팩트가 발생하기 전까지 얼마나 이동할 수 있는지), 그리고 알려진 좋은 편집 방향 목록(나이, 표정, 자세).

## 연습 문제

1. **쉬움.** `code/main.py`을 `adain_on=True` 및 `adain_on=False`와 함께 실행하세요. 고정된 잠재(latent)과 섭동된 잠재(latent)의 출력 분포를 비교하세요.
2. **중간.** 혼합 정규화(mixing regularization)를 구현하세요. 학습 배치에 대해 `w_a`, `w_b`을 계산하고, 합성의 첫 절반에는 `w_a`을, 두 번째 절반에는 `w_b`을 적용하세요. 디코더가 분리된(disentangled) 스타일을 학습하나요?
3. **난이도: 상.** 사전 학습된 StyleGAN3 FFHQ 모델(ffhq-1024.pkl)을 사용하여 레이블이 지정된 샘플에 SVM을 학습함으로써 "미소"를 제어하는 `w` 방향을 찾아보세요. 정체성이 변형되기 전까지 얼마나 강하게 조정할 수 있는지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 매핑 네트워크 | "MLP" | `f: Z → W`, 8층, 잠재 공간의 기하학적 구조를 데이터 통계와 분리합니다. |
| W 공간 | "스타일 공간" | 매핑 네트워크의 출력; 대략적으로 분리(disentangled)되어 있습니다. |
| AdaIN | "적응형 인스턴스 정규화" | 특징 맵을 정규화한 후 `w` 투영으로 스케일링 및 시프트합니다. |
| 절단 트릭 | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1은 다양성을 희생하여 품질을 높입니다. |
| 경로 길이 정규화 | "PL reg" | `w`의 단위 변화당 이미지 변화가 클 경우 페널티를 부과합니다; `W`을 더 매끄럽게 만듭니다. |
| 가중치 변조 | "StyleGAN2의 수정 사항" | 활성화 대신 컨볼루션 가중치를 정규화합니다; 물방울(droplet) 아티팩트를 제거합니다. |
| 에이리어스 프리 | "StyleGAN3의 트릭" | 창(windowed) sinc 필터; 텍스처가 픽셀 그리드에 달라붙는 현상을 제거합니다. |
| 역변환 | "실제 이미지에 대한 w 찾기" | `x → w`를 최적화하거나 인코딩하여 `G(w) ≈ x`를 수행합니다. |

## 프로덕션 노트: 2026년에도 StyleGAN이 여전히 사용되는 이유

StyleGAN3는 4090 GPU에서 10ms 미만의 시간으로 1024² FFHQ 얼굴을 생성합니다 — `num_steps = 1`, VAE 디코딩 없음, 교차 어텐션 패스 없음. 프로덕션 관점에서 이는 모든 이미지 생성기의 최소 지연 시간(floor latency)입니다. 같은 해상도의 50단계 SDXL + VAE 디코딩 파이프라인은 약 3초가 소요됩니다. 이는 **300배의 차이**이며, 좁은 도메인 제품(아바타 서비스, 신분증 파이프라인, 스톡 얼굴 생성)에서는 TCO(총 소유 비용) 측면에서 우위를 점합니다.

두 가지 운영적 결과:

- **스케줄러 없음, 배처 없음.** 목표 점유율에서의 정적 배치가 최적입니다. 연속 배치(LLM 및 확산 모델에 필수)는 모든 요청이 동일한 FLOPs를 필요로 하므로 이점이 전혀 없습니다.
- **절단 `ψ`가 안전 조절기(safety knob)입니다.** `ψ < 0.7`는 매핑 네트워크 범위의 좁은 원뿔(cone)에서 샘플링합니다. 이는 서빙 레이어가 샘플 분산에 대해 가지는 유일한 제어 수단입니다. 피크 부하에서는 `ψ`를 낮추고, 프리미엄 사용자를 위해서는 높여보세요.

## 추가 읽기

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2입니다.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3입니다.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 역전입니다.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL입니다.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 현대적 최소 GAN 레시피입니다.
