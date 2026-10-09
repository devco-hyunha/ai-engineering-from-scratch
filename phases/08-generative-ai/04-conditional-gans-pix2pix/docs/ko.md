# 조건부 GAN 및 Pix2Pix

> 2014-2017년의 첫 번째 큰 돌파구는 GAN이 무엇을 생성하는지 제어하는 것이었습니다. 레이블, 이미지, 문장을 첨부해 보세요. Pix2Pix는 이미지 버전을 수행했으며, 좁은 이미지-to-image 작업에서는 여전히 모든 범용 텍스트-to-image 모델을 능가합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 03강 (GAN), 4단계 · 06강 (U-Net), 3단계 · 07강 (CNN)
**시간:** 약 75분

## 문제점

무조건 GAN은 임의의 얼굴을 샘플링합니다. 데모에는 유용하지만 프로덕션에서는 쓸모가 없습니다. 다음을 원합니다: *스케치를 사진으로 매핑*, *지도를 항공 사진으로 매핑*, *낮 장면을 밤으로 매핑*, *그레이스케일 이미지를 컬러화*. 이 모든 경우, 입력 이미지 `x`가 주어지며 `y`을 의미론적 대응을 포함하여 출력해야 합니다. 각 `x`에 대해 많은 타당한 `y`가 존재합니다. 평균 제곱 오차는 이것들을 흐릿한 덩어리로 평평하게 만듭니다. 적대적 손실은 "실제처럼 보인다"가 날카롭기 때문에 그렇게 하지 않습니다.

조건부 GAN (Mirza & Osindero, 2014)은 `G`과 `D` 모두의 입력으로 조건 `c`을 추가합니다. Pix2Pix (Isola et al., 2017)는 이를 특화했습니다: 조건은 전체 입력 이미지, 생성기는 U-Net, 판별자는 *패치 기반* 분류기(PatchGAN)이며, 손실은 적대적 + L1입니다. 이 레시피는 *페어링된 데이터*로 훈련되기 때문에 *2026년에도* 좁은 이미지-to-image 도메인에서 처음부터 훈련된 텍스트-to-image 모델을 능가합니다. 필요한 신호가 정확히 있습니다.

## 개념

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**조건부 G.** `G(x, z) → y`. Pix2Pix에서는 `z`이 G 내부의 드롭아웃입니다 (입력 노이즈 없음 — Isola는 명시적인 노이즈가 무시되는 것을 발견했습니다).

**조건부 D.** `D(x, y) → [0, 1]`. 입력은 *쌍* (조건, 출력)입니다. 이것이 핵심적인 차이점입니다: D는 `y`이 실제처럼 보이는지 여부뿐만 아니라 `y`이 `x`와 일치하는지 판단해야 합니다.

**U-Net 생성기.** 병목 현상을 가로지르는 스킵 연결을 가진 인코더-디코더입니다. 입력과 출력이 저수준 구조(모서리, 윤곽)를 공유하는 작업에 중요합니다. 스킵이 없으면 고주파 세부 사항이 사라집니다.

**PatchGAN 판별자.** 단일한 진짜/가짜 점수를 출력하는 대신, D는 `N×N` 격자를 출력하며, 각 셀은 약 70×70 픽셀의 수용 영역을 판별합니다. 평균을 냅니다. 이는 마르코프 랜덤 필드 가정입니다: 현실성은 지역적입니다. 훈련이 훨씬 빠르고, 매개변수가 적으며, 출력이 더 선명합니다.

**손실.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 항은 훈련을 안정화하고 G를 알려진 목표로 밀어냅니다. L1은 L2보다 더 선명한 가장자리를 제공합니다 (평균이 아닌 중앙값). `λ = 100`은 Pix2Pix의 기본값이었습니다.

## CycleGAN — 쌍이 없을 때

Pix2Pix는 쌍을 이루는 `(x, y)` 데이터가 필요합니다. CycleGAN (Zhu et al., 2017)은 이 요구 사항을 제거하며, 추가 손실인 *순환 일관성(cycle consistency)* 손실을 대가로 치릅니다. 두 생성자 `G: X → Y`과 `F: Y → X`가 있습니다. `F(G(x)) ≈ x`과 `G(F(y)) ≈ y`이 되도록 훈련합니다. 이를 통해 쌍을 이루는 예시 없이 말에서 얼룩말, 여름에서 겨울로 변환할 수 있습니다.

2026년 현재, 쌍을 이루지 않는 이미지 간 변환은 대부분 CycleGAN이 아니라 확산 모델 (ControlNet, IP-Adapter)을 통해 수행되지만, 순환 일관성 개념은 거의 모든 쌍을 이루지 않는 도메인 적응 논문에서 살아남습니다.

```figure
gx-patchgan
```

## 구현하기

`code/main.py`은 1-D 데이터에 작은 조건부 GAN을 구현합니다. 조건 `c`은 클래스 레이블 (0 또는 1)입니다. 과제: 주어진 클래스에 대한 조건부 분포에서 샘플을 생성하는 것.

### 1단계: G와 D 입력에 조건 추가

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

원 핫(one-hot) 인코딩이 가장 단순한 방법입니다. 더 큰 모델은 학습된 임베딩, FiLM 변조, 또는 교차 어텐션을 사용합니다.

### 2단계: 조건부 훈련

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

생성자는 주변 분포가 아니라 *주어진 조건에 대한* 실제 분포를 따라야 합니다.

### 3단계: 클래스별 출력 검증

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 함정

- **조건 무시.** G는 주변 분포로 학습하고, D는 조건 신호가 약해서 벌점을 매기지 않습니다. 해결책: D에 조건을 더 공격적으로 적용 (초기 레이어, 후기 레이어만 아님), 투사 판별자(projection discriminator) 사용 (Miyato & Koyama 2018).
- **L1 가중치가 너무 낮음.** G는 충실한 출력 대신 임의의 실제처럼 보이는 출력으로 드리프트합니다. Pix2Pix 스타일 작업에서는 λ≈100으로 시작하세요.
- **L1 가중치가 너무 높음.** L1은 여전히 L_p 노름이므로 G는 흐릿한 출력을 생성합니다. 훈련이 안정화되면 가중치를 낮추세요.
- **D에서의 정답 유출.** `(x, y)`를 D 입력으로 연결하고, `y`만 사용하지 마세요. 이 없이는 D가 일관성을 확인할 수 없습니다.
- **클래스별 모드 붕괴.** 각 클래스는 독립적으로 붕괴할 수 있습니다. 클래스 조건부 다양성 검사를 실행해 보세요.

## 사용하기

2026년 이미지-to-image 작업 현황:

| 작업 | 최적 접근법 |
|------|---------------|
| 스케치 → 사진, 동일 도메인, 쌍 데이터 | Pix2Pix / Pix2PixHD (여전히 빠르고 선명함) |
| 스케치 → 사진, 비쌍 데이터 | Scribble 조건부 모델이 있는 ControlNet |
| 의미론적 세그멘테이션 → 사진 | SPADE / GauGAN2 또는 SD + ControlNet-Seg |
| 스타일 변환 | IP-Adapter 또는 LoRA가 있는 확산 모델; GAN 방법은 레거시 |
| 깊이 → 사진 | Stable Diffusion 기반의 ControlNet-Depth |
| 초해상도 | Real-ESRGAN (GAN), ESRGAN-Plus, 또는 SD-Upscale (확산) |
| 채색 | ColTran, 확산 기반 채색기, 또는 Pix2Pix-color |
| 낮 → 밤, 계절, 날씨 | CycleGAN 또는 ControlNet 기반 |

Pix2Pix는 (a) 수천 개의 쌍 예제가 있고, (b) 작업이 좁고 반복 가능하며, (c) 빠른 추론이 필요할 때 여전히 올바른 도구입니다. 일반적인 오픈 도메인 작업에서는 확산 모델이 승리합니다.

## 출시하기

`outputs/skill-img2img-chooser.md`를 저장하세요. 스킬은 작업 설명, 데이터 가용성(쌍 vs 비쌍, N 샘플), 지연/품질 예산을 받아 접근법(Pix2Pix, CycleGAN, ControlNet 변형, SDXL + IP-Adapter), 훈련 데이터 요구 사항, 추론 비용 및 평가 프로토콜(LPIPS, FID, 작업별)을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 수정하여 세 번째 클래스를 추가하세요. G가 여전히 각 클래스의 잡음을 올바른 모드로 매핑하는지 확인하세요.
2. **중간.** 1-D 설정에서 L1을 지각 스타일 손실(예: 기능 추출기 역할을 하는 작은 고정된 D)로 교체하세요. 조건부 분포의 선명도가 변합니까?
3. **어려움.** 1-D 설정에서 CycleGAN을 스케치하세요: 두 분포, 두 생성기, 사이클 손실. 쌍 데이터 없이 두 분포 간 매핑을 학습함을 보이세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 조건부 GAN | "레이블이 있는 GAN" | G(z, c), D(x, c). 두 네트워크 모두 조건을 참조합니다. |
| Pix2Pix | "이미지-이미지 GAN" | U-Net G와 PatchGAN D + L1 손실을 사용하는 쌍이 있는 cGAN입니다. |
| U-Net | "스킵 연결이 있는 인코더-디코더" | 대칭적인 합성곱 네트워크; 스킵 연결이 고주파 성분을 보존합니다. |
| PatchGAN | "국소적 현실성 분류기" | D는 전역 점수 대신 패치별 점수를 출력합니다. |
| CycleGAN | "쌍이 없는 이미지 변환" | 두 개의 G와 순환 일관성 손실; 쌍이 있는 데이터가 필요 없습니다. |
| SPADE | "GauGAN" | 중간 활성화를 세만틱 맵으로 정규화; 세만틱 맵에서 이미지로 변환합니다. |
| FiLM | "특징별 선형 변조" | 조건에서 파생된 특징별 아핀 변환; 저렴한 조건부 처리입니다. |

## 프로덕션 노트: 지연 시간 제한 기준선으로서의 Pix2Pix

쌍이 있는 데이터와 좁은 작업(스케치 → 렌더링, 세만틱 맵 → 사진, 낮 → 밤)이 있을 때, Pix2Pix의 원샷 추론은 지연 시간 측면에서 확산 모델보다 한 자릿수 이상 빠릅니다. 프로덕션 비교는 일반적으로 다음과 같습니다:

| 경로 | 단계 | 단일 L4에서 512² 기준의 일반적인 지연 시간 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net 순방향) | 1 | ~30 ms |
| SD-Inpaint 또는 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL 기본 모델 | 20-30 | ~3-5 s |

Pix2Pix는 정적 배치에서 처리량 측면에서 우세합니다(모든 요청이 동일한 FLOPs를 가짐). 확산 모델은 품질과 일반화 측면에서 우세합니다. 현대적인 전략은 좁은 작업에는 Pix2Pix 스타일의 증류된 모델을 출시하고, 꼬리 입력에는 확산 모델 폴백을 사용하는 경우가 많습니다.

## 추가 읽기

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 논문입니다.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix입니다.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN입니다.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD입니다.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN입니다.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 투사 D입니다.
