# Conditional GANs & Pix2Pix (조건부 GAN 및 Pix2Pix)

> 2014년에서 2017년 사이의 첫 번째 커다란 돌파구는 GAN이 생성하는 내용을 제어하는 것이었습니다. 레이블, 이미지, 또는 문장을 결합해 보세요. Pix2Pix는 이미지 버전을 구현했으며, 여전히 좁은 범위의 이미지 투 이미지(image-to-image) 작업에서는 모든 범용 텍스트 투 이미지(text-to-image) 모델을 능가합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## 문제점 (The Problem)

무조건부 GAN(Unconditional GAN)은 임의의 얼굴을 샘플링합니다. 데모용으로는 유용하지만, 실제 서비스(production) 단계에서는 한계가 있습니다. 여러분이 원하는 것은 다음과 같습니다: *스케치를 사진으로 매핑하기*, *지도를 항공 사진으로 매핑하기*, *낮 장면을 밤 장면으로 매핑하기*, *그레이스케일 이미지를 컬러로 만들기*. 이 모든 경우에서 여러분은 입력 이미지 `x`를 받으며, 어떤 의미론적 대응 관계(semantic correspondence)를 가진 `y`를 출력해야 합니다. 하나의 `x`에 대해 가능한 `y`는 매우 많습니다. 평균 제곱 오차(Mean-squared error)는 이들을 뭉개져서 흐릿하게(mush) 만들어 버립니다. 반면 적대적 손실(Adversarial loss)은 "실제처럼 보이는 것"이 선명하기 때문에 이를 뭉개지 않습니다.

조건부 GAN(Conditional GAN; Mirza & Osindero, 2014)은 조건 `c`를 `G`와 `D` 모두의 입력으로 추가합니다. Pix2Pix(Isola et al., 2017)는 이를 특화했습니다: 조건은 전체 입력 이미지이며, 생성기(generator)는 U-Net이고, 판별기(discriminator)는 *패치 기반(patch-based)* 분류기(PatchGAN)이며, 손실 함수는 적대적 손실 + L1입니다. 이 방식은 *쌍을 이룬 데이터(paired data)*—즉, 정확히 필요한 신호를 가지고 있음—를 통해 학습되기 때문에, 2026년에도 좁은 범위의 이미지 투 이미지(image-to-image) 도메인에서는 처음부터 학습시킨 텍스트 투 이미지(text-to-image) 모델보다 더 뛰어난 성능을 발휘합니다.

## 개념 (The Concept)

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**조건부 생성기 (Conditional G).** `G(x, z) → y`. Pix2Pix에서 `z`는 `G` 내부의 드롭아웃(dropout)입니다 (입력 노이즈는 사용되지 않습니다 — Isola는 명시적인 노이즈가 무시된다는 것을 발견했습니다).

**조건부 판별기 (Conditional D).** `D(x, y) → [0, 1]`. 입력은 (조건, 출력)의 *쌍(pair)*입니다. 이것이 핵심적인 차이점입니다. `D`는 단순히 `y`가 진짜처럼 보이는지를 판단하는 것이 아니라, `y`가 `x`와 일치하는지를 판단해야 합니다.

**U-Net 생성기 (U-Net generator).** 병목 구간(bottleneck)을 가로지르는 스킵 연결(skip connections)을 가진 인코더-디코더 구조입니다. 입력과 출력이 저수준 구조(가장자리, 실루엣 등)를 공유하는 작업에서 매우 중요합니다. 스킵 연결이 없다면 고주파 세부 정보(high-frequency detail)가 사라집니다.

**PatchGAN 판별기 (PatchGAN discriminator).** 단일한 진짜/가짜 점수를 출력하는 대신, `D`는 각 셀이 약 70×70 픽셀의 수용 영역(receptive field)을 판단하는 `N×N` 그리드를 출력합니다. 이후 평균을 냅니다. 이는 현실성이 국소적이라는 마르코프 무작위장(Markov random field) 가정을 따릅니다. 학습 속도가 훨씬 빠르고, 파라미터 수가 적으며, 더 선명한 출력을 생성합니다.

**손실 함수 (Loss).**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 항은 학습을 안정화하고 `G`가 알려진 타겟을 향하도록 유도합니다. L1은 L2보다 더 선명한 가장자리를 제공합니다 (평균이 아닌 중앙값 기반). Pix2Pix의 기본값은 `λ = 100`이었습니다.

## CycleGAN — 쌍(pair) 데이터가 없을 때

Pix2Pix는 쌍을 이루는 `(x, y)` 데이터가 필요합니다. CycleGAN (Zhu et al., 2017)은 추가적인 손실 함수인 *주기 일관성(cycle consistency)* 손실을 사용하는 대신 이 요구 사항을 제거합니다. 두 개의 생성기 `G: X → Y`와 `F: Y → X`를 사용합니다. `F(G(x)) ≈ x`와 `G(F(y)) ≈ y`가 되도록 학습시킵니다. 이를 통해 쌍을 이루는 예시 없이도 말을 얼룩말로, 여름을 겨울로 변환할 수 있습니다.

2026년 현재, 쌍이 없는 이미지 대 이미지(unpaired image-to-image) 변환은 CycleGAN보다는 주로 확산 모델(diffusion; ControlNet, IP-Adapter)을 통해 이루어지지만, 주기 일관성(cycle-consistency) 개념은 거의 모든 비지도 도메인 적응(unpaired domain adaptation) 논문에서 활용되고 있습니다.

```figure
gx-patchgan
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 1차원 데이터에 대한 아주 작은 조건부 GAN(conditional GAN)을 구현합니다. 조건 `c`는 클래스 레이블(0 또는 1)입니다. 작업 목표는 주어진 클래스에 대한 조건부 분포로부터 샘플을 생성하는 것입니다.

### 1단계: G와 D 입력 모두에 조건(condition) 추가하기

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

원-핫 인코딩(One-hot encoding)은 가장 간단한 방법입니다. 더 큰 모델들은 학습된 임베딩(learned embeddings), FiLM 변조(FiLM modulation), 또는 교차 주의 집중(cross-attention) 메커니즘을 사용합니다.

### 2단계: 조건부 학습 (train conditional)

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

생성자(Generator)는 주변 분포(marginal distribution)가 아니라, *주어진 조건(given condition)에 대하여* 실제 분포와 일치해야 합니다.

### 3단계: 클래스별 출력 검증 (verify per-class output)

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 함정 (Pitfalls)

- **조건 무시 (Condition ignored).** 생성자(G)가 조건을 무시하고 주변부 데이터(marginalize)를 학습하게 되며, 조건 신호가 약해 판별자(D)가 이를 전혀 페널티로 처리하지 못합니다. 해결책: 판별자(D)에 더 공격적으로 조건을 부여하세요 (후기 레이어뿐만 아니라 초기 레이어부터 적용). 또는 프로젝션 판별자(projection discriminator, Miyato & Koyama 2018)를 사용해 보세요.
- **L1 가중치가 너무 낮음 (L1 weight too low).** 생성자(G)가 실제와 유사해 보이기만 할 뿐, 입력에 충실하지 않은 임의의 결과물을 출력하도록 표류(drift)합니다. Pix2Pix 스타일의 작업에서는 $\lambda \approx 100$ 정도로 시작해 보세요.
- **L1 가중치가 너무 높음 (L1 weight too high).** L1은 여전히 $L_p$ 노름(norm)이기 때문에 생성자(G)가 흐릿한(blurry) 결과물을 생성합니다. 학습이 안정화되면 가중치를 점진적으로 줄여(anneal down) 보세요.
- **판별자(D)에서의 정답 누출 (Ground-truth leakage in D).** 판별자(D)의 입력으로 `y`만 넣지 말고, `(x, y)`를 결합(concatenate)하여 입력하세요. 이렇게 하지 않으면 판별자(D)가 일관성(consistency)을 확인할 수 없습니다.
- **클래스별 모드 붕괴 (Mode collapse per class).** 각 클래스가 독립적으로 붕괴될 수 있습니다. 클래스 조건부 다양성 검사(class-conditional diversity checks)를 수행해 보세요.

## 활용하기 (Use It)

2026년 이미지 투 이미지(image-to-image) 작업 현황:

| 작업 (Task) | 최적의 접근 방식 (Best approach) |
|------|---------------|
| 스케치 → 사진, 동일 도메인, 쌍을 이룬 데이터 (Sketch → photo, same domain, paired data) | Pix2Pix / Pix2PixHD (여전히 빠르고 선명함) |
| 스케치 → 사진, 쌍을 이루지 않은 데이터 (Sketch → photo, unpaired) | Scribble 조건부 모델을 사용한 ControlNet |
| 시맨틱 세그멘테이션 → 사진 (Semantic seg → photo) | SPADE / GauGAN2 또는 SD + ControlNet-Seg |
| 스타일 전이 (Style transfer) | IP-Adapter 또는 LoRA를 사용한 Diffusion; GAN 방식은 구식(legacy)임 |
| 깊이 → 사진 (Depth → photo) | Stable Diffusion 기반의 ControlNet-Depth |
| 초해상도 (Super-resolution) | Real-ESRGAN (GAN), ESRGAN-Plus, 또는 SD-Upscale (diffusion) |
| 채색 (Colorization) | ColTran, diffusion 기반 채색기, 또는 Pix2Pix-color |
| 낮 → 밤, 계절, 날씨 변화 (Daytime → nighttime, seasons, weather) | CycleGAN 또는 ControlNet 기반 방식 |

Pix2Pix는 (a) 수천 개의 쌍을 이룬(paired) 예시가 있고, (b) 작업 범위가 좁고 반복 가능하며, (c) 빠른 추론(inference)이 필요할 때 여전히 적합한 도구입니다. 일반적인 오픈 도메인(open-domain) 작업에서는 Diffusion 방식이 우세합니다.

## Ship It (실행하기)

`outputs/skill-img2img-chooser.md`를 저장하세요. 이 스킬은 작업 설명(task description), 데이터 가용성(paired vs unpaired, N개 샘플), 그리고 지연 시간/품질 예산(latency/quality budget)을 입력받아 다음을 출력합니다: 접근 방식(Pix2Pix, CycleGAN, ControlNet 변형, SDXL + IP-Adapter), 학습 데이터 요구 사항, 추론 비용, 그리고 평가 프로토콜(LPIPS, FID, 작업 특화 평가).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 수정하여 세 번째 클래스를 추가해 보세요. $G$가 여전히 각 클래스의 노이즈를 올바른 모드(mode)로 매핑하는지 확인합니다.
2. **중간 (Medium).** 1-D 설정에서 $L_1$ 손실을 지각적 스타일 손실(perceptual-style loss)로 교체해 보세요 (예: 특징 추출기 역할을 하는 작은 고정된 $D$ 사용). 이것이 조건부 분포(conditional distribution)의 선명도(sharpness)를 변화시키나요?
3. **어려움 (Hard).** 1-D 설정에서의 CycleGAN을 설계해 보세요: 두 개의 분포, 두 개의 생성기, 그리고 사이클 손실(cycle loss)을 포함합니다. 쌍을 이룬 데이터(paired data) 없이도 두 분포 사이를 매핑하는 법을 학습함을 보여주세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Conditional GAN | "라벨이 있는 GAN" | $G(z, c)$, $D(x, c)$. 두 네트워크 모두 조건(condition)을 참조합니다. |
| Pix2Pix | "이미지-투-이미지 GAN" | U-Net 구조의 $G$와 PatchGAN 구조의 $D$를 사용하며, $L1$ 손실을 결합한 쌍을 이루는(paired) cGAN입니다. |
| U-Net | "스킵 연결이 있는 인코더-디코더" | 대칭형 컨볼루션 네트워크로, 스킵 연결(skip connections)을 통해 고주파 정보를 보존합니다. |
| PatchGAN | "지역적 실사성 분류기" | $D$가 전체 점수 대신 패치(patch)별 점수를 출력합니다. |
| CycleGAN | "비쌍 데이터 이미지 변환" | 두 개의 $G$와 cycle-consistency loss를 사용하며, 쌍을 이루는 데이터가 필요하지 않습니다. |
| SPADE | "GauGAN" | 세그멘테이션 맵을 사용하여 중간 활성화 함수를 정규화하며, 세그멘테이션을 이미지로 변환합니다. |
| FiLM | "특징별 선형 변조(Feature-wise linear modulation)" | 조건으로부터 각 특징(feature)에 대한 아핀 변환(affine transform)을 수행하는 저비용 조건화 방식입니다. |

## 프로덕션 노트: 지연 시간 제한(latency-bound) 베이스라인으로서의 Pix2Pix

쌍을 이룬 데이터(paired data)와 좁은 범위의 작업(스케치 → 렌더링, 시맨틱 맵 → 사진, 낮 → 밤)이 있는 경우, Pix2Pix의 원샷 추론(one-shot inference)은 지연 시간 측면에서 디퓨전(diffusion)보다 한 자릿수(order of magnitude) 더 빠릅니다. 실제 프로덕션에서의 비교는 보통 다음과 같습니다:

| 경로 (Path) | 단계 (Steps) | 단일 L4 GPU 기준 512²에서의 일반적인 지연 시간 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint 또는 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix는 정적 배치(static batches, 모든 요청의 FLOPs가 동일함) 환경에서 처리량(throughput) 측면에서 우세합니다. 반면, 디퓨전은 품질과 일반화(generalization) 측면에서 우세합니다. 현대적인 전략은 종종 좁은 범위의 작업을 위해 Pix2Pix 스타일의 증류된(distilled) 모델을 배포하고, 예외적인 입력(tail inputs)에 대해서는 디퓨전을 폴백(fallback)으로 사용하는 것입니다.

## 추가 읽을거리 (Further Reading)

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 논문입니다.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix입니다.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN입니다.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD입니다.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN입니다.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — Projection Discriminator(projection D)에 관한 논문입니다.
