# 잠재 확산 및 Stable Diffusion

> 512×512 이미지에서의 픽셀 공간 확산은 연산적으로 치명적인 비효율입니다. Rombach 등(2022)은 이미지를 생성하는 데 모든 786k 차원이 필요하지 않으며, 의미적 구조를 포착하는 데 충분한 차원과 나머지 부분을 위한 별도의 디코더가 필요하다는 점을 발견했습니다. VAE의 잠재 공간(latent space) 내에서 확산을 수행하세요. 이 하나의 아이디어가 Stable Diffusion입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 02강 (VAE), 8단계 · 06강 (DDPM), 7단계 · 09강 (ViT)
**시간:** 약 75분

## 문제점

512²에서의 픽셀 공간 확산은 U-Net이 `[B, 3, 512, 512]` 형태의 텐서에서 실행됨을 의미합니다. 500M 매개변수를 가진 U-Net의 경우 각 샘플링 단계는 약 100 GFLOPS입니다. 50단계는 이미지당 5 TFLOPS입니다. 수십억 개의 이미지로 학습하면 연산 비용은 터무니없이 높아집니다.

이 FLOP의 대부분은 손실 VAE가 압축할 수 있는 고주파 텍스처와 같이 지각적으로 중요하지 않은 세부 사항을 네트워크를 통해 전달하는 데 사용됩니다. Rombach의 아이디어는 VAE를 한 번 학습(*1단계*)하고, 이를 동결한 후, 4채널 64×64 잠재 공간(*2단계*)에서 확산을 완전히 수행하는 것입니다. 동일한 U-Net을 사용하며, 픽셀 수는 1/16이고, 비교 가능한 품질을 위해 FLOP는 약 64배 적습니다.

이것이 Stable Diffusion 레시피입니다. SD 1.x / 2.x는 `64×64×4` 잠재 공간에서 860M U-Net을 사용했고, SDXL은 `128×128×4`에서 2.6B U-Net을 사용했으며, SD3는 U-Net을 흐름 매칭(flow matching)을 사용하는 Diffusion Transformer(DiT)로 교체했습니다. Flux.1-dev (Black Forest Labs, 2024)는 12B 매개변수를 가진 DiT-MMDiT를 제공합니다. 모두 동일한 2단계 기반 구조에서 실행됩니다.

## 개념

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**두 단계, 각각 독립적으로 학습.**

1. **1단계 — VAE.** 인코더 `E(x) → z`, 디코더 `D(z) → x`. 목표 압축: 각 공간 축에서 8× 다운샘플링 + 총 잠재 크기가 픽셀 수의 약 1/16이 되도록 채널 조정. 손실 = 재구성 (L1 + LPIPS 지각) + KL (`z`이 너무 가우시안으로 강제되지 않도록 작은 가중치 사용, `z`에서 정확한 샘플링이 필요하지 않기 때문). 디코딩된 이미지가 선명하도록 적대적 손실(adversarial loss)로 학습하는 경우가 많습니다.

2. **2단계 — `z`에서의 확산.** `z = E(x_real)`을 데이터로 취급하세요. U-Net (또는 DiT)을 학습하여 `z_t`를 디노이즈(denoise)하세요. 추론 시: 확산을 통해 `z_0`을 샘플링한 후 `x = D(z_0)`을 수행하세요.

**텍스트 조건부.** 두 개의 추가 구성 요소가 있습니다. 동결된 텍스트 인코더 (SD 1.x용 CLIP-L, SD 2/XL용 CLIP-L+OpenCLIP-G, SD3 및 Flux용 T5-XXL). 교차 어텐션(Cross-Attention) 주입: 모든 U-Net 블록이 `[Q = image features, K = V = text tokens]`를 받아 이를 혼합합니다. 토큰은 텍스트가 이미지에 영향을 미치는 유일한 방법입니다.

**손실 함수는 06강과 동일합니다.** 잡음에 대한 동일한 DDPM / 흐름 매칭 MSE를 사용합니다. 데이터 도메인만 교체하면 됩니다.

## 아키텍처 변형

| 모델 | 연도 | 백본 | 잠재 형상 | 텍스트 인코더 | 매개변수 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 토큰) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 리파이너 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 증류(Distilled) | 128×128×4 | 동일 | 1-4 스텝 샘플링 |
| SD3 | 2024 | MMDiT (멀티모달 DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 증류(Distilled) | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 스텝 |

추세: U-Net을 DiT (잠재 패치에 대한 트랜스포머)로 교체하고, 텍스트 인코더를 확장하며 (프롬프트 준수 측면에서 T5가 CLIP보다 우수함), 잠재 채널 수를 늘립니다 (4 → 16은 더 많은 디테일 여유를 제공합니다).

```figure
noise-schedule
```

## 구현하기

`code/main.py`는 06강의 DDPM 위에 장난감 1-D "VAE" (데모용 인코더 + 디코더; 실제 VAE는 합성곱 신경망)를 쌓고, 분류기 없는 가이드(Classifier-Free Guidance)를 사용하여 클래스 조건부를 추가합니다. 이는 원시 1-D 값에서 실행하든 인코딩된 값에서 실행하든 동일한 확산 손실이 작동한다는 핵심 통찰을 보여줍니다.

### 1단계: 인코더/디코더

```python
def encode(x):    return x * 0.5          # 더 작은 스케일로 "압축"
def decode(z):    return z * 2.0
```

실제 VAE는 학습된 가중치를 가집니다. 교육적 목적을 위해, 이 선형 맵은 확산이 원시 데이터 공간을 신경 쓰지 않고 `z`에서 작동한다는 것을 보여주기 위해 충분합니다.

### 2단계: `z` 공간에서의 확산

06강과 동일한 DDPM입니다. 네트워크가 보는 데이터는 `z = E(x)`입니다. `z_0`을 샘플링한 후 `D(z_0)`로 디코딩합니다.

### 3단계: 분류기 없는 가이드(Classifier-Free Guidance)

학습 중에는 클래스 레이블을 10% 확률로 드롭합니다(null 토큰으로 대체). 추론 시 `ε_cond`과 `ε_uncond`을 모두 계산한 후:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 가이드 없음(전체 다양성), `w = 3` = 기본값, `w = 7+` = 포화 / 과선명.

### 4단계: 텍스트 조건부(concept, not code)

클래스 레이블을 고정된 텍스트 인코더 출력으로 대체합니다. 텍스트 임베딩을 교차 어텐션(Cross-Attention)을 통해 U-Net에 전달합니다:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

이것이 클래스 조건부 확산 모델(Diffusion Model)과 Stable Diffusion 간의 유일한 실질적인 차이입니다.

## 문제점

- **VAE 스케일 불일치.** SD 1.x VAE는 인코딩 후 스케일링 상수(`scaling_factor ≈ 0.18215`)가 적용됩니다. 이를 잊으면 U-Net이 분산이 완전히 잘못된 잠재 변수(latent)로 학습됩니다. 모든 체크포인트(Checkpoint)가 이를 포함하고 있습니다.
- **텍스트 인코더가 조용히 잘못됨.** SD3는 >=128 토큰의 T5-XXL이 필요하며, CLIP 전용으로의 폴백은 손실이 있습니다. 항상 `use_t5=True`을 확인하거나 프롬프트 충실도가 급격히 떨어지는지 점검하세요.
- **잠재 공간(Latent Space) 혼합.** SDXL, SD3, Flux는 모두 다른 VAE를 사용합니다. SDXL 잠재 변수로 학습된 LoRA는 SD3에서 작동하지 않습니다. Hugging Face diffusers 0.30+는 불일치하는 체크포인트를 로드하지 않습니다.
- **CFG가 너무 높음.** `w > 10`은 포화되고 기름진 이미지를 생성하며 다양성 대가로 프롬프트에 과적합합니다. 최적점은 `w = 3-7`입니다.
- **네거티브 프롬프트 누수.** 빈 네거티브 프롬프트는 null 토큰이 되며, 채워진 네거티브 프롬프트는 `ε_uncond`이 됩니다. 이 둘은 같지 않으며, 일부 파이프라인은 조용히 null로 기본 설정합니다.

## 사용하기

2026년 프로덕션 스택:

| 대상 | 권장 백본 |
|--------|----------------------|
| 좁은 도메인, 쌍 데이터, 모델 처음부터 학습 | SDXL 미세 조정(Fine-tuning) (LoRA / 전체) — 가장 빠르게 출시 |
| 오픈 도메인 텍스트-이미지, 오픈 가중치 | Flux.1-dev (12B, Apache / 비상업적) 또는 SD3.5-Large |
| 가장 빠른 추론(Inference), 오픈 가중치 | Flux.1-schnell (1-4 스텝, Apache) 또는 SDXL-Lightning |
| 최고의 프롬프트 준수, 호스트형 | GPT-Image / DALL-E 3 (여전히), Midjourney v7, Imagen 4 |
| 편집 워크플로우 | Flux.1-Kontext (2024년 12월) — 이미지 + 텍스트를 네이티브로 수용 |
| 연구, 기준선 | SD 1.5 — 오래되었지만 잘 연구됨 |

## 출시하기

`outputs/skill-sd-prompter.md`을 저장하세요. 스킬은 텍스트 프롬프트와 대상 스타일을 입력받아 모델 + 체크포인트, CFG 스케일, 샘플러, 네거티브 프롬프트, 해상도, 선택적 ControlNet/IP-Adapter 조합, 그리고 단계별 QA 체크리스트를 출력합니다.

## 연습 문제

1. **쉬움.** `w ∈ {0, 1, 3, 7, 15}`의 가이던스로 `code/main.py`을 실행하세요. 클래스별 평균 샘플을 기록하세요. `w`에서 클래스 평균이 실제 데이터 평균을 얼마나 벗어납니까?
2. **중간.** 장난감 선형 인코더를 재구성 손실(reconstruction loss)을 사용하는 tanh-MLP 인코더/디코더 쌍으로 교체하세요. 새로운 잠재 공간(latent)에서 확산(diffusion)을 다시 학습하세요. 샘플 품질이 변합니까?
3. **어려움.** diffusers를 사용하여 실제 Stable Diffusion 추론을 설정하세요: `sdxl-base`을 로드하고, CFG=7로 30 Euler 단계를 실행하며 시간을 측정하세요. 이제 `sdxl-turbo`로 전환하여 4단계와 CFG=0으로 실행하세요. 같은 주제, 다른 품질 — 무엇이 변했는지와 그 이유를 설명하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 첫 번째 단계 | "VAE" | 학습된 인코더/디코더 쌍; 512²를 64²로 압축합니다. |
| 두 번째 단계 | "U-Net" | 잠재 공간(latent space)에서의 확산 모델입니다. |
| CFG | "가이던스 스케일" | `(1+w)·ε_cond - w·ε_uncond`; 조건부 강도를 조정합니다. |
| Null 토큰 | "빈 프롬프트 임베딩" | `ε_uncond`에 사용되는 무조건부 임베딩입니다. |
| 교차 어텐션(Cross-Attention) | "텍스트가 들어가는 방식" | 각 U-Net 블록은 텍스트 토큰을 K와 V로 어텐션합니다. |
| DiT | "확산 트랜스포머(Diffusion Transformer)" | U-Net을 잠재 패치(latent patches)에 대한 트랜스포머로 교체합니다; 더 잘 확장됩니다. |
| MMDiT | "멀티모달 DiT(Multi-modal DiT)" | SD3의 아키텍처: 텍스트와 이미지 스트림이 결합된 어텐션입니다. |
| VAE 스케일링 팩터 | "매직 넘버" | 잠재(latent)를 약 5.4로 나누어 확산이 단위 분산(unit-variance) 공간에서 작동하도록 합니다. |

## 프로덕션 노트: 8GB 소비자용 GPU에서 Flux-12B 실행하기

참고 Flux 통합은 "소비자용 GPU를 가지고 있는데, 이걸 출시할 수 있나요?"라는 질문에 대한 표준 레시피입니다. 핵심은 프로덕션 추론 문헌이 나열하는 세 가지 조절 변수 레시피를 확산 DiT에 적용하는 것입니다:

1. **단계적 로딩.** Flux는 VRAM에서 동시에 존재할 필요가 없는 세 개의 네트워크를 사용합니다: T5-XXL 텍스트 인코더(fp32에서 약 10 GB), CLIP-L(소형), 12B MMDiT, 그리고 VAE. 프롬프트를 먼저 인코딩하고, *인코더를 삭제*한 후, DiT를 로드하여 디노이즈(denoise)하고, *DiT를 삭제*한 후, VAE를 로드하여 디코딩합니다. 소비자용 8GB GPU는 한 번에 한 단계만 수용할 수 있습니다.
2. **bitsandbytes를 통한 4비트 양자화.** T5 인코더와 DiT 모두에 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`를 적용합니다. 메모리를 8배 줄이며, Aritra의 벤치마크(노트북에 링크됨)에 따르면 텍스트-이미지 생성에서의 품질 저하는 지각할 수 없을 정도입니다.
3. **CPU 오프로드.** `pipe.enable_model_cpu_offload()`는 각 순방향 전달(forward pass)이 진행됨에 따라 모듈을 CPU와 GPU 간에 자동으로 교체합니다. 지연 시간을 10-20% 증가시키지만, 파이프라인이 실행될 수 있게 합니다.

메모리 계산은 다음과 같습니다: `10 GB T5 / 8 = 1.25 GB` 양자화, `12 B params × 0.5 bytes = ~6 GB` 양자화된 DiT, 그리고 활성값(activations). stas00의 용어로 이는 TP=1 추론의 극단적인 형태입니다 — 모델 병렬화 없음, 최대 양자화. 프로덕션 환경에서는 H100에서 TP=2 또는 TP=4를 실행할 것입니다. 단일 개발자 노트북의 경우, 이것이 레시피입니다.

## 추가 읽기

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 계열.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 위의 모든 체크포인트에 대한 참조 구현.
