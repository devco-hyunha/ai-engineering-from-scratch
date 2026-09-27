# 잠재 확산 모델 & 스테이블 디퓨전 (Latent Diffusion & Stable Diffusion)

> 512×512 이미지에 대해 픽셀 공간(Pixel-space)에서 확산을 수행하는 것은 계산적 낭비가 너무 심합니다. Rombach et al. (2022)은 이미지를 생성하기 위해 786k개의 모든 차원이 필요하지 않다는 점에 주목했습니다. 즉, 의미적 구조(semantic structure)를 포착할 수 있을 만큼의 차원만 있으면 되며, 나머지는 별도의 디코더가 처리하면 됩니다. VAE의 잠재 공간(latent space) 내부에서 확산을 실행하십시오. 그 하나의 아이디어가 바로 스테이블 디퓨전(Stable Diffusion)입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## 문제점 (The Problem)

512² 해상도의 픽셀 공간(Pixel-space) 확산 모델은 U-Net이 `[B, 3, 512, 512]` 형태의 텐서에서 동작함을 의미합니다. 5억 개의 파라미터를 가진 U-Net의 경우, 샘플링 단계당 약 100 GFLOPS가 소요됩니다. 50단계를 거치면 이미지 한 장당 5 TFLOPS가 필요합니다. 10억 장의 이미지로 학습한다면 계산 비용은 터무니없이 높아집니다.

이러한 FLOPs의 대부분은 지각적으로 중요하지 않은 세부 사항, 즉 손실 압축 VAE(lossy VAE)가 압축하여 제거할 수 있는 고주파 텍스처를 네트워크를 통해 전달하는 데 사용됩니다. Rombach의 아이디어는 다음과 같습니다. VAE를 한 번 학습시키고(*1단계*), 이를 고정한 뒤, 확산 과정을 완전히 4채널 64×64 잠재 공간(latent space)에서 수행하는 것입니다(*2단계*). U-Net은 동일하게 사용하지만, 픽셀 수는 1/16로 줄어듭니다. 결과적으로 유사한 품질을 유지하면서도 FLOPs는 약 64배 적게 사용합니다.

이것이 바로 Stable Diffusion의 레시피입니다. SD 1.x / 2.x는 `64×64×4` 잠재 공간 위에서 860M U-Net을 사용했고, SDXL은 `128×128×4` 잠재 공간 위에서 2.6B U-Net을 사용했습니다. SD3는 U-Net을 플로우 매칭(flow matching) 기반의 Diffusion Transformer (DiT)로 교체했습니다. Flux.1-dev (Black Forest Labs, 2024)는 12B 파라미터 규모의 DiT-MMDiT를 탑재하고 있습니다. 이 모든 모델은 동일한 2단계 기반 구조(two-stage substrate) 위에서 동작합니다.

## 개념 (The Concept)

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**두 단계로 구성되며, 각각 별도로 학습됩니다.**

1. **1단계 — VAE.** 인코더 `E(x) → z`, 디코더 `D(z) → x`. 목표 압축률: 각 공간 축(spatial axis)에서 8배 다운샘플링 + 전체 잠재 공간(latent) 크기가 픽셀 수의 약 1/16이 되도록 채널 조정. 손실 함수(Loss) = 재구성 손실(reconstruction; L1 + LPIPS perceptual) + KL(KL divergence; `z`가 너무 가우시안 분포를 따르도록 강제되지 않도록 작은 가중치 부여, `z`로부터의 정확한 샘플링은 필요하지 않기 때문). 종종 디코딩된 이미지가 선명하도록 적대적 손실(adversarial loss)과 함께 학습됩니다.

2. **2단계 — `z`에 대한 확산(diffusion).** `z = E(x_real)`을 데이터로 취급합니다. `z_t`의 노이즈를 제거하도록 U-Net(또는 DiT)을 학습시킵니다. 추론 시에는 확산을 통해 `z_0`를 샘플링한 후, `x = D(z_0)`를 수행합니다.

**텍스트 조건화(Text conditioning).** 두 가지 추가 구성 요소가 있습니다. 동결된(frozen) 텍스트 인코더(SD 1.x의 경우 CLIP-L, SD 2/XL의 경우 CLIP-L+OpenCLIP-G, SD3 및 Flux의 경우 T5-XXL). 교차 주의 집중(cross-attention) 주입: 모든 U-Net 블록은 `[Q = 이미지 특징, K = V = 텍스트 토큰]`을 받아 이들을 혼합합니다. 토큰은 텍스트가 이미지에 영향을 미치는 유일한 방법입니다.

**손실 함수는 Lesson 06과 동일합니다.** 노이즈에 대한 동일한 DDPM / flow matching MSE를 사용합니다. 단지 데이터 도메인(data domain)만 바뀔 뿐입니다.

## 아키텍처 변형 (Architecture variants)

| 모델 (Model) | 연도 (Year) | 백본 (Backbone) | 잠재 공간 형태 (Latent shape) | 텍스트 인코더 (Text encoder) | 파라미터 (Params) |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | 동일 (same) | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

트렌드: U-Net을 DiT(잠재 패치 기반의 트랜스포머)로 교체하고, 텍스트 인코더의 규모를 확장하며(프롬프트 준수 측면에서 T5가 CLIP보다 우수함), 잠재 채널 수를 증가시킵니다(4 → 16으로 확장하여 더 많은 디테일 표현 여유 확보).

```figure
noise-schedule
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 Lesson 06의 DDPM 위에 토이(toy) 1-D "VAE"(시연을 위한 항등 인코더 + 디코더; 실제 VAE는 컨볼루션 네트워크를 사용합니다)를 쌓고, 분류기 없는 가이드(classifier-free guidance)를 통한 클래스 조건화(class conditioning)를 추가합니다. 이는 원본 1-D 값에서 실행하든 인코딩된 값에서 실행하든 동일한 확산 손실(diffusion loss)이 작동한다는 것을 보여주며, 이것이 핵심 통찰(key insight)입니다.

### 1단계: encoder/decoder (인코더/디코더)

```python
def encode(x):
    return x * 0.5          # 작은 스케일로의 장난스러운 "압축"
def decode(z):
    return z * 2.0
```

실제 VAE는 학습된 가중치를 가집니다. 교육적 목적을 위해, 이 선형 매핑(linear map)만으로도 확산 모델(diffusion)이 원래의 데이터 공간을 신경 쓰지 않고 `z` 위에서 동작한다는 것을 보여주기에 충분합니다.

### 2단계: `z`-공간에서의 확산 (diffusion in `z`-space)

레슨 06과 동일한 DDPM을 사용합니다. 네트워크가 보는 데이터는 `z = E(x)`입니다. `z_0`를 샘플링한 후, `D(z_0)`를 통해 디코딩합니다.

### 3단계: classifier-free guidance (CFG)

학습 과정에서 10%의 확률로 클래스 레이블을 누락(null token으로 대체)시킵니다. 추론 시에는 `ε_cond`와 `ε_uncond`를 모두 계산한 후, 다음과 같이 적용합니다:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0`은 가이던스 없음(최대 다양성), `w = 3`은 기본값, `w = 7+`는 포화(saturated) 또는 과도하게 날카로운(over-sharp) 상태를 의미합니다.

### 4단계: 텍스트 컨디셔닝 (text conditioning, 코드가 아닌 개념)

클래스 레이블(class label)을 동결된(frozen) 텍스트 인코더 출력값으로 대체합니다. 텍스트 임베딩을 크로스 어텐션(cross-attention)을 통해 U-Net에 전달합니다:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

이것이 클래스 조건부 확산 모델(class-conditional diffusion model)과 Stable Diffusion 사이의 유일하고 실질적인 차이점입니다.

## 주의 사항 (Pitfalls)

- **VAE 스케일 불일치 (VAE-scale mismatch).** SD 1.x VAE는 인코딩 후 스케일링 상수(`scaling_factor ≈ 0.18215`)를 적용합니다. 이를 누락하면 U-Net이 분산(variance)이 크게 잘못된 잠재 공간(latents)을 바탕으로 학습하게 됩니다. 모든 체크포인트에는 이 상수가 포함되어 있습니다.
- **텍스트 인코더의 조용한 오류 (Text encoder silently wrong).** SD3는 128개 이상의 토큰을 처리할 수 있는 T5-XXL이 필요하며, CLIP만 사용하는 폴백(fallback) 방식은 정보 손실이 발생합니다. `use_t5=True` 설정을 항상 확인하세요. 그렇지 않으면 프롬프트 충실도(prompt fidelity)가 급격히 떨어집니다.
- **잠재 공간 혼용 (Mixing latent spaces).** SDXL, SD3, Flux는 모두 서로 다른 VAE를 사용합니다. SDXL 잠재 공간에서 학습된 LoRA는 SD3에서 작동하지 않습니다. Hugging Face `diffusers` 0.30+ 버전은 일치하지 않는 체크포인트의 로드를 거부합니다.
- **너무 높은 CFG (CFG too high).** `w > 10` 설정은 채도가 과도하고 기름진(oily) 이미지를 생성하며, 다양성을 희생하면서 프롬프트에 과적합(over-fits)됩니다. 가장 적절한 값은 `w = 3-7` 사이입니다.
- **부정 프롬프트 누출 (Negative prompts leaking).** 비어 있는 부정 프롬프트는 null 토큰이 되지만, 내용이 채워진 부정 프롬프트는 `ε_uncond`가 됩니다. 이 둘은 서로 다르며, 일부 파이프라인은 조용히 null 토큰을 기본값으로 사용합니다.

## 활용하기 (Use It)

2026년의 프로덕션 스택(Production stacks):

| 대상 (Target) | 권장 백본 (Recommended backbone) |
|--------|----------------------|
| 좁은 도메인, 쌍을 이룬 데이터(paired data), 모델 처음부터 학습 | SDXL 미세 조정 (LoRA / full) — 가장 빠른 출시 가능 |
| 오픈 도메인 텍스트-이미지(text-to-image), 오픈 웨이트 | Flux.1-dev (12B, Apache / 비상업용) 또는 SD3.5-Large |
| 가장 빠른 추론(inference), 오픈 웨이트 | Flux.1-schnell (1-4 step, Apache) 또는 SDXL-Lightning |
| 최고의 프롬프트 준수(prompt adherence), 호스팅 서비스 | GPT-Image / DALL-E 3 (여전히), Midjourney v7, Imagen 4 |
| 편집 워크플로우 (Edit workflows) | Flux.1-Kontext (2024년 12월) — 이미지 + 텍스트를 기본적으로 수용 |
| 연구, 베이스라인 (Research, baseline) | SD 1.5 — 오래되었지만 연구가 많이 이루어짐 |

## Ship It (실전 적용)

`outputs/skill-sd-prompter.md`를 저장하세요. 이 스킬은 텍스트 프롬프트와 대상 스타일을 입력받아 다음 항목들을 출력합니다: 모델 + 체크포인트, CFG 스케일, 샘플러, 부정 프롬프트(negative prompt), 해상도, 선택 사항인 ControlNet/IP-Adapter 조합, 그리고 단계별 QA 체크리스트입니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 가이드 값 `w ∈ {0, 1, 3, 7, 15}`를 사용하여 `code/main.py`를 실행해 보세요. 클래스별 평균 샘플을 기록하세요. 어떤 `w` 값에서 클래스 평균이 실제 데이터의 평균을 넘어 발산하나요?
2. **중간 (Medium).** 기존의 토이 선형 인코더(toy linear encoder)를 재구성 손실(reconstruction loss)을 사용하는 `tanh-MLP` 인코더/디코더 쌍으로 교체해 보세요. 새로운 잠재 변수(latents)를 사용하여 확산 모델(diffusion)을 다시 학습시키세요. 샘플 품질이 변하나요?
3. **어려움 (Hard).** `diffusers`를 사용하여 실제 Stable Diffusion 추론 환경을 설정해 보세요: `sdxl-base`를 로드하고, CFG=7 설정으로 30번의 Euler 단계를 실행한 뒤 시간을 측정하세요. 이제 4단계와 CFG=0 설정을 사용하는 `sdxl-turbo`로 전환해 보세요. 동일한 피사체에 대해 품질이 어떻게 달라졌는지, 무엇이 왜 변했는지 설명해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 1단계 (First stage) | "VAE" | 학습된 인코더/디코더 쌍; 512²를 64²로 압축합니다. |
| 2단계 (Second stage) | "U-Net" | 잠재 공간(latent space) 상에서 작동하는 확산 모델(Diffusion model)입니다. |
| CFG | "가이던스 스케일(Guidance scale)" | `(1+w)·ε_cond - w·ε_uncond`; 조건부 강도(conditioning strength)를 조절합니다. |
| Null token | "빈 프롬프트 임베딩(Empty prompt embed)" | `ε_uncond`를 위해 사용되는 무조건부(unconditional) 임베딩입니다. |
| 교차 주의 집중 (Cross-attention) | "텍스트가 주입되는 방식" | 각 U-Net 블록이 텍스트 토큰을 K와 V로 참조(attend)합니다. |
| DiT | "Diffusion Transformer" | U-Net을 잠재 패치(latent patches) 상의 트랜스포머로 대체한 모델; 확장성(scaling)이 더 좋습니다. |
| MMDiT | "Multi-modal DiT" | SD3의 아키텍처: 텍스트와 이미지 스트림이 결합된 주의 집중(joint attention)을 사용합니다. |
| VAE 스케일링 인자 (VAE scaling factor) | "마법의 숫자(Magic number)" | 확산 모델이 단위 분산(unit-variance) 공간에서 작동하도록 잠재 변수를 ~5.4로 나눕니다. |

## 프로덕션 노트: 8GB 소비자용 GPU에서 Flux-12B 실행하기

참조용 Flux 통합 방식은 "소비자용 GPU를 가지고 있는데, 이것을 배포할 수 있을까요?"라는 질문에 대한 표준적인 레시피입니다. 핵심은 확산(diffusion) DiT에 적용되는, 프로덕션 추론 문헌에서 언급하는 세 가지 조절 방식(three-knob recipe)과 동일합니다:

1. **순차적 로딩(Staggered loading).** Flux에는 VRAM에 동시에 존재할 필요가 없는 세 가지 네트워크가 있습니다: T5-XXL 텍스트 인코더(fp32 기준 약 10GB), CLIP-L(소형), 12B MMDiT, 그리고 VAE입니다. 먼저 프롬프트를 인코딩한 후, 인코더를 *삭제*하고, DiT를 로드하여 노이즈를 제거한 뒤, DiT를 *삭제*하고, VAE를 로드하여 디코딩하세요. 8GB 소비자용 GPU는 한 번에 하나의 단계만 수용할 수 있습니다.
2. **bitsandbytes를 통한 4비트 양자화(4-bit quantization).** T5 인코더와 DiT 모두에 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`를 적용합니다. 메모리를 8배 절감하며, Aritra의 벤치마크(노트북에 링크됨)에 따르면 텍스트-이미지 생성 시 품질 저하는 감지할 수 없는 수준입니다.
3. **CPU 오프로드(CPU offload).** `pipe.enable_model_cpu_offload()`를 사용하면 각 순전파(forward pass)가 진행됨에 따라 모듈을 CPU와 GPU 사이에서 자동으로 교체합니다. 지연 시간(latency)이 10-20% 증가하지만, 파이프라인을 실행 가능하게 만듭니다.

메모리 계산은 다음과 같습니다: 양자화된 `10 GB T5 / 8 = 1.25 GB`, 양자화된 `12 B params × 0.5 bytes = ~6 GB` DiT, 그리고 활성화 함수(activations) 값입니다. stas00의 용어를 빌리자면, 이는 모델 병렬화가 없는 최대 양자화 상태인 TP=1 추론의 극한 단계입니다. 프로덕션 환경에서는 H100에서 TP=2 또는 TP=4로 실행하겠지만, 단일 개발용 노트북을 위해서는 이 레시피가 정답입니다.

## 추가 읽을거리 (Further Reading)

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 제품군.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 위에 언급된 모든 체크포인트에 대한 참조 구현체.
