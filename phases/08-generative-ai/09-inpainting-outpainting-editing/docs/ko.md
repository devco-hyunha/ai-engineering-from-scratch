# 인페인팅, 아웃페인팅 및 이미지 편집 (Inpainting, Outpainting & Image Editing)

> Text-to-image가 새로운 것을 만들어낸다면, Inpainting은 기존의 것을 수정합니다. 실제 실무에서는 이미지 작업 비용의 70%가 편집(배경 교체, 로고 제거, 캔버스 확장, 손 모양 재생성 등)에서 발생합니다. Inpainting은 디퓨전(diffusion) 모델이 그 진가를 발휘하는 영역입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## 문제점 (The Problem)

고객이 배경에 시선을 분산시키는 표지판이 포함된 완벽한 제품 사진을 보냈습니다. 당신은 표지판을 지우고 나머지 모든 부분은 픽셀 단위로 동일하게 유지하고 싶습니다. 처음부터 텍스트-투-이미지(text-to-image)를 실행할 수는 없습니다. 결과물의 색상, 조명, 제품의 각도가 달라질 것이기 때문입니다. 당신은 *오직* 마스크된 영역만 재생성하고 싶으며, 그 재생성이 주변 문맥(context)을 존중하기를 원합니다.

이것이 바로 인페인팅(inpainting)입니다. 변형 방식은 다음과 같습니다:

- **인페인팅(Inpainting).** 마스크 내부를 재생성하고, 마스크 외부의 픽셀은 유지합니다.
- **아웃페인팅(Outpainting).** 마스크 외부(또는 캔버스 너머)를 재생성하고, 마스크 내부를 유지합니다.
- **이미지 편집(Image editing).** 이미지 전체를 재생성하되, 원본의 의미적 또는 구조적 충실도(fidelity)를 유지합니다 (SDEdit, InstructPix2Pix).

2026년의 모든 확산 파이프라인(diffusion pipeline)은 인페인팅 모드를 탑재하여 출시됩니다. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit 등이 그 예입니다. 이들은 모두 동일한 원리로 작동합니다.

## 개념 (The Concept)

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 나이브한 접근 방식 (The naive approach, 그리고 이것이 잘못된 이유)

마스크(mask)를 사용하여 표준 텍스트-이미지(text-to-image) 모델을 실행합니다. 각 샘플링 단계(sampling step)마다, 노이즈가 섞인 잠재 공간(noisy latent)의 마스크되지 않은 영역을 순방향 확산(forward-diffused)된 깨끗한 이미지로 교체합니다. 이 방식은 작동은 하지만... 결과가 좋지 않습니다. 모델이 마스크된 영역 안에 무엇이 있는지에 대한 정보가 없기 때문에 경계 부분에서 아티팩트(boundary artifacts)가 발생하게 됩니다.

### 적절한 인페인팅 모델 (The proper inpainting model)

4개의 채널 대신 9개의 입력 채널을 받는 수정된 U-Net을 학습시킵니다:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

추가된 채널은 VAE로 인코딩된 원본 이미지의 복사본과 단일 채널 마스크로 구성됩니다. 학습 시에는 이미지의 특정 영역을 무작위로 마스킹하고, 마스킹되지 않은 영역은 깨끗한 컨디셔닝 신호(conditioning signal)로 제공되는 동안 모델이 마스킹된 영역만을 디노이징(denoise)하도록 학습합니다. 추론 시 모델은 마스킹된 영역을 둘러싼 환경을 "볼" 수 있으므로 일관성 있는 완성본을 생성할 수 있습니다.

SD-Inpaint, SDXL-Inpaint, Flux-Fill은 모두 이 9채널(또는 유사한 방식) 입력을 사용합니다. Diffusers의 `StableDiffusionInpaintPipeline`, `FluxFillPipeline`이 이를 지원합니다.

### SDEdit (Meng et al., 2022) — 자유로운 편집 (free editing)

소스 이미지에 특정 중간 단계 `t`까지 노이즈를 추가한 다음, 새로운 프롬프트를 사용하여 `t`에서 0까지 역방향 체인(reverse chain)을 실행합니다. 재학습은 필요하지 않습니다. 시작 지점인 `t`의 선택에 따라 충실도(fidelity)와 창의적 자유도(creative freedom) 사이의 균형이 결정됩니다:

- `t/T = 0.3` → 소스와 거의 동일하며, 미세한 스타일 변화만 발생
- `t/T = 0.6` → 중간 정도의 편집, 거친 구조(coarse structure)를 유지
- `t/T = 0.9` → 노이즈에 가까운 상태에서 생성되어, 소스 보존이 최소화됨

### InstructPix2Pix (Brooks et al., 2023)

`(input_image, instruction, output_image)` 세 쌍의 데이터를 사용하여 확산 모델(diffusion model)을 미세 조정(fine-tune)합니다. 추론(inference) 시에는 입력 이미지와 텍스트 지시어("make it sunset", "add a dragon") 모두를 조건(condition)으로 사용합니다. 두 가지 CFG(Classifier-Free Guidance) 스케일이 적용됩니다: 이미지 스케일(image scale)과 텍스트 스케일(text scale).

### RePaint (Lugmayr et al., 2022)

표준적인 무조건부 확산 모델(unconditional diffusion model)을 유지합니다. 각 역과정 단계(reverse step)에서 재샘플링(resample)을 수행합니다. 즉, 가끔 더 노이즈가 많은 상태로 되돌아갔다가 다시 생성하는 방식입니다. 이를 통해 경계면 아티팩트(boundary artifacts)를 방지할 수 있습니다. 별도로 학습된 인페인팅(inpainting) 모델이 없을 때 사용합니다.

```figure
inpaint-mask-reinject
```

## 구현하기 (Build It)

`code/main.py`는 5차원 데이터에 대한 간단한 1차원 인페인팅(inpainting) 방식을 구현합니다. 우리는 각 샘플이 두 개의 클러스터 중 하나에 속하는 5개의 부동 소수점으로 구성된 5차원 혼합 데이터(mixture data)를 사용하여 DDPM을 학습시킵니다. 추론 시에는 5개 차원 중 2개를 "마스킹(mask)"하고, 매 단계마다 마스킹되지 않은 나머지 3개 차원의 노이즈가 섞인 순방향(noisy-forward) 버전을 주입하여 마스킹된 차원만을 재생성합니다.

### 1단계: 5차원 DDPM 데이터 (5-D DDPM data)

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 2단계: 5개 차원 전체에 대해 디노이저(denoiser) 학습

표준 DDPM을 사용합니다. 네트워크는 5차원 노이즈가 섞인 입력에 대해 5차원 노이즈 예측값을 출력합니다.

### 3단계: 추론 시, 마스크 인지 역과정 (at inference, mask-aware reverse)

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 마스크되지 않은 차원을 깨끗한 소스의 새로 노이즈가 추가된 버전으로 교체합니다
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...그 다음 x_t에 대해 일반적인 역과정을 수행합니다
```

이것은 단순한(naive) 접근 방식이며, 간단한 1차원 데이터에서는 작동합니다. 실제 이미지 인페인팅(inpainting)에서는 질감의 일관성(texture coherence)이 더 중요하기 때문에 9채널 입력을 사용합니다.

### 4단계: 아웃페인팅 (Outpainting)

아웃페인팅(Outpainting)은 마스크를 반전시킨 인페인팅(inpainting)입니다. 즉, 새로운(이전에는 존재하지 않았던) 캔버스 영역을 마스크로 지정하고, 나머지 영역을 원본으로 채우는 방식입니다. 학습 목표는 인페인팅과 동일합니다.

## 주의 사항 (Pitfalls)

- **이음새 (Seams).** 단순한 방식은 마스크 경계를 가로질러 그래디언트 정보가 흐르지 않기 때문에 눈에 보이는 경계선을 남깁니다. 해결 방법: 마스크를 8~16픽셀 정도 확장(dilate)하거나, 적절한 인페인팅(inpainting) 모델을 사용해 보세요.
- **마스크 누출 (Mask leakage).** 컨디셔닝 이미지의 마스크되지 않은 영역의 품질이 낮거나 노이즈가 많으면, 마스크 내부의 생성 결과물까지 오염됩니다. 노이즈를 제거하거나 약간 블러(blur) 처리를 해 보세요.
- **CFG와 마스크 크기의 상호작용.** 작은 마스크 영역에 높은 CFG를 적용하면 해당 패치가 포화(saturated)될 수 있습니다. 작은 편집 작업에는 CFG를 낮추어 보세요.
- **SDEdit 충실도 급락 (Fidelity cliff).** `t/T = 0.5`에서 `t/T = 0.6`으로 넘어갈 때 피사체의 정체성(identity)을 잃을 수 있습니다. 값을 탐색(sweep)하며 체크포인트를 확인해 보세요.
- **프롬프트 불일치 (Prompt mismatch).** 프롬프트는 새로운 콘텐츠뿐만 아니라 *전체* 이미지를 묘사해야 합니다. "a cat"이 아니라 "A cat sitting on a chair"와 같이 작성해 보세요.

## 활용 방법 (Use It)

| 작업 (Task) | 파이프라인 (Pipeline) |
|------|----------|
| 객체 제거, 작은 마스크 (Remove object, small mask) | SD-Inpaint 또는 Flux-Fill, 표준 프롬프트 사용 |
| 하늘 교체 (Replace sky) | SD-Inpaint + "blue sky at sunset" |
| 캔버스 확장 (Extend canvas) | SDXL 아웃페인트(outpaint) 모드 (8px 페더링) 또는 아웃페인트 마스크를 사용한 Flux-Fill |
| 손 / 얼굴 재생성 (Regenerate hand / face) | 피사체를 재묘사하는 프롬프트를 사용한 SD-Inpaint + ControlNet-Openpose |
| 특정 영역의 스타일 변경 (Change style of one region) | 마스크 영역에 대해 `t/T=0.5`에서 SDEdit 적용 |
| "노을 지게 만들어줘" ("Make it sunset") | InstructPix2Pix 또는 Flux-Kontext |
| 배경 교체 (Background replacement) | SAM 마스크 → SD-Inpaint |
| 초고정밀도 (Ultra-high-fidelity) | 가장 어려운 케이스의 경우 Flux-Fill 또는 GPT-Image (호스팅 버전) 사용 |

SAM (Meta의 Segment Anything, 2023) + 확산 인페인트(diffusion inpaint)는 2026년형 배경 제거 파이프라인입니다. SAM 2 (2024)는 비디오에서도 작동합니다.

## Ship It (실행하기)

`outputs/skill-editing-pipeline.md`를 저장하세요. 이 스킬은 원본 이미지 + 편집 설명 + 선택적 마스크(또는 SAM 프롬프트)를 입력받아 다음을 출력합니다: 마스크 생성 방식(mask-generation approach), 베이스 모델(base model), CFG 스케일(이미지 + 텍스트), SDEdit-t 또는 인페인팅 모드(inpainting mode), 그리고 QA 체크리스트(QA checklist).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에서 마스킹되는 차원의 비율(fraction of dimensions masked)을 0.2에서 0.8까지 변화시켜 보세요. 인페인팅 품질(마스킹된 차원의 잔차, residual in masked dims)이 무조건부 생성(unconditional generation)과 같아지는 지점의 비율은 얼마인가요?
2. **중간 (Medium).** RePaint를 구현해 보세요: 매 10번째 역과정(reverse step)마다 5단계 뒤로 돌아가(노이즈 추가) 다시 디노이징(re-denoise)을 수행합니다. 이 방식이 마스크 경계면의 경계 잔차(boundary residual)를 줄여주는지 측정해 보세요.
3. **어려움 (Hard).** Hugging Face `diffusers`를 사용하여 다음 두 모델을 20개의 얼굴 재생성(face-regeneration) 작업에서 비교해 보세요: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill. 포즈 준수(pose adherence)와 정체성 보존(identity preservation) 점수를 각각 별도로 산출하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Inpainting (인페인팅) | "구멍 채우기" | 마스크 내부를 재생성하되, 외부 픽셀은 유지합니다. |
| Outpainting (아웃페인팅) | "캔버스 확장하기" | 캔버스 외부를 재생성하되, 내부 픽셀은 유지합니다. |
| 9-channel U-Net (9채널 U-Net) | "제대로 된 인페인팅 모델" | `noisy \| encoded-source \| mask`를 입력으로 받는 U-Net입니다. |
| SDEdit | "노이즈 레벨을 조절한 Img2img" | 노이즈를 시간 `t`만큼 추가한 후, 새로운 프롬프트로 디노이징합니다. |
| InstructPix2Pix | "텍스트만으로 편집하기" | (이미지, 지시어, 결과물) 삼중항 데이터를 통해 미세 조정된 확산 모델입니다. |
| RePaint | "재학습 없는 방식" | 경계선을 줄이기 위해 역과정(reverse) 중에 주기적으로 노이즈를 다시 추가합니다. |
| SAM | "Segment Anything" | 클릭이나 박스로 마스크를 생성하는 생성기이며, 인페인팅과 함께 사용됩니다. |
| Flux-Kontext | "문맥을 활용한 편집" | 편집을 위해 참조 이미지와 지시어를 함께 입력받는 Flux 변형 모델입니다. |

## 프로덕션 노트: 편집 파이프라인은 지연 시간(latency)에 민감합니다

이미지를 편집하는 사용자는 5초 미만의 왕복 시간(round trip)을 기대합니다. 1024² 해상도에서 30단계의 SDXL-Inpaint를 L4 GPU로 실행하면 3~4초가 소요되며, 여기에 SAM 마스크 생성(~200ms)과 VAE 인코딩/디코딩(~500ms 합계) 시간이 추가됩니다. 프로덕션 프레임워크 관점에서 이는 처리량(throughput)보다는 첫 토큰 생성 시간(TTFT, Time To First Token)에 구속되는 문제이며, 배치 크기 1, 낮은 동시성(concurrency) 환경에서 모든 단계를 최소화해야 합니다.

- **SAM-H가 가장 느린 병목 구간입니다.** 1024² 해상도에서 SAM-H는 약 200ms가 소요되지만, SAM-ViT-B는 약간의 품질 저하와 함께 약 40ms가 소요됩니다. SAM 2(비디오용)는 시간적 오버헤드(temporal overhead)를 추가하므로 단일 이미지 편집에는 사용하지 마세요.
- **가능한 경우 인코딩 단계를 건너뛰세요.** `pipe.image_processor.preprocess(img)`는 이미지를 잠재 공간(latents)으로 인코딩합니다. 만약 이전 생성 단계에서 얻은 `latents`가 있다면(반복 편집 UI의 일반적인 사례), 이를 `latents=...`를 통해 직접 전달하여 VAE 인코딩 단계를 하나 건너뛰세요.
- **마스크 확장(Mask dilation)은 처리량에도 영향을 미칩니다.** 마스크가 너무 작으면 U-Net 순전파(forward pass)의 대부분이 낭비됩니다(마스크가 지정되지 않은 픽셀은 어차피 클램핑(clamped)되기 때문입니다). `diffusers`의 `StableDiffusionInpaintPipeline`은 마스크 여부와 관계없이 전체 U-Net을 실행합니다. 오직 9채널 방식의 적절한 인페인트(proper-inpaint) 변형 모델들만이 마스크된 연산만을 활용합니다.
- **Flux-Kontext가 2025년의 해답입니다.** `(source_image, instruction)`에 대해 단 한 번의 순전파만 수행하며, 별도의 마스크나 SDEdit 노이즈 스윕(noise sweep)이 필요 없습니다. H100에서 약 1.5초 만에 편집을 완료합니다. 여기서 얻을 수 있는 아키텍처적 교훈은 단계를 통합(collapse)하라는 것입니다.

## 추가 학습 자료 (Further Reading)

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 별도의 학습이 필요 없는(training-free) 인페인팅.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 텍스트 지시어 기반 편집(text-instruction editing).
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — 마스크 소스인 SAM.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 비디오용 SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 어텐션 레벨 편집(attention-level editing).
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024년 최신 도구.
