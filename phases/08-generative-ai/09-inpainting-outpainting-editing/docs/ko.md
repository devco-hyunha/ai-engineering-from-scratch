# 인페인팅, 아웃페인팅 및 이미지 편집

> 텍스트-이미지 생성은 새로운 것을 만듭니다. 인페인팅은 기존 것을 수정합니다. 실제 생산 환경에서는 청구 가능한 이미지 작업의 70%가 편집입니다 — 배경을 교체하고, 로고를 제거하고, 캔버스를 확장하고, 손을 재생성하는 것 등입니다. 인페인팅은 확산 모델(diffusion)이 그 가치를 증명하는 영역입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 07강 (잠재 확산(Latent Diffusion)), 8단계 · 08강 (ControlNet 및 LoRA)
**시간:** 약 75분

## 문제점

클라이언트가 배경에 방해가 되는 간판이 있는 완벽한 제품 사진을 보내왔습니다. 간판을 지우고 나머지 모든 픽셀은 동일하게 유지해야 합니다. 처음부터 텍스트-이미지 생성을 실행할 수는 없습니다 — 결과가 색상, 조명, 제품 각도가 달라질 것이기 때문입니다. *마스크된 영역만* 재생성해야 하며, 재생성이 주변 컨텍스트를 존중하도록 해야 합니다.

이것이 인페인팅입니다. 변형은 다음과 같습니다:

- **인페인팅.** 마스크 내부에서 재생성하고, 외부 픽셀은 유지합니다.
- **아웃페인팅.** 마스크 외부(또는 캔버스 밖)에서 재생성하고, 내부 픽셀은 유지합니다.
- **이미지 편집.** 전체 이미지를 재생성하되, 원본과의 의미적 또는 구조적 충실도를 유지합니다(SDEdit, InstructPix2Pix).

2026년의 모든 확산 파이프라인은 인페인팅 모드를 제공합니다. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. 이들은 동일한 원리로 작동합니다.

## 개념

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 소박한 접근법(그리고 왜 틀렸는가)

마스크를 사용하여 표준 텍스트-이미지 생성을 실행합니다. 각 샘플링 단계에서 노이즈가 있는 잠재(latent)의 마스크되지 않은 영역을 전방 확산된(clean) 이미지로 대체합니다. 작동합니다... 하지만 형편없습니다. 모델이 마스크된 영역에 대한 정보를 가지고 있지 않기 때문에 경계 아티팩트가 흘러나옵니다.

### 적절한 인페인팅 모델

4개 대신 9개의 입력 채널을 받는 수정된 U-Net을 학습합니다:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

추가 채널은 VAE로 인코딩된 소스 이미지 사본과 단일 채널 마스크입니다. 학습 시에는 이미지의 영역을 랜덤하게 마스킹하고, 모델이 마스킹된 영역만 디노이즈하도록 학습하며, 마스킹되지 않은 영역은 깨끗한 조건 신호로 제공됩니다. 추론 시 모델은 마스킹된 영역 주변을 "볼" 수 있으며, 일관된 완성을 생성합니다.

SD-Inpaint, SDXL-Inpaint, Flux-Fill은 모두 이 9채널(또는 유사한) 입력을 사용합니다. Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — 자유 편집

소스 이미지에 중간 `t`까지 노이즈를 추가한 후, `t`에서 0까지의 역 체인을 새 프롬프트와 함께 실행합니다. 재학습이 필요 없습니다. 시작 `t`의 선택은 충실도와 창의적 자유 사이의 균형을 조정합니다:

- `t/T = 0.3` → 소스와 거의 동일하며, 작은 스타일적 변화만 발생
- `t/T = 0.6` → 중간 수준의 편집, 거친 구조는 보존
- `t/T = 0.9` → 거의 노이즈에서 생성되며, 소스 보존은 최소화

### InstructPix2Pix (Brooks et al., 2023)

`(input_image, instruction, output_image)` 삼중 쌍(triples)으로 확산 모델을 미세 조정합니다. 추론 시 입력 이미지와 텍스트 지시문("일몰로 만들어", "드래곤을 추가해") 모두를 조건으로 사용합니다. 두 개의 CFG 스케일: 이미지 스케일과 텍스트 스케일.

### RePaint (Lugmayr et al., 2022)

표준 무조건 확산 모델을 유지합니다. 각 역 단계에서 리샘플링을 수행하며, 가끔 더 노이즈가 많은 상태로 돌아가 재생성합니다. 경계 아티팩트를 피합니다. 학습된 인페인팅 모델이 없을 때 사용합니다.

```figure
inpaint-mask-reinject
```

## 구현하기

`code/main.py`은 5차원 데이터에 대한 장난감(toy) 1-D 인페인팅 방식을 구현합니다. 두 클러스터 중 하나에서 나온 5개의 float로 구성된 샘플인 5-D 혼합 데이터에 DDPM을 학습합니다. 추론 시 5개 차원 중 2개를 "마스킹"하고, 각 단계에서 마스킹되지 않은 3개 차원의 노이즈가 주입된 버전을 주입하며, 마스킹된 차원만 재생성합니다.

### 1단계: 5-D DDPM 데이터

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 2단계: 모든 5개 차원에 대해 디노이저 학습

표준 DDPM. 네트워크는 5-D 노이즈 입력에 대해 5-D 노이즈 예측을 출력합니다.

### 3단계: 추론 시 마스킹을 고려한 역 과정

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 마스킹되지 않은 차원을 깨끗한 소스의 새로 노이즈가 주입된 버전으로 교체
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...그 후 x_t에 대해 정상적인 역 단계를 실행
```

이것은 단순한 접근 방식이며 장난감 수준의 1-D 데이터에서는 작동합니다. 실제 이미지 인페인팅은 텍스처 일관성이 더 중요하므로 9채널 입력을 사용합니다.

### 4단계: 아웃페인팅

아웃페인팅은 마스크가 반전된 인페인팅입니다: 새로운 (이전에 존재하지 않았던) 캔버스 영역을 마스크로 지정하고, 나머지 영역은 원본으로 채웁니다. 학습 목표는 동일합니다.

## 주의할 점

- **이음새.** 단순한 접근 방식은 마스크를 가로지르는 기울기 정보가 흐르지 않아 눈에 띄는 경계선이 남습니다. 해결 방법: 마스크를 8-16픽셀 팽창(dilate)하거나, 적절한 인페인팅 모델을 사용하세요.
- **마스크 누수.** 조건 이미지(conditioning image)의 마스크되지 않은 영역이 저품질이거나 잡음이 많으면, 마스크 내부의 생성 결과에 오염을 일으킵니다. 디노이즈(denoise)하거나 약간 블러(blur) 처리하세요.
- **CFG가 마스크 크기와 상호작용합니다.** 작은 마스크에 높은 CFG를 적용하면 포화된 패치가 됩니다. 작은 편집에는 CFG를 낮추세요.
- **SDEdit 충실도 급락.** `t/T = 0.5`에서 `t/T = 0.6`로 변경하면 피사체의 정체성이 사라질 수 있습니다. 스윕(sweep)하고 체크포인트를 저장하세요.
- **프롬프트 불일치.** 프롬프트는 *전체* 이미지를 설명해야 하며, 새 내용만 설명해서는 안 됩니다. "a cat"이 아니라 "A cat sitting on a chair"로 작성하세요.

## 사용하기

| 작업 | 파이프라인 |
|------|----------|
| 객체 제거, 작은 마스크 | SD-Inpaint 또는 Flux-Fill, 표준 프롬프트 |
| 하늘 교체 | SD-Inpaint + "blue sky at sunset" |
| 캔버스 확장 | SDXL 아웃페인팅 모드(8px feather) 또는 Flux-Fill의 아웃페인팅 마스크 |
| 손/얼굴 재생성 | 피사체를 다시 설명하는 프롬프트를 사용한 SD-Inpaint + ControlNet-Openpose |
| 한 영역의 스타일 변경 | 마스크된 영역에서 `t/T=0.5`로 SDEdit |
| "Make it sunset" | InstructPix2Pix 또는 Flux-Kontext |
| 배경 교체 | SAM 마스크 → SD-Inpaint |
| 초고충실도 | 가장 어려운 경우 Flux-Fill 또는 GPT-Image(호스트형) |

SAM(Meta의 Segment Anything, 2023) + 확산 인페인팅은 2026년 배경 제거 파이프라인입니다. SAM 2(2024)는 비디오에서 작동합니다.

## 출시하기

`outputs/skill-editing-pipeline.md`을 저장하세요. 스킬은 원본 이미지 + 편집 설명 + 선택적 마스크(또는 SAM 프롬프트)를 입력받아 마스크 생성 접근 방식, 기본 모델, CFG 스케일(이미지 + 텍스트), SDEdit-t 또는 인페인팅 모드, QA 체크리스트를 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 가려진 차원의 비율을 0.2에서 0.8까지 변화시켜 보세요. 가려진 차원의 잔차가 무조건적 생성과 같아지는 비율은 어디입니까?
2. **중간.** RePaint를 구현해 보세요. 10번째 역방향 단계마다 5단계 뒤로 돌아가(잡음을 추가하고) 다시 디노이즈합니다. 마스크 가장자리에서 경계 잔차가 감소하는지 측정해 보세요.
3. **어려움.** Hugging Face diffusers를 사용하여 SD 1.5 Inpaint + ControlNet-Openpose와 Flux.1-Fill을 20개의 얼굴 재생성 작업에서 비교해 보세요. 자세 준수도와 정체성 보존도를 각각 점수화해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Inpainting | "구멍 채우기" | 마스크 내부에서 재생성하고 외부 픽셀은 유지합니다. |
| Outpainting | "캔버스 확장" | 캔버스 외부에서 재생성하고 내부는 유지합니다. |
| 9채널 U-Net | "적절한 Inpainting 모델" | 입력으로 `noisy \| encoded-source \| mask`를 사용하는 U-Net입니다. |
| SDEdit | "잡음 수준이 있는 Img2img" | 잡음을 시간 `t`로 변환하고, 새 프롬프트로 디노이즈합니다. |
| InstructPix2Pix | "텍스트 전용 편집" | (이미지, 지시문, 출력) 삼중 쌍에 대해 미세 조정된 확산 모델입니다. |
| RePaint | "재학습 없음" | 역방향 과정에서 주기적으로 재잡음을 추가하여 이음새를 줄입니다. |
| SAM | "Segment Anything" | 클릭이나 박스로 생성하는 마스크 생성기; Inpainting과 짝을 이룹니다. |
| Flux-Kontext | "컨텍스트로 편집" | 편집을 위해 참조 이미지와 지시문을 허용하는 Flux 변형입니다. |

## 프로덕션 노트: 편집 파이프라인은 지연 시간에 민감합니다

이미지를 편집하는 사용자는 5초 미만의 왕복 시간을 기대합니다. 1024² 해상도의 30단계 SDXL-Inpaint는 L4에서 3-4초가 걸리며, SAM 마스크 생성(~200ms)과 VAE 인코딩/디코딩(~500ms 합계)이 추가됩니다. 프로덕션 관점에서 이는 처리량(bound)이 아니라 TTFT(bound)에 제한됩니다. 배치 1, 낮은 동시성으로 모든 단계를 최소화해야 합니다:

- **SAM-H가 느린 부분입니다.** SAM-H는 1024²에서 ~200ms가 걸립니다. SAM-ViT-B는 ~40ms로 품질 손실이 약간 있습니다. SAM 2(비디오)는 시간적 오버헤드를 추가하므로 단일 이미지 편집에는 사용하지 마세요.
- **가능하면 인코딩을 건너뛰세요.** `pipe.image_processor.preprocess(img)`는 라텐트로 인코딩합니다. 이전 생성에서 라텐트를 가지고 있다면 (반복 편집 UI에서 흔한 경우), `latents=...`를 통해 직접 전달하여 VAE 인코딩 한 번을 건너뛰세요.
- **마스크 팽창은 처리량에도 영향을 미칩니다.** 작은 마스크는 U-Net 순방향 전파의 대부분이 낭비됨을 의미합니다 (마스크되지 않은 픽셀은 어차피 고정됩니다). `diffusers`의 `StableDiffusionInpaintPipeline`는 전체 U-Net을 실행합니다. 마스크된 연산을 활용하는 것은 9채널 proper-inpaint 변형뿐입니다.
- **Flux-Kontext는 2025년의 해답입니다.** `(source_image, instruction)`에 대한 단일 순방향 전파로, 별도의 마스크도 SDEdit 노이즈 스윕도 없습니다. H100에서는 약 1.5초 만에 편집을 완료합니다. 아키텍처적 교훈: 단계를 통합하세요.

## 추가 읽기

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 학습이 필요 없는 인페인팅.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 텍스트 지시 편집.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, 마스크 소스.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 비디오 SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 어텐션 수준 편집.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024년 도구.
