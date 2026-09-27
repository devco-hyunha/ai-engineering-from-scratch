# 비디오 생성 (Video Generation)

> 이미지는 2D 텐서입니다. 비디오는 3D 텐서입니다. 이론은 동일하지만, 연산량은 10~100배 더 어렵습니다. OpenAI의 Sora(2024년 2월)가 이것이 가능하다는 것을 증명했습니다. 2026년까지 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, 그리고 WAN 2.2는 텍스트로부터 1080p급 프로덕션 비디오를 생성하여 출시할 예정이며, 오픈 웨이트(open-weights) 스택(CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2)은 약 12개월 뒤처져 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 문제점 (The Problem)

24fps로 촬영된 10초 분량의 1080p 비디오는 1920×1080×3 픽셀로 구성된 240개의 프레임입니다. 이는 클립당 약 1.5GB의 원시 데이터(raw data)에 해당합니다. 따라서 픽셀 공간(pixel-space)에서의 디퓨전은 실행이 불가능합니다. 다음과 같은 요소들이 필요합니다:

1. **시공간 압축 (Spatiotemporal compression).** 프레임 단위가 아닌 비디오를 시공간 패치(spatiotemporal patches) 시퀀스로 인코딩하는 VAE가 필요합니다.
2. **시간적 일관성 (Temporal coherence).** 프레임들은 수 초 동안 콘텐츠, 조명, 객체의 정체성을 공유해야 합니다. 네트워크는 움직임을 모델링할 수 있어야 합니다.
3. **연산 예산 (Compute budget).** 비디오 학습은 동일한 모델 크기 기준으로 이미지보다 10~100배 더 많은 비용이 발생합니다.
4. **컨디셔닝 (Conditioning).** 텍스트, 이미지(첫 프레임), 오디오 또는 다른 비디오를 입력으로 받습니다. 대부분의 상용 모델은 이 네 가지를 모두 수용합니다.

이 문제를 해결한 아키텍처는 거대한 (프롬프트, 캡션, 비디오) 데이터셋으로 학습된, 시공간 패치에 적용된 **Diffusion Transformer (DiT)**입니다. 디퓨전 손실(diffusion loss) 방식은 Lesson 06과 동일합니다.

## 개념 (The Concept)

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### 패치화 (Patchify)

3D VAE(학습된 시공간 압축)를 사용하여 비디오를 인코딩합니다. 잠재 표현(latent)의 형태는 `[T_latent, H_latent, W_latent, C_latent]`입니다. 이를 `[t_p, h_p, w_p]` 크기의 패치로 분할합니다. Sora 스타일의 모델의 경우, `t_p = 1`(프레임당 패치) 또는 `t_p = 2`(두 프레임마다 하나)를 사용합니다. 10초 분량의 1080p 비디오는 약 20,000~100,000개의 패치로 압축됩니다.

### 시공간 DiT (Spatiotemporal DiT)

트랜스포머는 패치(patch)의 평탄화된 시퀀스를 처리합니다. 각 패치는 3D 위치 임베딩(time + y + x)을 가집니다. 어텐션(Attention)은 일반적으로 다음과 같이 분해됩니다:

- **공간 어텐션 (Spatial attention)**: 각 프레임 내의 패치들 사이에서 수행됩니다.
- **시간 어텐션 (Temporal attention)**: 동일한 공간 위치에 있는 프레임들 사이에서 수행됩니다.
- **전체 3D 어텐션 (Full 3D attention)**: 비용이 16~100배 더 많이 소요됩니다. 저해상도 작업이나 연구 목적으로만 사용됩니다.

### 텍스트 컨디셔닝 (Text conditioning)

대규모 텍스트 인코더(Sora의 경우 T5-XXL, CogVideoX-5B는 T5-XXL 사용)를 활용한 교차 주의 집중(Cross-attention) 방식입니다. 긴 프롬프트가 중요합니다. Sora의 학습 데이터셋에는 클립당 평균 200개 토큰에 달하는 GPT 생성 고밀도 재캡션(re-captions)이 포함되어 있었습니다.

### 학습 (Training)

시공간 잠재 변수(spatiotemporal latents)에 대한 표준 확산 손실(standard diffusion loss, $\epsilon$ 또는 $v$ 예측)을 사용합니다. 데이터: 웹 비디오 + 약 1억 개의 큐레이션된 클립 + 합성 텍스트 캡션(synthetic text captions). 연산량: 소규모 연구 실행에도 10,000시간 이상의 GPU 시간이 소요되며, Sora 규모의 학습에는 100,000시간 이상이 필요합니다.

## 2026년 프로덕션 환경 (The 2026 production landscape)

| 모델 (Model) | 날짜 (Date) | 최대 길이 (Max duration) | 최대 해상도 (Max res) | 오픈 웨이트 여부 (Open weights?) | 특징 (Notable) |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | 대규모 세계 시뮬레이터 특성을 보여준 최초의 모델 |
| Sora Turbo | 2024-12 | 20s | 1080p | No | 5배 빠른 추론 속도를 가진 프로덕션용 Sora |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | 2025년 기준 최고 품질 및 물리 엔진 성능 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | 네이티브 오디오 및 강력한 카메라 제어 기능 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | 2025년 1분기 기준 최고의 인체 움직임 구현 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | 전문적인 비디오 편집 도구 결합 |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | 가장 강력한 캐릭터 일관성 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | 최초의 오픈 5B 규모 비디오 모델 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | 2024년 말 기준 오픈 SOTA(State-of-the-Art) |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | 가장 허용적인 라이선스 정책 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | 2025년 중반 기준 가장 강력한 오픈 모델 |

오픈 웨이트(Open weights) 모델들이 이미지 분야보다 더 빠르게 격차를 줄이고 있습니다. HunyuanVideo와 WAN 2.2 LoRA는 2026년 중반에 이미 대부분의 오픈 소스 워크플로우를 주도하고 있습니다.

```figure
video-diffusion-denoise
```

## 구축하기 (Build It)

`code/main.py`는 핵심적인 시공간 DiT(spatiotemporal DiT) 아이디어를 시뮬레이션합니다. 작은 합성 비디오를 패치화(patchify)하고, 패치별 위치 임베딩(position embedding)을 추가한 뒤, 패치 간 트랜스포머 스타일의 어텐션(attention)을 통해 전체 시퀀스를 디노이징(denoise)합니다. `numpy`를 사용하지 않는 순수 파이썬(pure Python) 구현입니다. 인접한 프레임의 패치들이 디노이저와 위치 임베딩을 공유할 때, 1차원(1-D) 환경에서도 시간적 일관성(temporal coherence)이 나타남을 보여줍니다.

### 1단계: 합성 1차원 "비디오(video)" 패치화(patchify)하기

```python
def make_video(T_frames=8, rng=None):
    # "비디오"는 부드러운 궤적을 따르는 1차원 값들의 시퀀스입니다.
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 2단계: 프레임당 위치 임베딩 (position embedding per frame)

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 3단계: 전체 시퀀스를 확인하는 디노이저 (denoiser sees the whole sequence)

각 프레임을 독립적으로 디노이징하는 대신, 우리의 작은 네트워크(tiny net)는 모든 프레임 값과 위치 임베딩(position embeddings)을 결합(concatenate)하여 모든 프레임에 대한 노이즈를 공동으로 예측합니다.

### 4단계: 시간적 일관성 테스트 (Temporal Coherence Test)

학습이 완료된 후, 비디오를 샘플링합니다. 프레임 간의 변화량(delta)을 측정해 보세요. 모델이 시간적 구조(temporal structure)를 학습했다면, 각 프레임을 독립적으로 샘플링했을 때보다 프레임 간 변화량이 더 작게 유지됩니다.

## 주의 사항 (Pitfalls)

- **프레임별 독립적 샘플링 = 깜빡임(flicker).** 각 프레임에 대해 이미지 확산(image diffusion)을 개별적으로 실행하면, 각 프레임의 노이즈가 독립적이기 때문에 출력 결과가 깜빡거립니다. 비디오 확산(video diffusion)은 어텐션(attention)이나 공유된 노이즈를 통해 프레임들을 결합함으로써 이 문제를 해결합니다.
- **단순한 3D 어텐션(Naive 3D attention) = 메모리 부족(OOM).** 10초 분량의 1080p 잠재 공간(latent)에 대해 전체 3D 어텐션을 적용하면 수천억 번의 연산이 필요합니다. 이를 공간(spatial) + 시간(temporal) 단위로 분해(factorize)하세요.
- **데이터 캡셔닝(Data captioning)은 크기보다 중요합니다.** Sora가 이전 작업들에 비해 얻은 주요 업그레이드는 약 10배 더 상세한 캡션(GPT-4로 재라벨링된 클립)으로 학습했다는 점입니다. OpenAI의 기술 보고서는 이 점을 명시하고 있습니다.
- **첫 프레임 조건화(First-frame conditioning).** 대부분의 프로덕션 모델은 첫 프레임으로 이미지를 입력받을 수도 있습니다. 이는 "이미지-투-비디오(image-to-video)" 모드이며, 학습 과정에 이 변형 방식이 포함됩니다.
- **물리 법칙 드리프트(Physics drift).** 긴 클립(>10초)은 미세한 불일치가 누적됩니다. 슬라이딩 윈도우 생성(Sliding-window generation)과 키프레임 앵커링(keyframe anchoring)이 도움이 됩니다.

## 활용 방법 (Use It)

| 활용 사례 (Use case) | 2026년 추천 (2026 pick) |
|----------|-----------|
| 최고 품질의 텍스트-비디오(text-to-video), 호스팅형 | Veo 3 또는 Sora |
| 카메라 제어가 가능한 시네마틱 영상 | 모션 브러시(motion brushes)를 포함한 Runway Gen-3 |
| 클립 간 캐릭터 일관성 유지 | Pika 2.0 또는 Kling 2.1 |
| 오픈 웨이트(Open weights), 빠른 미세 조정(fine-tune) | WAN 2.2 + LoRA |
| 이미지-비디오(Image-to-video) | WAN 2.2-I2V, Kling 2.1 I2V 또는 Runway |
| 오디오-비디오(Audio-to-video) 립싱크 | Veo 3 (네이티브 오디오) 또는 전용 립싱크 모델 |
| 비디오 편집 | Runway Act-Two, Kling Motion Brush, Flux-Kontext (스틸 프레임) |

동일 품질 기준 비디오 초당 비용이 2024년과 2026년 사이에 20배 하락했습니다.

## Ship It (실행하기)

`outputs/skill-video-brief.md`를 저장하세요. 이 스킬은 비디오 브리프(길이, 종횡비, 스타일, 카메라 계획, 피사체 일관성, 오디오)를 입력받아 다음을 출력합니다: 모델 + 호스팅, 프롬프트 스캐폴딩(카메라 언어, 피사체 묘사, 동작 기술어), 시드 + 재현성 프로토콜, 그리고 프레임 단위 QA 체크리스트.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에서 (a) 프레임별 독립 샘플링(independent per-frame sampling)과 (b) 결합 시퀀스 샘플링(joint sequence sampling)에 대한 프레임 간 델타(frame-to-frame delta)를 비교해 보세요. 델타의 평균과 분산을 보고하세요.
2. **중간 (Medium).** 첫 번째 프레임 조건(first-frame condition)을 추가해 보세요: 0번 프레임을 특정 값으로 고정(pin)하고 나머지를 샘플링합니다. 고정된 값이 어떻게 전파되는지 측정해 보세요.
3. **어려움 (Hard).** HuggingFace `diffusers`를 사용하여 로컬 GPU에서 `CogVideoX-2B`를 실행해 보세요. 6초 분량의 720p 클립에 대해 20단계의 추론(inference steps) 시간을 측정하세요. 시공간 어텐션(spatiotemporal attention)을 프로파일링하여 병목 지점(bottleneck)을 찾아보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | `(T, H, W, C)`를 시공간 잠재 공간(spatiotemporal latent)으로 압축하는 인코더. |
| Patches | "토큰(The tokens)" | 잠재 공간의 고정된 크기를 가진 3-D 블록; DiT의 입력값. |
| Factorized attention | "공간 + 시간(Spatial + temporal)" | 전체 3-D 어텐션을 건너뛰고, 공간에 대해 어텐션을 수행한 뒤 시간에 대해 수행하는 방식. |
| Image-to-video (I2V) | "이 사진을 애니메이션으로 만들어줘" | 모델이 이미지와 텍스트를 입력받아, 해당 이미지에서 시작되는 비디오를 출력하는 방식. |
| Keyframe conditioning | "앵커 프레임(Anchor frames)" | 비디오의 흐름을 제어하기 위해 특정 프레임을 고정하는 방식. |
| Motion brush | "방향성 힌트(Directional hint)" | 사용자가 이미지 위에 모션 벡터를 그리는 UI 입력 방식. |
| Re-captioning | "상세 캡션(Dense captions)" | LLM을 사용하여 학습 클립을 상세한 프롬프트로 다시 라벨링하는 작업. |
| Flicker | "시간적 아티팩트(Temporal artifact)" | 프레임 간의 불일치 현상; 결합된 디노이징(coupled denoising)으로 해결 가능. |

## 프로덕션 노트: 비디오 잠재 변수(video latents)는 메모리 대역폭 문제입니다

24 fps로 촬영된 10초 분량의 1080p 클립은 240 프레임 × 1920 × 1080 × 3 ≈ 약 1.5 GB의 원본 픽셀 데이터를 가집니다. 4배 비디오 VAE 압축(`2 × 공간(spatial) × 2 × 시간(temporal)`)을 거치면 요청당 잠재 변수(latent)는 약 100 MB가 됩니다. 이를 배치 크기 1로 30단계 동안 시공간 DiT(spatiotemporal DiT)로 실행하면, 단계당 약 3 GB의 데이터가 HBM을 통과하게 됩니다. 즉, 병목 지점은 FLOPs가 아니라 메모리 대역폭(memory bandwidth)입니다.

프로덕션 추론(production-inference) 문헌의 추론 장(inference chapter)에서 발췌한 세 가지 프로덕션 조절 요소(knobs)는 다음과 같습니다:

- **DiT 전반에 걸친 텐서 병렬화(TP across the DiT).** 텍스트-비디오(Text-to-video) 모델은 통상적으로 10B 이상의 파라미터를 가집니다. 4개의 H100에서 TP=4를 사용하는 것이 표준이며, 405B급 모델의 경우 PP=2 × TP=2를 사용합니다. 단계당 지연 시간(latency)은 all-reduce 임계값에 도달할 때까지 TP에 따라 대략 선형적으로 감소합니다.
- **프레임 배칭 = 연속 배칭(Frame batching = continuous batching).** 생성 시점에 비디오는 개념적으로 어텐션(attention)으로 연결된 프레임들의 배치입니다. 모델 아키텍처가 슬라이딩 윈도우 생성(sliding-window generation)을 허용한다면, 프레임 `t-1`이 반환되는 동안 프레임 `t+1`의 렌더링을 시작하는 연속 배칭(in-flight scheduling)을 적용할 수 있습니다.
- **클립 수준 프리필 캐시(Clip-level prefill cache).** 이미지-비디오(image-to-video)의 경우, 첫 번째 프레임 조건화(conditioning)는 LLM의 프롬프트 프리필(prompt prefill)과 유사합니다. 이를 한 번만 계산하고 시간적 디코더 패스(temporal decoder passes) 전반에 걸쳐 재사용하세요. 이는 사실상 비디오를 위한 KV-캐시(KV-cache) 역할을 합니다.

## 추가 읽을거리 (Further Reading)

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 기술 보고서.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025년 중반 기준 오픈 소스 SOTA 모델.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 비디오 확산 모델의 선구적인 논문.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion의 모태가 된 연구.
