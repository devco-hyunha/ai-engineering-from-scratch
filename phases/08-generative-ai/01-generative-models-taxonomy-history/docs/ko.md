# 생성 모델 — 분류 및 역사

> 모든 이미지 모델, 텍스트 모델, 비디오 모델, 3D 모델은 다섯 가지 범주 중 하나에 속합니다. 범주를 잘못 선택하면 수학적 문제로 몇 주간 고생하게 됩니다. 올바른 범주를 선택하면 지난 12년간의 분야 발전이 머릿속에 깔끔하게 정리됩니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 2단계 (ML 기초), 3단계 (딥러닝 핵심), 7단계 · 14단계 (트랜스포머)
**시간:** 약 45분

## 문제점

생성 모델은 단 하나의 일을 수행합니다. 알려지지 않은 분포 `p_data(x)`에서 추출된 학습 샘플을 입력으로 받아, 동일한 분포에서 나온 것처럼 보이는 새로운 샘플을 출력하는 것입니다. 얼굴, 문장, MIDI 파일, 단백질 구조 — 모두 같은 문제입니다.

문제는 `p_data`이 수백만 차원의 공간에 존재한다는 점입니다 (512x512 RGB 이미지는 약 786k 차원). 샘플은 그 공간 내의 얇은 다양체(manifold) 위에 위치하며, 사용 가능한 예시는 최대 10M 정도입니다. 밀도를 브루트포스 방식으로 계산하는 것은 불가능합니다. 모든 생성 모델은 어려운 문제를 조금 덜 어려운 문제로 교환하는 절충안입니다.

지난 12년간 다섯 가지 계열이 살아남았습니다. 각 계열이 어떤 절충을 하는지 알면, 특정 작업에서 성공하고 다른 작업에서 실패하는 이유를 이해할 수 있습니다.

## 개념

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. 명시적 밀도, 계산 가능.** `log p(x)`을 실제로 평가할 수 있는 합으로 표현합니다. 자기회귀 모델 (PixelCNN, WaveNet, GPT)은 `p(x) = ∏ p(x_i | x_<i)`을 분해합니다. 정규화 플로우 (RealNVP, Glow)는 단순한 기본 분포의 가역 변환으로 `p(x)`을 구성합니다. 장점: 정확한 우도, 깔끔한 학습 손실. 단점: 자기회귀 추론은 순차적 (긴 시퀀스에 느림), 플로우에는 가역 아키텍처가 필요 (아키텍처적 제약이 있음).

**2. 명시적 밀도, 근사적.** `log p(x)`을 하한으로 바운드 (ELBO)하고 바운드를 최적화합니다. VAE (Kingma 2013)는 변분 사후 분포를 사용하는 인코더-디코더를 활용합니다. 확산 모델 (DDPM, Ho 2020)은 가중치가 적용된 ELBO를 암묵적으로 최적화하는 디노이저를 학습합니다. 확산 모델은 2026년 현재 이미지, 비디오, 3D 백본의 지배적인 기술입니다.

**3. 암시적 밀도.** 밀도를 완전히 생략합니다. 샘플을 생성하는 생성기 `G(z)`와 진짜와 가짜를 구분하는 판별기 `D(x)`를 학습합니다. GAN (생성적 적대 신경망)(GAN (Generative Adversarial Network)) (Goodfellow 2014). 추론 시는 빠릅니다 (한 번의 순방향 패스) 하지만 학습 중에는 악명 높게 불안정합니다. StyleGAN 1/2/3는 2026년에도 고정된 영역의 사진적 사실주의 (얼굴, 침실)에서 여전히 최첨단입니다.

**4. 점수 기반 / 연속 시간.** 로그 밀도 `∇_x log p(x)`의 기울기 (점수)를 직접 학습합니다. Song & Ermon (2019)은 점수 매칭이 확산을 SDE로 일반화함을 보였습니다. Flow matching (Lipman 2023)은 2024-2026년의 핫 이슈입니다: 시뮬레이션 없는 학습, 더 직선적인 경로, DDPM보다 4-10배 빠른 샘플링. Stable Diffusion 3, Flux, AudioCraft 2는 모두 flow matching을 사용합니다.

**5. 이산 코드에 대한 토큰 기반 자기회귀(Autoregressive).** VQ-VAE나 잔차 양자화기를 사용하여 고차원 데이터를 짧은 이산 토큰 시퀀스로 압축한 후, Transformer를 사용하여 토큰 시퀀스를 모델링합니다. Parti, MuseNet, AudioLM, VALL-E, Sora의 패치 토크나이저는 모두 이를 사용합니다. 이는 버킷 1에 학습된 토크나이저를 더한 것입니다.

## 간략한 역사

| 연도 | 모델 | 중요성 |
|------|-------|-----------------|
| 2013 | 변분 오토인코더 (VAE)(VAE (Variational Autoencoder)) (Kingma) | 사용 가능한 학습 손실을 가진 최초의 심층 생성 모델. |
| 2014 | GAN (생성적 적대 신경망)(GAN (Generative Adversarial Network)) (Goodfellow) | 암시적 밀도, 우도 없음 — 놀라울 정도로 선명한 샘플. |
| 2015 | DRAW, PixelCNN | 순차적 이미지 생성. |
| 2017 | Glow, RealNVP | 가역적 흐름; 깊이를 가진 정확한 우도. |
| 2017 | Progressive GAN | 최초의 메가픽셀 얼굴. |
| 2019 | StyleGAN / StyleGAN2 | 그 하나의 영역에서는 여전히 넘보기 어려운 사진적 사실주의 얼굴. |
| 2020 | DDPM (Ho) | 확산이 실용화됨. |
| 2021 | CLIP, DALL-E 1, VQGAN | 텍스트-이미지 생성이 주류가 됨. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 잠재 확산 + 텍스트 조건 = 상품화. |
| 2022 | ControlNet, LoRA (저랭크 적응)(LoRA (Low-Rank Adaptation)) | 사전 학습된 확산 모델에 대한 세밀한 제어. |
| 2023 | SDXL, Midjourney v5, Flow matching | 규모 + 더 나은 학습 동역학. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 비디오 확산; flow matching이 승리. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 프로덕션급 비디오. |
| 2026 | 일관성 + 정류 흐름(Rectified Flow) | 확산 백본(diffusion backbone)에서 단계 샘플링. |

## 5가지 질문 선별법

새로운 생성 모델 논문이 발표되면, 방법론 섹션을 읽기 전에 이 5가지 질문에 답해 보세요.

1. **무엇을 모델링하는가?** 픽셀, 잠재 변수(latent), 이산 토큰, 3D 가우시안, 메시, 파형?
2. **밀도는 명시적(explicit)인가, 암시적(implicit)인가?** `log p(x)`를 명시적으로 작성하는가?
3. **샘플링: 단계(one-shot)인가, 반복적(iterative)인가?** 반복적 샘플링은 추론이 느려집니다. 단계 샘플링은 보통 적대적(adversarial) 또는 증류(distilled) 방식을 의미합니다.
4. **조건부(conditioning): 무조건부, 클래스, 텍스트, 이미지, 포즈?** 이는 손실 함수와 아키텍처 골격(scaffolding)을 결정합니다.
5. **평가: FID, CLIP 점수, IS, 인간 선호도, 작업 정확도?** 각각 알려진 실패 모드(failure mode)가 있습니다 (14강 참조).

이 단계의 모든 강에서 이 5가지 질문에 다시 답하게 될 것입니다. 끝이 나면, 이 질문들은 반사적으로 답할 수 있게 될 것입니다.

```figure
autoencoder-bottleneck
```

## 구현하기

이 강의의 코드는 가벼운 시각화입니다. 세 가지 장난감 접근법(커널 밀도 추정, 이산 히스토그램, 최근접 샘플 "GAN-ish" 생성기)을 사용하여 샘플로부터 1-D 가우시안 혼합(mixture-of-Gaussians)을 피팅합니다. 이를 통해 한 화면에 출력할 수 있는 문제에서 명시적 밀도와 암시적 밀도의 차이를 확인할 수 있습니다.

`code/main.py`를 실행하세요. 2모드 가우시안 혼합(mixture)에서 2000개의 샘플을 추출한 후 다음을 출력합니다:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

주의: 첫 번째 두 방법은 "이 포인트가 얼마나 가능성 있는가?"라고 질문할 수 있습니다. 세 번째 방법은 그럴 수 없습니다. 이것이 *명시적 vs 암시적* 밀도의 구분이며, 모든 미래 강의에서 중요하게 작용할 것입니다.

## 사용하기

2026년, 어떤 태스크에 어떤 계열(family)을 사용해야 할까요?

| 태스크 | 최적 계열 | 이유 |
|------|-------------|-----|
| 사실적 얼굴, 좁은 도메인 | StyleGAN 2/3 | 여전히 가장 선명하고, 추론이 가장 빠릅니다. |
| 범용 텍스트-이미지 | 잠재 확산(latent diffusion) + 흐름 매칭(flow matching) | SD3, Flux.1, DALL-E 3. |
| 빠른 텍스트-이미지 | 정류 흐름(rectified flow) + 증류(distillation) | SDXL-Turbo, SD3-Turbo, LCM. |
| 텍스트-비디오 | 확산 트랜스포머(Diffusion Transformer) + 흐름 매칭(flow matching) | Sora, Veo 2, Kling. |
| 음성 + 음악 | 토큰 기반 AR (AudioLM, VALL-E, MusicGen) 또는 흐름 매칭 (AudioCraft 2) | 이산 토큰은 저렴하게 확장됩니다. |
| 3D 장면 | 가우시안 스플래팅 적합, 확산 사전 | 재구성에는 3D-GS, 신규 뷰 생성에는 확산 모델을 사용합니다. |
| 밀도 추정 (샘플링 없음) | 흐름 | 정확한 `log p(x)`을 제공하는 유일한 계열입니다. |
| 시뮬레이션 / 물리 | 흐름 매칭, 점수 SDE | 직선 경로, 매끄러운 벡터 필드를 사용합니다. |

## 출시하기

`outputs/skill-model-chooser.md`으로 저장하세요.

이 스킬은 작업 설명을 입력받아 (1) 사용할 계열, (2) 오픈 소스 및 호스팅 옵션 각각 세 가지를 순위를 매긴 목록, (3) 주의해야 할 예상 실패 모드, (4) 컴퓨팅/시간 예산을 출력합니다.

## 연습 문제

1. **쉬움.** ChatGPT 이미지, Midjourney v7, Sora, Runway Gen-3, ElevenLabs 중 각 제품에 대해 계열과 백본을 식별하세요. 근거는 공개된 기술 보고서에서 찾아야 합니다.
2. **중간.** 내일 읽을 논문이 확산 모델보다 100배 빠른 샘플링을 주장합니다. 조건부 생성 및 고해상도에서도 속도 향상이 유지되는지 확인하기 위해 세 가지 질문을 작성하세요.
3. **어려움.** 관심 있는 한 분야 (예: 단백질 구조, CAD, 분자, 궤적)를 선택하여 해당 분야의 현재 SOTA 모델에 대해 5가지 질문 트리아지를 답하고, 더 나은 모델이 무엇을 변경해야 하는지 스케치하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 생성 모델 | "새로운 것을 만든다" | `p_data(x)`에 대한 샘플러를 학습하며, 선택적으로 `log p(x)`을 노출합니다. |
| 명시적 밀도 | "평가할 수 있다" | 모델이 폐형 또는 추적 가능한 `log p(x)`을 제공합니다. |
| 암시적 밀도 | "GAN 스타일" | 샘플러만 존재하며, 주어진 점에 대한 `p(x)`을 평가할 방법이 없습니다. |
| ELBO | "증거 하한" | `log p(x)`에 대한 추적 가능한 하한입니다. VAE와 확산 모델이 이를 최적화합니다. |
| 점수 | "로그 밀도의 기울기" | `∇_x log p(x)`입니다. 확산 및 SDE 모델이 이 필드를 학습합니다. |
| 다양체 가설 | "데이터는 표면에 존재한다" | 고차원 데이터는 저차원 다양체에 집중됩니다. 차원 축소가 작동하는 이유입니다. |
| 자기회귀(Autoregressive) | "다음 조각을 예측" | 결합 확률을 조건부 확률의 곱으로 분해합니다. |
| 잠재 공간(Latent) | "압축된 코드" | 디코더가 입력을 재구성할 수 있는 저차원 표현입니다. |

## 프로덕션 노트: 5가지 계열, 5가지 추론 형태

각 계열은 서로 다른 추론 서버 비용 곡선에 대응합니다. 프로덕션 추론 문헌에서는 LLM 추론을 프리필(Prefill) + 디코딩으로 정의하며, 동일한 분해가 여기에도 적용됩니다:

- **자기회귀(Autoregressive) (계열 1 및 5).** 순차적 디코딩이 지연 시간을 지배합니다. KV 캐시, 연속 배치, 추론적 디코딩이 모두 직접 적용됩니다.
- **VAE / 확산 모델 / 흐름 매칭 (계열 2 및 4).** LLM 의미에서의 디코딩이 없습니다. 비용 = `num_steps × step_cost`이며, `step_cost`은 전체 잠재 해상도에서의 트랜스포머 또는 U-Net 순방향 연산입니다. 프로덕션 조정 변수는 스텝 수 (DDIM / DPM-Solver / 증류), 배치 크기, 정밀도 (bf16 / fp8 / int4)입니다.
- **GAN (계열 3).** 한 번의 순방향 연산. 스케줄이 없고, KV 캐시도 없습니다. TTFT ≈ 총 지연 시간. 이것이 StyleGAN이 좁은 도메인 UX에서 여전히 승리하는 이유입니다.

논문 초록에서 "확산 모델보다 빠름"을 보시면, "적은 스텝 수 × 동일한 스텝 비용" 또는 "동일한 스텝 수 × 더 저렴한 스텝 비용"으로 번역해 보세요. 나머지는 모두 마케팅입니다.

## 추가 읽기

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 논문.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 논문.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — SDE로서의 확산 모델.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 흐름 매칭 논문.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
