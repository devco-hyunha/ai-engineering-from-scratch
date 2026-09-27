# 생성 모델 — 분류 및 역사 (Generative Models — Taxonomy & History)

> 모든 이미지 모델, 텍스트 모델, 비디오 모델, 3D 모델은 다섯 가지 범주 중 하나에 속합니다. 잘못된 범주를 선택하면 몇 주 동안 수학적 원리와 씨름해야 할 것입니다. 올바른 범주를 선택하면 지난 12년간 이 분야가 이룩한 발전 과정을 머릿속에 깔끔하게 정리할 수 있습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## 문제점 (The Problem)

생성 모델(Generative model)의 역할은 단 하나입니다. 미지의 분포 `p_data(x)`에서 추출된 학습 샘플이 주어졌을 때, 동일한 분포에서 나온 것처럼 보이는 새로운 샘플을 출력하는 것입니다. 얼굴, 문장, MIDI 파일, 단백질 구조 — 자세히 들여다보면 모두 같은 문제입니다.

문제는 `p_data`가 수백만 차원의 공간(512x512 RGB 이미지는 약 786k 차원임)에 존재하며, 샘플들은 그 공간 내부의 얇은 매니폴드(manifold) 위에 놓여 있다는 점, 그리고 여러분에게는 고작 1,000만 개의 예시만이 있다는 점입니다. 밀도(density)를 무차별 대입(brute-force) 방식으로 계산하는 것은 불가능합니다. 모든 생성 모델은 하나의 어려운 문제를 약간 덜 어려운 문제로 맞바꾸는 타협안입니다.

지난 12년 동안 다섯 가지 계열(families)이 살아남았습니다. 각 계열이 어떤 타협을 선택했는지 이해하면, 왜 특정 모델이 어떤 작업에서는 승리하고 다른 작업에서는 붕괴(collapse)하는지 알 수 있습니다.

## 개념 (The Concept)

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. 명시적 밀도, 계산 가능 (Explicit density, tractable).** `log p(x)`를 실제로 계산할 수 있는 합의 형태로 작성합니다. 자기회귀 모델(Autoregressive models; PixelCNN, WaveNet, GPT)은 `p(x) = ∏ p(x_i | x_<i)`로 인수분해합니다. 노멀라이징 플로우(Normalizing flows; RealNVP, Glow)는 단순한 기본 분포를 가역 변환(invertible transform)하여 `p(x)`를 구축합니다. 장점: 정확한 가능도(likelihood), 깔끔한 학습 손실 함수. 단점: 자기회귀 추론은 순차적입니다(긴 시퀀스에서 느림), 플로우는 가역 아키텍처가 필요합니다(구조적 제약).

**2. 명시적 밀도, 근사적 (Explicit density, approximate).** `log p(x)`의 하한(ELBO)을 구하고 이 하한을 최적화합니다. VAE(Kingma 2013)는 변분 후험 분포(variational posterior)를 가진 인코더-디코더를 사용합니다. 확산 모델(Diffusion models; DDPM, Ho 2020)은 가중치가 적용된 ELBO를 암시적으로 최적화하는 디노이저(denoiser)를 학습합니다. 확산 모델은 2026년 이미지, 비디오, 3D 분야의 지배적인 백본(backbone)입니다.

**3. 암시적 밀도 (Implicit density).** 밀도 계산을 완전히 생략합니다. 샘플을 생성하는 생성기 `G(z)`와 진짜와 가짜를 구별하는 판별기 `D(x)`를 학습합니다. GAN(Goodfellow 2014). 추론 속도가 빠르지만(단 한 번의 순전파), 학습 과정이 매우 불안정한 것으로 유명합니다. StyleGAN 1/2/3는 2026년에도 특정 도메인의 실사성(얼굴, 침실 등) 측면에서 여전히 최첨단(state of the art) 기술로 남아 있습니다.

**4. 스코어 기반 / 연속 시간 (Score-based / continuous-time).** 로그 밀도의 그래디언트 `∇_x log p(x)`(스코어)를 직접 학습합니다. Song & Ermon (2019)은 스코어 매칭(score matching)이 확산을 SDE(확률 미분 방정식)로 일반화할 수 있음을 보여주었습니다. 플로우 매칭(Flow matching; Lipman 2023)은 2024-2026년의 핵심 기술입니다: 시뮬레이션이 필요 없는 학습, 더 직선적인 경로, DDPM보다 4-10배 빠른 샘플링이 가능합니다. Stable Diffusion 3, Flux, AudioCraft 2는 모두 플로우 매칭을 사용합니다.

**5. 이산 코드 기반 자기회귀 (Token-based autoregressive over discrete codes).** VQ-VAE 또는 잔차 양자화기(residual quantizer)를 사용하여 고차원 데이터를 짧은 이산 토큰 시퀀스로 압축한 다음, 트랜스포머(Transformer)를 사용하여 토큰 시퀀스를 모델링합니다. Parti, MuseNet, AudioLM, VALL-E, Sora의 패치 토크나이저(patch tokenizer)가 모두 이 방식을 사용합니다. 이는 1번 범주에 학습된 토크나이저가 추가된 형태입니다.

## 간략한 역사 (A brief history)

| 연도 | 모델 | 중요했던 이유 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 사용 가능한 학습 손실(training loss)을 가진 최초의 딥 생성 모델. |
| 2014 | GAN (Goodfellow) | 암시적 밀도(Implicit density), 우도(likelihood) 없음 — 놀라울 정도로 선명한 샘플 생성. |
| 2015 | DRAW, PixelCNN | 순차적 이미지 생성. |
| 2017 | Glow, RealNVP | 가역적 흐름(Invertible flows); 깊이를 가진 정확한 우도 계산. |
| 2017 | Progressive GAN | 최초의 메가픽셀급 얼굴 생성. |
| 2019 | StyleGAN / StyleGAN2 | 해당 도메인에서 여전히 극복하기 어려운 사진 같은 얼굴 생성. |
| 2020 | DDPM (Ho) | 확산(Diffusion) 모델의 실용화. |
| 2021 | CLIP, DALL-E 1, VQGAN | 텍스트-투-이미지(Text-to-image)의 주류화. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 잠재 확산(Latent diffusion) + 텍스트 조건화(text conditioning) = 대중화. |
| 2022 | ControlNet, LoRA | 사전 학습된 확산 모델에 대한 미세 제어. |
| 2023 | SDXL, Midjourney v5, Flow matching | 규모 확장(Scale) + 더 나은 학습 역학(training dynamics). |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 비디오 확산; 플로우 매칭(flow matching)의 승리. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 프로덕션급 비디오 생성. |
| 2026 | Consistency + Rectified Flow | 확산 백본(diffusion backbones)으로부터의 단일 단계 샘플링(One-step sampling). |

## 5가지 질문을 통한 분류 (The five-question triage)

새로운 생성 모델 논문이 발표되면, 방법론(method) 섹션을 읽기 전에 다음 다섯 가지 질문에 답해 보세요.

1. **무엇을 모델링하는가? (What is being modeled?)** 픽셀(pixels), 잠재 변수(latents), 이산 토큰(discrete tokens), 3D 가우시안(3D Gaussians), 메쉬(meshes), 파형(waveforms) 중 무엇인가요?
2. **밀도(density)가 명시적(explicit)인가, 암시적(implicit)인가?** `log p(x)`를 직접 기술하나요?
3. **샘플링 방식은 단판(one-shot)인가, 반복적(iterative)인가?** 반복적 방식은 추론 속도가 느림을 의미하며, 단판 방식은 대개 적대적 학습(adversarial) 또는 증류(distilled) 모델을 의미합니다.
4. **조건화(Conditioning) 방식은 무엇인가: 무조건부(unconditional), 클래스(class), 텍스트(text), 이미지(image), 포즈(pose)?** 이는 손실 함수(loss)와 아키텍처 구조(architecture scaffolding)를 결정합니다.
5. **평가 지표(Evaluation)는 무엇인가: FID, CLIP score, IS, 인간 선호도(human preference), 작업 정확도(task accuracy)?** 각 지표에는 알려진 결함 모드(failure modes)가 있습니다 (레슨 14 참조).

이 단계의 모든 레슨에서 이 다섯 가지 질문에 다시 답하게 될 것입니다. 과정을 마칠 때쯤이면 이 질문들은 반사적으로 나오게 될 것입니다.

```figure
autoencoder-bottleneck
```

## 직접 구현해 보기 (Build It)

이 레슨의 코드는 가벼운 시각화 도구입니다. 세 가지 장난감 접근 방식(커널 밀도 추정, 이산 히스토그램, 그리고 가장 가까운 샘플을 사용하는 "GAN 방식" 생성기)을 사용하여 샘플로부터 1차원 가우시안 혼합 모델(mixture-of-Gaussians)을 피팅합니다. 이를 통해 한 화면에 출력할 수 있는 문제를 가지고 명시적 밀도(explicit density)와 암시적 밀도(implicit density)의 차이를 직접 확인할 수 있습니다.

`code/main.py`를 실행해 보세요. 이 코드는 두 개의 모드를 가진 가우시안 혼합 모델에서 2000개의 샘플을 추출한 뒤, 다음과 같이 출력합니다:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

주의 깊게 살펴보세요: 처음 두 방식은 "이 지점이 나타날 확률이 얼마인가?"라는 질문에 답할 수 있습니다. 하지만 세 번째 방식은 불가능합니다. 이것이 바로 앞으로의 모든 레슨에서 중요하게 다뤄질 *명시적(explicit) vs 암시적(implicit)*의 차이입니다.

## 활용하기 (Use It)

2026년, 어떤 작업에 어떤 계열(family)을 사용해야 할까요?

| 작업 (Task) | 최적의 계열 (Best family) | 이유 (Why) |
|------|-------------|-----|
| 실사 얼굴, 좁은 도메인 | StyleGAN 2/3 | 여전히 가장 선명하며 추론 속도가 빠름. |
| 일반적인 텍스트-이미지 생성 | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| 빠른 텍스트-이미지 생성 | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| 텍스트-비디오 생성 | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| 음성 + 음악 | Token-based AR (AudioLM, VALL-E, MusicGen) 또는 flow matching (AudioCraft 2) | 이산 토큰(Discrete tokens)은 비용 효율적으로 확장 가능함. |
| 3D 장면 | Gaussian Splatting fit, diffusion prior | 재구성을 위한 3D-GS, 새로운 시점 생성을 위한 diffusion. |
| 밀도 추정 (샘플링 없음) | Flows | 정확한 `log p(x)`를 제공하는 유일한 계열. |
| 시뮬레이션 / 물리 | Flow matching, score SDE | 직선 경로 및 매끄러운 벡터장(vector fields). |

## Ship It (실행하기)

`outputs/skill-model-chooser.md`로 저장하세요.

이 스킬은 작업 설명(task description)을 입력받아 다음을 출력합니다: (1) 사용할 모델 계층(family), (2) 오픈 소스 3종 및 호스팅 모델 3종으로 구성된 순위 목록, (3) 주의해야 할 예상 실패 모드(failure mode), (4) 컴퓨팅/시간 예산(compute/time budget).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 다음 다섯 가지 제품 각각에 대해 제품군(family)과 백본(backbone)을 식별해 보세요: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. 근거는 공개된 기술 보고서(technical reports)를 바탕으로 해야 합니다.
2. **중간 (Medium).** 내일 읽게 될 논문은 확산 모델(diffusion)보다 100배 빠른 샘플링이 가능하다고 주장합니다. 이러한 속도 향상이 조건화(conditioning) 및 고해상도(high resolution) 환경에서도 유지되는지 확인하기 위한 세 가지 질문을 작성해 보세요.
3. **어려움 (Hard).** 관심 있는 한 가지 도메인(예: 단백질 구조, CAD, 분자, 궤적 등)을 선택하세요. 해당 도메인의 현재 SOTA(최첨단) 모델에 대해 '5가지 질문 분류 체계(five-question triage)'를 적용하여 답변하고, 더 나은 모델이 되기 위해 무엇을 개선해야 할지 스케치해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| 생성 모델 (Generative model) | "새로운 것을 만들어낸다" | `p_data(x)`를 위한 샘플러를 학습하며, 선택적으로 `log p(x)`를 노출함. |
| 명시적 밀도 (Explicit density) | "평가가 가능하다" | 모델이 폐쇄형(closed-form) 또는 다루기 쉬운(tractable) `log p(x)`를 제공함. |
| 암시적 밀도 (Implicit density) | "GAN 스타일" | 샘플러만 존재하며, 주어진 지점의 `p(x)`를 평가할 방법이 없음. |
| ELBO | "증거 하한 (Evidence lower bound)" | `log p(x)`에 대한 다루기 쉬운 하한값; VAE와 확산 모델(diffusion)은 이를 최적화함. |
| 스코어 (Score) | "로그 밀도의 기울기" | `∇_x log p(x)`; 확산 모델과 SDE 모델은 이 필드를 학습함. |
| 매니폴드 가설 (Manifold hypothesis) | "데이터는 표면 위에 존재한다" | 고차원 데이터가 저차원 매니폴드에 집중되어 있다는 가설; 차원 축소가 작동하는 이유. |
| 자기회귀 (Autoregressive) | "다음 조각을 예측한다" | 결합 확률(joint)을 조건부 확률(conditionals)의 곱으로 인수분해함. |
| 잠재 변수 (Latent) | "압축된 코드" | 디코더가 입력을 재구성할 수 있는 저차원 표현. |

## 프로덕션 노트: 5가지 계열, 5가지 추론 형태 (five families, five inference shapes)

각 계열은 서로 다른 추론 서버 비용 곡선(inference-server cost curve)에 매핑됩니다. 프로덕션 추론(production-inference) 문헌에서는 LLM 추론을 프리필(prefill) + 디코딩(decode)으로 정의하며, 동일한 분해 방식이 여기에도 적용됩니다:

- **자기회귀(Autoregressive) (버킷 1 및 5).** 순차적 디코딩이 지연 시간(latency)을 지배합니다. KV-cache, 연속 배칭(continuous batching), 그리고 투기적 디코딩(speculative decoding)이 모두 직접적으로 적용됩니다.
- **VAE / 확산(diffusion) / 플로우 매칭(flow-matching) (버킷 2 및 4).** LLM 의미에서의 디코딩은 존재하지 않습니다. 비용 = `num_steps × step_cost`이며, 여기서 `step_cost`는 전체 잠재 해상도(full latent resolution)에서의 트랜스포머(transformer) 또는 U-Net 순전파(forward)입니다. 프로덕션 조절 요소(knobs)는 스텝 수(DDIM / DPM-Solver / distillation), 배치 크기(batch size), 그리고 정밀도(bf16 / fp8 / int4)입니다.
- **GAN (버킷 3).** 단 한 번의 순전파(forward pass)로 이루어집니다. 스케줄도, KV-cache도 없습니다. TTFT ≈ 총 지연 시간입니다. 이것이 StyleGAN이 여전히 좁은 도메인의 UX에서 우위를 점하는 이유입니다.

논문 초록에서 "확산 모델보다 빠르다(faster than diffusion)"라는 문구를 본다면, 이를 "더 적은 스텝 × 동일한 스텝 비용" 또는 "동일한 스텝 × 더 저렴한 스텝 비용"으로 해석하십시오. 그 외의 모든 것은 마케팅입니다.

## 추가 읽을거리 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 논문입니다.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문입니다.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 논문입니다.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — SDE를 통한 확산 모델(diffusion) 연구입니다.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Flow Matching 논문입니다.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3에 관한 논문입니다.
