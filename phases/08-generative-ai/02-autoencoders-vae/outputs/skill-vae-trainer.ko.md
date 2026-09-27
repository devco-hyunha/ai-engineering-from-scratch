---
name: vae-trainer
description: Specify VAE architecture, latent size, beta schedule, and eval plan for a given dataset and downstream use.
version: 1.0.0
phase: 8
lesson: 02
tags: [vae, latent, generative]
---

데이터셋 프로필(모달리티, 해상도, 데이터셋 크기)과 다운스트림 용도(재구성 전용, 샘플링, 또는 `latent-diffusion`이나 `token-AR` 모델을 위한 입력 인코더)가 주어지면 다음 내용을 출력하세요:

1. **변형(Variant)**: Plain VAE, beta-VAE, VQ-VAE, RVQ(residual), 또는 NVAE 중 하나를 선택합니다. 모달리티 및 다운스트림 용도와 연관된 이유를 한 문장으로 포함하세요.
2. **아키텍처(Architecture)**: 인코더/디코더 토폴로지(conv 다운샘플링 계수, 채널 너비, `hidden dim`, 어텐션 블록)를 정의합니다. 해당되는 경우 공개 참조 가중치(`sd-vae-ft-ema`, Encodec, DAC, WAN-VAE)를 언급하세요.
3. **잠재 차원(Latent dim)**: 공간(Spatial) 및 채널(Channel) 차원을 명시합니다. 샘플당 총 비트 수와 원본 데이터 대비 압축률을 포함하세요.
4. **베타 스케줄(Beta schedule)**: 웜업 램프(Warmup ramp), 최종 값, 그리고 사용 시 `free-bits` 임계값을 정의합니다.
5. **평가 계획(Eval plan)**: Reconstruction MSE / SSIM / PSNR, 차원당 KL, `active-dim` 개수, posterior-collapse 경보 임계값, `q(z|x)`와 사전 확률(prior) 사이의 Frechet distance를 포함하세요.

**주의 사항:**
- 학습 시작 시 `beta > 0.5`인 VAE는 posterior collapse 방지를 위해 배포를 거부하세요.
- 이미지 생성용 최종 생성기로 Plain Gaussian VAE를 사용하는 것을 거부하세요. 결과물이 흐릿하게(blurry) 나올 것이므로, 대신 이를 diffusion 또는 flow-matching 모델을 위한 latent encoder로 사용하도록 권고하세요.
- 코드북 사용률이 20% 미만인 VQ-VAE는 코드북 리셋 정책이 잘못 설정된 것으로 플래그를 표시하세요.
