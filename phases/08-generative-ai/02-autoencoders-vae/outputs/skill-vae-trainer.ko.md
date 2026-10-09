---
name: vae-trainer
description: 주어진 데이터셋과 다운스트림 용도에 대해 VAE 아키텍처, 잠재 크기, beta 스케줄 및 평가 계획을 지정합니다.
version: 1.0.0
phase: 8단계
lesson: 02강
tags: [vae, latent, generative]
---

데이터셋 프로필(모달리티, 해상도, 데이터셋 크기)과 다운스트림 용도(재구성 전용, 샘플링, 또는 잠재 확산(latent-diffusion) 또는 토큰 AR 모델용 입력 인코더)이 주어지면 다음을 출력합니다:

1. 변형. Plain VAE, beta-VAE, VQ-VAE, RVQ(residual), 또는 NVAE. 모달리티와 다운스트림 용도와 연결된 한 문장 이유를 포함합니다.
2. 아키텍처. 인코더 / 디코더 토폴로지(conv downsampling factor, channel width, hidden dim, attention blocks). 해당되는 경우 공개 참조 가중치(`sd-vae-ft-ema`, Encodec, DAC, WAN-VAE)를 언급합니다.
3. 잠재 차원. 공간 및 채널 차원. 샘플당 총 비트 수. 원시 데이터 대비 압축 비율.
4. Beta 스케줄. Warmup ramp, 최종 값, 그리고 사용된 경우 free-bits threshold.
5. 평가 계획. 재구성 MSE / SSIM / PSNR, 차원별 KL, active-dim count, posterior-collapse alarm threshold, `q(z|x)`과 prior 간의 Frechet distance.

학습 시작 시 beta > 0.5인 VAE를 출시하는 것을 거부합니다(posterior collapse). 이미지를 위한 최종 생성기로 plain Gaussian VAE를 사용하는 것을 거부합니다 - 흐릿한 결과가 나올 것입니다. 대신 확산(diffusion) 또는 flow-matching 모델의 잠재 인코더로 사용하세요. codebook 사용량이 20% 미만인 VQ-VAE는 codebook reset policy가 잘못 설정된 것으로 플래그합니다.
