---
name: diffusion-trainer
description: 확산(diffusion) 학습 실행을 구성합니다: 스케줄, 예측 대상, 샘플러, 평가 계획.
version: 1.0.0
phase: 8단계
lesson: 06강
tags: [diffusion, ddpm, training]
---

데이터셋 프로필(모달리티, 해상도, 데이터셋 크기), 컴퓨팅 예산(GPU 시간, VRAM 하한), 품질 기준(FID 목표 또는 다운스트림 사용)이 주어지면 다음을 출력합니다:

1. 스케줄. 선형(linear), 코사인(cosine, Nichol), 또는 시그모이드(sigmoid). 단계 수 T (DDPM 기준선: 1000; 더 빠른 변형: 256).
2. 예측 대상. epsilon, v-prediction, 또는 x_0. 해상도와 스케줄 전반의 신호 대 잡음비(signal-to-noise)와 관련된 이유를 포함합니다.
3. 아키텍처. 픽셀 확산(pixel diffusion)의 경우 U-Net 깊이 + 채널 폭, 잠재 확산(latent diffusion)의 경우 DiT, 또는 비디오의 경우 3D U-Net / DiT. 시간 임베딩(time embedding) 방식(사인(sinusoidal) + MLP, FiLM, 또는 AdaLN)을 포함합니다.
4. 샘플러. DDIM (20-50 단계), DPM-Solver++ (10-20), Euler-A (창의적), 또는 증류(distilled) 1-4단계. 가이드 스케일(guidance scale, CFG w) 권장 사항을 포함합니다.
5. 평가 계획. FID / KID / CLIP-score / 인간 선호도(human-preference)와 샘플 수(FID의 경우 >=10k), CFG w에 대한 스윕(sweep) 프로토콜을 포함합니다.

잠재 확산(latent diffusion)이 FLOPs의 1/16로 동일한 품질을 달성할 때 >=256x256에서 픽셀 공간(pixel-space) 확산 학습을 권장하지 마십시오. 조건부 생성(conditional generation)을 위한 CFG 없이 모델을 출시하지 마십시오. 조건부 모델의 제로샷(zero-shot) 무조건(unconditional) 샘플은 일반적으로 퇴화(degenerate)됩니다. beta_T > 0.1인 모든 스케줄은 포화(saturated)되거나 불안정한 학습을 생성할 가능성이 높으므로 플래그(flag)를 지정하십시오.
