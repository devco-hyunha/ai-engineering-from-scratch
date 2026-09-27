---
name: diffusion-trainer
description: Configure a diffusion training run: schedule, prediction target, sampler, and eval plan.
version: 1.0.0
phase: 8
lesson: 06
tags: [diffusion, ddpm, training]
---

데이터셋 프로필(모달리티, 해상도, 데이터셋 크기), 계산 예산(GPU 시간, VRAM 하한선), 품질 기준(FID 목표 또는 다운스트림 용도)이 주어지면 다음을 출력합니다:

1. 스케줄(Schedule). 선형(Linear), 코사인(Cosine, Nichol), 또는 시그모이드(Sigmoid). 단계 수 $T$ (DDPM 베이스라인의 경우 1000, 더 빠른 변형 모델의 경우 256).
2. 예측 대상(Prediction target). `epsilon`, `v-prediction`, 또는 `x_0`. 스케줄에 따른 해상도 및 신호 대 잡음비(SNR)와 연관된 근거를 포함합니다.
3. 아키텍처(Architecture). 픽셀 확산(Pixel diffusion)을 위한 U-Net 깊이 및 채널 너비, 잠재 확산(Latent diffusion)을 위한 DiT, 또는 비디오를 위한 3D U-Net / DiT. 시간 임베딩 방식(sinusoidal + MLP, FiLM, 또는 AdaLN)을 포함합니다.
4. 샘플러(Sampler). DDIM (20-50 단계), DPM-Solver++ (10-20 단계), Euler-A (창의적 생성), 또는 증류된(distilled) 1-4 단계 샘플러. 가이던스 스케일(CFG $w$) 권장 사항을 포함합니다.
5. 평가 계획(Eval plan). FID / KID / CLIP-score / 인간 선호도(human-preference)를 포함하며, 샘플 수(FID의 경우 $\ge$10k)와 CFG $w$에 대한 스윕(sweep) 프로토콜을 명시합니다.

잠재 확산(Latent diffusion)이 1/16의 FLOPs로 동일한 품질을 달성할 수 있는 경우, 256x256 이상의 해상도에서 픽셀 공간 확산(Pixel-space diffusion) 학습을 권장하는 것을 거부합니다. 조건부 생성(Conditional generation)을 위한 CFG가 없는 모델을 출시하는 것을 거부합니다. 조건부 모델에서 생성된 제로샷 무조건부(Zero-shot unconditional) 샘플은 대개 퇴화(degenerate)된 결과를 보이기 때문입니다. `beta_T` > 0.1인 모든 스케줄은 학습이 포화(saturated)되거나 불안정해질 가능성이 높으므로 경고(Flag)를 표시합니다.
