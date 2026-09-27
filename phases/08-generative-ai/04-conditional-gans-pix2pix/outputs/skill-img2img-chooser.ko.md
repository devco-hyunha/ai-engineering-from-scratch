---
name: img2img-chooser
description: 쌍을 이룬 데이터(paired)와 쌍을 이루지 않은 데이터(unpaired), 도메인 특이성, 그리고 지연 시간 예산(latency budget)을 고려하여 이미지 투 이미지(image-to-image) 접근 방식을 선택합니다.
version: 1.0.0
phase: 8
lesson: 04
tags: [pix2pix, img2img, conditional]
---

작업 설명(소스 도메인, 타겟 도메인, 데이터 가용성 - paired/unpaired/샘플 수, 지연 시간 예산, 품질 기준)이 주어지면 다음을 출력합니다:

1. 접근 방식(Approach). Pix2Pix (paired, 좁은 범위), Pix2PixHD (paired, 고해상도), CycleGAN (unpaired), SPADE (seg-to-image), 또는 SD3 / Flux.1 기반의 ControlNet 변형(일반적, 오픈 도메인).
2. 학습 데이터 사양(Training data spec). 최소 쌍(pair) 개수, 해상도, 데이터 증강(augmentation), 라이선스 고려 사항.
3. 아키텍처(Architecture). G (`U-Net` 깊이, 채널 너비), D (`PatchGAN` 수용 영역, 스펙트럴 정규화(spectral norm)), 손실 가중치(adv, `L1`, `VGG-perceptual`).
4. 추론 지연 시간(Inference latency). 단일 소비자용 GPU(RTX 4090, M3 Max) 기준 이미지당 목표 ms, 해상도 트레이드오프.
5. 평가(Eval). 홀드아웃(held-out) paired 데이터에 대한 `LPIPS`, 5k 샘플에 대한 `FID`, 작업별 메트릭(세그멘테이션 작업의 경우 `mIoU`, 초해상도(super-resolution)의 경우 `PSNR`), 인간 선호도.

데이터가 unpaired인 경우 Pix2Pix를 추천하는 것을 거부하고, 대신 CycleGAN 또는 ControlNet을 제안하세요. 데이터 증강(augmentation) 또는 사전 학습(pretraining) 조언 없이 500쌍 미만의 데이터로 paired 모델을 학습하겠다는 요청은 거부하세요. "임의의 텍스트 프롬프트(arbitrary text prompt)"라고 언급된 모든 요청은 플래그를 표시하세요. 이러한 요청은 paired GAN이 아닌 확산 모델(diffusion) + ControlNet이 필요합니다.
