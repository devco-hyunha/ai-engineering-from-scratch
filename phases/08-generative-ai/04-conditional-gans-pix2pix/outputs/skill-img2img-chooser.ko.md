---
name: img2img-chooser
description: 짝이 있는 데이터 vs 짝이 없는 데이터, 도메인 특이성, 지연 시간 예산을 고려하여 이미지-to-이미지 접근 방식을 선택합니다.
version: 1.0.0
phase: 8단계
lesson: 04강
tags: [pix2pix, img2img, conditional]
---

작업 설명 (소스 도메인, 타겟 도메인, 데이터 가용성 - 짝이 있는/짝이 없는/N 샘플, 지연 시간 예산, 품질 기준)이 주어지면, 다음을 출력합니다:

1. 접근 방식. Pix2Pix (짝이 있는, 좁은 도메인), Pix2PixHD (짝이 있는, 고해상도), CycleGAN (짝이 없는), SPADE (분할-to-이미지), 또는 SD3 / Flux.1 기반 ControlNet 변형 (일반, 오픈 도메인).
2. 학습 데이터 사양. 최소 쌍 수, 해상도, 증강 기법, 라이선스 고려 사항.
3. 아키텍처. G (U-Net 깊이, 채널 폭), D (PatchGAN 수용 영역, 스펙트럴 노름), 손실 가중치 (adv, L1, VGG-perceptual).
4. 추론 지연. 단일 소비자 GPU (RTX 4090, M3 Max)에서의 이미지당 목표 ms, 해상도 트레이드오프.
5. 평가. 홀드아웃 짝이 있는 데이터에 대한 LPIPS, 5k 샘플에 대한 FID, 작업 특이적 지표 (분할 작업의 mIoU, 초해상도의 PSNR), 인간 선호도.

데이터가 짝이 없는 경우 Pix2Pix를 추천하지 마세요 - 대신 CycleGAN이나 ControlNet을 처방하세요. 증강 / 사전 학습 조언 없이 500 쌍 미만으로 짝이 있는 모델을 학습하는 것을 거부하세요. "임의의 텍스트 프롬프트"라고 요청하는 모든 요청을 플래그하세요 - 이러한 요청은 짝이 있는 GAN이 아니라 diffusion + ControlNet이 필요합니다.
