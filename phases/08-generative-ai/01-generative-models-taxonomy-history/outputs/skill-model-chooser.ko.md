---
name: generative-model-chooser
description: 주어진 작업과 예산에 따라 생성 모델 계열, 백본, 호스팅 대안을 선택합니다.
version: 1.0.0
phase: 8단계
lesson: 01강
tags: [generative, taxonomy]
---

작업 설명(모달리티, 도메인, 지연 시간 예산, 컴퓨팅 예산, 조건 신호)이 주어지면 다음을 출력합니다:

1. 계열. 명시적 추적 가능(explicit-tractable), 명시적 근사(explicit-approximate)(VAE / diffusion), 암시적(implicit)(GAN), 점수/유량 매칭(score / flow matching), 또는 토큰 자기회귀(token-AR) 중 하나를 선택합니다. 모달리티와 지연 시간에 연결된 한 문장 이유를 포함합니다.
2. 백본 + 오픈 참조. 사용자가 오늘 미세 조정(Fine-tuning)할 수 있는 사전 학습된 오픈 가중치 모델 하나(예: Stable Diffusion 3, Flux.1-dev, AudioCraft 2, StyleGAN3, 3D Gaussian Splatting).
3. 호스팅 대안. 품질/비용/지연 시간 트레이드오프에 따라 순위를 매긴 세 개의 프로덕션 API(fal.ai, Replicate, Stability, Runway, Veo, Kling, ElevenLabs 등).
4. 실패 모드. 선택된 계열의 알려진 병리 현상(모드 붕괴, 노출 편향, 샘플러 드리프트, 토크나이저 아티팩트, CLIP 점수 조작).
5. 예산. 단일 A100에서의 대략적인 학습 시간, 샘플당 추론 비용, VRAM 하한선.

작업이 가능성 점수(scoring)를 요구할 때 GAN을 추천하지 마세요. 고해상도 실시간 사용에 대해 픽셀 기반 자기회귀를 추천하지 마세요. 나열된 오픈 백본이 이미 도메인을 커버한다면 "처음부터 학습"을 추천하는 모든 경우를 플래그하세요.
