---
name: generative-model-chooser
description: 주어진 작업과 예산에 적합한 생성 모델 계열(family), 백본(backbone), 그리고 호스팅 대안(hosted alternative)을 선택합니다.
version: 1.0.0
phase: 8
lesson: 01
tags: [generative, taxonomy]
---

작업 설명(모달리티, 도메인, 지연 시간 예산, 컴퓨팅 예산, 컨디셔닝 신호)이 주어지면 다음을 출력하세요:

1. **계열(Family)**: 명시적-해석 가능(explicit-tractable), 명시적-근사(explicit-approximate, VAE / diffusion), 암시적(implicit, GAN), 스코어/플로우 매칭(score / flow matching), 또는 토큰-AR(token-AR) 중 하나를 선택하세요. 모달리티와 지연 시간을 근거로 한 한 문장 분량의 이유를 포함해야 합니다.
2. **백본(Backbone) + 오픈 레퍼런스(open reference)**: 사용자가 즉시 미세 조정(fine-tune)할 수 있는 사전 학습된 오픈 웨이트 모델 하나를 제시하세요 (예: `Stable Diffusion 3`, `Flux.1-dev`, `AudioCraft 2`, `StyleGAN3`, `3D Gaussian Splatting`).
3. **호스팅 대안(Hosted alternatives)**: 품질, 비용, 지연 시간 간의 트레이드오프에 따라 순위를 매긴 세 가지 프로덕션 API를 제시하세요 (예: `fal.ai`, `Replicate`, `Stability`, `Runway`, `Veo`, `Kling`, `ElevenLabs` 등).
4. **실패 모드(Failure mode)**: 선택한 계열에서 나타나는 알려진 병리적 현상(모드 붕괴(mode collapse), 노출 편향(exposure bias), 샘플러 드리프트(sampler drift), 토크나이저 아티팩트(tokenizer artifacts), CLIP-score 게이밍(CLIP-score gaming))을 기술하세요.
5. **예산(Budget)**: 단일 `A100` 기준 대략적인 학습 시간, 샘플당 추론 비용, 최소 VRAM 요구 사양을 제시하세요.

작업에 가능도 점수 산출(likelihood scoring)이 필요한 경우 GAN을 추천하는 것을 거부하세요. 고해상도 실시간 사용이 필요한 경우 픽셀 기반 자기회귀(autoregressive-over-pixels) 모델을 추천하는 것을 거부하세요. 나열된 오픈 백본이 이미 해당 도메인을 커버하고 있다면, "처음부터 학습(train from scratch)"하라는 권고에 주의 표시(flag)를 하세요.
