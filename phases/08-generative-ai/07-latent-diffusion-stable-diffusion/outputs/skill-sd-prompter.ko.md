---
name: sd-prompter
description: 주어진 프롬프트, 스타일, 품질 기준에 대해 Stable Diffusion / Flux 추론을 설정합니다.
version: 1.0.0
phase: 8단계
lesson: 07강
tags: [stable-diffusion, flux, latent-diffusion]
---

프롬프트, 목표 스타일, 품질 기준(빠른 미리보기 / 포트폴리오 품질 / 인쇄용)이 주어지면 다음을 출력합니다:

1. 모델 + 체크포인트. SD 1.5 (레거시 도구), SDXL-base + refiner, SDXL-Turbo (빠른), SD3.5-Large, Flux.1-dev (최고의 오픈), Flux.1-schnell (빠른 오픈), 또는 호스팅 API (DALL-E 3, Imagen 4, Midjourney v7). 한 문장 이유를 포함합니다.
2. 샘플러. Euler A (창의적), DPM-Solver++ 2M Karras (안정적), LCM (빠른), 또는 flow-matching 샘플러(SD3/Flux). 단계 수를 포함합니다.
3. CFG 스케일. turbo / LCM은 0, Flux는 3-4, SDXL은 5-7, SD1.5는 7-10. 트레이드오프를 문서화합니다.
4. 추가 기능. ControlNet (포즈, 깊이, canny, seg), IP-Adapter (참조 이미지), LoRA (스타일 또는 대상), SD3+용 T5 토글.
5. 네거티브 프롬프트. 명시적 빈 문자열 vs 채워진 내용(아티팩트, 저품질, 잘못된 해부학)이 중요합니다. 둘 다 지정합니다.

SDXL+에 대해 CFG > 10을 거부합니다(포화 출력). 레거시 체크포인트가 아닌 경우 샘플러 단계 > 50을 거부합니다(30에서 품질이 평평해짐). 서로 다른 기본 모델에서 학습된 LoRA를 혼합하는 것을 거부합니다(SD 1.5 LoRA를 SDXL에 적용하면 조용히 깨집니다). NSFW, 딥페이크, 저작권 정책에 대한 리마인더 없이 사진처럼 사실적인 인간을 요청하는 모든 요청을 플래그합니다.
