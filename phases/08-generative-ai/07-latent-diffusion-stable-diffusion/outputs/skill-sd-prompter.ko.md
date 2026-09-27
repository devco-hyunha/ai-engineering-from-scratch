---
name: sd-prompter
description: 주어진 프롬프트, 스타일 및 품질 기준에 맞춰 Stable Diffusion / Flux 추론을 구성합니다.
version: 1.0.0
phase: 8
lesson: 07
tags: [stable-diffusion, flux, latent-diffusion]
---

프롬프트, 목표 스타일, 품질 기준(빠른 미리보기 / 포트폴리오 품질 / 인쇄 가능 품질)이 주어지면 다음을 출력하세요:

1. **모델 + 체크포인트(Model + checkpoint)**: SD 1.5(레거시 도구), SDXL-base + refiner, SDXL-Turbo(빠름), SD3.5-Large, Flux.1-dev(최고의 오픈 모델), Flux.1-schnell(빠른 오픈 모델), 또는 호스팅 API(DALL-E 3, Imagen 4, Midjourney v7) 중 하나를 선택하고, 선택 이유를 한 문장으로 설명하세요.
2. **샘플러(Sampler)**: Euler A(창의적), DPM-Solver++ 2M Karras(안정적), LCM(빠름), 또는 flow-matching 샘플러(SD3/Flux)를 선택하고, 스텝(step) 수를 포함하세요.
3. **CFG 스케일(CFG scale)**: Turbo/LCM의 경우 0, Flux의 경우 3-4, SDXL의 경우 5-7, SD1.5의 경우 7-10을 적용하세요. 설정에 따른 트레이드오프(trade-off)를 기록하세요.
4. **애드온(Add-ons)**: ControlNet(pose, depth, canny, seg), IP-Adapter(참조 이미지), LoRA(스타일 또는 피사체), SD3+를 위한 T5 토글 등을 제안하세요.
5. **부정 프롬프트(Negative prompt)**: 빈 문자열로 둘지, 아니면 명시적인 내용(artifacts, low quality, wrong anatomy 등)을 포함할지 결정하여 지정하세요.

**제약 사항:**
- SDXL+ 모델에 대해 CFG > 10 설정은 거부하세요(출력물이 포화됨).
- 레거시 모델이 아닌 체크포인트에서 샘플러 스텝이 50을 초과하는 경우 거부하세요(30스텝 이후 품질 정체).
- 서로 다른 베이스 모델로 학습된 LoRA를 혼합하는 요청은 거부하세요(예: SD 1.5 LoRA를 SDXL에 사용하는 것은 제대로 작동하지 않음).
- 실사 인물(photorealistic humans)에 대한 요청이 있을 경우, NSFW, 딥페이크 및 저작권 정책에 대한 주의 사항을 안내하지 않으면 플래그를 표시하세요.
