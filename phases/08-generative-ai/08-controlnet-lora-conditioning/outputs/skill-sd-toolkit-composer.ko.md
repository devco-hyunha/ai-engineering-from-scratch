---
name: sd-toolkit-composer
description: 주어진 입력 세트에 대해 SD / Flux 베이스 모델 위에 ControlNet, LoRA, IP-Adapter를 조합합니다.
version: 1.0.0
phase: 8
lesson: 08
tags: [controlnet, lora, ip-adapter, diffusion]
---

작업(대상 이미지), 입력(프롬프트, 참조 이미지, 포즈 / 깊이 / 스케치 / 세그멘테이션, 피사체 정체성), 그리고 베이스 모델(SDXL, SD3.5, Flux.1-dev)이 주어지면 다음을 출력합니다:

1. ControlNet 스택. 어떤 ControlNet(`canny` / `openpose` / `depth` / `scribble` / `seg` / `lineart` / `tile`)을, 어떤 가중치로, 어떤 순서로 사용할지 결정합니다. 가중치의 총합은 1.5 이하여야 합니다.
2. LoRA 스택. 지정된 LoRA 이름, 랭크(`rank`), 알파(`alpha`). `alpha`가 1.5보다 크거나 여러 LoRA가 동일한 개념을 대상으로 하는 경우 경고를 표시합니다.
3. IP-Adapter. 없음, 일반형(plain), 또는 FaceID 변형 중 선택합니다. 일반적인 가중치는 0.4~0.8입니다.
4. 텍스트 프롬프트 + 네거티브 프롬프트. 키워드 순서, 토큰 예산, 네거티브 스캐폴딩(negative scaffolding)을 포함합니다.
5. 샘플러(Sampler) + CFG + 시드(seed). `Euler A` / `DPM-Solver++` / `LCM` 중 선택하며, CFG 스케일은 베이스 모델에 맞춥니다. 재현 가능한 시드 프로토콜을 제공합니다.
6. QA 체크리스트. ControlNet 드리프트(drift), LoRA 과포화(over-saturation), IP-Adapter 정체성 유출(identity leak), 해부학적 문제에 대한 시각적 점검 항목을 포함합니다.

SD 1.5 LoRA를 SDXL 베이스 모델에 적용하는 것은 거부합니다(차원 불일치). 가중치 1.0인 ControlNet을 3개 이상 사용하는 것은 거부합니다(특징 충돌). 사용자의 GPU 예산이 SDXL 또는 Flux를 지원할 수 있는 경우, SD 1.5를 추천하면 플래그를 표시합니다. 10장 미만의 이미지로 LoRA 정체성 학습을 수행하는 경우 과적합(overfit) 가능성이 높다고 플래그를 표시합니다.
