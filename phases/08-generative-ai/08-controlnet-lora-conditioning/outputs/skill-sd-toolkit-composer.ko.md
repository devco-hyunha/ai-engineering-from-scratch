---
name: sd-toolkit-composer
description: 주어진 입력에 대해 SD / Flux 기반 모델 위에 ControlNet, LoRA, IP-Adapter를 구성합니다.
version: 1.0.0
phase: 8단계
lesson: 08강
tags: [controlnet, lora, ip-adapter, diffusion]
---

작업(타겟 이미지), 입력(프롬프트, 참조 이미지, 포즈 / 깊이 / 스크리블 / 세그, 대상 신원), 그리고 기반 모델(SDXL, SD3.5, Flux.1-dev)이 주어지면 다음을 출력합니다:

1. ControlNet 스택. 어떤 ControlNet(canny / openpose / depth / scribble / seg / lineart / tile)을 어떤 가중치로 어떤 순서로 적용할지. 가중치 합은 &lt;= 1.5를 초과할 수 없습니다.
2. LoRA 스택. 이름 있는 LoRA, 랭크, 알파. 알파가 &gt; 1.5이거나 여러 LoRA가 같은 개념을 가리킬 때 경고합니다.
3. IP-Adapter. 없음, 일반, 또는 FaceID 변형; 가중치는 일반적으로 0.4-0.8입니다.
4. 텍스트 프롬프트 + 네거티브 프롬프트. 키워드 순서, 토큰 예산, 네거티브 스캐폴딩.
5. 샘플러 + CFG + 시드. Euler A / DPM-Solver++ / LCM; CFG 스케일은 기반 모델에 연결됩니다. 재현 가능한 시드 프로토콜.
6. QA 체크리스트. ControlNet 드리프트, LoRA 과포화, IP-Adapter 신원 누출, 해부학적 문제를 시각적으로 확인합니다.

SD 1.5 LoRA를 SDXL 기반 모델 위에 스택하는 것을 거부합니다(차원 불일치). 가중치가 각각 1.0인 3개 이상의 ControlNet을 실행하는 것을 거부합니다(특징 충돌). 사용자가 SDXL이나 Flux를 위한 GPU 예산을 가지고 있을 때 SD 1.5 권장 사항을 플래그합니다. 10장 미만의 이미지로 LoRA 신원 훈련을 하는 것은 과적합될 가능성이 높으므로 플래그합니다.
