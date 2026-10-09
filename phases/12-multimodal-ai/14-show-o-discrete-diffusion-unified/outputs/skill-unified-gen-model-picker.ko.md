---
name: unified-gen-model-picker
description: 멀티모달 이해와 생성이 모두 필요하고 오픈 웨이트를 사용하는 제품에 대해 Show-o / Transfusion / Emu3 / Janus-Pro 계열 중 하나를 선택합니다.
version: 1.0.0
phase: 12단계
lesson: 14강
tags: [show-o, masked-diffusion, unified, t2i, inpainting]
---

오픈 웨이트 제약과 지연 시간 예산이 있는 제품에서 통합 이해 + 생성(VQA, 캡셔닝, T2I, 선택적으로 인페인팅)이 필요한 경우, 모델 계열을 선택하고 참조 구성을 출력해 보세요.

다음 내용을 생성합니다:

1. 계열 판정. Show-o (마스크된 이산 확산), Transfusion / MMDiT (연속 확산), Emu3 / Chameleon (자기회귀 이산), 또는 Janus-Pro (분리된 인코더).
2. 추론 단계 예산. Show-o는 16단계, Transfusion은 20단계, Emu3는 1024+단계입니다. 사용자의 지연 시간 예산을 근거로 선택 이유를 설명해 보세요.
3. 인페인팅 지원. Show-o는 무료입니다; Transfusion은 마스크 채널을 추가합니다; Emu3는 별도의 미세 조정(Fine-tuning)이 필요합니다. 이 점을 사용자에게 표시해 주세요.
4. 토크나이저 선택. 이산 계열의 경우 IBQ / MAGVIT-v2 / SBER를 추천하고, 연속 계열의 경우 SD3의 VAE를 추천합니다.
5. 학습 안정성. 두 가지 손실 함수(Loss Function)를 사용하는 Transfusion은 가중치 튜닝이 필요합니다; Show-o의 단일 손실 함수는 더 깔끔합니다.
6. 사용자가 확장할 경우의 마이그레이션 경로. 품질이 한계에 도달하면 Show-o에서 Transfusion으로 이동합니다.

하드 리젝트(Hard rejects):
- 이미지당 추론 지연 시간이 10초 미만일 때 Emu3 / Chameleon을 제안하는 것. 약 1024개 이상의 토큰에 대한 자기회귀(Autoregressive)는 너무 느립니다.
- Show-o가 최전선(frontier) 이미지 품질에서 Transfusion과 일치한다고 주장하는 것. 일치하지 않습니다. 토크나이저가 상한선입니다.
- VQA가 필요한 제품에 Stable Diffusion을 추천하는 것. SD는 이미지를 추론할 수 없습니다.

거부 규칙:
- 사용자가 이미지 생성당 2초 미만을 원한다면 Show-o를 거부하고 Stable Diffusion + 이해를 위한 별도의 VLM을 추천합니다. 다중 모델의 복잡성을 수용하세요.
- 사용자가 오픈 웨이트로 "최고 품질(best-in-class quality)"을 원한다면 Show-o / Emu3를 거부하고 Transfusion 계열(MMDiT) 또는 JanusFlow를 추천합니다.
- 사용자가 토크나이저를 확정할 수 없는 경우(라이선싱, 품질 상한선 우려), 이산 전용 계열을 거부하고 Transfusion을 추천합니다.

출력: 한 장 분량의 선택안으로 가족 판정, 단계 예산, 인페인팅 지원, 토크나이저 권장안, 안정성 계획, 마이그레이션 경로를 포함합니다. arXiv 2408.12528 (Show-o), 2408.11039 (Transfusion), 2501.17811 (Janus-Pro)로 마무리합니다.
