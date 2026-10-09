---
name: skill-dcgan-scaffold
description: z_dim, image_size, num_channels를 기반으로 학습 루프와 샘플 저장기를 포함한 완전한 DCGAN 스캐폴드를 작성합니다
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN 스캐폴드

세 가지 매개변수를 입력으로 받아, 대상 이미지 해상도에 맞게 아키텍처가 적절하게 조정된 실행 가능한 DCGAN 프로젝트 골격을 생성합니다.

## 사용 시점

- 소규모 데이터셋으로 새로운 생성 실험을 시작할 때.
- 작동하는 최소 예제를 통해 DCGAN의 기본 원리를 가르칠 때.
- 조건부 GAN을 프로토타입할 때 (레이블 주입은 동일한 스캐폴드에서 이루어집니다).

## 입력

- `image_size`: 32, 64, 128 중 하나 (2의 거듭제곱이어야 합니다).
- `num_channels`: 1 (그레이스케일) 또는 3 (RGB).
- `z_dim`: 일반적으로 64 또는 128.
- `with_spectral_norm`: yes | no; 기본값은 yes입니다.

## 아키텍처 크기 조정

G의 전치 합성곱 블록 수와 D의 스트라이드 합성곱 블록 수는 `image_size`에 따라 달라집니다:

| image_size | G 블록 | D 블록 |
|------------|----------|----------|
| 32         | 4        | 4        |
| 64         | 5        | 5        |
| 128        | 6        | 6        |

추가 블록마다 G는 공간 차원이 두 배가 되고 D는 절반이 됩니다. 특징(feature) 개수는 32에서 시작하여 `feat_base * 2^block_index`에 따라 스케일링됩니다.

## 출력 파일

- `model.py` — 생성기 + 판별자 클래스
- `train.py` — 학습 루프, 손실, 옵티마이저 설정
- `sample.py` — 샘플 그리드 저장기
- `config.json` — 하이퍼파라미터
- `README.md` — 10줄 빠른 시작

## 보고서

```
[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [list]
  D blocks:         <N>, channels: [list]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 1 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md
```

## 규칙

- G의 출력에 항상 `nn.Tanh()`를 사용하며, 학습 중 데이터를 [-1, 1] 범위로 스케일링합니다.
- D에서는 항상 `LeakyReLU(0.2)`를 사용합니다.
- `with_spectral_norm == yes`인 경우, D의 모든 합성곱을 `spectral_norm()`로 감싸고 D에서 BatchNorm을 제거합니다. G에서는 BatchNorm을 유지합니다.
- image_size가 128을 초과하는 스캐폴드는 절대 생성하지 마세요 — 그 이상에서는 DCGAN이 불안정해집니다. 사용자에게 StyleGAN이나 확산 모델을 안내하세요.
