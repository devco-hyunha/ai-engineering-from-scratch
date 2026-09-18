---
name: skill-dcgan-scaffold
description: z_dim, image_size, num_channels로부터 학습 루프와 샘플 저장기를 포함한 완전한 DCGAN 스캐폴드를 작성합니다
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN 스캐폴드 (DCGAN Scaffold)

세 파라미터가 주어지면, 목표 이미지 해상도에 맞게 아키텍처 크기가 잡힌 실행 가능한 DCGAN 프로젝트 골격을 내보냅니다.

## 언제 쓰나요 (When to use)

- 작은 데이터셋에서 새 생성 실험을 시작할 때.
- 동작하는 최소 예제로 DCGAN 기초를 가르칠 때.
- 조건부 GAN을 프로토타이핑할 때(라벨 주입은 같은 스캐폴드에서 이뤄짐).

## 입력 (Inputs)

- `image_size`: 32, 64, 128 중 하나(2의 거듭제곱이어야 함).
- `num_channels`: 1(그레이스케일) 또는 3(RGB).
- `z_dim`: 보통 64 또는 128.
- `with_spectral_norm`: yes | no; 기본값 yes.

## 아키텍처 크기 (Architecture sizing)

G의 전치 컨볼루션 블록 수와 D의 stride 컨볼루션 블록 수는 `image_size`에 따라 달라집니다:

| image_size | G blocks | D blocks |
|------------|----------|----------|
| 32         | 4        | 4        |
| 64         | 5        | 5        |
| 128        | 6        | 6        |

블록이 하나 늘 때마다 G는 공간 차원을 두 배, D는 절반으로 만듭니다. 특징 수는 32에서 시작해 `feat_base * 2^block_index`로 스케일합니다.

## 출력 파일 (Output files)

- `model.py` — Generator + Discriminator 클래스
- `train.py` — 학습 루프, 손실, 옵티마이저 설정
- `sample.py` — 샘플 그리드 저장기
- `config.json` — 하이퍼파라미터
- `README.md` — 10줄 퀵스타트

## 보고 (Report)

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

## 규칙 (Rules)

- 항상 G 출력에 `nn.Tanh()`를 쓰고, 학습 중 데이터를 [-1, 1]로 스케일하세요.
- 항상 D에 `LeakyReLU(0.2)`를 쓰세요.
- `with_spectral_norm == yes`이면 D의 모든 conv를 `spectral_norm()`으로 감싸고 D에서 BatchNorm을 제거하세요. G에는 BatchNorm을 유지하세요.
- image_size > 128용 스캐폴드는 절대 내보내지 마세요 — 그 이상에서 DCGAN은 불안정해집니다; 사용자를 StyleGAN 또는 확산 모델로 안내하세요.
