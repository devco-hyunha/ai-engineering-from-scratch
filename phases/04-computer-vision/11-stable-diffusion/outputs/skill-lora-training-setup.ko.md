---
name: skill-lora-training-setup
description: 캡션, 랭크, 배치 크기, 학습률을 포함하여 사용자 정의 데이터셋에 대한 전체 LoRA 학습 구성을 작성합니다.
version: 1.0.0
phase: 4단계
lesson: 11강
tags: [computer-vision, stable-diffusion, lora, fine-tuning]
---

# LoRA 학습 설정

미세 조정 의도에 대한 설명을 `diffusers` 또는 `kohya_ss`에 전달할 준비가 된 구체적인 학습 구성으로 변환합니다.

## 사용 시점

- 주체(사람, 물건, 캐릭터), 스타일(아티스트, 브랜드), 또는 개념(포즈, 조명)에 대한 LoRA를 학습할 때.
- 기존 LoRA를 더 많은 데이터로 확장할 때.
- 출력이 학습 이미지를 과소 적합(underfit)하거나 과적합(overfit)하는 LoRA 실행을 디버깅할 때.

## 입력

- `purpose`: 주체 | 스타일 | 개념
- `num_images`: 사용 가능한 학습 이미지 수
- `base_model`: SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`: 8 | 12 | 16 | 24 | 48+
- `caption_source`: 수동 | BLIP2 생성 | 데이터셋 기본

## 랭크 선택기

| 목적 | 랭크 | Alpha |
|---------|------|-------|
| 주체 | 8-16 | 랭크 |
| 스타일 | 16-32 | 랭크 * 2 |
| 개념 | 32-64 | 랭크 |

랭크가 높을수록 용량이 커지지만, 작은 데이터셋에서는 과적합 위험이 증가합니다. Alpha는 LoRA의 효과 강도를 조절합니다. `alpha == rank`이 안전한 기본값입니다. 스타일은 문서화된 예외입니다: `alpha == rank * 2`은 스타일을 너무 강하게 고정(baking)할 위험을 감수하고 더 강한 스타일 푸시를 제공합니다. 주체 충실도가 목표가 아닐 때만 사용하세요.

## 학습 스텝 목표

- `subject`, 이미지 5-20장: 500-1500 스텝.
- `style`, 이미지 30-100장: 1500-4000 스텝.
- `concept`, 이미지 100장 이상: 4000-10000 스텝.

무분별한 과잉 학습은 위험합니다. 학습 이미지를 암기(memorised)한 LoRA는 일반화(generalise)할 수 없습니다.

## 학습률

- 텍스트 인코더 LoRA: SD 1.5는 `1e-4`, SDXL은 `5e-5`.
- U-Net LoRA: SD 1.5는 `1e-4`, SDXL은 `1e-4`.
- FLUX / SD3: 트랜스포머는 `5e-5`, 텍스트 인코더는 보통 동결(frozen)합니다.
- `num_images < 15` (주체)일 때 또는 3000 스텝 이상 학습할 때 학습률을 절반으로 줄여 보세요. 작은 데이터셋과 긴 학습 모두 더 부드러운 업데이트의 이점을 얻습니다.

## 스케줄러

- `cosine_with_warmup` (기본값): 첫 5-10% 스텝 동안 워밍업한 후 코사인 감쇠를 적용합니다. `steps >= 1000`일 때 사용하세요. 감쇠 꼬리(tail)가 더 선명한 최종 샘플을 제공합니다.
- `constant`: 매우 짧은 학습 (`steps < 500`)에만 사용하거나, 재어닐링(re-annealing) 없이 현재 학습된 특징을 보존하며 이전 LoRA를 재개할 때만 사용하세요.

## 캡션 형식

- 주체: 모든 캡션에 고유한 트리거 토큰("myperson")을 앞에 붙이세요. 트리거 토큰이 기존 개념을 덮어쓰지 않도록 희소하게 유지하세요. 실제 단어와 흔한 이름은 피하세요.
- 스타일: 모든 캡션 끝에 고유한 스타일 태그("...in mystyle style")를 붙이세요. 태그 자체를 희소 트리거 토큰으로 취급하세요. `mystyle`를 사용하며, 이미 실제 개념에 매핑되는 `impressionism`는 사용하지 마세요.
- 개념: 모든 캡션에서 개념을 설명하세요. 트리거 토큰은 없습니다. 개념 자체(예: "low-angle shot")가 앵커입니다.

## 출력 구성

```yaml
model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # 그리고/또는 unet.to_q, to_k, to_v, to_out

training:
  steps:          <int>
  batch_size:     <int, tuned to gpu_vram_gb>
  grad_accum:     <int, usually 1 on >=16 GB, 4 on <=12 GB>
  learning_rate:  <float>
  optimizer:      AdamW8bit | AdamW
  scheduler:      cosine_with_warmup | constant
  warmup_steps:   <int>
  save_every:     <int>

data:
  images_dir:     <path>
  caption_source: <manual | BLIP2 | native>
  trigger_token:   <string if purpose==subject>
  resolution:      <512 for SD 1.5, 1024 for SDXL>
  aspect_ratio_bucketing: true
  augmentation:
    flip:          true
    color_jitter:  false

validation:
  prompts:
    - "<trigger> ...test prompt..."
    - "<trigger> in a different scene"
  every_steps: 250
```

## 보고서

```
[lora setup]
  purpose:   <subject|style|concept>
  base:      <model>
  rank:      <int>
  steps:     <int>
  batch:     <int>   grad_accum: <int>
  lr:        <float>
  vram est.: <float> GB
```

## 규칙

- `rank > 64`를 절대 추천하지 마세요. 그 이상에서는 LoRA가 미니 미세 조정이 되어 '어댑터'로서의 특성을 잃습니다.
- `num_images < 5`에 대해서는 강하게 경고하세요. 1-3장의 이미지로 만든 정체성(identity) LoRA는 매번 과적합됩니다.
- `gpu_vram_gb < 12`에 대해서는 AdamW8bit와 기울기 체크포인팅(Gradient Checkpointing)을 요구하세요.
- `base_model == FLUX`이고 `gpu_vram_gb < 24`이라면 `schnell` 변형으로 라우팅하고 학습이 더 느려진다는 점을 명시하세요.
- 검증 프롬프트를 절대 건너뛰지 마세요. 샘플 그리드가 없는 LoRA는 평가가 불가능합니다.
