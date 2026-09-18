---
name: skill-lora-training-setup
description: 캡션, 랭크, 배치 크기, 학습률을 포함한 커스텀 데이터셋용 전체 LoRA 학습 설정을 작성합니다
version: 1.0.0
phase: 4
lesson: 11
tags: [computer-vision, stable-diffusion, lora, fine-tuning]
---

# LoRA 학습 설정 (LoRA Training Setup)

파인튜닝 의도 설명을 `diffusers` 또는 `kohya_ss`에 넘길 준비가 된 구체적 학습 설정으로 바꿉니다.

## 언제 쓰나요 (When to use)

- 피사체(사람, 물체, 캐릭터), 스타일(아티스트, 브랜드), 또는 개념(포즈, 조명)용 LoRA를 학습할 때.
- 기존 LoRA를 더 많은 데이터로 확장할 때.
- 출력이 학습 이미지에 과소적합하거나 과적합하는 LoRA 실행을 디버깅할 때.

## 입력 (Inputs)

- `purpose`: subject | style | concept
- `num_images`: 사용 가능한 학습 이미지 수
- `base_model`: SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`: 8 | 12 | 16 | 24 | 48+
- `caption_source`: manual | BLIP2-generated | dataset-native

## 랭크 선택기 (Rank picker)

| Purpose | Rank | Alpha |
|---------|------|-------|
| Subject | 8-16 | rank |
| Style | 16-32 | rank * 2 |
| Concept | 32-64 | rank |

더 높은 랭크 = 더 많은 용량, 작은 데이터셋에서 더 많은 과적합 위험. Alpha는 LoRA 효과 강도를 스케일합니다; `alpha == rank`가 안전한 기본값입니다. 스타일은 문서화된 예외입니다: `alpha == rank * 2`는 스타일을 너무 세게 굽는 위험을 대가로 더 강한 스타일 푸시를 줍니다 — 피사체 충실도가 목표가 아닐 때만 쓰세요.

## 학습 스텝 목표 (Training step target)

- 5–20장 `subject`: 500–1500 스텝.
- 30–100장 `style`: 1500–4000 스텝.
- 100장 이상 `concept`: 4000–10000 스텝.

과도하면 위험합니다 — 학습 이미지를 외운 LoRA는 일반화할 수 없습니다.

## 학습률 (Learning rate)

- 텍스트 인코더 LoRA: SD 1.5는 `1e-4`, SDXL은 `5e-5`.
- U-Net LoRA: SD 1.5는 `1e-4`, SDXL은 `1e-4`.
- FLUX / SD3: 트랜스포머는 `5e-5`, 텍스트 인코더는 보통 동결.
- `num_images < 15`(subject)이거나 3000 스텝 이상 학습할 때 LR을 절반으로; 아주 작은 데이터셋과 긴 실행 모두 더 부드러운 갱신의 이득을 봅니다.

## 스케줄러 (Scheduler)

- `cosine_with_warmup`(기본): 처음 5–10% 스텝에 걸쳐 워밍업, 그다음 코사인 감쇠. `steps >= 1000`일 때 사용; 감쇠 꼬리가 더 선명한 최종 샘플을 줍니다.
- `constant`: 매우 짧은 실행(`steps < 500`)이거나, 재어닐링 없이 현재 학습된 특징을 보존하며 이전 LoRA를 재개할 때만 사용.

## 캡션 형식 (Caption format)

- Subject: 모든 캡션 앞에 고유 트리거 토큰("myperson")을 붙이세요. 기존 개념을 덮어쓰지 않도록 트리거 토큰을 희귀하게 유지하세요. 실제 단어와 흔한 이름은 피하세요.
- Style: 모든 캡션 끝에 고유 스타일 태그를 붙이세요("...in mystyle style"). 태그 자체를 희귀 트리거 토큰으로 취급하세요 — 이미 실제 개념에 매핑된 `impressionism`이 아니라 `mystyle`.
- Concept: 모든 캡션에서 개념을 설명하세요; 트리거 토큰 없음. 개념 자체(예: "low-angle shot")가 앵커입니다.

## 출력 설정 (Output config)

```yaml
model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # and/or unet.to_q, to_k, to_v, to_out

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

## 보고 (Report)

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

## 규칙 (Rules)

- `rank > 64`를 절대 권하지 마세요; 그 이상이면 LoRA가 미니 파인튜닝이 되어 "어댑터" 성질을 잃습니다.
- `num_images < 5`이면 강하게 경고하세요 — 1–3장 이미지의 정체성 LoRA는 매번 과적합합니다.
- `gpu_vram_gb < 12`이면 AdamW8bit와 gradient checkpointing을 요구하세요.
- `base_model == FLUX`이고 `gpu_vram_gb < 24`이면 `schnell` 변형으로 보내고 학습이 더 느리다고 메모하세요.
- 검증 프롬프트를 절대 건너뛰지 마세요; 샘플 그리드 없는 LoRA는 평가할 수 없습니다.
