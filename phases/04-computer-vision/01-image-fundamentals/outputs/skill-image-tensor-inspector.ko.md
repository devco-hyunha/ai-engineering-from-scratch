---
name: skill-image-tensor-inspector
description: 이미지 형태 텐서나 배열을 검사해 dtype, 레이아웃, 범위, raw/normalized/standardized 여부를 보고합니다
version: 1.0.0
phase: 4
lesson: 1
tags: [computer-vision, debugging, preprocessing, tensors]
---

# Image Tensor Inspector

비전 파이프라인의 어느 지점에서든 이미지 형태 배열을 들고 있을 때, 정확히 어떤 상태인지 알아야 할 때 쓰는 진단 스킬입니다.

## When to use

- 사전학습 모델이 쓰레기 예측을 내고 전처리가 의심될 때.
- OpenCV와 torchvision 사이를 옮기며 채널 순서가 불명확할 때.
- 여러 프레임워크의 층을 쌓으며 배치 축이 계속 잘못된 자리에 나타날 때.
- 손실이 `log(num_classes)`에 멈춘 학습 루프를 디버깅할 때.

## Inputs

- `x`: 임의의 2-D, 3-D, 또는 4-D 배열류 (NumPy, PyTorch, JAX).
- Optional `expected`: 검사할 불변식 dict, 예: `{"layout": "CHW", "range": "standardized"}`.

## Steps

1. **Resolve backend** — `x`가 NumPy, Torch, JAX인지 감지합니다. 원본을 바꾸지 않고 검사용으로 NumPy로 변환합니다.

2. **Classify rank**:
   - rank 2 -> 단일 채널 이미지 (H, W).
   - rank 3 -> 마지막 축이 1, 3, 또는 4이고 다른 둘보다 엄밀히 작으면 `HWC`; 아니면 `CHW`.
   - rank 4 -> 축 1이 {1, 3, 4}에 있고 **그리고** 축 2나 축 3이 16보다 크면 `NCHW`를 선호; 아니면 `NHWC`를 선호. 순수 축-1 검사는 `(3, 4, 224, 3)` 같은 작은 이미지 NHWC 배치를 오분류합니다.
   - 모호한 경우(예: `(1, 3, 3, 3)`)는 추측하지 말고 `ambiguous`로 표시하고, 호출자가 `expected`를 제공하게 합니다.

3. **Classify dtype and range**:
   - `uint8` in [0, 255] -> `raw`.
   - `float*` with min >= 0 and max <= 1.01 -> `normalized`.
   - `float*` with min < 0 and |mean| < 0.5 and 0.5 <= std <= 1.5 -> `standardized`.
   - Anything else -> `unusual`, print the histogram.

4. **Per-channel stats** — 채널별 평균과 std를 보고합니다. 배열이 standardized처럼 보이면 ImageNet mean/std와 비교하고 일치 신뢰도를 드러냅니다.

5. **Report** in this exact block:

```
[inspector]
  backend:   numpy | torch | jax
  rank:      2 | 3 | 4
  layout:    HW | HWC | CHW | NHWC | NCHW
  dtype:     <dtype>
  shape:     <shape>
  range:     raw | normalized | standardized | unusual
  min/max:   <min> / <max>
  per-channel mean: [ ... ]
  per-channel std:  [ ... ]
  likely source:    camera | PIL | OpenCV | torchvision | random init
  likely target:    display | training | inference
```

6. **Recommend next action** based on the `likely target`:
   - For `display`: transpose to HWC, clip, convert to uint8.
   - For `training`: standardize with dataset stats, transpose to CHW, add batch axis.
   - For `inference`: match the exact invariants in the model card.

## Rules

- 입력을 절대 변형하지 마세요. 진단만 출력합니다.
- `expected`가 주어지면 모든 불일치를 `[expected X got Y]`로 표시합니다.
- 레이아웃이나 채널 순서가 모호하면 조용한 실패 위험을 지적합니다.
- 한 번에 행동 하나만 권하고, 옵션 목록은 내지 마세요.
