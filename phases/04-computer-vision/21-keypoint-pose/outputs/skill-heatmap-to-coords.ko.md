---
name: skill-heatmap-to-coords
description: 모든 프로덕션 포즈 모델이 쓰는 서브픽셀 히트맵→좌표 루틴을 작성
version: 1.0.0
phase: 4
lesson: 21
tags: [keypoint, pose, subpixel, inference]
---

# Heatmap to Coords

원시 키포인트 히트맵을 서브픽셀 정밀 좌표로 바꿉니다. 모든 포즈 파이프라인에서 가장 싼 정확도 업그레이드입니다.

## When to use

- 히트맵 기반 키포인트 모델을 배포할 때.
- 포즈 지표를 벤치마크할 때 — OKS는 서브픽셀 정확도에 매우 민감합니다.
- 한 프레임워크에서 다른 프레임워크로 포즈 코드를 옮길 때.

## Inputs

- `heatmaps`: `(N, K, H, W)` 텐서, 모델의 키포인트별 히트맵.
- `confidence_threshold`: 피크가 이 값 미만인 키포인트는 버림.

## Steps

1. **Argmax** — 각 히트맵에서 정수 피크 위치를 찾습니다.
2. **1차 차분 오프셋** — 이웃 픽셀에서 서브픽셀 오프셋을 추정합니다. `0.25` 계수는 `sigma >= 1`인 가우시안 히트맵에 보정된 휴리스틱입니다. 원칙적인 서브픽셀 복구에는 전체 2차 피팅(DARK) 또는 가우시안 피팅을 쓰세요.

```
dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
```

DARK / 2차 변형에서는 국소 2차로 근사합니다.

```
dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
        / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)
```

2차 피팅은 뾰족한 히트맵에서 더 정확하고, 부호 기반 오프셋은 히트맵이 노이즈일 때 더 안전한 기본값입니다.

3. **오프셋 추가** — 정수 피크에 더합니다.
4. **신뢰도** — 키포인트당 피크 값을 반환합니다. 클라이언트가 저신뢰 예측을 마스크하는 데 씁니다.
5. **경계 케이스** — 피크가 축의 첫/마지막 픽셀에 있으면 이웃 하나가 클램프되고, 오프셋은 0으로 붕괴합니다. 가장 안전한 폴백입니다.

## Output template

```python
import torch

def heatmap_to_coords_subpixel(heatmaps, threshold=0.2):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    ys_int = ys.long()
    xs_int = xs.long()

    x_minus = (xs_int - 1).clamp(min=0)
    x_plus = (xs_int + 1).clamp(max=W - 1)
    y_minus = (ys_int - 1).clamp(min=0)
    y_plus = (ys_int + 1).clamp(max=H - 1)

    batch_idx = torch.arange(N).view(-1, 1).expand(-1, K)
    kp_idx = torch.arange(K).view(1, -1).expand(N, -1)

    dx_raw = (heatmaps[batch_idx, kp_idx, ys_int, x_plus]
              - heatmaps[batch_idx, kp_idx, ys_int, x_minus])
    dy_raw = (heatmaps[batch_idx, kp_idx, y_plus, xs_int]
              - heatmaps[batch_idx, kp_idx, y_minus, xs_int])
    dx = 0.25 * torch.sign(dx_raw)
    dy = 0.25 * torch.sign(dy_raw)

    at_left = xs_int == 0
    at_right = xs_int == (W - 1)
    at_top = ys_int == 0
    at_bottom = ys_int == (H - 1)
    dx = torch.where(at_left | at_right, torch.zeros_like(dx), dx)
    dy = torch.where(at_top | at_bottom, torch.zeros_like(dy), dy)

    refined_x = xs + dx
    refined_y = ys + dy
    coords = torch.stack([refined_x, refined_y], dim=-1)
    mask = conf >= threshold
    return coords, conf, mask
```

## Report

```
[subpixel decode]
  keypoints:   K
  threshold:   <float>
  valid_rate:  fraction of keypoints above threshold
```

## Rules

- 이웃 인덱스를 항상 유효 범위로 클램프하세요. 가장자리 키포인트는 차분 오프셋이 0이지만 크래시는 없습니다.
- 클라이언트가 저신뢰 점을 마스크할 수 있도록 좌표와 함께 신뢰도를 반환하세요.
- 서브픽셀 정제는 피크 주변 히트맵이 매끄러울 때만 돕습니다 — 학습이 sigma >= 1인 가우시안 타깃을 썼는지 확인하세요.
- 히트맵 해상도가 매우 작으면(< 48×48) 좌표를 추출하기 전에 히트맵을 전체 이미지 크기로 업샘플하세요. 서브픽셀 오프셋은 stride와 함께 스케일합니다.
