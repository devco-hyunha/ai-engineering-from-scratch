---
name: skill-heatmap-to-coords
description: 모든 프로덕션 포즈 모델이 사용하는 서브픽셀 히트맵-좌표 변환 루틴 작성
version: 1.0.0
phase: 4단계
lesson: 21강
tags: [keypoint, pose, subpixel, inference]
---

# 히트맵에서 좌표로 변환

원시 키포인트 히트맵을 서브픽셀 정밀 좌표로 변환합니다. 모든 포즈 파이프라인에서 가장 비용 효율적인 정확도 향상 방법입니다.

## 사용 시점

- 히트맵 기반 키포인트 모델을 배포할 때.
- 포즈 지표 벤치마킹 시 — OKS는 서브픽셀 정확도에 매우 민감합니다.
- 포즈 코드를 한 프레임워크에서 다른 프레임워크로 이식할 때.

## 입력

- `heatmaps`: `(N, K, H, W)` 텐서, 모델에서 생성된 키포인트별 히트맵.
- `confidence_threshold`: 피크가 이 값 미만인 키포인트는 버립니다.

## 단계

1. **Argmax**를 각 히트맵에 적용하여 정수 피크 위치를 찾습니다.
2. **1차 차이 오프셋** — 인접 픽셀로부터 서브픽셀 오프셋을 추정합니다. `0.25` 계수는 `sigma >= 1`인 가우시안 히트맵에 대해 보정된 휴리스틱입니다. 원리적으로 서브픽셀을 복원하려면 완전한 2차 함수 적합(DARK)이나 가우시안 적합을 사용하세요.

```
dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
```

DARK / 2차 함수 변형의 경우, 지역 2차 함수로 근사합니다:

```
dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
        / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)
```

2차 함수 적합은 피크가 뚜렷한 히트맵에서 더 정확합니다. 히트맵이 잡음이 많을 때는 부호 기반 오프셋이 더 안전한 기본값입니다.

3. 정수 피크에 **오프셋을 더합니다**.
4. **신뢰도** — 키포인트별 피크 값을 반환합니다. 클라이언트는 이를 사용하여 신뢰도가 낮은 예측을 마스킹합니다.
5. **경계 케이스** — 피크가 축을 따라 첫 번째 또는 마지막 픽셀에 위치하면, 인접 픽셀 중 하나가 클램프됩니다. 이때 오프셋은 0으로 수렴하며, 이는 가장 안전한 폴백입니다.

## 출력 템플릿

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

## 보고서

```
[subpixel decode]
  keypoints:   K
  threshold:   <float>
  valid_rate:  fraction of keypoints above threshold
```

## 규칙

- 인덱스를 항상 유효한 범위로 클램프하세요. 가장자리 밖의 키포인트는 차이 오프셋이 0이 되지만, 크래시가 발생하지 않습니다.
- 좌표와 함께 신뢰도를 반환하여, 클라이언트가 신뢰도가 낮은 포인트를 마스킹할 수 있도록 하세요.
- 서브픽셀 정밀도는 히트맵이 피크 주변에서 매끄러운 경우에만 도움이 됩니다. 학습 시 sigma >= 1인 가우시안 타깃을 사용했는지 확인하세요.
- 매우 작은 히트맵 해상도(< 48x48)의 경우, 좌표를 추출하기 전에 히트맵을 전체 이미지 크기로 업샘플링하는 것을 고려해 보세요. 서브픽셀 오프셋은 스트라이드(stride)에 비례하여 증가합니다.
