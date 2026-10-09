---
name: skill-depth-to-pointcloud
description: Depth map에서 Point cloud를 구축하고, Intrinsics 처리를 정확하게 수행하여 .ply로 내보내기
version: 1.0.0
phase: 4단계
lesson: 26강
tags: [depth, point-cloud, 3d, intrinsics]
---

# Depth to Point Cloud

Depth map과 색상 이미지를 텍스처가 있는 Point cloud로 변환하여, 시각화나 추가 3D 작업에 사용할 수 있도록 내보냅니다.

## 언제 사용해야 하는가

- Depth 예측을 실제 3D 장면으로 시각화할 때.
- 단일 이미지에서 Sparse 3D 재구성(Sparse 3D Reconstruction)을 시작할 때.
- SfM이 실패했을 때 3DGS 훈련을 위한 입력을 생성할 때.
- 예측된 Depth를 LiDAR Ground truth와 비교할 때.

## 입력

- `depth`: `(H, W)` 출력에서 원하는 단위의 Depth numpy 배열 (미터 권장).
- `rgb`: `(H, W, 3)` 색상 numpy 배열 (uint8 또는 float32 [0, 1]).
- `intrinsics`: `(fx, fy, cx, cy)` 픽셀 단위.
- 선택적 `depth_scale`: 예측된 Depth 단위를 미터로 변환하는 곱셈 계수.

## 파이프라인

1. **검증** — 포함하려는 모든 위치에서 Depth는 양수이고 유한해야 합니다. 유효하지 않은 픽셀은 마스킹 처리하세요.
2. **리프트** — `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`를 픽셀별로 계산하세요.
3. RGB와 **페어링** — 각 3D 점은 매칭되는 픽셀에서 `(r, g, b)` 삼중항을 가져옵니다.
4. **내보내기** — PLY (Portable), `.xyz` (Lightweight), `.pcd` (Open3D-native), `.las`/`.laz` (Geospatial).

## 구현 템플릿

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics, depth_scale=1.0, min_depth=0.1, max_depth=100.0):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def write_ply(path, points, colors=None, valid_mask=None):
    p = points.reshape(-1, 3)
    if valid_mask is not None:
        p = p[valid_mask.flatten()]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {p.shape[0]}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.uint8)
        if valid_mask is not None:
            c = c[valid_mask.flatten()]
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is not None:
            for pt, col in zip(p, c):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
        else:
            for pt in p:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
```

## 보고서

```
[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised
```

## 규칙

- 유효하지 않은 Depth (0, NaN, inf, 포화)는 항상 마스킹 처리하세요. 이를 포함하면 원점에 쓰레기 Point cloud가 생성됩니다.
- Relative-depth 모델로부터 예측한 경우, Metric으로 내보내지 마세요. 출력 파일 이름에 `relative_`를 접두어로 붙여 관례를 표시하세요.
- 카메라 좌표계 관례를 일관되게 유지하세요 (OpenCV: +X 오른쪽, +Y 아래, +Z 앞). Downstream tool이 OpenGL (+Y 위)을 기대한다면 부호를 바꾸세요.
- Dense한 장면 (> 1M 점)의 경우, Subsample 매개변수를 제공하세요. PLY 파일이 500 MB를 넘으면 어디서나 로드하기가 번거롭습니다.
- "합리적"인 출력을 만들기 위해 깊이를 조용히 클리핑하지 마세요. 사용자가 버려진 내용을 알 수 있도록 경고 임계값을 명시적으로 설정하여 클리핑하세요.
