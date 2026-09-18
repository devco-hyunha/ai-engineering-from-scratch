---
name: skill-depth-to-pointcloud
description: 올바른 intrinsics 처리와 .ply보내기로 depth map에서 포인트 클라우드를 만듭니다
version: 1.0.0
phase: 4
lesson: 26
tags: [depth, point-cloud, 3d, intrinsics]
---

# Depth to Point Cloud

Depth map과 컬러 이미지를 텍스처 포인트 클라우드로 바꿔, 시각화나 추가 3D 작업에보낼 수 있게 합니다.

## 언제 쓰나요 (When to use)

- 깊이 예측을 실제 3D 장면으로 시각화할 때.
- 단일 이미지에서 희소 3D 재구성을 부트스트랩할 때.
- SfM이 실패할 때 3DGS 학습 입력을 만들 때.
- 예측 깊이를 LiDAR ground truth와 비교할 때.

## 입력 (Inputs)

- `depth`: 출력에 원하는 단위와 같은 깊이의 `(H, W)` numpy 배열(미터 권장).
- `rgb`: 색의 `(H, W, 3)` numpy 배열(uint8 또는 float32 [0, 1]).
- `intrinsics`: 픽셀 단위 `(fx, fy, cx, cy)`.
- 선택적 `depth_scale`: 예측 깊이 단위를 미터로 바꾸는 배수.

## 파이프라인 (Pipeline)

1. **검증** — 포함할 모든 곳에서 깊이가 양수이고 유한해야 합니다. 무효 픽셀을 마스킹합니다.
2. **리프트** — 픽셀당 `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`.
3. **RGB와 쌍** — 각 3D 점이 매칭 픽셀에서 `(r, g, b)` 삼중항을 받습니다.
4. **내보내기** — PLY(이식성), `.xyz`(경량), `.pcd`(Open3D 네이티브), `.las`/`.laz`(지리공간).

## 구현 템플릿 (Implementation template)

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

## 보고 (Report)

```
[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised
```

## 규칙 (Rules)

- 항상 무효 깊이(0, NaN, inf, 포화)를 마스킹하세요; 포함하면 원점에 쓰레기 점 구름이 생깁니다.
- 상대 깊이 모델의 예측은 메트릭으로보내지 마세요; 관례를 알리려고 출력 파일명에 `relative_` 접두어를 붙이세요.
- 카메라 좌표 관례를 일관되게 유지하세요(OpenCV: +X 오른쪽, +Y 아래, +Z 앞). 다운스트림 도구가 OpenGL(+Y 위)을 기대하면 부호를 바꿉니다.
- 조밀 장면(> 1M 점)에는 subsample 파라미터를 제공하세요; 500 MB가 넘는 PLY는 어디서나 로드하기 어색합니다.
- "합리적" 출력을 위해 깊이를 조용히 클립하지 마세요; 경고된 임계값으로 명시적으로 클립해 사용자가 무엇이 버려졌는지 알게 하세요.
