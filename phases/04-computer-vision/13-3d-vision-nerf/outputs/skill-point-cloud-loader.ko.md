---
name: skill-point-cloud-loader
description: .ply / .pcd / .xyz 파일을 위한 PyTorch Dataset 작성. 올바른 정규화, 중심화, 포인트 샘플링 포함
version: 1.0.0
phase: 4단계
lesson: 13강
tags: [3d-vision, point-cloud, data-loading, pytorch]
---

# 포인트 클라우드 로더

3D 스캔 파일이 있는 폴더를 바로 학습할 수 있는 PyTorch `Dataset`로 변환합니다.

## 사용 시점

- 새로운 포인트 클라우드 분류 / 분할 프로젝트를 시작할 때.
- `.ply`, `.pcd`, `.xyz` 형식 간에 전환할 때.
- 오류 없이 학습되지만 수렴이 잘 안 되는 모델을 디버깅할 때. 종종 데이터 로더의 정규화가 잘못된 경우입니다.

## 입력

- `data_root`: 포인트 클라우드 파일이 있는 폴더와 선택적 레이블 CSV 파일.
- `file_format`: ply | pcd | xyz | npy.
- `num_points`: 고정된 샘플링 크기, 일반적으로 1024 또는 2048.
- `augmentation`: none | rotate | jitter | mixup.

## 정규화 정책

모든 프로덕션 포인트 클라우드 파이프라인은 순서대로 다음을 적용합니다:

1. **중심화(Centre)**: 클라우드의 중심점(centroid)을 빼서 중심에 위치시킵니다.
2. **스케일링(Scale)**: 단위 구(unit sphere)로 맞추기 위해 중심으로부터의 최대 거리로 나눕니다.
3. **샘플링(Sample)**: `num_points`개의 포인트를 샘플링합니다. 클라우드에 더 많은 포인트가 있다면, 형태를 충실히 표현하기 위해 **최원점 샘플링(farthest point sampling, FPS)**을 사용하거나, 속도를 위해 랜덤 샘플링을 사용합니다. 포인트가 더 적다면 포인트를 반복합니다.
4. **셔플(Shuffle)**: 포인트 순서를 섞습니다 (모델에는 순서가 중요하지 않아야 하지만, 셔플은 우연한 순서 의존성을 깨뜨립니다).

## 출력 템플릿

```python
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def _read_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    # 폴백: 최소한의 ascii-ply 리더
    ...

def _fps(points, k):
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    seed = np.random.randint(len(points))
    idx[0] = seed
    for i in range(1, k):
        dist = np.minimum(dist, ((points - points[idx[i-1]]) ** 2).sum(axis=1))
        idx[i] = int(np.argmax(dist))
    return idx

def normalise(points):
    centre = points.mean(axis=0)
    points = points - centre
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / max(scale, 1e-8)

class PointCloudDataset(Dataset):
    def __init__(self, files, labels, num_points=1024, augment=False):
        self.files = files
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pts = _read_ply(self.files[i])
        pts = normalise(pts)
        if len(pts) >= self.num_points:
            idx = _fps(pts, self.num_points)
            pts = pts[idx]
        else:
            reps = int(np.ceil(self.num_points / len(pts)))
            pts = np.tile(pts, (reps, 1))[:self.num_points]
        # 우연한 의존성을 깨기 위해 포인트 순서를 셔플합니다 (특히
        # 타일링이 결정적인 순서로 포인트를 반복할 때 중요합니다).
        np.random.shuffle(pts)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
            pts = pts @ R
            pts = pts + np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        return torch.from_numpy(pts).transpose(0, 1), int(self.labels[i])
```

## 보고서

```
[dataset]
  files:          <N>
  format:         <ply|pcd|xyz|npy>
  points_per_sample: <int>
  normalise:      centre + unit sphere
  sampling:       FPS | random
  augmentation:   <list>
```

## 규칙

- 스케일링 전에 항상 중심화하세요. 순서를 바꾸면 "단위 구"의 의미가 달라집니다.
- 형태 관련 작업에서는 FPS를 랜덤 샘플링보다 선호하세요. 모든 포인트가 중요한 분할 작업에서는 랜덤 샘플링도 괜찮습니다.
- 평가 중에는 절대 증강하지 마세요. 학습 중에만 증강하세요.
- 점 구름 파일에 색상 또는 법선 벡터가 추가 채널로 포함되어 있다면, Dataset이 xyz뿐만 아니라 `(3 + C, num_points)` 텐서를 반환하도록 확장해 보세요.
