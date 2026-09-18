---
name: skill-point-cloud-loader
description: 올바른 정규화·중심화·점 샘플링으로 .ply / .pcd / .xyz 파일용 PyTorch Dataset을 작성합니다
version: 1.0.0
phase: 4
lesson: 13
tags: [3d-vision, point-cloud, data-loading, pytorch]
---

# 포인트 클라우드 로더 (Point Cloud Loader)

3D 스캔 파일 폴더를 학습 준비가 된 PyTorch `Dataset`으로 바꿉니다.

## 언제 쓰나요 (When to use)

- 새 포인트 클라우드 분류 / 세그멘테이션 프로젝트를 시작할 때.
- `.ply`, `.pcd`, `.xyz` 형식 사이를 전환할 때.
- 오류 없이 학습하지만 수렴이 나쁜 모델을 디버깅할 때; 종종 데이터 로더 정규화가 잘못됨.

## 입력 (Inputs)

- `data_root`: 포인트 클라우드 파일 폴더와 선택적 라벨 CSV.
- `file_format`: ply | pcd | xyz | npy.
- `num_points`: 고정 샘플링 크기, 보통 1024 또는 2048.
- `augmentation`: none | rotate | jitter | mixup.

## 정규화 정책 (Normalisation policy)

모든 프로덕션 포인트 클라우드 파이프라인은 순서대로 적용합니다:

1. 클라우드를 **중심화**: 중심을 뺍니다.
2. 단위 구로 **스케일**: 중심에서 최대 거리로 나눕니다.
3. `num_points`개 점을 **샘플링**. 클라우드가 더 많으면 충실한 형태 표현을 위해 **farthest point sampling**(FPS) 또는 속도를 위해 무작위 샘플링. 더 적으면 점을 반복.
4. 점 순서를 **셔플**(모델에는 순서가 중요하지 않아야 하지만, 셔플이 우연한 순서 의존을 깨뜨림).

## 출력 템플릿 (Output template)

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
    # Fallback: minimal ascii-ply reader
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
        # Shuffle point order to break any accidental dependencies (especially
        # important when tiling repeats points in deterministic order).
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

## 보고 (Report)

```
[dataset]
  files:          <N>
  format:         <ply|pcd|xyz|npy>
  points_per_sample: <int>
  normalise:      centre + unit sphere
  sampling:       FPS | random
  augmentation:   <list>
```

## 규칙 (Rules)

- 항상 스케일 전에 중심화하세요; 순서를 바꾸면 "단위 구"의 의미가 바뀝니다.
- 형태 태스크에는 무작위 샘플링보다 FPS를 선호하세요; 모든 점이 중요한 세그멘테이션에는 무작위도 괜찮습니다.
- 평가 중에는 절대 증강하지 마세요; 학습 중에만.
- 포인트 클라우드 파일에 색이나 법선이 추가 채널로 있으면, xyz만이 아니라 `(3 + C, num_points)` 텐서를 반환하도록 Dataset을 확장하세요.
