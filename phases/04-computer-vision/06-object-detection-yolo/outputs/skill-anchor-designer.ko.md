---
name: skill-anchor-designer
description: 정답 박스 데이터셋이 주어지면 (w, h)에 k-means를 돌리고 FPN 레벨별 앵커 집합과 커버리지 통계를 반환합니다
version: 1.0.0
phase: 4
lesson: 6
tags: [computer-vision, detection, anchors, kmeans]
---

# Anchor Designer

앵커는 앵커 기반 검출기에서 가장 데이터셋 특화된 하이퍼파라미터입니다. 기본 COCO 앵커는 세포 배양 이미지, 위성 타일, 소물체 감시에서 저성능입니다. 이 스킬은 타깃 데이터에 실제로 맞는 앵커를 도출합니다.

## When to use

- 새 데이터셋의 첫 학습 실행 전.
- 그 외에는 건강한 모델에서 아주 작거나 아주 큰 물체의 recall이 약할 때.
- 박스 크기 분포가 이동했을 수 있는 대규모 데이터셋 확장 후.

## Inputs

- `boxes`: `(cx, cy, w, h)` 또는 `(x1, y1, x2, y2)` 형식의 shape (N, 4) numpy 배열; 양성 박스 최소 1000개 권장.
- `num_anchors_per_level`: usually 3.
- `num_fpn_levels`: usually 3 (P3, P4, P5) or 4.
- `input_size`: training-resolution HxW.
- Optional `strides`: per-level strides; when omitted, take the first `num_fpn_levels` entries of `[8, 16, 32, 64]`. Pass a longer or shorter array explicitly if the detector's FPN has different strides.

## Steps

1. **Normalise boxes** to `(w, h)` pairs in pixel units at `input_size`. Drop any with w or h < 2 pixels.

2. **Run k-means** on `(w, h)` pairs, with `k = num_anchors_per_level * num_fpn_levels`. Use `1 - IoU(box, cluster)` as the distance function, not Euclidean distance — Euclidean on `(w, h)` collapses thin tall boxes and square boxes together. All boxes contribute equally (unweighted); if you have a class-imbalanced dataset and want larger-box recall, repeat rare-class boxes in the input array rather than passing a weight vector.

3. **Sort clusters by area** ascending. Split into `num_fpn_levels` groups of `num_anchors_per_level`. Smallest areas go to the highest-resolution level (smallest stride).

4. **Compute coverage statistics** per level:
   - `median IoU` of each ground-truth box to its best anchor at that level.
   - `recall@IoU=0.5` — percentage of boxes whose best anchor has IoU >= 0.5.
   - `area coverage` — fraction of boxes whose area falls within `[anchor_min_area / 4, anchor_max_area * 4]` of the level.

5. **Report per-level anchors** and flag levels where `recall@IoU=0.5 < 0.9`; that level's anchors do not match the data well and should be retuned or the number of anchors per level increased.

## Report format

```
[anchor-designer]
  total boxes:         <N>
  clusters:            <k>
  distance metric:     1 - IoU

[level P3  stride=8]
  anchors (w, h):      [(A, B), (C, D), (E, F)]
  median IoU:          <X>
  recall@IoU=0.5:      <X>
  coverage:            <X>
  flag:                ok | retune

[level P4  stride=16]
  ...

[summary]
  overall recall@IoU=0.5: <X>
  smallest anchor:        <w x h>
  largest anchor:         <w x h>
  recommendation:         <one sentence if any level flagged>
```

## Rules

- 항상 IoU 기반 거리를 쓰세요; 유클리드 k-means는 시각적으로는 합리적이지만 경험적으로 더 나쁜 앵커를 만듭니다.
- 클러스터를 면적으로 정렬한 뒤 오름차순으로 레벨에 할당하세요.
- `num_anchors_per_level = 1`이면 k-means를 아예 건너뛰세요: 박스를 면적 분위수로 `num_fpn_levels` 빈에 나누고(예: 3레벨이면 terciles), 각 레벨의 앵커를 빈별 중앙값 (w, h)로 설정하세요. 작은 데이터셋에서 `k = num_fpn_levels`로 k-means를 돌리는 것보다 견고합니다.
- 음수 앵커 차원을 절대 출력하지 마세요; 1에서 clamp하세요.
- 데이터셋에 박스 < 200개이면 앵커 검색이 신뢰할 수 없다고 경고하고 기본 COCO 앵커와 더 많은 학습 데이터를 권하세요.
