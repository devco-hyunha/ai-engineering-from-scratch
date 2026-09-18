---
name: skill-segmentation-mask-inspector
description: 클래스 분포, 예측 마스크 통계, 과소 예측되거나 경계가 흐릴 가능성이 큰 클래스를 보고합니다
version: 1.0.0
phase: 4
lesson: 7
tags: [computer-vision, segmentation, debugging, evaluation]
---

# Segmentation Mask Inspector

"손실이 내려갔다"와 "마스크가 실제로 맞아 보인다" 사이의 간격을 위한 진단입니다.

## When to use

- mIoU는 괜찮아 보이는데 시각 검사는 그렇지 않다고 말할 때 학습 실행 직후.
- 배포 전: 예측의 클래스 균형을 정답과 대조.
- 큰 물체의 클래스별 IoU는 높지만 작은 것은 낮을 때.
- 픽셀 수가 작아 IoU에 나타나지 않는 경계 아티팩트를 디버깅할 때.

## Inputs

- `preds`: 예측 클래스 ID의 (N, H, W) 텐서.
- `targets`: 정답 클래스 ID의 (N, H, W) 텐서.
- `num_classes`: integer.
- Optional `class_names`: list of C strings.

## Steps

1. **Class pixel histograms.** Compute the percentage of pixels per class for `preds` and `targets`. Flag any class where `|pred% - gt%| / max(gt%, 1e-6) > 0.30` (relative deviation above 30%). For classes absent from ground truth (`gt% == 0`), flag any predicted share above `0.3` directly.

2. **IoU per class** and **boundary F1 per class**. Boundary F1 is computed by dilating each mask by 3 pixels, intersecting, and scoring. Classes with IoU > 0.7 but boundary F1 < 0.5 are blurring edges.

3. **Small-object recall.** Separate every ground-truth connected component into size buckets (tiny < 100 px, small < 1000 px, medium < 10000 px, large >= 10000 px). Report recall per bucket per class. Small-object recall below 0.3 while large-object recall is above 0.9 indicates a resolution / receptive-field problem.

4. **Confusion pairs.** For each class, find the class it most often confuses with (most common wrong predicted class within its ground-truth mask). Report the top 3 pairs.

5. **Saturation check (requires `probs` or `logits`, not just `preds`).** If the caller passes the raw per-pixel probability distribution `probs: (N, C, H, W)`, compute the fraction of pixels where `probs.max(dim=1) > 0.99` per class. High saturation (>0.9 of a class's pixels) suggests overconfidence — candidate for label smoothing or calibration. When only argmaxed `preds` are available, skip this step and note it in the report.

## Report format

```
[mask-inspector]
  classes: C

[class distribution]
  name       gt %    pred %   delta
  ...

[metrics]
  class       IoU     bF1    recall_tiny  recall_small  recall_medium  recall_large
  ...

[confusion pairs]
  class A confused with class B: <N> pixels (most common)
  class B confused with class A: <N> pixels
  ...

[verdict]
  most impactful issue: <one sentence>
```

## Rules

- 클래스 행을 gt 픽셀 점유율 내림차순으로 정렬해 가장 잦은 클래스가 먼저 오게 하세요.
- IoU < 0.4 또는 boundary F1 < 0.3인 클래스를 `critical`로 표시하세요.
- 소물체 recall이 지배적 실패이면 권하세요: 더 높은 해상도 학습, 마지막 인코더 스테이지의 더 작은 스트라이드, 또는 feature-pyramid 디코더.
- Boundary F1이 지배적 실패이면 권하세요: 경계 인식 손실(Lovasz 또는 BoundaryLoss), 수평 플립 TTA, stride 없는 디코더.
- 클래스 인덱스를 유일한 식별자로 절대 출력하지 마세요; `class_names`가 주어지면 모든 행에서 쓰세요.
