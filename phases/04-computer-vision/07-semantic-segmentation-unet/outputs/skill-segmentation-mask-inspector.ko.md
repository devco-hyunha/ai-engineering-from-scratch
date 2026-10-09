---
name: skill-segmentation-mask-inspector
description: 클래스 분포, 예측 마스크 통계, 그리고 과소 예측되거나 경계가 흐려질 가능성이 가장 높은 클래스를 보고합니다.
version: 1.0.0
phase: 4단계
lesson: 07강
tags: [computer-vision, segmentation, debugging, evaluation]
---

# 분할 마스크 검사기

"손실이 감소했다"와 "마스크가 실제로 올바르게 보인다" 사이의 간격을 진단하는 도구입니다.

## 사용 시점

- mIoU는 정상으로 보이지만 시각적 검사가 이를 반증하는 경우, 학습 실행 직후에 사용하세요.
- 배포 전: 예측의 클래스 균형이 정답(ground truth)과 일치하는지 확인하세요.
- 큰 객체에 대한 클래스별 IoU는 높지만 작은 객체에 대한 IoU는 낮은 경우 사용하세요.
- 픽셀 수가 적어 IoU에 나타나지 않는 경계 아티팩트를 디버깅할 때 사용하세요.

## 입력

- `preds`: 예측된 클래스 ID의 (N, H, W) 텐서입니다.
- `targets`: 정답 클래스 ID의 (N, H, W) 텐서입니다.
- `num_classes`: 정수입니다.
- 선택적 `class_names`: C개의 문자열 목록입니다.

## 단계

1. **클래스 픽셀 히스토그램.** `preds`과 `targets`에 대해 클래스별 픽셀 백분율을 계산하세요. `|pred% - gt%| / max(gt%, 1e-6) > 0.30`인 모든 클래스(상대적 편차가 30% 이상)를 플래그로 표시하세요. 정답에 없는 클래스(`gt% == 0`)의 경우, 예측된 비율이 `0.3`를 초과하면 직접 플래그로 표시하세요.

2. **클래스별 IoU** 및 **클래스별 경계 F1.** 경계 F1은 각 마스크를 3픽셀 팽창(dilate)하고 교차한 후 점수를 매겨 계산합니다. IoU가 0.7보다 높지만 경계 F1이 0.5보다 낮은 클래스는 가장자리가 흐려지고 있습니다.

3. **작은 객체 재현율.** 모든 정답 연결 컴포넌트를 크기 버킷(tiny < 100 px, small < 1000 px, medium < 10000 px, large >= 10000 px)으로 분리하세요. 클래스별 버킷별 재현율을 보고하세요. 큰 객체 재현율이 0.9 이상인데 작은 객체 재현율이 0.3 미만인 경우, 해상도/수용장 문제(receptive-field problem)를 나타냅니다.

4. **혼동 쌍.** 각 클래스에 대해 가장 자주 혼동하는 클래스(정답 마스크 내에서 가장 흔한 잘못된 예측 클래스)를 찾으세요. 상위 3개 쌍을 보고하세요.

5. **포화도 확인 (`probs` 또는 `logits`이 필요하며, `preds`만으로는 부족합니다).** 호출자가 원시 픽셀별 확률 분포 `probs: (N, C, H, W)`을 전달하면, 각 클래스에서 `probs.max(dim=1) > 0.99`인 픽셀의 비율을 계산하세요. 높은 포화도(한 클래스 픽셀의 0.9 이상)는 과신뢰를 의미하며, 레이블 스무딩(label smoothing)이나 보정(calibration)의 후보가 됩니다. `preds`만 이용 가능한 경우, 이 단계를 건너뛰고 보고서에 이를 명시하세요.

## 보고서 형식

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

## 규칙

- 클래스 행을 GT 픽셀 점유율 내림차순으로 정렬하여 가장 빈번한 클래스가 먼저 오도록 하세요.
- IoU < 0.4 또는 경계 F1 < 0.3인 클래스는 `critical`으로 표시하세요.
- 작은 객체 재현율이 주요 실패 원인일 경우, 고해상도 훈련, 마지막 인코더 단계에서의 더 작은 스트라이드(stride), 또는 피라미드 디코더(feature-pyramid decoder)를 권장하세요.
- 경계 F1이 주요 실패 원인일 경우, 경계 인식 손실(boundary-aware loss, Lovasz 또는 BoundaryLoss), 수평 뒤집기(horizontal flip)를 포함한 TTA, 그리고 스트라이드 없는(stride-less) 디코더를 권장하세요.
- 클래스 인덱스를 유일한 식별자로 출력하지 마세요. `class_names`이 제공되면 모든 행에서 이를 사용하세요.
