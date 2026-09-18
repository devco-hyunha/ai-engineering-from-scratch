---
name: skill-mask-rcnn-head-swapper
description: 커스텀 num_classes에 맞춰 torchvision Mask R-CNN의 박스·마스크 헤드를 교체하는 정확한 코드를 생성합니다
version: 1.0.0
phase: 4
lesson: 8
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# Mask R-CNN 헤드 교체기 (Mask R-CNN Head Swapper)

Mask R-CNN 전용 헤드 교체 보일러플레이트를 만듭니다. 아래 템플릿은 `maskrcnn_resnet50_fpn`과 `maskrcnn_resnet50_fpn_v2`에만 존재하는 `model.roi_heads.box_predictor`와 `model.roi_heads.mask_predictor`를 가정합니다. Faster R-CNN은 박스 예측기는 있지만 마스크 예측기가 없고, RetinaNet은 `RetinaNetHead`를 쓰며 `roi_heads`가 아예 없습니다 — 둘 다 다른 스킬이 필요합니다.

## 언제 쓰나요 (When to use)

- 커스텀 클래스 집합에서 `maskrcnn_resnet50_fpn` 또는 `maskrcnn_resnet50_fpn_v2`를 파인튜닝할 때.
- COCO로 학습된 Mask R-CNN 체크포인트를 비-COCO 클래스 수로 옮길 때.
- `cls_score.out_features` 또는 `mask_predictor` 불일치로 학습이 크래시날 때 디버깅할 때.

## 범위 밖 (Out of scope)

- `fasterrcnn_*` — mask_predictor 없음. `box_predictor`만 교체; 별도 Faster R-CNN 헤드 교체 레시피 사용.
- `retinanet_*` — `roi_heads` 없음; 분류·회귀 헤드는 `model.head.classification_head`와 `model.head.regression_head` 아래. RetinaNet 전용 스킬 사용.
- `keypointrcnn_*` — `mask_predictor` 대신 `keypoint_predictor` 사용.

## 입력 (Inputs)

- `model_name`: torchvision 검출 모델 생성자, 예: `maskrcnn_resnet50_fpn_v2`.
- `num_classes`: 배경 포함. 객체 클래스 4개 데이터셋이면 `num_classes=5`.
- `freeze`: `backbone`, `backbone_fpn`, `none` 중 하나.

## 단계 (Steps)

1. 모델 생성자와 두 예측기 클래스(`FastRCNNPredictor`, `MaskRCNNPredictor`)를 import합니다.
2. 기본 가중치 사전학습 모델을 로드합니다.
3. `model.roi_heads.box_predictor`를 새 `FastRCNNPredictor(in_features, num_classes)`로 교체합니다.
4. `model.roi_heads.mask_predictor`를 새 `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)`로 교체합니다.
5. 요청된 동결 정책을 적용합니다.
6. 모듈별 학습 가능 파라미터를 나열하는 확인 블록을 출력합니다.

## 출력 코드 템플릿 (Output code template)

```python
from torchvision.models.detection import {MODEL_NAME}, {MODEL_WEIGHTS}
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes={NUM_CLASSES}):
    model = {MODEL_NAME}(weights={MODEL_WEIGHTS}.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    {FREEZE_BLOCK}

    return model
```

여기서 `{FREEZE_BLOCK}`은:

- `none` -> 비움
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  ```

## 보고 (Report)

```
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## 규칙 (Rules)

- 배경을 포함하지 않은 `num_classes`를 절대 권하지 마세요; 항상 사용자에게 상기시키세요.
- 가능하면 항상 torchvision 검출 모델의 `_v2` 변형을 쓰세요; 레거시보다 사전학습 가중치가 낫습니다.
- 이 스킬 안에서 모델을 인스턴스화하지 마세요 — 코드 블록을 만들고 사용자가 실행하게 하세요.
- 사용자가 10,000장 초과 데이터셋에서 `freeze backbone`을 요청하면, 백본도 파인튜닝하는 것을 고려하라고 제안하세요.
