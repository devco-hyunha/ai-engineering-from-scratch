---
name: skill-mask-rcnn-head-swapper
description: torchvision Mask R-CNN의 box 및 mask 헤드를 custom num_classes에 맞게 교체하는 정확한 코드를 생성합니다
version: 1.0.0
phase: 4단계
lesson: 08강
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# Mask R-CNN 헤드 스와퍼

Mask R-CNN 전용 헤드 교체 템플릿을 생성합니다. 아래 템플릿은 `model.roi_heads.box_predictor`와 `model.roi_heads.mask_predictor`을 가정하며, 이 요소들은 `maskrcnn_resnet50_fpn`와 `maskrcnn_resnet50_fpn_v2`에만 존재합니다. Faster R-CNN은 box predictor가 있지만 mask predictor가 없으며, RetinaNet은 `RetinaNetHead`를 사용하며 `roi_heads`가 전혀 없습니다. 두 경우 모두 다른 스킬이 필요합니다.

## 사용 시점

- custom class set에 대해 `maskrcnn_resnet50_fpn` 또는 `maskrcnn_resnet50_fpn_v2`을 미세 조정(Fine-tuning)할 때.
- COCO로 학습된 Mask R-CNN 체크포인트(Checkpoint)를 non-COCO class count에 적용할 때.
- `cls_score.out_features` 또는 `mask_predictor` 불일치로 인해 Mask R-CNN 학습 실행이 충돌할 때 디버깅할 때.

## 범위 밖

- `fasterrcnn_*` — mask_predictor가 없음. `box_predictor`만 교체하세요. Faster R-CNN 헤드 교체 레시피를 별도로 사용하세요.
- `retinanet_*` — `roi_heads`가 없음. classifier + regression 헤드는 `model.head.classification_head`와 `model.head.regression_head` 아래에 있습니다. RetinaNet 전용 스킬을 사용하세요.
- `keypointrcnn_*` — `mask_predictor` 대신 `keypoint_predictor`를 사용합니다.

## 입력

- `model_name`: torchvision detection model constructor, 예: `maskrcnn_resnet50_fpn_v2`.
- `num_classes`: background 포함. 4개 object class dataset은 `num_classes=5`를 의미합니다.
- `freeze`: `backbone`, `backbone_fpn`, `none` 중 하나.

## 단계

1. model constructor와 두 predictor classes (`FastRCNNPredictor`, `MaskRCNNPredictor`)를 import하세요.
2. default-weights pretrained model을 로드하세요.
3. `model.roi_heads.box_predictor`를 새로운 `FastRCNNPredictor(in_features, num_classes)`로 교체하세요.
4. `model.roi_heads.mask_predictor`를 새로운 `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)`로 교체하세요.
5. 요청된 freeze policy를 적용하세요.
6. module별 trainable params를 나열하는 확인 블록을 출력하세요.

## 출력 코드 템플릿

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

`{FREEZE_BLOCK}`는 다음과 같습니다:

- `none` -> empty
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters는 backbone.fpn 내부에 있습니다
  ```

## 보고서

```
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## 규칙

- 배경이 포함되지 않은 `num_classes`을 추천하지 마세요. 항상 사용자에게 상기시켜 주세요.
- 가능한 경우 torchvision 탐지 모델의 `_v2` 변형을 항상 사용하세요. 레거시 모델보다 사전 학습 가중치가 더 좋습니다.
- 이 스킬 내에서 모델을 인스턴스화하지 마세요. 코드 블록을 생성하고 사용자가 실행하도록 하세요.
- 사용자가 10,000장 이상의 이미지 데이터셋에서 `freeze backbone`을 요청하는 경우, 백본도 미세 조정하는 것을 고려하도록 제안하세요.
