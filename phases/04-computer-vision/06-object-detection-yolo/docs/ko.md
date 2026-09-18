# 물체 검출 — 처음부터 만드는 YOLO (Object Detection — YOLO from Scratch)

> 검출(detection)은 분류에 회귀를 더한 것으로, 특징 맵의 모든 위치에서 실행한 뒤 비최대 억제(non-maximum suppression)로 정리합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 05 (Transfer Learning)
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- 검출을 밀집 예측 문제로 바꾸는 격자·앵커 설계를 설명하고, 출력 텐서의 모든 숫자가 무엇을 의미하는지 말합니다
- 박스 간 Intersection-over-Union을 계산하고 비최대 억제를 처음부터 구현합니다
- 사전학습 백본 위에 최소 YOLO 스타일 헤드를 만들고, 분류·objectness·박스 회귀 손실을 포함합니다
- 검출 지표 행(precision@0.5, recall, mAP@0.5, mAP@0.5:0.95)을 읽고 다음에 돌릴 손잡이를 고릅니다

## 문제 상황 (The Problem)

분류는 "이 이미지는 개다"라고 말합니다. 검출은 "픽셀 (112, 40, 280, 210)에 개가 있고, (400, 180, 560, 310)에 고양이가 있으며, 프레임에 다른 것은 없다"라고 말합니다. 그 한 구조적 변화 — 이미지당 라벨 하나가 아니라 가변 개수의 라벨된 박스를 예측 — 가 모든 자율 시스템, 감시 제품, 문서 레이아웃 파서, 공장 비전 라인이 의존하는 것입니다.

검출은 또한 비전의 모든 공학 트레이드오프가 한꺼번에 나타나는 곳입니다. 정확한 박스(회귀 헤드), 각 박스의 올바른 클래스(분류 헤드), 검출할 것이 없을 때를 아는 모델(objectness 점수), 실제 물체당 정확히 하나의 예측(비최대 억제)이 필요합니다. 이 중 하나라도 놓치면 파이프라인은 물체를 놓치거나, 환각 박스를 보고하거나, 같은 물체를 약간 다른 위치에서 열다섯 번 예측합니다.

YOLO(You Only Look Once, Redmon et al. 2016)는 conv net의 단일 forward 패스로 이 모든 것을 실시간으로 돌리게 만든 설계이며, 같은 구조적 결정이 여전히 현대 검출기(YOLOv8, YOLOv9, YOLO-NAS, RT-DETR)의 백본입니다. 핵심을 배우면 모든 변형이 같은 부품의 재배열이 됩니다.

## 핵심 개념 (The Concept)

### 밀집 예측으로서의 검출

분류기는 이미지당 C개 숫자를 출력합니다. YOLO 스타일 검출기는 이미지당 `(S x S x (5 + C))`개 숫자를 출력하며, S는 공간 격자 크기입니다.

```mermaid
flowchart LR
    IMG["입력 416x416 RGB"] --> BB["백본<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["특징 맵<br/>(C_feat, 13, 13)"]
    FM --> HEAD["검출 헤드<br/>(1x1 convs)"]
    HEAD --> OUT["출력 텐서<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["디코드<br/>(격자 + sigmoid + exp)"]
    DEC --> NMS["비최대 억제"]
    NMS --> RESULT["최종 박스"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

`S * S` 격자 셀 각각이 `B`개 박스를 예측합니다. 각 박스마다:

- 4개 숫자가 기하를 기술합니다: `tx, ty, tw, th`.
- 1개 숫자가 objectness 점수입니다: "이 셀에 중심이 있는 물체가 있는가?"
- C개 숫자가 클래스 확률입니다.

셀당 합계: `B * (5 + C)`. VOC에서 `S=13, B=2, C=20`이면 셀당 숫자 50개입니다.

### 왜 격자와 앵커인가

평탄한 회귀는 모든 물체에 대해 `(x, y, w, h)`를 절대 좌표로 예측합니다. 이미지를 옮겨도 모든 예측이 같은 양만큼 옮겨져서는 안 되기 때문에 conv 네트워크에는 어렵습니다 — 각 물체는 공간적으로 고정됩니다. 격자는 각 정답 박스를 중심이 떨어지는 격자 셀에 할당해 이 문제에 답합니다; 그 셀만 그 물체에 책임집니다.

앵커는 두 번째 문제를 다룹니다. 3x3 conv는 16픽셀 수용 영역 특징 셀에서 500픽셀 너비 박스를 쉽게 회귀할 수 없습니다. 대신 셀당 `B`개의 사전 박스 형태(앵커)를 미리 정의하고 각 앵커로부터의 작은 델타를 예측합니다. 모델은 아무것도 없는 상태에서 회귀하는 대신 맞는 앵커를 고르고 살짝 밀치는 법을 배웁니다.

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

현대 검출기는 종종 해상도마다 다른 앵커 집합으로 FPN을 씁니다 — 얕고 고해상도 맵에 작은 앵커, 깊고 저해상도 맵에 큰 앵커. 같은 아이디어, 더 많은 스케일.

### 예측 디코딩

원시 `tx, ty, tw, th`는 박스 좌표가 아닙니다; 그리기 전에 변환할 회귀 타깃입니다:

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid`는 중심 오프셋을 셀 안에 둡니다. `exp`는 부호 뒤집힘 없이 앵커에서 너비가 자유롭게 스케일되게 합니다. `stride`는 격자 좌표를 다시 픽셀로 스케일합니다. 이 디코드 단계는 v2 이후 모든 YOLO 버전에서 같습니다.

### IoU

두 박스 사이의 검출 보편 유사도 지표:

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1은 동일; IoU = 0은 겹침 없음. 예측과 정답 박스 사이의 IoU가 예측이 진양성으로 세어지는지 결정합니다(보통 IoU >= 0.5). 두 예측 사이의 IoU는 NMS가 중복 제거에 씁니다.

### 비최대 억제 (Non-maximum suppression)

인접 앵커로 학습된 conv 네트워크는 같은 물체에 겹치는 박스를 자주 예측합니다. NMS는 최고 신뢰도 예측을 유지하고 IoU가 임계값 위인 다른 예측을 삭제합니다.

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

전형적 임계값: 물체 검출에서 0.45. 최근 검출기는 표준 NMS를 `soft-NMS`, `DIoU-NMS`로 바꾸거나 억제를 직접 학습하지만(RT-DETR) 구조적 목적은 같습니다.

### 손실

YOLO 손실은 가중치를 두고 더한 세 손실입니다:

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

물체를 담은 셀만 박스 회귀와 분류 손실에 기여합니다. 물체가 없는 셀은 objectness 손실에만 기여합니다(모델이 침묵하도록 가르침). `lambda_noobj`는 보통 작습니다(~0.5). 셀의 대다수가 비어 있어 그렇지 않으면 총 손실을 지배하기 때문입니다.

현대 변형은 MSE 박스 손실을 CIoU / DIoU로 바꾸고(IoU를 직접 최적화), 클래스 불균형에 focal loss를 쓰며, objectness를 quality focal loss로 균형합니다. 세 구성 요소 구조는 변하지 않습니다.

### 검출 지표

정확도는 검출로 전이되지 않습니다. 전이되는 숫자 넷:

- **Precision@IoU=0.5** — 양성으로 센 예측 중 실제로 맞는 비율.
- **Recall@IoU=0.5** — 실제 물체 중 찾은 비율.
- **AP@0.5** — IoU 임계값 0.5에서 precision-recall 곡선 면적; 클래스당 숫자 하나.
- **mAP@0.5:0.95** — IoU 임계값 0.5, 0.55, ..., 0.95에 걸친 AP 평균. COCO 지표; 가장 엄격하고 정보량이 큼.

넷 모두를 보고하세요. mAP@0.5에는 강하고 mAP@0.5:0.95에는 약한 검출기는 대략 위치하지만 타이트하지 않습니다; 더 나은 박스 회귀 손실로 고치세요. 높은 precision과 낮은 recall은 너무 보수적입니다; 신뢰도 임계값을 낮추거나 objectness 가중치를 올리세요.

```figure
object-detection-nms
```

## 직접 만들기 (Build It)

### 1단계: IoU

전체 레슨의 일꾼. `(x1, y1, x2, y2)` 형식의 두 박스 배열에서 동작합니다.

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

쌍별 IoU의 `(N_a, N_b)` 행렬을 반환합니다. 배열 하나를 shape `(1, 4)`로 만들어 단일 정답 박스에 대해 쓰세요.

### 2단계: 비최대 억제

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

결정적이며, 정렬에서 `O(N log N)`이고, 동일 입력에서 `torchvision.ops.nms` 동작과 맞습니다.

### 3단계: 박스 인코딩과 디코딩

픽셀 좌표와 네트워크가 실제로 회귀하는 `(tx, ty, tw, th)` 타깃 사이를 변환합니다.

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

테스트: 박스를 인코드한 뒤 디코드 — 원본에 매우 가까운 것을 얻어야 합니다(`tx`가 post-sigmoid 범위에 있지 않을 때 sigmoid 역이 완벽히 가역이 아닌 점까지).

### 4단계: 최소 YOLO 헤드

특징 맵 위의 1x1 conv 하나, `(B, S, S, num_anchors, 5 + C)`로 reshape.

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

출력 shape: `(N, H, W, num_anchors, 5 + C)`. 마지막 차원은 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`를 담습니다.

### 5단계: 정답 할당

모든 정답 박스에 대해 어느 `(cell, anchor)`가 책임인지 결정합니다.

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

앵커 선택은 "정답과의 최고 shape IoU" — YOLOv2/v3 할당과 맞는 싼 프록시입니다. v5 이후는 같은 아이디어를 정제하는 더 정교한 전략(task-aligned matching, dynamic k)을 씁니다.

### 6단계: 세 손실

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

모든 YOLO 튜토리얼이 하드코딩하거나 스윕하는 하이퍼파라미터 다섯. 비율이 중요합니다: `lambda_coord=5, lambda_noobj=0.5`는 원본 YOLOv1 논문을 반영하며 여전히 합리적인 기본값으로 동작합니다.

### 7단계: 추론 파이프라인

원시 헤드 출력을 디코드하고, sigmoid/exp를 적용하고, objectness로 임계값을 걸고, NMS합니다.

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

그것이 완전한 평가 경로입니다: head -> decode -> threshold -> NMS.

## 활용하기 (Use It)

`torchvision.models.detection`은 같은 개념 구조의 프로덕션 검출기를 싣습니다. 사전학습 모델 로드는 세 줄입니다.

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

실시간 추론 파이프라인에서는 `ultralytics`(YOLOv8/v9)가 표준입니다: `from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`. 모델이 디코딩과 NMS를 내부에서 처리하고 위에서 만든 것과 같은 `boxes / scores / labels` 삼중을 반환합니다.

## 산출물 (Ship It)

이 레슨이 만드는 것:

- `outputs/prompt-detection-metric-reader.md` — `precision, recall, AP, mAP@0.5:0.95` 행을 한 줄 진단과 가장 유용한 다음 실험 하나로 바꾸는 프롬프트.
- `outputs/skill-anchor-designer.md` — 정답 박스 데이터셋이 주어지면 `(w, h)`에 k-means를 돌리고 FPN 레벨별 앵커 집합과 올바른 앵커 수를 고르는 데 필요한 커버리지 통계를 반환하는 스킬.

## 연습 문제 (Exercises)

1. **(Easy)** `box_iou`를 구현하고 랜덤 박스 쌍 1,000개에서 `torchvision.ops.box_iou`와 대조하세요. 최대 절대 차이가 `1e-6` 미만인지 검증하세요.
2. **(Medium)** `yolo_loss`를 MSE 대신 `CIoU` 박스 손실을 쓰는 버전으로 포팅하세요. 100장 합성 데이터셋에서 같은 에폭 수에 CIoU가 MSE보다 더 나은 최종 mAP@0.5:0.95로 수렴함을 보이세요.
3. **(Hard)** 다중 스케일 추론을 구현하세요: 같은 이미지를 세 해상도로 모델에 넣고, 박스 예측을 합친 뒤, 끝에 단일 NMS를 돌리세요. 홀드아웃 집합에서 단일 스케일 추론 대비 mAP 상승을 측정하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Anchor | "박스 사전" | 네트워크가 절대 좌표 대신 델타를 예측하는 각 격자 셀의 미리 정의된 박스 형태 |
| IoU | "겹침" | 두 박스의 intersection-over-union; 검출의 보편 유사도 척도 |
| NMS | "중복 제거" | 최고 점수 예측을 유지하고 임계값 위 겹침을 제거하는 탐욕 알고리즘 |
| Objectness | "여기에 뭔가 있는가" | 해당 셀에 물체 중심이 있는지 예측하는 앵커·셀당 스칼라 |
| Grid stride | "다운샘플 인수" | 격자 셀당 픽셀; 13격자 헤드의 416px 입력은 스트라이드 32 |
| mAP | "평균 평균 정밀도" | precision-recall 곡선 아래 면적의 평균, 클래스와 (COCO의 경우) IoU 임계값에 걸쳐 평균 |
| AP@0.5 | "PASCAL VOC AP" | IoU 임계값 0.5의 평균 정밀도; 지표의 관대한 버전 |
| mAP@0.5:0.95 | "COCO AP" | IoU 임계값 0.5..0.95 step 0.05에 걸친 평균; 엄격한 버전이자 현재 커뮤니티 표준 |

## 더 읽을거리 (Further Reading)

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 창립 논문; 이후 모든 YOLO는 이 구조의 정제
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 다중 스케일 FPN 스타일 헤드를 도입한 논문; 여전히 가장 명확한 다이어그램
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) — 현재 프로덕션 참고; 데이터셋 형식, 증강, 학습 레시피 포함
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 전체 검출기 zoo의 최고 평이한 영어 투어; DETR, RetinaNet, FCOS, YOLO가 어떻게 관련되는지 이해하는 데 귀중함
