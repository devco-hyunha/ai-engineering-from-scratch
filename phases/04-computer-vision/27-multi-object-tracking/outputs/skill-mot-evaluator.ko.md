---
name: skill-mot-evaluator
description: MOTA / IDF1 / HOTA를 ground-truth 트랙에 대해 평가하는 완전한 평가 하네스를 작성합니다
version: 1.0.0
phase: 4단계
lesson: 27강
tags: [mot, evaluation, tracking, metrics]
---

# MOT 평가기

트래커의 출력을 표준 MOTA/IDF1/HOTA 파이프라인에 감싸서 문헌과 공정하게 비교할 수 있도록 해 보세요.

## 사용 시점

- MOT17 / MOT20 / DanceTrack / SportsMOT에서 새로운 트래커를 벤치마킹할 때.
- 자체 영상에서 ByteTrack, BoT-SORT, SAM 2를 비교할 때.
- 논문이나 PR 설명에 재현 가능한 수치를 산출할 때.

## 입력

- `predictions`: 프레임별 `(track_id, x, y, w, h, confidence)` 튜플 목록.
- `ground_truth`: 프레임별 `(gt_id, x, y, w, h)` 튜플 목록.
- `iou_threshold`: MOTA에서는 0.5가 일반적이며, HOTA는 스윕(sweep)을 사용합니다.
- `evaluator`: `py-motmetrics` (MOTA, IDF1) 또는 `TrackEval` (HOTA).

## 출력 형식 계약

`py-motmetrics`과 `TrackEval` 모두 디스크에 저장되는 특정 형식을 기대합니다:

```
# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

프레임은 1부터 시작하며, 박스는 (x1, y1, x2, y2)가 아닌 (x, y, w, h)입니다. 변환은 대부분의 통합 버그가 발생하는 지점입니다.

## 단계

1. 트래커의 출력을 MOT Challenge 텍스트 형식으로 변환하세요.
2. 두 파일 모두에 `py-motmetrics.io.loadtxt`을 실행하세요.
3. `mm.metrics.create().compute()`을 사용하여 MOTA + IDF1을 계산하세요.
4. HOTA의 경우, 동일한 파일과 `Metrics: HOTA`을 사용하여 `TrackEval`을 호출하세요.
5. 대시보드를 위해 결과를 JSON으로 저장하세요.

## 구현 개요

```python
import motmetrics as mm

def evaluate_mota_idf1(pred_path, gt_path):
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=0.5)
    metrics = mm.metrics.create().compute(
        acc, metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches"]
    )
    return metrics


def write_mot_txt(predictions, path):
    with open(path, "w") as f:
        for frame_idx, detections in enumerate(predictions, start=1):
            for tid, x, y, w, h, conf in detections:
                f.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
```

## 보고서

```
[mot evaluation]
  frames:     <int>
  gt tracks:  <int>
  pred tracks: <int>

[metrics]
  MOTA:       <float>
  MOTP:       <float>
  IDF1:       <float>
  IDP/IDR:    <float/float>
  ID switches: <int>
  HOTA:       <float>  (from TrackEval)
```

## 규칙

- 출력 텍스트 파일에서는 항상 1부터 시작하는 프레임 번호를 사용하세요. MOT 도구들이 이를 기대합니다.
- 쓰기 전에 (x1, y1, x2, y2)를 (x, y, w, h)로 변환하세요.
- 현대적인 비교에서는 MOTA만 보고하지 마세요. IDF01강 HOTA를 포함하세요.
- MOT17의 private vs public 검출에 주의하세요. 이들은 별도로 평가되며, 혼합하면 점수가 부풀려집니다.
- 시퀀스별 점수를 기록하세요. 집계는 단일 어려운 시퀀스의 실패를 숨깁니다.
