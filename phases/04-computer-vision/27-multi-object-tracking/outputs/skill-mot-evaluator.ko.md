---
name: skill-mot-evaluator
description: Ground-truth 트랙에 대한 MOTA / IDF1 / HOTA의 완전한 평가 하네스를 작성합니다
version: 1.0.0
phase: 4
lesson: 27
tags: [mot, evaluation, tracking, metrics]
---

# MOT Evaluator

트래커 출력을 표준 MOTA/IDF1/HOTA 파이프라인으로 감싸 문헌과 공정하게 비교합니다.

## 언제 쓰나요 (When to use)

- MOT17 / MOT20 / DanceTrack / SportsMOT에서 새 트래커를 벤치마크할 때.
- 자체 영상에서 ByteTrack vs BoT-SORT vs SAM 2를 비교할 때.
- 논문이나 PR 설명용으로 재현 가능한 숫자를 만들 때.

## 입력 (Inputs)

- `predictions`: 프레임당 `(track_id, x, y, w, h, confidence)` 튜플 목록.
- `ground_truth`: 프레임당 `(gt_id, x, y, w, h)` 튜플 목록.
- `iou_threshold`: MOTA에 전형적으로 0.5; HOTA는 스윕을 씀.
- `evaluator`: `py-motmetrics` (MOTA, IDF1) 또는 `TrackEval` (HOTA).

## 출력 형식 계약 (Output format contract)

`py-motmetrics`와 `TrackEval` 둘 다 특정 온디스크 포맷을 기대합니다:

```
# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

프레임은 1-indexed, 박스는 (x1, y1, x2, y2)가 아니라 (x, y, w, h)입니다. 변환이 대부분의 통합 버그가 사는 곳입니다.

## 단계 (Steps)

1. 트래커 출력을 MOT Challenge 텍스트 포맷으로 변환합니다.
2. 두 파일에 `py-motmetrics.io.loadtxt`를 돌립니다.
3. `mm.metrics.create().compute()`로 MOTA + IDF1을 계산합니다.
4. HOTA에는 같은 파일과 `Metrics: HOTA`로 `TrackEval`을 호출합니다.
5. 대시보드용으로 결과를 JSON으로 저장합니다.

## 구현 스케치 (Implementation sketch)

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

## 보고 (Report)

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

## 규칙 (Rules)

- 출력 텍스트 파일에서 항상 1-indexed 프레임을 쓰세요; MOT 도구가 이를 기대합니다.
- 쓰기 전에 (x1, y1, x2, y2)를 (x, y, w, h)로 변환하세요.
- 현대 비교에서 MOTA만 보고하지 마세요; IDF1과 HOTA를 포함하세요.
- MOT17의 private vs public 검출을 주의하세요 — 별도로 평가되며 섞으면 점수가 부풀려집니다.
- 시퀀스별 점수를 로깅하세요; 집계가 어려운 단일 시퀀스의 실패를 숨깁니다.
