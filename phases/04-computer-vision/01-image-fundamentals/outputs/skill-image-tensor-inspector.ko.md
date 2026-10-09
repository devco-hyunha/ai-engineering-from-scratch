---
name: skill-image-tensor-inspector
description: 이미지 형태의 텐서나 배열을 검사하여 dtype, 레이아웃, 범위, 그리고 원본(raw), 정규화(normalized), 표준화(standardized) 상태인지 보고합니다
version: 1.0.0
phase: 4단계
lesson: 01강
tags: [computer-vision, debugging, preprocessing, tensors]
---

# 이미지 텐서 검사기

비전 파이프라인의 어느 지점에서든 이미지 형태의 배열을 들고 있을 때, 그 배열이 정확히 어떤 상태인지 파악해야 하는 상황에서 사용하는 진단 스킬입니다.

## 사용 시점

- 사전 학습된 모델이 엉뚱한 예측을 반환하여 전처리 과정을 의심할 때.
- OpenCV와 torchvision 간에 파이프라인을 마이그레이션할 때 채널 순서가 불명확한 경우.
- 여러 프레임워크의 레이어를 스택할 때 배치 축이 계속 잘못된 위치에 나타나는 경우.
- 손실(loss)이 `log(num_classes)`에 고착된 학습 루프를 디버깅할 때.

## 입력

- `x`: 모든 2-D, 3-D, 4-D 배열 유사 객체(NumPy, PyTorch, JAX).
- 선택적 `expected`: 검사할 불변 조건(invariants)의 dict, 예: `{"layout": "CHW", "range": "standardized"}`.

## 단계

1. **백엔드 확인** — `x`가 NumPy, Torch, JAX 중 어떤 것인지 감지합니다. 원본을 변경하지 않고 검사하기 위해 NumPy로 변환합니다.

2. **랭크 분류**:
   - 랭크 2 -> 단일 채널 이미지 (H, W).
   - 랭크 3 -> 마지막 축이 1, 3, 4이고 다른 두 축보다 엄격하게 작다면 `HWC`; 그렇지 않으면 `CHW`.
   - 랭크 4 -> 축 1이 {1, 3, 4}에 속하고 **그리고** 축 2 또는 축 3이 16보다 크다면 `NCHW`을 선호합니다; 그렇지 않으면 `NHWC`을 선호합니다. 순수한 축 1 검사만으로는 `(3, 4, 224, 3)`와 같은 작은 이미지 NHWC 배치(batch)를 잘못 분류합니다.
   - 모호한 경우(예: `(1, 3, 3, 3)`)는 추측하지 말고 `ambiguous`으로 항상 플래그를 지정하세요; 호출자가 `expected`를 제공하도록 요구합니다.

3. **dtype 및 범위 분류**:
   - `uint8`가 [0, 255] 범위라면 -> `raw`.
   - `float*`의 min >= 0이고 max <= 1.01이라면 -> `normalized`.
   - `float*`의 min < 0이고 |mean| < 0.5이며 0.5 <= std <= 1.5라면 -> `standardized`.
   - 그 외의 경우 -> `unusual`, 히스토그램을 출력합니다.

4. **채널별 통계** — 채널별 평균과 표준 편차를 보고합니다. 배열이 표준화된 것으로 보이면 ImageNet 평균/표준 편차와 비교하고 일치 신뢰도를 표시합니다.

5. **이 블록 형식으로 보고합니다:**

```
[inspector]
  backend:   numpy | torch | jax
  rank:      2 | 3 | 4
  layout:    HW | HWC | CHW | NHWC | NCHW
  dtype:     <dtype>
  shape:     <shape>
  range:     raw | normalized | standardized | unusual
  min/max:   <min> / <max>
  per-channel mean: [ ... ]
  per-channel std:  [ ... ]
  likely source:    camera | PIL | OpenCV | torchvision | random init
  likely target:    display | training | inference
```

6. **`likely target`에 따라 다음 조치를 권장합니다:**
   - `display`의 경우: HWC로 전치(transpose), 클리핑(clip), uint8로 변환합니다.
   - `training`의 경우: 데이터셋 통계로 표준화하고, CHW로 전치(transpose), 배치 축을 추가합니다.
   - `inference`의 경우: 모델 카드의 정확한 불변 조건(invariants)과 일치시킵니다.

## 규칙

- 입력을 절대 변경하지 마세요. 진단 정보만 출력합니다.
- `expected`가 제공되면 모든 불일치를 `[expected X got Y]`으로 표시합니다.
- 레이아웃이나 채널 순서가 모호할 경우, 조용한 실패(silent-failure) 위험을 명시합니다.
- 옵션 목록이 아닌, 한 번에 하나의 조치만 권장합니다.
