---
name: skill-anchor-designer
description: 정답 박스 데이터셋이 주어지면 (w, h)에 대해 k-means를 실행하고 FPN 레벨별 앵커 세트 및 커버리지 통계를 반환합니다
version: 1.0.0
phase: 4단계
lesson: 06강
tags: [computer-vision, detection, anchors, kmeans]
---

# 앵커 디자이너

앵커는 앵커 기반 탐지 모델에서 데이터셋에 가장 특화된 하이퍼파라미터입니다. 기본 COCO 앵커는 세포 배양 이미지, 위성 타일, 소형 물체 감시 데이터셋에서 성능이 떨어집니다. 이 스킬은 실제 타겟 데이터와 일치하는 앵커를 도출합니다.

## 사용 시점

- 새로운 데이터셋에서 첫 번째 학습을 시작하기 전.
- 매우 작거나 매우 큰 물체에 대한 재현율이 다른 부분은 정상인 모델에서 약할 때.
- 박스 크기 분포가 변경될 수 있는 주요한 데이터셋 확장 이후.

## 입력

- `boxes`: (N, 4) 형태의 numpy 배열로, `(cx, cy, w, h)` 또는 `(x1, y1, x2, y2)` 형식이며, 최소 1000개의 양의 박스를 권장합니다.
- `num_anchors_per_level`: 보통 3입니다.
- `num_fpn_levels`: 보통 3 (P3, P4, P5) 또는 4입니다.
- `input_size`: 학습 해상도 HxW입니다.
- 선택적 `strides`: 레벨별 스트라이드입니다. 생략하면 `[8, 16, 32, 64]`의 첫 `num_fpn_levels` 항목을 사용합니다. 탐지 모델의 FPN 스트라이드가 다른 경우, 더 길거나 짧은 배열을 명시적으로 전달하세요.

## 단계

1. **박스를 정규화**하여 `input_size`에서 픽셀 단위의 `(w, h)` 쌍으로 변환합니다. w 또는 h가 2 픽셀 미만인 박스는 제거합니다.

2. **k-means를 실행**하여 `(w, h)` 쌍을 `k = num_anchors_per_level * num_fpn_levels`으로 클러스터링합니다. 유클리드 거리가 아닌 `1 - IoU(box, cluster)`를 거리 함수로 사용하세요. `(w, h)`에 대한 유클리드 거리는 가늘고 긴 박스와 정사각형 박스를 함께 묶습니다. 모든 박스는 동일하게 기여합니다 (가중치 없음). 클래스 불균형 데이터셋에서 큰 박스의 재현율을 높이고 싶다면, 가중치 벡터를 전달하는 대신 희소 클래스의 박스를 입력 배열에 반복하세요.

3. **클러스터를 면적 순으로 정렬**하여 오름차순으로 분할합니다. `num_anchors_per_level`개의 `num_fpn_levels` 그룹으로 나눕니다. 가장 작은 면적은 가장 높은 해상도 레벨 (가장 작은 스트라이드)에 할당됩니다.

4. **레벨별 커버리지 통계를 계산**합니다:
   - 각 레벨에서 각 정답 박스를 가장 좋은 앵커에 매핑합니다. `median IoU`
   - `recall@IoU=0.5` — 가장 좋은 앵커의 IoU가 0.5 이상인 박스의 비율입니다.
   - `area coverage` — 면적이 `[anchor_min_area / 4, anchor_max_area * 4]` 이내인 상자의 비율입니다.

5. **레벨별 앵커 보고**를 수행하고 `recall@IoU=0.5 < 0.9`인 레벨을 표시하세요. 해당 레벨의 앵커는 데이터와 잘 맞지 않으므로 재조정하거나 레벨별 앵커 수를 늘려야 합니다.

## 보고 형식

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

## 규칙

- 항상 IoU 기반 거리를 사용하세요. 유클리드 k-means는 시각적으로 합리적이지만 경험적으로 더 나쁜 앵커를 생성합니다.
- 클러스터를 면적 순으로 정렬한 후, 오름차순으로 레벨에 할당하세요.
- `num_anchors_per_level = 1`인 경우, k-means를 완전히 건너뛰세요. 상자를 면적 분위수(예: 3개 레벨의 경우 삼분위)에 따라 `num_fpn_levels`개의 빈으로 나누고, 각 레벨의 앵커를 빈별 중위값 (w, h)으로 설정하세요. 이는 소규모 데이터셋에서 `k = num_fpn_levels`로 k-means를 실행하는 것보다 더 견고합니다.
- 음수 앵커 차원을 출력하지 마세요. 1로 클램프(clamp)하세요.
- 데이터셋에 상자가 200개 미만인 경우, 앵커 검색이 신뢰할 수 없으며 기본 COCO 앵커와 더 많은 훈련 데이터를 사용할 것을 권장한다는 경고 메시지를 사용자에게 표시하세요.
