---
name: skill-conv-shape-calculator
description: CNN 사양을 레이어별로 순회하며 모든 블록의 출력 형상, 수용 영역 및 매개변수 수를 보고합니다
version: 1.0.0
phase: 4
lesson: 2
tags: [computer-vision, cnn, architecture, debugging]
---

# Conv 형상 계산기

CNN을 계획하거나 디버깅할 때 사용하는 결정론적 헬퍼입니다. 입력 형상과 레이어 사양 목록이 주어지면, 모델을 실행하지 않고도 형상, 수용 영역 및 매개변수 수를 추적합니다.

## 사용 시점

- 새로운 CNN을 설계할 때 모든 다운샘플링이 깔끔한 크기로 떨어지는지 확인하고 싶을 때.
- 논문에서 아키텍처 표를 코드로 변환할 때.
- 사전 학습된 백본이 분류기 헤드에서 형상 불일치로 충돌하며, 어떤 레이어가 공간 크기를 변경했는지 알아야 할 때.
- 학습 전에 두 백본의 매개변수 효율성을 비교할 때.

## 입력

- `input_shape`: `(C, H, W)`.
- `layers`: 레이어 딕셔너리의 순서 있는 목록. 각 항목은 다음을 지원합니다:
  - `{type: "conv", c_out, k, s, p, groups=1, bias=true}`
  - `{type: "pool", mode: "max"|"avg", k, s, p=0}`
  - `{type: "adaptive_pool", out_h, out_w}`
  - `{type: "flatten"}`
  - `{type: "linear", out_features, bias=true}`

## 단계

1. **추적 초기화**는 `(C, H, W)`, 수용 영역 `1`, 유효 스트라이드 `1`, 누적 매개변수 `0`으로 시작합니다.

2. **각 레이어에 대해**, 이 순서로 업데이트합니다:
   - `C_out` (conv/linear)를 계산하거나, `C_in`을 (pool)로 전달합니다.
   - conv와 pool에는 `(H + 2P - K) / S + 1`, adaptive pool에는 `out_h/out_w`, linear 앞의 flatten 출력 형상 `(1, 1)`, linear에는 스칼라 `(C * H * W, 1, 1)` `1x1` 를 사용하여 공간 출력 형상을 계산합니다.
   - 수용 영역과 유효 스트라이드를 업데이트합니다:
     - Conv/pool: `RF_new = RF_old + (K - 1) * effective_stride`, `effective_stride *= S`.
     - Adaptive pool: 유효 `S = H_in / out_h` (내림)를 가진 pool로 취급합니다. `RF_new = RF_old + (H_in - 1) * effective_stride_old`; `effective_stride *= S`. Adaptive pool의 수용 영역(RF)은 이전 전체 공간 범위와 같다는 점에 유의하세요.
     - Flatten / linear: 수용 영역과 유효 스트라이드는 더 이상 의미가 없으므로 flatten 이전 값으로 고정하고 이후 행에서는 생략합니다.
   - 매개변수를 계산합니다:
     - Conv: `C_out * (C_in / groups) * K * K + (C_out if bias else 0)`.
     - Linear: `out_features * in_features + (out_features if bias else 0)`.
     - 풀링 및 평탄화: 0.

3. **문제를 감지**하고 플래그를 지정합니다:
   - 정수가 아닌 출력 크기 (스트라이드/패딩 정렬 불일치).
   - 스택 끝내기 전에 `H_out <= 0`.
   - 입력 크기를 초과하는 수용 영역 (이 시점 이후 불필요한 연산 발생 가능성).
   - 레이어별 매개변수에서 갑작스러운 10배 급증은 잘못된 채널 계획임을 시사합니다.

4. **단일 표로 보고**합니다:

```
idx  layer                C_in  C_out  K  S  P  H_out  W_out  RF    params     cum_params
1    conv 3x3 s=1 p=1     3     32     3  1  1  224    224    3     896        896
2    conv 3x3 s=2 p=1     32    64     3  2  1  112    112    7     18,496     19,392
3    pool max 2x2         64    64     2  2  0  56     56     11    0          19,392
...
```

5. **요약 라인**: 최종 `(C, H, W)`, 최종 수용 영역, 총 매개변수, 경고.

## 규칙

- 공간 크기는 항상 정수로 반환합니다. 공식이 정수를 생성하지 않으면 오류로 플래그를 지정하고 조용히 내림하지 마세요.
- `groups > 1`일 경우 `C_in % groups == 0` 및 `C_out % groups == 0`를 검증합니다. 그렇지 않으면 오류로 처리합니다.
- Depthwise Convolution (`groups == C_in`)의 경우 `layer` 열에 라벨을 지정하여 매개변수가 낮은 이유를 독자가 파악할 수 있도록 합니다.
- 사용자가 BatchNorm 또는 활성화 레이어를 제공하면 형태 계산에는 포함하지 않지만 매개변수는 누적합니다 (BatchNorm당 `2 * C`).
- 누락된 필드에 대해 기본값을 추측하지 마세요. 모든 Convolution 및 Pooling에 `k`, `s`, `p`가 필요합니다.
