---
name: skill-conv-shape-calculator
description: CNN 스펙을 층별로 따라가며 모든 블록의 출력 shape, 수용 영역, 파라미터 수를 보고합니다
version: 1.0.0
phase: 4
lesson: 2
tags: [computer-vision, cnn, architecture, debugging]
---

# Conv Shape Calculator

CNN을 계획하거나 디버깅할 때 쓰는 결정적 도우미입니다. 입력 shape와 층 스펙 목록이 주어지면 모델을 실행하지 않고 shape, 수용 영역, 파라미터 수를 추적합니다.

## When to use

- 새 CNN을 설계하며 모든 다운샘플이 깔끔한 크기에 떨어지는지 확인할 때.
- 논문을 읽고 아키텍처 표를 코드로 옮길 때.
- 사전학습 백본이 분류기 헤드에서 shape 불일치로 깨지고, 어느 층이 공간 크기를 바꿨는지 알아야 할 때.
- 둘 중 하나를 학습하기 전에 두 백본의 파라미터 효율을 비교할 때.

## Inputs

- `input_shape`: `(C, H, W)`.
- `layers`: 층 dict의 순서 목록. 각각 지원:
  - `{type: "conv", c_out, k, s, p, groups=1, bias=true}`
  - `{type: "pool", mode: "max"|"avg", k, s, p=0}`
  - `{type: "adaptive_pool", out_h, out_w}`
  - `{type: "flatten"}`
  - `{type: "linear", out_features, bias=true}`

## Steps

1. **Initialise trace** with `(C, H, W)`, receptive field `1`, effective stride `1`, cumulative params `0`.

2. **For each layer**, update in this order:
   - Compute `C_out` (conv/linear), or carry `C_in` through (pool).
   - Compute spatial output using `(H + 2P - K) / S + 1` for conv and pool, `out_h/out_w` for adaptive pool, `(1, 1)` for flatten output shape `(C * H * W, 1, 1)` before the linear, and scalar `1x1` for linear.
   - Update receptive field and effective stride:
     - Conv/pool: `RF_new = RF_old + (K - 1) * effective_stride`, `effective_stride *= S`.
     - Adaptive pool: treat as a pool with effective `S = H_in / out_h` (round down). `RF_new = RF_old + (H_in - 1) * effective_stride_old`; `effective_stride *= S`. Note that adaptive pool's RF equals the full previous spatial extent.
     - Flatten / linear: RF and effective stride are no longer meaningful; freeze them to the values before the flatten and omit from subsequent rows.
   - Compute params:
     - Conv: `C_out * (C_in / groups) * K * K + (C_out if bias else 0)`.
     - Linear: `out_features * in_features + (out_features if bias else 0)`.
     - Pool and flatten: 0.

3. **Detect problems** and flag them:
   - Non-integer output size (misaligned stride/padding).
   - `H_out <= 0` before the end of the stack.
   - Receptive field exceeding input size (possible wasted compute after that point).
   - Sudden 10x jumps in per-layer params that suggest the wrong channel plan.

4. **Report** as a single table:

```
idx  layer                C_in  C_out  K  S  P  H_out  W_out  RF    params     cum_params
1    conv 3x3 s=1 p=1     3     32     3  1  1  224    224    3     896        896
2    conv 3x3 s=2 p=1     32    64     3  2  1  112    112    7     18,496     19,392
3    pool max 2x2         64    64     2  2  0  56     56     11    0          19,392
...
```

5. **Summary line**: final `(C, H, W)`, final receptive field, total params, warnings.

## Rules

- 공간 크기는 항상 정수를 반환하세요. 공식이 비정수를 내면 에러로 표시하고 조용히 floor하지 마세요.
- `groups > 1`이면 `C_in % groups == 0`과 `C_out % groups == 0`을 검증하고, 아니면 에러입니다.
- depthwise conv(`groups == C_in`)는 `layer` 열에 라벨을 붙여 파라미터가 낮은 이유를 보이게 하세요.
- 사용자가 BatchNorm이나 활성화 층을 주면 shape 목적에서는 무시하되 파라미터는 이어 가세요(BatchNorm당 `2 * C`).
- 빠진 필드의 기본값을 추측하지 마세요. 모든 conv와 pool에 `k`, `s`, `p`를 요구합니다.
