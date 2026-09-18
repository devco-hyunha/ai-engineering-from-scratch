---
name: skill-freeze-inspector
description: 어떤 파라미터가 학습 가능한지, 어떤 BatchNorm 층이 eval 모드인지, 옵티마이저가 실제로 학습 가능 파라미터를 소비하는지 보고합니다
version: 1.0.0
phase: 4
lesson: 5
tags: [computer-vision, transfer-learning, debugging, pytorch]
---

# Freeze Inspector

전이 학습 버그는 세 곳에 숨습니다: 동결되어야 하는데 안 된 파라미터, 학습 가능해야 하는데 안 된 파라미터, 동결 상태가 바뀌기 전에 만들어진 옵티마이저. 이 스킬이 한 패스에 셋 모두를 드러냅니다.

## When to use

- 파라미터 부분집합에 `requires_grad`를 설정한 직후.
- 미세조정 실행의 첫 학습 스텝 전.
- `freeze_bn_stats`나 BN 모드를 뒤집는 헬퍼를 호출한 후.
- val 정확도가 랜덤에 멈춰 있고 실제로 아무것도 학습되지 않는다고 의심할 때.

## Inputs

- `model`: PyTorch `nn.Module`.
- `optimizer`: 학습에 쓰려는 옵티마이저.
- Optional `expected_frozen_prefixes`: 동결되어야 하는 파라미터 이름 접두사 목록 (예: `["conv1", "bn1", "layer1"]`).

## Steps

1. **Walk parameters.** For each `(name, param)`:
   - record `requires_grad`
   - record `shape` and `numel`

2. **Walk modules.** For each module:
   - if it is BatchNorm, record whether it is in eval mode and whether its affine parameters are trainable.

3. **Inspect the optimizer.** For each parameter group:
   - flatten its `params` into a set of `id(p)`.
   - compare with the set of all `id(p)` for params where `requires_grad == True`.

4. **Detect the four failure modes:**
   - `leaked_train`: a param has `requires_grad=True` but does not appear in the optimizer (gradient is computed but never applied).
   - `ghost_train`: a param appears in the optimizer but has `requires_grad=False` (optimizer state is wasted; can also cause bugs if you later re-enable requires_grad).
   - `bn_mismatch`: either (a) a BN layer is in train mode (accumulates running stats) while its affine parameters (`weight`, `bias`) are frozen, or (b) a BN layer is in eval mode (frozen stats) while its affine parameters are trainable. Both states are inconsistent and almost always a bug.
   - `expected_vs_actual`: any prefix listed in `expected_frozen_prefixes` still has a trainable parameter.

## Report

```
[freeze-inspector]
  model trainable params: <N>
  model frozen params:    <N>
  batchnorm layers in eval mode: <count>
  batchnorm layers in train mode: <count>

[optimizer coverage]
  trainable params fed to optimizer: <M> of <N>
  leaked_train: <list of names> (trainable but not in optimizer)
  ghost_train:  <list of names> (in optimizer but frozen)

[bn audit]
  mismatched layers: <list of names>

[expectations]
  expected_frozen_prefixes: <...>
  violating params:         <list>

[verdict]
  ok | <one-line summary of the most severe issue>
```

## Rules

- 파라미터 이름만 보고하고, 가중치 자체는 절대 출력하지 마세요.
- 모든 목록을 파라미터 이름 알파벳 순으로 정렬하세요.
- 옵티마이저 커버리지가 100%이고 불일치가 없으면 `ok`를 반환하고 멈추세요.
- `leaked_train`에는 항상 동결 상태가 바뀐 뒤 옵티마이저를 다시 만들라고 권하세요.
- `ghost_train`에는 파라미터 그룹을 제거하거나, 학습 의도가 있었다면 `requires_grad=True`를 설정하라고 권하세요.
