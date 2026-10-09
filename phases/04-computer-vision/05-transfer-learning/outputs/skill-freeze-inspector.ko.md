---
name: skill-freeze-inspector
description: 학습 가능한 매개변수, eval 모드에 있는 BatchNorm 레이어, 옵티마이저가 실제로 학습 가능한 매개변수를 소비하는지 보고합니다
version: 1.0.0
phase: 4단계
lesson: 05강
tags: [computer-vision, transfer-learning, debugging, pytorch]
---

# Freeze Inspector

전이 학습 버그는 세 가지 곳에 숨어 있습니다: 동결되어야 하지만 동결되지 않은 매개변수, 학습 가능해야 하지만 학습 가능하지 않은 매개변수, 그리고 동결 상태가 변경되기 전에 생성된 옵티마이저입니다. 이 스킬은 한 번의 패스로 세 가지 문제를 모두 드러냅니다.

## 사용 시점

- 매개변수 하위 집합에 `requires_grad`을 설정한 직후에 사용하세요.
- 미세 조정 실행의 첫 학습 단계 전에 사용하세요.
- `freeze_bn_stats`을 호출하거나 BN 모드를 전환하는 헬퍼를 호출한 직후에 사용하세요.
- 검증 정확도가 무작위 수준에 고착되어 실제로 학습이 일어나지 않는다고 의심될 때 사용하세요.

## 입력

- `model`: PyTorch `nn.Module`입니다.
- `optimizer`: 학습에 사용될 옵티마이저입니다.
- 선택적 `expected_frozen_prefixes`: 동결되어야 하는 매개변수 이름 접두사 목록입니다 (예: `["conv1", "bn1", "layer1"]`).

## 단계

1. **매개변수 순회.** 각 `(name, param)`에 대해:
   - `requires_grad`을 기록하세요.
   - `shape`과 `numel`을 기록하세요.

2. **모듈 순회.** 각 모듈에 대해:
   - BatchNorm인 경우, eval 모드인지 및 affine 매개변수가 학습 가능한지 기록하세요.

3. **옵티마이저 검사.** 각 매개변수 그룹에 대해:
   - `params`을 `id(p)`의 집합으로 평탄화하세요.
   - `requires_grad == True`인 매개변수에 대한 모든 `id(p)`의 집합과 비교하세요.

4. **네 가지 실패 모드 감지:**
   - `leaked_train`: 매개변수가 `requires_grad=True`을 가지지만 옵티마이저에 나타나지 않습니다 (기울기가 계산되지만 적용되지 않습니다).
   - `ghost_train`: 매개변수가 옵티마이저에 나타나지만 `requires_grad=False`을 가집니다 (옵티마이저 상태가 낭비됩니다; 나중에 requires_grad를 다시 활성화하면 버그를 유발할 수도 있습니다).
   - `bn_mismatch`: (a) BN 레이어가 학습 모드(런닝 통계 누적)인데 그 아핀 매개변수(`weight`, `bias`)가 동결되어 있거나, (b) BN 레이어가 평가 모드(통계 동결)인데 그 아핀 매개변수가 학습 가능한 상태인 경우. 두 상태 모두 일관성이 없으며 거의 항상 버그입니다.
   - `expected_vs_actual`: `expected_frozen_prefixes`에 나열된 모든 접두어에 학습 가능한 매개변수가 남아 있습니다.

## 보고서

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

## 규칙

- 매개변수 이름만 보고하고, 가중치 자체는 절대 출력하지 마세요.
- 모든 목록을 매개변수 이름 기준으로 알파벳 순으로 정렬하세요.
- 옵티마이저 커버리지가 100%이고 불일치가 없다면 `ok`를 반환하고 중단하세요.
- `leaked_train`의 경우, 동결 상태가 변경된 후 항상 옵티마이저를 재구축할 것을 권장하세요.
- `ghost_train`의 경우, 학습하려는 의도였다면 매개변수 그룹을 제거하거나 `requires_grad=True`를 설정할 것을 권장하세요.
