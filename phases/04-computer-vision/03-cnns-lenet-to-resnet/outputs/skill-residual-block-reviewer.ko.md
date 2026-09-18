---
name: skill-residual-block-reviewer
description: PyTorch 잔차 블록의 스킵 연결 정확성, BN 배치, 활성화 순서, shape 정렬을 검토합니다
version: 1.0.0
phase: 4
lesson: 3
tags: [computer-vision, resnet, code-review, pytorch]
---

# Residual Block Reviewer

잔차 블록을 구현한다고 주장하는 모든 PyTorch `nn.Module`을 위한 집중 리뷰어입니다. 깨진 ResNet 재작성의 거의 전부를 설명하는 네 가지 실수를 잡습니다.

## When to use

- 누군가 커스텀 BasicBlock이나 Bottleneck을 썼는데 손실이 NaN이거나 정확도가 멈출 때.
- 한 프레임워크에서 다른 곳으로 블록을 이식하며 동등성을 검증하고 싶을 때.
- ResNet 내부를 바꾸는 PR을 리뷰할 때(pre-activation, squeeze-excite, anti-alias).
- CIFAR 크기 입력에서는 잘 나가다 ImageNet 해상도에서 숏컷이 틀려 깨질 때.

## Inputs

- 소스 텍스트 또는 import 가능한 경로로 된 PyTorch 클래스 정의.
- Optional `variant`: `basic` | `bottleneck` | `preact` | `seblock`.

## Four checks

### 1. Shortcut shape alignment

`stride != 1`이거나 `in_channels != out_channels`인 모든 블록에서 숏컷 경로는 **반드시** shape를 맞추는 모듈이어야 합니다 — 보통 1x1 conv plus BN. 이 경우 맨 `nn.Identity()`는 forward 시 보장된 shape 불일치 에러입니다.

Diagnostic:
```
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy
```

### 2. BN placement relative to the addition

덧셈 `out + shortcut(x)`는 최종 ReLU **전에** 일어나야 하고(post-activation, 원본 ResNet), 아니면 최종 ReLU가 아예 없어야 합니다(pre-activation ResNet v2). 주 가지에서 ReLU를 적용한 뒤 raw 숏컷을 더하는 블록은 비대칭 활성화 범위를 만들어 학습을 해칩니다.

Diagnostic:
```
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect
```

### 3. Bias on conv layers

바로 뒤에 BatchNorm이 오는 Conv는 `bias=False`여야 합니다. BN의 beta가 이미 편향을 파라미터화하므로, 추가 conv bias는 파라미터를 낭비하고 수렴을 늦출 수 있습니다.

Diagnostic:
```
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers
```

### 4. In-place ReLU and autograd

숏컷에 더해질 텐서에 대한 `nn.ReLU(inplace=True)`는 잔차 덧셈에 여전히 필요할 수 있는 값을 덮어씁니다. 덧셈 전에 새 텐서를 만드는 층이 따르지 않는 `inplace=True`를 모두 표시하세요.

Diagnostic:
```
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add
```

## Report

```
[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence
```

## Rules

- 블록을 다시 쓰지 마세요. 보고만 하세요.
- 블록이 맞으면 어디에나 `ok`라고 하고 멈추세요. 제안 없음.
- 여러 가지가 틀리면 위 순서로 나열하세요(숏컷이 가장 흔한 크래시 원인이므로 먼저).
- 사용자가 지정한 의도적 pre-activation이나 squeeze-excite 변형을 틀렸다고 표시하지 마세요.
