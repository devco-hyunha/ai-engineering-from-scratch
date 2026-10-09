---
name: skill-residual-block-reviewer
description: PyTorch 잔차 블록의 스킵 연결 정확성, BN 배치, 활성화 순서 및 형태 정렬을 검토합니다
version: 1.0.0
phase: 4
lesson: 3
tags: [computer-vision, resnet, code-review, pytorch]
---

# 잔차 블록 리뷰어

잔차 블록을 구현했다고 주장하는 모든 PyTorch `nn.Module`에 대한 집중 리뷰어입니다. 거의 모든 깨진 ResNet 재작성의 원인이 되는 네 가지 실수를 잡아냅니다.

## 사용 시점

- 누군가 사용자 정의 BasicBlock 또는 Bottleneck을 작성했는데 손실이 NaN이거나 정확도가 정체된 경우.
- 한 프레임워크에서 다른 프레임워크로 블록을 이식하고 동등성을 확인해야 하는 경우.
- ResNet 내부 구조(pre-activation, squeeze-excite, anti-alias)를 변경하는 PR을 검토하는 경우.
- CIFAR 크기의 입력에서는 모델이 잘 작동하지만, ImageNet 해상도에서는 스킵 연결이 잘못되어 충돌하는 경우.

## 입력

- PyTorch 클래스 정의. 소스 텍스트 또는 가져올 수 있는 경로로 제공됩니다.
- 선택적 `variant`: `basic` | `bottleneck` | `preact` | `seblock`.

## 네 가지 점검 항목

### 1. 스킵 연결 형태 정렬

`stride != 1` 또는 `in_channels != out_channels`이 있는 모든 블록에서 스킵 연결 경로는 **반드시** 형태가 일치하는 모듈이어야 합니다. 일반적으로 1x1 컨볼루션과 BN을 사용합니다. 이 경우 단순한 `nn.Identity()`는 forward 실행 시 형태 불일치 오류를 보장합니다.

진단:
```
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy
```

### 2. 더하기 연산에 대한 BN 배치

더하기 연산 `out + shortcut(x)`은 **최종 ReLU보다 먼저** 수행되어야 합니다(post-activation, 원본 ResNet) 또는 최종 ReLU가 완전히 없어야 합니다(pre-activation ResNet v2). 메인 분기에 ReLU를 적용한 후 원시 스킵 연결을 더하는 블록은 비대칭적인 활성화 범위를 생성하여 학습에 해를 끼칩니다.

진단:
```
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect
```

### 3. 컨볼루션 레이어의 편향

BatchNorm이 즉시 따라오는 컨볼루션은 `bias=False`이어야 합니다. BN의 beta가 이미 편향을 매개변수화하므로, 추가적인 컨볼루션 편향은 매개변수를 낭비하고 수렴을 늦출 수 있습니다.

진단:
```
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers
```

### 4. In-place ReLU 및 autograd

`nn.ReLU(inplace=True)`이 쇼트컷에 추가될 텐서에 적용되면 잔차 더셈에 아직 필요한 값이 덮어써집니다. 더셈 전에 새 텐서를 생성하는 레이어가 뒤따르지 않는 `inplace=True`을 모두 표시하세요.

진단:
```
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add
```

## 보고

```
[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence
```

## 규칙

- 블록을 재작성하지 마세요. 보고만 하세요.
- 블록이 정확하면 모든 곳에 `ok`이라고 말하고 멈추세요. 제안은 하지 마세요.
- 여러 문제가 있으면 위 순서대로 나열하세요 (쇼트컷이 충돌의 가장 흔한 원인이므로 먼저).
- 사용자가 명시한 의도적인 사전 활성화(pre-activation)나 squeeze-excite 변형은 잘못된 것으로 표시하지 마세요.
