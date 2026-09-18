---
name: skill-linear-probe-runner
description: 임의의 동결 인코더와 라벨 데이터셋에 대한 완전한 선형 프로브 평가를 작성
version: 1.0.0
phase: 4
lesson: 17
tags: [self-supervised, evaluation, linear-probe, pytorch]
---

# Linear Probe Runner

동결된 인코더의 특징을 단일 선형 분류기를 위에 학습해 평가합니다. 모든 자기지도 논문의 표준 평가입니다.

## When to use

- 자기지도 체크포인트를 비교할 때.
- 사전학습 에폭에 걸쳐 특징 품질을 추적할 때.
- 파인튜닝 없이 사전학습 인코더가 다운스트림 과제에 충분한지 결정할 때.

## Inputs

- `encoder`: 이미지당 고정 차원 특징을 반환하는 동결 `nn.Module`.
- `feature_dim`: 인코더 출력 차원.
- `train_dataset`: 라벨 데이터셋 (image, class_id).
- `val_dataset`: 홀드아웃 세트.
- `num_classes`: 과제 클래스 수.
- `epochs`: ImageNet 규모는 보통 100, 작은 데이터셋은 50.

## Steps

1. 인코더를 eval 모드로 두고 모든 파라미터에 `requires_grad=False`를 설정합니다.
2. train과 val 세트를 한 번 특징 추출합니다. numpy 배열이나 메모리 맵 파일로 저장합니다.
3. 캐시된 특징 위에서 SGD + cosine schedule로 `nn.Linear(feature_dim, num_classes)`를 학습합니다.
4. 표준 하이퍼파라미터: `lr=0.1`, `momentum=0.9`, `weight_decay=0`, `batch_size=1024`. 선형 프로브는 `lr`에 놀랍게 민감합니다 — 정확도가 나쁘면 스윕하세요.
5. 학습 끝에서 val top-1 정확도를 보고합니다.

## Output template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def extract(encoder, loader, device="cpu"):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x.to(device)).cpu()
            feats.append(f)
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(encoder, feature_dim, train_loader, val_loader,
                 num_classes, epochs=50, lr=0.1, device="cpu"):
    for p in encoder.parameters():
        p.requires_grad = False

    f_train, y_train = extract(encoder, train_loader, device)
    f_val, y_val = extract(encoder, val_loader, device)

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt = SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    ds = torch.utils.data.TensorDataset(f_train, y_train)
    train_iter = DataLoader(ds, batch_size=1024, shuffle=True)

    best_val = 0.0
    for ep in range(epochs):
        head.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            acc = (head(f_val.to(device)).argmax(-1).cpu() == y_val).float().mean().item()
        best_val = max(best_val, acc)
    return best_val
```

## Report

```
[linear probe]
  encoder:     <name + pretrain checkpoint>
  feature_dim: <int>
  epochs:      <int>
  best_val_top1: <float>
```

## Rules

- 선형 프로브 중 인코더 가중치를 절대 업데이트하지 마세요. 그러면 프로브가 아니라 파인튜닝입니다.
- 특징을 한 번만 미리 계산하세요. 매 에폭 인코더를 다시 돌리면 컴퓨트가 100배 낭비됩니다.
- weight decay 없는 SGD와 cosine schedule을 쓰세요. Adam은 여기서 종종 덜합니다.
- 인코더 계열마다 학습률을 적어도 한 번 스윕하세요. 최적값은 SSL 방법에 따라 달라집니다.
