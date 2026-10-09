---
name: skill-linear-probe-runner
description: 모든 동결된 인코더와 레이블이 지정된 데이터셋에 대한 완전한 선형 프로브 평가 작성
version: 1.0.0
phase: 4단계
lesson: 17강
tags: [self-supervised, evaluation, linear-probe, pytorch]
---

# 선형 프로브 실행기

동결된 인코더의 특징을 상단에 단일 선형 분류기를 학습하여 평가합니다. 모든 자기 지도 학습 논문에서 표준적으로 사용되는 평가 방법입니다.

## 사용 시점

- 자기 지도 학습 체크포인트를 비교할 때.
- 사전 학습 에포크에 걸쳐 특징 품질을 추적할 때.
- 미세 조정 없이 사전 학습된 인코더가 다운스트림 작업에 충분히 좋은지 결정할 때.

## 입력

- `encoder`: 이미지당 고정 차원의 특징을 반환하는 동결된 `nn.Module`.
- `feature_dim`: 인코더 출력의 차원 수.
- `train_dataset`: 레이블이 지정된 데이터셋 (이미지, class_id).
- `val_dataset`: 홀드아웃 세트.
- `num_classes`: 작업 클래스.
- `epochs`: ImageNet 규모에서는 일반적으로 100, 더 작은 데이터셋에서는 50.

## 단계

1. 인코더를 eval 모드로 설정하고 모든 매개변수에 `requires_grad=False`을 적용합니다.
2. 학습 및 검증 세트의 특징을 한 번만 추출합니다. numpy 배열이나 메모리 매핑 파일로 저장합니다.
3. 캐시된 특징에 대해 SGD + 코사인 스케줄로 `nn.Linear(feature_dim, num_classes)`을 학습합니다.
4. 표준 하이퍼파라미터: `lr=0.1`, `momentum=0.9`, `weight_decay=0`, `batch_size=1024`. 선형 프로브는 `lr`에 놀라울 정도로 민감합니다 — 정확도가 낮으면 스윕해 보세요.
5. 학습 종료 시 검증 세트에서의 top-1 정확도를 보고합니다.

## 출력 템플릿

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

## 보고

```
[linear probe]
  encoder:     <name + pretrain checkpoint>
  feature_dim: <int>
  epochs:      <int>
  best_val_top1: <float>
```

## 규칙

- 선형 프로브 중에는 절대 인코더 가중치를 업데이트하지 마세요. 이는 미세 조성이며 프로브가 아닙니다.
- 특징을 한 번만 사전 계산하세요. 매 에포크마다 인코더를 재학습하면 연산량이 100배 낭비됩니다.
- 가중치 감쇠 없이 SGD와 코사인 스케줄을 사용하세요. Adam은 여기서 성능이 떨어질 수 있습니다.
- 인코더 계열마다 학습률을 최소 한 번 스윕하세요. 최적값은 SSL 방법마다 다릅니다.
