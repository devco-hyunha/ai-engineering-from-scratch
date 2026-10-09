---
name: skill-pytorch-patterns
description: PyTorch 학습, 평가 및 배포를 위한 참조 패턴
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, training, deep-learning, gpu, patterns]
---

## 표준 학습 루프

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
```

## 혼합 정밀도 학습

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

사용 시점: float16을 지원하는 GPU 하드웨어(V100, A100, H100, RTX 3090+)에서 학습할 때. 약 1.5-2배의 속도 향상과 약 50%의 메모리 감소가 기대됩니다.

## 기울기 누적

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

사용 시점: GPU 메모리가 허용하는 것보다 더 큰 유효 배치 크기가 필요할 때. 손실을 accumulation_steps로 나누면 기울기 스케일이 일정하게 유지됩니다.

## 저장 및 로드

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

학습을 재개하기 위해 항상 옵티마이저 상태를 저장하세요. 추론 전용인 경우 `model.state_dict()`만 저장하세요.

## 사용자 정의 데이터셋

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = self._load_samples(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load_samples(self, data_dir):
        ...
```

## DataLoader 구성

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
```

| 매개변수 | 기능 | 사용 시점 |
|-----------|-------------|-------------|
| num_workers=4 | 병렬 데이터 로딩 | 멀티코어 머신에서는 항상 사용 |
| pin_memory=True | 페이지 고정 CPU 메모리 | GPU에서 학습할 때 |
| drop_last=True | 불완전한 마지막 배치 버리기 | BatchNorm을 사용할 때 |
| persistent_workers=True | 에포크 간 워커 유지 | num_workers > 0일 때 |

## 학습률 스케줄

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,
)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        ...
        optimizer.step()
        scheduler.step()
```

OneCycleLR: 대부분의 작업에 가장 좋은 기본값입니다. max_lr까지 워밍업한 후 코사인 감쇠를 적용합니다. 에포크마다가 아니라 배치마다 `scheduler.step()`을 호출하세요.

## 가중치 초기화

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)
```

## 추론 모드

```python
model.eval()

with torch.inference_mode():
    outputs = model(inputs)
```

`torch.inference_mode()`은 오토그라드(Autograd)를 완전히 비활성화하는 반면 `torch.no_grad()`은 단순히 기울기 계산만 억제하므로, 이 보다 더 빠릅니다.

## 공통 실수 체크리스트

1. CrossEntropyLoss 전에 softmax를 적용하는 것 (내부적으로 log_softmax를 포함하고 있습니다)
2. 검증 중에 model.eval()을 호출하지 않는 것
3. 텐서를 모델과 같은 디바이스로 이동하지 않는 것
4. optimizer.zero_grad()를 호출하지 않는 것 (기본적으로 기울기가 누적됩니다)
5. 학습 중에 torch.no_grad()를 사용하는 것 (기울기 계산이 비활성화됩니다)
6. num_workers를 너무 높게 설정하는 것 (너무 많은 프로세스가 생성되어 메모리가 스래싱됩니다)
7. GPU에서 학습할 때 pin_memory=True를 사용하지 않는 것
8. state_dict가 아닌 전체 모델 객체를 저장하는 경우 (리팩터링 시 깨짐)
