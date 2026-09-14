---
name: skill-pytorch-patterns
description: PyTorch 학습·평가·배포를 위한 레퍼런스 패턴
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, training, deep-learning, gpu, patterns]
---

## 정석 학습 루프 (Canonical Training Loop)

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

## 혼합 정밀도 학습 (Mixed Precision Training)

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

사용할 때: float16을 지원하는 GPU(V100, A100, H100, RTX 3090+)에서 학습할 때. ~1.5-2배 속도 향상과 ~50% 메모리 절감을 기대하세요.

## 기울기 누적 (Gradient Accumulation)

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

사용할 때: 유효 배치 크기가 GPU 메모리보다 커야 할 때. loss를 accumulation_steps로 나누면 기울기 스케일이 일정하게 유지됩니다.

## 저장과 로드 (Save and Load)

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

학습을 재개하려면 항상 옵티마이저 상태를 저장하세요. 추론만 할 때는 `model.state_dict()`만 저장하면 됩니다.

## 커스텀 Dataset

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

## DataLoader 설정 (DataLoader Configuration)

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

| Parameter | What it does | When to use |
|-----------|-------------|-------------|
| num_workers=4 | Parallel data loading | Always on multi-core machines |
| pin_memory=True | Page-locked CPU memory | When training on GPU |
| drop_last=True | Drop incomplete final batch | When using BatchNorm |
| persistent_workers=True | Keep workers alive across epochs | When num_workers > 0 |

## 학습률 스케줄 (Learning Rate Schedules)

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

OneCycleLR: 대부분의 작업에 가장 좋은 기본값입니다. max_lr까지 warmup한 뒤 cosine으로 감쇠합니다. `scheduler.step()`은 에포크마다가 아니라 매 배치 후에 호출하세요.

## 가중치 초기화 (Weight Initialization)

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

## 추론 모드 (Inference Mode)

```python
model.eval()

with torch.inference_mode():
    outputs = model(inputs)
```

`torch.inference_mode()`는 `torch.no_grad()`보다 빠릅니다. 기울기 계산만 억제하는 것이 아니라 autograd 자체를 끄기 때문입니다.

## 흔한 실수 체크리스트 (Common Mistakes Checklist)

1. CrossEntropyLoss 전에 softmax 적용 (내부적으로 log_softmax를 포함)
2. 검증 중 model.eval() 호출 잊기
3. 텐서를 모델과 같은 디바이스로 옮기지 않기
4. optimizer.zero_grad() 미호출 (기울기는 기본적으로 누적)
5. 학습 중 torch.no_grad() 사용 (기울기 계산 비활성)
6. num_workers를 너무 높게 설정 (프로세스가 너무 많아 메모리 thrashing)
7. GPU 학습 시 pin_memory=True 미사용
8. state_dict 대신 모델 객체 전체를 저장 (리팩터 시 깨짐)
