---
name: skill-rectified-flow-trainer
description: AdaLN DiT와 오일러 샘플링을 포함한 완전한 rectified-flow 학습 루프 작성
version: 1.0.0
phase: 4단계
lesson: 23강
tags: [diffusion, rectified-flow, DiT, training]
---

# Rectified Flow Trainer

임의의 이미지 텐서 데이터셋에서 작은 DiT를 rectified flow로 성공적으로 학습할 수 있는 깔끔하고 최소한의 학습 루프를 생성합니다.

## 사용 시점

- 소규모로 SD3 / FLUX 학습 목적을 재현할 때.
- 동일한 데이터에서 rectified flow와 DDPM을 벤치마킹할 때.
- 비표준 도메인(의료, 위성)용 custom rectified-flow 모델을 구축할 때.

## 입력

- `model`: `(x, t)`를 받아 예측된 속도를 반환하는 `nn.Module`.
- `dataset`: 모델 도메인의 깨끗한 이미지들의 iterable.
- `optimizer`: `lr=1e-4`, `weight_decay=0.01`, `betas=(0.9, 0.99)`가 포함된 AdamW.
- `scheduler`: warmup이 포함된 cosine, 기본값은 warmup 1000 steps.

## 학습 단계

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # [0, 1]에서 uniform
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # 속도 목표
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 샘플링 (Euler)

```python
@torch.no_grad()
def sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

## 팁

- `torch.rand` uniform `t`을 사용하세요. `t`의 logit-normal 또는 Sd3 스타일 가중 샘플링은 약간 도움이 되지만, 시작하는 데 필수적이지는 않습니다.
- 모델 가중치의 EMA는 표준 관행입니다. decay 0.9999로 `ema_model`를 유지하세요.
- 조건부 모델의 classifier-free guidance: 학습 중 10% 확률로 조건부 임베딩을 빈/null 임베딩으로 교체하세요. 추론 시 `v_uncond + w * (v_cond - v_uncond)`와 `w`를 3-5 정도로 혼합하세요.
- LDM 스타일 학습(FLUX, SD3)의 경우, 전체 루프는 VAE latent space에서 실행됩니다. 위의 깨끗한 `x0`는 실제로 `VAE.encode(image)`입니다.
- 32x32 toy dataset에서의 전형적인 수렴: 2000-5000 steps. 실제 latent SD3 학습에서는 수십만 steps가 필요합니다.

## 보고서

```
[rectified flow training]
  steps:        <int>
  final loss:   <float>
  ema decay:    <float>
  vae?:         yes | no
  cfg dropout:  <fraction>

[sampling]
  default steps: 20
  schnell / turbo target: 4
  full quality reference: 50+ (for comparison only)
```

## 규칙

- RGB `uint8` 데이터에 image-space velocity target으로 rectified flow를 학습하지 마세요. 먼저 평균 0, 분산 1로 정규화하세요.
- 각 시간 단계 버킷별로 학습 손실을 항상 기록하세요. 초기 시간 단계(0에 가까움)의 손실이 후기 시간 단계(1에 가까움)보다 높다면, 속도 매개변수화가 잘못 연결되었을 가능성이 높습니다.
- 동일한 학습 루프에서 rectified-flow 속도 목표와 DDPM 잡음 목표를 혼합하지 마세요. 하나만 선택하세요.
- Ampere+ GPU에서는 bfloat16 학습을 사용하세요. rectified flow에서는 속도 크기로 인해 float16이 가끔 NaN 기울기를 생성할 수 있습니다.
