---
name: skill-rectified-flow-trainer
description: AdaLN DiT와 Euler 샘플링으로 완전한 rectified-flow 학습 루프를 작성합니다
version: 1.0.0
phase: 4
lesson: 23
tags: [diffusion, rectified-flow, DiT, training]
---

# Rectified Flow Trainer

임의의 이미지 텐서 데이터셋에서 작은 DiT를 rectified flow로 성공적으로 학습할 수 있는 깔끔하고 최소인 학습 루프를 만듭니다.

## 언제 쓰나요 (When to use)

- 소규모에서 SD3 / FLUX 학습 목적을 재현할 때.
- 같은 데이터에서 rectified flow vs DDPM을 벤치마크할 때.
- 비표준 도메인(의료, 위성)용 커스텀 rectified-flow 모델을 만들 때.

## 입력 (Inputs)

- `model`: `(x, t)`를 받아 예측 속도를 반환하는 `nn.Module`.
- `dataset`: 모델 도메인의 깨끗한 이미지 iterable.
- `optimizer`: AdamW with `lr=1e-4`, `weight_decay=0.01`, `betas=(0.9, 0.99)`.
- `scheduler`: cosine with warmup, 기본 1000 warmup 스텝.

## 학습 스텝 (Training step)

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # uniform in [0, 1]
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # velocity target
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

## 팁 (Tips)

- `torch.rand` 균일 `t`를 쓰세요; logit-normal 또는 Sd3형 가중 `t` 샘플링은 약간 도움이 되지만 시작에는 필수가 아닙니다.
- 모델 가중치 EMA가 표준 관행입니다; decay 0.9999의 `ema_model`을 유지하세요.
- 조건부 모델의 Classifier-free guidance: 학습 중 10% 확률로 조건화를 빈/null 임베딩으로 교체; 추론 시 `w` 약 3–5로 `v_uncond + w * (v_cond - v_uncond)`를 혼합.
- LDM형 학습(FLUX, SD3)에서는 전체 루프가 VAE 잠재 공간에서 돌아갑니다; 위의 깨끗한 `x0`은 실제로 `VAE.encode(image)`입니다.
- 32x32 토이 데이터셋의 전형적 수렴: 2000–5000 스텝. 실제 잠재 SD3 학습: 수십만.

## 보고 (Report)

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

## 규칙 (Rules)

- RGB `uint8` 데이터에서 이미지 공간 속도 타깃으로 rectified flow를 절대 학습하지 마세요; 먼저 평균 0, 단위 분산으로 정규화하세요.
- 타임스텝 버킷별 학습 손실을 항상 로깅하세요; 이른 타임스텝(0 근처) 손실이 늦은 것(1 근처)보다 높으면 속도 파라미터가 잘못 배선되었을 가능성이 큽니다.
- 같은 학습 루프에서 rectified-flow 속도 타깃과 DDPM 노이즈 타깃을 섞지 마세요; 하나를 고르세요.
- Ampere+ GPU에서는 bfloat16 학습을 쓰세요; float16은 속도 크기 때문에 rectified flow에서 때로 NaN 기울기를 냅니다.
