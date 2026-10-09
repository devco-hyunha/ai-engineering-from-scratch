---
name: skill-noise-schedule-designer
description: T와 목표 손상 수준을 입력으로 받아 선형, 코사인, 시그모이드 beta 스케줄을 생성하고 SNR 플롯을 출력합니다
version: 1.0.0
phase: 4단계
lesson: 10강
tags: [computer-vision, diffusion, noise-schedule, training]
---

# 노이즈 스케줄 설계자

beta 스케줄은 각 확산(diffusion) 단계에서 신호가 얼마나 보존되는지 제어합니다. 부적절한 스케줄은 모든 후속 결정에서 학습 효율과 샘플 품질을 제한합니다.

## 사용 시점

- 새로운 확산(diffusion) 학습을 시작할 때 T와 beta를 선택하는 경우.
- 흐릿한 샘플을 생성하는 확산(diffusion) 모델(스케줄이 너무 공격적)이나 구조를 학습하지 못하는 모델(스케줄이 너무 온건)을 디버깅하는 경우.
- 서로 다른 스케줄을 보고하는 논문 간 설계를 비교하는 경우.

## 입력

- `T`: 시간 단계(timestep) 수, 일반적으로 100-1000.
- `type`: linear | cosine | sigmoid.
- `target_alpha_bar_final`: t=T에서 유지할 신호의 비율, 기본값은 0.001 (99.9% 손상됨).
- 선택적 `image_resolution` — 더 큰 이미지는 손상이 더 느린 스케줄(코사인 또는 시프트된 스케줄)의 혜택을 받습니다.

## 스케줄 공식

### 선형
```
beta_t = beta_start + (beta_end - beta_start) * (t - 1) / (T - 1)
```
기본값: beta_start=1e-4, beta_end=0.02 (DDPM 논문).

### 코사인 (Nichol & Dhariwal, 2021)
```
alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
```
s = 0.008. 신호를 더 오래 유지합니다. 낮은 단계 수에서 더 잘 작동합니다.

### 시그모이드
```
alpha_bar_t = 1 / (1 + exp(k * (t/T - 0.5)))
```
k = 6 to 12. 좋은 절충안입니다. 일부 SDXL 변형에서 사용됩니다.

## 단계

1. 공식에 따라 betas를 계산합니다.
2. `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`를 사전 계산합니다.
3. SNR_t = alpha_bar_t / (1 - alpha_bar_t)를 계산하고 시간에 따른 SNR 요약 생성합니다.
4. `alphas_cumprod[T-1]`가 `target_alpha_bar_final`의 10% 범위 내에 있는지 확인합니다. 그렇지 않으면 beta_end (선형), s (코사인), 또는 k (시그모이드)를 조정하고 재시도합니다.
5. 세 가지 체크포인트를 보고합니다:
   - `t=T*0.25` — 초기 손상
   - `t=T*0.5` — 중간
   - `t=T*0.75` — 최종 근처

## 보고

```
[schedule]
  type:   <name>
  T:      <int>
  beta_start: <float>   beta_end: <float>

[signal retention]
  t=0.25T:  alpha_bar=<X>  SNR=<X>
  t=0.5T:   alpha_bar=<X>  SNR=<X>
  t=0.75T:  alpha_bar=<X>  SNR=<X>
  t=T:      alpha_bar=<X>  SNR=<X>

[warnings]
  - <if alpha_bar collapses before 0.75T>
  - <if beta_end produces NaN in log-SNR>
```

## 규칙

- `alpha_bar_t <= 0`가 포함된 스케줄은 절대 생성하지 마세요. 1e-5 미만으로 값을 클램프(clamp)하고 경고합니다.
- 저단계 샘플링(< 30 단계)에서는 코사인이 기본 권장 사항입니다.
- `quality_target == research`에서는 선형이 기본값입니다 — DDPM 기준선은 선형 스케줄로 보고됩니다.
- `image_resolution > 256`일 때, 고해상도에서 더 많은 신호를 보존하기 위해 스케줄을 이동(Chen, 2023)하는 것을 권장합니다.
