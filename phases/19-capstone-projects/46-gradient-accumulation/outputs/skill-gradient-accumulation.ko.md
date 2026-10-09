---
name: gradient-accumulation
description: 마이크로 배치 손실을 스케일링하고 윈도우당 옵티마이저를 한 번 스텝하여, 장치 메모리보다 큰 유효 배치로 학습합니다.
version: 1.0.0
phase: 19단계
lesson: 46강
tags: [training, batch-size, distributed, scaling]
---

## 언제 사용해야 하는가

유효 배치는 기울기를 부드럽게 만들고 학습률 스케줄에 맞추는 레버입니다. 단일 순전파로 이를 감당할 수 없을 때, 이 레시피가 해답입니다.

## 레시피

1. 메모리에 Fits하고 가속기를 포화시키는 가장 큰 크기로 `micro_batch`를 선택하세요.
2. 학습률 스케줄에서 `effective_batch`를 선택하세요.
3. `accum_steps = effective_batch // (micro_batch * world_size)`를 설정하고 균등하게 나누어떨어짐을 어서트하세요.
4. 마이크로 배치마다: `loss = criterion(model(x), y) / accum_steps; loss.backward()`.
5. 마지막이 아닌 마이크로 배치에서는 `model.no_sync()`에 진입하여 DDP에서 기울기 올-리듀스를 건너뛰세요.
6. 마지막 마이크로 배치 후, `optimizer.step()`를 한 번 실행하세요. 다음 윈도우 전에 기울기를 0으로 초기화하세요.
7. 옵티마이저 상태는 유효 배치마다 한 번 진행되며, 학습률 스케줄은 유효 배치마다 한 번 틱합니다.

## 로깅

`samples_per_sec`, `median_step_ms`, `sync_calls`, `accum_steps`, `effective_batch`를 포함하여 유효 스텝마다 작은 JSON 레코드를 방출하세요. 이것이 없으면 비용 트레이드가 보이지 않습니다.

## 실패 모드

- `/ accum_steps` 스케일링을 잊는 것: 기울기가 N배로 폭발합니다.
- 윈도우 중간에 스텝하는 것: 파라미터가 드리프트합니다.
- 모든 마이크로 배치에서 동기화하는 것: 통계적 이득 없이 네트워크에 묶입니다.
- 이것을 혼합 정밀도 언스케일링과 혼합하는 것: 언스케일된 손실만 스케일링하세요.
