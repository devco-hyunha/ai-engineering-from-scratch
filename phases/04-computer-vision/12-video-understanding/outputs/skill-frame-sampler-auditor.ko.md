---
name: skill-frame-sampler-auditor
description: 비디오 파이프라인의 프레임 샘플러에서 off-by-one, 짧은 클립 처리, 크롭 일관성을 감사합니다
version: 1.0.0
phase: 4
lesson: 12
tags: [computer-vision, video, sampling, debugging]
---

# 프레임 샘플러 감사기 (Frame Sampler Auditor)

프레임 샘플링은 비디오 파이프라인이 깨지는 곳입니다. 여기 버그는 이후 모든 지표로 전파됩니다.

## 언제 쓰나요 (When to use)

- 새 비디오 데이터 로더를 작성할 때.
- 논문 숫자를 재현하는데 학습 정확도가 보고된 것보다 낮을 때.
- 평가 정확도가 실행마다 불안정한 비디오 모델을 디버깅할 때.

## 입력 (Inputs)

- `sampler_code`: (num_frames_total, T)를 받아 T개 인덱스를 반환하는 Python 함수.
- `T`: 목표 클립 길이.
- 선택적 테스트 케이스: 연습할 `num_frames_total` 값(예: `[3, T-1, T, T+1, 30, 300, 3000]`).

## 검사 (Checks)

### 1. 짧은 클립 처리 (Short clip handling)
`num_frames_total < T`를 넣으세요. 반환된 모든 인덱스는 `[0, num_frames_total - 1]` 안이어야 합니다. 표준 패딩 정책은 남은 위치에 마지막 프레임을 반복하는 것입니다.

### 2. 경계 인덱스 (Boundary indices)
`num_frames_total == T`를 넣으세요. 반환 인덱스는 정확히 `[0, 1, ..., T-1]`이어야 합니다.

### 3. 균등 분포 (Uniform distribution)
`num_frames_total == 10 * T`를 넣으세요. 반환 인덱스는 단조 증가하고 대략 고르게 간격이 있어야 합니다.

### 4. 밀집 창 경계 (Dense window bounds)
밀집 샘플링에는 `num_frames_total == 3 * T`를 넣으세요. 반환 인덱스는 연속 창을 이루고, 클립 끝을 절대 넘지 않아야 합니다.

### 5. 결정성 (Determinism)
같은 입력과 (결정적 샘플러의 경우) 같은 RNG로 샘플러를 두 번 호출하세요. 인덱스가 일치해야 합니다.

### 6. 크롭 일관성 (Crop consistency)
파이프라인이 프레임마다 공간 크롭도 반환하면, 같은 시드로 같은 클립에 대해 샘플러를 두 번 돌리고 모든 프레임이 같은 크롭 박스(같은 `(x, y, w, h)`)를 쓰는지 확인하세요. 한 클립 안에서 프레임마다 다른 크롭은 시간 일관성을 파괴하며 고전적인 조용한 버그입니다. 허용되는 변형: *클립당* 적용되고 클립 안에서 일관된 증강.

## 보고 (Report)

```
[sampler audit]
  name: <function name>
  T:    <int>

[short-clip handling]
  passed | failed (<details>)

[boundary]
  passed | failed

[uniform spacing]
  passed | failed (<stddev of gaps>)

[dense window]
  passed | failed (<details>)

[determinism]
  passed | failed

[crop consistency]
  passed | failed (<per-frame crop varies: yes/no>)

[verdict]
  ok | fix required
```

## 규칙 (Rules)

- 짧은 클립 처리가 범위 밖 인덱스를 반환하면 샘플러를 절대 "ok"로 표시하지 마세요.
- 밀집 샘플러는 `num_frames_total - 1`을 넘는 창을 절대 반환하지 않아야 합니다.
- 샘플러가 확률적(밀집)이면 명시적으로 시드된 RNG로만 결정성을 테스트하세요.
- 표준 정책을 제안하되 조용히 고치지 마세요: 마지막 프레임으로 패딩, 창을 끝에 클램프, 반열린 구간 반올림.
