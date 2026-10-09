---
name: skill-frame-sampler-auditor
description: 비디오 파이프라인의 프레임 샘플러에 대해 오프-바이-원, 짧은 클립 처리 및 크롭 일관성을 감사합니다
version: 1.0.0
phase: 4
lesson: 12
tags: [computer-vision, video, sampling, debugging]
---

# 프레임 샘플러 감사자

프레임 샘플링은 비디오 파이프라인이 깨지는 지점입니다.这里的 버그는 모든 다운스트림 지표로 전파됩니다.

## 사용 시점

- 새로운 비디오 데이터 로더를 작성할 때.
- 논문에서 보고된 수치와 훈련 정확도가 낮을 때 재현할 때.
- 실행 간 평가 정확도가 불안정한 비디오 모델을 디버깅할 때.

## 입력

- `sampler_code`: (num_frames_total, T)를 받아 T개의 인덱스를 반환하는 Python 함수.
- `T`: 목표 클립 길이.
- 선택적 테스트 케이스: `num_frames_total` 값으로 테스트할 값 (예: `[3, T-1, T, T+1, 30, 300, 3000]`).

## 검사 항목

### 1. 짧은 클립 처리
`num_frames_total < T`를 입력하세요. 반환된 모든 인덱스는 `[0, num_frames_total - 1]` 범위 내에 있어야 합니다. 표준 패딩 정책은 나머지 위치에 마지막 프레임을 반복하는 것입니다.

### 2. 경계 인덱스
`num_frames_total == T`를 입력하세요. 반환된 인덱스는 정확히 `[0, 1, ..., T-1]`이어야 합니다.

### 3. 균일한 분포
`num_frames_total == 10 * T`를 입력하세요. 반환된 인덱스는 단조 증가해야 하며 대략적으로 균일한 간격을 가져야 합니다.

### 4. 밀집 윈도우 경계
밀집 샘플링의 경우 `num_frames_total == 3 * T`를 입력하세요. 반환된 인덱스는 연속적인 윈도우를 형성해야 하며, 클립의 끝을 넘지 않아야 합니다.

### 5. 결정성
동일한 입력으로 샘플러를 두 번 호출하고 (결정적 샘플러의 경우) 동일한 RNG를 사용하세요. 인덱스가 일치해야 합니다.

### 6. 크롭 일관성
파이프라인이 프레임마다 공간 크롭을 반환하는 경우, 동일한 클립에 대해 동일한 시드로 샘플러를 두 번 실행하고 모든 프레임이 동일한 크롭 박스(동일한 `(x, y, w, h)`)를 사용하는지 확인하세요. 하나의 클립 내에서 프레임마다 다른 크롭을 사용하는 것은 시간적 일관성을 파괴하며, 전형적인 조용한 버그입니다. 허용되는 변형: 클립 *단위*로 적용되는 증강은 클립 내에서 일관되어야 합니다.

## 보고서

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

## 규칙

- 짧은 클립 처리가 범위 밖 인덱스를 반환하는 경우, 샘플러를 "ok"로 표시하지 마세요.
- 밀집 샘플러는 `num_frames_total - 1`을 넘나드는 윈도우를 반환해서는 안 됩니다.
- 샘플러가 확률적(밀집)인 경우, 명시적으로 시드를 설정한 RNG로 결정성 테스트만 수행해 보세요.
- 정식 정책(마지막 프레임으로 패딩, 윈도우를 끝에 클램프, 반열린 구간을 반올림)을 제안하되, 조용히 수정하지는 마세요.
