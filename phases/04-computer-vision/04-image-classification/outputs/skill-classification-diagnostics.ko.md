---
name: skill-classification-diagnostics
description: 혼동 행렬과 클래스 이름이 주어지면 클래스별 실패를 드러내고 가장 영향력 큰 수정 하나를 제안합니다
version: 1.0.0
phase: 4
lesson: 4
tags: [computer-vision, classification, evaluation, debugging]
---

# Classification Diagnostics

혼동 행렬을 읽는 렌즈입니다. 집계 정확도는 분류기가 동작한다고 말합니다. 혼동 행렬은 *아직 무엇을 모르는지* 말합니다.

## When to use

- 학습된 분류기의 검증 성능을 처음 볼 때.
- 다음에 무엇을 바꿀지 정하려고 학습 실행 사이에.
- 모델을 출시하기 전: 중요 클래스가 조용히 실패하지 않는지 검증.
- 전체 정확도가 한 포인트 떨어진 프로덕션 회귀를 디버깅하며 이유를 알아야 할 때.

## Inputs

- `cm`: CxC 혼동 행렬 (행 = 진짜, 열 = 예측).
- `labels`: 같은 순서의 C개 클래스 이름 목록.
- Optional `class_priors`: 클래스별 학습 빈도 (`cm`의 행 합이 기본).

## Steps

1. **Compute per-class metrics.** Treat any division by zero as the metric being undefined for that class, and report it as `n/a`; never substitute silently with 0.
   - precision_i = cm[i,i] / sum(cm[:, i])   (undefined when the class was never predicted)
   - recall_i    = cm[i,i] / sum(cm[i, :])   (undefined when the class has no ground-truth samples)
   - f1_i        = 2 * p * r / (p + r)        (undefined when either component is undefined)

2. **Rank up to three worst classes** by F1. If the confusion matrix has fewer than three classes, rank however many exist. Exclude classes with all metrics undefined.

3. **Find the top off-diagonal cell per row** — the one class that most commonly steals from this class. Report as `true -> predicted`.

4. **Classify the failure mode** for each worst class. Use these quantitative thresholds so the label is reproducible:
   - `ambiguity` — bidirectional confusion with another class: both `cm[i,j] / sum(cm[i, :]) >= 0.15` and `cm[j,i] / sum(cm[j, :]) >= 0.15`.
   - `imbalance` — the class has `< 0.5x` the training count of its top confuser.
   - `label_noise` — `|precision_i - recall_i| >= 0.2` and the class is not on the imbalance / ambiguity paths.
   - `systematic` — no single confuser exceeds 0.2 share of this class's errors; errors spread across three or more other classes.

5. **Recommend the single most impactful next action**:
   - `ambiguity` -> collect or synthesise discriminative examples, add targeted augmentation that preserves the distinguishing feature.
   - `imbalance` -> oversample the minority class or apply class-weighted loss.
   - `label_noise` -> audit a stratified sample of the class; fix mislabels before any other change.
   - `systematic` -> increase data for the class or fine-tune with a higher weight on this class's loss.

## Report

```
[diagnostics]
  aggregate accuracy: X.XX
  macro F1:           X.XX

[top-3 worst classes]
  1. class <name>  F1 = X.XX  prec = X.XX  rec = X.XX
     top confusion: <name> -> <other>  (N cases)
     failure mode:  ambiguity | imbalance | label_noise | systematic
     action:        <one sentence>

  2. ...
  3. ...

[recommendation]
  single biggest lever: <one sentence naming the class and the fix>
```

## Rules

- 클래스는 최대 셋만 반환하세요. 더 많으면 신호가 숨습니다.
- 각 최악 클래스의 지배적 혼동자를 이름 붙이세요; "많은 것과 혼동"으로 요약하지 마세요.
- 모든 권고를 혼동 행렬 증거에 근거하세요. 어느 클래스인지 없이 일반적인 "데이터 더 추가"는 안 됩니다.
- Precision과 recall이 0.2 이상 어긋나면 항상 라벨 노이즈를 후보로 표시하세요 — 실제 클래스는 학습 후 보통 P와 R이 맞춰집니다.
