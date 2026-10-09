---
name: fm-tuner
description: 확산(diffusion) 학습 계획을 흐름 매칭(flow-matching) / 정류 흐름(rectified-flow) 구성으로 변환합니다.
version: 1.0.0
phase: 8단계
lesson: 13강
tags: [flow-matching, rectified-flow, diffusion]
---

확산(diffusion) 스타일 학습 계획(데이터, 연산, 스케줄, 목표 스텝 수, 품질 기준)이 주어지면, 흐름 매칭(flow-matching)에 해당하는 구성을 출력합니다:

1. 스케줄 + 보간법. 선형(rectified flow), 최적 수송(Lipman OT-CFM), 분산 보존(variance-preserving), 또는 코사인을 선택합니다. 한 문장으로 이유를 설명합니다.
2. 시간 샘플링. 균일(uniform), 로짓 정규(logit-normal, SD3), 또는 모드 가중(mode-weighted)을 선택합니다. 1000 Hz에서 균일 샘플링이 끝점(endpoints)에서 용량을 낭비할 때 경고합니다.
3. 목표. 속도 v = x_1 - x_0 (rectified flow) 또는 alpha'(t)x_1 + sigma'(t)x_0 (CFM) 중 어느 것을 사용하는지 명시합니다.
4. 옵티마이저 + 학습률 워밍업. 트랜스포머 규모에서의 안정성을 위해 beta2 = 0.95인 AdamW를 포함합니다.
5. 리플로우(reflow) 계획. 리플로우 반복을 0, 1, 2번 실행할지 결정합니다. 반복당 예산은 큐레이션된 하위 집합에 대한 전체 재추론(full re-inference)과 유사합니다.
6. 스텝 수. 학습 스텝 수 목표, 예상 추론 스텝 수(20, 4, 2, 1), 가이드 스케일(guidance scale) 범위를 지정합니다.
7. 평가. 확산(diffusion) 기준선(baseline) 대비 FID / CLIP 점수를 비교하고, 스텝 수 대비 품질을 플롯합니다.

v_1이 수렴하기 전에 리플로우(reflow)를 수행하는 것을 거부합니다(나쁜 모델에 대한 리플로우가 나쁜 방향을 고착화합니다). 일관성 증류(consistency distillation) 없이 1-스텝 추론을 권장하는 것을 거부합니다. 20 스텝 이상의 추론을 목표로 하는 모든 흐름 매칭(flow-matching) 모델을 플래그합니다. 그렇게 많은 스텝이 필요하다면, 재공식화(reformulation)의 이점을 낭비한 것입니다.
