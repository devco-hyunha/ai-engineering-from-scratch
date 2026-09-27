---
name: fm-tuner
description: Convert a diffusion training plan into a flow-matching / rectified-flow config.
version: 1.0.0
phase: 8
lesson: 13
tags: [flow-matching, rectified-flow, diffusion]
---

확산 모델(diffusion-style) 방식의 학습 계획(데이터, 연산량, 스케줄, 목표 스텝 수, 품질 기준)이 주어지면, 이에 상응하는 플로우 매칭(flow-matching) 설정을 출력합니다:

1. 스케줄 + 보간법(Schedule + interpolant). 선형(Linear, rectified flow), 최적 운송(Optimal Transport, Lipman OT-CFM), 분산 보존(Variance-preserving), 또는 코사인(Cosine) 중 하나를 선택하고, 그 이유를 한 문장으로 설명합니다.
2. 시간 샘플링(Time sampling). 균등(Uniform), 로짓-노멀(Logit-normal, SD3), 또는 모드 가중(Mode-weighted) 중 하나를 선택합니다. 1000Hz에서 균등 샘플링을 사용할 경우 양 끝단에서 용량이 낭비될 수 있음을 경고합니다.
3. 타겟(Target). 속도 `v = x_1 - x_0` (rectified flow) 또는 `alpha'(t)x_1 + sigma'(t)x_0` (CFM) 중 무엇인지 명시합니다.
4. 옵티마이저 + 학습률 웜업(Optimizer + lr warmup). 트랜스포머 규모에서의 안정성을 위해 `beta2 = 0.95`를 적용한 `AdamW`를 포함합니다.
5. 리플로우 계획(Reflow plan). 0, 1, 또는 2회의 리플로우 반복 실행 여부를 결정합니다. 반복당 예산은 선별된 하위 집합(curated subset)에 대한 전체 재추론(full re-inference) 비용과 유사합니다.
6. 스텝 수(Step counts). 목표 학습 스텝 수, 예상 추론 스텝 수(20, 4, 2, 1), 가이던스 스케일(guidance scale) 범위를 명시합니다.
7. 평가(Eval). 확산 모델 베이스라인 대비 FID / CLIP-score, 스텝 수 대비 품질 그래프를 포함합니다.

`v_1`이 수렴하기 전에는 리플로우(reflow)를 수행하지 마십시오(잘못된 모델에 대한 리플로우는 잘못된 방향을 고착화할 뿐입니다). 일관성 증류(consistency distillation)가 병행되지 않은 상태에서 1-스텝 추론을 추천하는 것은 거부하십시오. 20단계 이상의 추론을 목표로 하는 플로우 매칭 모델이 있다면 경고하십시오. 만약 그만큼의 스텝이 필요하다면, 재정의(reformulation)를 시도한 의미가 없습니다.
