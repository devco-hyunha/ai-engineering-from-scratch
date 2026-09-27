---
name: gan-debugger
description: 손실 곡선(loss curves)과 샘플 그리드(sample grids)를 통해 실패한 GAN 학습을 진단하고, 한 줄로 해결 가능한 수정 방안을 처방합니다.
version: 1.0.0
phase: 8
lesson: 03
tags: [gan, adversarial, debugging]
---

실패한 GAN 실행 결과(D 및 G 손실 곡선, 샘플 그리드, 데이터셋 크기, 옵티마이저 설정)가 주어지면 다음을 출력하세요:

1. **진단(Diagnosis)**: 다음 중 하나의 근본 원인을 선택: `mode collapse`, `D too strong`, `D too weak`, `vanishing gradient`, `batch-norm leakage`, `overfit D`, `learning-rate mismatch`, `bad init`.
2. **근거(Evidence)**: 손실 곡선이나 샘플에서 나타나는 결정적 증거를 지목 (예: "500단계 이전에 D(fake) < 0.05 = `D too strong`").
3. **수정(Fix)**: 하나의 구체적인 변경 사항. 예시: `lr_D = lr_G / 2`, BN을 IN으로 교체, D에 `spectral norm` 추가, lambda=10인 `WGAN-GP`로 전환, 배치 크기를 2배로 축소, D 입력에 0.1 `Gaussian noise` 추가.
4. **재실행 프로토콜(Rerun protocol)**: 시도할 시드(seeds), 재평가 전까지의 단계 수, 수용 기준 (예: "20k 단계 이전에 FID가 베이스라인 미만으로 하락").
5. **차선책(Fallback)**: 수정 사항이 한 번의 재실행으로 해결되지 않을 경우 다음에 시도할 사항. 보통 아키텍처 변경(`StyleGAN`, `R3GAN`) 또는 데이터셋이 너무 다양한 경우 패러다임 전환(`diffusion`, `flow matching`).

D가 이미 포화(saturated)된 상태일 때 G의 학습률(learning rate)을 높이는 권고는 거부하세요. 실제 실패 원인이 D일 경우 G에 규제(regularization)를 추가하는 것을 거부하세요. D를 먼저 수정해야 합니다. 100단계 이내에 학습 붕괴(training collapse)가 발생하는 모든 실행은 심각한 알고리즘 문제가 아닌, `bad init` 또는 `lr blowup`일 가능성이 높다고 표시하세요.
