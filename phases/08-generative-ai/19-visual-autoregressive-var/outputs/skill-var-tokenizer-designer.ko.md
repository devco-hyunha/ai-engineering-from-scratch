---
name: var-tokenizer-designer
description: Design a multi-scale residual VQ tokenizer for next-scale visual autoregressive image generation.
version: 1.0.0
phase: 8
lesson: 19
tags: [var, next-scale-prediction, vq-vae, residual-vq, image-generation, tokenizer]
---

이미지 타겟(해상도, 채널, 컬러 vs 그레이스케일, 데이터셋 크기, 다운스트림 LM 연산 예산, 목표 FID)이 주어지면 다음을 출력하세요:

1. 스케일 스케줄(Scale schedule). 1x1부터 (H/p) x (W/p)까지의 K개 해상도 레벨을 나열하세요. 256x256의 경우 기본 10개 스케일, 512x512의 경우 14개 스케일을 권장합니다. LM의 유효 시퀀스 길이(스케일 면적의 합) 및 스케일 내 병렬 처리(per-pass parallel-within-scale) 예산에 근거하여 K를 정당화하세요.
2. 코드북(Codebook). 모든 스케일에 걸쳐 공유되는 단일 코드북 크기 `V`를 설정하세요(일반적으로 4096 / 8192 / 16384). 데이터셋 크기와 디코더 용량을 고려하여 `V`를 선택하세요. 캘리브레이션 배치(calibration batch)에서 코드북 사용률이 50% 이상을 유지하는지 확인하거나, 그렇지 않다면 `V`를 줄이세요.
3. 잔차 공유(Residual sharing). 스케일 1..K가 업샘플링된 임베딩의 합(residual VQ)을 통해 잠재 변수(latent)를 함께 재구성함을 확인하세요. 패치 크기 `p`와 VAE 백본(VQGAN 스타일의 판별기 사용 여부, 지각 손실(perceptual loss) 가중치)을 명시하세요.
4. 디코더(Decoder). 합산된 잠재 변수를 다시 픽셀로 매핑하는 VAE 디코더를 선택하세요. VQGAN 디코더, VAR 논문 디코더, 또는 더 가벼운 MAGVIT 스타일 디코더 중에서 선택하세요. 목표 FID 및 디코더 VRAM을 근거로 정당화하세요.
5. 위치 임베딩(Position embedding). 스케일당 학습된 임베딩과 스케일 내 2D sin-cos를 포함하는 `(scale_index, row, col)` 트리플(triple) 구조를 확인하세요. 평면적인 1D 위치 임베딩은 거부하세요. LM은 올바른 조건부(conditional)를 적용하기 위해 스케일 레이블이 필요합니다.

VAR를 위한 비잔차형(non-residual) 멀티스케일 토크나이저는 거부하세요. 잔차 합산이 없으면 다음 스케일 조건부(next-scale conditional)가 불분명해지며, LM은 논문에서 증명된 것과 다른 목적 함수를 최적화하게 됩니다. `V`가 더 작은 스케일의 픽셀 수에 맞춰 조정되고 코드북 붕괴(codebook collapse)가 완화되지 않는 한, 스케일별 별도 코드북은 거부하세요. `K` x 평균 스케일 면적이 LM의 최대 시퀀스 길이에서 텍스트 조건부(text conditioning)를 위한 여유 공간을 뺀 값보다 크면 다음 스케일 예측(next-scale prediction) 자체를 거부하세요.

입력 예시: "ImageNet class-conditional 256x256, dataset 1.2M, LM budget 1.5B params, target FID under 5.0."

출력 예시:
- 스케일 스케줄: K=10, 크기 1, 2, 3, 4, 5, 6, 8, 10, 13, 16. 총 토큰 수 671.
- 코드북: 공유됨, V=4096. 256 해상도의 ImageNet에서 70-80% 사용률 예상.
- 잔차 공유: 확인됨; p=16, 지각 손실(perceptual) + 적대적 손실(adversarial)을 사용하는 VQGAN 백본, 잔차 합산으로 f를 재구성함.
- 디코더: VQGAN 디코더, 4개의 업샘플링 블록, 추가 리파이너(refiner) 없음.
- 위치 임베딩: (scale, row, col) 트리플, 학습된 스케일 토큰 + 스케일 내 2D sin-cos.
