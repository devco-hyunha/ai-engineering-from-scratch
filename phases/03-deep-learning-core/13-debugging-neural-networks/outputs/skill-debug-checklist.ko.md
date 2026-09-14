---
name: skill-debug-checklist
description: 신경망 학습 실패 디버깅을 위한 결정 트리 체크리스트
version: 1.0.0
phase: 3
lesson: 13
tags: [debugging, neural-networks, training, diagnostics, deep-learning]
---

# 신경망 디버그 체크리스트 (Neural Network Debug Checklist)

학습이 잘못됐을 때의 체계적 디버깅 프로토콜입니다. 순서대로 진행하세요 — 대부분의 버그는 처음 3단계에서 잡힙니다.

## 학습 전 (버그 예방)

1. 모델 아키텍처와 파라미터 수를 출력하세요. 데이터에 맞는 크기인가?
2. 무작위 입력으로 단일 순전파를 돌리세요. 출력 shape가 타깃 shape와 맞는가?
3. 라벨 dtype이 올바른지 확인하세요 (CrossEntropyLoss는 Long, BCELoss는 Float)
4. 데이터 정규화 검증: 입력 평균이 ~0, std가 ~1이어야 함
5. 무작위 (input, label) 쌍 5개를 출력하세요. 라벨이 기대와 맞는가?
6. train/test 분할에 중복 샘플이 없는지 확인하세요

## 한 배치 과적합 테스트 (60초, 버그의 80%를 잡음)

1. 학습셋에서 샘플 8-32개를 가져오세요
2. 합리적인 학습률로 200 step 학습하세요
3. 손실은 0에 가까워지고, 학습 정확도는 100%에 도달해야 합니다
4. 실패하면: 버그는 데이터나 하이퍼파라미터가 아니라 모델·손실 함수·학습 루프에 있습니다
5. 통과하면: 전체 학습으로 진행하세요

## 손실이 줄지 않음

1. 학습률을 확인하세요. 3개 값 시도: current/10, current, current*10
2. 레이어별 기울기 노름을 출력하세요. 전부 0이면 dead 네트워크 또는 분리된 그래프
3. 파라미터에 `requires_grad=True`인지 확인하세요. `loss.backward()`가 호출되는지 확인하세요
4. `loss.backward()` 전에 `optimizer.zero_grad()`가 호출되는지 확인하세요
5. `loss.backward()` 후에 `optimizer.step()`이 호출되는지 확인하세요
6. 모델 파라미터가 옵티마이저에 전달되는지 확인하세요: `optimizer = Adam(model.parameters())`

## 손실이 NaN 또는 Inf

1. 학습률을 10배 줄이세요
2. 모든 log() 호출에 epsilon 추가: `torch.log(x + 1e-7)`
3. 모든 나눗셈에 epsilon 추가: `x / (y + 1e-8)`
4. BCE 손실 전에 예측 클램프: `torch.clamp(pred, 1e-7, 1 - 1e-7)`
5. `torch.autograd.detect_anomaly()`로 정확한 연산을 찾으세요
6. 입력 데이터의 NaN 확인: `assert not torch.isnan(x).any()`

## 손실이 진동함

1. 학습률을 3-10배 줄이세요
2. 배치 크기를 늘리세요 (기울기 노이즈 감소)
3. 기울기 클리핑 추가: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
4. SGD에서 Adam으로 전환 (파라미터별 적응적 LR)
5. 학습 초반 5-10%에 학습률 warmup을 추가하세요

## 과적합 (train acc 높고, test acc 낮음)

1. 드롭아웃 추가 (p=0.1부터, 0.5까지 올리기)
2. 옵티마이저에 weight decay 추가: `Adam(params, weight_decay=1e-4)`
3. 모델 크기 줄이기 (레이어 수 또는 폭 감소)
4. 데이터 증강 추가
5. Early stopping 사용: 검증 손실이 5+ 에포크 동안 오르면 중단
6. train과 test 세트 간 데이터 누수 확인

## 과소적합 (train·test acc 모두 낮음)

1. 모델 용량 늘리기 (더 많은·더 넓은 레이어)
2. 더 많은 에포크 학습
3. 학습률 올리기 (신중히)
4. 모델이 학습할 수 있는지 확인하기 위해 정규화를 일시적으로 제거
5. 모델이 작업에 충분히 표현력 있는지 확인

## Dead ReLU 뉴런

1. 레이어별 0 활성화 비율 확인. >50%면 문제
2. LeakyReLU(0.01) 또는 GELU로 전환
3. 가중치에 Kaiming 초기화 사용
4. 학습률 줄이기 (큰 갱신이 뉴런을 dead zone으로 밀어낼 수 있음)
5. 활성화 함수 전에 배치 정규화 추가

## 빠른 참조: 학습률 시작점

| Optimizer | Task | Starting LR |
|-----------|------|------------|
| Adam | Training from scratch | 1e-3 |
| Adam | Fine-tuning pretrained | 1e-5 |
| SGD + momentum | Training from scratch | 1e-1 |
| SGD + momentum | Fine-tuning pretrained | 1e-3 |
| AdamW | Transformer training | 3e-4 |

## 빠른 참조: 배치 크기 효과

| Batch size | Gradient noise | Memory | Generalization |
|-----------|---------------|--------|---------------|
| 8-16 | High (noisy) | Low | Often better |
| 32-64 | Moderate | Moderate | Good default |
| 128-256 | Low (smooth) | High | May need warmup |
| 512+ | Very low | Very high | Needs LR scaling |

## 아무것도 안 될 때

1. 모델을 은닉층 1개로 단순화하세요. 학습하는가?
2. 데이터를 100 샘플로 단순화하세요. 과적합하는가?
3. 손실을 MSE로 바꾸세요. 수렴하는가?
4. 옵티마이저를 SGD(lr=0.01)로 바꾸세요. 진전이 있는가?
5. 데이터를 합성 데이터로 바꾸세요(예: y = x[0] > 0). 학습하는가?
6. 이 중 하나도 안 되면: 보고 있지 않은 코드에 버그가 있습니다 (데이터 로딩, 전처리, 텐서 shape)
