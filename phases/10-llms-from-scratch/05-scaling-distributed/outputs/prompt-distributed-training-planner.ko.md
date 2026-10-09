---
name: prompt-distributed-training-planner
description: 모델 크기와 가용 하드웨어를 고려하여 분산 학습 실행을 계획합니다
version: 1.0.0
phase: 10단계
lesson: 05강
tags: [distributed-training, fsdp, deepspeed, tensor-parallelism, pipeline-parallelism, scaling]
---

# 분산 학습 계획 도구

대규모 언어 모델(LLM (대규모 언어 모델))의 분산 학습 실행을 계획할 때, 이 프레임워크를 사용하여 병렬화 전략, 메모리 예산, 통신 오버헤드 및 예상 처리량을 결정해 보세요.

## 입력 요구 사항

다음 정보를 제공하세요:
- **모델 크기** (단위: 십억 개 매개변수)
- **목표 학습 토큰 수** (단위: 조 개 토큰)
- **가용 GPU** (유형: A100/H100/H200, 개수, 상호 연결: NVLink/InfiniBand)
- **GPU 메모리** (A100/H100: 80GB, H200: 141GB)
- **노드** (노드당 GPU 수, 노드 수)
- **예산 제약** (최대 비용(달러), 최대 실시간(wall-clock time))

## 1단계: 메모리 예산

각 구성 요소에 대한 GPU당 메모리를 계산하세요:

| 구성 요소 | 공식 | FP16 | FP32 |
|-----------|---------|------|------|
| 가중치 | 매개변수 수 x 매개변수당 바이트 | 매개변수 수 x 2 | 매개변수 수 x 4 |
| Adam 옵티마이저 (m + v) | 매개변수 수 x 4 x 2 | 항상 매개변수당 8바이트 | 매개변수당 8바이트 |
| 기울기 | 매개변수 수 x 매개변수당 바이트 | 매개변수 수 x 2 | 매개변수 수 x 4 |
| 활성화 (추정) | 시퀀스 길이 x 배치 x 은닉층 x 레이어 x 2 | 변동 | 변동 |

총합이 GPU 메모리를 초과하면 샤딩(sharding)이 필요합니다. 다음 순서로 시도해 보세요:
1. ZeRO-1 (옵티마이저만 샤딩) -- 통신 비용이 가장 저렴
2. ZeRO-2 (+ 기울기) -- 중간 수준의 통신
3. FSDP/ZeRO-3 (+ 가중치) -- 통신 비용이 가장 높지만 메모리 절감 효과가 최대
4. 활성화가 여전히 너무 크다면 활성화 체크포인팅(Activation Checkpointing)을 추가하세요
5. 단일 레이어가 하나의 GPU에 맞지 않는다면 텐서 병렬화(Tensor Parallelism)를 추가하세요

## 2단계: 병렬화 전략

### 의사 결정 트리

1. **단일 레이어가 하나의 GPU에 맞습니까?**
   - 아니요: 텐서 병렬화가 필요합니다. TP를 2, 4 또는 8로 설정하세요 (노드 내에서).
   - 예: 텐서 병렬화를 건너뛰세요.

2. **샤딩을 적용한 전체 모델이 한 노드 내 GPU에 Fits합니까?**
   - 아니요: 파이프라인 병렬화(Pipeline Parallelism)가 필요합니다. PP = 노드 수 / 그룹 수로 설정하세요.
   - 예: 파이프라인 병렬화(Pipeline Parallelism)를 생략하세요.

3. **데이터 병렬화(Data Parallelism)에 남는 GPU는 몇 개입니까?**
   - DP = total_gpus / (TP x PP)

4. **데이터 병렬 그룹 내 샤딩 수준은 어떻게 설정합니까?**
   - FSDP (ZeRO-3)로 시작하세요. 통신이 병목이 되면 ZeRO-2 또는 ZeRO-1로 줄이세요.

### 일반적인 구성

| 모델 크기 | 총 GPU 수 | TP | PP | DP | 샤딩 |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## 3단계: 통신 분석

학습 스텝당 통신량을 추정하세요:

- **데이터 병렬 (all-reduce)**: 스텝당 2 x gradient_size x (N-1)/N
- **FSDP (all-gather + reduce-scatter)**: 스텝당 약 3 x weight_size x (N-1)/N (DP보다 높음)
- **텐서 병렬화 (레이어별 all-reduce)**: 스텝당 2 x activation_size x num_layers (NVLink 필요)
- **파이프라인 병렬화 (점-to-점)**: 스테이지 경계당 activation_size (최소)

통신 시간이 연산 시간의 20%를 초과하면 전략은 통신 병목(communication-bound)입니다. 해결책:
- 기울기 누적(Gradient Accumulation) (all-reduce 빈도 감소)
- 통신과 연산 겹치기 (FSDP는 기본적으로 수행)
- 마이크로 배치 크기 증가 (연산 대비 통신 비율 개선)
- 통신 부담이 적은 샤딩 단계로 전환

## 4단계: 처리량 및 비용 추정

**학습 스텝당 FLOPS:**
- 순전파: 약 2 x params x tokens_per_batch
- 역전파: 약 4 x params x tokens_per_batch (순전파의 2배)
- 총합: 약 6 x params x tokens_per_batch

**학습 시간:**
- total_flops = 6 x params x total_tokens
- time_seconds = total_flops / (num_gpus x gpu_tflops x 1e12 x utilization)
- 일반적인 활용률: 35-45% (통신, 파이프라인 버블, 메모리 오버헤드를 고려한 값)

**비용:**
- total_gpu_hours = num_gpus x time_seconds / 3600
- cost = total_gpu_hours x cost_per_gpu_hour

## 5단계: 검증 체크리스트

실행하기 전에:

1. GPU당 메모리가 하드웨어 한도 내에 fits (10% 여유 공간 포함)
2. 효과적인 배치 크기가 목표와 일치합니다 (per_gpu_batch x DP x gradient_accumulation_steps)
3. 통신 대 연산 비율이 20% 미만입니다
4. 파이프라인 버블 비율이 15% 미만입니다 (충분한 마이크로 배치)
5. 학습률이 효과적인 배치 크기에 맞춰 스케일링되었습니다
6. 체크포인팅 빈도가 실패 확률을 고려합니다 (대규모 실행의 경우 1-2시간마다 저장)
7. 기울기 클리핑이 설정되었습니다 (대규모 모델의 경우 일반적으로 1.0)
8. 워밍업 단계가 총 단계에 비례합니다 (일반적으로 총 단계의 0.1-1%)

## 적신호

- **TP > 8**: 노드 간 텐서 병렬화(Tensor parallelism)(Tensor parallelism)(InfiniBand를 통해)는 파이프라인 병렬화(Pipeline parallelism)(Pipeline parallelism)보다 거의 항상 느립니다
- **파이프라인 스테이지 > 32**: 많은 마이크로 배치에도 버블 오버헤드가 중요해집니다
- **효과적인 배치 크기 > 10M 토큰**: 수익이 감소합니다; 수렴에 해를 끼칠 수 있습니다
- **활용률 30% 미만**: 통신 병목 현상 -- 병렬화 전략을 재평가하세요
- **13B 이상에서 활성화 체크포인팅(Activation Checkpointing)(Activation Checkpointing) 없음**: 역전파(Backpropagation)(Backpropagation) 중에 메모리가 부족해집니다
- **작은 GPU당 배치로 기울기 누적(Gradient Accumulation)(Gradient Accumulation) 없음**: 기울기 잡음이 증가합니다; 효과적인 배치 256+ 샘플로 누적하세요
