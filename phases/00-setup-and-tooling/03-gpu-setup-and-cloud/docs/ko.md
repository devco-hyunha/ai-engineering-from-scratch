# GPU 설정 및 클라우드 (GPU Setup & Cloud)

> 학습을 위한 목적으로는 CPU로도 충분합니다. 하지만 본격적인 모델 훈련에는 GPU가 반드시 필요합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 학습 목표 (Learning Objectives)

- `nvidia-smi` 및 PyTorch CUDA API를 사용해 로컬 GPU의 가용 상태를 검증합니다.
- 무료 클라우드 환경인 Google Colab에서 T4 GPU를 구성하여 실험을 실행합니다.
- CPU와 GPU의 행렬 곱셈 연산 속도를 벤치마킹하고 성능 향상 폭을 측정합니다.
- fp16 어림셈 규칙을 활용하여 보유한 VRAM에 적재 가능한 최대 모델 크기를 추정합니다.

## 문제 상황 (The Problem)

Phase 1부터 3까지의 대부분의 레슨은 CPU에서도 원활하게 실행됩니다. 하지만 CNN, Transformer, LLM(Phase 4 이상) 훈련 단계로 넘어가면 GPU 가속이 필수적입니다. CPU로 8시간이 걸리는 학습 작업이 GPU에서는 10분 만에 끝납니다.

선택지는 크게 로컬 GPU, 클라우드 GPU, 그리고 무료 Google Colab 세 가지가 있습니다.

## 핵심 개념 (The Concept)

```
선택 가능한 환경:

1. 로컬 NVIDIA GPU
   비용: 0원 (이미 보유하고 있는 경우)
   설정: CUDA + cuDNN 설치
   추천: 정기적인 개발, 대용량 데이터셋 처리

2. Google Colab (무료 티어)
   비용: 0원
   설정: 별도 설정 불필요
   추천: 빠른 실험, 집에 GPU가 없는 경우

3. 클라우드 GPU (Lambda, RunPod, Vast.ai)
   비용: 시간당 약 $0.20 ~ $2.00
   설정: SSH 접속 + 라이브러리 설치
   추천: 본격적인 대규모 모델 훈련
```

```figure
s0-gpu-dispatch
```

## 구현하기 (Build It)

### 옵션 1: 로컬 NVIDIA GPU

GPU 장착 여부 확인:

```bash
nvidia-smi
```

CUDA 지원 PyTorch 설치 및 확인:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 옵션 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. 런타임 > 런타임 유형 변경 > T4 GPU 선택
3. `!nvidia-smi`를 실행하여 장치 확인

이 코스의 노트북 파일을 Colab에 직접 업로드하여 실행할 수 있습니다.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod, Vast.ai 등 사용 시:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없어도 괜찮습니다
대부분의 레슨은 CPU에서도 작동합니다. GPU가 꼭 필요한 레슨에는 별도 안내와 Colab 링크가 제공됩니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 구현하기: GPU vs CPU 연산 벤치마크

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 실습 과제 (Exercises)

1. 위의 벤치마크 코드를 실행하여 CPU와 GPU의 소요 시간을 직접 비교해 보세요.
2. 로컬 GPU가 없다면 Google Colab에서 실행하여 속도 차이를 비교해 보세요.
3. 보유한 GPU 메모리를 확인하고, 적재 가능한 최대 모델 크기를 계산해 보세요 (기준: fp16 기준 파라미터당 약 2바이트).

## 핵심 용어 정리 (Key Terms)

| 용어 | 흔히 하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | GPU에서 코드를 병렬 실행할 수 있게 해 주는 NVIDIA의 가속 컴퓨팅 플랫폼 |
| VRAM | "GPU 메모리" | 시스템 RAM과 분리된 GPU 전용 비디오 메모리. 적재 가능한 모델 크기를 결정 |
| fp16 | "반정밀도 (Half precision)" | 16비트 부동소수점. 정확도 손실을 최소화하면서 fp32 대비 메모리를 절반만 사용 |
| Tensor Core | "행렬 전용 하드웨어" | 행렬 곱셈 연산에 특화된 GPU 전용 코어로, 일반 코어 대비 4~8배 빠름 |
