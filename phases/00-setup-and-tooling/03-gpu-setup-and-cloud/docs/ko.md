# GPU 설정 및 클라우드

> 학습을 위해 CPU로 훈련하는 것은 괜찮습니다. 실제 훈련에는 GPU가 필요합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 0단계, 01강
**시간:** 약 45분

## 학습 목표

- `nvidia-smi` 및 PyTorch의 CUDA API를 사용하여 로컬 GPU 가용성을 확인하세요
- 무료 클라우드 기반 실험을 위해 Google Colab에 T4 GPU를 설정하세요
- CPU와 GPU에서 행렬 곱셈을 벤치마킹하고 속도 향상 속도를 측정하세요
- fp16 경험칙을 사용하여 VRAM에 적합한 가장 큰 모델을 추정하세요

## 문제점

0단계부터 3단계까지의 대부분의 강의는 CPU로 잘 실행됩니다. 그러나 CNN, 트랜스포머, LLM (4단계 이상) 훈련을 시작하면 GPU 가속이 필요합니다. CPU에서 8시간이 걸리는 훈련은 GPU에서 10분이 걸립니다.

세 가지 옵션이 있습니다: 로컬 GPU, 클라우드 GPU, Google Colab (무료).

## 개념

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 구현하기

### 옵션 1: 로컬 NVIDIA GPU

GPU가 있는지 확인하세요:

```bash
nvidia-smi
```

CUDA가 포함된 PyTorch를 설치하세요:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 옵션 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com)로 이동하세요
2. 실행 환경 > 실행 환경 유형 변경 > T4 GPU
3. `!nvidia-smi`를 실행하여 확인하세요

이 과정의 노트북을 Colab에 직접 업로드하세요.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod, Vast.ai의 경우:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없나요? 문제없습니다.

대부분의 강의는 CPU에서 작동합니다. GPU가 필요한 강의는 이를 명시하고 Colab 링크를 포함합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 구현하기: GPU vs CPU 벤치마크

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

## 연습 문제

1. 위의 벤치마크를 실행하고 CPU와 GPU 시간을 비교하세요
2. GPU가 없다면 Google Colab에서 실행하고 비교하세요
3. GPU 메모리 용량을 확인하고, 수용할 수 있는 가장 큰 모델을 추정해 보세요 (경험칙: fp16의 경우 매개변수당 2바이트)

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | GPU에서 코드를 실행할 수 있게 해주는 NVIDIA의 병렬 컴퓨팅 플랫폼 |
| VRAM | "GPU 메모리" | GPU의 비디오 RAM으로, 시스템 RAM과 분리되어 있습니다. 모델 크기를 제한합니다. |
| fp16 | "반정밀도" | 16비트 부동 소수점으로, fp32의 절반 메모리를 사용하며 최소한의 정확도 손실만 발생 |
| Tensor Core | "고속 행렬 하드웨어" | 행렬 곱셈을 위한 특수 GPU 코어로, 일반 코어보다 4-8배 빠름 |
