---
name: skill-latency-profiler
description: 워밍업, 동기화, 백분위수, 메모리 추적을 포함한 완전한 지연 시간 벤치마킹 스크립트를 작성합니다
version: 1.0.0
phase: 4단계
lesson: 15강
tags: [edge, deployment, profiling, benchmarking]
---

# 지연 시간 프로파일러

임의의 PyTorch 모델에 대해 규율 있는 지연 시간 벤치마크를 생성합니다. 하류 단계의 누구나 실제로 신뢰할 수 있는 보고서를 제공합니다.

## 사용 시점

- 배포할 백본을 선택하기 전에 여러 후보 백본을 비교할 때.
- 양자화(Quantization)나 가지치기(pruning) 전후.
- 런타임 변경(eager vs ONNX vs TensorRT) 후.
- 배포 준비 상태 보고서를 생성할 때.

## 입력

- `model`: PyTorch `nn.Module`.
- `input_shape`: `(1, 3, 224, 224)` 같은 튜플.
- `device`: `cpu` | `cuda` | `mps`.
- `warmup`: 기본값 10.
- `iters`: 기본값 100.

## 검사

### 1. 워밍업(Warmup)
타이밍 없이 모델을 `warmup`번 실행합니다. 첫 번째 순전파의 JIT 컴파일 및 캐시 냉기 효과를 포착합니다.

### 2. 동기화(Synchronisation)
`cuda`의 경우, 타이밍된 각 순전파 전후에 `torch.cuda.synchronize()`을 호출합니다.
`mps`의 경우, `torch.mps.synchronize()`을 호출합니다.

### 3. 타이머(Timer)
월클락(wall-clock) 측정을 위해 `time.perf_counter()`을 사용합니다. 밀리초로 변환합니다.

### 4. 백분위수(Percentiles)
타이밍의 전체 목록을 정렬합니다. `p50, p90, p95, p99, mean, std`을 보고합니다.

### 5. 메모리(Memory)
`cuda`의 경우, 실행 후 `torch.cuda.max_memory_allocated()`을 호출하고 기준선을 차감합니다.
`cpu`의 경우, 전후에 `tracemalloc` 또는 `psutil.Process().memory_info().rss`를 사용합니다.

### 6. 배치 크기 스윕(Batch-size sweep)
선택적으로 `batch_size in [1, 4, 16, 32]`에 대해 벤치마크를 반복하여 처리량과 지연 시간의 트레이드오프를 드러냅니다.

## 출력 템플릿

```python
import time
import torch
import psutil, os

def profile(model, input_shape, device="cpu", warmup=10, iters=100):
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss / 1e6

    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    mean = sum(times) / len(times)
    std  = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def pct(p):
        idx = max(0, min(len(times) - 1, int(len(times) * p) - 1))
        return times[idx]

    report = {
        "p50_ms":  pct(0.50),
        "p90_ms":  pct(0.90),
        "p95_ms":  pct(0.95),
        "p99_ms":  pct(0.99),
        "mean_ms": mean,
        "std_ms":  std,
        "rss_mb":  proc.memory_info().rss / 1e6 - baseline_rss,
    }
    if device == "cuda":
        report["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return report
```

## 규칙

- 항상 워밍업(Warmup)을 실행하세요. 첫 번째 순전파 타이밍은 절대 신뢰하지 마세요.
- 평균이 아닌 백분위수를 사용하세요. 단일 이상치가 평균을 두 배로 만들 수 있지만 p50은 거의 변하지 않습니다.
- 프로덕션과 동일한 input_shape를 사용하세요. 224x224에서의 지연 시간은 384x384에서의 지연 시간과 다릅니다.
- CUDA에서는 `torch.cuda.synchronize()`을 절대 생략하지 마세요. 이 값이 없으면 숫자는 의미가 없습니다.
- 숫자와 함께 torch 버전, CUDA 버전, 장치 이름을 기록하세요. 그렇지 않으면 비교가 불가능해집니다.
