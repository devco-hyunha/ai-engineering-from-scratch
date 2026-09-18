---
name: skill-latency-profiler
description: 워밍업, 동기화, 백분위수, 메모리 추적이 있는 완전한 지연 벤치마크 스크립트를 작성
version: 1.0.0
phase: 4
lesson: 15
tags: [edge, deployment, profiling, benchmarking]
---

# Latency Profiler

임의의 PyTorch 모델에 대해 규율 있는 지연 벤치마크를 만듭니다. 다운스트림 누구나 믿을 수 있는 보고서를 냅니다.

## When to use

- 배포할 후보 백본을 여러 개 비교할 때.
- 양자화나 가지치기 전후.
- 런타임 변경 후 (eager vs ONNX vs TensorRT).
- 배포 준비 보고서를 만들 때.

## Inputs

- `model`: PyTorch `nn.Module`.
- `input_shape`: `(1, 3, 224, 224)` 같은 튜플.
- `device`: `cpu` | `cuda` | `mps`.
- `warmup`: 기본 10.
- `iters`: 기본 100.

## Checks

### 1. Warmup
타이밍 없이 모델을 `warmup`회 실행합니다. 첫 순전파 JIT 컴파일과 차가운 캐시 효과를 잡습니다.

### 2. Synchronisation
`cuda`에서는 각 타이밍된 순전파 전후에 `torch.cuda.synchronize()`를 호출합니다.
`mps`에서는 `torch.mps.synchronize()`를 호출합니다.

### 3. Timer
벽시계 측정에 `time.perf_counter()`를 씁니다. 밀리초로 변환합니다.

### 4. Percentiles
전체 타이밍 목록을 정렬합니다. `p50, p90, p95, p99, mean, std`를 보고합니다.

### 5. Memory
`cuda`에서는 실행 후 `torch.cuda.max_memory_allocated()`를 호출하고 베이스라인을 뺍니다.
`cpu`에서는 전후에 `tracemalloc` 또는 `psutil.Process().memory_info().rss`를 씁니다.

### 6. Batch-size sweep
선택적으로 `batch_size in [1, 4, 16, 32]`에 대해 벤치마크를 반복해 처리량 vs 지연 트레이드오프를 드러냅니다.

## Output template

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

## Rules

- 항상 워밍업을 돌리세요. 첫 순전파 타이밍을 믿지 마세요.
- 평균이 아니라 백분위수 — 이상치 하나가 평균을 두 배로 올려도 p50은 거의 안 움직입니다.
- 프로덕션과 같은 input_shape를 쓰세요. 224×224 지연은 384×384 지연이 아닙니다.
- CUDA에서는 `torch.cuda.synchronize()`를 절대 빼지 마세요. 없으면 숫자가 무의미합니다.
- 숫자 옆에 torch 버전, CUDA 버전, 기기 이름을 로그하세요. 그렇지 않으면 비교가 불가능해집니다.
