# 기울기 체크포인팅 및 활성화 재계산

> 역전파는 모든 중간 활성화를 저장합니다. 70B 파라미터와 128K 컨텍스트에서는 랭크당 3 TB의 활성화가 필요합니다. 체크포인팅은 FLOPs를 메모리와 교환합니다: 저장하는 대신 재계산하는 것입니다. 문제는 어떤 세그먼트를 버릴지이며, 정답은 "모두"가 아닙니다.

**유형:** Build
**언어:** Python (numpy, 선택적으로 torch 사용)
**선수 요건:** 10단계 04강 (미니 GPT 사전 학습), 10단계 05강 (확장 및 분산)
**시간:** 약 70분

## 문제점

트랜스포머를 학습할 때, 각 레이어는 역전파에서 미분되는 모든 연산의 입력을 저장합니다: 어텐션 입력, Q/K/V 투영, 소프트맥스 출력, FFN 입력, 정규화 출력, 잔차 스트림. 숨겨진 크기가 `d`, 시퀀스 길이가 `L`, 배치가 `B`인 레이어의 경우, 레이어당 `12 * B * L * d`개의 floats가 필요합니다.

`d=8192, L=8192, B=1`인 경우, BF16에서 레이어당 800 MB입니다. 64 레이어 모델은 51 GB의 활성화가 필요하며, 이는 마이크로배치 크기를 곱하기 전, 어텐션-소프트맥스 중간값 (헤드당 `L^2`)을 더하기 전, 텐서 병렬 부분 복사본을 고려하기 전의 수치입니다.

양면의 청구서: BF16 가중치와 옵티마이저 상태는 80GB에 들어갈 수 있지만, 활성화가 이를 넘어서게 만듭니다. 기울기 체크포인팅 (활성화 재계산이라고도 함)이 표준적인 해결책입니다. 대부분의 활성화를 버리고, 역전파 중에 재계산하여 복원합니다. 비용: 추가 FLOPs. 이점: 체크포인팅 세그먼트와 전체 레이어의 비율만큼 메모리가 감소합니다.

단순하게 수행하면 체크포인팅은 단계당 순방향 패스 FLOPs를 약 33% 더 소모합니다. 잘 수행하면 — Korthikanti 등의 "스마트 선택"에 따른 선택적 체크포인팅 — FLOP 오버헤드 5% 미만으로 메모리를 5배 절약할 수 있습니다. FP8 행렬 곱, FSDP 오프로드, 전문가 병렬 MoE를 사용하면 이것이 정말 중요합니다: 메모리나 낭비된 연산 중 어느 것도 감당할 수 없기 때문입니다.

## 개념

### 역전파가 실제로 필요로 하는 것

`output = layer(input)`. 역전파는 `grad_input`과 `grad_params`를 원합니다. 이를 계산하려면 다음이 필요합니다:

- `input` (선형 레이어의 `grad_params = input.T @ grad_output`을 계산하기 위해)
- 일부 활성화 함수의 미분 중간값 (ReLU/GELU/softmax의 미분값은 활성화 값에 의존합니다)

순전파는 이러한 값을 오토그라드 그래프에 자동으로 저장합니다. 모든 `tensor.retain_grad()` 및 입력을 필요로 하는 모든 연산은 참조를 유지합니다.

### 소박한 전체 체크포인팅

네트워크를 `N`개의 세그먼트로 분할합니다. 순전파 동안 각 세그먼트의 *입력*만 저장합니다. 역전파가 중간값을 필요로 할 때, 세그먼트의 순전파를 다시 실행하여 중간값을 생성한 후 미분합니다.

예시: 32층 트랜스포머를 각각 1층인 32개의 세그먼트로 분할합니다.

- 메모리: 32개의 레이어 입력 (작음) vs 32 * (레이어당 활성화 볼륨) (거대).
- 추가 연산량: 세그먼트당 순전파가 1번 추가되므로, 전체 순전파 FLOPs가 약 33% 증가합니다 (역전파는 순전파의 2배이므로, 전체 단계는 1 + 1 + 2 = 4 유닛이 되며, 1 + 2 = 3 유닛이 아닙니다).

이것은 Chen et al. 2016의 원래 레시피입니다: 메모리와 연산량의 균형을 맞추기 위해 `sqrt(L)`층마다 하나의 체크포인팅을 수행합니다. L=64인 경우, 8개의 체크포인팅이 됩니다.

### 선택적 체크포인팅 (Korthikanti 2022)

모든 활성화 함수의 비용이 동일하지는 않습니다. 어텐션 softmax 출력은 `B*L*L*heads`이며 시퀀스 길이에 따라 *2차*로 증가합니다. FFN 숨겨진 활성화는 `B*L*4d`이며 선형으로 증가합니다. 긴 시퀀스에서는 softmax가 지배적입니다.

선택적 체크포인팅은 저장 비용이 낮은 활성화 (선형 투영, 잔차)를 유지하고, 비용이 높은 활성화 (어텐션)만 재계산합니다. 재계산에 최소한의 FLOPs를 지불하지만 O(L^2) 메모리를 절약합니다.

Megatron-Core는 이를 "선택적" 활성화 재계산으로 구현합니다. 2024년 이후 대부분의 최첨단 훈련 실행에서 사용됩니다.

### 오프로드

재계산의 대안: 순전파와 역전파 사이에 활성화 값을 CPU RAM으로 전송합니다. PCIe 대역폭이 필요하며, 유휴 대역폭이 재물화 비용보다 클 때 유리합니다. 혼합 전략이 일반적입니다: 일부 레이어는 체크포인팅하고, 나머지는 오프로드합니다.

FSDP2는 오프로드를 일급 옵션으로 제공합니다. GPU가 메모리 병목에 걸렸지만 CPU-GPU 전송에 여유가 있을 때 오프로드가 빛을 발합니다.

### 재계산 비용 모델

`L`층 중 `k`층마다 소박한 체크포인팅을 수행할 때 단계별 FLOPs:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 세그먼트 내 각 레이어에서 한 번의 추가 순전파
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

선택적 체크포인팅(Selective Checkpointing)을 사용하면 전체 레이어가 아닌 어텐션 커널만 재계산합니다:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 메모리 절감 모델

레이어별 활성화 볼륨: `A`. `L` 레이어의 경우 총 활성화 메모리: `L * A`.

전체 체크포인팅(세그먼트 크기 1): `L * input_volume`만 저장합니다(표준 트랜스포머의 경우 약 `L * 1/10 A`). 약 `9 * L * A * 1/10`을 절감합니다.

`k` 레이어마다 체크포인팅: 활성 세그먼트 내 `L/k * A` 및 `k-1` 레이어의 데이터를 저장합니다.

`k = sqrt(L)`에서 메모리와 재계산 비용 모두 `sqrt(L)`에 비례하여 증가합니다 — 균일한 비용의 레이어에 대한 최적의 트레이드오프입니다.

### 체크포인팅을 하지 말아야 할 경우

- 파이프라인 단계에서 이미 실행 중인 가장 안쪽 레이어. 어차피 완료해야 합니다.
- 단계의 연산을 지배하는 첫 번째 및 마지막 레이어(트랜스포머에서는 드문 경우).
- FlashAttention을 이미 사용 중인 어텐션 커널. Flash는 소프트맥스를 이미 빠르게 재계산하므로, 레이어 수준의 추가 체크포인팅은 거의 효과가 없습니다.

### 구현 패턴

1. **함수 래퍼:** 세그먼트를 `torch.utils.checkpoint.checkpoint(fn, input)`으로 래핑합니다. PyTorch는 `input`만 저장하고, 역전파 시 나머지 모든 것을 재계산합니다.

2. **데코레이터 기반:** 레이어를 체크포인팅 가능하도록 표시합니다. 트레이너는 설정 시점에 어떤 세그먼트를 래핑할지 결정합니다.

3. **수동 명시적 재계산:** 저장된 입력으로 순전파를 복제하는 커스텀 `recompute_forward`을 호출하여 역전파를 직접 작성합니다.

세 가지 방법 모두 동일한 기능적 결과를 제공합니다. 래퍼가 표준적인 관용구입니다.

### TP / PP / FP8과의 상호작용

- **텐서 병렬화(Tensor Parallel):** 체크포인팅 입력은 재계산 시 수집(gather)하거나 재분산(rescatter)해야 하며, 통신 비용을 처리해야 합니다.
- **파이프라인 병렬화(Pipeline Parallel):** 일반적인 패턴은 각 파이프라인 단계의 순전파를 체크포인팅하여 역순 마이크로배치가 활성화 메모리를 재사용할 수 있도록 하는 것입니다.
- **FP8 재계산:** 재계산 중 업데이트된 amax 히스토리는 원본 순전파와 일치해야 하며, 그렇지 않으면 FP8 스케일이 드리프트(drift)됩니다. 대부분의 프레임워크는 스케일을 스냅샷합니다.

```figure
activation-recompute
```

## 구현하기

### 1단계: 세그먼트가 있는 토이 모델

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 2단계: 모든 활성화가 필요한 단순 역전파

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 3단계: k마다 체크포인트 메모리

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 4단계: 비용 모델

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 5단계: 메모리 추정기

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 6단계: 최적 세그먼트 크기

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 7단계: 선택적 체크포인트 결정

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 사용하기

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorch의 표준 래퍼입니다. 함수를 래핑하며, 입력만 저장하고 역전파 시 재계산합니다.
- **Megatron-Core 활성화 재계산**: `selective`, `full`, `block` 모드를 지원합니다. 2024년 이후 최첨단 훈련에서 표준입니다.
- **FSDP2 오프로드**: `module.to_empty(device="cpu")`와 `offload_policy`를 사용하면 FSDP2가 재계산 대신 활성화(CPU)를 오프로드합니다.
- **DeepSpeed ZeRO-Offload**: 옵티마이저 상태와 활성화의 CPU 오프로드를 지원하며, 체크포인팅을 보완합니다.

## 출시하기

이 강의는 `outputs/prompt-activation-recompute-policy.md`를 생성합니다. 모델 구성(레이어, 은닉 크기, 시퀀스 길이, 배치 크기)과 가용 GPU 메모리를 입력으로 받아 레이어별 재계산 정책(없음 / 선택적 / 전체 / 오프로드)을 출력하는 프롬프트입니다.

## 연습 문제

1. 정확성을 검증하세요. `model_forward` + `model_backward` (전체 활성화)와 `model_forward_checkpointed` + `model_backward_checkpointed` (세그먼트)를 실행해 보세요. 매개변수 기울기가 기계 정밀도 수준에서 동일해야 합니다.

2. 세그먼트 크기 `k`을 1부터 `L`까지 스윕해 보세요. FLOP 오버헤드와 메모리를 플롯하고 곡선의 무릎(knee) 지점을 찾아보세요.

3. 선택적 체크포인팅을 구현해 보세요. 어텐션 모듈의 입력은 저장하되 중간값은 저장하지 마세요. 시퀀스 길이 8192인 32레이어 모델에서 전체 레이어 체크포인팅 대비 FLOP 오버헤드를 측정해 보세요.

4. 오프로드를 추가하세요. 세그먼트 입력을 시뮬레이션된 "CPU 버퍼"(별도 리스트)에 저장하세요. "PCIe 대역폭"을 바이트/시간으로 측정하고 오프로드와 재계산 사이의 손익분기점을 찾아보세요.

5. 실제 PyTorch 트랜스포머를 `torch.utils.checkpoint` 사용 여부에 따라 벤치마킹해 보세요. 메모리(`torch.cuda.max_memory_allocated`를 통해)와 스텝 시간을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 기울기 체크포인팅 | "순전파를 다시 수행해 메모리를 절약" | 세그먼트 입력만 저장하고, 역전파 시 중간값을 재계산하여 기울기 지원 텐서를 얻습니다 |
| 활성화 재계산 | "체크포인팅과 동일" | 동일한 기법의 HPC식 명칭 |
| 세그먼트 크기 (k) | "체크포인트당 레이어 수" | 중간값을 버리고 함께 재물화하는 레이어 수 |
| 선택적 체크포인팅 | "Korthikanti의 트릭" | 저장 비용이 큰 활성화(어텐션 소프트맥스)만 재계산하고, 비용이 작은 것은 유지 |
| 전체 체크포인팅 | "소박한 버전" | 모든 세그먼트에서 모든 레이어의 중간값을 재계산 |
| 블록 체크포인팅 | "거친 단위" | 전체 트랜스포머 블록을 체크포인팅; 가장 큰 단위 |
| FLOP 오버헤드 | "연산 세금" | 단계당 추가 FLOP = (재계산 FLOP) / (순전파 + 역전파 FLOP); 소박한 방식은 33%, 선택적 방식은 5% |
| 활성화 오프로드 | "CPU로 전송" | 순전파->역전파 동안 활성화를 CPU RAM으로 이동; 재계산의 대안 |
| sqrt-L 규칙 | "고전적 최적값" | 균일한 비용의 레이어에 대해, 최적의 체크포인트 간격은 sqrt(L) 레이어 |
| 어텐션-소프트맥스 볼륨 | "O(L^2) 문제" | L^2 * 헤더 * 배치 floats; 긴 컨텍스트에서 활성화 메모리를 지배 |

## 추가 읽기

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 기울기 체크포인팅을 공식화한 원 논문
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 선택적 활성화 재계산 및 공식 비용 분석
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 역방향 모드 재물화를 통한 대안적 상수 메모리 접근법
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 대규모 활성화 오프로드
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 표준 API
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) -- 선택적, 전체 및 블록 모드
