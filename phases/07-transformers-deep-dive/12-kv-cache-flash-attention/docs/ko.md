# KV 캐시, Flash Attention 및 추론 최적화 (KV Cache, Flash Attention & Inference Optimization)

> 학습(Training)은 병렬적이며 연산량(FLOPs)에 제한을 받습니다. 추론(Inference)은 직렬적이며 메모리 대역폭(Memory-bound)에 제한을 받습니다. 병목 지점이 다르므로, 해결 방법도 달라야 합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## 문제점 (The Problem)

단순한 자기회귀(autoregressive) 디코더는 `N`개의 토큰을 생성하기 위해 `O(N²)`의 연산을 수행합니다. 즉, 각 단계마다 전체 접두사(prefix)에 대해 어텐션을 다시 계산합니다. 4K 토큰 응답의 경우 1,600만 번의 어텐션 연산이 필요한데, 이 중 대부분은 중복된 연산입니다. 접두사 토큰의 모든 은닉 상태(hidden state)는 한 번 계산되면 결정론적(deterministic)입니다. 따라서 새로운 토큰의 쿼리(query)를 이전 모든 토큰의 캐시된 키(key) 및 값(value)과 대조하기만 하면 됩니다.

게다가 어텐션 자체는 매우 많은 데이터를 이동시킵니다. 표준 어텐션은 $N \times N$ 점수 행렬, $N \times d$ 소프트맥스(softmax) 출력, $N \times d$ 최종 출력을 실체화(materialize)하므로, HBM(High Bandwidth Memory)에 대한 읽기 및 쓰기 작업이 너무 많습니다. $N \ge 2K$인 경우, 어텐션은 연산량(FLOP) 제한에 도달하기 전에 메모리 대역폭 제한(memory-bound)에 먼저 도달합니다. 기존의 어텐션 커널은 현대적인 GPU의 성능을 4~10배 정도 제대로 활용하지 못합니다.

Dao 등이 제안한 두 가지 최적화 기술은 최첨단 추론(frontier inference) 성능을 "느림"에서 "빠름"의 단계로 끌어올렸습니다:

1. **KV 캐시(KV cache).** 모든 접두사 토큰의 K와 V 벡터를 저장합니다. 각 새로운 토큰의 어텐션은 캐시된 키에 대한 단 한 번의 쿼리가 됩니다. 이를 통해 생성 단계당 추론 복잡도가 `O(N²)`에서 `O(N)`으로 감소합니다.
2. **플래시 어텐션(Flash Attention).** 어텐션 계산을 타일링(tiling)하여 전체 $N \times N$ 행렬이 HBM에 기록되지 않도록 합니다. 모든 소프트맥스 및 행렬 곱셈(matmul)은 SRAM 내에서 수행됩니다. A100에서는 실제 실행 시간(wall-clock time) 기준 2~4배, H100의 FP8 환경에서는 5~10배의 속도 향상을 제공합니다.

2026년경에는 이 두 기술이 보편화될 것입니다. 모든 프로덕션 추론 스택(vLLM, TensorRT-LLM, SGLang, llama.cpp)은 이 기술들을 전제로 합니다. 모든 최첨단 모델은 플래시 어텐션이 활성화된 상태로 출시됩니다.

## 개념 (The Concept)

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV 캐시 수학 (KV cache math)

디코더 레이어당, 토큰당, 헤드당 계산식:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K와 V
```

32개 레이어, 32개 헤드, `d_head`=128, `fp16`을 사용하는 7B 모델의 경우:

```
레이어당 토큰당 = 2 * 128 * 2 = 512 bytes
토큰당 (32개 레이어) = 16 KB
32K 컨텍스트당 = 512 MB
```

Llama 3 70B (80개 레이어, `d_head`=128, 8개의 KV 헤드를 사용하는 GQA 적용)의 경우:

```
레이어당 토큰당 = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
32K 컨텍스트당 = 10.4 GB
```

이 10 GB라는 크기 때문에, Llama 3 70B를 128K 컨텍스트에서 배치 사이즈 1로 구동할 때 KV 캐시만으로도 40 GB A100 메모리의 대부분을 사용하게 됩니다.

**GQA는 KV 캐시 측면에서의 승리입니다.** 64개 헤드를 사용하는 MHA였다면 32 GB가 되었을 것입니다. MLA는 이보다 훨씬 더 압축합니다.

차원(dimensions)을 드래그하여 캐시 크기가 변하는 것을 확인해 보세요. 시퀀스 길이(sequence length)나 배치(batch)를 높여서 단일 GPU의 용량을 얼마나 빠르게 초과하는지 살펴보세요.

```figure
kv-cache-sizer
```

### Flash Attention — 타일링 기법 (the tiling trick)

표준 어텐션(Standard attention):

```
S = Q @ K^T          (HBM 읽기, N×N, HBM 쓰기)
P = softmax(S)       (HBM 읽기, HBM 쓰기)
O = P @ V            (HBM 읽기, HBM 쓰기)
```

총 세 번의 HBM 왕복(round trips)이 발생합니다. H100의 경우 HBM 대역폭은 3 TB/s인 반면, SRAM은 30 TB/s입니다. 모든 데이터를 온칩(on-chip)에 유지하는 것과 비교하면, 매 HBM 왕복은 10배의 속도 저하를 초래합니다.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

타일당 한 번의 HBM 왕복만 수행합니다. 전체 메모리 점유율(memory footprint)이 `O(N²)`에서 `O(N)`으로 감소합니다. 역전파(Backward pass) 단계에서는 값을 저장하는 대신 순전파(forward pass)의 일부 값을 재계산하여 메모리 효율을 한 번 더 높입니다.

**수치적 기법(Numerical trick).** 실행 중인 softmax는 타일 전반에 걸쳐 `(max, sum)`을 유지하므로 최종 정규화(normalization)가 정확하게 이루어집니다. 이는 근사치가 아닙니다. Flash Attention은 표준 어텐션과 비트 단위로 동일한 출력(bit-identical output)을 계산합니다 (fp16의 비결합성(non-associativity) 제외).

**버전 진화(Version evolution):**

| 버전 | 연도 | 주요 변경 사항 | 기준 하드웨어에서의 속도 향상 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 타일링된 SRAM 커널 | A100에서 2배 |
| Flash 2 | 2023 | 향상된 병렬성, causal-first 순서 지정 | A100에서 3배 |
| Flash 3 | 2024 | Hopper 비동기성, FP8 | H100에서 1.5–2배 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5단계 파이프라인, 소프트웨어 exp2 | 추론 우선 (초기에는 순전파만 지원) |

Flash 4는 출시 시점에 순전파(forward-pass)만 지원합니다. 학습(Training)에는 여전히 Flash 3가 사용됩니다. Flash 4의 GQA 및 가변 길이(varlen) 지원은 대기 중입니다 (2026년 중반 예정).

### Speculative decoding (추측 디코딩) — 또 다른 지연 시간(Latency) 개선책

저렴한 모델(Cheap model)이 $N$개의 토큰을 제안합니다. 거대 모델(Big model)은 이 $N$개를 병렬로 모두 검증합니다. 검증 결과 $k$개의 토큰이 수락되면, 단 한 번의 거대 모델 순전파(forward pass)로 $k$개의 토큰을 생성한 셈이 됩니다. 코드 및 산문(prose) 작업에서 일반적인 $k$값은 3~5 사이입니다.

2026년의 기본 기술들:
- **EAGLE 2 / Medusa.** 검증기의 은닉 상태(hidden states)를 공유하는 통합된 초안 헤드(draft heads)를 사용합니다. 품질 저하 없이 2~3배의 속도 향상을 제공합니다.
- **초안 모델을 사용한 추측 디코딩(Speculative decoding with draft model).** 소비자용 하드웨어에서 2~4배의 속도 향상을 제공합니다.
- **Lookahead decoding.** Jacobi iteration 방식을 사용하며, 별도의 초안 모델이 필요하지 않습니다. 특정 분야에 국한되지만 비용이 들지 않습니다.

### 연속 배칭 (Continuous batching)

전통적인 배치 추론(Classic batched inference): 가장 느린 시퀀스가 끝날 때까지 기다린 후 새로운 배치를 시작합니다. 짧은 응답이 일찍 끝날 경우 GPU 자원이 낭비됩니다.

연속 배칭 (Continuous batching, Orca에서 처음 도입되었으며 현재 vLLM, TensorRT-LLM, SGLang에 탑재됨): 기존 요청이 완료되는 즉시 새로운 요청을 배치로 교체합니다. 일반적인 채팅 워크로드에서 5~10배의 처리량(throughput) 향상을 제공합니다.

### PagedAttention — 가상 메모리로서의 KV 캐시(KV cache)

vLLM의 핵심 기능입니다. KV 캐시는 16개 토큰 단위의 블록으로 할당되며, 페이지 테이블(page table)이 논리적 위치를 물리적 블록에 매핑합니다. 이를 통해 병렬 샘플링(빔 서치, 병렬 샘플링 등) 간에 KV를 공유하고, 프롬프트 캐싱을 위해 접두사(prefix)를 핫스왑(hot-swap)하며, 메모리 단편화를 방지할 수 있습니다. 단순한 연속적 할당(contiguous allocation) 방식보다 처리량(throughput)을 4배 향상시킵니다.

```figure
flash-attention-memory
```

## 구현하기 (Build It)

`code/main.py`를 확인해 보세요. 다음을 구현합니다:

1. 나이브(naive)한 `O(N²)` 증분 디코더(incremental decoder).
2. `O(N)` KV 캐시 기반 디코더(KV-cached decoder).
3. Flash Attention의 running-max 알고리즘을 시뮬레이션하는 타일형 소프트맥스(tiled softmax).

### 1단계: KV 캐시 (KV cache)

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

단순한 방식: 레이어별, 헤드별 리스트에 토큰당 K, V 벡터를 계속해서 추가합니다.

### 2단계: 타일형 소프트맥스 (tiled softmax)

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

한 번에 수행하는 `softmax(qK) V`와 비트 단위로 동일한(Bit-identical) 출력을 생성하지만, 작업 세트(working set)는 전체 `N × d_head`가 아니라 항상 `tile × d_head` 블록 크기로 유지됩니다.

### 3단계: 100토큰 생성 시 Naive(기본) 디코딩과 Cached(캐싱) 디코딩 비교

Attention 연산 횟수를 계산합니다. Naive 방식: `O(N²)` = 5050. Cached 방식: `O(N)` = 100. 코드는 두 결과값을 모두 출력합니다.

## 사용 방법 (Use It)

```python
# HuggingFace transformers는 decoder-only 모델의 generate() 호출 시 KV 캐시를 자동으로 활성화합니다.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 아키텍처라면 FA3를 사용하세요
    torch_dtype="bfloat16",
)
# generate()는 KV 캐시를 자동으로 사용합니다
```

vLLM 프로덕션 환경:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

요청 간 프리픽스 캐싱(Prefix caching)은 2026년의 거대한 승리입니다. 동일한 시스템 프롬프트, 퓨샷(few-shot) 예시 또는 긴 컨텍스트 문서를 사용할 때 호출 간에 KV 캐시를 재사용할 수 있기 때문입니다. 반복적인 도구 프롬프트(tool prompts)가 발생하는 에이전트 워크로드의 경우, 프리픽스 캐싱을 통해 통상적으로 5배의 처리량(throughput) 향상을 얻을 수 있습니다.

## Ship It (실행하기)

`outputs/skill-inference-optimizer.md`를 참조하세요. 이 스킬은 새로운 추론 배포(inference deployment)를 위해 어텐션 구현(attention implementation), KV 캐시 전략(KV cache strategy), 양자화(quantization), 그리고 추측 디코딩(speculative decoding)을 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. Naive 디코더와 캐시된(cached) 디코더가 동일한 출력을 생성하는지 확인하고, 연산 횟수(op-count)의 차이를 기록해 보세요.
2. **중간 (Medium).** 프리픽스 캐싱(prefix caching)을 구현해 보세요. 프롬프트 `P`와 여러 개의 완성 문구(completions)가 주어졌을 때, `P`에 대해 한 번의 순전파(forward pass)를 수행하여 KV 캐시를 채운 뒤, 각 완성 문구별로 분기하여 실행합니다. 각 문구마다 `P`를 다시 인코딩하는 방식과 비교하여 속도 향상(speedup)을 측정해 보세요.
3. **어려움 (Hard).** 토이 버전의 PagedAttention을 구현해 보세요. KV 캐시를 고정된 16-토큰 블록 단위로 관리하며 프리 리스트(free-list)를 사용합니다. 시퀀스가 종료되면 해당 블록들을 풀(pool)로 반환합니다. 길이가 다양한 1,000개의 채팅 완성 문구를 시뮬레이션해 보세요. 메모리 파편화(memory fragmentation)와 연속 할당(contiguous allocation) 방식을 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 정의 | 실제 의미 |
|------|-----------------|-----------------------|
| KV cache | "디코딩을 빠르게 만드는 트릭" | 모든 접두사(prefix) 토큰으로부터 저장된 K와 V 값; 새로운 쿼리는 이를 재계산하는 대신 참조합니다. |
| HBM | "GPU 메인 메모리" | 고대역폭 메모리(High Bandwidth Memory); H100은 80GB, B200은 192GB를 탑재하며, 대역폭은 약 3 TB/s입니다. |
| SRAM | "온칩 메모리(On-chip memory)" | SM(Streaming Multiprocessor)별 고속 메모리; H100의 경우 SM당 약 256 KB이며, 대역폭은 약 30 TB/s입니다. |
| Flash Attention | "타일링된 어텐션 커널(Tiled attention kernel)" | HBM에 $N \times N$ 행렬을 실제로 생성하지 않고 어텐션을 계산합니다. |
| Continuous batching | "대기 없는 배칭(No-wait batching)" | 배치를 비우지 않고도 완료된 시퀀스는 교체(swap out)하고 새로운 시퀀스를 투입(in)합니다. |
| PagedAttention | "vLLM의 핵심 기술" | 페이지 테이블을 사용하여 KV cache를 고정된 블록 단위로 할당하며, 단편화(fragmentation)를 제거합니다. |
| Prefix caching | "긴 프롬프트 재사용" | 요청 간에 공유되는 접두사에 대한 KV를 캐싱합니다; 에이전트(agents) 운영 시 비용을 크게 절감합니다. |
| Speculative decoding | "초안 작성 + 검증(Draft + verify)" | 저렴한 초안 모델(draft model)이 토큰을 제안하면, 거대 모델이 한 번의 패스(pass)로 $k$개의 토큰을 검증합니다. |

## 추가 학습 자료 (Further Reading)

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5단계 파이프라인(5-stage pipeline) 및 software-exp2 트릭; 본 레슨에서 언급한 순방향 전용 실행(forward-only launch) 주의사항은 저장소의 README를 읽어보세요.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 논문.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 추측 디코딩(speculative decoding).
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 본 레슨에서 인용된 통합 초안(integrated-draft) 방식에 관한 EAGLE-1/2 논문.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLE과 함께 언급된 Medusa 방식.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16-토큰 블록 및 페이지 테이블(page-table) 설계에 관한 표준 심층 분석 자료.
