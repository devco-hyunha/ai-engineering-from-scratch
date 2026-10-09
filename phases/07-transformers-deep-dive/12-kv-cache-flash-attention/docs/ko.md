# KV 캐시, Flash Attention 및 추론 최적화

> 학습은 병렬적이고 FLOP에 제한됩니다. 추론은 순차적이고 메모리에 제한됩니다. 병목 현상이 다르므로, 트릭도 다릅니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션), 7단계 · 05강 (전체 트랜스포머), 7단계 · 07강 (GPT)
**시간:** 약 75분

## 문제점

소박한 자기회귀 디코더는 `O(N²)` 연산을 수행하여 `N` 토큰을 생성합니다. 각 단계에서 전체 접두어에 대한 어텐션을 다시 계산하기 때문입니다. 4K 토큰 응답의 경우 16M 어텐션 연산이 필요하며, 그 대부분은 중복됩니다. 접두어 토큰의 모든 은닉 상태는 한 번 계산되면 결정적입니다. 모든 이전 키와 값에 대해 캐시된 키와 값에 대해 새 토큰의 쿼리만 실행하면 됩니다.

게다가 어텐션은 많은 데이터를 이동시킵니다. 표준 어텐션은 N×N 점수 매트릭스, N×d 소프트맥스 출력, N×d 최종 출력을 구체화합니다. HBM에 대한 읽기와 쓰기가 너무 많습니다. N≥2K인 경우, 어텐션은 FLOP에 제한되기 전에 메모리에 제한됩니다. 고전적인 어텐션 커널은 최신 GPU를 4~10배 과소 활용합니다.

Dao 등이 제안한 두 가지 최적화가 최전선 추론을 "느림"에서 "빠름"으로 밀어냈습니다:

1. **KV 캐시.** 모든 접두어 토큰의 K 및 V 벡터를 저장합니다. 각 새 토큰의 어텐션은 캐시된 키에 대한 하나의 쿼리입니다. 추론은 생성 단계당 `O(N²)`에서 `O(N)`로 감소합니다.
2. **Flash Attention.** 어텐션 연산을 타일링하여 전체 N×N 매트릭스가 HBM에 도달하지 않도록 합니다. 모든 소프트맥스 + 매트릭스 곱이 SRAM에서 발생합니다. A100에서 2~4배의 벽시계 시간 단축; H100에서 FP8을 사용하면 5~10배 단축됩니다.

2026년까지 두 기술 모두 보편화되었습니다. 모든 프로덕션 추론 스택(vLLM, TensorRT-LLM, SGLang, llama.cpp)은 이를 전제로 합니다. 모든 최전선 모델은 Flash Attention이 활성화된 상태로 출시됩니다.

## 개념

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV 캐시 수학

디코더 레이어별, 토큰별, 헤드별:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

32 레이어, 32 헤드, d_head=128, fp16인 7B 모델의 경우:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Llama 3 70B (80 레이어, d_head=128, 8개 KV 헤드를 가진 GQA)의 경우:

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

이 10 GB는 Llama 3 70B가 128K 컨텍스트에서 배치 크기 1로 KV 캐시에 40 GB A100의 대부분을 필요로 하는 이유입니다.

**GQA는 KV 캐시 측면의 승리입니다.** 64개 헤드를 가진 MHA는 32 GB가 됩니다. MLA는 이를 더욱 압축합니다.

차원을 드래그하여 캐시 크기가 변하는 것을 지켜보세요. 시퀀스 길이 또는 배치를 늘려서 단일 GPU를 얼마나 빠르게 초과하는지 확인해 보세요:

```figure
kv-cache-sizer
```

### Flash Attention — 타일링 트릭

표준 어텐션:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

HBM 왕복이 세 번 발생합니다. H100에서 HBM 대역폭은 3 TB/s이고, SRAM은 30 TB/s입니다. 모든 HBM 왕복은 모든 것을 칩 내부에 유지하는 것에 비해 10배의 속도 저하를 의미합니다.

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

타일당 HBM 왕복이 한 번입니다. 총 메모리 사용량이 `O(N²)`에서 `O(N)`로 감소합니다. 역전파는 저장하는 대신 순전파에서 일부 값을 재계산합니다 — 이는 또 다른 메모리 측면의 승리입니다.

**수치적 트릭.** Softmax는 타일 전체에 걸쳐 `(max, sum)`을 유지하므로 최종 정규화가 정확합니다. 근사치가 아닙니다 — Flash Attention은 표준 어텐션과 비트 단위로 동일한 출력을 계산합니다(fp16 비결합성 제외).

**버전 발전:**

| 버전 | 연도 | 주요 변경 사항 | 참조 하드웨어에서의 속도 향상 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 타일링된 SRAM 커널 | A100에서 2배 |
| Flash 2 | 2023 | 개선된 병렬 처리, 인과 우선 순서 | A100에서 3배 |
| Flash 3 | 2024 | Hopper 비동기성, FP8 | H100에서 1.5–2배 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5단계 파이프라인, 소프트웨어 exp2 | 추론 우선(처음에는 순전파만) |

Flash 4는 출시 시 순전파 전용입니다. 학습은 여전히 Flash 3을 사용합니다. Flash 4의 GQA 및 varlen 지원은 보류 중입니다(2026년 중반).

### 추론적 디코딩(Speculative Decoding) — 다른 지연 측면의 승리

저비용 모델이 N개의 토큰을 제안합니다. 대형 모델이 모든 N개를 병렬로 검증합니다. 검증이 k개의 토큰을 승인하면, k개의 생성을 위해 대형 모델 순전파 한 번의 비용을 지불한 셈입니다. 코드와 산문에서는 일반적으로 k=3–5입니다.

2026년 기본값:
- **EAGLE 2 / Medusa.** 검증자의 은닉 상태를 공유하는 통합된 초안 헤드입니다. 품질 손실 없이 2–3배의 속도 향상.
- **초안 모델을 사용한 추론적 디코딩(Speculative Decoding).** 소비자 하드웨어에서 2–4배의 속도 향상.
- **선제적 디코딩(Lookahead decoding).** 야코비 반복; 초안 모델이 필요하지 않습니다. 틈새 영역이지만 무료입니다.

### 연속 배치(Continuous Batching)

전통적인 배치 추론: 가장 느린 시퀀스가 완료될 때까지 기다린 후 새 배치를 시작합니다. 짧은 응답이 일찍 끝나면 GPU가 낭비됩니다.

연속 배치(Continuous Batching)(Continuous Batching)(Orca에서 처음 출시되었으며, 현재 vLLM, TensorRT-LLM, SGLang에 포함됨): 오래된 요청이 완료되면 즉시 새 요청을 배치로 교체합니다. 일반적인 채팅 워크로드에서 처리량이 5~10배 향상됩니다.

### PagedAttention — 가상 메모리로서의 KV 캐시

vLLM의 핵심 기능입니다. KV 캐시는 16토큰 블록으로 할당되며, 페이지 테이블이 논리적 위치를 물리적 블록에 매핑합니다. 이를 통해 병렬 샘플링(빔 검색, 병렬 샘플링) 간에 KV를 공유하고, 프롬프트 캐싱을 위해 접두어를 핫 스왑하며, 메모리를 조각화 해제할 수 있습니다. 단순 연속 할당 대비 처리량이 4배 향상됩니다.

```figure
flash-attention-memory
```

## 구현하기

`code/main.py`를 참조하세요. 다음을 구현합니다:

1. 단순한 `O(N²)` 증분 디코더.
2. `O(N)` KV 캐시 디코더.
3. Flash Attention의 런닝-맥스 알고리즘을 시뮬레이션하는 타일링된 소프트맥스(Softmax)(Softmax).

### 1단계: KV 캐시

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

단순합니다: 레이어별, 헤드별 리스트에 토큰별 K, V 벡터를 계속 확장합니다.

### 2단계: 타일링된 소프트맥스(Softmax)

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

한 번에 `softmax(qK) V`와 비트 단위로 동일한 출력을 생성하지만, 작업 세트는 항상 `tile × d_head` 블록이며 전체 `N × d_head`가 아닙니다.

### 3단계: 100토큰 생성에서 단순 디코딩과 캐시된 디코딩 비교

어텐션 연산 횟수를 세어 보세요. 단순 방식: `O(N²)` = 5050. 캐시 방식: `O(N)` = 100. 코드가 두 값을 모두 출력합니다.

## 사용하기

```python
# HuggingFace transformers는 decoder-only generate()에서 KV 캐시를 자동으로 활성화합니다.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper인 경우 FA3 사용
    torch_dtype="bfloat16",
)
# generate()는 KV 캐시를 자동으로 사용
```

vLLM 프로덕션:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

요청 간 접두어 캐싱(Prefix Caching)(Prefix Caching)은 2026년의 큰 성과입니다. 동일한 시스템 프롬프트(System Prompt)(System Prompt), 소수 예시(Few-Shot)(Few-Shot) 예제, 또는 긴 컨텍스트 문서가 호출 간에 KV를 재사용합니다. 반복적인 도구 프롬프트가 있는 에이전트 워크로드에서는 접두어 캐싱이 처리량을 정기적으로 5배 향상시킵니다.

## 출시하기

`outputs/skill-inference-optimizer.md`를 참조하세요. 이 스킬은 새로운 추론 배포를 위해 어텐션 구현, KV 캐시 전략, 양자화(Quantization)(Quantization), 추론적 디코딩(Speculative Decoding)(Speculative Decoding)을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 네이브 및 캐시된 디코더가 동일한 출력을 생성하는지 확인하고, 연산 횟수 차이를 기록하세요.
2. **중간.** 접두어 캐싱을 구현하세요: 프롬프트 P와 여러 완료(completions)가 주어졌을 때, P에 대해 한 번의 순방향 패스를 실행하여 KV 캐시를 채운 후, 완료별로 분기하세요. 각각에 대해 P를 다시 인코딩하는 것과 비교하여 속도 향상을 측정하세요.
3. **어려움.** 토이 PagedAttention을 구현하세요: 고정된 16-토큰 블록과 자유 목록(free-list)을 사용하여 KV 캐시를 관리하세요. 시퀀스가 완료되면 해당 블록을 풀(pool)에 반환하세요. 길이가 다양한 1,000개의 채팅 완료를 시뮬레이션하세요. 메모리 단편화와 연속 할당을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| KV 캐시 | "디코딩을 빠르게 만드는 비법" | 모든 접두어 토큰의 K와 V가 저장됨; 새로운 쿼리는 재계산 대신 이를 참조합니다. |
| HBM | "GPU 주 메모리" | 고대역폭 메모리(High Bandwidth Memory); H100에서 80 GB, B200에서 192 GB. 대역폭은 약 3 TB/s입니다. |
| SRAM | "온칩 메모리" | SM별 고속 메모리, H100의 경우 SM당 약 256 KB. 대역폭은 약 30 TB/s입니다. |
| Flash Attention | "타일링된 어텐션 커널" | HBM에 N×N을 물리적으로 생성하지 않고 어텐션을 계산합니다. |
| 연속 배치 | "대기 없는 배치" | 완료된 시퀀스를 빼고 새로운 시퀀스를 넣으며, 배치를 비우지 않습니다. |
| PagedAttention | "vLLM의 헤드라인" | 페이지 테이블을 사용하여 고정 블록에 KV 캐시를 할당; 단편화를 제거합니다. |
| 접두어 캐싱 | "긴 프롬프트 재사용" | 요청 간 공유 접두어에 대한 KV를 캐시; 에이전트에서 비용이 크게 절감됩니다. |
| 추론적 디코딩 | "초안 + 검증" | 저렴한 초안 모델이 토큰을 제안하고, 큰 모델이 한 번의 패스로 k개를 검증합니다. |

## 추가 읽기

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5단계 파이프라인 및 소프트웨어 exp2 트릭; 이 강의에서 언급된 forward-only 런치 주의사항은 저장소 README를 읽어 보세요.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 논문.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 추론적 디코딩.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 강의에서 인용된 통합 초안 접근 방식에 대한 EAGLE-1/2 논문.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLE과 함께 참조된 Medusa 접근 방식.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16토큰 블록 및 페이지 테이블 설계에 대한 표준 심층 분석입니다.
