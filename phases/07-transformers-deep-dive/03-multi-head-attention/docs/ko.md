# 멀티 헤드 어텐션 (Multi-Head Attention)

> 하나의 어텐션 헤드는 한 번에 하나의 관계를 학습합니다. 여덟 개의 헤드는 여덟 개의 관계를 학습합니다. 헤드는 비용이 들지 않습니다. 더 많이 활용해 보세요.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## 문제점 (The Problem)

단일 셀프 어텐션 헤드(single self-attention head)는 하나의 어텐션 행렬을 계산합니다. 이 행렬은 한 가지 종류의 관계만을 포착하는데, 대개는 주어진 학습 신호에서 손실(loss)을 최소화하는 관계입니다. 만약 데이터에 주어-동사 일치(subject-verb agreement), 상호 참조(co-reference), 장거리 담화(long-range discourse), 구문론적 청킹(syntactic chunking)이 모두 뒤섞여 있다면, 단일 헤드는 이를 하나의 소프트맥스(softmax) 분포로 뭉뚱그려 버려 신호의 절반을 잃게 됩니다.

2017년 Vaswani 논문에서 제시한 해결책은 다음과 같습니다: 각각 고유한 `Q`, `K`, `V` 투영(projection)을 가진 여러 개의 어텐션 함수를 병렬로 실행하고 그 출력들을 연결(concatenate)하는 것입니다. 각 헤드는 `d_model / n_heads` 차원의 더 작은 부분 공간(subspace)에서 작동합니다. 전체 파라미터 수는 동일하게 유지되면서 표현력(expressive power)은 향상됩니다.

멀티 헤드 어텐션(Multi-head attention)은 2026년에 출시되는 모든 트랜스포머의 기본 사양입니다. 유일한 논쟁점은 헤드를 *얼마나 많이* 사용할 것인지, 그리고 키(key)와 값(value)이 투영을 공유할 것인지 여부(Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention)입니다.

## 개념 (The Concept)

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**분할(Split).** `(N, d_model)` 형태의 `X`를 가져옵니다. 이를 각각 `(N, d_model)` 형태의 Q, K, V로 투영(Project)합니다. 이후 `d_head = d_model / n_heads`가 되도록 `(N, n_heads, d_head)` 형태로 재구성(Reshape)합니다. 마지막으로 `(n_heads, N, d_head)`로 전치(Transpose)합니다.

**병렬 어텐션(Attend in parallel).** 각 헤드 내부에서 스케일드 닷-프로덕트 어텐션(scaled dot-product attention)을 실행합니다. 각 헤드는 `(N, d_head)`를 생성합니다. 각 헤드는 임베딩의 서로 다른 하위 공간(subspaces)에서 작동하며, 어텐션 계산 과정 자체에서는 서로 통신하지 않습니다.

**결합 및 투영(Concatenate and project).** 헤드들을 다시 `(N, d_model)` 형태로 쌓고(Stack), `(d_model, d_model)` 형태의 학습된 출력 행렬 `W_o`를 곱합니다. `W_o`는 헤드들이 서로 섞일 수 있는 지점입니다.

**작동 원리.** 각 헤드는 표현 예산(representational budget)을 두고 다른 헤드와 경쟁하지 않고도 전문화될 수 있습니다. 2019~2024년의 프로빙(Probing) 연구들은 다음과 같은 뚜렷한 헤드 역할들을 보여줍니다: 위치 헤드(positional heads), 이전 토큰에 어텐션을 주는 헤드, 복사 헤드(copy heads), 개체명 헤드(named-entity heads), 인덕션 헤드(induction heads, 인컨텍스트 학습의 기초가 됨).

**2026년 변형 모델 계보(The 2026 lineage of variations):**

| 변형 모델 (Variant) | Q 헤드 수 | K/V 헤드 수 | 사용 사례 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (예: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | 저차원(low-rank)으로 압축 | DeepSeek-V2, V3 |

GQA는 품질을 거의 그대로 유지하면서 KV-캐시 메모리를 `N/G` 배만큼 절감할 수 있기 때문에 현대적인 기본값(default)으로 자리 잡았습니다. MLA는 한 걸음 더 나아가 K/V를 잠재 공간(latent space)으로 압축한 뒤 연산 시점에 다시 투영합니다. 이는 연산량(FLOPs)을 소모하지만, 훨씬 더 많은 메모리를 절약합니다.

```figure
multihead-split
```

## 직접 구현해 보기 (Build It)

### 1단계: 이미 보유한 싱글 헤드 어텐션(single-head attention)에서 헤드 분할하기

Lesson 02의 `SelfAttention`을 가져와 split/concat 쌍으로 감싸보세요. numpy 구현 방식은 `code/main.py`를 참조하시기 바랍니다. 로직은 다음과 같습니다:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

한 번의 reshape과 한 번의 transpose만 사용합니다. 루프는 없습니다. 이것이 바로 PyTorch의 `nn.MultiheadAttention` 내부에서 수행되는 방식과 정확히 일치합니다.

### 2단계: 헤드별 스케일드 닷-프로덕트 어텐션(scaled dot-product attention) 실행

각 헤드는 `Q`, `K`, `V`의 고유한 슬라이스를 할당받습니다. 어텐션은 배치 행렬 곱셈(batched matmul)이 됩니다:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

실제 하드웨어에서 `Qh @ Kh.transpose(...)`는 하나의 `bmm`(batch matrix multiplication) 연산입니다. GPU는 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 형태의 단일 배치 행렬 곱셈으로 인식합니다. 따라서 헤드를 추가하는 데 드는 비용은 거의 없습니다.

### 3단계: Grouped-Query Attention 변형 (Grouped-Query Attention variant)

Key와 Value 프로젝션만 변경됩니다. Q는 `n_heads`개의 그룹을 가지며, K와 V는 `n_kv_heads < n_heads`개의 그룹을 가진 뒤 이를 맞추기 위해 반복됩니다:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

추론(Inference) 시에는 `n_heads`가 아닌 `n_kv_heads`만큼의 복사본만 KV 캐시에 존재하므로 메모리를 절약할 수 있습니다. Llama 3 70B는 64개의 Query 헤드와 8개의 KV 헤드를 사용하여 캐시 크기를 8배 줄였습니다.

### 4단계: 각 헤드가 무엇을 학습했는지 조사하기 (Probe what each head learned)

4개의 헤드를 가진 짧은 문장에 대해 MHA(Multi-Head Attention)를 실행합니다. 각 헤드에 대해 `(N, N)` 어텐션 행렬(attention matrix)을 출력해 보세요. 무작위 초기화 상태에서도 각 헤드가 서로 다른 구조를 포착하는 것을 확인할 수 있습니다. 이는 부분적으로는 신호(signal) 때문이며, 부분적으로는 서브스페이스(subspaces) 내의 회전 대칭성(rotational symmetry) 때문입니다.

## 사용 방법 (Use It)

PyTorch에서의 한 줄 구현 버전:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 버전 기준 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention은 CUDA 환경에서 Flash Attention을 자동으로 호출합니다.
# GQA를 사용하려면 (B, n_heads, N, d_head) 형태의 Q와 
# (B, n_kv_heads, N, d_head) 형태의 K, V를 전달하세요. PyTorch가 반복(repeat) 작업을 처리합니다.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**헤드 수는 몇 개로 해야 할까요?** 2026년 프로덕션 모델들의 경험칙(Rules of thumb)은 다음과 같습니다:

| 모델 크기 (Model size) | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`는 거의 항상 64 또는 128로 설정됩니다. 이는 하나의 헤드가 "볼 수 있는" 양의 단위입니다. 32 미만으로 떨어지면 헤드가 스케일링 인자인 `sqrt(d_head)`와 충돌하기 시작하며, 256을 넘어가면 "다수의 작은 전문가(many small specialists)"가 갖는 이점을 잃게 됩니다.

## Ship It (실행해 보기)

`outputs/skill-mha-configurator.md`를 참조하세요. 이 스킬은 파라미터 예산(parameter budget), 시퀀스 길이(sequence length), 배포 대상(deployment target)이 주어졌을 때 새로운 트랜스포머(transformer)를 위한 헤드 수(head count), KV 헤드 수(kv-head count), 그리고 프로젝션 전략(projection strategy)을 추천합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 있는 MHA를 가져와서 `d_model=64`를 고정한 상태로 `n_heads`를 1에서 16으로 변경해 보세요. 합성 데이터 복제 작업(synthetic copy task)에서 아주 작은 1계층 모델의 손실(loss)을 그래프로 그려 보세요. 헤드 수가 많아지면 도움이 되나요, 정체되나요, 아니면 성능이 저하되나요?
2. **중간 (Medium).** MQA(Multi-Query Attention: 하나의 KV 헤드를 모든 Query 헤드가 공유)를 구현해 보세요. 전체 MHA와 비교했을 때 파라미터 수가 얼마나 감소하는지 측정해 보세요. 또한 $N=2048$일 때 추론 과정에서 KV 캐시(KV-cache) 크기가 얼마나 줄어드는지 계산해 보세요.
3. **어려움 (Hard).** Multi-head Latent Attention의 축소 버전을 구현해 보세요: $K, V$를 랭크(rank) `r`인 잠재 벡터(latent)로 압축하고, 이 잠재 벡터를 KV 캐시에 저장한 뒤, 어텐션 시점에 압축을 해제합니다. 검증 데이터의 perplexity(ppl)가 1비트 이내의 차이를 유지하면서, 캐시 메모리가 전체 MHA의 1/8 미만으로 떨어지는 `r` 값은 얼마인가요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| Head (헤드) | "단일 어텐션 회로" | 고유한 어텐션 행렬을 가진 `d_head = d_model / n_heads` 차원의 Q/K/V 프로젝션 하나. |
| d_head | "헤드 차원" | 헤드당 은닉층 너비(hidden width); 실제 서비스 환경에서는 거의 항상 64 또는 128임. |
| Split / combine (분할/결합) | "Reshape 트릭" | 어텐션 전후로 수행되는 `(N, d_model) ↔ (n_heads, N, d_head)` 형태의 reshape 및 transpose 연산. |
| W_o | "출력 프로젝션" | 헤드들을 결합(concatenate)한 후 적용되는 `(d_model, d_model)` 행렬; 헤드들이 서로 섞이는 지점. |
| MQA | "하나의 KV 헤드" | Multi-Query Attention: 단일 공유 K/V 프로젝션을 사용함. KV 캐시가 가장 작지만, 품질 저하가 발생할 수 있음. |
| GQA | "Llama 2 이후의 기본값" | Grouped-Query Attention: `n_kv_heads < n_heads` 구조이며, Q의 개수에 맞추기 위해 반복 사용됨. |
| MLA | "DeepSeek의 트릭" | Multi-head Latent Attention: K, V를 저차원 잠재 공간(low-rank latent)으로 압축한 뒤, 어텐션 시점에 다시 복원함. |
| Induction head (유도 헤드) | "인컨텍스트 학습(ICL)의 배후에 있는 회로" | 이전의 출현을 감지하고 그 뒤에 따라왔던 내용을 복사하는 한 쌍의 헤드. |

## 추가 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 멀티 헤드(multi-head)에 대한 원본 명세입니다.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA(Multi-Query Attention) 논문입니다.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 학습 후 MHA를 GQA로 변환하는 방법에 관한 내용입니다.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA와 이것이 캐시 메모리 측면에서 왜 MHA/GQA보다 우수한지에 대한 보고서입니다.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 헤드가 실제로 수행하는 역할에 대한 기계론적 분석(mechanistic look)을 다룹니다.
