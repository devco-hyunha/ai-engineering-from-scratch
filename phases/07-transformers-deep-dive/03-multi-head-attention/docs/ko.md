# 멀티 헤드 어텐션(Multi-Head Attention)

> 하나의 어텐션 헤드는 한 번에 하나의 관계를 학습합니다. 8개의 헤드는 8개를 학습합니다. 헤드는 자유롭습니다. 더 많이 가져가세요.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션(Self-Attention) 직접 구현)
**시간:** 약 75분

## 문제점

단일 셀프 어텐션 헤드는 하나의 어텐션 행렬을 계산합니다. 이 행렬은 하나의 종류의 관계를 포착합니다. 보통은 학습 신호에 대해 손실을 최소화하는 관계입니다. 데이터에 주어-동사 일치, 참조(co-reference), 장거리 담화, 구문 청킹이 모두 얽혀 있다면, 단일 헤드는 이것들을 하나의 소프트맥스 분포로 뭉개버려 신호의 절반을 잃게 됩니다.

2017년 Vaswani 논문에서의 해결책: 여러 어텐션 함수를 병렬로 실행하고, 각각 고유한 Q, K, V 투영을 가지며, 출력들을 연결(concatenate)합니다. 각 헤드는 차원이 `d_model / n_heads`인 더 작은 부분 공간에서 작동합니다. 총 매개변수 수는 그대로입니다. 표현력은 향상됩니다.

멀티 헤드 어텐션은 2026년 모든 트랜스포머(transformer)가 기본으로 탑재하는 기능입니다. 유일한 논쟁은 *헤드의 개수*가 몇 개인지, 그리고 키와 값이 투영을 공유하는지 여부(Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention)입니다.

## 개념

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**분할.** `(N, d_model)` 모양의 `X`를 가져오세요. Q, K, V 각각 `(N, d_model)` 모양으로 투영하세요. `d_head = d_model / n_heads`인 `(N, n_heads, d_head)`로 재형상(reshape)하세요. `(n_heads, N, d_head)`로 전치(transpose)하세요.

**병렬로 어텐션.** 각 헤드 내부에서 스케일된 점곱 어텐션(scaled dot-product attention)을 실행하세요. 각 헤드는 `(N, d_head)`를 생성합니다. 헤드는 임베딩의 서로 다른 부분 공간에서 작동하며, 어텐션 계산 자체 중에는 서로 대화하지 않습니다.

**연결 및 투영.** 헤드를 `(N, d_model)`로 다시 쌓고, `(d_model, d_model)` 모양의 학습된 출력 행렬 `W_o`과 곱하세요. `W_o`는 헤드가 혼합되는 지점입니다.

**작동하는 이유.** 각 헤드는 표현력 예산을 서로 경쟁하지 않고 전문화할 수 있습니다. 2019–2024년 프로빙(probing) 연구는 명확한 헤드 역할을 보여줍니다: 위치 헤드, 이전 토큰에 어텐션하는 헤드, 복사 헤드, 고유 명사 헤드, 유도 헤드(induction heads, 인컨텍스트 학습의 기반).

**2026년 변형 계보:**

| 변형 | Q 헤드 | K/V 헤드 | 사용 모델 |
|---------|---------|-----------|---------|
| 멀티 헤드 (MHA) | N | N | GPT-2, BERT, T5 |
| 멀티 쿼리 (MQA) | N | 1 | PaLM, Falcon |
| 그룹 쿼리 (GQA) | N | G (예: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 멀티 헤드 잠재 (MLA) | N | 저랭크로 압축 | DeepSeek-V2, V3 |

GQA는 `N/G`만큼 KV 캐시 메모리를 줄이면서 거의 완전한 품질을 유지하기 때문에 현대의 기본 선택입니다. MLA는 K/V를 잠재 공간으로 압축한 후 계산 시점에 다시 투영하는 방식으로 더 나아가며, FLOPs가 증가하지만 메모리를 훨씬 더 많이 절약합니다.

```figure
multihead-split
```

## 구현하기

### 1단계: 이미 있는 단일 헤드 어텐션에서 헤드를 분리하기

NN강의 `SelfAttention`를 가져와서 분리/연결 쌍으로 감싸 보세요. numpy 구현은 `code/main.py`를 참고하세요. 로직은 다음과 같습니다:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

리셰이프 한 번과 트랜스포즈 한 번으로 충분합니다. 루프가 없습니다. 이는 PyTorch가 `nn.MultiheadAttention` 내부에서 정확히 수행하는 작업입니다.

### 2단계: 헤드별로 스케일된 점곱 어텐션 실행하기

각 헤드는 Q, K, V의 자체 슬라이스를 받습니다. 어텐션은 배치된 행렬 곱이 됩니다:

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

실제 하드웨어에서는 `Qh @ Kh.transpose(...)`가 하나의 `bmm`입니다. GPU는 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 형태의 단일 배치된 행렬 곱을 봅니다. 헤드를 추가하는 것은 무료입니다.

### 3단계: 그룹 쿼리 어텐션 변형

키와 값 투영만 변경됩니다. Q는 `n_heads` 그룹을 가지며, K와 V는 `n_kv_heads < n_heads` 그룹을 가지고 매칭되도록 반복됩니다:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

추론 시에는 KV 캐시에 `n_kv_heads`개의 사본만 존재하고 `n_heads`개는 존재하지 않으므로 메모리를 절약합니다. Llama 3 70B는 64개의 쿼리 헤드와 8개의 KV 헤드를 사용하며, 이는 캐시를 8배로 줄입니다.

### 4단계: 각 헤드가 학습한 내용 탐지하기

4개의 헤드를 가진 MHA를 짧은 문장에 대해 실행하세요. 각 헤드에 대해 `(N, N)` 어텐션 행렬을 출력해 보세요. 랜덤 초기화에서도 서로 다른 헤드가 서로 다른 구조를 선택하는 것을 볼 수 있습니다. 이는 부분적으로 신호이며, 부분적으로 하위 공간의 회전 대칭성 때문입니다.

## 사용하기

PyTorch에서는 한 줄 버전이 다음과 같습니다:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+에서의 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention은 CUDA에서 Flash Attention을 자동으로 디스패치합니다.
# GQA의 경우, (B, n_heads, N, d_head) 형태의 Q와
# (B, n_kv_heads, N, d_head) 형태의 K, V를 전달하세요. PyTorch가 반복 처리를 담당합니다.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**헤드 수는 몇 개?** 2026년 생산 모델에서의 경험칙:

| 모델 크기 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`는 거의 항상 64 또는 128이 됩니다. 이는 하나의 헤드가 "볼 수 있는" 단위입니다. 32 미만으로 떨어지면 헤드들이 스케일링 팩터 `sqrt(d_head)`와 싸우기 시작하고; 256 이상으로 올리면 "많은 작은 전문가"의 이점을 잃게 됩니다.

## 출시하기

`outputs/skill-mha-configurator.md`을 참고하세요. 이 스킬은 매개변수 예산, 시퀀스 길이, 배포 대상을 고려하여 새로운 트랜스포머에 대한 헤드 수, kv-head 수, 투영 전략을 권장합니다.

## 연습 문제

1. **쉬움.** `code/main.py`의 MHA를 가져와 `n_heads`을 1에서 16으로 변경하고 `d_model=64`는 고정하세요. 합성 복사 작업에서 작은 단일 레이어 모델의 손실을 플롯하세요. 더 많은 헤드가 도움이 되는지, 정체되는지, 해가 되는지 확인해 보세요.
2. **중간.** MQA (모든 쿼리 헤드가 하나의 KV 헤드를 공유)를 구현하세요. 전체 MHA와 비교하여 매개변수 수가 얼마나 감소하는지 측정하세요. 추론 시 N=2048에서 KV 캐시 크기가 얼마나 줄어드는지 계산하세요.
3. **어려움.** Multi-head Latent Attention의 작은 버전을 구현하세요: K, V를 랭크 `r`의 잠재(latent) 공간으로 압축하고, 잠재 공간을 KV 캐시에 저장하며, 어텐션 시점에 압축을 해제하세요. `r`이 얼마일 때 캐시 메모리가 전체 MHA의 1/8 미만으로 떨어지면서 검증 ppl이 1 비트 이내로 유지됩니까?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Head | "단일 어텐션 회로" | 자체 어텐션 매트릭스를 가진 `d_head = d_model / n_heads` 차원의 Q/K/V 투영. |
| d_head | "헤드 차원" | 헤드별 숨겨진 너비; 생산 환경에서는 거의 항상 64 또는 128. |
| Split / combine | "리셰이프 트릭" | 어텐션 주위의 `(N, d_model) ↔ (n_heads, N, d_head)` 리셰이프 + 트랜스포즈. |
| W_o | "출력 투영" | 헤드를 연결한 후 적용하는 `(d_model, d_model)` 행렬; 헤드가 혼합되는 지점. |
| MQA | "하나의 KV 헤드" | 다중 쿼리 어텐션(Multi-Query Attention): 단일 공유 K/V 투영. 가장 작은 KV 캐시, 일부 품질 손실. |
| GQA | "Llama 2 이후의 기본값" | `n_kv_heads < n_heads`을 사용하는 그룹 쿼리 어텐션(Grouped-Query Attention); Q와 일치하도록 반복. |
| MLA | "DeepSeek의 트릭" | 다중 헤드 잠재 어텐션(Multi-head Latent Attention): K,V를 저랭크 잠재 공간으로 압축하고 어텐션 시에 복원. |
| Induction head | "인컨텍스트 학습 뒤의 회로" | 이전 발생을 감지하고 그 뒤를 따르는 내용을 복사하는 헤드 쌍. |

## 추가 읽기

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 원본 다중 헤드 사양.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 논문.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 학습 후 MHA를 GQA로 변환하는 방법.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 및 캐시 메모리에서 MHA/GQA를 능가하는 이유.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 헤드가 실제로 수행하는 작업에 대한 기계론적 분석.
