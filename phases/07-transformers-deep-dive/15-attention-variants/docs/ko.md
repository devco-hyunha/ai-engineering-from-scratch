# Attention 변형 모델 — 슬라이딩 윈도우, 희소, 차분 (Attention Variants — Sliding Window, Sparse, Differential)

> 전체 어텐션(Full attention)은 원형입니다. 모든 토큰이 모든 토큰을 바라보며, 그 대가로 막대한 메모리를 소모합니다. 네 가지 변형 모델은 이 원의 형태를 구부려 비용을 획기적으로 절감합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 문제점 (The Problem)

전체 어텐션(Full attention)은 시퀀스 길이 $N$에 대해 `O(N²)`의 메모리와 `O(N²)`의 연산 비용이 발생합니다. 128K 컨텍스트를 가진 Llama 3 70B 모델의 경우, 레이어당 160억 개의 어텐션 엔트리가 존재하며, 이는 80개 레이어에 걸쳐 적용됩니다. Flash Attention(12강)은 `O(N²)`의 활성화(activation) 메모리 사용량은 숨겨주지만, 산술 연산 비용 자체를 바꾸지는 않습니다. 즉, 모든 토큰이 여전히 다른 모든 토큰을 참조합니다.

어텐션 행렬의 토폴로지(topology) 자체를 변경하는 세 가지 변형 방식이 있습니다:

1. **슬라이딩 윈도우 어텐션 (Sliding window attention, SWA).** 각 토큰은 전체 접두사(prefix)가 아닌 고정된 크기의 이웃 윈도우만을 참조합니다. 메모리와 연산 비용은 `O(N · W)`로 감소하며, 여기서 $W$는 윈도우 크기입니다. Gemma 2/3, Mistral 7B의 초기 레이어, Phi-3-Long 등이 이 방식을 사용합니다.
2. **희소 / 블록 어텐션 (Sparse / block attention).** 선택된 특정 쌍 `(i, j)`에 대해서만 점수를 계산하며, 나머지는 가중치가 0이 되도록 강제합니다. Longformer, BigBird, OpenAI sparse transformer가 이에 해당합니다.
3. **차분 어텐션 (Differential attention).** 별도의 Q/K 투영(projection)을 통해 두 개의 어텐션 맵을 계산한 뒤, 한 쪽에서 다른 쪽을 뺍니다. 이는 초기 몇 개의 토큰으로 가중치가 쏠리는 "어텐션 싱크(attention sink)" 현상을 제거합니다. Microsoft의 DIFF Transformer (2024)가 대표적입니다.

이 방식들은 공존합니다. 2026년의 프런티어 모델은 종종 이들을 혼합하여 사용합니다. 예를 들어, 대부분의 레이어는 SWA-1024를 사용하고, 매 다섯 번째 레이어는 글로벌 전체 어텐션(global full attention)을 사용하며, 소수의 레이어는 정보 검색을 정제하는 차분 헤드(differential heads)를 사용합니다. Gemma 3의 5:1 SWA 대 글로벌 어텐션 비율은 현재 교과서적인 기본 설정입니다.

## 개념 (The Concept)

### 슬라이딩 윈도우 어텐션 (Sliding Window Attention, SWA)

위치 `i`에 있는 각 쿼리(query)는 `[i - W, i]` 범위(인과적 SWA, causal SWA) 또는 `[i - W/2, i + W/2]` 범위(양방향, bidirectional) 내의 위치에만 어텐션을 수행합니다. 윈도우 외부의 토큰은 스코어 행렬(score matrix)에서 `-inf` 값을 가집니다.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192`이고 `W = 1024`인 경우, 스코어 행렬은 기대치상 1024 × 8192개의 0이 아닌 행을 가집니다. 이는 8배의 감소를 의미합니다.

**SWA를 사용하면 KV 캐시(KV cache)가 축소됩니다.** 레이어당 K와 V의 마지막 `W`개 토큰만 유지하면 됩니다. Gemma-3 스타일의 설정(윈도우 1024, 컨텍스트 128K)의 경우, KV 캐시가 128배 감소합니다.

**품질 비용(Quality cost).** SWA만 사용하는 트랜스포머는 장거리 검색(long-range retrieval)에 어려움을 겪습니다. 해결책은 SWA 레이어와 전체 어텐션(full-attention) 레이어를 교차 배치하는 것입니다. Gemma 3는 SWA와 글로벌 어텐션의 비율을 5:1로 사용합니다. Mistral 7B는 정보가 중첩된 윈도우를 통해 "앞으로 흐르는(flows forward)" 인과적 SWA 스택을 사용했습니다. 즉, 각 레이어는 유효 수용 영역(effective receptive field)을 `W`만큼 확장하며, `L`개의 레이어를 거친 후 모델은 `L × W`개의 토큰까지 과거를 참조할 수 있습니다.

### 희소 / 블록 어텐션 (Sparse / Block Attention)

사전에 `N × N` 희소 패턴(sparsity pattern)을 선택합니다. 세 가지 대표적인 형태는 다음과 같습니다:

- **로컬 + 스트라이드 (Local + strided, OpenAI sparse transformer).** 마지막 `W`개의 토큰과 그 이전의 매 `stride`번째 토큰에 어텐션을 수행합니다. `O(N · sqrt(N))`의 연산량으로 로컬 정보와 장거리 정보를 모두 포착합니다.
- **Longformer / BigBird.** 로컬 윈도우(Local window)와 소수의 글로벌 토큰(예: `[CLS]`)을 사용합니다. 글로벌 토큰은 모든 토큰을 참조하며 모든 토큰으로부터 참조를 받습니다. 여기에 무작위 희소 연결(random-sparse links)을 추가합니다. 동일한 품질에서 컨텍스트 길이를 2배로 확장할 수 있습니다.
- **네이티브 희소 어텐션 (Native Sparse Attention, DeepSeek, 2025).** 어떤 `(Q, K)` 블록이 중요한지 학습하며, 커널 레벨에서 값이 0인 블록을 건너뜁니다. FlashAttention과 호환됩니다.

희소 어텐션은 커널 엔지니어링(kernel-engineering)의 영역입니다. 수학적 원리는 간단하지만(점수 행렬에 마스크 적용), 핵심적인 이점은 0인 항목을 SRAM에 로드하지 않는다는 점에 있습니다. FlashAttention-3와 2026년 출시될 FlexAttention API는 PyTorch에서 커스텀 희소 패턴을 일급 객체(first-class)로 지원하게 만듭니다.

### 차분 어텐션 (Differential Attention, DIFF Transformer, 2024)

일반적인 어텐션(Regular attention)에는 "어텐션 싱크(attention sink)" 문제가 있습니다. softmax는 모든 행의 합이 1이 되도록 강제하기 때문에, 특별히 주목할 대상이 없는 토큰들이 첫 번째 토큰(또는 처음 몇 개의 토큰)에 가중치를 쏟아붓게 됩니다. 이는 실제 콘텐츠에 할당되어야 할 용량을 가로채는 결과를 초래합니다.

차분 어텐션(Differential attention)은 **두 개**의 어텐션 맵을 계산하여 그 차이를 구함으로써 이 문제를 해결합니다:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

여기서 `λ`는 학습 가능한 스칼라 값(일반적으로 0.5–0.8)입니다. `A1`은 실제 콘텐츠 가중치를 포착하고, `A2`는 싱크(sink)를 포착합니다. 뺄셈을 통해 싱크를 상쇄하고, 가중치를 관련 있는 토큰으로 재할당합니다.

보고된 결과(Microsoft 2024): 퍼플렉서티(perplexity) 5–10% 감소, 동일한 학습 길이에서 1.5–2배 더 긴 유효 컨텍스트(effective context) 확보, 더 정교한 니들 인 어 헤이스택(needle-in-a-haystack) 검색 성능을 보여주었습니다.

### 변형 비교 (Variant Comparison)

| 변형 (Variant) | 연산량 (Compute) | KV 캐시 (KV cache) | 전체 대비 품질 (Quality vs full) | 실제 적용 사례 (Production use) |
|---------|---------|----------|-----------------|----------------|
| Full attention | $O(N^2)$ | 레이어당 $O(N)$ | 기준점 (baseline) | 모든 모델의 기본 레이어 |
| SWA (윈도우 1024) | $O(N \cdot W)$ | 레이어당 $O(W)$ | -0.1 ppl, 글로벌 레이어와 조합 시 양호 | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | $O(N \cdot \sqrt{N})$ | 혼합형 (mixed) | SWA와 유사 | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | 약 $O(N)$ | 혼합형 (mixed) | 2배 컨텍스트에서 Full attention과 일치 | 초기 롱 컨텍스트(long-context) BERT |
| Native Sparse (DeepSeek-V3.2) | $O(N \cdot \text{active fraction})$ | $O(N)$ | 0.05 ppl 이내 | DeepSeek-V3.2, 2025 |
| Differential | $O(2 \cdot N^2)$ | $O(2N)$ | -5 ~ -10% ppl | DIFF Transformer, 2026년 초 모델 |

```figure
gqa-kv-sharing
```

## 직접 구현해 보기 (Build It)

`code/main.py`를 확인해 보세요. 간단한 시퀀스(toy sequence)를 사용하여 전체(full), SWA, local+strided, 그리고 차분 어텐션(differential attention)을 나란히 비교하여 보여주는 인과적 마스크 비교기(causal mask comparator)를 구현했습니다.

### 1단계: 전체 인과적 마스크 (full causal mask, baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07의 베이스라인입니다. 하삼각 행렬(Lower triangular) 형태이며, 대각선 위쪽의 가중치는 0입니다.

### 2단계: 슬라이딩 윈도우 인과적 마스크 (sliding window causal mask)

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

매개변수는 `window` 하나입니다. `window >= n`인 경우, 전체 인과적 어텐션(full causal attention)을 복구합니다. `window = 1`인 경우, 각 토큰은 자기 자신에게만 어텐션을 수행합니다.

### 3단계: 로컬 + 스트라이드 희소 마스크 (local + strided sparse mask)

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

밀집된 로컬 윈도우(dense local window)와 시퀀스 시작점부터 매 `stride`번째 토큰을 포함합니다. 추가 레이어를 거칠수록 수용 영역(receptive field)이 로그 단계로 성장합니다.

### 4단계: 차분 어텐션 (Differential Attention)

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

두 번의 어텐션 패스를 수행한 뒤, 학습 가능한 혼합 계수(mixing coefficient)를 사용하여 차이를 구합니다. 코드에서는 단일 어텐션과 차분 어텐션의 어텐션 싱크(attention-sink) 히트맵을 비교하며 싱크가 붕괴(collapse)되는 과정을 관찰합니다.

### 5단계: KV 캐시 크기 (KV cache sizes)

각 변형 모델(variant)에 대해 `N = 131072`일 때 레이어당 캐시 크기를 출력해 보세요. SWA 및 sparse 변형 모델은 크기가 10~100배 감소합니다. Differential 모델은 두 배로 증가합니다. 메모리 비용을 의식적으로 관리하세요.

## 사용 방법 (Use It)

2026년 프로덕션 패턴:

```python
from transformers import AutoModelForCausalLM
# Gemma 3는 SWA (window=1024)와 global 레이어를 5:1 비율로 혼합합니다.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+의 FlexAttention은 마스크 함수를 지원합니다:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

이 코드는 커스텀 Triton 커널로 컴파일됩니다. 일반적인 패턴에서 FlashAttention-3 속도의 10% 이내 성능을 보여주며, 마스크 함수는 Python 호출 가능 객체(callable)입니다.

**각 방식의 선택 기준:**

- **순수 전체 어텐션 (Pure full attention)** — 약 16K 컨텍스트까지의 모든 레이어, 또는 검색(retrieval) 품질이 가장 중요할 때 선택합니다.
- **SWA + global 혼합 (SWA + global mix)** — 긴 컨텍스트(>32K), 학습 및 추론 시 메모리 대역폭이 제한적일 때 선택합니다. 32K 이상의 컨텍스트에서는 위에서 언급한 방식이 2026년의 기본값이 될 것입니다.
- **희소 블록 어텐션 (Sparse block attention)** — 커스텀 커널이나 커스텀 패턴이 필요할 때 선택합니다. 특수 워크로드(검색, 오디오)를 위해 예약되어 있습니다.
- **차분 어텐션 (Differential attention)** — 어텐션 싱크 오염(attention-sink contamination)이 성능을 저해하는 모든 워크로드(긴 컨텍스트 RAG, needle-in-a-haystack)에서 선택합니다.

## Ship It (실행하기)

`outputs/skill-attention-variant-picker.md`를 참조하세요. 이 스킬은 목표 컨텍스트 길이(target context length), 검색 요구 사항(retrieval demands), 그리고 학습/추론 컴퓨팅 프로필(training/inference compute profile)이 주어졌을 때 새로운 모델을 위한 어텐션 토폴로지(attention topology)를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. `window=4`일 때 SWA가 각 행의 마지막 4개 토큰 이외의 모든 값을 0으로 만드는지 확인하세요. `window=n`일 때 전체 인과적 어텐션(causal attention)과 비트 단위로 동일하게 재현되는지 확인하세요.
2. **중간 (Medium).** Lesson 07 캡스톤 프로젝트를 바탕으로 `window=1024`인 인과적 SWA(causal SWA)를 구현해 보세요. tinyshakespeare 데이터셋으로 1,000 스텝 동안 학습시켜 보세요. 전체 어텐션(full attention)과 비교했을 때 검증 손실(val loss)이 얼마나 퇴보(regress)하나요? 피크 메모리(peak memory)는 얼마나 감소하나요?
3. **어려움 (Hard).** 캡스톤 모델에 Gemma-3 스타일의 5:1 레이어 혼합(5개의 SWA, 1개의 global)을 구현해 보세요. 파라미터 수를 동일하게 맞춘 상태에서 순수 SWA(pure-SWA) 및 순수 global(pure-global) 베이스라인과 손실, 메모리, 생성 품질을 비교해 보세요.
4. **어려움 (Hard).** 헤드당 학습 가능한 `λ`를 사용하는 차분 어텐션(differential attention)을 구현해 보세요. 합성 검색 작업(하나의 바늘, 2,000개의 방해 요소)으로 학습시켜 보세요. 파라미터 수를 동일하게 맞춘 상태에서 단일 어텐션(single-attention) 베이스라인 대비 검색 정확도를 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 슬라이딩 윈도우 어텐션 (Sliding window attention, SWA) | "로컬 어텐션(Local attention)" | 각 쿼리가 마지막 `W`개의 토큰에만 어텐션을 수행합니다. KV 캐시가 `O(W)`로 축소됩니다. |
| 유효 수용 영역 (Effective receptive field) | "모델이 얼마나 멀리까지 보는지" | 윈도우 크기가 `W`인 `L`계층 SWA 스택의 경우, 최대 `L × W`개의 토큰까지 볼 수 있습니다. |
| Longformer / BigBird | "로컬 + 글로벌 + 랜덤" | 소수의 항상 어텐션되는 글로벌 토큰을 포함하는 희소 패턴(Sparse patterns)을 사용하는 초기 롱 컨텍스트 접근 방식입니다. |
| 네이티브 희소 어텐션 (Native Sparse Attention) | "DeepSeek의 커널 트릭" | 블록 단위의 희소성(Sparsity)을 학습합니다. 품질을 유지하면서 커널 수준에서 값이 0인 블록을 건너뜁니다. |
| 차분 어텐션 (Differential attention) | "두 개의 맵, 하나를 뺌" | DIFF Transformer: 첫 번째 어텐션 맵에서 학습된 `λ`배만큼의 두 번째 어텐션 맵을 빼서 어텐션 싱크(attention sinks)를 상쇄합니다. |
| 어텐션 싱크 (Attention sink) | "가중치가 0번 토큰으로 쏠림" | Softmax 정규화로 인해 행의 합이 1이 되어야 하므로, 정보가 없는 쿼리가 0번 위치에 가중치를 쏟아붓는 현상입니다. |
| FlexAttention | "Mask-as-Python" | 임의의 마스크 함수를 FlashAttention 형태의 커널로 컴파일하는 PyTorch 2.5+ API입니다. |
| 레이어 유형 혼합 (Layer type mix) | "SWA 대 글로벌 비율 5:1" | 메모리 사용량을 낮추면서 품질을 유지하기 위해 스택 내에 희소 어텐션과 전체 어텐션 레이어를 교차 배치합니다. |

## 추가 학습 자료 (Further Reading)

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 슬라이딩 윈도우(sliding-window)와 글로벌 토큰(global-token) 방식을 다룬 표준적인 논문입니다.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 로컬(local) + 글로벌(global) + 랜덤(random) 방식을 제안합니다.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI의 로컬(local) + 스트라이드(strided) 패턴을 다룹니다.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 비율의 SWA:global 혼합 방식을 보여줍니다.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 현재 교과서적인 기본값으로 사용되는 window=1024 기반의 5:1 혼합 방식을 다룹니다.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 논문입니다.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2의 학습된 희소성 어텐션(learned-sparsity attention)을 다룹니다.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 'Use It' 섹션의 mask-as-callable 패턴에 대한 API 레퍼런스입니다.
