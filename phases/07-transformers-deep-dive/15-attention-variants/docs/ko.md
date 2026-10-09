# 어텐션 변형 — 슬라이딩 윈도우, 희소, 차분 어텐션

> 풀 어텐션은 원형입니다. 모든 토큰이 모든 토큰을 보며, 메모리가 그 비용을 지불합니다. 네 가지 변형이 원의 형태를 변형하여 비용의 절반을 회복합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션), 7단계 · 03강 (멀티 헤드), 7단계 · 12강 (KV 캐시 / Flash Attention)
**시간:** 약 60분

## 문제점

풀 어텐션은 시퀀스 길이에 대해 `O(N²)` 메모리와 `O(N²)` 연산 비용을 요구합니다. 128K 컨텍스트의 Llama 3 70B의 경우, 레이어당 160억 개의 어텐션 항목이 있으며, 이는 80개 레이어에 곱해집니다. Flash Attention (12강)은 `O(N²)` 활성화 메모리를 숨기지만 연산 비용을 변경하지는 않습니다. 모든 토큰은 여전히 다른 모든 토큰에 어텐션합니다.

세 가지 변형 클래스가 어텐션 매트릭스 자체의 위상을 변경합니다:

1. **슬라이딩 윈도우 어텐션 (SWA).** 각 토큰은 전체 접두어가 아닌 이웃의 고정된 윈도우에 어텐션합니다. 메모리와 연산이 `O(N · W)`로 감소하며, 여기서 `W`는 윈도우입니다. Gemma 2/3, Mistral 7B의 첫 레이어, Phi-3-Long.
2. **희소 / 블록 어텐션.** 선택된 쌍 `(i, j)`만 점수를 받으며, 나머지는 가중치가 0으로 강제됩니다. Longformer, BigBird, OpenAI 희소 트랜스포머.
3. **차분 어텐션.** 분리된 Q/K 투영으로 두 개의 어텐션 맵을 계산하고, 하나를 다른 하나에서 뺍니다. 첫 몇 개 토큰에 가중치를 흘려보내는 "어텐션 싱크"를 제거합니다. Microsoft의 DIFF Transformer (2024).

이들은 공존합니다. 2026년 프론티어 모델은 종종 이들을 혼합합니다. 대부분의 레이어는 SWA-1024이며, 5번째마다 전역 풀 어텐션이고, 몇 개는 검색을 정리하는 차분 헤드입니다. Gemma 3의 5:1 SWA 대 전역 비율은 현재 교과서 기본값입니다.

## 개념

### 슬라이딩 윈도우 어텐션 (SWA)

`i` 위치의 각 쿼리는 `[i - W, i]` (인과 SWA) 또는 `[i - W/2, i + W/2]` (양방향) 내의 위치에만 어텐션합니다. 윈도우 밖의 토큰은 점수 매트릭스에서 `-inf`를 받습니다.

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

`N = 8192` 및 `W = 1024`의 경우, 점수 매트릭스는 기대값에서 1024 × 8192개의 비영(non-zero) 행을 가지며, 이는 8배 감소입니다.

**SWA로 KV 캐시가 축소됩니다.** 각 레이어에서 K와 V의 마지막 `W` 토큰만 유지하면 됩니다. Gemma-3 유사 구성(1024 윈도우, 128K 컨텍스트)에서는 KV 캐시가 128배 감소합니다.

**품질 비용.** SWA 전용 트랜스포머는 장거리 검색에 어려움을 겪습니다. 해결책: SWA 레이어를 전체 어텐션 레이어와 교대로 배치하는 것입니다. Gemma 3은 5:1 SWA:전체 어텐션 비율을 사용합니다. Mistral 7B는 겹치는 윈도우를 통해 정보가 "앞으로 흐르는" 인과적 SWA 스택을 사용했습니다. 각 레이어는 유효 수용 영역을 `W`만큼 확장하며, `L` 레이어 이후 모델은 `L × W` 토큰까지 거슬러 어텐션할 수 있습니다.

### 희소 / 블록 어텐션

미리 `N × N` 희소 패턴을 선택하세요. 세 가지 표준 형태가 있습니다:

- **지역 + 스트라이드 (OpenAI 희소 트랜스포머).** 마지막 `W` 토큰과 그 이전의 모든 `stride`번째 토큰에 어텐션합니다. `O(N · sqrt(N))` 연산으로 지역 및 장거리 패턴을 모두 포착합니다.
- **Longformer / BigBird.** 지역 윈도우 + 모든 토큰이 어텐션하고 모든 토큰이 어텐션하는 소량의 전역 토큰(예: `[CLS]`) + 랜덤 희소 링크. 동일한 품질에서 경험적으로 2배 컨텍스트를 달성합니다.
- **네이티브 희소 어텐션 (DeepSeek, 2025).** `(Q, K)` 블록 중 어떤 것이 중요한지 학습하며, 커널 수준에서 제로 블록을 건너뜁니다. FlashAttention 호환입니다.

희소 어텐션은 커널 엔지니어링의 이야기입니다. 수학은 간단합니다(점수 매트릭스를 마스킹). 승리는 제로 항목을 SRAM에 로드하지 않는 데서 옵니다. FlashAttention-3와 2026 FlexAttention API는 PyTorch에서 커스텀 희소 패턴을 일급 객체로 만듭니다.

### 차분 어텐션 (DIFF 트랜스포머, 2024)

일반 어텐션에는 "어텐션 싱크" 문제가 있습니다: softmax는 모든 행이 1로 합쳐지도록 강제하므로, 특정 토큰에 어텐션하지 않으려는 토큰은 가중치를 첫 번째 토큰(또는 첫 몇 개)에 투하합니다. 이는 실제 콘텐츠에 사용되어야 할 용량을 빼앗습니다.

차분 어텐션은 **두 개의** 어텐션 맵을 계산하고 빼서 이를 해결합니다:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

여기서 `λ`는 학습된 스칼라입니다(일반적으로 0.5–0.8). A1은 실제 콘텐츠 가중치를 포착하고, A2는 싱크를 포착합니다. 뺄셈이 싱크를 상쇄하고, 가중치를 관련 토큰에 재분배합니다.

보고된 결과 (Microsoft 2024): 퍼플렉시티(Perplexity)가 5–10% 낮아짐, 동일한 학습 길이에서 유효 컨텍스트가 1.5–2배 길어짐, 바늘 찾기(haystack retrieval) 검색이 더 날카로워짐.

### 변형 비교

| 변형 | 연산량 | KV 캐시 | 전체 대비 품질 | 프로덕션 사용 |
|---------|---------|----------|-----------------|----------------|
| 전체 어텐션 | O(N²) | 레이어당 O(N) | 기준선 | 모든 모델의 기본 레이어 |
| SWA (윈도우 1024) | O(N·W) | 레이어당 O(W) | -0.1 ppl, 전역 레이어와 함께 사용 시 좋음 | Gemma 2/3, Phi-3-Long |
| 지역 + 스트라이드 희소 | O(N·√N) | 혼합 | SWA와 유사 | OpenAI 희소 트랜스포머, Longformer |
| BigBird (지역 + 전역 + 랜덤) | O(N) 근사 | 혼합 | 2배 컨텍스트에서 전체와 일치 | 초기 장문 컨텍스트 BERT |
| 네이티브 희소 (DeepSeek-V3.2) | O(N · 활성 비율) | O(N) | 0.05 ppl 이내 | DeepSeek-V3.2, 2025 |
| 차등 | O(2·N²) | O(2N) | ppl -5 ~ -10% | DIFF 트랜스포머, 초기 2026 모델 |

```figure
gqa-kv-sharing
```

## 구현하기

`code/main.py`를 참조하세요. 장난감 시퀀스에서 전체, SWA, 지역+스트라이드, 차등 어텐션을 나란히 보여주는 인과 마스크 비교기를 구현합니다.

### 1단계: 전체 인과 마스크 (기준선)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

07강의 기준선. 하단 삼각형; 대각선 위에는 가중치가 0입니다.

### 2단계: 슬라이딩 윈도우 인과 마스크

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

하나의 매개변수 — `window`. `window >= n`이면 전체 인과 어텐션을 복원합니다. `window = 1`이면 각 토큰이 자기 자신만 참조합니다.

### 3단계: 지역 + 스트라이드 희소 마스크

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

밀집 지역 윈도우와 시퀀스 시작부터 매 `stride`번째 토큰까지 포함합니다. 추가 레이어가 쌓이면 수용 영역이 로그 단계로 증가합니다.

### 4단계: 차등 어텐션

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

두 번의 어텐션 패스를 수행하고 학습된 혼합 계수로 차이를 계산합니다. 코드에서는 단일 어텐션과 차등 어텐션의 어텐션 싱크(sink) 히트맵을 비교하며 싱크 붕괴(sink collapse)를 관찰합니다.

### 5단계: KV 캐시 크기

각 변형에 대해 `N = 131072`에서의 레이어별 캐시 크기를 출력합니다. SWA 및 희소 변형은 10–100배 감소합니다. 차등은 2배 증가합니다. 메모리 비용을 의식적으로 관리하세요.

## 사용하기

2026 프로덕션 패턴:

```python
from transformers import AutoModelForCausalLM
# Gemma 3는 SWA (window=1024)와 전역 레이어를 5:1 비율로 혼합합니다.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+의 FlexAttention은 마스크 함수를 허용합니다:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

이것은 커스텀 Triton 커널로 컴파일됩니다. 일반적인 패턴에 대해 FlashAttention-3 속도의 10% 이내이며, 마스크 함수는 Python 호출 가능한 객체입니다.

**각 방식을 선택하는 시점:**

- **순수 전체 어텐션** — ~16K 컨텍스트까지 모든 레이어에 사용하거나, 검색 품질이 최우선일 때.
- **SWA + 전역 혼합** — 긴 컨텍스트 (>32K), 학습 및 추론이 메모리 바운드인 경우. 32K 이상에서의 2026년 기본값입니다.
- **희소 블록 어텐션** — 커스텀 커널, 커스텀 패턴. 특수화된 워크로드 (검색, 오디오)에 예약됩니다.
- **차분 어텐션** — 어텐션 싱크 오염이 해로운 모든 워크로드 (긴 컨텍스트 RAG, haystack 속의 바늘 찾기).

## 출시하기

`outputs/skill-attention-variant-picker.md`를 참조하세요. 이 스킬은 목표 컨텍스트 길이, 검색 요구 사항, 학습/추론 컴퓨팅 프로필을 고려하여 새 모델의 어텐션 토폴로지를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. `window=4`에서 SWA가 각 행의 마지막 4개 토큰을 제외한 모든 것을 0으로 만드는지 확인하세요. `window=n`가 전체 인과적 어텐션을 비트 단위로 동일하게 재현하는지 확인하세요.
2. **중간.** 07강 캡스톤 위에 `window=1024`를 사용하여 인과적 SWA를 구현하세요. tinyshakespeare에서 1,000 스텝 동안 학습하세요. val 손실이 전체 어텐션 대비 얼마나 악화되나요? 피크 메모리가 얼마나 감소하나요?
3. **어려움.** 캡스톤 모델에 Gemma-3 스타일의 5:1 레이어 혼합 (5 SWA, 1 전역)을 구현하세요. 매개변수가 동일한 상태에서 순수 SWA 및 순수 전역 기준선과 손실, 메모리, 생성 품질을 비교하세요.
4. **어려움.** 각 헤드에 학습된 `λ`를 사용하여 차분 어텐션을 구현하세요. 합성 검색 작업 (바늘 하나, 방해물 2,000개)에서 학습하세요. 매개변수가 동일한 상태에서 단일 어텐션 기준선 대비 검색 정확도를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 슬라이딩 윈도우 어텐션 (SWA) | "지역 어텐션" | 각 쿼리가 마지막 `W` 토큰에 어텐션하며; KV 캐시가 `O(W)`로 축소됩니다. |
| 유효 수용 필드 | "모델이 얼마나 멀리까지 보는가" | `L`층 SWA 스택에서 윈도우 `W`를 사용하면 최대 `L × W` 토큰까지 볼 수 있습니다. |
| Longformer / BigBird | "지역 + 전역 + 랜덤" | 항상 어텐션하는 몇 개의 전역 토큰을 포함하는 희소 패턴; 초기 롱 컨텍스트 접근 방식입니다. |
| Native Sparse Attention | "DeepSeek의 커널 트릭" | 블록 단위 희소성을 학습합니다. 품질을 유지하면서 커널 수준에서 제로 블록을 건너뜁니다. |
| Differential attention | "두 개의 맵, 하나가 차감" | DIFF Transformer: 학습된 `λ`을 두 번째 어텐션 맵에 곱하여 첫 번째 맵에서 차감함으로써 어텐션 싱크를 제거합니다. |
| Attention sink | "가중치가 토큰 0으로 흘러감" | Softmax 정규화로 인해 행의 합이 1이 되어야 하므로, 정보량이 낮은 쿼리가 가중치를 위치 0에 집중시킵니다. |
| FlexAttention | "Mask-as-Python" | 임의의 마스크 함수를 FlashAttention 형태 커널로 컴파일하는 PyTorch 2.5+ API입니다. |
| Layer type mix | "5:1 SWA-to-global" | 스택 내에서 희소 어텐션 레이어와 전체 어텐션 레이어를 교대로 배치하여 더 낮은 메모리로 품질을 유지합니다. |

## 추가 읽기

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 표준적인 슬라이딩 윈도우 + 전역 토큰 논문입니다.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 지역 + 전역 + 랜덤 패턴입니다.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI의 지역+스트라이드 패턴입니다.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global 혼합입니다.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — window=1024인 5:1 혼합으로, 현재 교과서 기본값입니다.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 논문입니다.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2의 학습된 희소성 어텐션입니다.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It 섹션의 mask-as-callable 패턴에 대한 API 레퍼런스입니다.
