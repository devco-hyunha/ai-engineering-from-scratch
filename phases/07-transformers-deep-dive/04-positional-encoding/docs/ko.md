# 위치 인코딩 (Positional Encoding) — Sinusoidal, RoPE, ALiBi

> 어텐션(Attention) 메커니즘은 순열 불변(permutation-invariant)적입니다. 위치 신호가 없다면 "The cat sat on the mat"과 "mat the on sat cat the"는 동일한 출력을 생성합니다. 세 가지 알고리즘이 이 문제를 해결하며, 각 알고리즘은 "위치"가 무엇을 의미하는지에 대해 서로 다른 가설을 세웁니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## 문제점 (The Problem)

스케일드 닷 프로덕트 어텐션(Scaled dot-product attention)은 순서에 무관합니다. 어텐션 행렬 `softmax(Q K^T / √d) V`는 쌍별 유사도(pairwise similarities)를 통해 계산됩니다. `X`의 행을 섞으면, 출력의 행도 동일한 방식으로 섞이게 됩니다. 어텐션 내부의 그 어떤 요소도 위치(position)를 고려하지 않습니다.

이는 단어 가방(bag-of-words) 모델에서는 버그가 아닙니다. 하지만 언어, 코드, 오디오, 비디오와 같이 순서가 의미를 갖는 모든 대상에게 이는 치명적입니다.

해결책은 임베딩에 어떤 방식으로든 위치 정보를 주입하는 것입니다. 세 가지 시대별 답변은 다음과 같습니다:

1. **절대적 사인 함수 방식 (Absolute sinusoidal)** (Vaswani 2017). 임베딩에 위치의 `sin/cos` 값을 더합니다. 단순하고 학습이 필요 없지만, 학습된 길이를 벗어나는 외삽(extrapolation) 능력이 떨어집니다.
2. **RoPE — 회전 위치 임베딩 (Rotary Position Embeddings)** (Su 2021). `Q`와 `K` 벡터를 위치에 비례하는 각도로 회전시킵니다. 닷 프로덕트 내에 *상대적* 위치를 직접 인코딩합니다. 2026년 현재 주류 기술입니다.
3. **ALiBi — 선형 편향 어텐션 (Attention with Linear Biases)** (Press 2022). 임베딩을 완전히 생략합니다. 대신 거리에 따라 어텐션 점수에 헤드별 선형 페널티(linear penalty)를 추가합니다. 길이 외삽 능력이 매우 뛰어납니다.

2026년 현재, 사실상 거의 모든 프런티어 오픈 모델이 RoPE를 사용합니다: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. 소수의 롱 컨텍스트(long-context) 모델들이 ALiBi 또는 그 현대적 변형을 사용합니다. 절대적 사인 함수 방식은 이제 역사 속의 기술이 되었습니다.

## 개념 (The Concept)

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 절대 사인파 (Absolute sinusoidal)

`(max_len, d_model)` 형상의 고정된 행렬 `PE`를 미리 계산합니다:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

그 다음 어텐션(attention)을 수행하기 전에 `X' = X + PE[:N]`을 적용합니다. 각 차원은 서로 다른 주파수를 가진 사인파(sinusoid)입니다. 모델은 위상 패턴(phase pattern)으로부터 위치를 읽는 법을 학습합니다. `max_len`을 벗어나면 실패합니다. 모델이 0~2047 위치만 보았다면, 2048 위치에서 어떤 일이 일어나는지에 대해 아무런 정보도 전달받지 못했기 때문입니다.

### RoPE (Rotary Positional Embedding)

Q와 K 벡터(임베딩이 아님)를 회전시킵니다. 차원 쌍 `(2i, 2i+1)`에 대하여:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 기본값 10000
```

위치 `pos_k`를 가진 키(key)에도 동일한 회전을 적용합니다. 내적(dot product) `q'_m · k'_n`은 오직 `(m - n)`만의 함수가 됩니다. 즉, **회전은 절대적 위치를 기준으로 수행되었음에도 불구하고, 어텐션 점수(attention score)는 오직 상대적 거리(relative distance)에만 의존하게 됩니다.** 매우 아름다운 기법입니다.

RoPE의 확장: `base`를 스케일링(NTK-aware, YaRN, LongRoPE 방식)함으로써 재학습 없이 더 긴 컨텍스트로 외삽(extrapolate)할 수 있습니다. Llama 3는 이러한 방식으로 컨텍스트를 8K에서 128K로 확장했습니다.

### ALiBi

임베딩 트릭을 건너뛰고, 어텐션 점수(attention scores)에 직접 편향(bias)을 부여합니다:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

여기서 `m_h`는 헤드별 기울기(head-specific slope)입니다 (예: `1 / 2^(8·h/H)`). 가까운 토큰은 점수가 높아지고, 먼 토큰은 페널티를 받습니다. 학습 시간 비용은 발생하지 않습니다. 논문에 따르면, ALiBi의 길이 외삽(length extrapolation) 성능은 사인파(sinusoidal) 방식보다 뛰어나며, 원래 학습된 길이에서는 RoPE와 대등한 성능을 보여줍니다.

### 2026년에 선택할 것 (What to pick in 2026)

| 변형 (Variant) | 외삽 능력 (Extrapolation) | 학습 비용 (Training cost) | 사용 사례 (Used by) |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | 낮음 (poor) | 무료 (free) | original transformer, 초기 BERT |
| Learned absolute | 없음 (none) | 매우 낮음 (tiny) | GPT-2, GPT-3 |
| RoPE | 스케일링 시 좋음 (good with scaling) | 무료 (free) | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 매우 우수 (excellent) | 미세 조정 단계 (fine-tune stage) | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 매우 우수 (excellent) | 무료 (free) | BLOOM, MPT, Baichuan |

RoPE가 승리한 이유는 아키텍처를 변경하지 않고도 어텐션(attention) 메커니즘에 자연스럽게 삽입될 수 있고, 상대적 위치를 인코딩하며, `base` 하이퍼파라미터를 통해 긴 문맥 미세 조정(long-context fine-tuning)을 위한 깔끔한 조절 장치를 제공하기 때문입니다.

```figure
rope-explorer
```

## 직접 구현해 보기 (Build It)

### 1단계: 사인 함수 인코딩 (sinusoidal encoding)

`code/main.py`를 참조하세요. 다음은 4줄로 구성된 계산식입니다:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

첫 번째 어텐션 레이어(attention layer) 이전에 이 값을 임베딩 행렬(embedding matrix)에 더해 주세요.

### 2단계: Q와 K에 적용되는 RoPE

RoPE는 `Q`와 `K`에 제자리(in-place) 방식으로 동작합니다. 각 차원 쌍에 대해 다음과 같이 수행됩니다:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

중요한 점은: 위치 `m`에 있는 `Q`와 위치 `n`에 있는 `K`에 동일한 함수를 적용해야 한다는 것입니다. 이들의 내적(dot product)은 모든 좌표 쌍에서 `cos((m-n)·θ_i)` 인자를 추출하게 됩니다. 이를 통해 어텐션(Attention)은 별도의 학습 없이도 상대적 위치를 파악할 수 있습니다.

### 3단계: ALiBi 기울기(slopes) 및 편향(bias)

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # softmax 적용 전 attention score에 더해줍니다
```

`h`번째 헤드의 `(seq_len, seq_len)` 어텐션 스코어(attention score) 행렬에 `bias[h]`를 더한 후, softmax를 적용해 보세요.

### 4단계: RoPE의 상대적 거리 속성(relative-distance property) 검증

두 개의 무작위 벡터 `a, b`를 선택합니다. 먼저 `(pos_a, pos_b)`만큼 회전시킨 후, 다시 `(pos_a + k, pos_b + k)`만큼 회전시켜 보세요. 두 경우의 내적(dot product) 값은 부동 소수점 오차 범위 내에서 반드시 일치해야 합니다. 이 속성이 바로 RoPE의 핵심입니다. 즉, 절대적인 오프셋(absolute offset)에는 불변하며, 오직 상대적인 간격(relative gap)만이 중요하게 작용합니다.

## 사용 방법 (Use It)

PyTorch 2.5+ 버전부터는 `torch.nn.functional`에 RoPE 유틸리티가 포함되어 있습니다. 대부분의 프로덕션 코드에서는 RoPE가 어텐션 커널 내부에서 적용되는 `flash_attn` 또는 `xformers`를 사용합니다.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026년 기준 롱 컨텍스트(Long-context) 트릭:**

- **NTK-aware interpolation (NTK 인식 보간).** 4K에서 16K 이상으로 확장할 때 `base` 값을 `base * (scale_factor)^(d/(d-2))`로 재조정합니다.
- **YaRN.** 긴 컨텍스트에서 어텐션 엔트로피(attention entropy)를 보존하는 더 스마트한 보간 방식입니다. Llama 3.1 128K에서 이를 사용합니다.
- **LongRoPE.** 진화적 탐색(evolutionary search)을 사용하여 차원별 스케일 인자를 선택하는 Microsoft의 2024년 방식입니다. Phi-3-Long에서 이를 사용합니다.
- **Position interpolation + fine-tuning (위치 보간 + 미세 조정).** 위치를 확장 인자만큼 축소한 뒤 1~5B 토큰에 대해 미세 조정을 수행합니다. 놀라울 정도로 효과적입니다.

## Ship It (실행해 보기)

`outputs/skill-positional-encoding-picker.md`를 참조하세요. 이 기술(skill)은 목표 컨텍스트 길이, 외삽(extrapolation) 필요성 및 학습 예산을 고려하여 새로운 모델을 위한 인코딩 전략을 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `max_len=512, d=128` 설정에서 사인파 형태의 `PE` 행렬을 히트맵(heatmap)으로 시각화해 보세요. "차원 인덱스가 커질수록 줄무늬가 넓어지는" 패턴을 확인해 보세요.
2. **중간 (Medium).** NTK-aware RoPE 스케일링(scaling)을 구현해 보세요. 길이 256의 시퀀스로 아주 작은 언어 모델(tiny LM)을 학습시킨 후, 스케일링을 적용했을 때와 적용하지 않았을 때를 비교하며 길이 1024에서 테스트해 보세요. 퍼플렉서티(perplexity)를 측정해 보세요.
3. **어려움 (Hard).** 동일한 어텐션 모듈(attention module) 내에 ALiBi와 RoPE를 모두 구현해 보세요. 길이 512의 시퀀스를 사용하는 복사 작업(copy task)으로 4계층 트랜스포머(4-layer transformer)를 학습시키세요. 테스트 시점에 2048 길이로 외삽(extrapolate)하여 성능 저하를 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| Positional encoding (위치 인코딩) | "어텐션에 순서를 알려줌" | 임베딩이나 어텐션에 추가되어 위치 정보를 인코딩하는 모든 신호. |
| Sinusoidal (사인 함수 방식) | "오리지널 방식" | 임베딩에 기하학적 주파수의 `sin/cos`를 더하는 방식; 외삽(extrapolation)이 되지 않음. |
| RoPE | "회전 임베딩(Rotary embeddings)" | 위치에 따라 결정되는 각도로 `Q`, `K`를 회전시킴; 내적(dot product)을 통해 상대적 거리를 인코딩함. |
| ALiBi | "선형 편향 트릭(Linear bias trick)" | 어텐션 점수에 `-m·\|i-j\|`를 더함; 임베딩이 필요 없으며 외삽 성능이 뛰어남. |
| base | "RoPE의 조절 노브(knob)" | RoPE의 주파수 스케일러; 추론 시 컨텍스트를 확장하려면 이 값을 높임. |
| NTK-aware | "RoPE 스케일링 트릭" | 컨텍스트가 확장될 때 고주파 차원이 압축되지 않도록 `base`를 재조정함. |
| YaRN | "고급 기술" | 어텐션 엔트로피를 보존하면서 차원별로 보간(interpolation) 및 외삽(extrapolation)을 수행함. |
| Extrapolation (외삽) | "학습된 길이를 넘어 작동함" | 위치 체계가 학습 시 보았던 `max_len`을 넘어 올바른 출력을 생성할 수 있는가? |

## 추가 학습 자료 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 오리지널 사인파(sinusoidal) 방식.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 논문.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 최신 RoPE 스케일링 기술.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta의 Llama 2 롱 컨텍스트(long-context) 논문.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long에서 사용되었으며 'Use It' 섹션에서 인용된 Microsoft의 방식.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 모든 RoPE 스케일링 방식(default, linear, dynamic, YaRN, LongRoPE, Llama-3)에 대한 프로덕션급 구현체.
