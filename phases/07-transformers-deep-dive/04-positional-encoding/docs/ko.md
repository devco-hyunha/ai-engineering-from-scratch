# 위치 인코딩 — 사인 함수, RoPE, ALiBi

> 어텐션은 순열 불변입니다. "The cat sat on the mat"과 "mat the on sat cat the"는 위치 신호가 없으면 동일한 출력을 생성합니다. 세 가지 알고리즘이 이를 해결합니다 — 각각 "위치"의 의미에 대해 다른 가정을 합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션), 7단계 · 03강 (멀티 헤드 어텐션)
**시간:** 약 45분

## 문제점

스케일된 점곱 어텐션은 순서에 대해 맹목적입니다. 어텐션 행렬 `softmax(Q K^T / √d) V`는 쌍별 유사성에서 계산됩니다. `X`의 행을 섞으면, 출력의 행도 동일하게 섞입니다. 어텐션 내부에서는 위치에 대해 아무것도 신경 쓰지 않습니다.

이는 단어 주머니(bag-of-words) 모델의 버그가 아닙니다. 언어, 코드, 오디오, 비디오 — 순서가 의미를 담는 모든 것에서는 치명적입니다.

해결책은 임베딩에 위치를 주입하는 것입니다. 세 시대의 답변이 있습니다:

1. **절대 사인 함수** (Vaswani 2017). 임베딩에 위치 `sin/cos`를 더합니다. 단순하고 학습 매개변수가 없으며, 학습된 길이 밖에서는 외삽이 잘 작동하지 않습니다.
2. **RoPE — 회전 위치 임베딩(Rotary Position Embeddings)** (Su 2021). Q와 K 벡터를 위치에 비례하는 각도로 회전시킵니다. 점곱에서 *상대적* 위치를 직접 인코딩합니다. 2026년 현재 지배적입니다.
3. **ALiBi — 선형 편향이 있는 어텐션(Attention with Linear Biases)** (Press 2022). 임베딩을 완전히 생략하고, 거리에 기반하여 어텐션 점수에 헤드별 선형 페널티를 더합니다. 길이 외삽이 뛰어납니다.

2026년 현재, 사실상 모든 프론티어 오픈 모델이 RoPE를 사용합니다: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. 몇몇 긴 컨텍스트 모델은 ALiBi나 그 현대적 변형을 사용합니다. 절대 사인 함수는 역사적입니다.

## 개념

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 절대 사인 함수

`(max_len, d_model)` 형태의 고정 행렬 `PE`를 사전 계산합니다:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

그런 다음 어텐션 전에 `X' = X + PE[:N]`를 적용합니다. 각 차원은 서로 다른 주파수의 사인 함수입니다. 모델은 위상 패턴에서 위치를 읽는 법을 학습합니다. `max_len` 밖에서는 실패합니다: 모델이 위치 0–2047만 보았을 때 위치 2048에서 무슨 일이 일어나는지 알려지지 않았습니다.

### RoPE

Q와 K 벡터를 회전시킵니다 (임베딩이 아닙니다). 차원 쌍 `(2i, 2i+1)`에 대해:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

포지션 `pos_k`인 키에 동일한 회전을 적용합니다. 내적 `q'_m · k'_n`은 `(m - n)`만의 함수가 됩니다. 즉: **어텐션 점수는 상대적 거리에만 의존합니다**, 회전이 절대적 포지션을 기준으로 이루어졌음에도 불구하고. 아름다운 트릭입니다.

RoPE 확장: `base`는 재학습 없이 더 긴 컨텍스트로 외삽하기 위해 스케일링할 수 있습니다 (NTK-aware, YaRN, LongRoPE). Llama 3은 이 방식으로 컨텍스트를 8K에서 128K로 확장했습니다.

### ALiBi

임베딩 트릭을 건너뛰세요. 어텐션 점수에 직접 편향을 더합니다:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

여기서 `m_h`는 헤드별 기울기입니다 (예: `1 / 2^(8·h/H)`). 가까운 토큰은 부스트되고, 먼 토큰은 페널티를 받습니다. 학습 시간 비용이 없습니다. 논문은 길이 외삽이 사인 함수 방식보다 뛰어나며, 원래 학습된 길이에서는 RoPE와 동등한 성능을 보인다고 보여줍니다.

### 2026년에 선택할 것

| 변형 | 외삽 | 학습 비용 | 사용처 |
|---------|---------------|---------------|---------|
| 절대 사인 함수 | 낮음 | 무료 | 원조 트랜스포머, 초기 BERT |
| 학습된 절대 위치 | 없음 | 매우 적음 | GPT-2, GPT-3 |
| RoPE | 스케일링으로 좋음 | 무료 | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 매우 좋음 | 미세 조정 단계 | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 매우 좋음 | 무료 | BLOOM, MPT, Baichuan |

RoPE가 승리한 이유는 아키텍처를 변경하지 않고 어텐션에 쉽게 통합되며, 상대적 위치를 인코딩하고, `base` 하이퍼파라미터가 긴 컨텍스트 미세 조정을 위한 깔끔한 조절旋钮을 제공하기 때문입니다.

```figure
rope-explorer
```

## 구현하기

### 1단계: 사인 함수 인코딩

`code/main.py`을 참고하세요. 4줄 계산:

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

첫 번째 어텐션 레이어 전에 임베딩 행렬에 이 값을 더하세요.

### 2단계: Q, K에 RoPE 적용

RoPE는 Q와 K에 제자리(in-place)로 작동합니다. 각 차원 쌍에 대해:

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

중요: 포지션 `m`의 Q와 포지션 `n`의 K에 동일한 함수를 적용하세요. 내적은 모든 좌표 쌍에 `cos((m-n)·θ_i)` 요인을 포함합니다. 어텐션은 상대적 위치를 무료로 학습합니다.

### 3단계: ALiBi 기울기와 편향

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 소프트맥스 전에 어텐션 점수에 더합니다
```

`(seq_len, seq_len)` 어텐션 점수 행렬에 `bias[h]`을 더한 후, `h` 헤드에서 소프트맥스를 적용합니다.

### 4단계: RoPE의 상대 거리 속성 검증하기

두 개의 랜덤 벡터 `a, b`을 선택합니다. `(pos_a, pos_b)`만큼 회전한 후, `(pos_a + k, pos_b + k)`만큼 더 회전합니다. 두 내적 값이 부동 소수점 오차 범위 내에서 일치해야 합니다. 이 속성이 RoPE의 핵심입니다. 절대 오프셋에는 변하지 않으며, 상대적 간격만 중요합니다.

## 사용하기

PyTorch 2.5+는 `torch.nn.functional`에 RoPE 유틸리티를 제공합니다. 대부분의 프로덕션 코드는 어텐션 커널 내부에서 RoPE가 적용되는 `flash_attn` 또는 `xformers`를 사용합니다.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026년 롱 컨텍스트 트릭:**

- **NTK 인식 보간.** 4K에서 16K+로 확장할 때 `base`을 `base * (scale_factor)^(d/(d-2))`로 재스케일합니다.
- **YaRN.** 긴 컨텍스트에서 어텐션 엔트로피를 보존하는 더 스마트한 보간법입니다. Llama 3.1 128K가 이를 사용합니다.
- **LongRoPE.** Microsoft의 2024년 방법으로, 진화적 검색을 통해 차원별 스케일 팩터를 선택합니다. Phi-3-Long가 이를 사용합니다.
- **위치 보간 + 미세 조정.** 확장 팩터만큼 위치를 축소하고 1~5B 토큰으로 미세 조정합니다. 놀라울 정도로 효과적입니다.

## 출시하기

`outputs/skill-positional-encoding-picker.md`을 참조하세요. 이 스킬은 목표 컨텍스트 길이, 외삽 필요성, 학습 예산을 고려하여 새 모델에 대한 인코딩 전략을 선택합니다.

## 연습 문제

1. **쉬움.** `max_len=512, d=128`에 대해 사인 코사인 `PE` 행렬을 히트맵으로 플롯합니다. "차원 인덱스가 커질수록 줄무늬가 넓어지는" 패턴을 확인합니다.
2. **중간.** NTK 인식 RoPE 스케일링을 구현합니다. 길이 256의 시퀀스로 작은 LM를 학습한 후, 스케일링 유무에 따라 길이 1024에서 테스트합니다. 퍼플렉시티를 측정합니다.
3. **어려움.** 동일한 어텐션 모듈에서 ALiBi와 RoPE를 구현합니다. 길이 512의 시퀀스로 복사 작업에 4층 트랜스포머를 학습합니다. 테스트 시 2048로 외삽합니다. 성능 저하를 비교합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 위치 인코딩 | "순서를 어텐션에 알려주는 역할" | 임베딩이나 어텐션에 위치를 인코딩하기 위해 추가되는 모든 신호. |
| Sinusoidal | "원래 방식" | `sin/cos`를 기하급수적 주파수로 임베딩에 더하는 방식; 외삽이 불가능합니다. |
| RoPE | "회전 임베딩" | Q, K를 위치에 의존하는 각도로 회전시킴; 내적(dot product)이 상대적 거리를 인코딩합니다. |
| ALiBi | "선형 편향 트릭" | 어텐션 점수에 `-m·\|i-j\|`를 더하는 방식; 임베딩이 필요 없으며 외삽 성능이 뛰어납니다. |
| base | "RoPE의 조절旋钮" | RoPE의 주파수 스케일러; 추론 시 컨텍스트를 확장하려면 값을 증가시킵니다. |
| NTK-aware | "RoPE 스케일링 트릭" | `base`를 재스케일링하여 컨텍스트가 확장될 때 고주파 차원이 압축되지 않도록 합니다. |
| YaRN | "고급 방식" | 차원별 보간+외삽으로 어텐션 엔트로피를 보존합니다. |
| Extrapolation | "학습된 길이를 넘어서 작동" | 위치 인코딩 방식이 학습에서 본 `max_len`을 넘어서도 올바른 출력을 생성할 수 있는가? |

## 추가 읽기

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 원본 sinusoidal 방식.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 논문.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 최신 RoPE 스케일링 기법.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta의 Llama 2 롱 컨텍스트 논문.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long이 사용하며 '사용하기' 섹션에서 인용된 Microsoft 방식.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 모든 RoPE 스케일링 방식(default, linear, dynamic, YaRN, LongRoPE, Llama-3)의 프로덕션급 구현.
