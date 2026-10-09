# 전체 트랜스포머 — 인코더 + 디코더

> 어텐션이 핵심입니다. 나머지 요소들 — 잔차 연결, 정규화, 피드포워드, 교차 어텐션 — 는 어텐션을 깊게 쌓을 수 있게 해주는 뼈대입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션), 7단계 · 03강 (멀티헤드 어텐션), 7단계 · 04강 (위치 인코딩)
**시간:** 약 75분

## 문제점

단일 어텐션 레이어는 모델이 아니라 특징 추출기입니다. 레이어당 하나의 행렬 곱은 언어를 처리하기에는 용량이 부족합니다. 깊이가 필요하며, 올바른 배관(plumbing)이 없으면 깊이가 무너집니다.

2017년 Vaswani 논문은 하나의 어텐션 레이어를 쌓을 수 있는 블록으로 만든 6가지 설계 결정을 제시했습니다. 이후의 모든 트랜스포머 — 인코더 전용(BERT), 디코더 전용(GPT), 인코더-디코더(T5) — 는 동일한 뼈대를 상속받습니다. 2026년에는 블록이 정교해졌지만(RMSNorm, SwiGLU, pre-norm, RoPE), 뼈대는 동일합니다.

이 강의는 뼈대를 다룹니다. 다음 강의에서는 이를 특화합니다 — 06강은 인코더, 07강은 디코더, 08강은 인코더-디코더를 다룹니다.

## 개념

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 6가지 구성 요소

1. **임베딩 + 위치 신호.** 토큰을 벡터로 변환합니다. 위치는 RoPE(현대적) 또는 사인 함수(전통적)로 주입됩니다.
2. **셀프 어텐션.** 모든 위치가 다른 모든 위치를 참조합니다. 디코더에서는 마스킹됩니다.
3. **피드포워드 네트워크(FFN).** 위치별 2층 MLP: `W_2 · activation(W_1 · x)`. 기본 확장 비율은 4배입니다.
4. **잔차 연결.** `x + sublayer(x)`. 이 연결이 없으면 약 6레이어 이후에서 기울기가 사라집니다.
5. **레이어 정규화.** `LayerNorm` 또는 `RMSNorm` (현대적). 잔차 스트림을 안정화합니다.
6. **교차 어텐션 (디코더 전용).** 쿼리는 디코더에서, 키와 값은 인코더 출력에서 가져옵니다.

벡터가 하나의 블록을 통과하는 과정을 살펴보세요: 어텐션이 위치 간에 혼합하고, 잔차가 이를 전달하며, FFN이 이를 변환하고, 정규화가 스트림을 안정적으로 유지합니다.

```figure
transformer-block
```

### 인코더 블록 (BERT, T5 인코더에서 사용)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

인코더는 양방향입니다. 마스킹이 없으며, 모든 위치가 모든 위치를 참조합니다.

### 디코더 블록 (GPT, T5 디코더에서 사용)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

디코더는 각 블록에 세 개의 하위 계층이 있습니다. 가운데 계층인 교차 어텐션(Cross-Attention)은 인코더에서 디코더로 정보가 흐르는 유일한 위치입니다. 순수 디코더 전용 아키텍처(GPT)에서는 교차 어텐션이 생략되며, 마스킹된 셀프 어텐션(Self-Attention)과 FFN만 남습니다.

### 사전 정규화 vs 사후 정규화

원 논문: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. 사후 정규화는 2019년경부터 선호도가 떨어졌습니다. 세심한 워밍업(Warmup) 없이 깊은 모델을 학습하기가 더 어렵기 때문입니다. 사전 정규화(서브레이어 *이전*에 `LN` 적용)는 2026년 기본값입니다. Llama, Qwen, GPT-3+, Mistral 모두 이를 사용합니다.

### 2026년 현대화된 블록

Vaswani 2017은 LayerNorm과 ReLU를 사용했습니다. 현대 스택은 두 요소 모두를 대체했습니다. 실제 프로덕션 블록의 모습은 다음과 같습니다:

| 구성 요소 | 2017 | 2026 |
|-----------|------|------|
| 정규화 | LayerNorm | RMSNorm |
| FFN 활성화 함수 | ReLU | SwiGLU |
| FFN 확장 | 4× | 2.6× (SwiGLU는 세 개의 행렬을 사용하며, 총 매개변수 수는 동일) |
| 위치 인코딩 | 사인 절대 위치 | RoPE |
| 어텐션 | 전체 MHA | GQA (또는 MLA) |
| 편향(Bias) 항 | 있음 | 없음 |

RMSNorm은 LayerNorm의 평균 중심화(mean-centering)를 제거합니다(뺄셈 한 번 감소). 이는 연산량을 절약하며, 경험적으로 적어도 동일한 안정성을 제공합니다. SwiGLU (`Swish(W1 x) ⊙ W3 x`)는 Llama, PaLM, Qwen 논문에서 ReLU/GELU FFN보다 약 0.5 ppl(perplexity) 포인트 일관되게 더 좋은 성능을 보입니다.

### 매개변수 수

`d_model = d`이고 FFN 확장 비율이 `r`인 단일 블록의 경우:

- MHA: `4 · d²` (Q, K, V, O 투영)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- 정규화: 무시할 수 있는 수준

`d = 4096, r = 2.6, layers = 32` (대략 Llama 3 8B)일 때, 총합: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (임베딩과 헤드를 포함). 공개된 매개변수 수와 일치합니다.

## 구현하기

### 1단계: 구성 요소

3강의 작은 `Matrix` 클래스를 사용합니다(독립성을 위해 이 파일에 복사했습니다):

- `layer_norm(x, eps=1e-5)` — 평균을 빼고 표준 편차로 나눕니다.
- `rms_norm(x, eps=1e-6)` — RMS로 나눕니다. 평균 뺄셈이 없습니다.
- `gelu(x)` 및 `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` 및 `decoder_block(x, enc_out, params)`.

전체 연결 구조는 `code/main.py`를 참조하세요.

### 2단계: 2층 인코더와 2층 디코더를 연결합니다

이들을 쌓습니다. 인코더의 출력을 모든 디코더의 교차 어텐션(Cross-Attention)에 전달합니다. 출력 투사(output projection) 전에 최종 LN(Layer Normalization)을 추가합니다.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 3단계: 장난감 예제(toy example)에 대해 forward를 실행합니다

6개 토큰의 소스(source)와 5개 토큰의 타겟(target)을 입력으로 전달합니다. 출력 형상이 `(5, vocab)`인지 확인합니다. 학습은 하지 않습니다 — 이 강의는 아키텍처에 관한 것이며 손실(loss)에 관한 것이 아닙니다.

### 4단계: RMSNorm + SwiGLU로 교체합니다

LayerNorm과 ReLU-FFN을 RMSNorm과 SwiGLU로 교체합니다. 형상이 여전히 일치하는지 확인합니다. 이는 하나의 함수 치환으로 이루어진 2026년 현대화입니다.

## 사용하기

PyTorch/TF 참조 구현: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. 하지만 대부분의 2026년 프로덕션 코드는 자체 블록을 구현합니다. 그 이유는:

- Flash Attention은 어텐션(Attention) 내부에서 호출되며 `nn.MultiheadAttention`를 통해 호출되지 않습니다.
- GQA / MLA는 표준 라이브러리 참조에 포함되어 있지 않습니다.
- RoPE, RMSNorm, SwiGLU는 PyTorch의 기본값이 아닙니다.

HF `transformers`에는 읽어야 할 깔끔한 참조 블록이 있습니다: `modeling_llama.py`은 정통적인 2026년 디코더 전용(decoder-only) 블록입니다. 약 500줄이며 한 번 훑어볼 가치가 있습니다.

**인코더 vs 디코더 vs 인코더-디코더 — 선택 기준:**

| 필요 | 선택 | 예시 |
|------|------|---------|
| 분류, 임베딩, 텍스트 기반 QA | 인코더 전용 | BERT, DeBERTa, ModernBERT |
| 텍스트 생성, 채팅, 코드, 추론 | 디코더 전용 | GPT, Llama, Claude, Qwen |
| 구조화된 입력 → 구조화된 출력 (번역, 요약) | 인코더-디코더 | T5, BART, Whisper |

디코더 전용(decoder-only)이 우세한 이유는 가장 깔끔하게 확장(scale)되며 이해(comprehension)와 생성(generation)을 모두 처리하기 때문입니다. 인코더-디코더는 입력이 명확한 "소스 시퀀스(source sequence)" 정체성을 가질 때(번역, 음성 인식, 구조화된 작업) 여전히 최선입니다.

## 출시하기

`outputs/skill-transformer-block-reviewer.md`을 참조하세요. 이 스킬은 새로운 트랜스포머 블록 구현을 2026년 기본값과 비교하여 누락된 부분(pre-norm, RoPE, RMSNorm, GQA, FFN 확장 비율)을 표시합니다.

## 연습 문제

1. **쉬움.** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`에서 encoder_block의 매개변수(Parameter) 수를 세어 보세요. 블록을 구현하고 `sum(p.numel() for p in block.parameters())`를 사용하여 검증합니다.
2. **중간 난이도.** post-norm에서 pre-norm으로 전환하세요. 두 방식을 모두 초기화하고, 랜덤 입력으로 12층을 쌓은 후 활성화 노름(activation norm)을 측정해 보세요. post-norm의 활성화는 폭발해야 하며, pre-norm의 활성화는 유한한 값으로 유지되어야 합니다.
3. **고난이도.** 장난감 복사 작업(copy `x` reversed)에 4층 인코더-디코더를 구현하세요. 100 스텝 동안 학습하고 손실(loss)을 보고하세요. RMSNorm + SwiGLU + RoPE로 교체했을 때 손실이 감소하는지 확인해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 블록(Block) | "하나의 트랜스포머 레이어" | 정규화 + 어텐션 + 정규화 + FFN의 스택으로, 잔여 연결(residual connections)로 감싸져 있습니다. |
| 잔여 연결(Residual) | "스킵 연결(Skip connection)" | `x + f(x)` 출력; 깊은 스택을 통해 기울기 흐름을 가능하게 합니다. |
| Pre-norm | "사후가 아닌 사전에 정규화" | 현대적 방식: `x + sublayer(LN(x))`. 워밍업(warmup)의 복잡한 조정 없이 더 깊은 모델도 학습할 수 있습니다. |
| RMSNorm | "평균을 제외한 LayerNorm" | RMS로 나누는 방식; 연산이 하나 줄어들며, 경험적으로 동일한 안정성을 제공합니다. |
| SwiGLU | "모두가 채택한 FFN" | `Swish(W1 x) ⊙ W3 x → W2`. LM 퍼플렉시티(perplexity)에서 ReLU/GELU보다 우수합니다. |
| 교차 어텐션(Cross-attention) | "디코더가 인코더를 보는 방법" | Q는 디코더에서, K/V는 인코더 출력에서 가져오는 MHA입니다. |
| FFN 확장(FFN expansion) | "중간 MLP의 너비" | hidden-size와 d_model의 비율로, 보통 4 (LayerNorm) 또는 2.6 (SwiGLU)입니다. |
| 편향 제거(Bias-free) | "+b 항을 제거" | 현대적 스택은 선형 레이어의 편향(bias)을 생략합니다; 퍼플렉시티가 약간 개선되고 모델이 더 작아집니다. |

## 추가 읽기

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 원본 블록 사양.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm이 post-norm보다 깊은 모델에서 더 좋은 이유.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026년 표준 decoder-only 블록.
