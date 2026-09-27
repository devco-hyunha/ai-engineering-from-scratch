# 전체 트랜스포머(The Full Transformer) — 인코더 + 디코더

> 어텐션(Attention)이 주인공입니다. 잔차 연결(residual connection), 정규화(normalization), 피드포워드(feed-forward), 크로스 어텐션(cross-attention) 등 그 외의 모든 요소는 어텐션을 깊게 쌓을 수 있도록 돕는 비계(scaffolding) 역할을 합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## 문제점 (The Problem)

단일 어텐션 레이어(attention layer)는 특징 추출기(feature extractor)일 뿐, 모델이 아닙니다. 레이어당 한 번의 행렬 곱셈(matmul)만으로는 언어를 처리하기 위한 충분한 용량(capacity)을 확보할 수 없습니다. 깊이(depth)가 필요하며, 적절한 배관(plumbing) 구조가 없다면 깊이는 제대로 작동하지 않습니다.

2017년 Vaswani의 논문은 단일 어텐션 레이어를 쌓을 수 있는 블록(stackable block)으로 변모시킨 6가지 설계 결정을 제시했습니다. 이후 등장한 모든 트랜스포머 — 인코더 전용(BERT), 디코더 전용(GPT), 인코더-디코더(T5) — 는 동일한 골격(skeleton)을 상속받습니다. 2026년 현재, 블록들은 더욱 정교하게 다듬어졌지만(RMSNorm, SwiGLU, pre-norm, RoPE), 그 골격은 동일합니다.

이 레슨은 그 골격에 대한 내용입니다. 다음 레슨들에서 이를 전문화합니다 — 06은 인코더(encoders), 07은 디코더(decoders), 08은 인코더-디코더(encoder-decoder)를 다룹니다.

## 개념 (The Concept)

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 6가지 구성 요소 (The six pieces)

1. **임베딩 + 위치 신호 (Embedding + positional signal).** 토큰을 벡터로 변환합니다. 위치 정보는 RoPE(현대적 방식) 또는 sinusoidal(고전적 방식)을 통해 주입됩니다.
2. **셀프 어텐션 (Self-attention).** 모든 위치가 서로를 참조합니다. 디코더에서는 마스킹(Masked) 처리됩니다.
3. **피드포워드 네트워크 (Feed-forward network, FFN).** 위치별(Position-wise) 2층 MLP 구조입니다: `W_2 · activation(W_1 · x)`. 기본 확장 비율은 4배입니다.
4. **잔차 연결 (Residual connection).** `x + sublayer(x)` 형태입니다. 이것이 없다면 약 6개 층을 지나면서 기울기 소실(gradients vanish)이 발생합니다.
5. **레이어 정규화 (Layer normalization).** `LayerNorm` 또는 `RMSNorm`(현대적 방식)을 사용합니다. 잔차 스트림(residual stream)을 안정화합니다.
6. **크로스 어텐션 (Cross-attention, 디코더 전용).** 쿼리(Queries)는 디코더에서 오고, 키(Keys)와 값(Values)은 인코더 출력에서 옵니다.

하나의 블록을 통과하는 벡터의 흐름을 살펴보세요: 어텐션이 위치 간 정보를 혼합하고, 잔차 연결이 이를 앞으로 전달하며, FFN이 이를 변환하고, 정규화가 스트림을 안정적으로 유지합니다.

```figure
transformer-block
```

### 인코더 블록 (Encoder block, BERT 및 T5 인코더에서 사용)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

인코더는 양방향(bidirectional)입니다. 마스킹(masking)을 사용하지 않습니다. 모든 위치가 모든 위치를 참조합니다.

### 디코더 블록 (Decoder block, GPT 및 T5 디코더에서 사용)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

디코더는 블록당 세 개의 서브레이어(sublayer)를 가집니다. 그중 중간에 위치한 크로스 어텐션(cross-attention)은 인코더에서 디코더로 정보가 흐르는 유일한 경로입니다. 순수 디코더 전용 아키텍처(GPT)에서는 크로스 어텐션이 생략되며, 마스크드 셀프 어텐션(masked self-attention)과 FFN만 존재합니다.

### Pre-norm vs post-norm (프리-노름 대 포스트-노름)

원문 논문: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm은 2019년경부터 선호도가 낮아졌습니다. 세심한 웜업(warmup) 없이는 깊은 모델을 학습시키기가 더 어렵기 때문입니다. Pre-norm(`sublayer` *이전*에 `LN` 적용)은 2026년의 기본 방식입니다: Llama, Qwen, GPT-3+, Mistral 모두 이를 사용합니다.

### 2026년형 현대화된 블록 (The 2026 modernized block)

Vaswani 2017 모델은 `LayerNorm` + `ReLU`를 탑재하여 출시되었습니다. 현대적인 스택은 이 두 가지를 모두 대체했습니다. 실제 프로덕션 환경의 블록 구성은 다음과 같습니다:

| 구성 요소 (Component) | 2017 | 2026 |
|-----------|------|------|
| 정규화 (Normalization) | `LayerNorm` | `RMSNorm` |
| FFN 활성화 함수 (FFN activation) | `ReLU` | `SwiGLU` |
| FFN 확장 계수 (FFN expansion) | 4× | 2.6× (`SwiGLU`는 세 개의 행렬을 사용하며, 총 파라미터 수는 동일함) |
| 위치 인코딩 (Position) | Sinusoidal absolute | `RoPE` |
| 어텐션 (Attention) | Full `MHA` | `GQA` (또는 `MLA`) |
| 편향 항 (Bias terms) | 있음 (Yes) | 없음 (No) |

`RMSNorm`은 `LayerNorm`의 평균 중심화(mean-centering) 과정을 생략하여(뺄셈 연산 1회 감소) 연산량을 절약하며, 경험적으로 최소한 `LayerNorm`만큼의 안정성을 보여줍니다. `SwiGLU` (`Swish(W1 x) ⊙ W3 x`)는 Llama, PaLM, Qwen 논문에서 `ReLU`/`GELU` FFN보다 일관되게 약 0.5 point ppl(per-token perplexity)만큼 우수한 성능을 보입니다.

### 파라미터 수 (Parameter count)

`d_model = d` 및 FFN 확장 계수 `r`을 가진 하나의 블록에 대하여:

- MHA: `4 · d²` (Q, K, V, O 프로젝션)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: 무시할 수 있는 수준

`d = 4096, r = 2.6, layers = 32` (대략 Llama 3 8B 모델)일 때, 총합: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 파라미터/레이어 × 32 ≈ 7B` (임베딩 및 헤드 제외). 이는 발표된 수치와 일치합니다.

## 직접 구현해 보기 (Build It)

### 1단계: 구성 요소 (the building blocks)

Lesson 03의 작은 `Matrix` 클래스를 사용합니다 (독립성을 위해 이 파일로 복사되었습니다):

- `layer_norm(x, eps=1e-5)` — 평균을 빼고 표준편차로 나눕니다.
- `rms_norm(x, eps=1e-6)` — RMS로 나눕니다. 평균을 빼지 않습니다.
- `gelu(x)` 및 `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` 및 `decoder_block(x, enc_out, params)`.

전체적인 연결 구조는 `code/main.py`를 참조하세요.

### 2단계: 2계층 인코더(encoder)와 2계층 디코더(decoder) 연결하기

두 모듈을 쌓으세요. 인코더의 출력을 모든 디코더의 크로스 어텐션(cross-attention) 메커니즘에 전달합니다. 출력 투영(output projection) 직전에 최종 레이어 정규화(LN)를 추가하세요.

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

### 3단계: 토이 예제(toy example)로 순전파(forward) 실행하기

6개의 토큰으로 구성된 소스(source)와 5개의 토큰으로 구성된 타겟(target)을 입력합니다. 출력 형태(output shape)가 `(5, vocab)`인지 확인해 보세요. 학습(training)은 진행하지 않습니다. 이번 레슨의 목적은 손실(loss)이 아니라 아키텍처(architecture)를 이해하는 것입니다.

### 4단계: RMSNorm + SwiGLU로 교체하기

`LayerNorm`과 `ReLU-FFN`을 `RMSNorm`과 `SwiGLU`로 교체해 보세요. 텐서의 형상(shape)이 여전히 일치하는지 확인해야 합니다. 이는 단 한 번의 함수 교체만으로 이루어지는 2026년형 현대화 방식입니다.

## 사용 방법 (Use It)

PyTorch/TF의 참조 구현체로는 `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`가 있습니다. 하지만 2026년 기준 대부분의 프로덕션 코드는 다음과 같은 이유로 자체적인 블록을 직접 구현하여 사용합니다:

- Flash Attention이 `nn.MultiheadAttention`을 통하지 않고 어텐션 내부에서 직접 호출됩니다.
- GQA / MLA가 표준 라이브러리 참조 구현에 포함되어 있지 않습니다.
- RoPE, RMSNorm, SwiGLU는 PyTorch의 기본값이 아닙니다.

HF `transformers`에는 반드시 읽어보아야 할 깔끔한 참조 블록들이 있습니다: `modeling_llama.py`는 2026년 기준 표준적인 decoder-only 블록입니다. 약 500줄 정도의 분량이며, 한 번쯤 차근차근 살펴보는 가치가 있습니다.

**인코더(Encoder) vs 디코더(Decoder) vs 인코더-디코더(Encoder-decoder) — 선택 기준:**

| 요구 사항 | 선택 | 예시 |
|------|------|---------|
| 분류(Classification), 임베딩(Embeddings), 텍스트 기반 질의응답(QA) | Encoder-only | BERT, DeBERTa, ModernBERT |
| 텍스트 생성, 채팅, 코드, 추론 | Decoder-only | GPT, Llama, Claude, Qwen |
| 구조화된 입력 → 구조화된 출력 (번역, 요약) | Encoder-decoder | T5, BART, Whisper |

Decoder-only 모델은 확장이 가장 용이하고 이해(Comprehension)와 생성(Generation)을 모두 처리할 수 있기 때문에 언어 모델의 주류가 되었습니다. 인코더-디코더 모델은 입력 데이터가 명확한 "소스 시퀀스(source sequence)" 정체성을 가질 때(번역, 음성 인식, 구조화된 작업 등) 여전히 가장 효과적입니다.

## Ship It (실전 적용)

`outputs/skill-transformer-block-reviewer.md`를 참조하세요. 이 스킬은 새로운 트랜스포머 블록(transformer block) 구현을 2026년 기본 표준(defaults)과 비교하여 검토하고, 누락된 요소(pre-norm, RoPE, RMSNorm, GQA, FFN expansion ratio)를 찾아냅니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 설정에서 `encoder_block`의 파라미터 수를 계산해 보세요. 블록을 구현한 뒤 `sum(p.numel() for p in block.parameters())`를 사용하여 검증해 보세요.
2. **중간 (Medium).** Post-norm 방식을 Pre-norm 방식으로 전환해 보세요. 두 방식을 모두 초기화한 후, 무작위 입력(random input)에 대해 12개의 레이어를 쌓았을 때의 활성화 노름(activation norm)을 측정해 보세요. Post-norm의 활성화 값은 폭발(explode)해야 하며, Pre-norm의 활성화 값은 유계(bounded) 상태를 유지해야 합니다.
3. **어려움 (Hard).** 간단한 복사 작업(toy copy task, `x`를 역순으로 복사)을 수행하는 4계층 인코더-디코더(encoder-decoder)를 구현해 보세요. 100단계(steps) 동안 학습시킨 후 손실(loss)을 보고하세요. 이후 RMSNorm + SwiGLU + RoPE로 교체했을 때 손실이 감소하나요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| 블록 (Block) | "트랜스포머 레이어 하나" | 잔차 연결(residual connections)로 감싸진 norm + attention + norm + FFN의 스택. |
| 잔차 (Residual) | "스킵 연결 (Skip connection)" | `x + f(x)` 출력; 깊은 스택을 통한 그래디언트 흐름을 가능하게 함. |
| 프리-노름 (Pre-norm) | "사후가 아닌 사전 정규화" | 현대적 방식: `x + sublayer(LN(x))`. 웜업(warmup) 기술 없이도 더 깊은 모델 학습 가능. |
| RMSNorm | "평균을 제외한 LayerNorm" | RMS로 나누는 방식; 연산은 하나 줄어들지만 경험적 안정성은 동일함. |
| SwiGLU | "모두가 갈아탄 FFN" | `Swish(W1 x) ⊙ W3 x → W2`. 언어 모델의 perplexity(ppl) 측면에서 ReLU/GELU보다 우수함. |
| 크로스 어텐션 (Cross-attention) | "디코더가 인코더를 보는 방식" | 디코더의 Q와 인코더 출력의 K/V를 사용하는 MHA(Multi-Head Attention). |
| FFN 확장 (FFN expansion) | "중간 MLP의 너비" | `d_model` 대비 은닉층 크기의 비율로, 보통 4(LayerNorm) 또는 2.6(SwiGLU)임. |
| 바이어스 프리 (Bias-free) | "+b 항 제거" | 현대적 스택은 선형 레이어에서 바이어스를 생략함; 약간의 ppl 개선 및 모델 크기 감소 효과. |

## 추가 읽기 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 오리지널 블록 사양(block spec).
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 왜 pre-norm이 post-norm보다 우수한지에 대한 심층 분석.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 표준적인 2026 decoder-only 블록 구현.
