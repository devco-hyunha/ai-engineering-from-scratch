# GPT — 인과적 언어 모델링 (Causal Language Modeling)

> BERT는 양방향을 봅니다. GPT는 오직 과거만을 봅니다. 삼각형 마스크(triangle mask)는 현대 AI에서 가장 중요한 단 한 줄의 코드입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## 문제 (The Problem)

언어 모델은 하나의 질문에 답합니다. 즉, 주어진 첫 `t-1`개의 토큰이 있을 때, 토큰 `t`에 대한 확률 분포는 무엇인가? 이 신호, 즉 다음 토큰 예측(next-token prediction)을 통해 학습하면 한 번에 하나의 토큰씩 임의의 텍스트를 생성할 수 있는 모델을 얻게 됩니다.

전체 시퀀스에 대해 엔드 투 엔드(end-to-end)로 병렬 학습시키려면, 각 위치의 예측이 오직 이전 위치들에만 의존해야 합니다. 그렇지 않으면 모델은 정답을 미리 보고 보는 방식으로 아주 쉽게 속임수를 쓰게 됩니다.

인과적 마스크(Causal mask)가 이 역할을 수행합니다. 이는 소프트맥스(softmax)를 적용하기 전 어텐션 점수(attention scores)에 더해지는 `-inf` 값들로 이루어진 단일 상삼각 행렬(upper-triangular matrix)입니다. 소프트맥스를 거치고 나면 해당 위치의 값들은 0이 됩니다. 이를 통해 각 위치는 자기 자신과 이전 위치들에만 어텐션을 줄 수 있습니다. 또한 시퀀스 전체에 한 번만 적용하면 되기 때문에, 단 한 번의 순전파(forward pass)로 $N$개의 병렬적인 다음 토큰 예측을 수행할 수 있습니다.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — 이들은 모두 동일한 핵심 루프를 가진 디코더 전용 인과적 트랜스포머(decoder-only causal transformers)입니다. 이들을 구분 짓는 요소는 데이터의 품질, 규모, 아키텍처의 개선, 그리고 사후 학습(post-training: SFT, RLHF, DPO 및 그 후속 기술들)입니다.

## 개념 (The Concept)

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 마스크 (The mask)

길이가 `N`인 시퀀스가 주어졌을 때, 다음과 같은 `N × N` 행렬을 생성합니다:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Softmax를 적용하기 전, 원본 어텐션 점수(raw attention scores)에 `M`을 더합니다. `exp(-inf) = 0`이므로, 마스킹된 위치는 가중치에 0을 기여하게 됩니다. 어텐션 행렬의 각 행은 이전 위치들에 대해서만 확률 분포를 형성합니다.

구현 비용: `torch.tril()` 호출 한 번. 계산 시간: 나노초 단위. 분야에 미치는 영향: 모든 것.

### 삼각형이 어디에서 오는가 (Where the triangle comes from)

마스크(mask)는 보통 어텐션(attention)에 덧붙여진 패치처럼 제시됩니다. 하지만 유도 과정을 반대 방향으로 돌려보면 신비로움이 사라집니다. 어텐션은 접두사 평균(prefix average)의 세 번째 정제 단계이며, 삼각형은 해당 평균의 루프 범위(loop bounds)를 행렬로 표현한 것입니다.

**1단계 — 접두사 평균 (prefix average).** 시퀀스에 대한 가장 단순한 인과적 요약(causal summary)입니다. 위치 `i`는 위치 `0…i`의 평균이 됩니다. 루프로 표현하면 `out[i] = X[:i+1].mean(0)`입니다. 동일한 계산을 행렬 곱셈 한 번으로 수행할 수 있습니다. 1로 이루어진 하삼각 행렬(lower-triangular matrix)을 가져와 각 행을 해당 개수로 나누고 곱하면 됩니다.

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

`A`의 `i`번째 행은 `[1/(i+1), …, 1/(i+1), 0, …, 0]`이 됩니다. 대각선 위의 0들은 인과성(causality)을 의미합니다. 미래의 정보가 마스킹된 것이 아니라, 미래의 정보가 애초에 합계에 포함되지 않았던 것입니다.

**2단계 — 학습된 가중치 (learned weights).** 균등 평균(uniform average)은 모든 과거 토큰을 동일하게 중요하다고 취급합니다. 이 1들을 학습된 점수 행렬 `S`로 교체합니다. 이제 행의 합이 구조적으로 1이 되지 않으므로, 개수로 나누는 대신 소프트맥스(softmax)를 사용하여 각 행을 정규화합니다. 소프트맥스는 정확히 0을 출력하지 않으므로 인과성이 깨질 수 있습니다. 단, 미래의 점수를 `-inf`로 입력하여 `exp(-inf) = 0`이 되게 한다면 예외입니다.

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

동일한 삼각형, 동일한 행 확률 행렬(row-stochastic matrix), 동일한 단일 행렬 곱셈입니다. `-inf` 마스크는 새로운 장치가 아닙니다. 이는 1단계의 0인 항목들이 소프트맥스의 입력 도메인으로 번역된 것뿐입니다.

**3단계 — 콘텐츠 의존적 가중치 (content-dependent weights).** 2단계에서 `S`는 학습 후 고정됩니다. 즉, 토큰의 내용과 상관없이 위치 7은 항상 위치 3에 동일한 가중치를 부여합니다. 이제 점수가 토큰 자체에 의존하게 만듭니다: `S = Q @ K.T / sqrt(d_k)`. 그 외에는 아무것도 변하지 않습니다. 마스킹, 소프트맥스, 행렬 곱셈 — 모두 동일합니다.

세 단계, 하나의 불변량(invariant): 하삼각 행렬인 행 확률 행렬과 시퀀스의 곱입니다. 균등 평균, 학습된 정적 가중치, 콘텐츠 의존적 가중치. 마스크는 어텐션에 추가된 것이 아닙니다. 평균 단계에서부터 살아남은 것입니다.

```figure
mask-derivation
```

### 병렬 학습, 직렬 추론 (Parallel training, serial inference)

**학습(Training):** 전체 `(N, d_model)` 시퀀스를 한 번에 순전파(forward-pass)하고, $N$개의 교차 엔트로피 손실(cross-entropy losses, 위치당 하나씩)을 계산한 뒤, 이를 합산하여 역전파(backprop)합니다. 시퀀스를 따라 병렬로 처리됩니다. 이것이 GPT 학습이 확장 가능한 이유입니다. 즉, 한 번의 GPU 패스로 배치 내의 100만 개 토큰을 처리할 수 있습니다.

**추론(Inference):** 토큰을 하나씩 생성합니다. `[t1, t2, t3]`를 입력하여 `t4`를 얻습니다. `[t1, t2, t3, t4]`를 입력하여 `t5`를 얻습니다. `[t1, t2, t3, t4, t5]`를 입력하여 `t6`를 얻습니다. KV 캐시(Lesson 12)는 `t1…tn`의 은닉 상태(hidden states)를 저장하여 매 단계마다 이를 다시 계산하지 않도록 돕습니다. 하지만 추론 시의 직렬 깊이(serial depth)는 출력 길이에 비례합니다. 이것이 자기회귀 비용(autoregressive tax)이며, 디코딩이 모든 LLM의 지연 시간(latency) 병목 구간인 이유입니다.

### 손실(Loss) — 시프트-바이-원(shift-by-one)

토큰 `[t1, t2, t3, t4]`가 주어졌을 때:

- 입력(Input): `[t1, t2, t3]`
- 타겟(Targets): `[t2, t3, t4]`

모든 위치 `i`에 대하여, `-log P(target_i | inputs[:i+1])`를 계산합니다. 이를 모두 합산합니다. 이것이 전체 시퀀스에 대한 교차 엔트로피(cross-entropy)입니다.

여러분이 들어본 모든 트랜스포머 언어 모델(Transformer LM)은 이 손실 함수를 사용하여 학습합니다. 사전 학습(Pre-training), 미세 조정(Fine-tuning), SFT 모두 동일한 손실 함수를 사용하며, 단지 데이터가 다를 뿐입니다.

### 디코딩 전략 (Decoding strategies)

학습이 완료된 후에는 샘플링(sampling) 선택이 생각보다 훨씬 중요합니다.

| 방법 (Method) | 동작 방식 | 사용 시점 |
|--------|--------------|-------------|
| Greedy | 매 단계에서 `argmax` 수행 | 결정론적 작업, 코드 완성 |
| Temperature | 로짓(logits)을 $T$로 나누고 샘플링 | 창의적인 작업, $T$가 높을수록 다양성 증가 |
| Top-k | 상위 $k$개의 토큰에서만 샘플링 | 낮은 확률의 꼬리(tail) 부분 제거 |
| Top-p (nucleus) | 누적 확률이 $p$ 이상인 최소 토큰 집합에서 샘플링 | 2020년 이후 기본값; 분포 형태에 적응함 |
| Min-p | `p > min_p * max_p`인 토큰 유지 | 2024년 이후; top-p보다 긴 꼬리를 더 잘 제거함 |
| Speculative decoding | 초안 모델(draft model)이 $N$개 토큰을 제안하고, 큰 모델이 검증 | 동일한 품질에서 지연 시간(latency) 2~3배 감소 |

2026년 기준으로, 오픈 웨이트(open-weights) 모델에는 `min-p`와 `temperature 0.7` 조합이 합리적인 기본값입니다. Speculative decoding은 모든 프로덕션 추론 스택(production inference stack)에서 필수적인 요소입니다.

### "GPT 레시피"가 성공한 이유 (What made the "GPT recipe" work)

1. **디코더 전용 (Decoder-only).** 인코더 오버헤드가 없습니다. 레이어당 한 번의 어텐션(Attention) + FFN(Feed-Forward Network) 통과로 구성됩니다.
2. **스케일링 (Scaling).** 124M → 1.5B → 175B → 수조(trillions) 단위로 확장되었습니다. Chinchilla 스케일링 법칙(Lesson 13)은 컴퓨팅 자원을 어떻게 사용해야 하는지 알려줍니다.
3. **인컨텍스트 학습 (In-context learning).** 6B~13B 파라미터 규모에서 나타나기 시작했습니다. 모델은 미세 조정(fine-tuning) 없이도 퓨샷(few-shot) 예시를 따를 수 있습니다.
4. **RLHF.** 인간의 선호도에 맞춘 사후 학습(Post-training)을 통해 가공되지 않은 사전 학습된 텍스트를 채팅 어시스턴트로 전환했습니다.
5. **Pre-norm + RoPE + SwiGLU.** 대규모 스케일에서도 안정적인 학습을 가능하게 합니다.

핵심 아키텍처는 GPT-2 이후로 크게 변하지 않았습니다. 흥미로운 모든 변화는 데이터, 규모(scale), 그리고 사후 학습(post-training)에서 일어났습니다.

```figure
causal-mask
```

## 직접 구현해 보기 (Build It)

### 1단계: 인과적 마스크 (the causal mask)

`code/main.py`를 참조하세요. 한 줄로 구현하면 다음과 같습니다:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Softmax를 적용하기 전, 어텐션 점수(attention scores)에 이 마스크를 더해 주세요. 이것이 전체 메커니즘의 전부입니다.

### 2단계: 2계층 GPT 스타일 모델 (a 2-layer GPT-ish model)

두 개의 디코더 블록(masked self-attention + FFN, cross-attention 제외)을 쌓습니다. 토큰 임베딩(token embedding), 위치 인코딩(positional encoding), 그리고 언임베딩(unembedding, GPT-2 이후 표준 기법인 토큰 임베딩 행렬과 가중치를 공유하는 방식)을 추가합니다.

### 3단계: 다음 토큰 예측, 엔드투엔드 (next-token prediction, end-to-end)

20개의 토큰으로 구성된 장난감 어휘집(toy vocab)을 사용하여 모든 위치에서 로짓(logits)을 생성합니다. 1만큼 이동시킨(shift-by-one) 타겟에 대해 교차 엔트로피 손실(cross-entropy loss)을 계산합니다. 그래디언트(gradient)는 계산하지 않습니다 — 이는 순전파(forward-pass)를 통한 무결성 검사(sanity check)입니다.

### 4단계: 샘플링 (Sampling)

Greedy, temperature, top-k, top-p, min-p를 구현해 보세요. 각각을 고정된 프롬프트에 실행하고 출력을 비교해 보세요. 샘플링 함수는 10줄 내외로 작성할 수 있습니다.

## 사용 방법 (Use It)

PyTorch, 2026년 관용구(idiom):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

내부적으로 `generate()`는 순전파(forward pass)를 실행하고, 마지막 위치의 로짓(logits)을 추출하며, 다음 토큰을 샘플링하고, 이를 추가한 뒤 과정을 반복합니다. 모든 프로덕션 LLM 추론 스택(vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX)은 배치 프리필(batched prefill), 연속 배칭(continuous batching), KV 캐시 페이징(KV cache paging), 투기적 디코딩(speculative decoding)과 같은 강력한 최적화를 적용하여 동일한 루프를 구현합니다.

**GPT vs BERT, 각각 한 줄 요약:** GPT는 $P(x_t | x_{<t})$를 예측합니다. BERT는 $P(x_{masked} | x_{unmasked})$를 예측합니다. 손실(loss)은 모델이 생성(generate)할 수 있는지 여부를 결정합니다.

## Ship It (실행해 보기)

`outputs/skill-sampling-tuner.md`를 참조하세요. 이 스킬은 새로운 생성 작업(generation task)을 위한 샘플링 파라미터를 선택하고, 결정론적 디코딩(deterministic decoding)이 필요한 경우 이를 표시합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하고, softmax를 거친 후의 인과적 어텐션 행렬(causal attention matrix)이 하삼각 행렬(lower-triangular)인지 확인해 보세요. 샘플 확인: 3번 행은 0~3번 열에만 가중치가 있어야 합니다.
2. **중간 (Medium).** 너비(width) 4인 빔 서치(beam search)를 구현해 보세요. 10개의 짧은 프롬프트에 대해 빔-4(beam-4)와 그리디(greedy) 방식의 퍼플렉서티(perplexity)를 비교해 보세요. 빔 서치가 항상 더 나은 결과를 보이나요? (힌트: 보통 번역 작업에는 유리하지만, 개방형 채팅에는 그렇지 않을 수 있습니다.)
3. **어려움 (Hard).** 투기적 디코딩(speculative decoding)을 구현해 보세요. 2개 층(layer)의 아주 작은 모델을 초안 모델(draft model)로 사용하고, 6개 층의 모델을 검증 모델(verifier model)로 사용합니다. 길이가 64인 100개의 문장 생성 시 실제 실행 시간(wall-clock speedup)이 얼마나 빨라지는지 측정해 보세요. 생성된 결과물이 검증 모델의 그리디 방식 결과와 일치하는지 확인해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Causal mask (인과적 마스크) | "삼각형(The triangle)" | 위치 `i`가 `≤ i`인 위치만 볼 수 있도록 어텐션 점수에 더해지는 상삼각 `-inf` 행렬. |
| Next-token prediction (다음 토큰 예측) | "손실(The loss)" | 모든 위치에서 모델의 분포와 실제 다음 토큰 사이의 크로스 엔트로피(Cross-entropy). |
| Autoregressive (자기회귀) | "한 번에 하나씩 생성" | 출력을 다시 입력으로 피드백하는 방식; 훈련 중에는 병렬 처리가 가능하지만 생성 중에는 불가능함. |
| Logits (로짓) | "Pre-softmax 점수" | 소프트맥스(softmax) 적용 전 LM 헤드의 가공되지 않은 출력값; 샘플링은 이 값을 바탕으로 수행됨. |
| Temperature (온도) | "창의성 조절 노브" | 로짓을 `T`로 나눔; `T`→0은 탐욕적(greedy) 생성, `T`→∞는 균등(uniform) 분포를 의미함. |
| Top-p | "핵심 샘플링(Nucleus sampling)" | 합계가 `≥p`가 되는 가장 작은 토큰 집합으로 분포를 절단; 남은 집합에서 샘플링함. |
| Min-p | "Top-p보다 나은 방식" | `p ≥ min_p × max_p`인 토큰을 유지; 분포의 날카로움(sharpness)에 따라 컷오프를 적응시킴. |
| Speculative decoding (추측적 디코딩) | "초안 + 검증" | 가벼운 모델이 `N`개의 토큰을 제안하고, 큰 모델이 이를 병렬로 검증함. |
| Teacher forcing (교사 강요) | "훈련 트릭" | 훈련 중 모델의 예측값이 아닌 실제 이전 토큰을 입력으로 제공함. 모든 seq2seq LM의 표준 방식. |

## 추가 읽을거리 (Further Reading)

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 및 인컨텍스트 학습(in-context learning).
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 추측 디코딩(speculative decoding) 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 표준적인 인과적 언어 모델(causal-LM) 참조 코드.
