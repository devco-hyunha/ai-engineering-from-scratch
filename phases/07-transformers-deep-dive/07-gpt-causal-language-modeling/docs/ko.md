# GPT — 인과적 언어 모델링

> BERT는 양쪽을 모두 봅니다. GPT는 과거만 봅니다. 삼각형 마스크는 현대 AI에서 가장 중요한 한 줄의 코드입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 02강 (셀프 어텐션), 7단계 · 05강 (전체 트랜스포머), 7단계 · 06강 (BERT)
**시간:** 약 75분

## 문제점

언어 모델은 한 가지 질문에 답합니다: 첫 `t-1` 토큰이 주어졌을 때, 토큰 `t`에 대한 확률 분포는 무엇일까요? 이 신호(다음 토큰 예측)로 학습하면, 한 번에 한 토큰씩 임의의 텍스트를 생성할 수 있는 모델을 얻을 수 있습니다.

전체 시퀀스를 병렬로 엔드투엔드 학습하려면, 각 위치의 예측이 이전 위치에만 의존해야 합니다. 그렇지 않으면 모델은 정답을 보며 쉽게 속임수를 쓰게 됩니다.

인과 마스크가 이 역할을 수행합니다. 이는 소프트맥스 전에 어텐션 점수에 더하는 `-inf` 값의 상삼각 행렬입니다. 소프트맥스 이후, 해당 위치는 0이 됩니다. 각 위치는 자기 자신과 이전 위치에만 어텐션할 수 있습니다. 그리고 전체 시퀀스에 한 번만 적용하므로, 한 번의 순전파로 N개의 병렬 다음 토큰 예측을 얻을 수 있습니다.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — 이들은 모두 동일한 핵심 루프를 가진 디코더 전용 인과 트랜스포머입니다. 이들을 구분하는 것은 데이터 품질, 규모, 아키텍처 개선, 그리고 사후 학습(지도 미세 조정 (SFT)(SFT (Supervised Fine-Tuning)), RLHF (인간 피드백 기반 강화 학습)(RLHF (Reinforcement Learning from Human Feedback)), DPO (직접 선호 최적화)(DPO (Direct Preference Optimization)) 및 그 후속 기술들)입니다.

## 개념

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 마스크

길이 `N`인 시퀀스가 주어지면, `N × N` 행렬을 만드세요:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

소프트맥스 전에 원시 어텐션 점수에 `M`를 더하세요. `exp(-inf) = 0`이므로, 마스킹된 위치는 가중치에 기여하지 않습니다. 어텐션 행렬의 각 행은 이전 위치들에 대한 확률 분포입니다.

구현 비용: `torch.tril()` 호출 한 번. 계산 시간: 나노초. 분야에 미치는 영향: 모든 것.

### 삼각형이 유래한 곳

마스크는 보통 어텐션에 붙여진 패치처럼 제시됩니다. 유도를 반대 방향으로 해 보세요. 그러면 더 이상 신비롭지 않게 됩니다. 어텐션은 접두어 평균의 세 번째 정제이며, 삼각형은 그 평균의 루프 경계를 행렬로 표현한 것입니다.

**1단계 — 접두어 평균.** 시퀀스에 대한 가장 단순한 인과적 요약입니다. 위치 `i`는 위치 `0…i`의 평균이 됩니다. 루프로 표현하면 `out[i] = X[:i+1].mean(0)`입니다. 동일한 연산은 하나의 행렬 곱셈으로 수행됩니다. 1로 채워진 하삼각 행렬을 취하고, 각 행을 그 개수로 나눈 뒤 곱합니다.

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

`A`의 `i` 행은 `[1/(i+1), …, 1/(i+1), 0, …, 0]`입니다. 대각선 위의 0들은 인과성을 나타냅니다. 미래에 대한 것은 마스킹되지 않았습니다. 미래는 애초에 합에 포함되지 않았습니다.

**2단계 — 학습된 가중치.** 균일한 평균은 모든 과거 토큰을 동일하게 관련 있는 것으로 취급합니다. 1들을 학습된 점수 행렬 `S`으로 대체합니다. 이제 행은 구성상 더 이상 합이 1이 아니므로, 개수로 나누는 대신 각 행을 softmax로 정규화합니다. Softmax는 정확히 0을 출력하지 않으며, 이는 인과성을 깨뜨립니다. 단, 미래 점수가 `-inf`로 입력되는 경우를 제외합니다. `exp(-inf) = 0`이기 때문입니다.

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

동일한 삼각형, 동일한 행 확률 행렬, 동일한 하나의 행렬 곱셈. `-inf` 마스크는 새로운 메커니즘이 아닙니다. 이는 1단계의 0 항목을 softmax의 입력 영역으로 변환한 것입니다.

**3단계 — 콘텐츠 의존 가중치.** 2단계에서는 `S`이 학습 후 고정됩니다. 토큰이 무엇을 말하든 위치 7은 항상 위치 3을 동일하게 가중합니다. 점수가 토큰 자체에 의존하도록 하세요. `S = Q @ K.T / sqrt(d_k)`. 다른 것은 변하지 않습니다. 마스크, softmax, 행렬 곱 — 동일합니다.

세 단계, 하나의 불변 조건: 하삼각 행 확률 행렬이 시퀀스와 곱해집니다. 균일한 평균, 학습된 정적 가중치, 콘텐츠 의존 가중치. 마스크는 어텐션에 추가된 적이 없습니다. 평균에서 살아남은 것입니다.

```figure
mask-derivation
```

### 병렬 학습, 직렬 추론

학습: 전체 `(N, d_model)` 시퀀스를 한 번에 순전파하고, N개의 교차 엔트로피 손실(각 위치당 하나)을 계산한 뒤, 합산하고 역전파합니다. 시퀀스를 따라 병렬화됩니다. 이것이 GPT 학습이 확장되는 이유입니다. 한 번의 GPU 패스로 배치에서 100만 토큰을 처리합니다.

추론(Inference): 토큰을 하나씩 생성합니다. `[t1, t2, t3]`을 입력하면 `t4`을 얻습니다. `[t1, t2, t3, t4]`을 입력하면 `t5`을 얻습니다. `[t1, t2, t3, t4, t5]`을 입력하면 `t6`을 얻습니다. KV 캐시(KV Cache)(12강)는 `t1…tn`의 은닉 상태를 저장하므로 매 단계마다 재계산하지 않아도 됩니다. 그러나 추론 시의 순차적 깊이는 출력 길이에 해당합니다. 이것이 자기회귀(Autoregressive) 비용이며, 모든 LLM (대규모 언어 모델)(LLM (Large Language Model))에서 디코딩이 지연 병목의 원인인 이유입니다.

### 손실(Loss) — 한 칸 이동(shift-by-one)

토큰 `[t1, t2, t3, t4]`이 주어졌을 때:

- 입력: `[t1, t2, t3]`
- 타겟: `[t2, t3, t4]`

각 위치 `i`에 대해 `-log P(target_i | inputs[:i+1])`을 계산합니다. 합산합니다. 이것이 전체 시퀀스에 대한 교차 엔트로피(Cross-Entropy)입니다.

들어본 모든 트랜스포머(Transformer) 언어 모델은 이 손실(Loss Function)로 학습합니다. 사전 학습, 미세 조정(Fine-tuning), SFT (지도 미세 조정)(SFT (Supervised Fine-Tuning)) — 손실은 동일하고 데이터만 다릅니다.

### 디코딩 전략(Decoding Strategy)

학습 후, 샘플링 선택은 사람들이 생각하는 것보다 더 중요합니다.

| 방법 | 동작 | 사용 시점 |
|--------|--------------|-------------|
| Greedy | 매 단계 Argmax | 결정론적 작업, 코드 완성 |
| 온도(Temperature) | 로짓(Logits)을 T로 나누고 샘플링 | 창의적 작업, T가 높을수록 다양성 증가 |
| Top-k 샘플링(Top-k Sampling) | 상위 k개 토큰(Token)에서만 샘플링 | 낮은 확률의 꼬리 제거 |
| Top-p (핵 샘플링(Top-p)) | 누적 확률이 p 이상인 최소 집합에서 샘플링 | 2020년 이후 기본값; 분포 형태에 적응 |
| Min-p | `p > min_p * max_p`인 토큰 유지 | 2024년 이후; top-p보다 긴 꼬리를 더 잘 거부 |
| 추론적 디코딩(Speculative Decoding) | 초안 모델이 N개 토큰을 제안하고 큰 모델이 검증 | 동일한 품질에서 2–3배 지연 감소 |

2026년, 오픈 웨이트 모델에 대해 min-p + 온도(Temperature) 0.7은 합리적인 기본값입니다. 추론적 디코딩(Speculative Decoding)은 모든 프로덕션 추론 스택의 필수 조건입니다.

### "GPT 레시피"가 작동한 이유

1. **디코더 전용(Decoder-only).** 인코더(Encoder) 오버헤드 없음. 각 레이어마다 어텐션(Attention) + FFN 한 번의 패스.
2. **스케일링.** 124M → 1.5B → 175B → 수조. Chinchilla 스케일링 법칙(13강)은 컴퓨팅을 어떻게 사용해야 하는지 알려줍니다.
3. **인컨텍스트 학습(In-Context Learning).** 6B–13B 부근에서 등장. 모델은 미세 조정(Fine-tuning) 없이 소수 예시(Few-Shot)를 따를 수 있습니다.
4. **RLHF (인간 피드백 기반 강화 학습)(RLHF (Reinforcement Learning from Human Feedback)).** 인간 선호에 대한 사후 학습이 원시 사전 학습된 텍스트를 채팅 어시스턴트로 변환했습니다.
5. **사전 정규화 + RoPE + SwiGLU.** 대규모에서도 안정적인 학습이 가능합니다.

GPT-2 이후 핵심 아키텍처는 크게 변하지 않았습니다. 흥미로운 발전은 모두 데이터, 규모, 그리고 학습 후(post-training) 단계에서 이루어졌습니다.

```figure
causal-mask
```

## 구현하기

### 1단계: 인과적 마스크(causal mask)

`code/main.py`를 참고해 보세요. 한 줄 코드로 구현할 수 있습니다:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

소프트맥스 전에 어텐션 점수에 추가합니다. 이것이 메커니즘의 전부입니다.

### 2단계: 2층 GPT 스타일 모델

두 개의 디코더 블록(마스크된 셀프 어텐션 + FFN, 교차 어텐션 없음)을 쌓습니다. 토큰 임베딩, 위치 인코딩, 그리고 언임베딩(토큰 임베딩 행렬과 연결(tied) — GPT-2 이후의 표준적인 기법)을 추가합니다.

### 3단계: 엔드투엔드 다음 토큰 예측

20개 토큰의 장난감 어휘(vocab)에서 모든 위치에 로짓(Logits)을 생성합니다. 한 칸 이동(shift-by-one)한 타겟에 대해 교차 엔트로피(Cross-Entropy) 손실을 계산합니다. 기울기는 계산하지 않습니다 — 이는 순전파(forward-pass) 무결성 체크입니다.

### 4단계: 샘플링

그리디(greedy), 온도(Temperature), top-k, top-p, min-p를 구현합니다. 고정된 프롬프트에서 각각 실행하고 출력 결과를 비교해 보세요. 샘플링 함수는 10줄이면 됩니다.

## 사용하기

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

내부적으로, `generate()`는 순전파를 실행하고, 마지막 위치의 로짓을 가져와, 다음 토큰을 샘플링하고, 이를 추가한 후 반복합니다. 모든 프로덕션 LLM 추론 스택(vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX)은 동일한 루프를 무거운 최적화(배치 프리필, 연속 배치, KV 캐시 페이징, 추론적 디코딩)와 함께 구현합니다.

**GPT vs BERT, 한 줄씩:** GPT는 `P(x_t | x_{<t})`를 예측합니다. BERT는 `P(x_masked | x_unmasked)`를 예측합니다. 손실(loss)은 모델이 생성할 수 있는지를 결정합니다.

## 출시하기

`outputs/skill-sampling-tuner.md`를 참고해 보세요. 이 스킬은 새로운 생성 작업에 대한 샘플링 매개변수를 선택하고, 결정적 디코딩(deterministic decoding)이 필요한 경우를 표시합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고, 소프트맥스 이후 인과적 어텐션 행렬이 하삼각(lower-triangular)인지 확인하세요. 체크해 보세요: 3번째 행은 0–3번째 열에만 가중치가 있어야 합니다.
2. **중간.** 너비(width)가 4인 빔 서치(beam search)를 구현하세요. 짧은 프롬프트 10개에서 beam-4와 그리디(greedy)의 퍼플렉시티(perplexity)를 비교하세요. 빔이 항상 이길까요? (힌트: 번역에서는 보통 이기지만, 개방형 채팅에서는 그렇지 않습니다.)
3. **난이도: 상.** 추론적 디코딩(Speculative Decoding)을 구현하세요: 2층 모델을 초안(draft)으로, 6층 모델을 검증자(verifier)로 사용하세요. 길이 64인 완성(completions) 100개에 대해 실시간(wall-clock) 속도 향상을 측정하세요. 출력 결과가 검증자의 탐욕(greedy) 디코딩과 일치하는지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 인과 마스크(Causal mask) | "삼각형" | 어텐션 점수에 더하는 상삼각 `-inf` 행렬로, 위치 `i`가 위치 `≤ i`만 볼 수 있도록 합니다. |
| 다음 토큰 예측 | "손실 함수" | 모든 위치에서 모델의 분포와 실제 다음 토큰 간의 교차 엔트로피(Cross-Entropy)입니다. |
| 자기회귀(Autoregressive) | "한 번에 하나씩 생성" | 출력을 입력으로 되돌려 공급합니다. 병렬 처리는 학습 중에만 가능하고, 생성 중에는 불가능합니다. |
| 로짓(Logits) | "소프트맥스 이전 점수" | LM 헤드(LM head)의 원시 출력으로, 소프트맥스(Softmax) 적용 전 상태입니다. 샘플링은 이 값에 대해 수행됩니다. |
| 온도(Temperature) | "창의성 조절旋钮" | 로짓을 T로 나눕니다. T→0은 탐욕(greedy) 디코딩, T→∞는 균일(uniform) 샘플링입니다. |
| Top-p | "핵 샘플링(Top-p)" | 합이 ≥p가 되는 가장 작은 집합으로 분포를 잘라내고, 남은 부분에서 샘플링합니다. |
| Min-p | "Top-p보다 더 좋음" | `p ≥ min_p × max_p`인 토큰을 유지합니다. 분포의 날카로움(sharpness)에 따라 컷오프를 적응적으로 조정합니다. |
| 추론적 디코딩(Speculative Decoding) | "초안 + 검증" | 값싼 모델이 N개의 토큰을 제안하고, 큰 모델이 병렬로 검증합니다. |
| 티처 포싱(Teacher forcing) | "학습 트릭" | 학습 중에 모델의 예측이 아닌 실제 이전 토큰을 공급합니다. 모든 시퀀스 투 시퀀스(seq2seq) LM의 표준입니다. |

## 추가 읽기

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 및 인컨텍스트 학습(In-Context Learning).
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 추론적 디코딩 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 표준 인과 LM(causal-LM) 참조 코드.
