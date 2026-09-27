# 어텐션 메커니즘 (Attention Mechanism) — 돌파구 (The Breakthrough)

> 디코더는 압축된 요약본을 뚫어지게 쳐다보는 것을 멈추고, 소스 전체를 보기 시작합니다. 이 이후의 모든 내용은 어텐션과 엔지니어링에 관한 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## 문제점 (The Problem)

레슨 09는 측정 가능한 실패로 끝났습니다. toy copy task로 학습된 GRU 인코더-디코더는 길이가 5일 때 89%의 정확도를 보이다가, 길이가 80이 되면 거의 찍는 수준(near-chance)으로 떨어집니다. 그 이유는 학습 버그가 아니라 구조적인 문제입니다: 인코더가 수집한 모든 정보가 하나의 고정된 크기의 은닉 상태(hidden state)에 담겨야 하며, 디코더는 그 외의 다른 것을 볼 수 없기 때문입니다.

Bahdanau, Cho, Bengio는 2014년에 단 세 줄로 해결할 수 있는 방법을 발표했습니다. 디코더에게 인코더의 마지막 상태만 주는 대신, 모든 인코더 상태를 유지합니다. 각 디코더 단계에서, '지금 디코더가 인코더의 `i`번째 위치를 얼마나 봐야 하는가?'를 나타내는 가중치를 사용하여 인코더 상태들의 가중 평균을 계산합니다. 그 가중 평균이 컨텍스트(context)이며, 이는 디코더의 매 단계마다 변합니다.

이것이 핵심 아이디어입니다. 트랜스포머(Transformers)는 이를 확장했습니다. 셀프 어텐션(Self-attention)은 이를 단일 시퀀스에 적용했습니다. 멀티 헤드 어텐션(Multi-head attention)은 이를 병렬로 실행합니다. 하지만 2014년 버전이 이미 병목 현상을 해결했으므로, 일단 이를 이해하고 나면 트랜스포머로 넘어가는 과정은 개념적인 변화가 아니라 엔지니어링의 영역입니다.

## 개념 (The Concept)

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

각 디코더 단계 `t`에서:

1. 이전 디코더 은닉 상태 `s_{t-1}`을 **쿼리(query)**로 사용합니다.
2. 모든 인코더 은닉 상태 `h_1, ..., h_T`와 비교하여 점수를 매깁니다. 인코더 위치당 하나의 스칼라 값이 생성됩니다.
3. 점수에 소프트맥스(Softmax)를 적용하여 합이 1이 되는 어텐션 가중치 `α_{t,1}, ..., α_{t,T}`를 얻습니다.
4. 컨텍스트 벡터 `c_t = Σ α_{t,i} * h_i`. 인코더 상태들의 가중 평균입니다.
5. 디코더는 `c_t`와 이전 출력 토큰을 받아 다음 토큰을 생성합니다.

가중 평균이 핵심입니다. 디코더가 "Je"를 "I"로 번역해야 할 때, "Je"에 해당하는 인코더 상태의 가중치는 높이고 나머지는 낮게 설정합니다. "not"이 필요할 때는 "pas"의 가중치를 높입니다. 컨텍스트 벡터는 매 단계를 재구성합니다.

## 형태 (Shapes) — 모두를 괴롭히는 부분

모든 어텐션 구현이 처음 시도할 때 실수하는 부분입니다. 천천히 읽어 보세요.

| 항목 | 형태 (Shape) | 비고 |
|-------|-------|-------|
| 인코더 은닉 상태 `H` | `(T_enc, d_h)` | BiLSTM인 경우, `d_h = 2 * d_hidden` |
| 디코더 은닉 상태 `s_{t-1}` | `(d_s,)` | 하나의 벡터 |
| 어텐션 점수 `e_{t,i}` | 스칼라 | 인코더 위치당 하나 |
| 어텐션 가중치 `α_{t,i}` | 스칼라 | 모든 `i`에 대해 소프트맥스를 적용한 후 |
| 컨텍스트 벡터 `c_t` | `(d_h,)` | 인코더 상태와 동일한 형태 |

**Bahdanau (additive, 가산형) 점수.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}`은 `(d_s,)` 형태를, `h_i`는 `(d_h,)` 형태를 가집니다.
- `W_a`는 `(d_attn, d_s)` 형태를, `U_a`는 `(d_attn, d_h)` 형태를 가집니다.
- tanh 내부의 합은 `(d_attn,)` 형태를 가집니다.
- `v_α`는 `(d_attn,)` 형태를 가집니다. `v_α`와의 내적은 스칼라로 축소됩니다. **이것이 `v_α`가 하는 역할입니다.** 마법이 아닙니다. 어텐션 차원의 벡터를 스칼라 점수로 변환하는 투영(projection)입니다.

**Luong (multiplicative, 곱셈형) 점수.** 세 가지 변형이 있습니다:

- `dot`: `e_{t,i} = s_t^T * h_i`. `d_s == d_h`가 필요합니다. 엄격한 제약 조건입니다. 인코더가 양방향(bidirectional)이라면 건너뛰세요.
- `general`: `e_{t,i} = s_t^T * W * h_i`이며 `W`의 형태는 `(d_s, d_h)`입니다. 차원이 같아야 한다는 제약을 제거합니다.
- `concat`: 본질적으로 Bahdanau 형태입니다. 앞의 두 방식이 더 저렴하기 때문에 거의 사용되지 않습니다.

**주의해야 할 Bahdanau / Luong의 차이점(gotcha)이 하나 있습니다.** Bahdanau는 `s_{t-1}`(현재 단어를 생성하기 *전*의 디코더 상태)을 사용합니다. Luong는 `s_t`(생성 *후*의 상태)를 사용합니다. 이 둘을 혼동하면 디버깅하기 매우 어려운 미세하게 잘못된 그래디언트(gradient)가 발생합니다. 논문 하나를 선택하여 그 관례를 따르세요.

```figure
attention-heatmap
```

## 구현하기 (Build It)

### 1단계: 가산형 (additive, Bahdanau) 어텐션

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

위의 표와 형태를 대조해 보세요. `encoder_states`는 `(T_enc, d_h)` 형태를 가집니다. `projected_enc`는 `(T_enc, d_attn)` 형태를 가집니다. `projected_dec`는 `(d_attn,)` 형태이며 브로드캐스팅됩니다. `combined`는 `(T_enc, d_attn)` 형태를 가집니다. `scores`는 `(T_enc,)` 형태를 가집니다. `weights`는 `(T_enc,)` 형태를 가집니다. `context`는 `(d_h,)` 형태를 가집니다. 이제 완료되었습니다.

### 2단계: Luong의 dot 및 general 방식

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

각각 세 줄이면 충분합니다. 이것이 Luong의 논문이 주목받은 이유입니다. 대부분의 작업에서 동일한 정확도를 유지하면서 코드 양은 훨씬 적습니다.

### 3단계: 수치적 예시

세 개의 인코더 상태(대략 "cat", "sat", "mat")와 첫 번째 상태와 가장 잘 일치하는 디코더 상태가 주어지면, 어텐션 분포는 0번 위치에 집중됩니다. 디코더 상태가 마지막 상태와 일치하도록 이동하면, 어텐션은 2번 위치로 이동합니다. 컨텍스트 벡터가 이를 추적합니다.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

첫 번째 행이 승리합니다. 그런 다음 디코더 상태를 세 번째 인코더 상태에 더 가깝게 이동시키며 가중치가 변하는 것을 관찰해 보세요. 그게 전부입니다. 어텐션은 명시적인 정렬(alignment)입니다.

### 4단계: 이것이 왜 트랜스포머로 가는 가교인가

위의 언어를 Q/K/V로 변환해 보세요:

- **Query** = 디코더 상태 `s_{t-1}`
- **Key** = 인코더 상태 (점수를 매기는 대상)
- **Value** = 인코더 상태 (가중치를 곱하고 합산하는 대상)

고전적인 어텐션에서 Key와 Value는 동일한 것입니다. 셀프 어텐션은 이를 분리합니다. 즉, K와 V에 서로 다른 학습된 투영(projection)을 사용하여 시퀀스 자신에 대해 쿼리를 수행할 수 있습니다. 멀티 헤드 어텐션은 이를 서로 다른 학습된 투영과 함께 병렬로 실행합니다. 트랜스포머는 이 전체 단계를 여러 번 쌓고 RNN을 제거합니다.

수식은 같습니다. 형태도 같습니다. Bahdanau 어텐션에서 스케일드 닷-프로덕트(scaled dot-product) 어텐션으로 넘어가는 교육적 도약은 대부분 표기법의 차이일 뿐입니다.

## 사용하기 (Use It)

PyTorch와 TensorFlow는 어텐션을 직접 제공합니다.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

이것이 트랜스포머 어텐션 레이어입니다. 쿼리는 5개 위치의 배치, 키/값은 10개 위치의 배치, 각각 128차원이며 8개의 헤드를 가집니다. `output`은 컨텍스트가 보강된 새로운 쿼리입니다. `weights`는 시각화할 수 있는 5x10 정렬 행렬입니다.

### 고전적 어텐션이 여전히 중요한 경우

- 교육적 목적. 싱글 헤드, 싱글 레이어, RNN 기반 버전은 모든 개념을 명확하게 보여줍니다.
- 트랜스포머를 적용하기 어려운 온디바이스(on-device) 시퀀스 작업.
- 2014-2017년 사이의 논문들. Bahdanau의 관례를 모르면 잘못 읽을 수 있습니다.
- 기계 번역(MT)에서의 미세한 정렬 분석. 가공되지 않은 어텐션 가중치는 트랜스포머 모델에서도 해석 가능성 도구로 사용되며, 이를 읽으려면 그것이 무엇인지 알아야 합니다.

### 어텐션 가중치를 설명으로 사용하는 함정

어텐션 가중치는 해석 가능해 보입니다. 위치에 따라 합이 1이 되는 가중치이며, 그래프로 그릴 수 있고, 값이 높으면 "이 부분을 보았다"는 뜻입니다. 리뷰어들은 이를 좋아합니다.

하지만 보이는 것만큼 해석 가능하지는 않습니다. Jain과 Wallace(2019)는 일부 작업에서 어텐션 분포를 순열(permute)하거나 임의의 대안으로 교체해도 모델의 예측이 바뀌지 않을 수 있음을 보여주었습니다. 어블레이션(ablation)이나 반사실적(counterfactual) 검증 없이 어텐션 가중치를 추론의 증거로 보고하지 마세요.

## 완성하기 (Ship It)

`outputs/prompt-attention-shapes.md`로 저장하세요:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 연습 문제 (Exercises)

1. **쉬움.** 인코더의 패딩 토큰이 어텐션 가중치 0을 갖도록 `softmax` 마스킹을 구현해 보세요. 가변 길이 시퀀스가 포함된 배치에서 테스트해 보세요.
2. **중간.** Luong의 `general` 방식에 멀티 헤드 어텐션을 추가해 보세요. `d_h`를 `n_heads` 그룹으로 나누고, 헤드별로 어텐션을 실행한 뒤 결합(concatenate)합니다. 싱글 헤드 케이스가 이전 구현과 일치하는지 확인해 보세요.
3. **어려움.** 레슨 09의 toy copy task에서 Bahdanau 어텐션을 사용하는 GRU 인코더-디코더를 학습시켜 보세요. 시퀀스 길이에 따른 정확도를 그래프로 그려 보세요. 어텐션이 없는 베이스라인과 비교해 보세요. 길이가 길어질수록 격차가 벌어지는 것을 볼 수 있으며, 이는 어텐션이 병목 현상을 해결함을 확인시켜 줍니다.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Attention | 무언가를 보는 것 | 쿼리-키 유사도에서 계산된 가중치를 사용하여 값(value) 시퀀스의 가중 평균을 구하는 것. |
| Query, Key, Value | QKV | 세 가지 투영: Q는 질문하고, K는 매칭 대상이며, V는 반환할 값입니다. |
| Additive attention | Bahdanau | 피드포워드 점수: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | 점수는 `q^T k` 또는 `q^T W k`입니다. 더 저렴하며 대부분의 작업에서 동일한 정확도를 보입니다. |
| Alignment matrix | 예쁜 그림 | `(T_dec, T_enc)` 그리드 형태의 어텐션 가중치입니다. 모델이 무엇에 집중했는지 확인하기 위해 읽어 보세요. |

## 추가 읽을거리 (Further Reading)

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 논문 원문.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 세 가지 점수 변형과 그 비교.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 해석 가능성에 대한 주의사항.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch를 사용한 실행 가능한 단계별 안내.
