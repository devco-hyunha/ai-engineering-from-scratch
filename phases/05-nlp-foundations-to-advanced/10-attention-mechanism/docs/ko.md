# 어텐션 메커니즘 — 돌파구

> 디코더는 압축된 요약에 집중하는 것을 멈추고 전체 소스 문장을 보기 시작합니다. 이후 모든 내용은 어텐션과 엔지니어링의 결합입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 09강 (Sequence-to-Sequence Models)
**시간:** 약 45분

## 문제점

09강은 측정된 실패로 마무리되었습니다. 장난감 복사 작업(toy copy task)으로 훈련된 GRU 인코더-디코더는 길이 5에서 89%의 정확도를 기록하지만, 길이 80에서는 거의 무작위 수준으로 떨어집니다. 그 이유는 훈련 버그가 아니라 구조적 한계입니다. 인코더가 수집한 모든 정보는 하나의 고정 크기 은닉 상태(hidden state)에 담겨야 하며, 디코더는 그 외의 정보를 전혀 볼 수 없습니다.

Bahdanau, Cho, Bengio는 2014년에 세 줄짜리 해결책을 발표했습니다. 디코더에 최종 인코더 상태만 주는 대신, 모든 인코더 상태를 유지합니다. 각 디코더 단계에서 가중 평균을 계산하며, 가중치는 "디코더가 지금 인코더 위치 `i`를 얼마나 봐야 하는가?"를 나타냅니다. 이 가중 평균이 컨텍스트가 되며, 디코더 단계마다 변합니다.

이것이 전체 아이디어입니다. 트랜스포머는 이를 확장했습니다. 셀프 어텐션은 단일 시퀀스에 적용했습니다. 멀티 헤드 어텐션은 병렬로 실행했습니다. 그러나 2014년 버전은 이미 병목 현상을 깨뜨렸으며, 이를 습득하면 트랜스포머로의 전환은 개념적 문제가 아니라 엔지니어링 문제입니다.

## 개념

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

각 디코더 단계 `t`에서:

1. 이전 디코더 은닉 상태 `s_{t-1}`를 **쿼리**로 사용합니다.
2. 모든 인코더 은닉 상태 `h_1, ..., h_T`에 대해 점수를 매깁니다. 인코더 위치마다 하나의 스칼라 값이 생성됩니다.
3. 점수에 소프트맥스를 적용하여 합이 1이 되는 어텐션 가중치 `α_{t,1}, ..., α_{t,T}`를 얻습니다.
4. 컨텍스트 벡터 `c_t = Σ α_{t,i} * h_i`. 인코더 상태의 가중 평균입니다.
5. 디코더는 `c_t`와 이전 출력 토큰을 받아 다음 토큰을 생성합니다.

가중 평균이 핵심입니다. 디코더가 "Je"를 "I"로 번역해야 할 때, "Je"에 해당하는 인코더 상태에 높은 가중치를 부여하고 나머지는 낮게 부여합니다. "not"이 필요할 때는 "pas"에 높은 가중치를 부여합니다. 컨텍스트 벡터는 매 단계마다 형태를 바꿉니다.

## 형태 (모두가 겪는 어려움)

모든 어텐션 구현이 처음에 틀리는 지점입니다. 천천히 읽어 보세요.

| 항목 | 형태 | 비고 |
|-------|-------|-------|
| 인코더 은닉 상태 `H` | `(T_enc, d_h)` | BiLSTM인 경우, `d_h = 2 * d_hidden` |
| 디코더 은닉 상태 `s_{t-1}` | `(d_s,)` | 하나의 벡터 |
| 어텐션 점수 `e_{t,i}` | 스칼라 | 인코더 위치마다 하나 |
| 어텐션 가중치 `α_{t,i}` | 스칼라 | 모든 `i`에 대해 소프트맥스 적용 후 |
| 컨텍스트 벡터 `c_t` | `(d_h,)` | 인코더 상태와 동일한 형태 |

**Bahdanau (가산) 점수.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}`의 형태는 `(d_s,)`이고, `h_i`의 형태는 `(d_h,)`입니다.
- `W_a`의 형태는 `(d_attn, d_s)`입니다. `U_a`의 형태는 `(d_attn, d_h)`입니다.
- tanh 내부의 합은 `(d_attn,)`의 형태를 가집니다.
- `v_α`의 형태는 `(d_attn,)`입니다. `v_α`와의 내적은 스칼라로 축소됩니다. **이것이 `v_α`이 하는 일입니다.** 마법이 아닙니다. 어텐션 차원 벡터를 스칼라 점수로 변환하는 투영입니다.

**Luong (곱) 점수.** 세 가지 변형:

- `dot`: `e_{t,i} = s_t^T * h_i`. `d_s == d_h`가 필요합니다. 하드 제약 조건입니다. 인코더가 양방향인 경우 건너뛰세요.
- `general`: `W`의 형태가 `(d_s, d_h)`인 `e_{t,i} = s_t^T * W * h_i`. 동일한 차원 제약 조건을 제거합니다.
- `concat`: 본질적으로 Bahdanau 형태입니다. 앞의 두 가지가 더 저렴하므로 거의 사용되지 않습니다.

**언급할 가치가 있는 Bahdanau / Luong 함정 하나.** Bahdanau는 `s_{t-1}` (현재 단어를 생성하기 *전*의 디코더 상태)를 사용합니다. Luong은 `s_t` (*후의* 상태)를 사용합니다. 이 둘을 혼동하면 디버깅이 매우 어려운 미묘하게 잘못된 기울기가 생성됩니다. 하나의 논문을 선택하고 그 규칙을 따르세요.

```figure
attention-heatmap
```

## 구현하기

### 1단계: 가산 (Bahdanau) 어텐션

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

위 표와 형태를 대조해 보세요. `encoder_states`의 형태는 `(T_enc, d_h)`입니다. `projected_enc`의 형태는 `(T_enc, d_attn)`입니다. `projected_dec`의 형태는 `(d_attn,)`이며 브로드캐스트됩니다. `combined`의 형태는 `(T_enc, d_attn)`입니다. `scores`의 형태는 `(T_enc,)`입니다. `weights`의 형태는 `(T_enc,)`입니다. `context`의 형태는 `(d_h,)`입니다. 출시하세요.

### 2단계: Luong 점 및 일반

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

각각 세 줄입니다. Luong의 논문이 성공한 이유입니다. 대부분의 작업에서 정확도는 동일하고, 코드는 훨씬 적습니다.

### 3단계: 실제 수치 예제

세 개의 인코더 상태(대략 "cat", "sat", "mat")와 첫 번째 상태와 가장 잘 정렬되는 디코더 상태가 주어지면, 어텐션 분포는 위치 0에 집중됩니다. 디코더 상태가 마지막 상태와 정렬되도록 이동하면, 어텐션은 위치 2로 이동합니다. 컨텍스트 벡터가 이를 추적합니다.

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

첫 번째 행이 승리합니다. 그런 다음 디코더 상태를 세 번째 인코더 상태에 더 가깝게 이동하고 가중치가 이동하는 것을 관찰하세요. 이것이 전부입니다. 어텐션은 명시적인 정렬입니다.

### 4단계: 이것이 트랜스포머로 이어지는 다리인 이유

위의 언어를 Q/K/V로 번역하세요:

- **Query** = 디코더 상태 `s_{t-1}`
- **Key** = 인코더 상태 (점수를 매기는 대상)
- **Value** = 인코더 상태 (가중치를 부여하고 합산하는 대상)

고전적 어텐션에서는 키와 값이 같은 것입니다. 셀프 어텐션은 이를 분리합니다: K와 V에 대해 서로 다른 학습된 투영을 사용하여 시퀀스를 자체에 대해 쿼리할 수 있습니다. 멀티헤드 어텐션은 서로 다른 학습된 투영을 사용하여 병렬로 실행합니다. 트랜스포머는 전체 단계를 여러 번 스택하고 RNN을 제거합니다.

수식은 동일합니다. 형태는 동일합니다. Bahdanau 어텐션에서 스케일된 점곱 어텐션으로의 교육적 도약은 대부분 표기법입니다.

## 사용하기

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

이것은 트랜스포머 어텐션 레이어입니다. 5개 위치의 Query 배치, 10개 위치의 key/value 배치, 각각 128차원, 8개의 헤드. `output`는 새로운 컨텍스트 증강 쿼리입니다. `weights`는 시각화할 수 있는 5x10 정렬 매트릭스입니다.

### 고전적 어텐션이 여전히 중요한 경우

- 교육. 단일 헤드, 단일 레이어, RNN 기반 버전은 모든 개념을 명확하게 보여줍니다.
- 트랜스포머가 적합하지 않은 온디바이스 시퀀스 작업.
- 2014-2017년의 모든 논문. Bahdanau의 관례를 모르면 오독할 것입니다.
- MT에서의 세밀한 정렬 분석. 트랜스포머 모델에서도 원시 어텐션 가중치는 해석 가능성 도구이며, 이를 읽으려면 그 의미를 알아야 합니다.

### 어텐션 가중치를 설명으로 사용하는 함정

어텐션 가중치는 해석이 가능한 것처럼 보입니다. 위치 전체에 대해 합이 1이 되는 가중치이며, 플롯할 수 있고, 값이 높으면 "여기를 보았다"는 의미입니다. 리뷰어들은 이것을 좋아합니다.

그러나 실제 해석 가능성은 겉으로 보이는 것만큼 높지 않습니다. Jain과 Wallace (2019)는 일부 작업에서 어텐션 분포를 순열로 바꾸거나 임의의 대안으로 대체해도 모델 예측이 변하지 않는다는 것을 보였습니다. 어블레이션(ablation)이나 반사실적(counterfactual) 검증 없이 어텐션 가중치를 추론의 증거로 보고하지 마세요.

## 출시하기

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

## 연습 문제

1. **쉬움.** 인코더에서 패딩 토큰이 어텐션 가중치 0을 받도록 `softmax` 마스킹을 구현하세요. 가변 길이 시퀀스가 포함된 배치로 테스트해 보세요.
2. **중간.** Luong `general` 형태에 멀티헤드 어텐션을 추가하세요. `d_h`을 `n_heads` 그룹으로 나누고, 각 헤드별로 어텐션을 실행한 후 연결하세요. 단일 헤드 케이스가 이전 구현과 일치하는지 확인하세요.
3. **어려움.** 09강의 장난감 복사 작업에 Bahdanau 어텐션을 사용하는 GRU 인코더-디코더를 학습하세요. 시퀀스 길이에 따른 정확도를 플롯하세요. 어텐션이 없는 baseline과 비교하세요. 길이가 증가함에 따라 격차가 넓어지는 것을 볼 수 있으며, 이는 어텐션이 병목 현상을 해소함을 확인합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 어텐션 | 무언가를 보는 것 | 쿼리-키 유사성으로 계산된 가중치를 사용하여 값 시퀀스의 가중 평균을 구하는 것. |
| 쿼리, 키, 값 | QKV | 세 가지 투영: Q는 묻고, K는 매칭할 대상이며, V는 반환할 대상입니다. |
| 가산 어텐션 | Bahdanau | 피드포워드 점수: `v^T tanh(W q + U k)`. |
| 곱셈 어텐션 | Luong dot / general | 점수는 `q^T k` 또는 `q^T W k`입니다. 더 저렴하며, 대부분의 작업에서 정확도가 동일합니다. |
| 정렬 매트릭스 | 예쁜 그림 | `(T_dec, T_enc)` 그리드 형태의 어텐션 가중치. 모델이 무엇에 어텐션했는지 보기 위해 읽습니다. |

## 추가 읽기

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 논문.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 세 가지 점수 변형과 비교.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 해석 가능성에 대한 주의 사항.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch를 사용한 실행 가능한 워크스루.
