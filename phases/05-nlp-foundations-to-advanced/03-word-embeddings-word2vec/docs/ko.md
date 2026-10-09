# 단어 임베딩 — Word2Vec을 처음부터 구현하기

> 단어는 주변에 있는 단어로 알 수 있습니다. 그 아이디어에 얕은 신경망을 학습하면 기하학적 구조가 자연스럽게 드러납니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 3단계 · 03강 (역전파(Backpropagation)를 처음부터 구현하기)
**시간:** 약 75분

## 문제점

TF-IDF는 `dog`와 `puppy`가 서로 다른 단어임을 알지만, 거의 같은 의미라는 사실은 알지 못합니다. `dog`에 대해 학습한 분류기는 `puppy`에 대한 리뷰로 일반화되지 않습니다. 유의어를 나열하여 이 문제를 해결할 수 있지만, 희소한 용어, 도메인 전문 용어, 그리고 예상하지 못한 모든 언어에서는 실패합니다.

`dog`와 `puppy`가 공간상에서 가깝게 위치하고, `king - man + woman`가 `queen` 근처에 위치하며, `dog`에 대해 학습한 모델이 `puppy`에 대해 무료로 신호를 전달하는 표현을 원합니다.

Word2Vec은 그 공간을 제공했습니다. 2013년에 발표된 두 층 신경망과 조 단위 토큰 학습 실행이 그 결과입니다. 아키텍처는 거의 당혹스러울 정도로 단순합니다. 그 결과는 10년간 NLP를 재편했습니다.

## 개념

**분포 가설** (Firth, 1957): "단어는 주변에 있는 단어로 알 수 있습니다." 두 단어가 유사한 맥락에 나타나면, 그들은 유사한 것을 의미할 가능성이 높습니다.

Word2Vec은 그 아이디어를 활용하는 두 가지 변형이 있습니다.

- **Skip-gram.** 중심 단어가 주어지면 주변 단어를 예측합니다. `cat -> (the, sat, on)`는 윈도우 크기가 2입니다.
- **CBOW (연속 단어 주머니).** 주변 단어가 주어지면 중심 단어를 예측합니다. `(the, sat, on) -> cat`.

Skip-gram은 학습 속도가 느리지만 희소한 단어를 더 잘 처리합니다. 이것이 기본값이 되었습니다.

신경망은 비선형성이 없는 하나의 은닉층을 가지고 있습니다. 입력은 어휘에 대한 원-hot 벡터입니다. 출력은 어휘에 대한 소프트맥스(Softmax)입니다. 학습 후, 출력층을 버립니다. 은닉층 가중치가 임베딩(Embedding)입니다.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

핵심은 이겁니다. 10만 개의 단어를 대상으로 소프트맥스(Softmax)를 계산하는 것은 비용이 너무 높습니다. Word2Vec은 **음성 샘플링(negative sampling)**을 사용하여 이를 이진 분류 작업으로 바꿉니다. "이 컨텍스트 단어가 중심 단어 근처에 나타났는가, 예/아니오"를 예측하는 방식입니다. 전체 어휘에 대해 소프트맥스를 계산하는 대신, 학습 쌍마다 몇 개의 음성(non-co-occurring) 단어를 샘플링합니다.

```figure
word-vector-arithmetic
```

## 구현하기

### 1단계: 코퍼스로부터 학습 쌍 생성

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

윈도우 내의 모든 (중심, 컨텍스트) 쌍은 양성 학습 예제입니다.

### 2단계: 임베딩 테이블

두 개의 행렬이 있습니다. `W`는 중심 단어 임베딩 테이블(유지하는 테이블)입니다. `W'`는 컨텍스트 단어 테이블(종종 폐기되며, 때로는 `W`와 평균 내어 사용)입니다.

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

작은 랜덤 초기값을 사용합니다. 어휘 크기 10k와 차원 100은 현실적인 설정이며, 교육용으로는 어휘 50 x 차원 16만으로도 기하학적 구조를 확인하기에 충분합니다.

### 3단계: 음성 샘플링 목적 함수

각 양성 쌍 `(center, context)`에 대해, 어휘에서 `k`개의 랜덤 단어를 음성으로 샘플링합니다. 모델이 양성 쌍에서는 내적 `W[center] · W'[context]`이 높고, 음성 쌍에서는 낮도록 학습합니다.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

마법 같은 공식: 양성 쌍에 대한 로지스틱 손실(sigmoid가 1에 가깝기를 원함)과 음성 쌍에 대한 로지스틱 손실(sigmoid가 0에 가깝기를 원함)을 더합니다. 기울기는 두 테이블 모두로 전파됩니다. 전체 유도는 원 논문에서 확인할 수 있으며, 이해가 잘 안 되면 연필과 종이로 한 번 풀어보세요.

### 4단계: 장난감 코퍼스로 학습

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

대규모 코퍼스에서 충분한 에포크(Epoch)를 학습하면, 컨텍스트를 공유하는 단어들의 중심 임베딩이 유사해집니다. 장난감 코퍼스를 사용해서는 효과가 미미하게 나타나며, 수십억 개의 토큰을 사용해서는 극적으로 나타납니다.

### 5단계: 유추(trick) 기법

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

사전 학습된 300차원 Google News 벡터에서:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. 모델이 왕실(royalty)이 무엇인지 알기 때문이 아닙니다. 벡터 `(king - man)`가 "royal"과 유사한 무언가를 포착하고, 이를 `woman`에 더하면 왕실 여성 영역 근처에 위치하기 때문입니다.

## 사용하기

Word2Vec을 처음부터 작성하는 것은 교육 목적입니다. 실제 NLP에서는 `gensim`을 사용합니다.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

실제 작업에서는 Word2Vec을 직접 학습하는 경우가 거의 없습니다. 사전 학습된 벡터를 다운로드합니다.

- **GloVe** — 스탠퍼드의 동시 발생 행렬 분해 접근법. 50d, 100d, 200d, 300d 체크포인트. 일반적인 커버리지에 적합합니다. 04강에서 GloVe를 구체적으로 다룹니다.
- **fastText** — Facebook의 Word2Vec 확장으로 문자 n-gram을 임베딩합니다. 하위 단어를 조합하여 어휘 목록 밖(OOV) 단어를 처리합니다. 04강.
- **Google News의 사전 학습된 Word2Vec** — 300d, 300만 단어 어휘, 2013년 공개. 여전히 매일 다운로드됩니다.

### 2026년에도 Word2Vec이 승리하는 경우

- 경량 도메인 특화 검색. 노트북에서 한 시간 동안 의료 초록으로 학습하면, 일반 모델이 포착하지 못하는 특화 벡터를 얻을 수 있습니다.
- 유사성 기반 기능 엔지니어링. `gender_vector = mean(man - woman pairs)`. 다른 단어에서 이를 빼서 성중립 축을 얻습니다. 공정성 연구에서 여전히 사용됩니다.
- 해석 가능성. 100d는 PCA나 t-SNE로 플롯하여 클러스터가 형성되는 것을 실제로 볼 수 있을 만큼 작습니다.
- GPU 없이 디바이스에서 추론을 실행해야 하는 모든 곳. Word2Vec 조회는 단일 행 가져오기입니다.

### Word2Vec이 실패하는 지점

다의어 벽. `bank`는 하나의 벡터를 가집니다. `river bank`과 `financial bank`는 이를 공유합니다. `table` (스프레드시트 vs. 가구)도 이를 공유합니다. 다운스트림 분류기는 벡터로부터 의미를 구분할 수 없습니다.

맥락 임베딩 (ELMo, BERT, 이후의 모든 트랜스포머)은 주변 맥락에 기반하여 단어의 각 발생에 대해 다른 벡터를 생성함으로써 이 문제를 해결했습니다. 이것이 Word2Vec에서 BERT로의 도약입니다: 정적에서 맥락 기반으로. 7단계는 트랜스포머 부분을 다룹니다.

어휘 목록 밖(OOV) 문제가 다른 실패 지점입니다. 학습 데이터에 없다면 Word2Vec은 `Zoomer-approved`를 본 적이 없습니다. 폴백이 없습니다. fastText는 하위 단어 조합으로 이를 해결합니다 (04강).

## 출시하기

`outputs/skill-embedding-probe.md`로 저장:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 연습 문제

1. **쉬움.** 작은 코퍼스 (고양이와 개에 대한 20문장)로 학습 루프를 실행합니다. 200 에포크 후, `nearest(vocab, W, W[vocab["cat"]])`가 상위 3개 안에 `dog`를 반환하는지 확인합니다. 그렇지 않다면 에포크 수나 어휘를 늘리세요.
2. **중간.** 빈번한 단어의 서브샘플링을 추가하세요. 빈도가 `10^-5` 이상인 단어는 빈도에 비례하는 확률로 학습 쌍에서 드롭됩니다. 희소 단어 유사성에 미치는 영향을 측정하세요.
3. **난이도: 높음.** 20 Newsgroups 코퍼스로 모델을 학습하세요. 두 편향 축 `he - she`와 `doctor - nurse`을 계산하세요. 직업 단어를 두 축에 투영하세요. 편향 격차가 가장 큰 직업을 보고하세요. 이는 공정성 연구자들이 사용하는 탐지 방식입니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 단어 임베딩 | 단어를 벡터로 표현 | 문맥에서 학습된 밀집(low-dim, 보통 100-300) 표현입니다. |
| Skip-gram | Word2Vec 기법 | 중심 단어에서 문맥 단어를 예측합니다. CBOW보다 느리지만 희단어에 더 잘 작동합니다. |
| 음 샘플링 | 학습 단축 기법 | 전체 어휘에 대한 softmax를 `k`개의 랜덤 단어에 대한 이진 분류로 대체합니다. |
| 정적 임베딩 | 단어당 하나의 벡터 | 문맥과 무관하게 동일한 벡터를 사용합니다. 다의어에 실패합니다. |
| 문맥 임베딩 | 문맥에 민감한 벡터 | 주변 단어에 기반해 각 출현마다 다른 벡터를 생성합니다. 트랜스포머가 생성하는 결과입니다. |
| OOV | 어휘 외 단어 | 학습에서 본 적 없는 단어입니다. Word2Vec은 이런 단어를 위한 벡터를 생성할 수 없습니다. |

## 추가 읽기

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 음 샘플링 논문입니다. 짧고 읽기 쉽습니다.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 원 논문이 수학적으로 밀집하게 느껴진다면, 기울기 유도가 가장 명확합니다.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 실제로 작동하는 프로덕션 학습 설정입니다.
