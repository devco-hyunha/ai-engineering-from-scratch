# 단어 임베딩 — Word2Vec 처음부터 만들기 (Word Embeddings — Word2Vec from Scratch)

> 단어는 함께 나타나는 친구들로 정의됩니다. 그 아이디어로 얕은 신경망을 학습하면 기하학이 따라옵니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 문제 (The Problem)

TF-IDF는 `dog`와 `puppy`가 다른 단어라는 것은 압니다. 거의 같은 의미라는 것은 모릅니다. `dog`로 학습한 분류기는 `puppy`에 대한 리뷰로 일반화하지 못합니다. 동의어 목록으로 메울 수는 있지만, 희귀 용어·도메인 전문어·예측하지 못한 모든 언어에서는 실패합니다.

원하는 표현은 `dog`와 `puppy`가 공간에서 가깝게 놓이는 것입니다. `king - man + woman`이 `queen` 근처에 떨어지는 것입니다. `dog`로 학습한 모델이 `puppy`에도 신호를 공짜로 전달하는 것입니다.

Word2Vec이 그 공간을 주었습니다. 2층 신경망, 조(兆) 토큰 학습 실행, 2013년 발표. 아키텍처는 거의 창피할 정도로 단순합니다. 결과는 NLP를 10년간 바꿨습니다.

## 개념 (The Concept)

**분포 가설(Distributional hypothesis)** (Firth, 1957): "단어는 함께 나타나는 친구들로 알 수 있다." 두 단어가 비슷한 문맥에 나타나면, 비슷한 의미일 가능성이 큽니다.

Word2Vec에는 그 아이디어를 쓰는 두 가지 변형이 있습니다.

- **Skip-gram.** 중심 단어가 주어지면 주변 단어를 예측합니다. 윈도우 크기 2일 때 `cat -> (the, sat, on)`.
- **CBOW (continuous bag of words).** 주변 단어가 주어지면 중심을 예측합니다. `(the, sat, on) -> cat`.

Skip-gram은 학습이 더 느리지만 희귀 단어를 더 잘 다룹니다. 기본값이 되었습니다.

네트워크는 비선형성이 없는 은닉층 하나입니다. 입력은 어휘에 대한 원-핫 벡터입니다. 출력은 어휘에 대한 소프트맥스입니다. 학습 후 출력층을 버립니다. 은닉층 가중치가 임베딩입니다.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

트릭: 10만 단어에 대한 소프트맥스는 비용이 너무 큽니다. Word2Vec은 **네거티브 샘플링(negative sampling)**으로 이를 이진 분류 작업으로 바꿉니다. "이 문맥 단어가 이 중심 단어 근처에 나타났는가, 예 또는 아니오"를 예측합니다. 전체 어휘에 소프트맥스를 계산하는 대신, 학습 쌍마다 음의(공출현하지 않은) 단어를 소수 샘플링합니다.

```figure
word-vector-arithmetic
```

## 직접 만들기 (Build It)

### 1단계: 코퍼스에서 학습 쌍 만들기

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

윈도우 안의 모든 (중심, 문맥) 쌍이 양의 학습 예제입니다.

### 2단계: 임베딩 테이블

행렬 두 개. `W`는 중심 단어 임베딩 테이블(유지하는 쪽)입니다. `W'`는 문맥 단어 테이블(보통 버리고, 때로 `W`와 평균)입니다.

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

작은 랜덤 초기화. 어휘 1만·차원 100이 현실적이고, 교육용으로는 어휘 50 × 차원 16이면 기하학을 보기에 충분합니다.

### 3단계: 네거티브 샘플링 목적함수

각 양의 쌍 `(center, context)`에 대해 어휘에서 음수로 `k`개 단어를 샘플링합니다. `W[center] · W'[context]` 내적이 양수에서는 높고 음수에서는 낮도록 학습합니다.

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

핵심 공식: 양의 쌍에 대한 로지스틱 손실(시그모이드를 1 근처로) + 음의 쌍에 대한 로지스틱 손실(시그모이드를 0 근처로). 기울기는 두 테이블 모두로 흐릅니다. 전체 유도는 원 논문에 있습니다. 남기고 싶으면 연필과 종이로 한 번 따라가 보세요.

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

큰 코퍼스에서 에포크를 충분히 돌리면, 문맥을 공유하는 단어들은 비슷한 중심 임베딩을 갖습니다. 장난감 코퍼스에서는 효과가 희미하게 보이고, 수십억 토큰에서는 극적으로 보입니다.

### 5단계: 유추 트릭

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

`king - man + woman = queen`. 모델이 왕족이 무엇인지 알아서가 아닙니다. 벡터 `(king - man)`이 "왕족" 같은 무언가를 담고, 이를 `woman`에 더하면 왕족-여성 영역 근처에 떨어지기 때문입니다.

## 활용하기 (Use It)

Word2Vec을 처음부터 쓰는 것은 교육입니다. 프로덕션 NLP는 `gensim`을 씁니다.

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

실무에서는 Word2Vec을 직접 학습하는 일이 거의 없습니다. 사전 학습 벡터를 다운로드합니다.

- **GloVe** — Stanford의 공출현 행렬 분해 접근. 50d, 100d, 200d, 300d 체크포인트. 일반 커버리지가 좋습니다. 레슨 04에서 GloVe를 다룹니다.
- **fastText** — Facebook의 Word2Vec 확장으로 문자 n-그램을 임베딩합니다. 서브워드를 합성해 어휘 밖(out-of-vocabulary) 단어를 다룹니다. 레슨 04.
- **Google News 사전 학습 Word2Vec** — 300d, 어휘 300만, 2013년 발표. 지금도 매일 다운로드됩니다.

### 2026년에도 Word2Vec이 이기는 경우

- 가벼운 도메인 특화 검색. 노트북에서 한 시간 안에 의학 초록으로 학습하면, 일반 모델이 잡지 못하는 전문 벡터를 얻습니다.
- 유추 스타일 특징 공학. `gender_vector = mean(man - woman pairs)`. 다른 단어에서 빼서 성별 중립 축을 얻습니다. 공정성 연구에서 여전히 쓰입니다.
- 해석 가능성. 100d는 PCA나 t-SNE로 그려 클러스터가 실제로 생기는 것을 볼 수 있을 만큼 작습니다.
- GPU 없이 온디바이스에서 추론해야 하는 곳. Word2Vec 조회는 한 행 fetch입니다.

### Word2Vec이 실패하는 곳

다의어 벽. `bank`는 벡터 하나입니다. `river bank`와 `financial bank`가 공유합니다. `table`(스프레드시트 vs. 가구)도 공유합니다. 하류 분류기는 벡터만으로 의미를 구분할 수 없습니다.

문맥 임베딩(ELMo, BERT, 이후의 모든 트랜스포머)은 주변 문맥에 따라 단어 출현마다 다른 벡터를 만들어 이를 해결했습니다. Word2Vec에서 BERT로의 도약: 정적에서 문맥적으로. Phase 7이 트랜스포머 절반을 다룹니다.

다른 실패는 어휘 밖(out-of-vocabulary) 문제입니다. 학습 데이터에 없으면 Word2Vec은 `Zoomer-approved`를 본 적이 없습니다. 대안이 없습니다. fastText는 서브워드 합성으로 이를 고칩니다(레슨 04).

## 배포하기 (Ship It)

`outputs/skill-embedding-probe.md`로 저장하세요:

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

## 연습 문제 (Exercises)

1. **Easy.** 고양이·개에 대한 짧은 코퍼스(문장 20개)로 학습 루프를 돌리세요. 200 에포크 후 `nearest(vocab, W, W[vocab["cat"]])`가 top 3에 `dog`를 반환하는지 확인하세요. 아니면 에포크나 어휘를 늘리세요.
2. **Medium.** 빈도 높은 단어의 서브샘플링을 추가하세요. 빈도가 `10^-5` 초과인 단어는 빈도에 비례한 확률로 학습 쌍에서 제거됩니다. 희귀 단어 유사도에 미치는 영향을 측정하세요.
3. **Hard.** 20 Newsgroups 코퍼스로 모델을 학습하세요. 편향 축 두 개를 계산하세요: `he - she`와 `doctor - nurse`. 직업 단어를 두 축에 투영하세요. 편향 격차가 가장 큰 직업을 보고하세요. 공정성 연구자가 쓰는 종류의 프로브입니다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|----------|
| Word embedding | 단어를 벡터로 | 문맥에서 학습한 밀집·저차원(보통 100–300) 표현. |
| Skip-gram | Word2Vec 트릭 | 중심 단어로 문맥 단어를 예측. CBOW보다 느리고 희귀 단어에 더 좋음. |
| Negative sampling | 학습 지름길 | 전체 어휘 소프트맥스를 `k`개 랜덤 단어에 대한 이진 분류로 대체. |
| Static embedding | 단어당 벡터 하나 | 문맥과 무관한 동일 벡터. 다의어에서 실패. |
| Contextual embedding | 문맥 민감 벡터 | 주변 단어에 따라 출현마다 다른 벡터. 트랜스포머가 만드는 것. |
| OOV | Out of vocabulary | 학습에서 보지 못한 단어. Word2Vec은 이들에 대한 벡터를 만들 수 없음. |

## 더 읽어보기 (Further Reading)

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 네거티브 샘플링 논문. 짧고 읽기 쉽습니다.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 원 논문 수학이 빽빽하면, 기울기 유도가 가장 명확한 글입니다.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 실제로 통하는 프로덕션 학습 설정.
