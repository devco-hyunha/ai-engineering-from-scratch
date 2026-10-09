# GloVe, FastText, 서브워드 임베딩

> Word2Vec는 단어당 하나의 임베딩을 학습했습니다. GloVe는 동시 발생(co-occurrence) 행렬을 분해했습니다. FastText는 조각(piece)을 임베딩했습니다. BPE는 트랜스포머(transformer)로 연결하는 역할을 했습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 03강 (Word2Vec 직접 구현)
**시간:** 약 45분

## 문제점

Word2Vec는 두 가지 미해결 질문을 남겼습니다.

첫째, 온라인 스킵그램(skip-gram) 업데이트 대신 동시 발생(co-occurrence) 행렬을 직접 분해하는 병행 연구 라인(LSA, HAL)이 존재했습니다. Word2Vec의 반복적 접근 방식이 근본적으로 더 나은 것일까요, 아니면 두 방법의 카운트 처리 방식 차이로 인한 인위적 결과일까요? **GloVe**가 이에 대한 답을 제시했습니다. 신중하게 선택한 손실 함수를 사용하는 행렬 분해는 Word2Vec과 동등하거나 더 나은 성능을 내며, 학습 비용도 더 적습니다.

둘째, 두 방법 모두 본 적 없는 단어에 대한 스토리가 없었습니다. `Zoomer-approved`, `dogecoin`, 지난 주에 만들어진 고유 명사, 희소 어근의 모든 활용형. **FastText**는 문자 n-gram을 임베딩하여 이를 해결했습니다. 단어는 형태소(morpheme)를 포함해 그 부분들의 합이므로, 어휘 목록에 없는(out-of-vocabulary) 단어라도 합리적인 벡터를 얻습니다.

셋째, 트랜스포머(transformer)가 등장하자 질문이 다시 바뀌었습니다. 단어 수준 어휘는 약 백만 항목에서 한계에 도달합니다. 실제 언어는 그보다 더 열려 있습니다. **바이트 쌍 인코딩 (BPE)(Byte-pair encoding (BPE))** 및 관련 기법들은 모든 것을 커버하는 빈번한 서브워드(subword) 단위 어휘를 학습하여 이를 해결했습니다. 모든 최신 LLM의 모든 최신 토크나이저는 서브워드 토크나이저입니다.

이 강의는 세 가지를 모두 다루며, 언제 무엇을 선택해야 하는지 설명합니다.

## 개념

**GloVe (Global Vectors).** 단어-단어 동시 발생(co-occurrence) 행렬 `X`를 구축합니다. 여기서 `X[i][j]`은 단어 `j`가 단어 `i`의 컨텍스트에서 얼마나 자주 나타나는지를 나타냅니다. `v_i · v_j + b_i + b_j ≈ log(X[i][j])`가 되도록 벡터를 학습합니다. 손실 함수에 가중치를 적용하여 빈번한 쌍이 지배하지 않도록 합니다. 끝입니다.

**FastText.** 단어는 문자 n-gram과 단어 자체의 합입니다. `where`는 `<wh, whe, her, ere, re>, <where>`이 됩니다. 단어 벡터는 이러한 구성 요소 벡터들의 합입니다. Word2Vec처럼 학습합니다. 장점: 본 적 없는 단어(`whereupon`)는 알려진 n-gram으로 구성됩니다.

**BPE (바이트 쌍 인코딩)(Byte Pair Encoding (BPE)).** 개별 바이트(또는 문자)의 어휘(Vocabulary)로 시작합니다. 코퍼스(corpus)에서 모든 인접 쌍을 세어 보세요. 가장 빈번한 쌍을 새로운 토큰(Token)으로 병합합니다. `k` 반복을 수행합니다. 결과: `k + 256`개의 토큰으로 이루어진 어휘(Vocabulary)가 생성되며, 빈번한 시퀀스(`ing`, `tion`, `the`)는 단일 토큰이 되고, 희소한 단어는 친숙한 조각으로 분해됩니다. 모든 문장은 무언가로 토큰화(Tokenization)됩니다.

```figure
n5-subword-merge
```

## 구현하기

### GloVe: 공현 빈도 행렬(factorize the co-occurrence matrix) 분해

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

언급할 가치가 있는 두 가지 이동 요소가 있습니다. 가중치 함수 `f(x) = (x/x_max)^alpha`는 매우 빈번한 쌍(예: `(the, and)`)의 가중치를 낮추어 손실(loss)을 지배하지 않도록 합니다. 최종 임베딩(Embedding)은 `W`(center) 및 `W_tilde`(context) 테이블의 합입니다. 두 테이블을 합산하는 것은 공개된 트릭(trick)으로, 하나만 사용하는 것보다 성능이 더 좋은 경향이 있습니다.

### FastText: 하위 단어(subword) 인식 임베딩(Embedding)

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

각 단어는 n-gram 집합(보통 3~6자)으로 표현됩니다. 단어 임베딩(Embedding)은 n-gram 임베딩의 합입니다. Skip-gram 학습에서는 Word2Vec이 단일 벡터를 사용했던 부분에 이를 적용합니다.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

보지 못한 단어(unseen word)의 경우에도 일부 n-gram이 알려진다면 벡터를 얻을 수 있습니다. `whereupon`는 `<wh`, `her`, `ere`, `<where`를 `where`와 공유하므로, 두 단어가 서로 가까운 위치에 배치됩니다.

### BPE: 학습된 하위 단어(subword) 어휘(Vocabulary)

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

첫 번째 반복에서는 가장 흔한 인접 쌍을 병합합니다. 충분한 반복 후, 빈번한 하위 문자열(`low`, `est`, `tion`)은 단일 토큰이 되고, 희소한 단어는 깔끔하게 분해됩니다.

실제 GPT / BERT / T5 토크나이저(tokenizer)는 30k~100k의 병합을 학습합니다. 결과: 모든 텍스트는 알려진 ID의 유한 길이 시퀀스로 토큰화(Tokenization)되며, OOV(Out-Of-Vocabulary)는 발생하지 않습니다.

## 사용하기

실제로는 이러한 것들을 직접 학습하는 경우가 거의 없습니다. 사전 학습된 체크포인트(Checkpoint)를 로드합니다.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

트랜스포머(Transformer) 시대의 BPE 스타일 하위 단어(subword) 토큰화(Tokenization)의 경우:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 접두어(prefix)는 단어 경계를 표시합니다(GPT-2 관례). 모든 현대적인 토크나이저(tokenizer)는 BPE 변형, WordPiece (BERT), 또는 SentencePiece (T5, LLaMA)입니다.

### 어떤 것을 선택해야 하는가

| 상황 | 선택 |
|-----------|------|
| 사전 학습된 범용 단어 벡터, OOV 허용 필요 없음 | GloVe 300d |
| 사전 학습된 범용 단어 벡터, 오탈자 / 신조어 / 형태론적으로 풍부한 언어를 처리해야 함 | FastText |
| 트랜스포머에 입력되는 모든 것 (학습 또는 추론) | 모델에 포함된 토크나이저. 절대 교체하지 마세요. |
| 자체 언어 모델을 처음부터 학습 | 먼저 코퍼스에 BPE 또는 SentencePiece 토크나이저를 학습하세요 |
| 선형 모델을 사용한 프로덕션 텍스트 분류 | 여전히 TF-IDF입니다. 02강. |

## 출시하기

`outputs/skill-embeddings-picker.md`로 저장하세요:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 연습 문제

1. **쉬움.** `char_ngrams("playing")`과 `char_ngrams("played")`을 실행하세요. 두 n-gram 집합의 자카드 중복을 계산하세요. 상당한 공유 조각(`pla`, `lay`, `play`)이 보일 것입니다. 이것이 FastText가 형태론적 변형 간에 잘 전이되는 이유입니다.
2. **중간.** `learn_bpe`을 확장하여 어휘 성장을 추적하세요. 병합 횟수의 함수로 코퍼스 문자당 토큰 수를 플롯하세요. 처음에는 급격한 압축이 일어나고, 토큰당 약 2-3자 부근에서 점근하는 것을 볼 수 있습니다.
3. **어려움.** 셰익스피어의 전집에 대해 1,000번 병합한 BPE를 학습하세요. 일반 단어와 희귀 고유 명사의 토큰화를 비교하세요. 병합 전후의 단어당 평균 토큰 수를 측정하세요. 놀라웠던 점을 작성하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 동시 발생 행렬 | 단어-단어 빈도 표 | `X[i][j]` = `j` 단어가 `i` 단어 주변 윈도우에서 얼마나 자주 나타나는지. |
| 서브워드 | 단어의 조각 | 문자 n-gram (FastText) 또는 학습된 토큰 (BPE/WordPiece/SentencePiece). |
| BPE | 바이트 쌍 인코딩 | 어휘가 목표 크기에 도달할 때까지 가장 빈번한 인접 쌍을 반복적으로 병합하는 것. |
| OOV | 어휘 외 | 모델이 본 적 없는 단어. Word2Vec/GloVe는 실패합니다. FastText와 BPE는 이를 처리합니다. |
| 바이트 수준 BPE | 원시 바이트에 대한 BPE | GPT-2의 방식. 어휘가 256바이트로 시작하므로 OOV는 절대 발생하지 않습니다. |

## 추가 읽기

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 논문, 7페이지, 손실 함수의 가장 좋은 유도 과정입니다.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 현대 NLP에 BPE를 도입한 논문.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE, WordPiece, SentencePiece가 실제로 어떻게 다른지 확인해 보세요.
