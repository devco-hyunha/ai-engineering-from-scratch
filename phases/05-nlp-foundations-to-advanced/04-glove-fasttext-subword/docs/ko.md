# GloVe, FastText, 서브워드 임베딩 (GloVe, FastText, and Subword Embeddings)

> Word2Vec은 단어마다 임베딩 하나를 학습했습니다. GloVe는 공출현 행렬을 분해했습니다. FastText는 조각들을 임베딩했습니다. BPE는 트랜스포머로 이어지는 다리였습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 문제 (The Problem)

Word2Vec은 두 가지 열린 질문을 남겼습니다.

첫째, 온라인 skip-gram 업데이트 대신 공출현 행렬을 직접 분해하는 병렬 연구 흐름(LSA, HAL)이 있었습니다. Word2Vec의 반복적 접근이 본질적으로 더 나았던 것일까요, 아니면 두 방법이 카운트를 다루는 방식의 부산물이었을까요? **GloVe**가 답했습니다. 신중히 고른 손실로 행렬 분해하면 Word2Vec과 맞먹거나 더 낫고, 학습 비용은 더 적습니다.

둘째, 어느 방법도 본 적 없는 단어에 대한 이야기가 없었습니다. `Zoomer-approved`, `dogecoin`, 지난주에 생긴 고유명사, 희귀 어근의 모든 굴절형. **FastText**는 문자 n-그램을 임베딩해 이를 고쳤습니다. 단어는 형태소를 포함한 부분들의 합이므로, 어휘 밖(out-of-vocabulary) 단어도 괜찮은 벡터를 얻습니다.

셋째, 트랜스포머가 등장하자 질문이 다시 바뀌었습니다. 단어 수준 어휘는 대략 백만 개에서 막히고, 실제 언어는 그보다 더 열려 있습니다. **바이트 페어 인코딩(Byte-pair encoding, BPE)**과 그 친척들은 모든 것을 덮는 빈번한 서브워드 단위 어휘를 학습해 이를 풀었습니다. 현대 LLM의 모든 현대 토크나이저는 서브워드 토크나이저입니다.

이 레슨은 셋을 모두 살펴본 뒤, 언제 무엇을 고를지 설명합니다.

## 개념 (The Concept)

**GloVe (Global Vectors).** 단어-단어 공출현 행렬 `X`를 만듭니다. `X[i][j]`는 단어 `j`가 단어 `i`의 문맥에 나타난 횟수입니다. `v_i · v_j + b_i + b_j ≈ log(X[i][j])`가 되도록 벡터를 학습합니다. 빈번한 쌍이 지배하지 않도록 손실에 가중치를 둡니다. 끝입니다.

**FastText.** 단어는 문자 n-그램들과 단어 자체의 합입니다. `where`는 `<wh, whe, her, ere, re>, <where>`가 됩니다. 단어 벡터는 그 구성 벡터들의 합입니다. Word2Vec처럼 학습합니다. 이점: 미지 단어(`whereupon`)도 알려진 n-그램에서 합성됩니다.

**BPE (Byte-Pair Encoding).** 개별 바이트(또는 문자) 어휘로 시작합니다. 코퍼스에서 인접 쌍을 모두 셉니다. 가장 빈번한 쌍을 새 토큰으로 병합합니다. `k`번 반복합니다. 결과: `k + 256`개 토큰 어휘. 빈번한 시퀀스(`ing`, `tion`, `the`)는 단일 토큰이 되고, 희귀 단어는 익숙한 조각으로 나뉩니다. 모든 문장이 무언가로 토큰화됩니다.

```figure
n5-subword-merge
```

## 직접 만들기 (Build It)

### GloVe: 공출현 행렬 분해

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

이름 붙일 가치가 있는 움직이는 조각이 두 개 있습니다. 가중 함수 `f(x) = (x/x_max)^alpha`는 `(the, and)`처럼 매우 빈번한 쌍의 가중치를 낮춰 손실을 지배하지 않게 합니다. 최종 임베딩은 `W`(중심)와 `W_tilde`(문맥) 테이블의 합입니다. 둘을 더하는 것은 하나만 쓰는 것보다 나은 경향이 있는, 발표된 트릭입니다.

### FastText: 서브워드를 아는 임베딩

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

각 단어는 n-그램 집합(보통 3~6자)으로 표현됩니다. 단어 임베딩은 그 n-그램 임베딩들의 합입니다. skip-gram 학습에서는 Word2Vec이 단일 벡터를 쓰던 자리에 이것을 끼워 넣습니다.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

미지 단어라도 n-그램 일부가 알려져 있으면 벡터를 얻습니다. `whereupon`은 `where`와 `<wh`, `her`, `ere`, `<where`를 공유하므로 둘이 근처에 놓입니다.

### BPE: 학습된 서브워드 어휘

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

첫 반복은 가장 흔한 인접 쌍을 병합합니다. 충분히 반복하면 빈번한 부분문자열(`low`, `est`, `tion`)이 단일 토큰이 되고, 희귀 단어는 깔끔하게 나뉩니다.

실제 GPT / BERT / T5 토크나이저는 3만~10만 번의 병합을 학습합니다. 결과: 어떤 텍스트든 알려진 ID의 유한 길이 시퀀스로 토큰화되며, OOV는 없습니다.

## 활용하기 (Use It)

실무에서는 이것들을 직접 학습하는 경우가 거의 없습니다. 사전학습 체크포인트를 불러옵니다.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

트랜스포머 시대의 BPE 스타일 서브워드 토큰화:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 접두사는 단어 경계를 표시합니다(GPT-2 관례). 현대 토크나이저는 모두 BPE 변형, WordPiece(BERT), 또는 SentencePiece(T5, LLaMA)입니다.

### 무엇을 고를지

| 상황 | 선택 |
|------|------|
| 사전학습 범용 단어 벡터, OOV 허용이 필요 없음 | GloVe 300d |
| 사전학습 범용 단어 벡터, 오타 / 신조어 / 형태론적으로 풍부한 언어를 다뤄야 함 | FastText |
| 트랜스포머에 들어가는 모든 것(학습 또는 추론) | 모델과 함께 배포된 토크나이저. 절대 바꾸지 마세요. |
| 언어 모델을 처음부터 직접 학습 | 먼저 코퍼스에서 BPE 또는 SentencePiece 토크나이저를 학습 |
| 선형 모델을 쓰는 프로덕션 텍스트 분류 | 여전히 TF-IDF. 레슨 02. |

## 배포하기 (Ship It)

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

## 연습 문제 (Exercises)

1. **쉬움.** `char_ngrams("playing")`과 `char_ngrams("played")`를 실행하세요. 두 n-그램 집합의 Jaccard 겹침을 계산하세요. 공유 조각(`pla`, `lay`, `play`)이 상당해야 하며, 그래서 FastText가 형태론적 변형 간에 잘 전이됩니다.
2. **보통.** `learn_bpe`를 확장해 어휘 성장을 추적하세요. 병합 횟수에 따른 코퍼스 문자당 토큰 수를 그리세요. 처음에는 빠른 압축이 보이고, 토큰당 약 2~3자 근처에서 점근해야 합니다.
3. **어려움.** 셰익스피어 전집에 1천 병합 BPE를 학습하세요. 흔한 단어와 희귀 고유명사의 토큰화를 비교하세요. 전후의 단어당 평균 토큰 수를 측정하세요. 놀라웠던 점을 정리하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|--------------------|-------------------|
| 공출현 행렬 (Co-occurrence matrix) | 단어-단어 빈도 표 | `X[i][j]` = 단어 `j`가 단어 `i` 주변 윈도우에 나타난 횟수. |
| 서브워드 (Subword) | 단어의 조각 | 문자 n-그램(FastText) 또는 학습된 토큰(BPE/WordPiece/SentencePiece). |
| BPE | 바이트 페어 인코딩 | 가장 빈번한 인접 쌍을 반복 병합해 어휘가 목표 크기에 이를 때까지. |
| OOV | Out of vocabulary | 모델이 본 적 없는 단어. Word2Vec/GloVe는 실패. FastText와 BPE는 처리. |
| 바이트 수준 BPE (Byte-level BPE) | 원시 바이트 위의 BPE | GPT-2의 방식. 어휘가 256바이트로 시작하므로 OOV가 절대 없음. |

## 더 읽어보기 (Further Reading)

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 논문. 일곱 쪽. 여전히 손실의 가장 좋은 유도.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 현대 NLP에 BPE를 도입한 논문.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE, WordPiece, SentencePiece가 실무에서 실제로 어떻게 다른지.
