# Bag of Words, TF-IDF, 그리고 텍스트 표현

> 먼저 세고, 나중에 생각하세요. 2026년에도 명확히 정의된 작업에서는 TF-IDF가 임베딩을 능가합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 01강 (텍스트 처리), 2단계 · 02강 (처음부터 만드는 선형 회귀)
**시간:** 약 75분

## 문제점

모델은 숫자를 필요로 합니다. 당신은 문자열을 가지고 있습니다.

모든 NLP 파이프라인은 동일한 질문에 답해야 합니다. 가변 길이의 토큰 스트림을 분류기가 소비할 수 있는 고정 크기의 벡터로 어떻게 변환할 것인가? 이 분야가 도달한 첫 번째 답은 가장 단순하지만 작동하는 방법입니다. 단어를 세고, 벡터를 만드세요.

이 벡터는 어떤 임베딩 모델보다도 더 많은 프로덕션 NLP를 담당해 왔습니다. 스팸 필터, 주제 분류기, 로그 이상 감지, 검색 순위 (BM25 이전), 초기 감성 분석, 학술 NLP 벤치마크의 첫 10년. 2026년 실무자들도 좁은 분류 작업에서는 여전히 이것을 먼저 선택합니다. 빠르고, 해석 가능하며, 단어의 존재 여부가 중요한 작업에서는 4억 매개변수 임베딩 모델과 구별할 수 없는 경우가 많습니다.

이 강의는 Bag of Words와 TF-IDF를 처음부터 구축합니다. 그 다음 scikit-learn이 세 줄로 동일한 작업을 수행하는 것을 보여줍니다. 그리고 임베딩을 찾게 만드는 실패 모드에 대해 설명합니다.

## 개념

**Bag of Words (BoW)**는 순서를 버립니다. 각 문서에 대해 각 어휘 단어가 몇 번 나타나는지 세세요. 벡터 길이는 어휘 크기입니다. 위치 `i`는 단어 `i`의 개수입니다.

**TF-IDF**는 BoW의 가중치를 재조정합니다. 모든 문서에 나타나는 단어는 정보가 없으므로 축소하세요. 코퍼스 전체에서는 드물지만 단일 문서에서는 자주 나타나는 단어는 신호이므로 확대하세요.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

여기서 `TF`는 문서 내의 용어 빈도, `df`는 문서 빈도 (단어를 포함하는 문서의 수), `N`는 총 문서 수입니다. `log`은 보편적인 단어에 대해 가중치를 제한합니다.

핵심 속성: 둘 다 해석 가능한 축을 가진 희소 벡터를 생성합니다. 학습된 분류기의 가중치를 보고, 단어가 문서를 각 클래스로 어떻게 밀어내는지 읽을 수 있습니다. 768차원 BERT 임베딩으로는 이 작업을 할 수 없습니다.

```figure
bow-tfidf
```

## 구현하기

### 1단계: 어휘(Vocabulary) 구축하기

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

입력: 토큰화된 문서 목록 (단어 단위 토크나이저는 무엇이든 사용 가능; 이 강의의 `code/main.py`는 단순화된 소문자 변형을 사용). 출력: `{word: index}` dict. 안정적인 삽입 순서 덕분에 단어 인덱스 0은 첫 번째 문서에서 처음 발견된 단어가 됩니다. 관례는 다양합니다; scikit-learn은 알파벳 순으로 정렬합니다.

### 2단계: 단어 주머니(Bag of Words)

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

행은 문서입니다. 열은 어휘(Vocabulary) 인덱스입니다. 항목 `[i][j]`은 "문서 `i`에서 단어 `j`이 몇 번 나타나는가"를 의미합니다. 문서 1에는 `cat`이 두 번 나타납니다. 문서 0에는 `ran`이 0번 나타납니다.

### 3단계: 용어 빈도(Term Frequency) 및 문서 빈도(Document Frequency)

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

언급할 가치가 있는 두 가지 스무딩(smoothing) 트릭이 있습니다. `(n+1)/(d+1)`는 `log(x/0)`을 피합니다. 끝의 `+1`는 모든 문서에 있는 단어가 IDF가 1 (0이 아님)이 되도록 보장하며, scikit-learn의 기본값과 일치합니다. 다른 구현은 원시 `log(N/df)`을 사용합니다. 둘 다 작동합니다; 스무딩된 버전이 더 친화적입니다.

### 4단계: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

세 개의 문서, 다섯 개의 어휘(Vocabulary) 단어 (`the`, `cat`, `sat`, `dog`, `ran`). `the`는 세 문서 모두에 나타나므로 IDF가 낮습니다. `dog`는 한 문서에만 나타나므로 IDF가 높습니다. 벡터는 희소(sparse)합니다 (대부분의 항목이 작음)이며, 판별적인 단어가 튀어나옵니다.

### 5단계: 행을 L2 정규화(Normalization)하기

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

정규화(Normalization)가 없으면, 더 긴 문서가 더 큰 벡터를 얻어 유사성 점수를 지배합니다. L2 정규화(Normalization)는 모든 문서를 단위 초구(unit hypersphere) 위에 놓습니다. 이제 행 간의 코사인 유사도(Cosine Similarity)는 단순한 내적(dot product)입니다.

## 사용하기

scikit-learn은 프로덕션 버전을 제공합니다.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`는 토큰화(Tokenization), 어휘(Vocabulary), BoW를 한 번의 호출로 처리합니다. `TfidfVectorizer`는 IDF 가중치와 L2 정규화(Normalization)를 추가합니다. 둘 다 희소(sparse) 행렬을 반환합니다. 10만 개의 문서의 경우, 밀집(dense) 버전은 메모리에 맞지 않습니다; 분류기가 밀집(dense)을 요구할 때까지 희소(sparse) 상태를 유지하세요.

모든 것을 바꾸는 조절 변수(Knobs):

| 인자 | 효과 |
|-----|--------|
| `ngram_range=(1, 2)` | 바이그램(bigram) 포함. 보통 분류 성능을 높입니다. |
| `min_df=2` | 2개 미만 문서에 있는 단어를 제거합니다. 잡음 데이터에서 어휘(Vocabulary)를 줄입니다. |
| `max_df=0.95` | 95% 이상의 문서에 있는 단어를 제거합니다. 하드코딩된 목록 없이 스톱워드(stopword) 제거를 근사합니다. |
| `stop_words="english"` | scikit-learn의 내장 불용어 목록. 작업에 따라 다름 — 감정 분석에서는 부정어(negation)를 제거하지 *않아야* 합니다. |
| `sublinear_tf=True` | `1 + log(tf)`을 `tf` 대신 사용하세요. 한 문서에서 특정 용어가 반복될 때 도움이 됩니다. |

### TF-IDF가 여전히 우세한 경우 (2026년 기준)

- 스팸 탐지, 주제 라벨링, 로그 이상 감지. 단어의 존재 여부가 중요하며, 의미적 미묘한 차이는 중요하지 않습니다.
- 저데이터 환경 (수백 개의 레이블된 예시). TF-IDF와 로지스틱 회귀는 사전 학습 비용이 없습니다.
- 레이턴시가 중요한 모든 곳. TF-IDF와 선형 모델은 마이크로초 단위로 응답합니다. 트랜스포머를 통해 문서를 임베딩하는 데는 10-100ms가 걸립니다.
- 예측을 설명해야 하는 시스템. 분류기의 계수를 확인하세요. 가장 높은 양의 값을 가진 단어가 이유입니다.

### TF-IDF가 실패하는 경우

의미적 맹점 실패. 다음 두 문서를 고려해 보세요:

- "The movie was not good at all."
- "The movie was excellent."

하나는 부정적인 리뷰이고, 다른 하나는 긍정적인 리뷰입니다. 두 문서의 TF-IDF 겹침은 정확히 `{the, movie, was}`입니다. 단어 주머니(bag-of-words) 분류기는 `good` 근처의 `not`가 레이블을 뒤집는다는 것을 기억해야 합니다. 충분한 데이터가 있으면 이를 학습할 수 있지만, 구문을 이해하는 모델만큼 우아하게 학습하지는 못합니다.

다른 실패: 추론 시 어휘 외(out-of-vocabulary) 단어. IMDb 리뷰로 학습된 BoW 모델은 학습 중에 등장하지 않은 `Zoomer-approved` 토큰을 어떻게 처리해야 할지 모릅니다. 서브워드 임베딩(04강)은 이를 처리합니다. TF-IDF는 처리할 수 없습니다.

### 하이브리드: TF-IDF 가중 임베딩

중간 규모 데이터 분류에 대한 2026년 실용적 기본값: 단어 임베딩에 대한 어텐션으로 TF-IDF 가중치를 사용하세요.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

임베딩에서 의미적 용량을 얻고, TF-IDF에서 희소 단어 강조를 얻습니다. 분류기는 풀링된 벡터로 학습합니다. 약 50k개의 레이블된 예시 미만인 감정, 주제, 의도 분류에서는 각각 단독으로 사용하는 것보다 성능이 더 좋습니다.

## 출시하기

`outputs/prompt-vectorization-picker.md`로 저장하세요:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 연습 문제

1. **쉬움.** L2 정규화된 TF-IDF 출력에 `cosine_similarity(doc_vec_a, doc_vec_b)`을 구현하세요. 동일한 문서는 1.0 점수를 받고, 어휘가 겹치지 않는 문서는 0.0 점수를 받는지 확인하세요.
2. **중간 난이도.** `bag_of_words`에 `n-gram` 지원을 추가하세요. 매개변수 `n`는 `n`-그램에 대한 개수를 생성합니다. `["the", "cat", "sat"]`에 `n=2`를 적용하면 `["the cat", "cat sat"]`에 대한 바이그램 개수가 생성되는지 테스트해 보세요.
3. **어려운 난이도.** GloVe 100d 벡터(한 번 다운로드하고 캐시)를 사용하여 위의 TF-IDF 가중 임베딩 하이브리드를 구축하세요. 20 Newsgroups 데이터셋에서 순수 TF-IDF 및 순수 평균 풀링 임베딩과 분류 정확도를 비교하세요. 어떤 방법이 어떤 상황에서 더 우월한지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BoW | 단어 빈도 벡터 | 한 문서 내 어휘 단어의 개수. 순서를 버립니다. |
| TF | 단어 빈도 | 문서 내 단어의 개수. 문서 길이에 따라 정규화할 수 있습니다. |
| DF | 문서 빈도 | 해당 단어를 최소 한 번 포함하는 문서의 개수. |
| IDF | 역 문서 빈도 | `log(N / df)`가 스무딩(smoothed)된 형태. 모든 곳에 나타나는 단어의 가중치를 낮춥니다. |
| 희소 벡터 | 대부분 0 | 어휘는 일반적으로 10k-100k 단어이며, 특정 문서에는 대부분 단어가 없습니다. |
| 코사인 유사도(Cosine similarity) | 벡터 각도 | L2 정규화된 벡터의 내적. 1은 동일, 0은 직교를 의미합니다. |

## 추가 읽기

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 표준 API 레퍼런스이며, 모든 설정 옵션에 대한 노트가 포함되어 있습니다.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — TF-IDF를 10년간 기본값으로 만든 논문입니다.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026년 관점에서의 구식 방법이 승리하는 시점과 이유에 대한 내용입니다.
