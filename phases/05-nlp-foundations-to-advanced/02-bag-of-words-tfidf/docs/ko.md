# Bag of Words, TF-IDF, 텍스트 표현 (Bag of Words, TF-IDF, and Text Representation)

> 먼저 세고, 나중에 생각하세요. 2026년에도 잘 정의된 작업에서는 TF-IDF가 임베딩을 이깁니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## 문제 (The Problem)

모델은 숫자가 필요합니다. 당신에게는 문자열이 있습니다.

모든 NLP 파이프라인은 같은 질문에 답해야 합니다. 가변 길이 토큰 스트림을, 분류기가 소비할 수 있는 고정 크기 벡터로 어떻게 바꿀 것인가. 이 분야가 처음 도달한 답은 작동하는 답 중 가장 단순한 것이었습니다. 단어를 세요. 벡터를 만드세요.

그 벡터는 어떤 임베딩 모델보다도 더 많은 프로덕션 NLP를 지탱해 왔습니다. 스팸 필터, 토픽 분류기, 로그 이상 탐지, 검색 랭킹(BM25 이전), 감성 분석의 첫 물결, 학술 NLP 벤치마크의 첫 10년. 2026년 실무자도 좁은 분류 작업에서는 여전히 이것을 먼저 꺼냅니다. 빠르고, 해석 가능하며, 단어의 출현 여부가 핵심인 작업에서는 4억 파라미터 임베딩 모델과 거의 구분이 안 됩니다.

이 레슨에서는 Bag of Words를 만든 뒤 TF-IDF를 처음부터 구현합니다. 이어서 scikit-learn이 같은 일을 세 줄로 하는 모습을 보고, 임베딩으로 넘어가게 만드는 실패 모드를 이름 붙입니다.

## 개념 (The Concept)

**Bag of Words(BoW)**는 순서를 버립니다. 각 문서에 대해 어휘의 각 단어가 몇 번 나타나는지 셉니다. 벡터 길이는 어휘 크기입니다. 위치 `i`는 단어 `i`의 출현 횟수입니다.

**TF-IDF**는 BoW를 재가중합니다. 모든 문서에 나오는 단어는 정보가 적으니 스케일을 낮춥니다. 코퍼스 전체에서는 드물지만 한 문서에서는 자주 나오는 단어는 신호이니 스케일을 올립니다.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

여기서 `TF`는 문서 내 용어 빈도(term frequency), `df`는 문서 빈도(document frequency, 그 단어를 포함하는 문서 수), `N`은 전체 문서 수입니다. `log`는 어디에나 있는 단어의 가중치를 제한합니다.

핵심 성질: 둘 다 해석 가능한 축을 가진 희소 벡터를 만듭니다. 학습된 분류기의 가중치를 보고 어떤 단어가 문서를 각 클래스로 밀어내는지 읽을 수 있습니다. 768차원 BERT 임베딩으로는 이렇게 할 수 없습니다.

```figure
bow-tfidf
```

## 직접 만들기 (Build It)

### 1단계: 어휘 구축

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

입력: 토큰화된 문서 리스트(단어 단위 토크나이저면 됩니다. 이 레슨의 `code/main.py`는 단순화된 소문자 변형을 씁니다). 출력: `{word: index}` 딕셔너리. 안정적인 삽입 순서 덕분에 단어 인덱스 0은 첫 문서에서 처음 본 단어입니다. 관례는 다양합니다. scikit-learn은 알파벳순으로 정렬합니다.

### 2단계: Bag of Words

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

행은 문서입니다. 열은 어휘 인덱스입니다. 항목 `[i][j]`는 "문서 `i`에 단어 `j`가 몇 번 나타나는가"입니다. 문서 1에는 `cat`이 두 번 나오므로 2이고, 문서 0에는 `ran`이 전혀 나오지 않으므로 0입니다.

### 3단계: 용어 빈도와 문서 빈도

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

짚고 넘어갈 만한 스무딩 트릭이 두 가지 있습니다. `(n+1)/(d+1)`은 0으로 나누는 상황(`log(x/0)`)을 방지합니다. 식 끝의 `+1`은 모든 문서에 등장하는 단어라도 IDF가 0이 아닌 1이 되도록 보장하여 scikit-learn의 기본 동작과 일치시킵니다. 다른 구현에서는 단순 `log(N/df)`를 쓰기도 합니다. 둘 다 동작하지만, 스무딩을 적용한 버전이 훨씬 안정적입니다.

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

문서 세 개, 어휘 단어 다섯 개(`the`, `cat`, `sat`, `dog`, `ran`). `the`는 세 문서 모두에 나오므로 IDF가 낮습니다. `dog`는 하나에만 나오므로 IDF가 높습니다. 벡터는 희소하고(대부분 항목이 작음) 변별력 있는 단어가 두드러집니다.

### 5단계: 행 L2 정규화

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

정규화를 거치지 않으면 긴 문서일수록 벡터 크기가 커져 유사도 점수를 왜곡하게 됩니다. L2 정규화는 모든 문서를 단위 초구(unit hypersphere) 위에 위치시킵니다. 정규화를 거치면 행 간 코사인 유사도는 단순 내적(dot product) 계산이 됩니다.

## 활용하기 (Use It)

scikit-learn이 프로덕션 버전을 제공합니다.

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

`CountVectorizer`는 토큰화, 어휘 사전 구축, BoW 생성을 단 한 번의 호출로 처리합니다. `TfidfVectorizer`는 여기에 IDF 가중치 계산과 L2 정규화를 추가합니다. 두 벡터화기 모두 희소 행렬(sparse matrix)을 반환합니다. 문서가 10만 건쯤 되면 밀집(dense) 행렬은 메모리에 다 들어가지 않습니다. 분류기가 반드시 밀집 행렬을 요구하기 전까지는 희소 형식을 유지하세요.

결과를 바꾸는 주요 설정(knob)들입니다.

| Arg | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | 바이그램(bigram)을 포함합니다. 일반적으로 분류 성능을 향상시킵니다. |
| `min_df=2` | 2개 미만의 문서에 나타나는 단어를 제외합니다. 노이즈가 많은 데이터에서 어휘 크기를 줄여줍니다. |
| `max_df=0.95` | 95%를 초과하는 문서에 등장하는 단어를 제외합니다. 불용어 목록을 하드코딩하지 않고도 불용어 제거와 유사한 효과를 냅니다. |
| `stop_words="english"` | scikit-learn의 기본 불용어 목록입니다. 작업에 따라 달라지며, 감성 분석에서는 부정어(negation)를 절대 제거해서는 안 됩니다. |
| `sublinear_tf=True` | 원시 `tf` 대신 `1 + log(tf)`를 사용합니다. 한 문서 안에서 특정 단어가 지나치게 많이 반복될 때 유용합니다. |

### TF-IDF가 여전히 이기는 경우 (2026 기준)

- 스팸 탐지, 토픽 라벨링, 로그 이상 플래깅. 단어 출현이 핵심이고 의미적 뉘앙스는 중요하지 않습니다.
- 데이터가 적은 환경(수백 개의 라벨링 예제). TF-IDF와 로지스틱 회귀의 조합은 사전학습 비용이 전혀 들지 않습니다.
- 응답 지연 시간(latency)이 중요한 모든 환경. TF-IDF와 선형 모델은 마이크로초(µs) 단위로 결과를 냅니다. 트랜스포머로 문서를 임베딩하려면 10~100ms가 소요됩니다.
- 예측을 설명해야 하는 시스템. 분류기 계수를 검사하세요. 상위 양의 가중 단어가 이유입니다.

### TF-IDF가 실패하는 경우

첫 번째는 의미적 맹목(semantic blindness)으로 인한 실패입니다. 다음 두 문서를 보세요.

- "The movie was not good at all."
- "The movie was excellent."

하나는 부정 리뷰이고 다른 하나는 긍정 리뷰입니다. 두 문서 간 TF-IDF 중복 단어는 `{the, movie, was}`뿐입니다. Bag-of-words 분류기는 `good` 주변의 `not`이 라벨을 반전시킨다는 사실을 직접 암기해야만 합니다. 학습 데이터가 충분하다면 학습할 수는 있지만, 문장 구조와 구문을 이해하는 모델만큼 자연스럽고 매끄럽게 처리하지는 못합니다.

두 번째 실패: 추론 시 발생하는 미등록 어휘(out-of-vocabulary, OOV) 문제입니다. IMDb 리뷰 데이터셋으로 학습한 BoW 모델은 학습 데이터에 등장한 적 없는 `Zoomer-approved` 같은 신조어를 어떻게 다뤄야 할지 전혀 알지 못합니다. 서브워드 임베딩(레슨 04)은 이를 분해하여 처리할 수 있지만, TF-IDF는 처리할 수 없습니다.

### 하이브리드: TF-IDF 가중 임베딩

중간 규모 데이터 분류의 2026년 실용 기본값: TF-IDF 가중치를 단어 임베딩 위의 어텐션처럼 씁니다.

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

임베딩으로부터 풍부한 의미 표현력을 얻고, TF-IDF로부터 희귀 단어 강조 효과를 함께 얻을 수 있습니다. 분류기는 이렇게 풀링된 벡터를 입력받아 학습합니다. 라벨링된 예제가 약 5만 개 이하인 환경에서는 감성·주제·의도 분류 작업에서 두 방식 중 하나만 단독으로 사용하는 것보다 더 뛰어난 성능을 보입니다.

## 배포하기 (Ship It)

`outputs/prompt-vectorization-picker.md`로 저장하세요.

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

## 연습 문제 (Exercises)

1. **Easy.** L2 정규화된 TF-IDF 출력에 `cosine_similarity(doc_vec_a, doc_vec_b)`를 구현하세요. 동일 문서는 1.0, 어휘가 겹치지 않는 문서는 0.0이 되는지 검증하세요.
2. **Medium.** `bag_of_words`에 `n-gram` 지원을 추가하세요. 파라미터 `n`은 `n`-gram 위의 카운트를 만듭니다. `["the", "cat", "sat"]`에 `n=2`를 주면 `["the cat", "cat sat"]`의 바이그램 카운트가 나오는지 테스트하세요.
3. **Hard.** 위의 TF-IDF 가중 임베딩 하이브리드를 GloVe 100d 벡터로 만드세요(한 번 다운로드, 캐시). 20 Newsgroups 데이터셋에서 순수 TF-IDF, 순수 평균 풀링 임베딩과 분류 정확도를 비교하고, 어디서 무엇이 이기는지 보고하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BoW | 단어 빈도 벡터 | 한 문서 내 어휘별 단어 출현 횟수. 단어 순서는 무시함. |
| TF | 용어 빈도 | 문서 내 단어 카운트. 선택적으로 문서 길이로 정규화. |
| DF | 문서 빈도 | 그 단어를 한 번 이상 포함하는 문서 수. |
| IDF | 역문서 빈도 | 스무딩된 `log(N / df)`. 어디에나 나오는 흔한 단어의 가중치를 낮춤. |
| Sparse vector | 대부분 0 | 어휘 사전은 보통 1만~10만 단어로 구성되나, 특정 문서에는 그중 극히 일부만 등장함. |
| Cosine similarity | 벡터 각도 | L2 정규화 벡터 간의 내적. 1은 완전히 일치, 0은 직교(공통 단어 없음). |

## 더 읽을거리 (Further Reading)

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 표준 API 참고서와 모든 노브에 대한 메모.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — TF-IDF를 한 시대의 기본값으로 만든 논문.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 구방법이 언제 이기는지와 그 이유에 대한 2026년 관점.
