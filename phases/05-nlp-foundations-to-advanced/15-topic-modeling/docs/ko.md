# 토픽 모델링(Topic Modeling) — LDA 및 BERTopic

> LDA: 문서는 토픽의 혼합이며, 토픽은 단어들의 분포입니다. BERTopic: 문서는 임베딩 공간에서 클러스터링되며, 클러스터가 곧 토픽입니다. 목표는 같지만, 분해 방식이 다릅니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## 문제 (The Problem)

10,000개의 고객 지원 티켓, 50,000개의 뉴스 기사, 또는 200,000개의 트윗이 있다고 가정해 봅시다. 내용을 직접 읽지 않고도 이 컬렉션이 무엇에 관한 것인지 알아내야 합니다. 레이블이 지정된 카테고리는 없으며, 카테고리가 몇 개 존재하는지도 모르는 상태입니다.

토픽 모델링(Topic modeling)은 지도 학습 없이 이 문제를 해결합니다. 코퍼스(corpus)를 입력하면 일관된 토픽들의 작은 집합을 반환하며, 각 문서에 대해 해당 토픽들에 대한 분포를 제공합니다.

두 가지 주요 알고리즘 체계(algorithmic families)가 시장을 주도하고 있습니다. LDA (2003)는 각 문서를 잠재 토픽(latent topics)의 혼합으로 취급하며, 각 토픽을 단어들의 분포로 취급합니다. 추론은 베이지안(Bayesian) 방식을 따릅니다. 혼합 멤버십(mixed-membership) 토픽 할당과 설명 가능한 단어 수준의 확률 분포가 필요한 프로덕션 환경에서 여전히 널리 사용됩니다.

BERTopic (2020)은 BERT로 문서를 인코딩하고, UMAP으로 차원을 축소하며, HDBSCAN으로 클러스터링을 수행한 뒤, class-based TF-IDF를 통해 토픽 단어를 추출합니다. 짧은 텍스트, 소셜 미디어, 그리고 단어의 중복보다 의미적 유사성(semantic similarity)이 더 중요한 모든 분야에서 우수한 성능을 보입니다. 다만, 하나의 문서가 하나의 토픽만 할당받는다는 점은 긴 형태의 콘텐츠(long-form content)에서는 한계가 될 수 있습니다.

이 레슨에서는 두 방식 모두에 대한 직관을 기르고, 주어진 코퍼스에 따라 어떤 방식을 선택해야 하는지 알아봅니다.

## 개념 (The Concept)

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 생성 과정(LDA generative story).** 각 토픽은 단어들에 대한 분포입니다. 각 문서는 토픽들의 혼합체입니다. 문서 내의 단어를 생성하려면, 문서의 혼합체에서 토픽을 샘플링한 다음, 해당 토픽의 분포에서 단어를 샘플링합니다. 추론(Inference)은 이 과정을 역으로 수행합니다. 즉, 관찰된 단어들이 주어졌을 때 문서별 토픽 분포와 토픽별 단어 분포를 추론합니다. Collapsed Gibbs sampling 또는 variational Bayes가 이 계산을 수행합니다.

주요 LDA 출력값:

- `doc_topic`: `(n_docs, n_topics)` 행렬, 각 행의 합은 1입니다 (문서의 토픽 혼합체).
- `topic_word`: `(n_topics, vocab_size)` 행렬, 각 행의 합은 1입니다 (토픽의 단어 분포).

**BERTopic 파이프라인(BERTopic pipeline).**

1. sentence transformer(예: `all-MiniLM-L6-v2`)를 사용하여 각 문서를 인코딩합니다. 384차원 벡터가 생성됩니다.
2. UMAP을 사용하여 차원을 약 5차원으로 축소합니다. BERT 임베딩은 클러스터링을 수행하기에 차원이 너무 높습니다.
3. HDBSCAN으로 클러스터링합니다. 밀도 기반(Density-based) 방식이며, 다양한 크기의 클러스터와 "outlier" 레이블을 생성합니다.
4. 각 클러스터에 대해, 해당 클러스터의 문서들을 대상으로 class-based TF-IDF를 계산하여 상위 단어들을 추출합니다.

출력값은 문서당 하나의 토픽입니다 (또한 -1인 outlier 레이블이 포함됩니다). 선택적으로 HDBSCAN의 확률 벡터를 통해 소프트 멤버십(soft membership)을 얻을 수 있습니다.

```figure
topic-drift
```

## 직접 구현해 보기 (Build It)

### 1단계: scikit-learn을 이용한 LDA (LDA via scikit-learn)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

참고: 불용어(stopwords)가 제거되었으며, min_df와 max_df는 희귀하거나 너무 흔한 용어를 필터링합니다. LDA는 원시 카운트(raw counts)를 기대하므로 TfidfVectorizer가 아닌 CountVectorizer를 사용합니다.

### 2단계: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 필터는 BERTopic의 이상치 버킷(outlier bucket, HDBSCAN이 클러스터링하지 못한 문서들)을 제외합니다. `min_topic_size`는 HDBSCAN의 최소 클러스터 크기를 제어하며, BERTopic 라이브러리의 기본값은 10입니다. 이 예제에서는 학습 규모에 맞춰 이를 15로 명시적으로 설정했습니다. 10,000개 이상의 문서로 구성된 코퍼스의 경우, 이 값을 50 또는 100으로 늘려 보세요.

### 3단계: 평가 (evaluation)

두 방법 모두 토픽 단어(topic words)를 출력합니다. 핵심은 해당 단어들이 일관성(cohere)을 갖는지 여부입니다.

- **토픽 일관성 (Topic coherence, c_v).** 슬라이딩 윈도우 컨텍스트(sliding-window contexts) 내 상위 단어 쌍의 NPMI(normalized pointwise mutual information)를 결합하고, 점수를 토픽 벡터로 집계한 뒤, 코사인 유사도(cosine similarity)를 통해 이 벡터들을 비교합니다. 값이 높을수록 좋습니다. `gensim.models.CoherenceModel`을 사용하며 `coherence="c_v"` 옵션을 설정하세요.
- **토픽 다양성 (Topic diversity).** 모든 토픽의 상위 단어들 중 고유한 단어가 차지하는 비율입니다. 값이 높을수록 좋습니다 (토픽이 서로 중복되지 않음을 의미합니다).
- **정성적 검사 (Qualitative inspection).** 각 토픽의 상위 단어들을 읽어보세요. 실제로 존재하는 개념을 나타내나요? 인간의 판단은 여전히 최후의 방어선입니다.

## 상황별 선택 기준 (When to pick which)

| 상황 | 선택 |
|-----------|------|
| 짧은 텍스트 (트윗, 리뷰, 헤드라인) | BERTopic |
| 다양한 주제가 섞인 긴 문서 | LDA |
| GPU 없음 / 제한된 컴퓨팅 자원 | LDA 또는 NMF |
| 문서 수준의 다중 주제 분포가 필요한 경우 | LDA |
| 주제 라벨링을 위한 LLM 통합 | BERTopic (직접 지원) |
| 자원이 제한된 엣지 배포 | LDA |
| 최대 의미론적 일관성 (semantic coherence) | BERTopic |

가장 중요한 실무적 고려 사항은 문서의 길이입니다. BERT 임베딩은 길이를 자르지만(truncate), LDA의 카운트 방식은 어떤 길이에서도 작동합니다. 임베딩 모델의 컨텍스트보다 긴 문서의 경우, 청크 단위로 나누어 합산(chunk + aggregate)하거나 LDA를 사용해 보세요.

## 활용 방법 (Use It)

2026년 스택:

- **BERTopic.** 짧은 텍스트 및 의미론적(semantics) 요소가 중요한 모든 경우에 사용하는 기본 도구입니다.
- **`gensim.models.LdaModel`.** 프로덕션 환경을 위한 클래식 LDA로, 성숙하고 검증된(battle-tested) 모델입니다.
- **`sklearn.decomposition.LatentDirichletAllocation`.** 실험용으로 사용하기 쉬운 LDA입니다.
- **NMF.** 비음수 행렬 분해(Non-negative matrix factorization)입니다. LDA의 빠른 대안이며, 짧은 텍스트에서 유사한 품질을 제공합니다.
- **Top2Vec.** BERTopic과 유사한 설계 방식입니다. 커뮤니티 규모는 작지만 일부 벤치마크에서 우수한 성능을 보입니다.
- **FASTopic.** 최신 기술로, 매우 큰 코퍼스(corpora)에서 BERTopic보다 빠릅니다.
- **LLM 기반 레이블링 (LLM-based labeling).** 어떤 클러스터링이든 수행한 후, 모델에 프롬프트를 입력하여 각 클러스터의 이름을 지정합니다.

## Ship It (실행하기)

`outputs/skill-topic-picker.md`로 저장하세요:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 연습 문제 (Exercises)

1. **쉬움.** 20 Newsgroups 데이터셋에 대해 5개의 토픽으로 LDA를 학습시켜 보세요. 토픽당 상위 10개 단어를 출력하세요. 각 토픽에 수동으로 라벨을 붙여 보세요. 알고리즘이 실제 카테고리를 찾아냈나요?
2. **중간.** 동일한 20 Newsgroups 서브셋에 BERTopic을 학습시켜 보세요. LDA와 비교하여 발견된 토픽의 수, 상위 단어, 그리고 정성적 일관성(qualitative coherence)을 비교해 보세요. 어떤 방법이 실제 카테고리를 더 명확하게 드러내나요?
3. **어려움.** 코퍼스(corpus)에 대해 LDA와 BERTopic 모두에 대해 c_v coherence를 계산하세요. 각 모델을 5, 10, 20, 50개의 토픽으로 실행하세요. 토픽 수 대비 일관성(coherence)을 그래프로 그리세요. 토픽 수 변화에 따라 어떤 방법이 더 안정적인지 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| Topic | 코퍼스가 다루는 주제 | 단어들에 대한 확률 분포(LDA) 또는 유사한 문서들의 클러스터(BERTopic)입니다. |
| Mixed membership | 문서가 여러 토픽을 가짐 | LDA는 각 문서에 모든 토픽에 대한 분포를 할당합니다. |
| UMAP | 차원 축소 | 국소적 구조를 보존하는 매니폴드 학습(manifold learning)이며, BERTopic에서 사용됩니다. |
| HDBSCAN | 밀도 기반 클러스터링 | 가변 크기의 클러스터를 찾으며, 이상치(outlier)에 대해 "noise" 레이블(-1)을 생성합니다. |
| c_v coherence | 토픽 품질 지표 | 슬라이딩 윈도우 내 상위 토픽 단어들의 평균 점별 상호 정보량(pointwise mutual information)입니다. |

## 추가 읽을거리 (Further Reading)

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 논문입니다.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 논문입니다.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_v 등을 소개한 논문입니다.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 실무 참조용 문서입니다. 훌륭한 예제들이 포함되어 있습니다.
