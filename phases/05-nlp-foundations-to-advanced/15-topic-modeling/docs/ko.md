# 주제 모델링 — LDA와 BERTopic

> LDA: 문서는 주제들의 혼합이며, 주제는 단어에 대한 분포입니다. BERTopic: 문서는 임베딩 공간에서 클러스터링되며, 클러스터가 주제입니다. 목표는 같지만 분해 방식이 다릅니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 5단계 · 03강 (Word2Vec)
**시간:** 약 45분

## 문제점

10,000개의 고객 지원 티켓, 50,000개의 뉴스 기사, 또는 200,000개의 트윗이 있습니다. 전체를 읽지 않고도 컬렉션의 내용을 파악해야 합니다. 레이블이 지정된 카테고리가 없으며, 카테고리가 몇 개 존재하는지도 알지 못합니다.

주제 모델링은 지도 학습 없이 이 질문에 답합니다. 코퍼스를 입력하면 일관된 주제들의 작은 집합과 각 문서에 대한 주제 분포를 반환합니다.

두 가지 알고리즘 계열이 지배적입니다. LDA (2003)는 각 문서를 잠재적 주제들의 혼합으로, 각 주제를 단어에 대한 분포로 취급합니다. 추론은 베이지안 방식입니다. 혼합 멤버십 주제 할당과 설명 가능한 단어 수준 확률 분포가 필요한 생산 환경에서는 여전히 사용되고 있습니다.

BERTopic (2020)은 BERT로 문서를 인코딩하고, UMAP으로 차원을 축소하며, HDBSCAN으로 클러스터링하고, 클래스 기반 TF-IDF로 주제 단어를 추출합니다. 짧은 텍스트, 소셜 미디어, 단어 겹침보다 의미적 유사성이 더 중요한 모든 상황에서 우세합니다. 한 문서가 하나의 주제를 가지므로 긴 형식 콘텐츠에는 한계가 있습니다.

이 강의는 두 방식 모두에 대한 직관을 구축하고, 주어진 코퍼스에 대해 어떤 방식을 선택해야 하는지 알려줍니다.

## 개념

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 생성 스토리.** 각 주제는 단어에 대한 분포입니다. 각 문서는 주제들의 혼합입니다. 문서에서 단어를 생성하려면, 문서의 혼합에서 주제를 샘플링한 후, 해당 주제의 분포에서 단어를 샘플링합니다. 추론은 이를 역전합니다: 관측된 단어가 주어지면, 문서별 주제 분포와 주제별 단어 분포를 추론합니다. Collapsed Gibbs 샘플링이나 변분 베이지안 방식이 이를 계산합니다.

LDA의 주요 출력:

- `doc_topic`: `(n_docs, n_topics)` 행렬, 각 행의 합은 1입니다 (문서의 주제 혼합).
- `topic_word`: `(n_topics, vocab_size)` 행렬, 각 행의 합이 1입니다 (주제의 단어 분포).

**BERTopic 파이프라인.**

1. 문장 트랜스포머 (예: `all-MiniLM-L6-v2`)로 각 문서를 인코딩합니다. 384차원 벡터가 됩니다.
2. UMAP를 사용하여 차원을 약 5차원으로 줄입니다. BERT 임베딩은 클러스터링에 너무 높은 차원을 가집니다.
3. HDBSCAN으로 클러스터링합니다. 밀도 기반이며, 크기가 다양한 클러스터와 "아웃라이어" 레이블을 생성합니다.
4. 각 클러스터에 대해 클러스터 내 문서의 클래스 기반 TF-IDF를 계산하여 상위 단어를 추출합니다.

출력은 문서당 하나의 주제 (그리고 -1 아웃라이어 레이블)입니다. 선택적으로, HDBSCAN의 확률 벡터를 통해 부드러운 멤버십을 얻을 수 있습니다.

```figure
topic-drift
```

## 구현하기

### 1단계: scikit-learn을 통한 LDA

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

참고: 불용어(stopwords)가 제거되며, min_df와 max_df는 희소하고 흔한 용어를 필터링합니다. LDA는 원시 카운트를 기대하므로 CountVectorizer (TfidfVectorizer가 아님)를 사용합니다.

### 2단계: BERTopic (프로덕션)

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

`Topic != -1` 필터는 BERTopic의 아웃라이어 버킷(HDBSCAN이 클러스터링하지 못한 문서)을 제거합니다. `min_topic_size`는 HDBSCAN의 최소 클러스터 크기를 제어합니다. BERTopic 라이브러리 기본값은 10입니다. 이 예제에서는 강의 규모에 맞춰 명시적으로 15로 설정합니다. 10,000개 이상의 문서 코퍼레이션을 사용하는 경우, 50 또는 100으로 늘리세요.

### 3단계: 평가

두 방법 모두 주제 단어를 출력합니다. 문제는 그 단어들이 응집되는지 여부입니다.

- **주제 응집도(c_v).** 슬라이딩 윈도우 컨텍스트에서 상위 단어 쌍의 NPMI (정규화된 포인트와이즈 상호 정보)를 결합하고, 점수를 주제 벡터로 집계한 후 코사인 유사도로 비교합니다. 높을수록 좋습니다. `gensim.models.CoherenceModel`를 `coherence="c_v"`와 함께 사용하세요.
- **주제 다양성.** 모든 주제의 상위 단어 중 고유 단어의 비율입니다. 높을수록 좋습니다 (주제들이 겹치지 않음).
- **정성적 검사.** 각 주제의 상위 단어를 읽어보세요. 실제 대상을 지칭하나요? 인간의 판단은 여전히 최후의 방어선입니다.

## 어떤 것을 선택해야 할 때

| 상황 | 선택 |
|-----------|------|
| 짧은 텍스트 (트윗, 리뷰, 헤드라인) | BERTopic |
| 주제 혼합이 있는 긴 문서 | LDA |
| GPU 없음 / 제한된 컴퓨팅 자원 | LDA 또는 NMF |
| 문서 수준 다중 주제 분포가 필요 | LDA |
| 주제 라벨링을 위한 LLM 통합 | BERTopic (직접 지원) |
| 리소스가 제한된 엣지 배포 | LDA |
| 최대 의미적 일관성 | BERTopic |

가장 중요한 실무적 고려 사항은 문서 길이입니다. BERT 임베딩은 잘라내며, LDA 카운팅은 어떤 길이에서도 작동합니다. 임베딩 모델의 컨텍스트보다 긴 문서의 경우, 청킹 후 집계하거나 LDA를 사용하세요.

## 사용하기

2026년 스택:

- **BERTopic.** 짧은 텍스트와 의미가 중요한 모든 경우의 기본값입니다.
- **`gensim.models.LdaModel`.** 프로덕션용 클래식 LDA, 성숙하고 검증된 도구입니다.
- **`sklearn.decomposition.LatentDirichletAllocation`.** 실험용 쉬운 LDA입니다.
- **NMF.** 비음수 행렬 분해. LDA의 빠른 대안이며, 짧은 텍스트에서 비교 가능한 품질을 제공합니다.
- **Top2Vec.** BERTopic과 유사한 설계입니다. 커뮤니티는 작지만 일부 벤치마크에서 좋은 성능을 보입니다.
- **FASTopic.** 최신 도구로, 매우 대규모 코퍼스에서 BERTopic보다 빠릅니다.
- **LLM 기반 라벨링.** 임의의 클러스터링을 실행한 후, 모델에 프롬프트하여 각 클러스터의 이름을 지정합니다.

## 출시하기

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

## 연습 문제

1. **쉬움.** 20 Newsgroups 데이터셋에 5개 주제를 가진 LDA를 적합하세요. 각 주제별 상위 10개 단어를 출력하세요. 각 주제를 수동으로 라벨링하세요. 알고리즘이 실제 카테고리를 찾았나요?
2. **중간.** 동일한 20 Newsgroups 하위 집합에 BERTopic을 적합하세요. 발견된 주제 수, 상위 단어, LDA 대비 질적 일관성을 비교하세요. 어느 방법이 실제 카테고리를 더 명확하게 드러내나요?
3. **어려움.** 코퍼스에 대해 LDA와 BERTopic 모두의 c_v 일관성을 계산하세요. 각각 5, 10, 20, 50개 주제로 실행하세요. 주제 수 대비 일관성을 플롯하세요. 주제 수 변화에 대해 어느 방법이 더 안정적인지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 주제 | 코퍼스가 다루는 것 | 단어에 대한 확률 분포 (LDA) 또는 유사한 문서의 클러스터 (BERTopic). |
| 혼합 멤버십 | 문서가 여러 주제에 속함 | LDA는 각 문서에 모든 주제에 대한 분포를 할당합니다. |
| UMAP | 차원 축소 | 지역 구조를 보존하는 매니폴드 학습; BERTopic에서 사용됩니다. |
| HDBSCAN | 밀도 클러스터링 | 크기가 다양한 클러스터를 찾습니다; 이상치에 대해 "noise" 레이블(-1)을 생성합니다. |
| c_v 일관성 | 주제 품질 지표 | 슬라이딩 윈도우 내에서 상위 주제 단어들의 평균 점별 상호 정보입니다. |

## 추가 읽기

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 논문입니다.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 논문입니다.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_v 및 관련 지표들을 도입한 논문입니다.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 생산 환경 참고 자료입니다. 훌륭한 예제들이 있습니다.
