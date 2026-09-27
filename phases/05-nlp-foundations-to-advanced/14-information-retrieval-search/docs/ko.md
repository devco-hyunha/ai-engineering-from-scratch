# 정보 검색 및 탐색 (Information Retrieval and Search)

> BM25는 정밀하지만 취약합니다. Dense는 넓게 탐색하지만 키워드를 놓칩니다. Hybrid가 2026년의 기본값입니다. 그 외의 모든 것은 튜닝입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 문제 (The Problem)

사용자가 "누군가 돈을 벌기 위해 거짓말을 하면 어떻게 되나요"라고 입력하면, 실제로 이를 다루는 법 조항인 "IPC 제420조"를 찾기를 기대합니다. 키워드 검색은 이를 완전히 놓칩니다(공유된 어휘가 없음). 시맨틱 검색은 임베딩이 법률 텍스트로 학습되지 않았다면 이를 놓칩니다. 실제 검색은 이 두 가지를 모두 처리해야 합니다.

IR은 모든 RAG 시스템, 모든 검색창, 모든 문서 사이트의 퍼지 검색(fuzzy lookup) 아래에 있는 파이프라인입니다. 프로덕션에서 작동하는 2026년 아키텍처는 단일 방법이 아닙니다. 그것은 상호 보완적인 방법들의 체인이며, 각 방법은 이전 방법의 실패를 보완합니다.

이 레슨에서는 각 구성 요소를 구축하고, 각 요소가 어떤 실패를 보완하는지 설명합니다.

## 개념 (The Concept)

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

네 가지 계층이 있습니다. 필요한 것을 선택해 보세요.

1. **희소 검색 (Sparse retrieval, BM25).** 빠르고 정확한 일치에는 정밀하지만, 의미론적(semantics) 측면에서는 형편없습니다. 역색인(inverted index) 위에서 실행됩니다. 수백만 개의 문서에 대해 쿼리당 10ms 미만입니다. 법 조항 참조, 제품 코드, 오류 메시지, 개체명(named entities)을 정확하게 찾아줍니다.
2. **밀집 검색 (Dense retrieval).** 쿼리와 문서를 벡터로 인코딩합니다. 최근접 이웃(Nearest neighbor) 검색을 수행합니다. 패러프레이징(paraphrases)과 의미적 유사성을 포착합니다. 한 글자만 달라도 정확한 키워드 일치를 놓칠 수 있습니다. FAISS 또는 벡터 DB를 사용하면 쿼리당 50-200ms가 소요됩니다.
3. **퓨전 (Fusion).** 희소 검색과 밀집 검색의 순위 목록을 병합합니다. Reciprocal Rank Fusion (RRF)은 원시 점수(서로 다른 스케일에 있음)를 무시하고 순위 위치만 사용하기 때문에 쉬운 기본값입니다. 특정 도메인에서 하나의 신호가 지배적이라는 것을 안다면 가중치 퓨전(Weighted fusion)을 옵션으로 사용할 수 있습니다.
4. **크로스 인코더 재순위화 (Cross-encoder rerank).** 퓨전에서 상위 30개를 가져옵니다. 크로스 인코더(쿼리와 문서를 함께 입력하여 각 쌍의 점수를 매김)를 실행합니다. 상위 5개를 유지합니다. 크로스 인코더는 바이 인코더(bi-encoders)보다 쌍당 속도는 느리지만 훨씬 더 정확합니다. 상위 30개에 대해서만 실행함으로써 비용을 분산(amortize)시킵니다.

3방향 검색(Three-way retrieval, BM25 + dense + SPLADE와 같은 학습된 희소 검색)은 2026년 벤치마크에서 2방향보다 성능이 뛰어나지만, 학습된 희소 인덱스를 위한 인프라가 필요합니다. 대부분의 팀에게는 2방향에 크로스 인코더 재순위화를 더하는 것이 최적의 지점(sweet spot)입니다.

```figure
gx-hybrid-retrieval
```

## 구축하기 (Build It)

### 1단계: BM25 처음부터 구현하기

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

알아두어야 할 두 가지 파라미터입니다. `k1=1.5`는 단어 빈도 포화(term-frequency saturation)를 제어합니다. 값이 높을수록 단어 반복에 더 많은 가중치를 둡니다. `b=0.75`는 길이 정규화(length normalization)를 제어합니다. 0은 문서 길이를 무시하고, 1은 완전히 정규화합니다. 기본값은 원본 논문의 Robertson의 권장 사항이며 튜닝이 거의 필요하지 않습니다.

### 2단계: 바이 인코더를 사용한 밀집 검색

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

내적(dot product)이 코사인 유사도와 같아지도록 임베딩을 L2 정규화합니다. `all-MiniLM-L6-v2`는 384차원이며 빠르고 대부분의 영어 검색에 충분히 강력합니다. 다국어 작업에는 `paraphrase-multilingual-MiniLM-L12-v2`를 사용하세요. 최고의 정확도를 위해서는 `bge-large-en-v1.5` 또는 `e5-large-v2`를 사용하세요.

### 3단계: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 상수는 원본 RRF 논문에서 가져온 것입니다. `k`가 높을수록 순위 차이의 기여도가 평탄해지며, `k`가 낮을수록 상위 순위가 지배적이 됩니다. 60은 발표된 기본값이며 튜닝이 거의 필요하지 않습니다.

### 4단계: 하이브리드 검색 + 재순위화

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

세 단계가 구성되었습니다. BM25는 어휘적 일치를 찾습니다. Dense는 의미적 일치를 찾습니다. RRF는 점수 보정(score calibration) 없이 두 순위를 병합합니다. 크로스 인코더는 쿼리-문서 쌍을 함께 사용하여 상위 30개를 다시 점수 매기며, 이는 바이 인코더가 놓친 미세한 관련성을 포착합니다. 상위 5개를 유지하세요.

### 5단계: 평가

| 지표 | 의미 |
|--------|---------|
| Recall@k | 정답 문서가 존재하는 쿼리 중, 상위 k개 안에 포함되는 빈도는 얼마인가요? |
| MRR (Mean Reciprocal Rank) | 첫 번째 관련 문서의 1/rank의 평균입니다. |
| nDCG@k | 단순한 이진 관련/비관련이 아닌, 관련성의 단계(gradations)를 고려합니다. |

특히 RAG의 경우, 검색기(retriever)의 **Recall@k**가 가장 중요한 수치입니다. 올바른 구절이 검색된 세트에 없다면 독자는 답을 할 수 없습니다.

디버깅 팁: 실패하는 쿼리에 대해 희소 검색과 밀집 검색의 순위를 비교(diff)해 보세요. 하나는 올바른 문서를 찾고 다른 하나는 찾지 못한다면, 어휘 불일치(vocabulary mismatch, 해결책: 누락된 절반을 추가) 또는 의미적 모호성(semantic ambiguity, 해결책: 더 나은 임베딩 또는 재순위화 모델)이 있는 것입니다.

## 사용하기 (Use It)

2026년 스택:

| 규모 | 스택 |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF. 별도의 DB 없음. |
| 100k-10M docs | 밀집 검색용 FAISS 또는 pgvector + BM25용 Elasticsearch / OpenSearch. 병렬 실행. |
| 10M+ docs | 하이브리드 지원 Qdrant / Weaviate / Vespa / Milvus. 상위 30개에 대해 크로스 인코더 재순위화. |
| 최고 품질 프런티어(Best-quality frontier) | 3방향(BM25 + dense + SPLADE와 같은 학습된 희소 검색) + ColBERT 후기 상호작용(late-interaction) 재순위화 |

무엇을 선택하든 평가를 위한 예산을 확보하세요. 엔드 투 엔드(end-to-end) RAG 정확도를 벤치마킹하기 전에 검색 재현율(retrieval recall)을 먼저 벤치마킹하세요. 검색기가 놓친 것은 독자가 고칠 수 없습니다.

### 2026년 프로덕션 RAG에서 얻은 값진 교훈들

- **RAG 실패의 80%는 모델이 아니라 인제스션(ingestion)과 청킹(chunking)에서 기인합니다.** 팀들은 세 번째 쿼리마다 검색기가 잘못된 컨텍스트를 반환하고 있는데도 LLM을 교체하고 프롬프트를 튜닝하는 데 몇 주를 보냅니다. 청킹을 먼저 해결하세요.
- **청킹 전략은 청크 크기보다 중요합니다.** 고정 크기 분할은 표, 코드, 중첩된 헤더를 깨뜨립니다. 문장 인식(Sentence-aware) 방식이 기본값이며, 기술 문서나 제품 매뉴얼의 경우 의미론적(semantic) 또는 LLM 기반 청킹이 효과적입니다.
- **부모-자식 문서 패턴(Parent-doc pattern).** 정밀도를 위해 작은 "자식(child)" 청크를 검색합니다. 동일한 부모 섹션에서 여러 자식이 나타나면, 컨텍스트를 유지하기 위해 부모 블록으로 교체합니다. 이는 재학습 없이도 답변 품질을 일관되게 높여줍니다.
- **`k_rerank=3`이 보통 최적입니다.** 그 이상의 추가 청크는 답변 품질을 높이지 않으면서 토큰 비용과 생성 지연 시간만 추가합니다. 만약 `k=8`이 여전히 `k=3`보다 낫다면, 재순위화 모델의 성능이 떨어지는 것입니다.
- **HyDE / 쿼리 확장(query expansion).** 쿼리로부터 가상의 답변을 생성하고, 이를 임베딩하여 검색합니다. 짧은 질문과 긴 문서 사이의 표현 차이를 메워줍니다. 학습 없이 정밀도를 높일 수 있는 무료 방법입니다.
- **컨텍스트 예산 8K 토큰 미만.** 이 한계에서 일관된 히트가 발생한다면 재순위화 임계값이 너무 느슨한 것입니다.
- **모든 것을 버전 관리하세요.** 프롬프트, 청킹 규칙, 임베딩 모델, 재순위화 모델. 어떤 변화(drift)라도 답변 품질을 조용히 망가뜨릴 수 있습니다. 충실도(faithfulness), 컨텍스트 정밀도(context precision), 미답변 질문 비율(unanswered-question rate)에 대한 CI 게이트를 통해 사용자가 인지하기 전에 퇴보를 차단하세요.
- **3방향 검색(BM25 + dense + SPLADE와 같은 학습된 희소 검색)은 2방향보다 성능이 뛰어납니다.** 특히 고유 명사와 의미론이 섞인 쿼리에서 그렇습니다. 인프라가 SPLADE 인덱스를 지원할 때 도입하세요.

2026년 산업 측정에 따르면 적절한 검색 설계는 환각(hallucination)을 70-90% 줄여줍니다. 대부분의 RAG 성능 향상은 모델 미세 조정이 아니라 더 나은 검색에서 옵니다.

## Ship It

`outputs/skill-retrieval-picker.md`로 저장하세요:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 연습 문제 (Exercises)

1. **쉬움.** 위에서 구현한 `hybrid_search`를 500개 문서 코퍼스에 구현해 보세요. 20개의 쿼리로 테스트하세요. BM25 전용, 밀집 전용, 하이브리드 방식의 Recall@5를 비교해 보세요.
2. **중간.** MRR 계산을 추가해 보세요. 정답 문서가 알려진 각 테스트 쿼리에 대해, BM25, 밀집, 하이브리드 순위에서 정답 문서의 순위를 찾으세요. 각각의 MRR을 보고하세요.
3. **어려움.** MultipleNegativesRankingLoss(Sentence Transformers)를 사용하여 도메인에 맞는 밀집 인코더를 미세 조정해 보세요. 500개의 쿼리-문서 쌍으로 학습 세트를 구축하세요. 미세 조정 전후의 Recall을 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BM25 | 키워드 검색 | Okapi BM25. 단어 빈도, IDF, 길이에 따라 문서 점수를 매깁니다. |
| Dense retrieval | 벡터 검색 | 쿼리 + 문서를 벡터로 인코딩하여 최근접 이웃을 찾습니다. |
| Bi-encoder | 임베딩 모델 | 쿼리와 문서를 독립적으로 인코딩합니다. 쿼리 시점에 빠릅니다. |
| Cross-encoder | 재순위화 모델 | 쿼리 + 문서를 함께 인코딩합니다. 느리지만 정확합니다. |
| RRF | 순위 퓨전 | `1/(k + rank)`를 합산하여 두 순위를 결합합니다. |
| Recall@k | 검색 지표 | 관련 문서가 상위 k개 안에 포함되는 쿼리의 비율입니다. |

## 더 읽어볼 거리 (Further Reading)

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 결정적인 BM25 다룸.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — 표준적인 바이 인코더인 DPR.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 밀집 검색과의 격차를 줄이는 학습된 희소 검색기.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 논문.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 후기 상호작용(late-interaction) 검색.
