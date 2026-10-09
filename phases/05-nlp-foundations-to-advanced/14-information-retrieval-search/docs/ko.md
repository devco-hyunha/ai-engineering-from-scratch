# 정보 검색 및 검색

> BM25는 정밀하지만 취약합니다. Dense는 넓은 범위를 포착하지만 키워드를 놓칩니다. Hybrid는 2026년 기본값입니다. 나머지는 모두 튜닝입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 5단계 · 04강 (GloVe, FastText, Subword)
**시간:** 약 75분

## 문제점

사용자가 "누군가 돈을 얻기 위해 거짓말을 하면 어떻게 되는가"를 입력하고, 실제로 이를 다루는 조항인 "IPC 제420조"를 찾으려 합니다. 키워드 검색은 공유 어휘가 없어 이를 완전히 놓칩니다. 임베딩이 법률 텍스트로 학습되지 않았다면 시맨틱 검색도 이를 놓칩니다. 실제 검색은 두 가지를 모두 처리해야 합니다.

IR는 모든 RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) 시스템, 모든 검색 바, 모든 문서 사이트의 퍼지 검색을 뒷받침하는 파이프라인입니다. 프로덕션에서 작동하는 2026년 아키텍처는 단일 방법이 아닙니다. 이전 방법의 실패를 각각 보완하는 상호 보완적인 방법들의 체인입니다.

이 강의는 각 부분을 구축하고, 각 부분이 어떤 실패를 보완하는지 명시합니다.

## 개념

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

네 가지 계층입니다. 필요한 것을 선택하세요.

1. **Sparse 검색 (BM25).** 빠르고 정확한 일치에 강하며, 시맨틱에는 취약합니다. 역전 인덱스(inverted index)에서 실행됩니다. 수백만 개의 문서에서 쿼리당 10ms 미만입니다. 조항 참조, 제품 코드, 오류 메시지, 명명된 엔티티를 정확히 찾아냅니다.
2. **Dense 검색.** 쿼리와 문서를 벡터로 인코딩합니다. 최근접 이웃 검색을 수행합니다. 패러프레이즈와 시맨틱 유사성을 포착합니다. 한 글자 차이로 다른 정확한 키워드 일치는 놓칩니다. FAISS나 벡터 DB를 사용하면 쿼리당 50-200ms가 소요됩니다.
3. **Fusion.** Sparse와 Dense의 랭킹된 목록을 병합합니다. 상호 랭킹 융합 (RRF)(Reciprocal Rank Fusion (RRF))는 원시 점수(서로 다른 스케일에 존재)를 무시하고 랭킹 위치만 사용하므로 쉬운 기본값입니다. 특정 도메인에서 하나의 신호가 지배적이라면 가중치 융합이 옵션입니다.
4. **Cross-encoder 리랭킹.** Fusion에서 상위 30개를 가져옵니다. Cross-encoder (쿼리 + 문서를 함께 입력하여 각 쌍을 점수화)를 실행합니다. 상위 5개를 유지합니다. Cross-encoder는 Bi-encoder보다 쌍당 속도가 느리지만 훨씬 더 정확합니다. 상위 30개에만 실행함으로써 비용을 상쇄합니다.

3-way 검색(BM25 + 밀집 검색 + SPLADE와 같은 학습된 희소 검색)은 2026년 벤치마크에서 2-way 검색보다 성능이 뛰어나지만, 학습된 희소 인덱스를 위한 인프라가 필요합니다. 대부분의 팀에게는 2-way 검색 + 크로스 인코더 리랭킹이 최적의 균형점입니다.

```figure
gx-hybrid-retrieval
```

## 구현하기

### 1단계: BM25를 처음부터 구현하기

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

알아두어야 할 두 가지 매개변수가 있습니다. `k1=1.5`는 용어 빈도 포화를 제어합니다. 값이 높을수록 용어 반복에 더 많은 가중치를 부여합니다. `b=0.75`는 길이 정규화를 제어합니다. 0은 문서 길이를 무시하고, 1은 완전히 정규화합니다. 기본값은 원 논문에서 로버트슨이 권장한 값이며, 거의 튜닝이 필요하지 않습니다.

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

임베딩을 L2 정규화하여 내적(dot product)이 코사인 유사도와 같아지도록 합니다. `all-MiniLM-L6-v2`는 384차원으로, 빠르고 대부분의 영어 검색에 충분히 강력합니다. 다국어 작업에는 `paraphrase-multilingual-MiniLM-L12-v2`를 사용하세요. 최고 정확도를 원한다면 `bge-large-en-v1.5` 또는 `e5-large-v2`를 사용하세요.

### 3단계: 상호 랭킹 융합 (RRF)

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 상수는 원 RRF 논문에서 유래했습니다. `k`가 높으면 순위 차이의 기여도가 평평해지고, `k`가 낮으면 상위 순위가 지배하게 됩니다. 60은 공식적으로 발표된 기본값이며, 거의 튜닝이 필요하지 않습니다.

### 4단계: 하이브리드 검색 + 리랭킹

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

세 단계가 조합됩니다. BM25는 어휘적 일치 항목을 찾습니다. 밀집 검색은 의미적 일치 항목을 찾습니다. RRF는 점수 보정 없이 두 순위를 병합합니다. 크로스 인코더는 쿼리-문서 쌍을 함께 사용하여 상위 30개 항목을 재점수화하며, 이는 바이 인코더가 놓친 세밀한 관련성을 포착합니다. 상위 5개를 유지하세요.

### 5단계: 평가

| 지표 | 의미 |
|--------|---------|
| Recall@k | 올바른 문서가 존재하는 쿼리 중, 상위 k개에 포함되는 빈도는 얼마나 되는가? |
| MRR (평균 역 순위) | 첫 번째 관련 문서의 순위의 역수 평균. |
| nDCG@k | 관련/비관련의 이진 값뿐만 아니라 관련성의 등급을 고려합니다. |

RAG의 경우, 검색기의 **Recall@k**가 가장 중요한 수치입니다. 올바른 구문이 검색된 집합에 없다면, 사용자는 답변할 수 없습니다.

디버깅 팁: 실패한 쿼리에 대해 희소 검색과 밀집 검색의 순위를 비교해 보세요. 한쪽이 올바른 문서를 찾고 다른 쪽이 찾지 못한다면, 어휘 불일치(해결: 누락된 부분을 추가)나 의미적 모호성(해결: 더 나은 임베딩이나 리랭커 사용)이 존재하는 것입니다.

## 사용하기

2026년 스택:

| 규모 | 스택 |
|-------|-------|
| 1k-100k 문서 | 메모리 내 BM25 + `all-MiniLM-L6-v2` 임베딩 + RRF. 별도 DB 없음. |
| 100k-10M 문서 | 밀집 검색용 FAISS 또는 pgvector + BM25용 Elasticsearch / OpenSearch. 병렬로 실행. |
| 10M+ 문서 | 하이브리드 지원이 있는 Qdrant / Weaviate / Vespa / Milvus. 상위 30개에 대해 교차 인코더 리랭킹 수행. |
| 최고 품질 프론티어 | 3-way (BM25 + 밀집 + SPLADE) + ColBERT 후기 상호작용 리랭킹 |

무엇을 선택하든 평가에 예산을 배정하세요. 엔드투엔드 RAG 정확도를 벤치마킹하기 전에 검색 재현율을 벤치마킹하세요. 리더는 리트리버가 놓친 것을 고칠 수 없습니다.

### 2026년 프로덕션 RAG에서 얻은 뼈아픈 교훈

- **RAG 실패의 80%는 모델이 아니라 인제이션과 청킹에서 비롯됩니다.** 팀은 LLM을 교체하고 프롬프트를 튜닝하는 데 몇 주를 보내는 동안, 리트리버는 세 번째 쿼리마다 조용히 잘못된 컨텍스트를 반환합니다. 청킹을 먼저 고치세요.
- **청킹 전략이 청킹 크기보다 중요합니다.** 고정 크기 분할은 표, 코드, 중첩 헤더를 깨뜨립니다. 문장 인식 방식이 기본이며, 기술 문서 및 제품 매뉴얼의 경우 시맨틱 또는 LLM 기반 청킹이 효과적입니다.
- **부모 문서 패턴.** 정확도를 위해 작은 "자식" 청크를 검색하세요. 동일한 부모 섹션에서 여러 자식 청크가 나타날 경우, 컨텍스트를 보존하기 위해 부모 블록으로 교체하세요. 이는 재학습 없이 답변 품질을 일관되게 높입니다.
- **k_rerank=3이 일반적으로 최적입니다.** 그 이상의 추가 청크는 토큰 비용과 생성 지연을 증가시키며 답변 품질을 높이지 못합니다. k=8이 k=3보다 여전히 더 좋다면, 리랭커가 성능을 발휘하지 못하고 있습니다.
- **HyDE / 쿼리 확장.** 쿼리에서 가상의 답변을 생성하고, 이를 임베딩하여 검색하세요. 짧은 질문과 긴 문서 간의 표현 격차를 해소합니다. 학습 없이 무료 정밀도 향상을 제공합니다.
- **컨텍스트 예산은 8K 토큰 미만.** 그 한계에서 일관된 히트가 발생한다면, 리랭커 임계값이 너무 느슨합니다.
- **모든 것을 버전화하세요.** 프롬프트, 청킹 규칙, 임베딩 모델, 리랭커. 모든 드리프트는 답변 품질을 조용히 깨뜨립니다. 충실도, 컨텍스트 정밀도, unanswered-question rate에 대한 CI 게이트는 사용자가 보기 전에 회귀를 차단합니다.
- **3방향 검색(BM25 + 밀집 + SPLADE와 같은 학습된 희소)은 2026년 벤치마크에서 2방향 검색보다 성능이 우수합니다.** 특히 고유 명사와 의미론이 혼합된 쿼리에 효과적입니다. 인프라가 SPLADE 인덱스를 지원할 때 출시하세요.

2026년 산업 측정 결과에 따르면, 적절한 검색 설계는 환각(Hallucination)을 70-90% 감소시킵니다. RAG 성능 향상은 대부분 더 나은 검색에서 나오며, 모델 미세 조정(Fine-tuning)에서 나오지 않습니다.

## 출시하기

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

## 연습 문제

1. **쉬움.** 500개 문서 코퍼스에 위 `hybrid_search`를 구현하세요. 20개 쿼리를 테스트하세요. BM25 전용, 밀집 전용, 하이브리드 검색의 Recall@5를 비교하세요.
2. **중간.** MRR 계산을 추가하세요. 알려진 정답 문서가 있는 각 테스트 쿼리에 대해 BM25, 밀집, 하이브리드 랭킹에서 정답 문서의 순위를 찾으세요. 각 랭킹의 MRR을 보고하세요.
3. **어려움.** MultipleNegativesRankingLoss (Sentence Transformers)를 사용하여 도메인에 밀집 인코더를 미세 조정(Fine-tuning)하세요. 500개 쿼리-문서 쌍으로 훈련 세트를 구축하세요. 미세 조정 전후의 Recall을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BM25 | 키워드 검색 | Okapi BM25. 용어 빈도, IDF, 길이로 문서를 점수화합니다. |
| 밀집 검색(Dense Retrieval) | 벡터 검색 | 쿼리 + 문서를 벡터로 인코딩하고, 최근접 이웃을 찾습니다. |
| 바이인코더(Bi-encoder) | 임베딩 모델 | 쿼리와 문서를 독립적으로 인코딩합니다. 쿼리 시간에 빠릅니다. |
| 크로스인코더(Cross-encoder) | 리랭커(Reranker) 모델 | 쿼리 + 문서를 함께 인코딩합니다. 느리지만 정확합니다. |
| RRF | 랭킹 융합 | `1/(k + rank)`를 합산하여 두 랭킹을 결합합니다. |
| Recall@k | 검색 지표 | 관련 문서가 상위 k개에 포함된 쿼리의 비율입니다. |

## 추가 읽기

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25에 대한 결정적인 자료입니다.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, 표준 바이인코더입니다.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 밀집 검색과의 격차를 줄이는 학습된 희소 검색기입니다.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 논문입니다.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 후기 상호작용 검색입니다.
