# 임베딩 모델(Embedding Models) — 2026년 심층 분석(The 2026 Deep Dive)

> Word2Vec은 단어당 하나의 벡터를 제공했습니다. 현대적인 임베딩 모델은 구절(passage)당 하나의 벡터를 제공하며, 교차 언어(cross-lingual) 지원은 물론 희소(sparse), 밀집(dense), 다중 벡터(multi-vector) 뷰를 제공하고 인덱스 크기에 맞춰 조정할 수 있습니다. 잘못된 모델을 선택하면 RAG가 잘못된 정보를 검색하게 됩니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## 문제점 (The Problem)

귀하의 RAG 시스템이 40%의 확률로 잘못된 구절을 검색합니다. 원인은 벡터 데이터베이스나 프롬프트인 경우가 거의 없습니다. 바로 임베딩 모델이 문제입니다.

2026년에 임베딩 모델을 선택한다는 것은 다음 다섯 가지 축을 기준으로 결정함을 의미합니다:

1. **밀집(Dense) vs 희소(Sparse) vs 멀티 벡터(Multi-vector).** 구절당 하나의 벡터를 사용할 것인지, 토큰당 하나의 벡터를 사용할 것인지, 아니면 희소 가중치 단어 가방(sparse weighted bag of words)을 사용할 것인지 결정해야 합니다.
2. **언어 커버리지(Language coverage).** 영어 전용 작업에서는 여전히 단일 언어 영어 모델이 우세합니다. 혼합 코퍼스(corpora)의 경우에는 다국어 모델이 유리합니다.
3. **컨텍스트 길이(Context length).** 512 토큰 vs 8,192 vs 32,768 — 그리고 실제 유효 용량은 종종 광고된 최대치의 60-70%에 불과합니다.
4. **차원 예산(Dimension budget).** 전체 정밀도에서 3,072개의 부동 소수점(floats)은 벡터당 12KB입니다. 1억 개의 벡터가 있다면 저장 비용은 월 $1,300입니다. 마트료시카 절단(Matryoshka truncation)은 이를 4배 줄여줍니다.
5. **오픈(Open) vs 호스팅형(Hosted).** 오픈 웨이트(Open-weight)는 스택과 데이터를 직접 제어할 수 있음을 의미합니다. 호스팅형은 제어권을 대가로 항상 최신 상태를 유지하는 것을 의미합니다.

이 레슨에서는 단순히 지난 분기에 유행했던 것이 아니라, 근거에 기반하여 선택할 수 있도록 트레이드오프(tradeoffs)를 명시합니다.

## 개념 (The Concept)

![밀집, 희소 및 멀티 벡터 임베딩](../assets/embedding-modes.svg)

**밀집 임베딩 (Dense embeddings).** 구절당 하나의 벡터를 생성합니다 (일반적으로 384~3,072 차원). 코사인 유사도(Cosine similarity)를 통해 의미적 근접성(semantic proximity)에 따라 구절의 순위를 매깁니다. OpenAI `text-embedding-3-large`, BGE-M3 밀집 모드, Voyage-3 등이 있으며, 가장 기본적으로 선택되는 방식입니다.

**희소 임베딩 (Sparse embeddings).** SPLADE 스타일입니다. 트랜스포머가 모든 어휘 사전(vocab) 토큰에 대한 가중치를 예측한 후, 대부분의 값을 0으로 만듭니다. 결과물은 |vocab| 크기의 희소 벡터(sparse vector)입니다. 학습된 용어 가중치를 사용하여 BM25와 같은 어휘 매칭(lexical matching)을 수행합니다. 키워드가 중요한 쿼리에 강력합니다.

**멀티 벡터 (Multi-vector, 후기 상호작용).** ColBERTv2, Jina-ColBERT 등이 있습니다. 토큰당 하나의 벡터를 생성합니다. MaxSim을 통해 점수를 산출합니다: 각 쿼리 토큰에 대해 가장 유사한 문서 토큰을 찾아 점수를 합산합니다. 저장 및 점수 산출 비용은 더 높지만, 긴 쿼리와 특정 도메인 코퍼스(corpora)에서 우수한 성능을 보입니다.

**BGE-M3: 세 가지 방식 모두 한 번에 (all three at once).** 단일 모델이 밀집, 희소, 멀티 벡터 표현을 동시에 출력합니다. 각각 독립적으로 쿼리할 수 있으며, 가중 합(weighted sum)을 통해 점수를 결합합니다. 하나의 체크포인트(checkpoint)에서 유연성을 확보하고자 할 때 2026년의 기본 선택지가 될 것입니다.

**마트료시카 표현 학습 (Matryoshka Representation Learning).** 벡터의 처음 N개 차원이 유용한 독립형 임베딩을 형성하도록 학습됩니다. 1,536차원 벡터를 256차원으로 잘라내면(truncate), 약 1%의 정확도 손실로 6배의 저장 공간을 절약할 수 있습니다. OpenAI `text-3`, Cohere `v4`, Voyage-4, Jina `v5`, Gemini Embedding 2, Nomic `v1.5+` 등에서 지원합니다.

### MTEB 리더보드는 부분적인 정보만을 제공합니다 (The MTEB leaderboard tells a partial story)

Massive Text Embedding Benchmark — 출시 당시(2022년) 8개 작업 유형에 걸친 56개 작업으로 시작하여, MTEB v2에서는 100개 이상의 작업으로 확장되었습니다. 2026년 초, Gemini Embedding 2가 검색(retrieval) 부문 1위를 차지했습니다(67.71 MTEB-R). Cohere `embed-v4`가 일반(general) 부문을 선도하고 있습니다(65.2 MTEB). BGE-M3가 오픈 웨이트(open-weight) 다국어(multilingual) 부문을 선도하고 있습니다(63.0). 리더보드는 유용하지만 충분하지는 않습니다. 항상 여러분의 도메인에서 직접 벤치마크를 수행해 보세요.

### 3계층 패턴 (The three-tier pattern)

| 사용 사례 | 패턴 |
|----------|---------|
| 빠른 1차 통과 | Dense bi-encoder (BGE-M3, text-3-small) |
| 재현율 향상 | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Top-50 정밀도 | Multi-vector (ColBERTv2) 또는 cross-encoder reranker |

대부분의 프로덕션 스택은 이 세 가지를 모두 사용합니다.

```figure
gx-matryoshka
```

## 직접 구현해 보기 (Build It)

### 1단계: 베이스라인(baseline) — Sentence-BERT를 이용한 밀집 임베딩(dense embeddings)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`를 설정하면 내적(dot product)이 코사인 유사도(cosine similarity)와 같아집니다. 항상 이 옵션을 설정하세요.

### 2단계: 마트료시카 절단 (Matryoshka truncation)

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

절단 후에는 다시 정규화합니다. Nomic `v1.5`, OpenAI `text-3`, Voyage-4는 초기 몇 단계 수준에서는 정보 손실이 없도록(lossless) 학습되었습니다. 마트료시카 방식이 아닌 모델(기존 Sentence-BERT)은 절단 시 성능이 급격히 저하됩니다.

### 3단계: BGE-M3의 다기능성 (BGE-M3 multi-functionality)

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: {token_id: weight} 형태의 dict를 포함하는 리스트
# output["colbert_vecs"]:  (n_tokens, 1024) 배열을 포함하는 리스트
```

세 가지 인덱스를 한 번의 추론 호출로 얻을 수 있습니다. 점수 융합(Score fusion) 예시:

```python
dense_score = ... # dense_vecs에 대한 코사인 유사도
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

사용자의 도메인에 맞춰 가중치를 조정해 보세요.

### 4단계: 커스텀 태스크에 대한 MTEB 평가 (MTEB eval on a custom task)

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

후보 모델들을 *대표적인(representative)* 서브셋에서 실행해 보세요. 리더보드 순위만 신뢰하지 마세요. 여러분의 도메인이 가장 중요합니다.

### 5단계: 직접 구현한 코사인 유사도 (hand-rolled cosine from scratch)

`code/main.py`를 확인해 보세요. 평균화된 해싱 트릭(Averaged Hashing Trick) 임베딩(표준 라이브러리만 사용)입니다. 트랜스포머(transformer) 임베딩만큼 성능이 뛰어나지는 않지만, 전체적인 흐름(tokenize → vector → normalize → dot product)을 보여줍니다.

## 주의 사항 (Pitfalls)

- **쿼리와 문서에 동일한 모델 사용.** 일부 모델(Voyage, Jina-ColBERT)은 비대칭 인코딩(asymmetric encoding)을 사용하며, 쿼리와 문서가 서로 다른 경로를 거칩니다. 항상 모델 카드(model card)를 확인하세요.
- **접두사(Prefix) 누락.** `bge-*` 모델은 쿼리 앞에 `"Represent this sentence for searching relevant passages: "`를 붙여야 합니다. 이를 누락하면 재현율(recall) 점수가 3~5점 차이 날 수 있습니다.
- **마트료시카(Matryoshka) 임베딩의 과도한 잘라내기(Over-trimming).** 1,536 → 256은 대개 안전하지만, 1,536 → 64는 그렇지 않습니다. 평가 데이터셋(eval set)에서 검증해 보세요.
- **컨텍스트 절단(Context truncation).** 대부분의 모델은 최대 길이를 초과하는 입력을 조용히 절단합니다. 긴 문서는 청킹(chunking)이 필요합니다(23강 참조).
- **지연 시간 꼬리(Latency tail) 무시.** MTEB 점수는 p99 지연 시간을 나타내지 않습니다. 600M 모델이 335M 모델보다 점수가 2점 높을 수 있지만, 쿼리당 비용은 3배 더 들 수 있습니다.

## 활용하기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어 전용, 빠른 속도, API | `text-embedding-3-large` 또는 `voyage-3-large` |
| 오픈 웨이트(Open-weight), 영어 | `BAAI/bge-large-en-v1.5` |
| 오픈 웨이트(Open-weight), 다국어 | `BAAI/bge-m3` 또는 `Qwen3-Embedding-8B` |
| 긴 컨텍스트 (32k+) | Voyage-3-large, Cohere `embed-v4`, Qwen3-Embedding-8B |
| CPU 전용 배포 | Nomic Embed v2 (137M 파라미터, MoE) |
| 저장 공간 제약 | 마트료시카 절단(Matryoshka-truncated) + int8 양자화(quantization) |
| 키워드 중심 쿼리 | SPLADE 희소(sparse) 추가, 밀집(dense) 모델과 RRF-fuse 결합 |

2026년 패턴: BGE-M3 또는 `text-3-large`로 시작하여, MTEB를 통해 해당 도메인에서 평가해 보세요. 도메인 특화 모델이 3점 이상의 차이로 앞선다면 교체하세요.

## Ship It (실행해 보세요)

`outputs/skill-embedding-picker.md`로 저장하세요:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 연습 문제

1. **쉬움.** 100개의 문장을 `bge-small-en-v1.5`를 사용하여 전체 차원(384)으로 인코딩한 후, 마트료시카 128 차원으로 인코딩해 보세요. 10개의 쿼리에 대해 MRR 하락 폭을 측정해 보세요.
2. **중간.** 본인의 도메인에서 가져온 500개의 구절(passages)을 대상으로 BGE-M3의 dense, sparse, colbert 방식을 비교해 보세요. 재현율(`recall@10`)에서는 어떤 방식이 가장 우수한가요? RRF fusion이 가장 성능이 좋은 단일 모드보다 뛰어난가요?
3. **어려움.** 본인의 상위 2개 도메인 태스크에 대해 세 가지 후보 모델을 대상으로 MTEB를 실행해 보세요. MTEB 점수, 100개 쿼리 배치에서의 `p99` 지연 시간(latency), 그리고 100만 쿼리당 비용($/1M queries)을 보고해 보세요. 파레토 최적(Pareto-optimal)인 모델을 선택해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Dense embedding | 벡터 | 텍스트당 하나의 고정된 크기의 벡터. 순위 산정을 위해 코사인 유사도(Cosine similarity) 사용. |
| Sparse embedding | 학습된 BM25 | 어휘 토큰당 하나의 가중치; 대부분이 0; 엔드투엔드(end-to-end)로 학습. |
| Multi-vector | ColBERT 스타일 | 토큰당 하나의 벡터; MaxSim 스코어링; 더 큰 인덱스, 더 높은 재현율(recall). |
| 마트료시카(Matryoshka) | 러시아 인형 기법 | 처음 N개의 차원 자체가 독립적으로 유효한 더 작은 임베딩임. |
| MTEB | 벤치마크 | Massive Text Embedding Benchmark — 출시 당시 56개 작업, v2에서는 100개 이상의 작업 포함. |
| BEIR | 검색 벤치마크 | 18개의 제로샷(zero-shot) 검색 작업; 교차 도메인 강건성(cross-domain robustness)으로 자주 인용됨. |
| Asymmetric encoding | 쿼리 ≠ 문서 경로 | 모델이 쿼리와 문서에 대해 서로 다른 프로젝션(projection)을 사용함. |

## 추가 학습 자료 (Further Reading)

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder 논문입니다.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 리더보드 관련 논문입니다.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 통합된 3가지 모드(three-mode) 모델에 관한 논문입니다.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 차원 사다리(dimension-ladder) 학습 목적 함수에 관한 논문입니다.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 실제 운영 환경에서의 late interaction(지연 상호작용)에 관한 논문입니다.
- [Hugging Face의 MTEB 리더보드](https://huggingface.co/spaces/mteb/leaderboard) — 실시간 순위입니다.
