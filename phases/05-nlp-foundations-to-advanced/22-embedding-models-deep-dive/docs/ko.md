# 임베딩 모델 — 2026년 심층 분석

> Word2Vec는 단어마다 벡터를 제공했습니다. 현대의 임베딩 모델은 문단마다 벡터를 제공하며, 다국어 지원, 희소(sparse), 밀집(dense), 다중 벡터(multi-vector) 뷰를 포함하고, 인덱스에 맞게 크기를 조정할 수 있습니다. 선택을 잘못하면 RAG가 잘못된 것을 검색합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 03강 (Word2Vec), 5단계 · 14강 (정보 검색)
**시간:** 약 60분

## 문제점

RAG 시스템이 40%의 확률로 잘못된 문단을 검색합니다. 원인은 벡터 데이터베이스나 프롬프트가 아닌, 임베딩 모델인 경우가 대부분입니다.

2026년에 임베딩을 선택한다는 것은 다음 다섯 가지 축에 걸쳐 선택하는 것을 의미합니다:

1. **밀집(dense) vs 희소(sparse) vs 다중 벡터(multi-vector).** 문단마다 하나의 벡터, 토큰마다 하나의 벡터, 또는 희소 가중치 단어 주머니(sparse weighted bag of words)를 사용합니다.
2. **언어 커버리지.** 단언어 영어 모델은 영어 전용 작업에서 여전히 우세합니다. 다국어 모델은 코퍼스가 혼합된 경우 우세합니다.
3. **컨텍스트 길이.** 512 토큰 vs 8,192 vs 32,768 — 실제 유효 용량은 종종 광고된 최대치의 60-70%입니다.
4. **차원 예산.** 전체 정밀도의 3,072 floats는 벡터당 12 KB입니다. 1억 벡터에서 저장 비용은 월 $1,300입니다. Matryoshka 절단은 이를 4배 줄입니다.
5. **오픈 vs 호스트.** 오픈 가중치(open-weight)는 스택과 데이터를 통제할 수 있음을 의미합니다. 호스트(hosted)는 통제력을 최신 상태로 유지하는 것과 교환하는 것을 의미합니다.

이 강의는 트레이드오프를 명시하여, 지난 분기에 유행했던 것에 의존하지 않고 증거에 기반하여 선택할 수 있도록 합니다.

## 개념

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**밀집 임베딩(dense embeddings).** 문단마다 하나의 벡터 (보통 384-3,072 차원). 코사인 유사도(Cosine Similarity)가 의미적 근접성에 따라 문단을 순위를 매깁니다. OpenAI `text-embedding-3-large`, BGE-M3 밀집 모드, Voyage-3. 기본 선택입니다.

**희소 임베딩(sparse embeddings).** SPLADE 스타일. 트랜스포머가 어휘 토큰마다 가중치를 예측한 후 대부분을 0으로 만듭니다. 결과는 크기가 |vocab|인 희소 벡터입니다. BM25와 유사한 어휘 매칭을 포착하지만 학습된 용어 가중치를 사용합니다. 키워드 중심 쿼리에 강합니다.

**멀티 벡터 (후기 상호작용).** ColBERTv2, Jina-ColBERT. 토큰당 하나의 벡터. MaxSim으로 점수 산정: 각 쿼리 토큰에 대해 가장 유사한 문서 토큰을 찾고 점수를 합산합니다. 저장 및 점수 산정이 더 비싸지만, 긴 쿼리와 도메인 특화 코퍼스에서 더 좋은 성능을 냅니다.

**BGE-M3: 세 가지 모두를 한 번에.** 단일 모델이 밀집(dense), 희소(sparse), 멀티 벡터 표현을 동시에 출력합니다. 각각 독립적으로 쿼리할 수 있으며, 가중 합으로 점수를 융합합니다. 하나의 체크포인트에서 유연성을 원할 때 2026년 기본 선택입니다.

**Matryoshka 표현 학습.** 벡터의 첫 N개 차원이 독립적으로 유용한 임베딩을 형성하도록 학습됩니다. 1,536차원 벡터를 256차원으로 잘라내면 약 1%의 정확도 손실로 6배의 저장 공간 절감을 달성합니다. OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+에서 지원됩니다.

### MTEB 리더보드는 부분적인 이야기만 전달합니다

Massive Text Embedding Benchmark — 출시 시점(2022)에 8가지 작업 유형에 걸쳐 56개 작업, MTEB v2에서는 100개 이상의 작업으로 확장. 2026년 초, Gemini Embedding 2가 검색 분야에서 최고(MTEB-R 67.71)입니다. Cohere embed-v4가 일반 분야에서 선두(MTEB 65.2)입니다. BGE-M3가 오픈 웨이트 다국어 분야에서 선두(MTEB 63.0)입니다. 리더보드는 필요하지만 충분하지는 않습니다 — 항상 자신의 도메인에서 벤치마킹하세요.

### 3단계 패턴

| 사용 사례 | 패턴 |
|----------|---------|
| 빠른 1차 패스 | 밀집 바이인코더 (BGE-M3, text-3-small) |
| 재현율 향상 | 희소 (SPLADE, BGE-M3 sparse) + RRF 융합 |
| 상위 50개 정밀도 | 멀티 벡터 (ColBERTv2) 또는 크로스 인코더 리랭커 |

대부분의 프로덕션 스택은 세 가지를 모두 사용합니다.

```figure
gx-matryoshka
```

## 구현하기

### 1단계: 기본선 — Sentence-BERT로 밀집 임베딩

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

`normalize_embeddings=True`는 내적 곱을 코사인 유사도와 동일하게 만듭니다. 항상 설정하세요.

### 2단계: Matryoshka 잘라내기

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

잘라낸 후 재정규화하세요. Nomic v1.5, OpenAI text-3, Voyage-4는 처음 몇 단계에서 손실 없이 잘라낼 수 있도록 학습되었습니다. Matryoshka가 아닌 모델(원본 Sentence-BERT)은 잘라내면 성능이 급격히 저하됩니다.

### 3단계: BGE-M3 다기능성

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
# output["lexical_weights"]: dict {token_id: weight}의 리스트
# output["colbert_vecs"]: (n_tokens, 1024) 배열의 리스트
```

세 개의 인덱스, 한 번의 추론 호출. 점수 융합:

```python
dense_score = ... # dense_vecs에 대한 코사인 유사도(Cosine Similarity)
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

해당 도메인에 맞춰 가중치를 조정해 보세요.

### 4단계: 사용자 정의 작업에 대한 MTEB 평가(Evaluation (Eval))

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

후보 모델을 *대표적인* 하위 집합에서 실행하세요. 리더보드 순위만 신뢰하지 마세요. 도메인이 중요합니다.

### 5단계: 처음부터 직접 만든 코사인 유사도(Cosine Similarity)

`code/main.py`을 참고하세요. 평균화된 해싱 트릭 임베딩(Embedding) (표준 라이브러리 전용). 트랜스포머(Transformer) 임베딩(Embedding)과 경쟁할 수준은 아니지만, 형태를 보여줍니다: 토큰화(Tokenization) → 벡터 → 정규화(Normalization) → 내적.

## 함정

- **쿼리와 문서에 동일한 모델 사용.** 일부 모델(Voyage, Jina-ColBERT)은 비대칭 인코딩을 사용합니다. 쿼리와 문서가 서로 다른 경로를 거칩니다. 항상 모델 카드(Model Card)를 확인하세요.
- **접두어 누락.** `bge-*` 모델은 쿼리에 `"Represent this sentence for searching relevant passages: "`을 앞에 붙여야 합니다. 잊으면 재현율(recall)이 3-5점 떨어집니다.
- **Matryoshka 과다 절단.** 1,536 → 256은 보통 안전합니다. 1,536 → 64는 안전하지 않습니다. 평가 세트(Eval Set)에서 검증하세요.
- **컨텍스트 잘림.** 대부분의 모델은 최대 길이를 초과하는 입력을 조용히 잘라냅니다. 긴 문서는 청킹(Chunking)이 필요합니다 (23강 참고).
- **꼬리 지연(Tail Latency) 무시.** MTEB 점수는 p99 지연을 숨깁니다. 600M 모델이 335M 모델보다 2점 높을 수 있지만, 쿼리당 비용이 3배 더 들 수 있습니다.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어 전용, 빠름, API | `text-embedding-3-large` 또는 `voyage-3-large` |
| 오픈 웨이트, 영어 | `BAAI/bge-large-en-v1.5` |
| 오픈 웨이트, 다국어 | `BAAI/bge-m3` 또는 `Qwen3-Embedding-8B` |
| 긴 컨텍스트 (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU 전용 배포 | Nomic Embed v2 (137M 매개변수(Parameter), MoE (혼합 전문가)(MoE (Mixture of Experts))) |
| 저장 공간 제한 | Matryoshka 절단 + int8 양자화(Quantization) |
| 키워드 중심 쿼리 | SPLADE 희소 검색 추가, 밀집 검색(Dense Retrieval)과 RRF (상호 랭킹 융합)(Reciprocal Rank Fusion (RRF)) 융합 |

2026년 패턴: BGE-M3 또는 text-3-large로 시작하여, MTEB로 도메인에서 평가하고, 도메인 특화 모델이 3점 이상 이기면 교체하세요.

## 출시하기

`outputs/skill-embedding-picker.md`으로 저장하세요:

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

1. **쉬움.** `bge-small-en-v1.5`으로 100개 문장을 전체 차원(384)에서 인코딩한 후, Matryoshka 128 차원에서 인코딩하세요. 10개 쿼리에 대해 MRR 감소량을 측정해 보세요.
2. **중간.** 도메인에서 500개 패시지를 가져와 BGE-M3의 dense, sparse, colbert 방식을 비교하세요. recall@10에서 어떤 방식이 이기나요? RRF 융합이 단일 모드 중 최상의 성능을 넘어서나요?
3. **어려움.** 상위 2개 도메인 작업에 대해 세 후보 모델로 MTEB를 실행하세요. MTEB 점수, 100개 쿼리 배치에서의 p99 지연 시간, 쿼리 100만 건당 비용($/1M queries)을 보고하세요. 파레토 최적 모델을 선택하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Dense 임베딩 | 벡터 | 텍스트당 고정 크기의 벡터 하나. 랭킹을 위해 코사인 유사도(Cosine Similarity)를 사용. |
| Sparse 임베딩 | 학습된 BM25 | 어휘 토큰당 가중치 하나; 대부분 0; 엔드투엔드(end-to-end)로 학습. |
| Multi-vector | ColBERT 스타일 | 토큰당 벡터 하나; MaxSim 점수 매기기; 더 큰 인덱스, 더 나은 재현율(recall). |
| Matryoshka | 러시아 인형 트릭 | 첫 N개 차원만으로도 유효한 더 작은 임베딩이 됨. |
| MTEB | 벤치마크 | Massive Text Embedding Benchmark — 출시 시 56개 작업, v2에서는 100개 이상. |
| BEIR | 검색 벤치마크 | 18개 제로샷(Zero-Shot) 검색 작업; 도메인 간 강건성(cross-domain robustness)으로 자주 인용됨. |
| 비대칭 인코딩 | 쿼리 ≠ 문서 경로 | 모델이 쿼리와 문서에 서로 다른 투영(projection)을 사용. |

## 추가 읽기

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 바이인코더(bi-encoder) 논문.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 리더보드 논문.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 통합 3-모드 모델.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 차원 사다리 학습 목표.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 프로덕션에서의 후기 상호작용(late interaction).
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 실시간 랭킹.
