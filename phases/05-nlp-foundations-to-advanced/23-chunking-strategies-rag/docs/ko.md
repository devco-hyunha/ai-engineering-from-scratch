# RAG를 위한 청킹 전략 (Chunking Strategies for RAG)

> 청킹 설정은 임베딩 모델의 선택만큼이나 검색 품질에 큰 영향을 미칩니다 (Vectara NAACL 2025). 청킹을 잘못하면 아무리 리랭킹(reranking)을 해도 소용이 없습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 문제점 (The Problem)

50페이지 분량의 계약서를 RAG 시스템에 넣었습니다. 사용자가 "해지 조항(termination clause)이 무엇인가요?"라고 질문합니다. 리트리버(retriever)는 표지를 반환합니다. 왜일까요? 모델이 512토큰 청크(chunk) 단위로 학습되었고, 해지 조항은 20페이지 뒤에 위치하며 페이지 구분선에 걸쳐 나뉘어 있는 데다, 질문과 연결될 만한 로컬 키워드가 없기 때문입니다.

해결책은 "더 나은 임베딩 모델을 사는 것"이 아닙니다. 해결책은 청킹(chunking)입니다. 크기는 어느 정도로 할까요? 오버랩(overlap)은요? 어디서 나눌까요? 주변 문맥(context)을 포함해야 할까요?

2026년 2월 벤치마크는 놀라운 결과를 보여줍니다:

- Vectara의 2026년 연구: 재귀적 512토큰 청킹(recursive 512-token chunking)이 시맨틱 청킹(semantic chunking)을 앞질렀습니다 (정확도 69% → 54%).
- Natural Questions에 대한 SPLADE + Mistral-8B 테스트: 오버랩은 측정 가능한 이점을 전혀 제공하지 않았습니다.
- 문맥 절벽(Context cliff): 문맥이 2,500토큰 근처에 도달하면 응답 품질이 급격히 떨어집니다.

"뻔한" 정답(시맨틱 청킹, 20% 오버랩, 1000토큰)은 종종 틀립니다. 이 레슨에서는 6가지 전략에 대한 직관을 기르고, 언제 어떤 전략을 사용해야 하는지 알려줍니다.

## 개념 (The Concept)

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**고정 청킹 (Fixed chunking).** 매 N개의 문자 또는 토큰 단위로 분할합니다. 가장 단순한 베이스라인입니다. 문장 중간에서 끊길 수 있습니다. 압축률은 좋지만 일관성이 떨어집니다.

**재귀적 청킹 (Recursive).** LangChain의 `RecursiveCharacterTextSplitter`가 대표적입니다. 먼저 `\n\n`으로 분할을 시도한 다음, `\n`, `.`, 공백 순으로 시도합니다. 깔끔하게 폴백(fallback)됩니다. 2026년의 기본 방식입니다.

**의미론적 청킹 (Semantic).** 각 문장을 임베딩합니다. 인접한 문장 간의 코사인 유사도를 계산합니다. 유사도가 임계값 아래로 떨어지는 지점에서 분할합니다. 주제의 일관성을 유지합니다. 속도는 더 느리며, 때로는 검색 성능을 저해하는 40토큰 미만의 아주 작은 조각을 생성하기도 합니다.

**문장 단위 청킹 (Sentence).** 문장 경계에서 분할합니다. 청크당 하나의 문장을 담거나 N개의 문장 윈도우를 사용합니다. 비용은 훨씬 적게 들면서 약 5k 토큰까지는 의미론적 청킹과 유사한 성능을 냅니다.

**부모 문서 청킹 (Parent-document).** 검색을 위한 작은 자식 청크(child chunk)와 문맥을 위한 더 큰 부모 청크(parent chunk)를 함께 저장합니다. 자식 청크로 검색하고 부모 청크를 반환합니다. 성능 저하가 완만하게 일어납니다. 즉, 자식 청크가 좋지 않더라도 여전히 합리적인 부모 청크를 반환할 수 있습니다.

**레이트 청킹 (Late chunking, 2024).** 먼저 문서 전체를 토큰 수준에서 임베딩한 다음, 토큰 임베딩을 풀링(pool)하여 청크 임베딩으로 만듭니다. 청크 간의 문맥을 보존합니다. 긴 문맥을 지원하는 임베딩 모델(BGE-M3, Jina v3)과 함께 작동합니다. 연산량이 더 많습니다.

**문맥적 검색 (Contextual retrieval, Anthropic, 2024).** 각 청크 앞에 문서 내 위치에 대한 LLM 생성 요약을 추가합니다 (예: "이 청크는 해지 조항의 섹션 3.2입니다..."). Anthropic 자체 벤치마크에서 검색 성능이 35-50% 향상되었습니다. 인덱싱 비용이 많이 듭니다.

### 모든 기본 설정을 압도하는 규칙 (The rule that beats every default)

쿼리 유형에 맞춰 청크 크기(chunk size)를 조정하세요:

| 쿼리 유형 (Query type) | 청크 크기 (Chunk size) |
|------------|-----------|
| 사실 관계형 (Factoid) ("CEO의 이름은 무엇인가요?") | 256-512 tokens |
| 분석형 / 멀티홉 (Analytical / multi-hop) | 512-1024 tokens |
| 섹션 전체 이해 (Whole-section comprehension) | 1024-2048 tokens |

NVIDIA의 2026년 벤치마크입니다. 청크는 정답과 주변 문맥(local context)을 포함할 수 있을 만큼 충분히 커야 하며, 검색기(retriever)의 top-K 결과가 문맥 노이즈(context noise)가 아닌 정답에 집중할 수 있을 만큼 충분히 작아야 합니다.

```figure
n5-chunk-cuts
```

## 직접 구현해 보기 (Build It)

### 1단계: 고정 및 재귀적 청킹 (fixed and recursive chunking)

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 2단계: 의미론적 청킹 (semantic chunking)

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

도메인에 맞춰 `threshold`를 조정해 보세요. 너무 높으면 파편화(fragments)되고, 너무 낮으면 하나의 거대한 청크(one giant chunk)가 됩니다.

### 3단계: 부모 문서 청킹 (parent-document)

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

핵심 통찰(Key insight): 부모 문서의 중복을 제거하세요. 여러 개의 자식 청크가 동일한 부모 청크에 매핑될 수 있으므로, 모두 반환하면 문맥(context)이 낭비될 수 있습니다.

### 4단계: 문맥적 검색 (contextual retrieval, Anthropic 패턴)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

문맥화된 청크를 인덱싱하세요. 쿼리 시점에 검색은 추가적인 주변 신호(surrounding signal)의 이점을 얻을 수 있습니다.

### 5단계: 평가 (evaluate)

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

항상 벤치마크를 수행하세요. 여러분의 코퍼스(corpus)에 대한 "최적의" 전략은 어떤 블로그 포스트와도 일치하지 않을 수 있습니다.

## 주의 사항 (Pitfalls)

- **사실 기반 질의(factoid queries)로만 청킹을 평가함.** 멀티홉(Multi-hop) 질의에서는 결과가 매우 다르게 나타납니다. 질의 유형별로 계층화된(stratified) 평가 세트를 사용하세요.
- **최소 크기를 고려하지 않은 의미론적 청킹(Semantic chunking).** 검색 성능을 저하시키는 40토큰 정도의 파편화된 조각들을 생성합니다. 항상 `min_tokens`를 적용하세요.
- **맹목적인 오버랩(Overlap) 적용.** 2026년 연구에 따르면 오버랩은 이득이 없는 경우가 많으며 인덱스 비용을 두 배로 늘립니다. 추측하지 말고 직접 측정하세요.
- **최소/최대 크기 제한 미비.** 5토큰이나 5000토큰 크기의 청크는 모두 검색 성능을 망가뜨립니다. 범위를 제한(Clamp)하세요.
- **문서 간 청킹(Cross-doc chunking).** 하나의 청크가 두 개의 문서를 가로지르게 하지 마세요. 항상 문서별로 청킹한 다음 병합하세요.

## 활용하기 (Use It)

2026년 스택:

| 상황 | 전략 |
|-----------|----------|
| 최초 구축, 미지의 코퍼스 | 재귀적(Recursive), 512 토큰, 오버랩 없음 |
| 사실 기반 질의응답 (Factoid QA) | 재귀적(Recursive), 256-512 토큰 |
| 분석적 / 멀티홉 (Analytical / multi-hop) | 재귀적(Recursive), 512-1024 토큰 + 부모 문서(parent-document) |
| 대량의 교차 참조 (계약서, 논문) | 레이트 청킹(Late chunking) 또는 문맥적 검색(contextual retrieval) |
| 대화형 / 대화 코퍼스 | 턴(Turn) 단위 청크 + 화자 메타데이터 |
| 짧은 발화 (트윗, 리뷰) | 문서 하나 = 청크 하나 |

재귀적 512 토큰 방식으로 시작해 보세요. 50개의 쿼리로 구성된 평가 세트에서 `recall@5`를 측정하세요. 그 결과에 따라 튜닝을 진행해 보세요.

## Ship It

`outputs/skill-chunker.md`로 저장하세요:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 연습 문제 (Exercises)

1. **쉬움.** `fixed(512, 0)`, `recursive(512, 0)`, 그리고 `recursive(512, 100)` 방식을 사용하여 20페이지 분량의 문서 하나를 청킹(chunking)해 보세요. 청크 개수와 경계 품질(boundary quality)을 비교해 보세요.
2. **중간.** 5개의 문서를 대상으로 30개의 쿼리로 구성된 평가 세트(eval set)를 구축해 보세요. `recursive`, `semantic`, 그리고 `parent-document` 방식에 대해 `recall@5`를 측정해 보세요. 어떤 방식이 가장 우수한가요? 블로그 포스트의 내용과 일치하나요?
3. **어려움.** 문맥적 검색(contextual retrieval)을 구현해 보세요. 기본(baseline)인 `recursive` 방식 대비 MRR 개선 정도를 측정해 보세요. 인덱싱 비용(LLM 호출 횟수) 대비 정확도 향상(accuracy gain)을 보고해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 청크 (Chunk) | 문서의 조각 | 임베딩, 인덱싱 및 검색되는 하위 문서 단위. |
| 오버랩 (Overlap) | 안전 마진 | 인접한 청크 간에 공유되는 N개의 토큰; 2026년 벤치마크에서는 종종 무용지물임. |
| 시맨틱 청킹 (Semantic chunking) | 스마트한 청킹 | 인접한 문장 간의 임베딩 유사도가 떨어지는 지점에서 분할. |
| 부모 문서 (Parent-document) | 2단계 검색 | 작은 자식 청크를 검색하여 더 큰 부모 청크를 반환. |
| 레이트 청킹 (Late chunking) | 임베딩 후 청킹 | 전체 문서를 토큰 수준에서 임베딩한 후, 청크 벡터로 풀링(pool). |
| 문맥적 검색 (Contextual retrieval) | Anthropic의 기법 | 인덱싱 전 각 청크 앞에 LLM이 생성한 요약을 추가. |
| 문맥 절벽 (Context cliff) | 2500토큰의 벽 | RAG에서 문맥 토큰이 약 2.5k에 도달할 때 관찰되는 품질 저하 (2026년 1월). |

## 추가 학습 자료 (Further Reading)

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 프로덕션 환경의 기본 설정입니다.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 청킹(chunking)은 임베딩 선택만큼이나 중요합니다.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 레이트 청킹(late chunking) 논문입니다.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM으로 생성된 컨텍스트 접두사(context prefixes)를 사용하여 검색 성능을 35-50% 향상시킵니다.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 쿼리 유형별 청크 크기(chunk size)에 관한 요약입니다.
