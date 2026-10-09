# RAG를 위한 청킹 전략

> 청킹 구성은 임베딩 모델 선택만큼 검색 품질에 영향을 미칩니다 (Vectara NAACL 2025). 청킹을 잘못하면 리랭킹으로 회복할 수 없습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 14강 (정보 검색), 5단계 · 22강 (임베딩 모델)
**시간:** 약 60분

## 문제점

50페이지 분량의 계약을 RAG 시스템에 넣었습니다. 사용자가 "종료 조항이 무엇인가요?"라고 묻습니다. 검색기는 표지 페이지를 반환합니다. 왜일까요? 모델이 512토큰 청킹으로 학습되었고, 종료 조항은 20페이지 깊숙이 위치하며 페이지 나누기를 통해 분리되어 있고, 쿼리와 연결하는 지역 키워드가 없기 때문입니다.

해결책은 "더 나은 임베딩 모델을 구매하는 것"이 아닙니다. 해결책은 청킹입니다. 크기는 얼마나? 겹침은? 어디서 나누는가? 주변 컨텍스트를 포함하는가?

2026년 2월 벤치마크는 놀라운 결과를 보여줍니다:

- Vectara의 2026년 연구: 재귀적 512토큰 청킹이 시맨틱 청킹을 69% → 54% 정확도로 능가했습니다.
- Natural Questions에서의 SPLADE + Mistral-8B: 겹침은 측정 가능한 이점을 제공하지 않았습니다.
- 컨텍스트 절벽: 컨텍스트가 약 2,500토큰에 도달하면 응답 품질이 급격히 떨어집니다.

"명백한" 답변 (시맨틱 청킹, 20% 겹침, 1000토큰)은 종종 틀립니다. 이 강의는 6가지 전략에 대한 직관을 구축하고, 어떤 전략을 언제 사용해야 하는지 알려줍니다.

## 개념

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**고정 청킹.** 매 N자 또는 토큰마다 나눕니다. 가장 단순한 기준선입니다. 문장 중간에서 끊어집니다. 압축은 좋지만, 일관성은 나쁩니다.

**재귀적.** LangChain의 `RecursiveCharacterTextSplitter`입니다. `\n\n`으로 먼저 나누고, 그 다음 `\n`, 그 다음 `.`, 그 다음 공백으로 나눕니다. 깔끔하게 폴백합니다. 2026년 기본값입니다.

**시맨틱.** 각 문장을 임베딩합니다. 인접한 문장 간의 코사인 유사도(Cosine Similarity)를 계산합니다. 유사도가 임계값 아래로 떨어지는 지점에서 나눕니다. 주제 일관성을 보존합니다. 느립니다. 때로는 검색에 해로운 40토큰의 작은 조각을 생성합니다.

**문장.** 문장 경계에서 나눕니다. 청킹당 한 문장 또는 N문장 윈도우를 사용합니다. 비용의 일부로 약 5k토큰까지 시맨틱 청킹과 유사한 성능을 보입니다.

**부모 문서.** 검색을 위해 작은 자식 청크를 저장하고, 컨텍스트를 위해 더 큰 부모 청크를 저장합니다. 자식 청크로 검색하고, 부모 청크를 반환합니다. 우아한 저하(Graceful Degradation)가 가능합니다: 나쁜 자식 청크가 있어도 합리적인 부모 청크를 반환합니다.

** 후기 청킹(Late Chunking) (2024).** 먼저 토큰 수준에서 전체 문서를 임베딩(Embedding)한 후, 토큰 임베딩을 청크 임베딩으로 풀링합니다. 청크 간 컨텍스트를 보존합니다. 긴 컨텍스트 임베더(BGE-M3, Jina v3)와 잘 작동합니다. 연산량이 더 높습니다.

** 컨텍스트 검색(Contextual Retrieval) (Anthropic, 2024).** 각 청크 앞에 LLM이 생성한 문서 내 위치 요약("이 청크는 종료 조항의 3.2 섹션입니다...")을 붙입니다. Anthropic의 자체 벤치마크에서 검색 성능이 35-50% 향상되었습니다. 인덱싱 비용이 높습니다.

### 모든 기본값을 능가하는 규칙

청크 크기를 쿼리 유형에 맞춰 매칭하세요:

| 쿼리 유형 | 청크 크기 |
|------------|-----------|
| 팩토이드(Factoid) ("CEO의 이름은 무엇인가요?") | 256-512 토큰 |
| 분석적 / 다중 홉(Multi-hop) | 512-1024 토큰 |
| 전체 섹션 이해 | 1024-2048 토큰 |

NVIDIA의 2026년 벤치마크. 청크는 답변과 지역 컨텍스트를 포함할 만큼 충분히 크고, 검색기의 Top-K가 컨텍스트 잡음 대신 답변에 집중할 만큼 충분히 작아야 합니다.

```figure
n5-chunk-cuts
```

## 구현하기

### 1단계: 고정 및 재귀 청킹

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

### 2단계: 시맨틱 청킹(Semantic Chunking)

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

`threshold`를 도메인에 맞게 튜닝하세요. 너무 높으면 → 파편화됩니다. 너무 낮으면 → 하나의 거대한 청크가 됩니다.

### 3단계: 부모 문서

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

핵심 인사이트: 부모 청크를 중복 제거하세요. 여러 자식 청크가 같은 부모 청크에 매핑될 수 있으며, 모두 반환하면 컨텍스트를 낭비합니다.

### 4단계: 컨텍스트 검색 (Anthropic 패턴)

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

컨텍스트화된 청크를 인덱싱하세요. 쿼리 시 검색은 추가적인 주변 신호의 이점을 얻습니다.

### 5단계: 평가(Evaluation (Eval))

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

항상 벤치마킹하세요. 코퍼pus에 대한 "최고의" 전략은 블로그 포스트와 일치하지 않을 수 있습니다.

## 함정

- ** 팩토이드(Factoid) 쿼리에서만 청킹을 평가하는 것.** 다중 홉(Multi-hop) 쿼리는 매우 다른 승자를 드러냅니다. 쿼리 유형별로 층화(stratified)된 평가 세트(Eval Set)를 사용하세요.
- ** 최소 크기가 없는 시맨틱 청킹(Semantic Chunking).** 검색 성능을 해치는 40 토큰 파편을 생성합니다. 항상 `min_tokens`을 적용하세요.
- **중복은 의식처럼 여겨지는 경향.** 2026년 연구에 따르면 중복은 종종 효과가 없고 인덱스 비용을 두 배로 늘립니다. 측정하세요, 추측하지 마세요.
- **최소/최대 강제 없음.** 5개 토큰이나 5000개 토큰의 청크는 모두 검색을 망가뜨립니다. 클램프하세요.
- **문서 간 청킹.** 청크가 두 문서를 걸쳐서는 안 됩니다. 항상 문서별로 청킹한 후 병합하세요.

## 사용하기

2026년 스택:

| 상황 | 전략 |
|-----------|----------|
| 첫 빌드, 코퍼스 미지수 | 재귀적, 512 토큰, 중복 없음 |
| 사실성 QA | 재귀적, 256-512 토큰 |
| 분석적 / 다중 홉 | 재귀적, 512-1024 토큰 + 부모 문서 |
| 많은 상호 참조 (계약서, 논문) | 후기 청킹 또는 컨텍스트 검색 |
| 대화 / 대화 코퍼스 | 턴 단위 청크 + 화자 메타데이터 |
| 짧은 발화 (트윗, 리뷰) | 한 문서 = 한 청크 |

재귀적 512로 시작하세요. 50개 쿼리 평가 세트에서 recall@5를 측정하세요. 거기서부터 튜닝하세요.

## 출시하기

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

## 연습 문제

1. **쉬움.** 20페이지 문서를 fixed(512, 0), recursive(512, 0), recursive(512, 100)으로 청킹하세요. 청크 수와 경계 품질을 비교하세요.
2. **중간.** 5개 문서에 대해 30개 쿼리 평가 세트를 구축하세요. 재귀적, 시맨틱, 부모 문서 방식에 대해 recall@5를 측정하세요. 어떤 방식이 이기나요? 블로그 포스트와 일치하나요?
3. **어려움.** 컨텍스트 검색을 구현하세요. 기본 재귀적 방식 대비 MRR 개선도를 측정하세요. 인덱스 비용 (LLM 호출)과 정확도 향상도를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 청크 | 문서의 일부 | 임베딩, 인덱싱, 검색되는 하위 문서 단위. |
| 중복 | 안전 마진 | 인접 청크 간에 공유되는 N개 토큰; 2026년 벤치마크에서는 종종 무용함. |
| 시맨틱 청킹 | 스마트 청킹 | 인접 문장 임베딩 유사도가 떨어지는 곳에서 분할. |
| 부모 문서 | 2단계 검색 | 작은 자식을 검색하고, 더 큰 부모를 반환. |
| 늦은 청킹 | 임베딩 후 청킹 | 문서 전체를 토큰 수준에서 임베딩한 후 청크 벡터로 풀링합니다. |
| 컨텍스트 검색 | Anthropic의 기법 | 인덱싱 전에 각 청크에 LLM이 생성한 요약문을 앞에 붙입니다. |
| 컨텍스트 절벽 | 2500 토큰 벽 | RAG에서 약 2.5k 컨텍스트 토큰 부근에서 품질 저하가 관찰됩니다 (2026년 1월). |

## 추가 읽기

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 프로덕션의 기본값입니다.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 청킹은 임베딩 선택만큼 중요합니다.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 늦은 청킹 논문입니다.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM이 생성한 컨텍스트 접두어로 검색 성능이 35-50% 향상됩니다.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 쿼리 유형별 청크 크기입니다.
