# 엔티티 연결 및 모호성 해소 (Entity Linking & Disambiguation)

> NER이 "Paris"를 찾았습니다. 엔티티 연결(Entity linking)은 다음과 같이 결정합니다: 프랑스의 파리? 파리스 힐튼? 텍사스의 파리? 파리스(트로이의 왕자)? 엔티티 연결이 없다면, 여러분의 지식 그래프(knowledge graph)는 모호한 상태로 남게 됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 문제점 (The Problem)

"Jordan beat the press."라는 문장이 있습니다. NER 모델은 "Jordan"을 `PERSON`으로 태깅합니다. 좋습니다. 하지만 *어떤* Jordan일까요?

- Michael Jordan (농구 선수)?
- Michael B. Jordan (배우)?
- Michael I. Jordan (버클리 대학의 ML 교수 — 네, ML 논문에서 실제로 발생하는 혼동입니다)?
- Jordan (국가)?
- Jordan (히브리어 이름)?

엔티티 연결(Entity Linking, EL)은 각 언급(mention)을 지식 베이스(Wikidata, Wikipedia, DBpedia 또는 도메인 KB)의 고유한 항목으로 연결합니다. 여기에는 두 가지 하위 작업이 있습니다:

1. **후보 생성(Candidate generation).** "Jordan"이 주어졌을 때, 어떤 KB 항목이 가능성이 있습니까?
2. **모호성 해소(Disambiguation).** 문맥을 고려할 때, 어떤 후보가 정답입니까?

두 단계 모두 학습 가능합니다. 또한 두 단계 모두 벤치마크 성능이 측정됩니다. 통합 파이프라인은 지난 10년 동안 안정적으로 유지되어 왔으며, 변화하는 것은 모호성 해소기(disambiguator)의 품질입니다.

## 개념 (The Concept)

![엔티티 연결 파이프라인: 언급(mention) → 후보(candidates) → 모호성이 해소된 엔티티(disambiguated entity)](../assets/entity-linking.svg)

**후보 생성(Candidate generation).** 언급의 표면형(mention surface form, 예: "Jordan")이 주어지면, 별칭 인덱스(alias index)에서 후보를 검색합니다. 위키피디아 별칭 사전은 대부분의 고유 명사를 포함합니다: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK airport, JFK (movie). 일반적인 인덱스는 언급당 10~30개의 후보를 반환합니다.

**모호성 해소(Disambiguation): 세 가지 접근 방식.**

1. **사전 확률 + 문맥(Prior + context) (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. 성능이 좋고 빠르며, 별도의 학습이 필요하지 않습니다.
2. **임베딩 기반(Embedding-based) (ESS / REL / Blink).** 언급과 문맥을 인코딩합니다. 각 후보의 설명을 인코딩한 후, 최대 코사인 유사도를 선택합니다. 2020~2024년 사이의 표준 방식입니다.
3. **생성형(Generative) (GENRE, 2021; LLM 기반, 2023+).** 엔티티의 정형화된 이름(canonical name)을 토큰 단위로 디코딩합니다. 유효한 엔티티 이름의 트라이(trie)로 제약 디코딩(constrained decoding)을 수행하여, 출력이 반드시 유효한 KB ID가 되도록 보장합니다.

**엔드투엔드(End-to-end) vs 파이프라인(Pipeline).** 최신 모델(ELQ, BLINK, ExtEnD, GENRE)은 NER + 후보 생성 + 모호성 해소를 한 번의 과정으로 수행합니다. 파이프라인 시스템은 구성 요소를 교체할 수 있기 때문에 여전히 프로덕션 환경에서 주로 사용됩니다.

### 두 가지 측정 지표 (The two measurements)

- **Mention recall (후보 생성 단계).** 정답(gold) 언급 중 올바른 KB 엔트리가 후보 목록에 포함된 비율입니다. 전체 파이프라인의 하한선(floor) 역할을 합니다.
- **Disambiguation accuracy / F1 (모호성 해소 정확도 / F1).** 올바른 후보가 주어졌을 때, `top-1`이 얼마나 자주 맞는지 나타냅니다.

항상 두 지표를 모두 보고해야 합니다. 후보 재현율(candidate recall)이 80%인데 모호성 해소(disambiguation) 정확도가 99%인 시스템은 결국 80% 성능의 파이프라인입니다.

```figure
gx-entity-linking
```

## 구현하기 (Build It)

### 1단계: Wikipedia 리다이렉트(redirects)로부터 별칭 인덱스(alias index) 구축하기

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 별칭 데이터: 약 1,800만(18M) 개의 (alias, entity) 쌍이 존재합니다. Wikidata 덤프(dumps)에서 다운로드하여 역색인(inverted index)으로 저장합니다.

### 2단계: 문맥 기반 모호성 해소 (context-based disambiguation)

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap(자카드 중첩)은 예시용(toy)입니다. 임베딩에 대한 코사인 유사도(cosine similarity)로 교체해 보세요 (`code/main.py`의 step-2에서 트랜스포머 버전을 참조하세요).

### 3단계: 임베딩 기반(embedding-based, BLINK-style) 방식

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

인덱싱 시점에는 모든 KB 엔티티를 한 번씩 임베딩합니다. 쿼리 시점에는 언급(mention)과 문맥(context)을 한 번 임베딩한 후, 후보군(candidate pool)과 내적(dot-product)을 수행하여 최댓값을 선택합니다.

### 4단계: 생성형 엔티티 연결 (Generative Entity Linking (개념))

GENRE는 엔티티의 Wikipedia 제목을 한 글자씩 디코딩합니다. 제약 디코딩(constrained decoding, 레슨 20 참조)을 통해 유효한 제목만 출력되도록 보장합니다. 지식 베이스(KB) 기반의 트라이(trie)와 긴밀하게 통합됩니다. 현대의 후계 모델은 REL-GEN 및 구조화된 출력을 사용하는 LLM 프롬프트 기반 EL입니다.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

화이트리스트(Outlines의 `choice`)와 결합하면, 이는 2026년에 출시할 수 있는 가장 단순한 EL 파이프라인입니다.

### 5단계: AIDA-CoNLL로 평가하기 (evaluate on AIDA-CoNLL)

AIDA-CoNLL는 표준 EL 벤치마크입니다: 1,393개의 Reuters 기사, 34k개의 언급(mentions), Wikipedia 엔티티로 구성됩니다. in-KB 정확도(`P@1`)와 out-of-KB NIL 탐지율(NIL-detection rate)을 보고하세요.

## 주의 사항 (Pitfalls)

- **`NIL` 처리 (NIL handling).** 일부 언급(mention)은 `KB`에 존재하지 않습니다 (신규 등장 엔티티, 알려지지 않은 인물 등). 시스템은 잘못된 엔티티를 추측하는 대신 `NIL`을 예측해야 합니다. 이는 별도로 측정됩니다.
- **언급 경계 오류 (Mention boundary errors).** 상위 단계의 `NER`이 부분적인 스팬(span)을 놓치는 경우입니다 ("Bank of America"가 "Bank"로만 태깅됨). 이 경우 `EL` 재현율(recall)이 하락합니다.
- **인기 편향 (Popularity bias).** 학습된 시스템은 빈도가 높은 엔티티를 과도하게 예측하는 경향이 있습니다. 머신러닝 논문에 등장하는 "Michael I. Jordan"은 종종 농구 선수 Jordan으로 연결됩니다.
- **교차 언어 `EL` (Cross-lingual EL).** 중국어 텍스트의 언급을 영어 위키피디아 엔티티에 매핑하는 작업입니다. 다국어 인코더(multilingual encoder) 또는 번역 단계가 필요합니다.
- **`KB` 노후화 (KB staleness).** 새로운 기업, 사건, 인물은 작년의 위키피디아 덤프에 포함되어 있지 않습니다. 운영 파이프라인에는 데이터 갱신 루프(refresh loop)가 필요합니다.

## 활용하기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 범용 영어 + 위키피디아 | BLINK 또는 REL |
| 교차 언어(Cross-lingual), KB = 위키피디아 | mGENRE |
| LLM 친화적, 일일 언급 횟수 적음 | 후보 목록 + 제약 JSON(constrained JSON)을 사용하여 Claude/GPT-4에 프롬프트 입력 |
| 도메인 특화 KB (의료, 법률) | KB 인지 검색(KB-aware retrieval) 기능이 포함된 커스텀 BERT + 도메인 AIDA 스타일 데이터셋으로 미세 조정(fine-tune) |
| 극도로 낮은 지연 시간 | 정확한 일치(Exact-match) 사전 확률만 사용 (Milne-Witten 베이스라인) |
| 연구용 SOTA | GENRE / ExtEnD / 생성형 LLM-EL |

2026년에 배포될 프로덕션 패턴: NER → coref → 각 언급(mention)에 대한 EL → 클러스터를 하나의 정준 엔티티(canonical entity)로 병합. 출력은 언급당 하나가 아닌, 문서 내 엔티티당 하나의 KB ID를 출력합니다.

## Ship It (실행하기)

`outputs/skill-entity-linker.md`로 저장하세요:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 연습 문제

1. **쉬움(Easy).** `code/main.py`에서 10개의 모호한 언급(Paris, Jordan, Apple)을 대상으로 사전 정보+문맥 모호성 해소기(prior+context disambiguator)를 구현해 보세요. 정답 엔티티를 수동으로 레이블링하고 정확도(accuracy)를 측정하세요.
2. **중간(Medium).** Sentence Transformer를 사용하여 50개의 모호한 언급을 인코딩하세요. 각 후보의 설명을 임베딩하세요. 임베딩 기반 모호성 해소(embedding-based disambiguation)를 Jaccard 문맥 중첩(Jaccard context overlap)과 비교해 보세요.
3. **어려움(Hard).** 1,000개의 엔티티로 구성된 도메인 지식 베이스(KB)(예: 회사의 직원 + 제품)를 구축하세요. NER + EL을 엔드 투 엔드(end-to-end)로 구현하세요. 100개의 홀드아웃(held-out) 문장에 대해 정밀도(precision)와 재현율(recall)을 측정하세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 엔티티 연결 (Entity linking, EL) | Wikipedia 링크 연결 | 언급(mention)을 고유한 KB 엔티티로 매핑합니다. |
| 후보 생성 (Candidate generation) | 누구일까요? | 언급에 대해 가능성 있는 KB 엔티티들의 후보 목록을 반환합니다. |
| 모호성 해소 (Disambiguation) | 올바른 것 선택하기 | 문맥을 사용하여 후보들의 점수를 매기고, 최적의 후보를 선택합니다. |
| 별칭 인덱스 (Alias index) | 조회 테이블 | 표면형(surface form) → 후보 엔티티로 매핑합니다. |
| NIL | KB에 없음 | 일치하는 KB 엔티티가 없음을 명시적으로 예측하는 것입니다. |
| KB | 지식 베이스 (Knowledge base) | Wikidata, Wikipedia, DBpedia 또는 해당 도메인의 KB입니다. |
| AIDA-CoNLL | 벤치마크 | 정답(gold) 엔티티 링크가 포함된 1,393개의 Reuters 기사입니다. |

## 추가 학습 자료 (Further Reading)

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 기초적인 사전 지식 및 문맥(prior+context) 접근 방식입니다.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 임베딩 기반의 핵심 모델(workhorse)입니다.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 제약 디코딩(constrained decoding)을 사용하는 생성형 EL입니다.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 벤치마크 논문입니다.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 오픈 소스 프로덕션 스택입니다.
