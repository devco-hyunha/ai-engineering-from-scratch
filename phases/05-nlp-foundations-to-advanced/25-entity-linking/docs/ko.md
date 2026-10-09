# 엔티티 링크 및 모호성 해소

> NER이 "Paris"를 발견했습니다. 엔티티 링크는 다음을 결정합니다: 프랑스의 Paris? Paris Hilton? 텍사스의 Paris? (트로이 왕자) Paris? 링크가 없으면 지식 그래프는 모호한 상태로 남습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 06강 (NER), 5단계 · 24강 (개명 해소)
**시간:** 약 60분

## 문제점

문장이 "Jordan beat the press."라고 읽힙니다. NER이 "Jordan"을 PERSON으로 태깅했습니다. 좋습니다. 하지만 *어떤* Jordan일까요?

- Michael Jordan (농구)?
- Michael B. Jordan (배우)?
- Michael I. Jordan (버클리 ML 교수 — 네, ML 논문에서 이런 혼동은 실제로 존재합니다)?
- Jordan (국가)?
- Jordan (히브리어 이름)?

엔티티 링크(EL)는 각 언급을 지식 베이스(Wikidata, Wikipedia, DBpedia 또는 도메인 KB)의 고유한 항목으로 해소합니다. 두 가지 하위 작업이 있습니다:

1. **후보 생성.** "Jordan"이 주어졌을 때, 어떤 KB 항목이 타당합니까?
2. **모호성 해소.** 컨텍스트를 고려할 때, 어떤 후보가 올바른 후보입니까?

두 단계 모두 학습 가능하며 벤치마킹됩니다. 결합된 파이프라인은 10년 동안 안정적이었습니다. 변하는 것은 모호성 해소기의 품질입니다.

## 개념

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**후보 생성.** 언급 표면 형식("Jordan")이 주어지면 별칭 인덱스에서 후보를 조회합니다. Wikipedia 별칭 사전은 대부분의 명명된 엔티티를 포함합니다: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK 공항, JFK (영화). 일반적인 인덱스는 언급당 10-30개의 후보를 반환합니다.

**모호성 해소: 세 가지 접근법.**

1. **사전 확률 + 컨텍스트 (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. 잘 작동하며, 빠르고, 학습이 필요 없습니다.
2. **임베딩 기반 (ESS / REL / Blink).** 언급 + 컨텍스트를 인코딩합니다. 각 후보의 설명을 인코딩합니다. 최대 코사인 유사도를 선택합니다. 2020-2024년의 기본값입니다.
3. **생성형 (GENRE, 2021; LLM 기반, 2023+).** 엔티티의 표준 이름을 토큰 단위로 디코딩합니다. 유효한 엔티티 이름의 트라이(trie)로 제약되어 출력은 유효한 KB id로 보장됩니다.

**엔드투엔드 vs 파이프라인.** 최신 모델(ELQ, BLINK, ExtEnD, GENRE)은 NER + 후보 생성 + 모호성 해소(NER + candidate generation + disambiguation)를 한 번의 패스로 수행합니다. 파이프라인 시스템은 구성 요소를 교체할 수 있기 때문에 프로덕션 환경에서 여전히 주류입니다.

### 두 가지 측정 지표

- **멘션 재현율(후보 생성).** 골드 멘션(gold mentions) 중 올바른 KB 항목이 후보 목록에 포함되는 비율입니다. 전체 파이프라인의 하한선입니다.
- **모호성 해소 정확도 / F1.** 올바른 후보가 주어졌을 때, top-1이 정답인 빈도입니다.

두 지표를 항상 보고하세요. 후보 재현율이 80%인 시스템에서 모호성 해소 정확도가 99%라면, 전체 파이프라인 성능은 80%입니다.

```figure
gx-entity-linking
```

## 구현하기

### 1단계: 위키피디아 리다이렉트에서 별칭 인덱스 구축

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

위키피디아 별칭 데이터: 약 1,800만 개의 (별칭, 엔티티) 쌍. Wikidata 덤프에서 다운로드하세요. 역방향 인덱스로 저장하세요.

### 2단계: 컨텍스트 기반 모호성 해소

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

자카드 겹침(Jaccard overlap)은 단순한 예시입니다. 임베딩의 코사인 유사도(Cosine Similarity)로 대체하세요(트랜스포머 버전은 `code/main.py`의 2단계 참조).

### 3단계: 임베딩 기반(BLINK 스타일)

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

인덱싱 시점에 KB 엔티티를 한 번 임베딩하세요. 쿼리 시점에 멘션 + 컨텍스트를 한 번 임베딩하고, 후보 풀과 내적(dot-product)을 계산하여 최대값을 선택하세요.

### 4단계: 생성형 엔티티 링크(concept)

GENRE는 엔티티의 위키피디아 제목을 문자 단위로 디코딩합니다. 제약 디코딩(제약 디코딩)(20강 참조)은 유효한 제목만 출력되도록 보장합니다. KB 기반 트라이(trie)와 긴밀하게 통합됩니다. 현대적인 후속 기술은 REL-GEN과 구조화된 출력(structured output)을 사용하는 LLM 프롬프트 기반 EL입니다.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

화이트리스트(Outlines `choice`)와 결합하면, 2026년에 출시할 수 있는 가장 단순한 EL 파이프라인이 됩니다.

### 5단계: AIDA-CoNLL로 평가

AIDA-CoNLL은 표준 EL 벤치마크입니다: 1,393개의 Reuters 기사, 34,000개의 멘션, 위키피디아 엔티티. KB 내 정확도(`P@1`)와 KB 밖 NIL 감지율을 보고하세요.

## 문제점

- **NIL 처리.** 일부 멘션은 KB에 없습니다(신생 엔티티, 잘 알려지지 않은 인물). 시스템은 잘못된 엔티티를 추측하는 대신 NIL을 예측해야 합니다. 별도로 측정됩니다.
- **멘션 경계 오류.** 상류 NER이 부분적인 스팬(span)을 놓칩니다("Bank of America"가 "Bank"로만 태그됨). EL 재현율이 떨어집니다.
- **인기 편향.** 학습된 시스템은 빈번한 엔티티를 과잉 예측합니다. ML 논문에서 "Michael I. Jordan"을 언급하면 농구 선수 Jordan으로 연결되는 경우가 많습니다.
- **언어 간 EL.** 중국어 텍스트의 언급을 영어 Wikipedia 엔티티로 매핑합니다. 다국어 인코더나 번역 단계가 필요합니다.
- **KB 노후화.** 새로운 기업, 이벤트, 인물은 작년 Wikipedia 덤프에 없습니다. 프로덕션 파이프라인은 갱신 루프가 필요합니다.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 범용 영어 + Wikipedia | BLINK 또는 REL |
| 언어 간, KB = Wikipedia | mGENRE |
| LLM 친화적, 하루에 몇 번의 언급 | 후보 목록 + 제약 JSON으로 Claude/GPT-4에 프롬프트 |
| 도메인 특화 KB (의료, 법률) | KB 인식 검색을 사용하는 커스텀 BERT + 도메인 AIDA 스타일 세트에 미세 조정 |
| 극도로 낮은 지연 | 정확한 일치 사전만 사용 (Milne-Witten 기준선) |
| 연구 SOTA | GENRE / ExtEnD / 생성형 LLM-EL |

2026년 출시되는 프로덕션 패턴: NER → 코어퍼런스 → 각 언급에 대한 EL → 클러스터를 클러스터당 하나의 정준 엔티티로 축약. 출력: 문서 내 엔티티당 하나의 KB id, 언급당 하나가 아님.

## 출시하기

`outputs/skill-entity-linker.md`로 저장:

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

1. **쉬움.** `code/main.py`에서 사전+컨텍스트 모호성 해소기를 10개의 모호한 언급(Paris, Jordan, Apple)에 대해 구현하세요. 올바른 엔티티를 수동으로 라벨링하세요. 정확도를 측정하세요.
2. **중간.** 문장 트랜스포머로 50개의 모호한 언급을 인코딩하세요. 각 후보의 설명을 임베딩하세요. 임베딩 기반 모호성 해소와 Jaccard 컨텍스트 중복을 비교하세요.
3. **어려움.** 1k 엔티티 도메인 KB (예: 회사 내 직원 + 제품)를 구축하세요. NER + EL을 엔드투엔드로 구현하세요. 100개의 홀드아웃 문장에 대해 정밀도와 재현율을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 엔티티 링크 (EL) | Wikipedia로 링크 | 언급을 고유 KB 항목에 매핑합니다. |
| 후보 생성 | 누가 될 수 있을까요? | 언급에 대한 타당한 KB 항목의 짧은 목록을 반환합니다. |
| 모호성 해소 | 올바른 하나 선택 | 컨텍스트를 사용해 후보를 점수화하고, 승자를 선택합니다. |
| 별칭 인덱스 | 조회 테이블 | 표면 형태 → 후보 엔티티로 매핑합니다. |
| NIL | KB에 없음 | KB 항목이 일치하지 않는다는 명시적 예측입니다. |
| KB | 지식 베이스 | Wikidata, Wikipedia, DBpedia 또는 도메인 KB입니다. |
| AIDA-CoNLL | 벤치마크 | 골드 엔티티 링크가 포함된 1,393개의 Reuters 기사입니다. |

## 추가 읽기

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — 기본 사전+컨텍스트 접근법입니다.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 임베딩 기반의 주력 도구입니다.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 제약 디코딩(제약 디코딩)을 사용한 생성형 엔티티 링크입니다.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 벤치마크 논문입니다.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 오픈 프로덕션 스택입니다.
