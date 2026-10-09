# 관계 추출 및 지식 그래프 구축

> NER가 엔티티를 찾았습니다. 엔티티 링크가 이를 고정했습니다. 관계 추출은 엔티티 간의 간선을 찾습니다. 지식 그래프는 노드, 간선 및 그 출처의 합입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 06강 (NER), 5단계 · 25강 (엔티티 링크)
**시간:** 약 60분

## 문제점

분석가가 "Tim Cook became CEO of Apple in 2011."를 읽습니다. 네 가지 사실:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

관계 추출(RE)은 자유 텍스트를 구조화된 삼중항 `(subject, relation, object)`으로 변환합니다. 코퍼스를 집계하면 지식 그래프가 됩니다. 집계하고 쿼리하면 RAG, 분석, 컴플라이언스 감사의 추론 기반이 됩니다.

2026년의 문제: LLM은 관계를 열정적으로 추출합니다. 너무 열정적입니다. 원문 텍스트가 지지하지 않는 삼중항을 환각합니다. 출처가 없으면 실제 삼중항과 그럴듯한 허구를 구별할 수 없습니다. 2026년의 답은 AEVS 스타일의 앵커 및 검증 파이프라인입니다.

## 개념

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**삼중항 형식.** `(subject_entity, relation_type, object_entity)`. 관계는 폐쇄적 온톨로지(Wikidata 속성, FIBO, UMLS) 또는 개방형 세트(OpenIE 스타일, 무엇이든 허용)에서 나옵니다.

**세 가지 추출 접근법.**

1. **규칙 / 패턴 기반.** Hearst 패턴: "X such as Y" → `(Y, isA, X)`. 그리고 수작업 정규식. 취약하지만, 정밀하고 설명 가능합니다.
2. **지도 학습 분류기.** 문장 내 두 엔티티 언급이 주어지면 고정된 세트에서 관계를 예측합니다. TACRED, ACE, KBP로 학습됩니다. 2015–2022년 표준입니다.
3. **생성형 LLM.** 모델이 삼중항을 출력하도록 프롬프트합니다. 즉시 작동합니다. 출처가 필요하며, 그렇지 않으면 그럴듯한 쓰레기를 환각합니다.

**AEVS (앵커-추출-검증-보충, 2026).** 현재 환각 완화 프레임워크:

- **앵커.** 모든 엔티티 스팬 및 관계 구문 스팬을 정확한 위치로 식별합니다.
- **추출.** 앵커 스팬에 연결된 삼중항을 생성합니다.
- **검증.** 각 삼중 요소가 원문과 일치하는지 확인하고, 근거가 없는 내용은 거부합니다.
- **보충.** 커버리지 패스를 통해 앵커링된 스팬이 누락되지 않도록 보장합니다.

환각(Hallucination)가 급격히 감소합니다. 더 많은 연산 자원이 필요하지만 감사(audit)가 가능합니다.

**개방형 대 폐쇄형 트레이드오프.**

- **폐쇄형 온톨로지.** 고정된 속성 목록(예: Wikidata의 11,000개 이상의 속성). 예측 가능하고, 쿼리 가능하며, 발명하기 어렵습니다.
- **개방형 IE.** 모든 동사구가 관계가 됩니다. 재현율이 높습니다. 정밀도가 낮습니다. 쿼리하기 복잡합니다.

프로덕션 KG는 보통 혼합 방식을 사용합니다: 발견을 위해 개방형 IE를 사용하고, 메인 그래프에 병합하기 전에 관계를 폐쇄형 온톨로지로 정규화합니다.

```figure
relation-triples
```

## 구현하기

### 1단계: 패턴 기반 추출

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

`code/main.py`에서 전체 토이 추출기를 참조하세요. Hearst 패턴은 디버깅이 쉽기 때문에 도메인 특화 파이프라인에 여전히 사용되고 있습니다.

### 2단계: 지도 학습 관계 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL은 시퀀스 투 시퀀스(seq2seq) 관계 추출기입니다: 텍스트가 입력되고, 삼중(triples)이 출력되며, 이미 Wikidata 속성 ID가 포함됩니다. 원격 지도 학습(distant-supervision) 데이터로 미세 조정(Fine-tuning)되었습니다. 표준 오픈 웨이츠(open-weights) 기준선입니다.

### 3단계: 앵커링을 포함한 LLM 프롬프트 기반 추출

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

반환된 모든 스팬이 원문과 일치하는지 검증합니다. `text[start:end] != triple_entity`인 경우 거부합니다. 이는 AEVS의 "검증" 단계를 최소한으로 구현한 것입니다.

### 4단계: 폐쇄형 온톨로지로 정규화

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (주어/목적어 역전)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 매핑되지 않은 개방형 관계를 버리거나 수동 검토로 라우팅
```

정규화(Canonicalization)는 종종 엔지니어링 작업의 60-80%를 차지합니다. 이를 위해 예산을 배정하세요.

### 5단계: 작은 그래프를 구축하고 쿼리

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

이는 모든 RAG-over-KG 시스템의 기본 단위입니다. RDF 삼중 스토어(Blazegraph, Virtuoso), 속성 그래프(Neo4j) 또는 벡터 증강 그래프 스토어를 사용하여 확장하세요.

## 함정

- **RE 전의 상호참조(Coreference).** "He founded Apple" — RE는 "he"가 누구인지 알아야 합니다. 먼저 상호참조 처리를 수행하세요(24강).
- **엔트리 정규화.** "Apple Inc"와 "Apple"은 동일한 노드로 해석되어야 합니다. 엔티티 링크를 먼저 수행하세요 (25강).
- **환각된 삼중항.** LLM은 텍스트가 지지하지 않는 삼중항을 생성합니다. 스팬 검증을 강제하세요.
- **관계 정규화 드리프트.** Open IE 관계는 일관성이 없습니다 ("was born in," "came from," "is a native of"). 정규화된 id로 수렴하지 않으면 그래프가 쿼리 불가능해집니다.
- **시간적 오류.** "Tim Cook is CEO of Apple" — 현재는 참이지만 2005년에는 거짓입니다. 많은 관계는 시간 범위가 있습니다. 수식어를 사용하세요 (`P580` 시작 시간, `P582` 종료 시간, Wikidata).
- **도메인 불일치.** REBEL은 Wikipedia로 학습되었습니다. 법률, 의료 및 과학 텍스트는 도메인 미세 조정된 RE 모델이 필요한 경우가 많습니다.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 빠른 프로덕션, 일반 도메인 | REBEL 또는 LlamaPred + Wikidata 정규화 |
| 도메인 특화 (생물 의학, 법률) | SciREX 스타일 도메인 미세 조정 + 사용자 정의 온톨로지 |
| LLM 프롬프트, 감사된 출력 | AEVS 파이프라인: 앵커 → 추출 → 검증 → 보충 |
| 대량 뉴스 IE | 패턴 기반 + 지도 학습 하이브리드 |
| KG를 처음부터 구축 | Open IE + 수동 정규화 패스 |
| 시간 KG | 수식어와 함께 추출 (시작/종료 시간, 시점) |

통합 패턴: NER → 코어퍼런스 → 엔티티 링크 → 관계 추출 → 온톨로지 매핑 → 그래프 로드. 모든 단계는 잠재적인 품질 게이트입니다.

## 출시하기

`outputs/skill-re-designer.md`로 저장:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 패턴 추출기를 5개의 뉴스 기사 문장에 대해 실행하세요. 정밀도를 수동으로 확인하세요.
2. **중간.** 동일한 문장에 대해 REBEL (또는 작은 LLM)을 사용하세요. 삼중항을 비교하세요. 어느 추출기가 정밀도가 더 높습니까? 재현율이 더 높습니까?
3. **어려움.** AEVS 파이프라인을 구축하세요: LLM으로 추출 + 스팬을 원본과 대조하여 검증하세요. 50개의 Wikipedia 스타일 문장에서 검증 단계 전후의 환각률을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 트리플 | 주체-관계-객체 | `(s, r, o)` KG의 원자 단위인 튜플입니다. |
| Open IE | 무엇이든 추출 | 오픈 어휘 관계 구문; 높은 재현율, 낮은 정밀도. |
| 폐쇄적 온톨로지 | 고정된 스키마 | 관계 유형의 유한한 집합 (Wikidata, UMLS, FIBO). |
| 정규화 | 모든 것을 표준화 | 표면 이름/관계를 표준 ID로 매핑합니다. |
| AEVS | 그라운딩된 추출 | 앵커-추출-검증-보충 파이프라인 (2026). |
| 출처 | 진실의 원천 링크 | 모든 트리플은 문서 ID와 문자 범위를 통해 출처를 가집니다. |
| 원격 감독 | 저렴한 레이블 | 기존 KG와 텍스트를 정렬하여 학습 데이터를 생성합니다. |

## 추가 읽기

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 원격 감독 논문입니다.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE의 핵심 도구입니다.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 결합 IE입니다.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026년 환각 완화 설계입니다.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 표준화된 그래프 쿼리입니다.
