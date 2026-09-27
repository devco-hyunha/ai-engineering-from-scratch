# 관계 추출 및 지식 그래프 구축 (Relation Extraction & Knowledge Graph Construction)

> NER은 개체(entity)를 찾아냅니다. 개체 연결(Entity linking)은 이를 고정합니다. 관계 추출(Relation extraction)은 개체 사이의 엣지(edge)를 찾아냅니다. 지식 그래프(Knowledge graph)는 노드, 엣지, 그리고 그 출처(provenance)의 총합입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## 문제 (The Problem)

한 분석가가 다음과 같은 문장을 읽습니다: "Tim Cook은 2011년에 Apple의 CEO가 되었습니다." 네 가지 사실이 도출됩니다:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

관계 추출(Relation Extraction, RE)은 자유 형식의 텍스트를 구조화된 트리플 `(subject, relation, object)`로 변환합니다. 코퍼스 전체에 걸쳐 이를 집계하면 지식 그래프(Knowledge Graph)가 됩니다. 이를 집계하고 쿼리하면 RAG, 분석 또는 컴플라이언스 감사를 위한 추론 기반(reasoning substrate)을 갖추게 됩니다.

2026년의 문제: LLM은 관계를 매우 열정적으로 추출합니다. 너무 열정적입니다. 원문이 뒷받침하지 않는 트리플을 환각(hallucinate)하여 생성합니다. 출처(provenance)가 없다면, 실제 트리플과 그럴듯한 허구를 구분할 수 없습니다. 2026년의 해답은 AEVS 방식의 앵커 및 검증(anchor-and-verify) 파이프라인입니다.

## 개념 (The Concept)

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**트리플 형태 (Triple form).** `(subject_entity, relation_type, object_entity)`. 관계(Relations)는 폐쇄형 온톨로지(Wikidata 속성, FIBO, UMLS) 또는 개방형 세트(OpenIE 스타일, 모든 것이 가능)에서 가져옵니다.

**세 가지 추출 접근 방식 (Three extraction approaches).**

1. **규칙 / 패턴 기반 (Rule / pattern-based).** Hearst 패턴: "X와 같은 Y" → `(Y, isA, X)`. 여기에 수동으로 제작된 정규 표현식(regex)이 추가됩니다. 취약하지만, 정밀하며 설명 가능합니다.
2. **지도 학습 분류기 (Supervised classifier).** 문장 내 두 엔티티 언급(entity mentions)이 주어지면, 정해진 세트에서 관계를 예측합니다. TACRED, ACE, KBP 데이터셋으로 학습됩니다. 2015~2022년 사이의 표준 방식입니다.
3. **생성형 LLM (Generative LLM).** 모델이 트리플을 출력하도록 프롬프트를 작성합니다. 즉시 사용 가능합니다. 출처(provenance) 확인이 필요하며, 그렇지 않으면 그럴듯해 보이는 쓰레기 정보(hallucinates plausible-looking junk)를 생성할 수 있습니다.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** 현재의 환각 완화 프레임워크입니다:

- **앵커 (Anchor).** 모든 엔티티 스팬(entity span)과 관계 구절(relation-phrase) 스팬의 정확한 위치를 식별합니다.
- **추출 (Extract).** 앵커 스팬과 연결된 트리플을 생성합니다.
- **검증 (Verify).** 각 트리플 요소를 원문 텍스트와 다시 대조합니다. 근거가 없는 것은 모두 거부합니다.
- **보완 (Supplement).** 커버리지 패스(coverage pass)를 통해 앵커링된 스팬이 누락되지 않도록 보장합니다.

환각 현상이 급격히 감소합니다. 더 많은 연산량이 필요하지만 감사(auditable)가 가능합니다.

**개방형 대 폐쇄형 트레이드오프 (The open-vs-closed tradeoff).**

- **폐쇄형 온톨로지 (Closed ontology).** 고정된 속성 목록(예: Wikidata의 11,000개 이상의 속성). 예측 가능합니다. 쿼리가 가능합니다. 새로운 것을 만들어내기 어렵습니다.
- **개방형 IE (Open IE).** 모든 동사 구절이 관계가 됩니다. 재현율(recall)이 높습니다. 정밀도(precision)가 낮습니다. 쿼리하기에 복잡합니다.

실제 운영되는 지식 그래프(KG)는 대개 이를 혼합하여 사용합니다: 발견을 위해 개방형 IE를 사용한 다음, 메인 그래프에 병합하기 전에 관계를 폐쇄형 온톨로지로 정규화(canonicalize)합니다.

```figure
relation-triples
```

## 구축하기 (Build It)

### 1단계: 패턴 기반 추출 (pattern-based extraction)

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

전체 예제 추출기(toy extractor)는 `code/main.py`를 참조하세요. Hearst 패턴은 디버깅이 가능하다는 장점 때문에 여전히 도메인 특화 파이프라인(domain-specific pipelines)에서 사용되고 있습니다.

### 2단계: 지도 관계 분류 (Supervised Relation Classification)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL은 seq2seq 관계 추출기(relation extractor)입니다. 텍스트를 입력하면 Wikidata 속성 ID가 포함된 트리플(triples)을 출력합니다. 원격 지도 학습(distant-supervision) 데이터로 미세 조정(fine-tuned)되었으며, 표준적인 오픈 웨이트(open-weights) 베이스라인 모델입니다.

### 3단계: 앵커링을 활용한 LLM 프롬프트 기반 추출 (LLM-prompted extraction with anchoring)

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

반환된 모든 `span`을 원문과 대조하여 검증하세요. `text[start:end] != triple_entity`인 경우 모두 거부해야 합니다. 이것이 AEVS의 "검증(verify)" 단계의 가장 단순한 형태입니다.

### 4단계: 폐쇄형 온톨로지로 정규화(Canonicalize)하기

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 매핑되지 않은 개방형 관계는 삭제하거나 수동 검토로 라우팅합니다.
```

정규화(Canonicalization) 작업은 종종 엔지니어링 업무의 60-80%를 차지합니다. 이에 대한 예산을 충분히 확보해 두세요.

### 5단계: 작은 그래프 구축 및 쿼리(Query)

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

이것은 모든 RAG-over-KG(지식 그래프 기반 RAG) 시스템의 최소 단위(atom)입니다. RDF 트리플 스토어(Blazegraph, Virtuoso), 프로퍼티 그래프(Neo4j), 또는 벡터 증강 그래프 저장소(vector-augmented graph stores)를 사용하여 이를 확장해 보세요.

## 주의 사항 (Pitfalls)

- **RE(관계 추출) 전 상호 참조 해결(Coreference before RE).** "He founded Apple" — RE를 수행하려면 "he"가 누구인지 알아야 합니다. 먼저 상호 참조 해결(lesson 24)을 수행하세요.
- **개체 정규화(Entity canonicalization).** "Apple Inc"와 "Apple"은 동일한 노드로 수렴되어야 합니다. 먼저 개체 연결(Entity linking, lesson 25)을 수행하세요.
- **환각 트리플(Hallucinated triples).** LLM은 본문이 지원하지 않는 트리플을 생성할 수 있습니다. 구간 검증(span verification)을 강제하세요.
- **관계 정규화 드리프트(Relation canonicalization drift).** Open IE 관계는 일관성이 없습니다 ("was born in", "came from", "is a native of"). 관계를 정규화된 ID로 통합하지 않으면 그래프를 쿼리할 수 없습니다.
- **시간적 오류(Temporal errors).** "Tim Cook is CEO of Apple" — 현재는 참이지만, 2005년에는 거짓입니다. 많은 관계는 시간적 경계가 있습니다. 한정자(qualifiers)를 사용하세요 (Wikidata의 `P580` 시작 시간, `P582` 종료 시간 등).
- **도메인 불일치(Domain mismatch).** REBEL은 Wikipedia로 학습되었습니다. 법률, 의료 및 과학 텍스트에는 종종 도메인 미세 조정(domain-fine-tuned)된 RE 모델이 필요합니다.

## 활용 방법 (Use It)

2026년형 스택:

| 상황 (Situation) | 선택 (Pick) |
|-----------|------|
| 빠른 생산, 일반 도메인 | REBEL 또는 Wikidata 정규화(canonicalization)를 적용한 LlamaPred |
| 특정 도메인 (바이오메디컬, 법률) | SciREX 스타일의 도메인 미세 조정(fine-tune) + 커스텀 온톨로지 |
| LLM 프롬프트 기반, 검증된 출력 | AEVS 파이프라인: 앵커(anchor) → 추출(extract) → 검증(verify) → 보완(supplement) |
| 대량의 뉴스 정보 추출(IE) | 패턴 기반 + 지도 학습 하이브리드 방식 |
| 처음부터 지식 그래프(KG) 구축 | Open IE + 수동 정규화 단계 |
| 시계열 지식 그래프 (Temporal KG) | 한정자(qualifiers)를 포함한 추출 (시작/종료 시간, 특정 시점) |

통합 패턴: NER → coref → 엔티티 연결(entity linking) → 관계 추출(relation extraction) → 온톨로지 매핑(ontology mapping) → 그래프 로드(graph load). 모든 단계는 잠재적인 품질 게이트(quality gate) 역할을 합니다.

## Ship It

`outputs/skill-re-designer.md`로 저장하세요:

```markdown
---
name: re-designer
description: 출처(provenance) 및 정규화(canonicalization)를 포함한 관계 추출(relation extraction) 파이프라인을 설계합니다.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

코퍼스(도메인, 언어, 규모)와 하위 작업 용도(KG-RAG, 분석, 컴플라이언스)가 주어졌을 때, 다음을 출력하세요:

1. 추출기(Extractor). 패턴 기반 / 지도 학습 / LLM / AEVS 하이브리드 방식 중 선택. 정밀도(precision) 대 재현율(recall) 목표에 근거한 이유를 제시할 것.
2. 온톨로지(Ontology). 폐쇄형 속성 목록(Wikidata / 도메인 기반) 또는 정규화 과정을 거치는 개방형 IE(open IE).
3. 출처(Provenance). 모든 트리플(triple)은 소스 문자 범위(char-span)와 문서 ID를 포함해야 함. 감사를 위해 필수 사항임.
4. 병합 전략(Merge strategy). 정규화된 엔티티 ID + 관계 ID + 시간적 한정자(temporal qualifiers); 중복 제거(dedup) 정책.
5. 평가(Evaluation). 수동으로 라벨링된 200개 트리플에 대한 정밀도/재현율 + LLM 추출 샘플에 대한 환각률(hallucination-rate).

스팬 검증(span verification, 소스 출처)이 없는 LLM 기반 RE 파이프라인은 거부하세요. 정규화 과정 없이 프로덕션 그래프로 유입되는 개방형 IE 출력도 거부하세요. 시간 제한이 있는 관계(고용주, 배우자, 직책)에 시간적 한정자가 없는 파이프라인은 경고(flag)를 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 있는 패턴 추출기(pattern extractor)를 5개의 뉴스 기사 문장에 실행해 보세요. 정밀도(precision)를 수동으로 확인해 보세요.
2. **중간 (Medium).** 동일한 문장에 REBEL(또는 소형 LLM)을 사용해 보세요. 추출된 트리플(triples)을 비교해 보세요. 어떤 추출기가 더 높은 정밀도(precision)를 보이나요? 어떤 추출기가 더 높은 재현율(recall)을 보이나요?
3. **어려움 (Hard).** AEVS 파이프라인을 구축해 보세요: LLM으로 추출한 후 소스 문장과 대조하여 스팬(spans)을 검증합니다. 50개의 위키피디아 스타일 문장을 사용하여 검증(verify) 단계 전후의 환각률(hallucination rate)을 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 (Term) | 통용되는 표현 (What people say) | 실제 의미 (What it actually means) |
|------|-----------------|-----------------------|
| Triple (트리플) | 주어-관계-목적어 | 지식 그래프(KG)의 원자 단위인 `(s, r, o)` 튜플. |
| Open IE (개방형 정보 추출) | 무엇이든 추출하기 | 개방형 어휘 관계 구문; 높은 재현율(recall), 낮은 정밀도(precision). |
| Closed ontology (폐쇄형 온톨로지) | 고정된 스키마 | 제한된 관계 유형 집합 (Wikidata, UMLS, FIBO 등). |
| Canonicalization (정규화) | 모든 것을 표준화하기 | 표면적 이름/관계를 표준 ID(canonical ids)로 매핑. |
| AEVS | 근거 기반 추출 | Anchor-Extraction-Verification-Supplement 파이프라인 (2026). |
| Provenance (출처/계보) | 진실의 근원 링크 | 모든 트리플은 소스 문서의 doc id와 char-span 정보를 포함함. |
| Distant supervision (원격 지도 학습) | 저렴한 레이블링 | 기존 KG와 텍스트를 정렬하여 학습 데이터를 생성하는 방식. |

## 추가 읽을거리 (Further Reading)

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 원격 지도 학습(distant-supervision)에 관한 논문입니다.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq 방식의 관계 추출(RE) 핵심 연구입니다.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 통합 정보 추출(joint IE) 연구입니다.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026년형 환각 완화(hallucination-mitigation) 설계 프레임워크입니다.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 표준 그래프 쿼리(canonical graph queries) 튜토리얼입니다.
