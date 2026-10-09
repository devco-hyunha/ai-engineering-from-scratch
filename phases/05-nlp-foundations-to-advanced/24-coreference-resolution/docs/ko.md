# 개체 참조 해석(Coreference Resolution)

> "그녀가 그에게 전화를 걸었다. 그는 답하지 않았다. 의사는 점심 식사 중이었다." 두 사람을 가리키는 세 번의 참조가 있고, 아무도 이름이 언급되지 않았습니다. 개체 참조 해석은 누가 누구인지 파악합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 06강 (NER), 5단계 · 07강 (품사 및 구문 분석)
**시간:** 약 60분

## 문제점

300단어짜리 기사에서 Apple Inc.에 대한 모든 언급을 추출해 보세요. 기사에 "Apple"이라고 적혀 있으면 쉽습니다. "the company", "they", "Cupertino's technology giant", "Jobs's firm"이라고 적혀 있으면 어렵습니다. 이러한 언급들을 동일한 개체로 해석하지 않으면 NER 파이프라인이 언급의 60-80%를 놓칩니다.

개체 참조 해석은 동일한 실제 세계 개체를 가리키는 모든 표현을 하나의 클러스터로 연결합니다. 이는 표면 수준의 NLP(NER, 구문 분석)와 다운스트림 의미론(정보 추출, 질의 응답, 요약, 지식 그래프)을 연결하는 접착제입니다.

2026년 기준 중요성:

- 요약: "CEO가 발표했습니다..." vs "Tim Cook이 발표했습니다..." — 요약은 CEO의 이름을 명시해야 합니다.
- 질의 응답: "그녀가 누구에게 전화를 걸었나요?"는 "그녀"를 해석해야 합니다.
- 정보 추출: "PER1이 Apple을 설립했다"와 "Jobs가 Apple을 설립했다"를 별도 항목으로 가진 지식 그래프는 잘못된 것입니다.
- 다중 문서 정보 추출: 동일한 사건에 대한 기사들 간에 언급을 병합하는 것은 문서 간 개체 참조 해석입니다.

## 개념

![Coreference clustering: mentions → entities](../assets/coref.svg)

**작업.** 입력: 문서. 출력: 각 클러스터가 하나의 개체를 가리키는 언급(span)들의 클러스터링.

**언급 유형.**

- **고유 명사.** "Tim Cook"
- **명사구.** "the CEO", "the company"
- **대명사.** "he", "she", "they", "it"
- **동격어.** "Tim Cook, Apple's CEO,"

**아키텍처.**

1. **규칙 기반(Hobbs, 1978).** 문법 규칙을 사용하는 구문 트리 기반 대명사 해석. 좋은 기준선. 대명사에서 놀라울 정도로 넘기 어렵습니다.
2. **언급 쌍 분류기.** 모든 언급 쌍(m_i, m_j)에 대해 개체 참조 여부를 예측합니다. 전달 폐폐(transitive closure)로 클러스터링합니다. 2016년 이전 표준.
3. **언급 순위 매기기.** 각 언급에 대해 후보 선행어(“선행어 없음” 포함)를 순위 매깁니다. 최상위를 선택합니다.
4. **스팬 기반 엔드투엔드(Lee et al., 2017).** 트랜스포머 인코더. 길이 상한까지 모든 후보 스팬을 열거합니다. 언급 점수를 예측합니다. 각 스팬에 대해 선행어 확률을 예측합니다. 탐욕적으로 클러스터링합니다. 현대의 기본값입니다.
5. **생성형(2024+).** LLM에 프롬프트를 제공합니다: “이 텍스트의 모든 대명사와 그 선행어를 나열하세요.” 쉬운 사례에서는 잘 작동하지만, 긴 문서와 드문 참조 대상에서는 어려움을 겪습니다.

**평가 지표.** 다섯 가지 표준 지표(MUC, B³, CEAF, BLANC, LEA)를 사용합니다. 단일 지표로는 클러스터링 품질을 완전히 포착할 수 없기 때문입니다. 첫 세 지표의 평균을 CoNLL F1로 보고합니다. 2026년 CoNLL-2012 기준 최신 기술은 약 83 F1입니다.

**알려진 어려운 사례.**

- 수 페이지 전에 도입된 엔티티를 참조하는 정관사 구.
- 브리지잉 애나포라(“the wheels” → 이전에 언급된 자동차).
- 중국어와 일본어 같은 언어의 제로 애나포라.
- 카타포라(참조 대상보다 대명사가 먼저 옴): “When **she** walked in, Mary smiled.”

```figure
coref-links
```

## 구현하기

### 1단계: 사전 학습된 신경망 코어퍼런스(AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 실험적 모델
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

더 긴 문서에서는 다음과 같은 결과를 얻습니다:
- 클러스터 1: [Apple, The company, they]
- 클러스터 2: [new products]

### 2단계: 규칙 기반 대명사 해석기(교육용)

`code/main.py`를 참고하여 stdlib 전용 구현을 보세요:

1. 언급 추출: 고유 명사(대문자 스팬), 대명사(사전 조회), 정관사 구(“the X”).
2. 각 대명사에 대해 이전 K개 언급을 살펴보고 다음 기준으로 점수를 매깁니다:
   - 성별/수 일치(휴리스틱)
   - 최근성(가까운 것이 우선)
   - 통사론적 역할(주어 선호)
3. 가장 높은 점수의 선행어를 연결합니다.

신경망 모델과 경쟁할 수준은 아닙니다. 하지만 엔드투엔드 모델이 결정해야 하는 탐색 공간과 의사결정을 보여줍니다.

### 3단계: 코어퍼런스에 LLM 사용

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

주의해야 할 두 가지 실패 모드입니다. 첫째, LLM이 과잉 병합하는 경우("him"과 "her"가 두 명의 서로 다른 사람을 가리키는데도 하나로 처리)입니다. 둘째, LLM이 긴 문서에서 언급을 조용히 누락하는 경우입니다. 항상 span-offset 검사를 통해 검증해 보세요.

### 4단계: 평가

표준 conll-2012 스크립트는 MUC, B³, CEAF-φ4를 계산하고 평균을 보고합니다. 사내 평가를 위해, 먼저 주석 처리된 테스트 세트에서 span-level 정밀도와 재현율을 시작하고, 그 다음 mention-linking F1을 추가해 보세요.

## 함정

- **싱글톤 폭발.** 일부 시스템은 모든 언급을 각각 자체 클러스터로 보고합니다. B³는 이를 관대하게 처리합니다. MUC는 이를 벌점 처리합니다. 항상 세 가지 지표 모두를 확인하세요.
- **긴 컨텍스트에서의 대명사.** 2,000 토큰이 넘는 문서에서는 성능이 약 15 F1만큼 떨어집니다. 신중하게 청킹하세요.
- **성별 가정.** 하드코딩된 성별 규칙은 비이진 참조 대상, 조직, 동물에 대해 작동하지 않습니다. 학습된 모델이나 중립적 스코어링을 사용하세요.
- **긴 문서에서의 LLM 드리프트.** 단일 API 호출로는 50개 이상의 단서에 걸친 언급을 신뢰할 수 있게 클러스터링할 수 없습니다. 슬라이딩 윈도우 + 병합을 사용하세요.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 단일 문서 | `en_coreference_web_trf` (spaCy-experimental) 또는 AllenNLP 신경 코어프 |
| 다국어 | OntoNotes 또는 Multilingual CoNLL로 학습된 SpanBERT / XLM-R |
| 문서 간 이벤트 코어프 | 전문화된 엔드투엔드 모델 (2025–26 SOTA) |
| 빠른 LLM 기준선 | 구조화된 출력 코어프 프롬프트를 사용한 GPT-4o / Claude |
| 프로덕션 대화 시스템 | 규칙 기반 폴백 + 신경 주 모델 + 중요한 슬롯에 대한 수동 검토 |

2026년에 출시되는 통합 패턴은 다음과 같습니다: 먼저 NER을 실행하고, 코어프를 실행한 후, 코어프 클러스터를 NER 엔티티에 병합합니다. 다운스트림 작업은 언급당 하나의 엔티티가 아니라 클러스터당 하나의 엔티티를 보게 됩니다.

## 출시하기

`outputs/skill-coref-picker.md`로 저장하세요:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 연습 문제

1. **쉬움.** `code/main.py`의 규칙 기반 리졸버를 5개의 수작업 단서에 대해 실행하세요. 정답 대비 mention-link 정확도를 측정하세요.
2. **중간.** 뉴스 기사에 사전 학습된 신경 코어프 모델을 사용하세요. 클러스터를 자체 수동 주석과 비교하세요. 어디에서 실패했나요?
3. **난이도: 상.** 코어퍼런스 강화 NER 파이프라인을 구축해 보세요: 먼저 NER을 수행한 후 코어퍼런스 클러스터를 통해 병합합니다. 100개 기사에서 NER 전용 대비 엔티티 커버리지 개선도를 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 언급(Mention) | 참조 | 엔티티(이름, 대명사, 명사구)를 가리키는 텍스트 스팬입니다. |
| 선행어(Antecedent) | "it"이 가리키는 것 | 이후 언급이 코어퍼런스하는 이전 언급입니다. |
| 클러스터(Cluster) | 엔티티의 언급들 | 모두 동일한 실세계 엔티티를 가리키는 언급들의 집합입니다. |
| 후행 참조(Anaphora) | 후방 참조 | 이후 언급이 이전 언급을 가리킵니다("he" → "John"). |
| 선행 참조(Cataphora) | 전방 참조 | 이전 언급이 이후 언급을 가리킵니다("When he arrived, John..."). |
| 브리징(Bridging) | 암시적 참조 | "I bought a car. The wheels were bad." (그 자동차의 바퀴입니다.) |
| CoNLL F1 | 리더보드에 있는 숫자 | MUC, B³, CEAF-φ4 F1 점수의 평균입니다. |

## 추가 읽기

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 표준 교과서 챕터입니다.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 스팬 기반 엔드투엔드입니다.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 코어퍼런스를 개선하는 사전 학습입니다.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 벤치마크입니다.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 규칙 기반 고전입니다.
