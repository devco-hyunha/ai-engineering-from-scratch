# 상호 참조 해결 (Coreference Resolution)

> "그녀가 그에게 전화했습니다. 그는 받지 않았습니다. 의사는 점심 식사 중이었습니다." 두 명의 인물을 가리키는 세 개의 참조가 있지만, 이름은 언급되지 않았습니다. 상호 참조 해결(Coreference resolution)은 이들이 각각 누구를 가리키는지 찾아냅니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## 문제점 (The Problem)

300단어 분량의 기사에서 Apple Inc.에 대한 모든 언급을 추출한다고 가정해 봅시다. 기사에 "Apple"이라고 명시되어 있다면 쉽습니다. 하지만 "그 회사(the company)", "그들(they)", "쿠퍼티노의 기술 거물(Cupertino's technology giant)", 또는 "Jobs의 기업(Jobs's firm)"이라고 표현되어 있다면 어려워집니다. 이러한 언급들을 동일한 엔티티(entity)로 해결(resolve)하지 못하면, NER 파이프라인은 언급된 내용의 60~80%를 놓치게 됩니다.

상호 참조 해결(Coreference resolution)은 동일한 실제 세계의 엔티티를 가리키는 모든 표현을 하나의 클러스터(cluster)로 연결합니다. 이는 표면 수준의 NLP(NER, 파싱)와 다운스트림 의미론적 작업(IE, QA, 요약, KG)을 이어주는 접착제 역할을 합니다.

2026년에 이것이 중요한 이유:

- **요약(Summarization):** "CEO가 발표했습니다..." vs "Tim Cook이 발표했습니다..." — 요약문에는 CEO의 이름이 명시되어야 합니다.
- **질의응답(Question answering):** "그녀는 누구에게 전화했나요?"라는 질문에 답하려면 "그녀(she)"가 누구인지 해결해야 합니다.
- **정보 추출(Information extraction):** "PER1이 Apple을 설립했다"와 "Jobs가 Apple을 설립했다"를 별개의 항목으로 기록하는 지식 그래프(knowledge graph)는 잘못된 것입니다.
- **다중 문서 정보 추출(Multi-document IE):** 동일한 사건에 관한 여러 기사에서 언급된 내용을 병합하는 것은 교차 문서 상호 참조(cross-document coreference)에 해당합니다.

## 개념 (The Concept)

![Coreference clustering: mentions → entities](../assets/coref.svg)

**태스크.** 입력: 문서. 출력: 각 클러스터가 하나의 엔티티를 가리키도록 언급(mention, span)들을 클러스터링한 결과.

**언급(Mention) 유형.**

- **고유 명사(Named entity).** "Tim Cook"
- **명사구(Nominal).** "the CEO", "the company"
- **대명사(Pronominal).** "he", "she", "they", "it"
- **동격(Appositive).** "Tim Cook, Apple's CEO,"

**아키텍처.**

1. **규칙 기반(Rule-based, Hobbs, 1978).** 문법 규칙을 사용하여 구문 트리(syntactic tree)를 기반으로 대명사를 해소(resolution)합니다. 훌륭한 베이스라인이며, 대명사 처리 능력은 놀라울 정도로 강력합니다.
2. **언급 쌍 분류기(Mention-pair classifier).** 모든 언급 쌍 $(m_i, m_j)$에 대해 두 언급이 동일한 엔티티를 가리키는지 예측합니다. 이후 이행적 폐쇄(transitive closure)를 통해 클러스터링합니다. 2016년 이전의 표준 방식입니다.
3. **언급 순위 지정(Mention-ranking).** 각 언급에 대해 후보 선행사(antecedent, "선행사 없음" 포함)의 순위를 매깁니다. 그중 가장 높은 순위의 항목을 선택합니다.
4. **스팬 기반 엔드투엔드(Span-based end-to-end, Lee et al., 2017).** Transformer 인코더를 사용합니다. 길이 제한 내에서 모든 후보 스팬을 나열합니다. 언급 점수(mention score)를 예측하고, 각 스팬에 대한 선행사 확률을 예측합니다. 이후 탐욕적(greedy) 방식으로 클러스터링합니다. 현재 가장 표준적인 방식입니다.
5. **생성형(Generative, 2024+).** LLM에 "이 텍스트의 모든 대명사와 그 선행사를 나열하라"고 프롬프트를 입력합니다. 쉬운 사례에서는 잘 작동하지만, 문서가 길거나 희귀한 지시 대상(referent)이 등장하면 어려움을 겪습니다.

**평가 지표.** 단일 지표로는 클러스터링 품질을 온전히 포착할 수 없으므로 다섯 가지 표준 지표(MUC, B³, CEAF, BLANC, LEA)를 사용합니다. 처음 세 지표의 평균을 CoNLL F1으로 보고합니다. 2026년 기준 CoNLL-2012 데이터셋의 SOTA(State-of-the-art)는 약 83 F1입니다.

**주요 난제.**

- 몇 페이지 전에 등장한 엔티티를 가리키는 한정 기술(Definite descriptions).
- 가교 조응(Bridging anaphora): "바퀴(the wheels)" $\rightarrow$ 이전에 언급된 "자동차(car)".
- 중국어 및 일본어와 같은 언어에서의 영조응(Zero anaphora).
- 후방 조응(Cataphora, 지시 대상보다 대명사가 먼저 등장): "**She** walked in, and Mary smiled." (그녀가 들어오자, Mary가 미소 지었다.)

```figure
coref-links
```

## 직접 구현해 보기 (Build It)

### 1단계: 사전 학습된 신경망 상호 참조 (pretrained neural coreference) (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 실험적 모델
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

더 긴 문서의 경우, 다음과 같은 결과를 얻을 수 있습니다:
- 클러스터 1: [Apple, The company, they]
- 클러스터 2: [new products]

### 2단계: 규칙 기반 대명사 해소기 (rule-based pronoun resolver) (교육용)

표준 라이브러리만 사용한 구현 예시는 `code/main.py`를 참조하세요:

1. 언급(mentions) 추출: 개체명(named entities, 대문자로 시작하는 구간), 대명사(pronouns, 사전 조회), 한정 기술구(definite descriptions, "the X").
2. 각 대명사에 대해 이전 K개의 언급을 살펴보고 다음 기준에 따라 점수를 매깁니다:
   - 성별/수 일치(gender/number agreement, heuristic)
   - 최신성(recency, 가까울수록 높은 점수)
   - 통사적 역할(syntactic role, 주어 선호)
3. 가장 높은 점수를 받은 선행사(antecedent)와 연결합니다.

신경망 모델(neural models)만큼 성능이 뛰어나지는 않지만, 탐색 공간(search space)과 엔드투엔드(end-to-end) 모델이 내려야 하는 결정들을 이해하는 데 도움이 됩니다.

### 3단계: 상호 참조(coreference)를 위한 LLM 활용

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

주의해야 할 두 가지 실패 모드(failure modes)가 있습니다. 첫째, LLM이 과도하게 병합(over-merge)하는 경우입니다(예: 서로 다른 두 사람을 가리키는 "him"과 "her"를 하나로 묶는 경우). 둘째, 긴 문서에서 LLM이 언급(mention)을 조용히 누락시키는 경우입니다. 항상 span-offset 체크를 통해 검증해 보세요.

### 4단계: 평가 (evaluation)

표준 CoNLL-2012 스크립트는 MUC, B³, CEAF-φ4를 계산하고 평균값을 보고합니다. 사내 평가(in-house eval)를 수행할 때는, 먼저 주석이 달린 테스트 세트에 대한 span-level 정밀도(precision)와 재현율(recall)을 산출한 뒤, mention-linking F1을 추가해 보세요.

## 주의 사항 (Pitfalls)

- **싱글톤 폭발 (Singleton explosion).** 일부 시스템은 모든 언급(mention)을 개별 클러스터로 보고합니다. B³는 관대하지만, MUC는 이를 감점합니다. 항상 세 가지 지표를 모두 확인하세요.
- **긴 문맥에서의 대명사 (Pronouns in long context).** 2,000 토큰이 넘는 문서에서는 F1 점수 기준 성능이 약 15점 하락합니다. 신중하게 청크(chunk)를 나누세요.
- **성별 가정 (Gender assumptions).** 하드코딩된 성별 규칙은 비이진(non-binary) 지칭 대상, 조직, 동물 등에 적용할 때 오류가 발생합니다. 학습된 모델이나 중립적인 점수 산정 방식을 사용하세요.
- **긴 문서에서의 LLM 드리프트 (LLM drift on long docs).** 단일 API 호출만으로는 50개 이상의 단락에 걸쳐 있는 언급들을 안정적으로 클러스터링할 수 없습니다. 슬라이딩 윈도우(sliding-window)와 병합(merge) 방식을 사용하세요.

## 활용하기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 단일 문서 | `en_coreference_web_trf` (spaCy-experimental) 또는 AllenNLP neural coref |
| 다국어 | OntoNotes 또는 Multilingual CoNLL로 학습된 SpanBERT / XLM-R |
| 문서 간 이벤트 상호 참조 (Cross-document event coref) | 특화된 엔드투엔드(end-to-end) 모델 (2025–26 SOTA) |
| 빠른 LLM 베이스라인 | 구조화된 출력(structured-output) coref 프롬프트를 사용한 GPT-4o / Claude |
| 프로덕션 대화 시스템 | 규칙 기반 폴백(fallback) + 신경망 기반 기본 모델 + 중요 슬롯에 대한 수동 검토 |

2026년에 적용될 통합 패턴: 먼저 NER을 실행하고, coref를 실행한 다음, coref 클러스터를 NER 엔티티로 병합합니다. 다운스트림 태스크(downstream tasks)는 언급(mention)당 하나의 엔티티가 아닌, 클러스터당 하나의 엔티티를 보게 됩니다.

## Ship It (실행해 보세요)

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

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 직접 작성한 5개의 단락에 대해 `code/main.py`에 있는 규칙 기반 리졸버(rule-based resolver)를 실행해 보세요. 정답(ground truth)과 비교하여 언급-연결(mention-link) 정확도를 측정해 보세요.
2. **중간 (Medium).** 뉴스 기사에 사전 학습된 신경망 상호 참조(neural coref) 모델을 사용해 보세요. 생성된 클러스터를 직접 수동으로 주석(manual annotation)을 단 결과와 비교해 보세요. 어느 부분에서 실패했나요?
3. **어려움 (Hard).** 상호 참조(coref)가 강화된 NER 파이프라인을 구축해 보세요: 먼저 NER을 수행한 다음, 상호 참조 클러스터를 통해 병합합니다. 100개의 기사를 대상으로 NER만 사용했을 때와 비교하여 개체 커버리지(entity-coverage) 개선 정도를 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| 언급 (Mention) | 참조 | 엔티티(이름, 대명사, 명사구)를 지칭하는 텍스트 구간(스팬). |
| 선행사 (Antecedent) | "it"이 가리키는 대상 | 나중에 나오는 언급이 상호 참조(corefer)하는 이전의 언급. |
| 클러스터 (Cluster) | 엔티티의 언급들 | 모두 동일한 현실 세계의 엔티티를 지칭하는 언급들의 집합. |
| 전방 조응 (Anaphora) | 역방향 참조 | 나중에 나온 언급이 이전에 나온 것을 지칭함 ("he" → "John"). |
| 후방 조응 (Cataphora) | 순방향 참조 | 이전에 나온 언급이 나중에 나올 대상을 지칭함 ("When he arrived, John..."). |
| 가교 조응 (Bridging) | 암시적 참조 | "차를 샀다. 바퀴가 나빴다." (그 차의 바퀴.) |
| CoNLL F1 | 리더보드에 표시되는 숫자 | MUC, B³, CEAF-φ4 F1 점수의 평균. |

## 추가 학습 자료 (Further Reading)

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 표준적인 교과서 챕터입니다.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span 기반의 엔드투엔드(end-to-end) 방식입니다.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 상호 참조(coref) 성능을 향상시키는 사전 학습(pretraining) 방식입니다.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 벤치마크입니다.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 규칙 기반(rule-based)의 고전적인 연구입니다.
