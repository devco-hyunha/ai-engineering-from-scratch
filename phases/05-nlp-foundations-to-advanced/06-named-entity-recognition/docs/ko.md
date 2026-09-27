# 개체명 인식 (Named Entity Recognition)

> 텍스트에서 이름을 추출해 보세요. 모호한 경계, 중첩된 개체, 도메인 전문 용어를 마주하기 전까지는 쉬워 보입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 문제 (The Problem)

"Apple sued Google over its iPhone search deal in the US." 개체는 다섯 개입니다: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (판단에 따라), US (GPE). 좋은 NER 시스템은 이들을 모두 올바른 유형으로 추출합니다. 반면 취약한 시스템은 iPhone을 놓치고, 과일 Apple과 기업 Apple을 혼동하며, "US"를 PERSON으로 잘못 분류합니다.

NER는 모든 구조화 추출 파이프라인 밑바닥을 지탱하는 핵심 기반(workhorse)입니다. 이력서 파싱, 컴플라이언스 로그 분석, 의료 기록 익명화, 검색 쿼리 이해, 챗봇 응답의 그라운딩, 법률 계약 추출까지 겉으로 잘 드러나지는 않지만 우리는 언제나 NER에 의존하고 있습니다.

이 레슨에서는 고전적 접근 방식(규칙 기반, HMM, CRF)에서 출발하여 현대적인 접근 방식(BiLSTM-CRF, 트랜스포머)까지 차례대로 살펴봅니다. 각 단계는 이전 단계가 지닌 구체적인 한계를 해결하며, 이러한 발전 흐름을 이해하는 것이 곧 이번 레슨의 핵심입니다.

## 개념 (The Concept)

**BIO 태깅**(또는 BILOU)은 개체 추출을 시퀀스 라벨링 문제로 바꿉니다. 각 토큰에 `B-TYPE`(개체 시작), `I-TYPE`(개체 내부), 또는 `O`(어떤 개체에도 속하지 않음)를 붙입니다.

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

다중 토큰 개체는 이어집니다: `New B-GPE`, `York I-GPE`, `City I-GPE`. BIO를 이해하는 모델은 임의 스팬을 추출할 수 있습니다.

아키텍처 진행:

- **규칙 기반(Rule-based).** 정규식 + 가제티어(gazetteer) 조회. 이미 알려진 개체에는 높은 정밀도를 보이지만, 새로운 개체에 대해서는 인식 범위(커버리지)가 전혀 없습니다.
- **HMM.** 은닉 마르코프 모델(Hidden Markov Model). 태그가 주어졌을 때의 토큰 방출 확률과 태그 간 전이 확률을 모델링하며, Viterbi 알고리즘으로 디코딩합니다. 라벨링된 데이터로 학습합니다.
- **CRF.** 조건부 무작위장(Conditional Random Field). HMM과 유사하지만 판별적(discriminative) 모델이므로 임의의 다양한 특징(단어 형태, 대소문자 여부, 주변 단어 등)을 유연하게 결합할 수 있습니다. 2026년 현재에도 저자원 환경 배포 시 고전적이지만 가장 신뢰받는 프로덕션 일꾼입니다.
- **BiLSTM-CRF.** 수작업 특징 엔지니어링 대신 신경망 특징 표현을 활용합니다. LSTM이 문장을 양방향으로 읽고, 최상단의 CRF 층이 일관된 태그 시퀀스를 강제합니다.
- **트랜스포머 기반(Transformer-based).** BERT에 토큰 분류 헤드를 얹어 파인튜닝합니다. 가장 높은 정확도를 자랑하지만, 연산 비용 또한 가장 큽니다.

```figure
ner-bio-tagging
```

## 직접 만들기 (Build It)

### 1단계: BIO 태깅 헬퍼

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 2단계: 수작업 특징

고전적(비신경망) NER에서는 어떤 특징(feature)을 설계하느냐가 성능을 결정짓습니다. 유용한 특징은 다음과 같습니다:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`은 `xXxxxx`를 반환합니다. `word_shape("USA-2024")`는 `XXX-dddd`를 반환합니다. 대소문자 패턴은 고유명사를 식별하는 데 매우 유용한 단서(high-signal)가 됩니다.

### 3단계: 간단한 규칙 기반 + 사전 베이스라인

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

실제 프로덕션 가제티어는 Wikipedia나 DBpedia에서 수집한 수백만 개의 항목을 포함합니다. 개체 인식 범위(커버리지)는 훌륭하지만, 문맥에 따른 중의성 해소(과일 `Apple` vs 기업 `Apple`)는 거의 불가능합니다. 이것이 바로 통계 기반 머신러닝 모델이 승리한 이유입니다.

### 4단계: CRF 단계 (스케치, 전체 구현 아님)

확률론 기초 이론 없이 밑바닥부터 50줄짜리 CRF를 구현하는 것은 큰 학습 효과를 주기 어렵습니다. 대신 신뢰할 수 있는 라이브러리인 `sklearn-crfsuite`를 사용하는 것이 좋습니다:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`과 `c2`는 각각 L1, L2 정규화 강도입니다. `all_possible_transitions=True`를 설정하면 모델이 문법에 맞지 않는 시퀀스(예: `O` 바로 뒤에 `I-ORG`가 오는 경우)의 발생 가능성이 매우 낮다는 점을 직접 학습하므로, 별도의 제약 규칙 코드를 작성하지 않아도 CRF가 BIO 일관성을 자연스럽게 보장합니다.

### 5단계: BiLSTM-CRF가 더하는 것

특징을 수작업으로 설계하는 대신 모델이 스스로 학습합니다. 입력으로는 토큰 임베딩(GloVe 또는 fastText)이 주어집니다. LSTM은 문장을 좌→우 및 우→좌 양방향으로 읽고, 연결된 은닉 상태는 최상단의 CRF 출력 층을 거칩니다. CRF는 여전히 태그 시퀀스의 일관성을 강제하며, LSTM은 수작업 특징을 학습된 특징 표현으로 대체합니다.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 층에는 `torchcrf.CRF`를 쓰세요(`pip install pytorch-crf`). 수작업 특징 기반 CRF 대비 성능 향상은 측정 가능하지만, 라벨링된 문장이 수만 개 이상 확보되지 않는다면 기대보다 개선 폭이 작을 수 있습니다.

## 활용하기 (Use It)

spaCy는 프로덕션급 NER를 기본으로 제공합니다.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

`iPhone`이 `PRODUCT`가 아니라 `ORG`로 라벨링된 점에 주목하세요 — spaCy의 소형 모델은 제품 개체 커버리지가 다소 약합니다. 대형 모델(`en_core_web_lg`)이나 트랜스포머 모델(`en_core_web_trf`)을 사용하면 훨씬 더 나은 결과를 얻을 수 있습니다.

Hugging Face의 BERT 기반 NER:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`은 연속된 B-X, I-X 토큰을 하나의 개체 스팬으로 결합합니다. 이 옵션이 없으면 토큰 단위 라벨만 반환되어 직접 병합 로직을 구현해야 합니다.

### LLM 기반 NER (2026년 옵션)

현재 제로샷 및 퓨샷 LLM 기반 NER는 다양한 도메인에서 파인튜닝 모델과 경쟁할 수 있을 만큼 강력하며, 라벨링된 데이터가 부족할 때는 훨씬 뛰어납니다.

- **제로샷 프롬프팅.** LLM에 개체 유형 목록과 예시 스키마를 제공하고 JSON 형식 출력을 요청합니다. 별도 학습 없이 바로 동작하며, 새로운 도메인에서도 준수한 정확도를 보입니다.
- **ZeroTuneBio 스타일 프롬프팅.** 작업을 [후보 추출 → 의미 설명 → 판단 → 재검토]로 분해합니다. 단일 호출이 아닌 다단계 프롬프팅은 바이오메디컬 NER에서 정확도를 대폭 끌어올립니다. 법률, 금융, 과학 등 전문 도메인에도 동일한 패턴을 적용할 수 있습니다.
- **RAG를 활용한 동적 프롬프팅.** 매 추론마다 소규모 주석 시드 데이터셋에서 가장 유사한 라벨 예시들을 검색해 퓨샷 프롬프트를 즉석에서 구성합니다. 2026년 벤치마크에 따르면, 이 방식은 정적 프롬프팅 대비 GPT-4의 바이오메디컬 NER F1 점수를 11~12% 끌어올렸습니다.
- **개체 유형별 단계적 분해.** 긴 문서에서 모든 개체 유형을 한 번에 추출하려고 하면 문서 길이가 늘어날수록 재현율(recall)이 떨어집니다. 개체 유형별로 독립적인 추출 패스를 실행하세요. 추론 비용은 증가하지만 정확도는 비약적으로 향상됩니다. 임상 기록과 법률 계약서 처리의 표준 패턴입니다.

2026년 기준 프로덕션 권장 사항: 대규모 학습 데이터를 수집하기 전에 먼저 LLM 제로샷 베이스라인부터 구축해 보세요. 상당수의 경우 제로샷 F1만으로도 요구 사항을 충족하여 파인튜닝이 전혀 필요 없을 수도 있습니다.

### 고전 NER가 여전히 이기는 경우

LLM을 쉽게 쓸 수 있는 환경에서도 다음과 같은 조건에서는 고전적 NER가 여전히 우위를 점합니다:

- 지연 시간 예산이 50ms 미만으로 극히 짧을 때.
- 수천 개 이상의 라벨 예제가 이미 있고 98%+ 수준의 높은 F1이 필요할 때.
- 사전학습 CRF나 BiLSTM이 잘 전이되는 안정적인 온톨로지를 가진 도메인일 때.
- 규제 및 컴플라이언스 제약으로 온프레미스 비생성형 모델이 필수적일 때.

### 무너지는 지점

- **도메인 시프트(Domain shift).** CoNLL-2003 등 일반 뉴스 데이터로 학습된 NER 모델은 법률 계약서 같은 특수 도메인에서 단순 가제티어보다 못한 성능을 보입니다. 해당 도메인 데이터로 반드시 파인튜닝해야 합니다.
- **중첩 개체(Nested entities).** "Bank of America Tower"는 기업(ORG)이자 건물/시설(FACILITY)입니다. 표준 BIO 태깅 방식은 겹치는 스팬을 표현할 수 없으므로, 멀티패스 또는 스팬 기반의 중첩 NER 모델이 필요합니다.
- **긴 개체명.** "United States Federal Deposit Insurance Corporation"처럼 길이가 긴 개체는 토큰 단위 모델이 중간에 분절해 버리기 쉽습니다. `aggregation_strategy`를 사용하거나 후처리 병합 규칙을 추가해야 합니다.
- **희소하거나 전문적인 개체 유형.** DRUG_BRAND(약품 브랜드), ADVERSE_EVENT(이상 반응), DOSE(용량) 같은 의료 NER 라벨은 범용 모델이 제대로 인식하지 못합니다. 이런 도메인에서는 ScispaCy나 BioBERT를 출발점으로 삼아야 합니다.

## 내보내기 (Ship It)

`outputs/skill-ner-picker.md`로 저장하세요:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 연습 문제 (Exercises)

1. **쉬움.** `bio_to_spans`(`spans_to_bio`의 역변환)를 구현하고 10개 문장에서 상호 변환 일관성을 검증해 보세요.
2. **중간.** 앞서 다룬 `sklearn-crfsuite` 기반 CRF를 CoNLL-2003 영어 NER 데이터셋으로 학습시켜 보세요. `seqeval`을 사용하여 개체별 F1을 보고하세요. (전형적인 기준 성능: 약 84 F1)
3. **어려움.** 도메인 특화 NER 데이터셋(의료, 법률 또는 금융)에 `distilbert-base-cased`를 파인튜닝해 보세요. spaCy 스몰 모델과 비교하고, 데이터 누수(leakage) 검사 과정을 문서화하며 새롭게 발견한 점을 정리하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|----------|
| NER | 이름 추출 | 토큰 스팬에 유형(PERSON, ORG, GPE, DATE, ...)을 라벨링. |
| BIO | 태깅 스키마 | `B-X` 시작, `I-X` 계속, `O` 외부. |
| BILOU | 더 나은 BIO | 경계를 더 깔끔하게 하려고 `L-X`(마지막), `U-X`(단일)를 추가. |
| CRF | 구조화 분류기 | 방출뿐 아니라 라벨 간 전이를 모델링. 유효한 시퀀스를 강제. |
| Nested NER | 겹치는 개체 | 한 스팬이 그 부분 스팬과 다른 개체. BIO로는 표현 불가. |
| Entity-level F1 | 올바른 NER 지표 | 예측 스팬이 정답 스팬과 정확히 일치해야 함. 토큰 수준 F1은 정확도를 과장함. |

## 더 읽어보기 (Further Reading)

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 아키텍처를 정립한 필독 논문입니다.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 표준이 된 토큰 분류 패턴을 소개합니다.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents`와 `Span`의 모든 속성에 대한 실무 참고서.
- [seqeval](https://github.com/chakki-works/seqeval) — 개체 수준 평가를 위한 표준 지표 라이브러리입니다. 항상 사용하세요.
