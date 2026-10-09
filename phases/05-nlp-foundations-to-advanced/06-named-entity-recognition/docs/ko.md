# 고유 명칭 인식(Named Entity Recognition)

> 이름을 추출하는 작업은 쉬워 보이지만, 모호한 경계, 중첩된 엔티티, 도메인 전문 용어를 다루게 되면 상황이 복잡해집니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 5단계 · 03강 (단어 임베딩)
**시간:** 약 75분

## 문제점

"Apple sued Google over its iPhone search deal in the US." 다섯 개의 엔티티가 있습니다: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (아마도), US (GPE). 좋은 NER 시스템은 이 모든 것을 올바른 유형으로 추출합니다. 나쁜 시스템은 iPhone을 놓치고, 과일로서의 Apple과 기업으로서의 Apple을 혼동하며, "US"를 PERSON으로 라벨링합니다.

NER는 모든 구조화된 추출 파이프라인의 기반이 되는 핵심 작업입니다. 이력서 파싱, 컴플라이언스 로그 스캐닝, 의료 기록 익명화, 검색 쿼리 이해, 챗봇 응답을 위한 그라운딩(Grounding), 법률 계약 추출 등. NER는 직접 눈에 띄지 않지만, 항상 그 의존성에 기반하여 작동합니다.

이 강의는 고전적인 경로(규칙 기반, HMM, CRF)에서 현대적인 경로(BiLSTM-CRF, 그리고 트랜스포머)로 이어지는 과정을 다룹니다. 각 단계는 이전 단계의 특정 한계를 해결합니다. 이 패턴 자체가 강의의 핵심입니다.

## 개념

**BIO 태깅**(또는 BILOU)은 엔티티 추출을 시퀀스 라벨링 문제로 변환합니다. 각 토큰에 `B-TYPE` (엔티티 시작), `I-TYPE` (엔티티 내부), 또는 `O` (엔티티 외부)로 라벨링합니다.

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

다중 토큰 엔티티는 체인을 형성합니다: `New B-GPE`, `York I-GPE`, `City I-GPE`. BIO를 이해하는 모델은 임의의 스팬(span)을 추출할 수 있습니다.

아키텍처의 발전 과정은 다음과 같습니다:

- **규칙 기반.** 정규식 + 사전(gazetteer) 조회. 알려진 엔티티에 대해서는 높은 정확도를 가지지만, 새로운 엔티티에 대한 커버리지는 없습니다.
- **HMM.** 은닉 마르코프 모델(Hidden Markov Model). 태그가 주어졌을 때 토큰의 방출 확률, 태그 간 전이 확률. 비터비(Viterbi) 디코딩. 라벨링된 데이터로 학습합니다.
- **CRF.** 조건부 랜덤 필드(Conditional Random Field). HMM과 유사하지만 판별적(discriminative)이므로 임의의 특징(단어 형태, 대문자화, 주변 단어)을 혼합할 수 있습니다. 2026년에도 저자원 배포 환경에서의 고전적인 생산용 핵심 도구로 여전히 사용되고 있습니다.
- **BiLSTM-CRF.** 수작업으로 만든 특징 대신 신경망 기반 특징을 사용합니다. LSTM이 문장을 양방향으로 읽고, 그 위에 CRF 레이어가 일관된 태그 시퀀스를 강제합니다.
- **트랜스포머 기반.** 토큰 분류 헤드를 사용하여 BERT를 미세 조정합니다. 가장 높은 정확도를 제공하며, 연산량이 가장 많습니다.

```figure
ner-bio-tagging
```

## 구현하기

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

### 2단계: 수작업으로 만든 특징

고전적(비신경망) NER에서는 특징(feature)이 핵심입니다. 유용한 특징은 다음과 같습니다:

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

`word_shape("iPhone")`는 `xXxxxx`를 반환합니다. `word_shape("USA-2024")`는 `XXX-dddd`를 반환합니다. 대문자 패턴은 고유 명사를 식별하는 데 매우 유용한 신호입니다.

### 3단계: 단순한 규칙 기반 + 사전 기반 기준선

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

프로덕션 가제트(gazetteer)는 Wikipedia와 DBpedia에서 수집한 수백만 개의 항목을 포함합니다. 커버리지(coverage)는 좋습니다. 모호성 해소(disambiguation, `Apple`가 회사인지 과일인지 구분하는 것)는 매우 어렵습니다. 그래서 통계적 모델이 승리했습니다.

### 4단계: CRF 단계 (스케치, 전체 구현 아님)

확률 이론적 기초 없이 50줄로 CRF를 처음부터 구현하는 것은 도움이 되지 않습니다. 대신 `sklearn-crfsuite`를 사용하세요:

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

`c1`와 `c2`는 L1 및 L2 정규화(regularization)입니다. `all_possible_transitions=True`는 모델이 비정상적인 시퀀스(예: `O` 뒤에 `I-ORG`)가 발생하기 어렵다는 것을 학습하도록 허용합니다. 이는 CRF가 사용자가 제약 조건을 직접 작성하지 않아도 BIO 일관성을 강제하는 방식입니다.

### 5단계: BiLSTM-CRF가 추가하는 것

특징이 학습된 형태로 변환됩니다. 입력: 토큰 임베딩(embedding)(GloVe 또는 fastText). LSTM은 왼쪽에서 오른쪽, 오른쪽에서 왼쪽으로 읽습니다. 연결된 은닉 상태(hidden state)는 CRF 출력 레이어를 거칩니다. CRF는 여전히 태그 시퀀스 일관성을 강제하며, LSTM은 수작업으로 만든 특징을 학습된 특징으로 대체합니다.

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

CRF 레이어에는 `torchcrf.CRF`를 사용하세요(pip install pytorch-crf). 수작업 CRF 대비 성능 향상은 측정 가능하지만, 수만 개의 레이블이 지정된 문장이 없다면 기대보다 작습니다.

## 사용하기

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

`iPhone`가 `PRODUCT`가 아닌 `ORG`로 레이블링된 것을 주목하세요. spaCy의 소형 모델은 제품 엔티티(product entity) 커버리지가 약합니다. 대형 모델(`en_core_web_lg`)은 더 잘 작동합니다. 트랜스포머 모델(`en_core_web_trf`)은 그보다 더 잘 작동합니다.

BERT 기반 NER을 위한 Hugging Face:

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

`aggregation_strategy="simple"`는 연속적인 B-X, I-X 토큰을 하나의 스팬(span)으로 병합합니다. 이 기능이 없으면 토큰 단위 레이블만 얻게 되며, 직접 병합해야 합니다.

### LLM 기반 NER (2026년 옵션)

제로샷 및 소수 예시 LLM NER은 현재 많은 분야에서 미세 조정된 모델과 경쟁할 수 있으며, 레이블이 지정된 데이터가 부족한 경우 훨씬 더 뛰어난 성능을 보입니다.

- **제로샷 프롬팅.** LLM에 엔티티 유형 목록과 예시 스키마를 제공합니다. JSON 출력을 요청합니다. 즉시 작동하며, 새로운 분야에서의 정확도는 중간 수준입니다.
- **ZeroTuneBio 스타일 프롬팅.** 작업을 후보 추출 → 의미 설명 → 판단 → 재검토로 분해합니다. 다단계 프롬프트(원샷이 아님)는 생물의학 NER의 정확도를 크게 높입니다. 동일한 패턴은 법률, 금융 및 과학 분야에서도 작동합니다.
- **RAG를 활용한 동적 프롬팅.** 모든 추론 호출마다 작은 주석된 시드 세트에서 가장 유사한 레이블 지정된 예제를 검색하고, 즉석에서 소수 예시 프롬프트를 구축합니다. 2026년 벤치마크에서는 이 방법이 GPT-4 생물의학 NER의 F1을 정적 프롬팅 대비 11-12% 높입니다.
- **엔티티 유형별 분해.** 긴 문서의 경우 모든 엔티티 유형을 한 번에 추출하는 단일 호출은 길이가 증가함에 따라 재현율이 떨어집니다. 엔티티 유형마다 추출 패스를 한 번 실행하세요. 추론 비용은 더 높지만, 정확도는 훨씬 더 높습니다. 이는 임상 노트 및 법률 계약의 표준 패턴입니다.

2026년 기준 프로덕션 권장 사항: 훈련 데이터를 수집하기 전에 LLM 제로샷 기준선을 시작하세요. F1이 충분히 좋아 미세 조정이 필요하지 않은 경우가 많습니다.

### 고전적 NER이 여전히 승리하는 경우

LLM이available 있음에도 불구하고, 고전적 NER이 승리하는 경우는 다음과 같습니다:

- 지연 시간 예산이 50ms 미만인 경우.
- 수천 개의 레이블 지정된 예제가 있고 98% 이상의 F1이 필요한 경우.
- 분야가 안정된 온톨로지를 가지고 있어 사전 훈련된 CRF 또는 BiLSTM이 잘 전이되는 경우.
- 규제 제약이 온프레미스 비생성형 모델을 요구하는 경우.

### 성능이 저하되는 경우

- **분야 이동.** CoNLL로 훈련된 NER은 법률 계약에서 가제터보다 성능이 떨어집니다. 해당 분야에서 미세 조정하세요.
- **중첩 엔티티.** "Bank of America Tower"는 동시에 ORG와 FACILITY입니다. 표준 BIO는 겹치는 스팬을 표현할 수 없습니다. 중첩 NER(다중 패스 또는 스팬 기반 모델)가 필요합니다.
- **긴 엔티티.** "United States Federal Deposit Insurance Corporation." 토큰 수준 모델은 때때로 이를 분할합니다. `aggregation_strategy`을 사용하거나 후처리하세요.
- **희소 유형.** `DRUG_BRAND`, `ADVERSE_EVENT`, `DOSE`와 같은 의료 NER 레이블입니다. 범용 모델은 이를 전혀 인식하지 못합니다. `Scispacy`와 `BioBERT`가 출발점입니다.

## 출시하기

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

## 연습 문제

1. **쉬움.** `bio_to_spans` (`spans_to_bio`의 역연산)을 구현하고 10개 문장에서 왕복 일관성을 검증해 보세요.
2. **중간.** 위 `sklearn-crfsuite` CRF를 CoNLL-2003 영어 NER 데이터셋으로 학습하세요. `seqeval`를 사용하여 엔티티별 F1을 보고하세요. 일반적인 결과: 약 84 F1.
3. **어려움.** `distilbert-base-cased`를 도메인 특화 NER 데이터셋(의료, 법률, 금융)으로 미세 조정(Fine-tuning)하세요. `spaCy` 소형 모델과 비교하세요. 데이터 누수(Data Leakage) 검사를 문서화하고 놀라웠던 점을 작성해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NER | 이름 추출 | 토큰 스팬에 유형(`PERSON`, `ORG`, `GPE`, `DATE`, ...)을 레이블링합니다. |
| BIO | 태깅 스킴 | `B-X`은 시작, `I-X`은 연속, `O`은 외부입니다. |
| BILOU | 개선된 BIO | 더 깔끔한 경계를 위해 `L-X` (마지막), `U-X` (단위)을 추가합니다. |
| CRF | 구조화된 분류기 | 방출(emission)뿐만 아니라 레이블 간 전이를 모델링합니다. 유효한 시퀀스를 강제합니다. |
| 중첩 NER | 겹치는 엔티티 | 하나의 스팬이 그 하위 스팬과 다른 엔티티인 경우입니다. BIO는 이를 표현할 수 없습니다. |
| 엔티티 수준 F1 | 올바른 NER 지표 | 예측된 스팬이 실제 스팬과 정확히 일치해야 합니다. 토큰 수준 F1은 정확도를 과대평가합니다. |

## 추가 읽기

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 논문입니다. 표준입니다.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 표준이 된 토큰 분류 패턴을 소개합니다.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 및 `Span`의 모든 속성에 대한 실용적인 참고 자료입니다.
- [seqeval](https://github.com/chakki-works/seqeval) — 올바른 지표 라이브러리입니다. 항상 사용하세요.
