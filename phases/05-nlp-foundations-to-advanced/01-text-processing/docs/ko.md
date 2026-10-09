# 텍스트 처리 — 토큰화, 어간 추출, 원형 복원

> 언어는 연속적입니다. 모델은 이산적입니다. 전처리가 그 다리를 놓습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 2단계 · 14강 (나이브 베이즈)
**시간:** 약 45분

## 문제점

모델은 "The cats were running."을 읽을 수 없습니다. 모델은 정수를 읽습니다.

모든 NLP 시스템은 동일한 세 가지 질문으로 시작합니다. 단어는 어디서 시작하는가. 단어의 어근은 무엇인가. 도움이 될 때는 "run", "running", "ran"을 같은 것으로 취급하고, 도움이 되지 않을 때는 다른 것으로 취급하려면 어떻게 해야 하는가.

토큰화를 잘못하면 모델은 잡음에서 학습합니다. 토크나이저가 `don't`을 하나의 토큰으로 취급하지만 `do n't`을 두 개의 토큰으로 취급한다면, 학습 분포가 분할됩니다. 어간 추출기가 `organization`와 `organ`를 동일한 어간으로 축약한다면, 주제 모델링은 실패합니다. 원형 복원기가 품사 컨텍스트를 필요로 하는데 이를 전달하지 않으면, 동사가 명사로 취급됩니다.

이 강의에서는 세 가지 전처리 단계를 처음부터 구축한 후, NLTK와 spaCy가 동일한 작업을 수행하는 방식을 보여 주어 트레이드오프를 확인할 수 있도록 합니다.

## 개념

세 가지 연산. 각각에는 역할과 실패 모드(실패 양상)가 있습니다.

**토큰화(Tokenization)**는 문자열을 토큰으로 분할합니다. "토큰"은 의도적으로 모호한 용어이며, 올바른 세분화(granularity)는 작업에 따라 달라집니다. 고전 NLP에서는 단어 단위, 트랜스포머에서는 서브워드, 공백이 없는 언어에서는 문자 단위를 사용합니다.

**어간 추출(Stemming)**은 규칙으로 접미사를 잘라냅니다. 빠르고, 공격적이며, 단순합니다. `running -> run`. `organization -> organ`. 두 번째가 실패 모드입니다.

**원형 복원(Lemmatization)**은 문법 지식을 사용하여 단어를 사전 형태(dictionary form)로 줄입니다. 느리지만 정확하며, 룩업 테이블이나 형태소 분석기가 필요합니다. `ran -> run` ("ran"이 "run"의 과거형임을 알아야 함). `better -> good` (비교급 형태를 알아야 함).

경험칙. 속도가 중요하고 잡음을 허용할 수 있을 때(검색 인덱싱, 대략적 분류)는 어간 추출을 사용하세요. 의미가 중요할 때(질문 답변, 시맨틱 검색, 사용자가 읽는 모든 것)는 원형 복원을 사용하세요.

```figure
edit-distance
```

## 구현하기

### 1단계: 정규식 단어 토크나이저

가장 단순한 유용한 토크나이저는 비알파뉴메릭 문자를 기준으로 분할하면서 구두점을 자체 토큰으로 유지합니다. 완벽하지는 않고 최종 형태도 아니지만, 한 줄로 실행됩니다.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

우선순위에 따른 세 가지 패턴. 내부에 선택적 아포스트로피가 포함된 단어(`don't`, `it's`). 순수 숫자. 공백이 아닌 단일 비알파뉴메릭 문자를 독립적인 토큰으로 처리(구두점).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

주의해야 할 실패 모드. `3pm`는 `['3', 'pm']`로 분할됩니다. 문자 연속과 숫자 연속이 번갈아 나타났기 때문입니다. 대부분의 작업에는 충분합니다. URL, 이메일, 해시태그는 모두 깨집니다. 프로덕션 환경에서는 일반적인 패턴보다 먼저 특정 패턴을 추가하세요.

### 2단계: Porter 어간 추출기(1a 단계만)

전체 Porter 알고리즘은 다섯 단계의 규칙으로 구성됩니다. 1a 단계만으로도 가장 흔한 영어 접미사를 다루며 패턴을 가르쳐 줍니다.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

규칙을 위에서 아래로 읽어 보세요. `ies -> i` 규칙 때문에 `ponies -> poni`가 `pony`가 됩니다. 실제 Porter 알고리즘에는 이를 수정하는 1b 단계가 있습니다. 규칙은 서로 경쟁하며, 앞선 규칙이 승리합니다. 단일 규칙보다 순서가 더 중요합니다.

### 3단계: 조회 기반 어원 복원기

proper 어원 복원에는 형태론이 필요합니다. 가르치기 쉬운 버전은 작은 어원 표와 폴백을 사용합니다.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

마지막 케이스가 핵심적인 가르침의 순간입니다. `watched`는 우리 표에 없고, 폴백은 `ing`만 처리합니다. 실제 어원 복원은 `ed`, 불규칙 동사, 비교급 형용사, 소리 변화가 있는 복수형(`children -> child`)을 다룹니다. 그래서 프로덕션 시스템은 WordNet, spaCy의 morphologizer, 또는 완전한 형태 분석기를 사용합니다.

### 4단계: 이들을 파이프라인으로 연결하기

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

누락된 부분은 POS 태거입니다. 5단계 · 07강(POS 태깅)에서 이를 구축합니다. 지금은 모든 것을 `NOUN`로 기본 설정하고 한계를 인정하세요.

## 사용하기

NLTK과 spaCy는 프로덕션 버전을 제공합니다. 각각 몇 줄이면 됩니다.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`는 축약형, 유니코드, 정규식이 놓친 엣지 케이스를 처리합니다. `PorterStemmer`는 다섯 단계 전체를 실행합니다. `WordNetLemmatizer`는 NLTK의 Penn Treebank 스키마를 WordNet의 약어 집합으로 변환한 POS 태그가 필요합니다. 위 번역 연결은 대부분의 튜토리얼이 건너뛰는 부분입니다.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy는 전체 파이프라인을 `nlp(text)` 뒤에 숨깁니다. 토큰화, 품사 태깅, 어간 추출이 모두 실행됩니다. 대규모 환경에서 NLTK보다 빠릅니다. 기본 설정으로 더 정확한 결과를 제공합니다. 단점으로는 개별 구성 요소를 쉽게 교체할 수 없다는 점입니다.

### 언제 무엇을 선택해야 하는가

| 상황 | 선택 |
|-----------|------|
| 교육, 연구, 구성 요소 교체 | NLTK |
| 프로덕션, 다국어 지원, 속도 중요 | spaCy |
| 트랜스포머 파이프라인 (모델의 토크나이저로 토큰화하므로) | `tokenizers` / `transformers`를 사용하고 고전적인 전처리 생략 |

### 누구도 경고하지 않는 두 가지 실패 모드

대부분의 튜토리얼은 알고리즘을 가르치고 멈춥니다. 실제 전처리 파이프라인을 위협하는 두 가지가 있으며, 거의 다루어지지 않습니다.

**재현성 드리프트.** NLTK과 spaCy는 버전 간에 토큰화 및 어간 추출기 동작이 변경됩니다. spaCy 2.x에서 `['do', "n't"]`를 생성했던 것이 3.x에서는 `["don't"]`를 생성할 수 있습니다. 모델은 하나의 분포로 학습되었습니다. 추론은 이제 다른 분포에서 실행됩니다. 정확도가 조용히 저하되고 아무도 이유를 알지 못합니다. `requirements.txt`에서 라이브러리 버전을 고정하세요. 20개 샘플 문장의 예상 토큰화를 고정하는 전처리 회귀 테스트를 작성하세요. 모든 업그레이드 시에 실행하세요.

**학습 / 추론 불일치.** 공격적인 전처리(소문자화, 불용어 제거, 어간 추출)로 학습하고, 원시 사용자 입력에 배포하면 성능이 급격히 떨어집니다. 이는 가장 흔한 프로덕션 NLP 실패입니다. 학습 중에 전처리한다면, 추론 중에도 동일한 함수를 실행해야 합니다. 전처리를 모델 패키지 내의 함수로 출시하세요. 서빙 팀이 다시 작성하는 노트북 셀로 남기지 마세요.

## 출시하기

엔지니어가 세 권의 교과서를 읽지 않고 전처리 전략을 선택하도록 돕는 재사용 가능한 프롬프트입니다.

`outputs/prompt-preprocessing-advisor.md`로 저장하세요:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 연습 문제

1. **쉬움.** `tokenize`를 확장하여 URL을 단일 토큰으로 유지하세요. 테스트: `tokenize("Visit https://example.com today.")`는 하나의 URL 토큰을 생성해야 합니다.
2. **중간.** Porter 단계 1b를 구현하세요. 단어가 모음을 포함하고 `ed` 또는 `ing`으로 끝나면 제거하세요. 이중 자음 규칙(`hopping -> hop`, `hopp` 아님)을 처리하세요.
3. **난이도: 상.** WordNet을 조회 테이블로 사용하되, WordNet에 항목이 없을 경우 Porter 스템머로 폴백하는 어간 추출기를 만들어 보세요. 태깅된 코퍼스에서 순수 WordNet 및 순수 Porter 대비 정확도를 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 토큰(Token) | 단어 | 모델이 소비하는 단위. 단어, 하위 단어, 문자, 바이트가 될 수 있습니다. |
| 어간(Stem) | 단어의 뿌리 | 규칙 기반 접미사 제거의 결과. 항상 실제 단어는 아닙니다. |
| 어휘(Lemma) | 사전 형태 | 사전에서 찾을 형태. 정확하게 계산하려면 문법적 컨텍스트가 필요합니다. |
| 품사 태그(POS tag) | 품사 | NOUN, VERB, ADJ와 같은 범주. 정확하게 어간 추출을 하려면 필요합니다. |
| 형태론(Morphology) | 단어 형태 규칙 | 시제, 수, 격에 따라 단어가 형태가 변하는 방식. 어간 추출은 이에 의존합니다. |

## 추가 읽기

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 원본 논문, 5페이지, 여전히 가장 명확한 설명입니다.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 실제 파이프라인이 연결되는 방식입니다.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 아직 생각하지 못한 토큰화 엣지 케이스입니다.
