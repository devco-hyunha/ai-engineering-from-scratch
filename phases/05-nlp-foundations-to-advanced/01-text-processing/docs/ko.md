# 텍스트 처리 — 토큰화, 스테밍, 표제어 추출 (Text Processing — Tokenization, Stemming, Lemmatization)

> 언어는 연속적입니다. 모델은 이산적입니다. 전처리가 그 다리입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제 (The Problem)

모델은 "The cats were running."을 읽지 못합니다. 정수를 읽습니다.

모든 NLP 시스템은 같은 세 가지 질문으로 시작합니다. 단어는 어디서 시작하는가. 단어의 어근은 무엇인가. "run", "running", "ran"을 도움이 될 때는 같은 것으로, 아닐 때는 다른 것으로 어떻게 다룰 것인가.

토큰화를 잘못하면 모델은 쓰레기로부터 학습합니다. 토크나이저가 `don't`를 한 토큰으로, `do n't`를 두 토큰으로 다루면 학습 분포가 갈라집니다. 스테머가 `organization`과 `organ`을 같은 어간으로 합치면 토픽 모델링이 망가집니다. 표제어 추출기가 품사 맥락이 필요한데 넘기지 않으면 동사가 명사로 취급됩니다.

이 레슨에서는 세 전처리 단계를 처음부터 만든 뒤, NLTK와 spaCy가 같은 일을 어떻게 하는지 보여 트레이드오프를 확인합니다.

## 개념 (The Concept)

세 연산입니다. 각각 역할과 실패 모드가 있습니다.

**토큰화(Tokenization)**는 문자열을 토큰으로 나눕니다. "토큰"은 의도적으로 모호합니다. 올바른 단위는 작업에 따라 달라지기 때문입니다. 고전 NLP에는 단어 단위. 트랜스포머에는 서브워드. 공백이 없는 언어에는 문자.

**스테밍(Stemming)**은 규칙으로 접미사를 잘라냅니다. 빠르고, 공격적이며, 단순합니다. `running -> run`. `organization -> organ`. 두 번째가 실패 모드입니다.

**표제어 추출(Lemmatization)**은 문법 지식을 사용해 단어를 사전 형태로 줄입니다. 더 느리고, 정확하며, 룩업 테이블이나 형태소 분석기가 필요합니다. `ran -> run`("ran"이 "run"의 과거형임을 알아야 함). `better -> good`(비교급 형태를 알아야 함).

경험 법칙입니다. 속도가 중요하고 노이즈를 감수할 수 있으면 스테밍을 쓰세요(검색 인덱싱, 거친 분류). 의미가 중요하면 표제어 추출을 쓰세요(질의응답, 의미 검색, 사용자가 읽을 모든 것).

```figure
edit-distance
```

## 직접 만들기 (Build It)

### 1단계: 정규식 단어 토크나이저

가장 단순하면서도 유용한 토크나이저는 영숫자가 아닌 문자로 나누되 구두점은 자체 토큰으로 유지합니다. 완벽하지도, 최종본도 아니지만 한 줄로 동작합니다.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

우선순위 순의 세 패턴입니다. 선택적 내부 아포스트로피가 있는 단어(`don't`, `it's`). 순수 숫자. 공백·영숫자가 아닌 단일 문자를 독립 토큰으로(구두점).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

주목할 실패 모드입니다. `3pm`은 문자 연속과 숫자 연속을 번갈아 다루기 때문에 `['3', 'pm']`으로 갈라집니다. 대부분의 작업에는 충분합니다. URL, 이메일, 해시태그는 모두 깨집니다. 프로덕션에서는 일반 패턴보다 앞에 패턴을 추가하세요.

### 2단계: Porter 스테머 (1a 단계만)

전체 Porter 알고리즘은 다섯 단계의 규칙을 가집니다. 1a 단계만으로도 가장 흔한 영어 접미사를 다루며 패턴을 가르칩니다.

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

규칙을 위에서 아래로 읽으세요. `ies -> i` 규칙 때문에 `ponies -> poni`이지 `pony`가 아닙니다. 실제 Porter에는 이를 고치는 1b 단계가 있습니다. 규칙은 경쟁합니다. 앞선 규칙이 이깁니다. 순서 하나가 어떤 단일 규칙보다 중요합니다.

### 3단계: 룩업 기반 표제어 추출기

본격적인 표제어 추출에는 형태론이 필요합니다. 교육용으로 다루기 쉬운 버전은 작은 표제어 테이블과 폴백을 씁니다.

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

마지막 사례가 핵심 교육 포인트입니다. `watched`는 테이블에 없고 폴백은 `ing`만 처리합니다. 실제 표제어 추출은 `ed`, 불규칙 동사, 비교급 형용사, 소리 변화가 있는 복수형(`children -> child`)을 다룹니다. 그래서 프로덕션 시스템은 WordNet, spaCy의 morphologizer, 또는 완전한 형태소 분석기를 씁니다.

### 4단계: 파이프로 연결하기

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

빠진 조각은 POS 태거입니다. Phase 5 · 07 (POS Tagging)에서 하나를 만듭니다. 지금은 모두 `NOUN`으로 기본값을 두고 한계를 인정합니다.

## 활용하기 (Use It)

NLTK와 spaCy는 프로덕션 버전을 제공합니다. 각각 몇 줄입니다.

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

`word_tokenize`는 축약형, 유니코드, 정규식이 놓치는 엣지 케이스를 처리합니다. `PorterStemmer`는 다섯 단계를 모두 실행합니다. `WordNetLemmatizer`는 NLTK의 Penn Treebank 체계에서 WordNet 약어 집합으로 변환된 POS 태그가 필요합니다. 위의 매핑 연결 코드가 대부분의 튜토리얼에서 건너뛰는 부분입니다.

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

spaCy는 전체 파이프라인을 `nlp(text)` 뒤에 숨깁니다. 토큰화, POS 태깅, 표제어 추출이 모두 실행됩니다. 규모에서는 NLTK보다 빠릅니다. 기본으로 더 정확합니다. 트레이드오프는 개별 컴포넌트를 쉽게 교체할 수 없다는 점입니다.

### 무엇을 고를지

| 상황 | 선택 |
|------|------|
| 교육, 연구, 컴포넌트 교체 | NLTK |
| 프로덕션, 다국어, 속도가 중요 | spaCy |
| 트랜스포머 파이프라인 (어차피 모델 토크나이저로 토큰화) | `tokenizers` / `transformers`를 쓰고 고전 전처리는 건너뛰기 |

### 아무도 경고하지 않는 두 가지 실패 모드

대부분의 튜토리얼은 알고리즘을 가르치고 끝냅니다. 실제 전처리 파이프라인의 발목을 잡는 두 가지 문제가 있지만 거의 다루어지지 않습니다.

**재현성 드리프트(Reproducibility drift).** NLTK와 spaCy는 버전 간에 토큰화·표제어 추출 동작을 바꿉니다. spaCy 2.x에서 `['do', "n't"]`를 만든 것이 3.x에서는 `["don't"]`가 될 수 있습니다. 모델은 한 분포로 학습되었습니다. 추론은 이제 다른 분포에서 돌아갑니다. 정확도는 조용히 떨어지고 아무도 이유를 모릅니다. `requirements.txt`에 라이브러리 버전을 고정하세요. 샘플 문장 20개의 기대 토큰화를 고정하는 전처리 회귀 테스트를 작성하세요. 업그레이드마다 실행하세요.

**학습 / 추론 불일치(Training / inference mismatch).** 공격적인 전처리(소문자화, 불용어 제거, 스테밍)로 학습하고, 원시 사용자 입력으로 배포하면 성능이 무너집니다. 이것이 가장 흔한 프로덕션 NLP 실패입니다. 학습 중 전처리했다면 추론 중에도 동일한 함수를 실행해야 합니다. 전처리를 서빙 팀이 다시 쓰는 노트북 셀이 아니라 모델 패키지 안의 함수로 배포하세요.

## 산출물 (Ship It)

엔지니어가 세 권의 교과서를 읽지 않고도 전처리 전략을 고르도록 돕는 재사용 프롬프트입니다.

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

## 연습 문제 (Exercises)

1. **쉬움.** `tokenize`를 확장해 URL을 단일 토큰으로 유지하세요. 테스트: `tokenize("Visit https://example.com today.")`는 URL 토큰 하나를 만들어야 합니다.
2. **보통.** Porter 1b 단계를 구현하세요. 단어에 모음이 있고 `ed` 또는 `ing`로 끝나면 제거합니다. 이중 자음 규칙을 처리하세요(`hopping -> hop`, `hopp`가 아님).
3. **어려움.** WordNet을 룩업 테이블로 쓰되 WordNet에 항목이 없으면 Porter 스테머로 폴백하는 표제어 추출기를 만드세요. 태깅된 코퍼스에서 순수 WordNet·순수 Porter 대비 정확도를 측정하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|----------|
| Token | 단어 | 모델이 소비하는 단위. 단어, 서브워드, 문자, 또는 바이트일 수 있음. |
| Stem | 단어의 어근 | 규칙 기반 접미사 제거의 결과. 항상 실제 단어는 아님. |
| Lemma | 사전 형태 | 찾아볼 때의 형태. 올바르게 계산하려면 문법 맥락이 필요함. |
| POS tag | 품사 | NOUN, VERB, ADJ 같은 범주. 정확한 표제어 추출에 필요함. |
| Morphology | 단어 형태 규칙 | 시제, 수, 격에 따라 단어 형태가 어떻게 바뀌는지. 표제어 추출이 이에 의존함. |

## 더 읽어보기 (Further Reading)

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 원논문, 다섯 페이지, 여전히 가장 명확한 설명.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 실제 파이프라인이 어떻게 연결되는지.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 아직 생각해 보지 못한 토큰화 엣지 케이스.
