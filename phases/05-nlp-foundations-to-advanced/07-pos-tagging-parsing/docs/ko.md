# 품사 태깅 및 통사 분석

> 문법은 한동안 유행에서 밀려났습니다. 하지만 모든 LLM 파이프라인이 구조화된 추출을 검증해야 했고, 문법은 다시 돌아왔습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 01강 (텍스트 처리), 2단계 · 14강 (나이브 베이즈)
**시간:** 약 45분

## 문제점

01강은 어간 원형화(lemmatization)가 품사 태깅을 필요로 한다고 약속했습니다. `running`이 동사인지 알지 못하면, 어간 원형화기는 이를 `run`로 줄일 수 없습니다. `better`이 형용사인지 알지 못하면, `good`로 줄일 수 없습니다.

그 약속은 하나의 하위 분야를 숨기고 있었습니다. 품사 태깅은 문법 범주를 할당합니다. 통사 분석은 문장의 트리 구조를 복원합니다: 어떤 단어가 어떤 단어를 수식하는지, 어떤 동사가 어떤 인수를 지배하는지. 고전 NLP는 이 두 가지를 정교화하는 데 20년을 보냈습니다. 이후 딥러닝은 이를 사전 학습된 트랜스포머 위의 토큰 분류 작업으로 통합했고, 연구 커뮤니티는 다른 곳으로 이동했습니다.

응용 커뮤니티는 아닙니다. 모든 구조화된 추출 파이프라인은 여전히 내부적으로 품사와 의존 트리(dependency tree)를 사용합니다. LLM이 생성한 JSON은 문법적 제약 조건에 대해 검증됩니다. 질의응답 시스템은 의존 분석을 사용하여 쿼리를 분해합니다. 기계 번역 품질 평가기는 분석 트리의 정렬을 확인합니다.

알아둘 가치가 있습니다. 이 강의는 태그셋, 기준선(baseline), 그리고 처음부터 구현하는 것을 멈추고 spaCy를 호출하는 지점을 소개합니다.

## 개념

**품사 태깅**은 각 토큰에 문법 범주를 라벨링합니다. **Penn Treebank (PTB)** 태그셋은 영어의 기본값입니다. 36개의 태그는 casual reader가 까다롭게 느낄 구별을 포함합니다: `NN` 단수 명사, `NNS` 복수 명사, `NNP` 단수 고유 명사, `VBD` 동사 과거형, `VBZ` 동사 3인칭 단수 현재형 등. **Universal Dependencies (UD)** 태그셋은 더 거칠며(17개 태그) 언어에 독립적입니다; 이는 언어 간 작업의 기본값이 되었습니다.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**통사 분석**은 트리를 생성합니다. 두 가지 주요 스타일:

- **구성 성분 분석(Constituency parsing).** 명사구, 동사구, 전치사구가 서로 중첩됩니다. 출력은 단어를 잎으로 하는 비종결 범주(NP, VP, PP)의 트리입니다.
- **의존 구문 분석.** 각 단어는 의존하는 하나의 중심어(head word)를 가지며, 문법적 관계로 라벨링됩니다. 출력은 모든 간선이 (중심어, 의존어, 관계) 세 쌍(triple)으로 이루어진 트리입니다.

의존 구문 분석은 언어 전반에 걸쳐 깔끔하게 일반화되므로, 특히 어순이 자유로운 언어에서 2010년대에 우세한 방법론이 되었습니다.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 구현하기

### 1단계: 최빈 태그(most-frequent-tag) 기준선

작동하는 가장 단순한 POS 태거입니다. 각 단어에 대해 학습 데이터에서 가장 자주 나타났던 태그를 예측합니다.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brown 코퍼스에서 이 기준선은 약 85%의 정확도를 달성합니다. 좋은 성능은 아니지만, 진지한 모델이 이보다 낮아져서는 안 되는 하한선입니다.

### 2단계: 바이그램 HMM 태거

시퀀스의 결합 확률을 모델링합니다:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

두 개의 테이블: 전이 확률(이전 태그가 주어졌을 때의 태그)과 방출 확률(태그가 주어졌을 때의 단어). 두 확률 모두 카운트와 라플라스 스무딩(Laplace smoothing)을 통해 추정합니다. Viterbi 알고리즘(태그 격자(lattice)에 대한 동적 프로그래밍)으로 디코딩합니다.

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 코퍼스에 대한 바이그램 HMM은 약 93%의 정확도를 달성합니다. 85%에서 93%로의 도약은 주로 전이 확률 때문입니다. 모델은 `DET NOUN`이 흔하고 `NOUN DET`이 희귀하다는 것을 학습합니다.

### 3단계: 현대 태거가 이보다 잘하는 이유

전이 및 방출 확률은 국소적입니다. "I bought a saw"에서 `saw`이 명사이지만 "I saw the movie"에서는 동사라는 것을 포착할 수 없습니다. 임의의 특징(접미사, 단어 형태, 앞뒤 단어, 단어 자체)을 사용하는 CRF는 약 97%의 정확도를 달성합니다. BiLSTM-CRF나 트랜스포머는 약 98% 이상을 달성합니다.

이 작업의 상한은 주석자 간 불일치에 의해 결정됩니다. Penn Treebank에서 인간 주석자는 약 97%의 경우에만 일치합니다. 98%를 넘는 모델은 테스트 세트에 과적합(overfitting)하고 있을 가능성이 높습니다.

### 4단계: 의존 구문 분석 개요

처음부터 완전한 의존 구문 분석을 구현하는 것은 범위 밖입니다. 표준 교과서적 처리는 Jurafsky와 Martin의 저서에 있습니다. 알아야 할 두 가지 고전적인 계열은 다음과 같습니다:

- **전이 기반(Transition-based)** 파서(arc-eager, arc-standard)는 shift-reduce 파서처럼 작동합니다: 토큰을 읽고 스택으로 shift하며, 간선을 생성하는 reduce 동작을 적용합니다. 탐욕(greedy) 디코딩은 빠릅니다. 고전적인 구현은 MaltParser입니다. 현대적인 신경망 버전은 Chen과 Manning의 전이 기반 파서입니다.
- **그래프 기반** 파서(Eisner 알고리즘, Dozat-Manning biaffine)는 모든 가능한 head-dependent edge를 점수화하고 최대 신장 트리를 선택합니다. 느리지만 더 정확합니다.

대부분의 응용 작업에서는 spaCy를 호출하세요:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep` 열을 아래에서 위로 읽으면 문장의 문법적 구조가 드러납니다.

## 사용하기

모든 프로덕션 NLP 라이브러리는 표준 파이프라인의 일부로 POS 및 의존성 파서를 제공합니다.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). 빠르고 정확하며 토큰화 + NER + 어간 추출과 통합되어 있습니다. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (의존 관계).
- **Stanford NLP (stanza)**. CoreNLP의 후속인 Stanford의 라이브러리입니다. 60개 이상의 언어에서 최신 기술(state-of-the-art)을 구현합니다.
- **trankit**. 트랜스포머 기반이며, UD 정확도가 좋습니다.
- **NLTK**. `pos_tag`. 사용 가능하지만 느리고 오래되었습니다. 교육용으로는 적당합니다.

### 2026년에도 여전히 중요한 부분

- **어간 추출(Lemmatization).** 01강에서는 POS가 어간 추출을 올바르게 수행하는 데 필요합니다. 항상.
- **LLM 출력에서의 구조화된 추출.** 생성된 문장이 문법적 제약(예: 주어-동사 일치, 필수 수식어)을 준수하는지 검증하세요.
- **측면 기반 감정 분석.** 의존성 파싱은 어떤 형용사가 어떤 명사를 수식하는지 알려줍니다.
- **쿼리 이해.** "Wes Anderson이 감독하고 Bill Murray가 출연한 영화"는 파싱을 통해 구조화된 제약 조건으로 분해됩니다.
- **언어 간 전이.** UD 태그와 의존 관계는 언어에 독립적이므로, 새로운 언어에 대한 제로샷 구조 분석이 가능합니다.
- **저연산 파이프라인.** 트랜스포머를 배포할 수 없는 경우, POS + 의존성 파싱 + 사전(gazetteer)만으로도 놀라울 정도로 좋은 결과를 얻을 수 있습니다.

## 출시하기

`outputs/skill-grammar-pipeline.md`로 저장하세요:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 연습 문제

1. **쉬움.** 작은 태그가 붙은 코퍼스(예: NLTK의 Brown 하위 집합)에서 가장 빈번한 태그 기준선을 사용하여 홀드아웃 문장에 대한 정확도를 측정하세요. 약 85%의 결과를 확인해 보세요.
2. **중간.** 위의 bigram HMM을 학습하고 태그별 정밀도 및 재현율을 보고하세요. HMM이 가장 많이 혼동하는 태그는 무엇인가요?
3. **난이도: 상.** spaCy의 의존 구문 분석을 사용하여 1000문장 샘플에서 주어-동사-목적어(SVO) 삼중항을 추출하세요. 50개의 수동으로 라벨링된 삼중항에 대해 평가하세요. 추출이 실패하는 경우(수동태, 병렬 구조, 생략된 주어 등)를 문서화하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| POS 태그 | 단어의 유형 | 문법적 범주. PTB는 36개, UD는 17개입니다. |
| Penn Treebank | 표준 태그셋 | 영어 전용. 동사의 시제와 명사의 수를 세분화합니다. |
| Universal Dependencies | 다국어 태그셋 | PTB보다 거친 분류이며, 언어 중립적이고 다국어 작업의 기본값으로 사용됩니다. |
| 의존 구문 분석 | 문장 트리 | 각 단어는 하나의 헤드를 가지며, 각 간선에는 문법적 관계가 있습니다. |
| Viterbi | 동적 프로그래밍 | 방출(emission)과 전이(transition)가 주어졌을 때 확률이 가장 높은 태그 시퀀스를 찾습니다. |

## 추가 읽기

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 및 구문 분석에 대한 표준 교과서적 처리.
- [Universal Dependencies project](https://universaldependencies.org/) — 모든 다국어 파서가 사용하는 다국어 태그셋 및 트리뱅크 컬렉션.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token`에 노출된 모든 속성에 대한 실용적 참조.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 신경망 파서를 주류로 도입한 논문.
