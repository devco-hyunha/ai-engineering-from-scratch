# 품사 태깅과 구문 파싱 (POS Tagging and Syntactic Parsing)

> 문법은 한동안 유행이 아니었습니다. 그러다 모든 LLM 파이프라인이 구조화 추출을 검증해야 하게 되면서 다시 돌아왔습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제 (The Problem)

레슨 01은 표제어 추출(lemmatization)에 품사(part-of-speech) 태그가 필요하다고 약속했습니다. `running`이 동사인 줄 모르면 lemmatizer는 `run`으로 줄일 수 없습니다. `better`가 형용사인 줄 모르면 `good`으로 줄일 수 없습니다.

그 약속 뒤에는 한 하위 분야 전체가 숨어 있습니다. 품사 태깅은 문법 범주를 부여합니다. 구문 파싱(syntactic parsing)은 문장의 트리 구조를 복원합니다. 어떤 단어가 어떤 단어를 수식하는지, 어떤 동사가 어떤 논항을 지배하는지. 고전 NLP는 둘을 다듬는 데 20년을 썼습니다. 그다음 딥러닝이 이를 사전학습 트랜스포머 위의 토큰 분류 과제로 압축했고, 연구 커뮤니티는 관심을 돌렸습니다.

응용 커뮤니티는 아닙니다. 모든 구조화 추출 파이프라인은 여전히 내부에서 POS와 의존성 트리를 씁니다. LLM이 생성한 JSON은 문법 제약에 대해 검증됩니다. 질의응답 시스템은 의존성 파스를 써서 질의를 분해합니다. 기계번역 품질 평가기는 파스 트리 정렬을 확인합니다.

알아 둘 가치가 있습니다. 이 레슨은 태그셋, 베이스라인, 그리고 직접 구현을 멈추고 spaCy를 호출하는 지점을 소개합니다.

## 개념 (The Concept)

**POS 태깅**은 각 토큰에 문법 범주를 붙입니다. **Penn Treebank (PTB)** 태그셋이 영어의 기본입니다. 태그 36개이며, 일반 독자가 까다롭다고 느끼는 구분을 둡니다. `NN` 단수 명사, `NNS` 복수 명사, `NNP` 고유명사 단수, `VBD` 동사 과거형, `VBZ` 동사 3인칭 단수 현재 등입니다. **Universal Dependencies (UD)** 태그셋은 더 거칠고(17개), 언어에 중립적이며, 교차언어 작업의 기본이 되었습니다.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**구문 파싱(Syntactic parsing)**은 트리를 만듭니다. 주요 스타일은 두 가지입니다.

- **구성 파싱(Constituency parsing).** 명사구, 동사구, 전치사구가 서로 안에 중첩됩니다. 출력은 비단말 범주(NP, VP, PP) 트리이며 단어가 잎입니다.
- **의존성 파싱(Dependency parsing).** 각 단어는 의존하는 단일 머리(head) 단어를 가지며, 문법 관계로 라벨링됩니다. 출력은 모든 엣지가 (head, dependent, relation) 트리플인 트리입니다.

의존성 파싱은 2010년대에 이겼습니다. 특히 어순이 자유로운 언어를 포함해, 언어 전반에 깔끔하게 일반화되기 때문입니다.

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

## 직접 만들기 (Build It)

### 1단계: 최빈 태그(most-frequent-tag) 베이스라인

동작하는 가장 단순한 POS 태거입니다. 각 단어에 대해, 학습에서 가장 자주 붙은 태그를 예측합니다.

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

Brown 코퍼스에서 이 베이스라인은 약 85% 정확도에 도달합니다. 훌륭하지는 않지만, 진지한 모델이 내려가서는 안 되는 바닥입니다.

### 2단계: 바이그램 HMM 태거

시퀀스의 결합 확률을 모델링합니다.

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

표는 두 개입니다. 전이 확률(이전 태그가 주어졌을 때의 태그), 방출 확률(태그가 주어졌을 때의 단어). 둘 다 Laplace 스무딩을 넣은 카운트로 추정합니다. 디코딩은 Viterbi(태그 격자 위의 동적 프로그래밍)입니다.

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

Brown에서 바이그램 HMM은 약 93% 정확도에 도달합니다. 85%에서 93%로의 도약은 대부분 전이 확률 덕분입니다. 모델은 `DET NOUN`이 흔하고 `NOUN DET`이 드물다는 것을 배웁니다.

### 3단계: 현대 태거가 이것을 이기는 이유

전이 + 방출 확률은 국소적입니다. "I bought a saw"에서 `saw`가 명사이고 "I saw the movie"에서 동사인 것을 잡지 못합니다. 임의 특징(접미사, 단어 형태, 앞뒤 단어, 단어 자체)을 쓰는 CRF는 약 97%에 도달합니다. BiLSTM-CRF나 트랜스포머는 약 98%+입니다.

이 과제의 천장(ceiling)은 주석자 불일치가 정합니다. 인간 주석자는 Penn Treebank에서 약 97% 일치합니다. 98%를 넘는 모델은 아마 테스트셋을 과적합하고 있을 가능성이 큽니다.

### 4단계: 의존성 파싱 스케치

처음부터 완전한 의존성 파싱은 범위 밖입니다. 표준 교과서 처리는 Jurafsky and Martin에 있습니다. 알아 둘 고전 계열은 두 가지입니다.

- **전이 기반(Transition-based)** 파서(arc-eager, arc-standard)는 shift-reduce 파서처럼 동작합니다. 토큰을 읽고, 스택에 shift한 뒤, 아크를 만드는 reduce 액션을 적용합니다. 탐욕 디코딩이 빠릅니다. 고전 구현은 MaltParser입니다. 현대 신경망 버전: Chen and Manning의 전이 기반 파서.
- **그래프 기반(Graph-based)** 파서(Eisner's algorithm, Dozat-Manning biaffine)는 가능한 모든 head-dependent 엣지를 점수화하고 최대 신장 트리를 고릅니다. 더 느리지만 더 정확합니다.

대부분의 응용 작업에서는 spaCy를 호출하세요.

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

`dep` 열을 아래에서 위로 읽으면 문장의 문법 구조가 드러납니다.

## 활용하기 (Use It)

모든 프로덕션 NLP 라이브러리는 표준 파이프라인의 일부로 POS와 의존성 파서를 제공합니다.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). 빠르고 정확하며, 토큰화 + NER + lemmatization과 통합됩니다. `token.tag_`(Penn), `token.pos_`(UD), `token.dep_`(의존 관계).
- **Stanford NLP (stanza).** Stanford의 CoreNLP 후속. 60개 이상 언어에서 최첨단.
- **trankit.** 트랜스포머 기반, UD 정확도가 좋습니다.
- **NLTK.** `pos_tag`. 쓸 수 있고, 느리며, 오래되었습니다. 교육용으로는 괜찮습니다.

### 2026년에도 여전히 중요한 곳

- **표제어 추출(Lemmatization).** 레슨 01은 올바른 lemmatization에 POS가 필요합니다. 항상.
- **LLM 출력의 구조화 추출.** 생성된 문장이 문법 제약(예: 주어-동사 일치, 필수 수식어)을 지키는지 검증합니다.
- **측면 기반 감성(Aspect-based sentiment).** 의존성 파스는 어떤 형용사가 어떤 명사를 수식하는지 알려 줍니다.
- **질의 이해.** "movies directed by Wes Anderson starring Bill Murray"는 파스를 통해 구조화된 제약으로 분해됩니다.
- **교차언어 전이.** UD 태그와 의존 관계는 언어에 중립적이라, 새 언어에 대한 제로샷 구조화 분석을 가능하게 합니다.
- **저연산 파이프라인.** 트랜스포머를 배포할 수 없다면, POS + 의존성 파스 + 가제티어로 놀라울 만큼 멀리 갑니다.

## 산출물 (Ship It)

`outputs/skill-grammar-pipeline.md`로 저장하세요.

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

## 연습 문제 (Exercises)

1. **쉬움.** 작은 태깅 코퍼스(예: NLTK의 Brown 부분집합)에서 최빈 태그 베이스라인을 쓰고, 보류 문장에 대한 정확도를 측정하세요. 약 85% 결과를 확인하세요.
2. **보통.** 위의 바이그램 HMM을 학습하고 태그별 precision/recall을 보고하세요. HMM이 가장 헷갈리는 태그는 무엇인가요?
3. **어려움.** spaCy의 의존성 파스를 써서 1000문장 샘플에서 주어-동사-목적어 트리플을 추출하세요. 수동으로 라벨링한 50개 트리플로 평가하세요. 추출이 실패하는 곳(종종 수동태, 등위 구조, 생략된 주어)을 문서화하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| POS tag | 단어의 유형 | 문법 범주. PTB는 36개, UD는 17개. |
| Penn Treebank | 표준 태그셋 | 영어 특화. 세분화된 동사 시제와 명사 수. |
| Universal Dependencies | 다국어 태그셋 | PTB보다 거칠고, 언어 중립적이며, 교차언어 작업의 기본. |
| Dependency parse | 문장 트리 | 각 단어에 머리(head) 하나, 각 엣지에 문법 관계. |
| Viterbi | 동적 프로그래밍 | 방출과 전이가 주어졌을 때 최고 확률 태그 시퀀스를 찾음. |

## 더 읽어보기 (Further Reading)

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS와 파싱의 표준 교과서 처리.
- [Universal Dependencies project](https://universaldependencies.org/) — 모든 다국어 파서가 쓰는 교차언어 태그셋과 트리뱅크 모음.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token`에 노출된 모든 속성의 실무 참고.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 신경망 파서를 주류로 끌어온 논문.
