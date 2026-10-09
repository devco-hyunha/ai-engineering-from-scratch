# 트랜스포머 이전의 텍스트 생성 — N-gram 언어 모델

> 단어가 놀랍다면 모델이 나쁜 것입니다. 퍼플렉시티(Perplexity)는 놀라움을 수치로 만듭니다. 스무딩(Smoothing)은 이를 유한하게 유지합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 01강 (텍스트 처리), 2단계 · 14강 (나이브 베이즈)
**시간:** 약 45분

## 문제점

트랜스포머 이전, RNN 이전, 워드 임베딩(Word Embedding) 이전에는 언어 모델이 이전 `n-1` 단어가 얼마나 자주 뒤따랐는지 세어서 다음 단어를 예측했습니다. "the cat" → "sat"이 47번, "the cat" → "jumped"가 12번, "the cat" → "refrigerator"가 0번出现的했습니다. 정규화하여 확률 분포를 얻습니다.

이것이 n-gram 언어 모델입니다. 1980년부터 2015년까지 모든 음성 인식기, 모든 철자 검사기, 모든 구문 기반 기계 번역 시스템이 이를 실행했습니다. 값싼 온디바이스 언어 모델링이 필요할 때에도 여전히 실행됩니다.

흥미로운 문제는 본 적 없는 n-gram을 어떻게 처리하느냐입니다. 원시적인 카운트 기반 모델은 본 적 없는 것에 대해 확률을 0으로 할당하는데, 이는 문장이 길고 거의 모든 긴 문장에 본 적 없는 시퀀스가 적어도 하나 포함되기 때문에 치명적입니다. 50년간의 스무딩 연구가 이를 해결했습니다. Kneser-Ney 스무딩이 그 결과이며, 현대 딥러닝은 그 경험적 전통을 계승했습니다.

## 개념

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### 예측 게임

이 모든 기계가 존재하기 전에, 한 실험이 언어 모델이 무엇인지 정의했습니다. 영어 문장의 다음 글자를 가리고, 한 번에 하나씩 정답을 맞출 때까지 추측하게 하세요. 추측 횟수를 기록하세요. 몇백 개의 글자에 대해 반복하세요.

추측 횟수는 잡담이 아닙니다. 이는 텍스트의 무손실 재인코딩입니다: 추측 횟수 시퀀스를 두 번째 동일한 추측자에게 넘겨주면 모든 글자를 복원할 수 있습니다. 각 위치에서 어떤 추측이 먼저 오는지 정확히 알기 때문입니다. 더 적은 기호로 재인코딩할 수 있는 메시지는 기호당 정보가 적으므로, 추측 횟수 통계는 영어의 엔트로피에 상한을 설정합니다.

Shannon은 1951년에 이 실험을 수행하여 오늘날까지 이 분야를 지배하는 수치를 도출했습니다. 27개 문자 알파벳(26개 문자와 공백)은 문자당 `log2(27) ≈ 4.75` 비트를 전달할 수 있습니다. 100자 문맥을 가진 인간 추론자는 문자당 0.6에서 1.3 비트 사이의 값을 기록했습니다. 영어는 약 4분의 3이 강제된 이동입니다. 모델이 학습해야 할 구조는 모델이 학습하기 전에 측정되었습니다.

이후의 모든 언어 모델은 이 게임의 기계적 플레이어이며, 이 강의의 모든 평가 수치는 게임 점수입니다:

- **교차 엔트로피 손실(Cross-Entropy)**은 모델이 문자당 필요로 하는 평균 비트 수입니다. LM를 학습하는 것은 문자 그대로 추론 게임에서의 점수를 최소화하는 것입니다.
- **퍼플렉시티(Perplexity)**는 `2^bits` (또는 `e^nats`)입니다: 모델의 추론 후에도 남아 있는 분기 계수입니다. 27개 문자에 대한 균일한 추론은 퍼플렉시티 27입니다. 문자당 1비트 플레이어는 퍼플렉시티 2입니다.
- **컨텍스트 윈도우는 플레이어의 기억입니다.** 트라이그램 모델은 2개 토큰의 기억으로 플레이합니다. 트랜스포머는 100K 토큰으로 같은 게임을 플레이합니다. 규칙은 변하지 않았습니다. 플레이어는 더 잘하게 되었습니다.

추적해야 할 단위 전환이 하나 있습니다. 게임은 문자당 비트(`log2`)로 점수를 매기지만, 아래 n-gram 공식은 단어 토큰당 nats(자연 로그)로 점수를 매깁니다. nats에서의 퍼플렉시티 `e^H`는 bits에서의 `2^H`와 같으므로, 두 관점은 서로 다른 단위로 측정된 동일한 측정값입니다.

```figure
prediction-game
```

**N-gram 확률:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. `n`를 고정합니다(트라이그램은 보통 3, 4-gram은 4). 카운트로부터 계산합니다:

```text
P(w | context) = count(context, w) / count(context)
```

**0 카운트 문제.** 학습에서 보지 못한 모든 n-gram은 확률이 0입니다. 2007년 Brown 코퍼스에 대한 연구에 따르면, 4-gram 모델조차도 홀드아웃된 4-gram의 30%가 학습에서 보지 못한 것이었습니다. 스무딩(smoothing) 없이 실제 텍스트를 평가할 수 없습니다.

**스무딩 접근법, 정교한 순서대로:**

1. **Laplace (add-one).** 모든 카운트에 1을 더합니다. 간단하지만 희귀한 사건에는 끔찍합니다.
2. **Good-Turing.** 빈도-빈도(frequency-of-frequencies)에 기반하여 고빈도 사건에서 확률 질량을 재분배하여 보지 못한 사건에 할당합니다.
3. **보간법(Interpolation).** n-gram, (n-1)-gram 등의 추정치를 조정 가능한 가중치로 결합합니다.
4. **백오프(Backoff).** n-gram의 카운트가 0이면 (n-1)-gram으로 폴백합니다. Katz 백오프는 이를 정규화합니다.
5. **절대 할인.** 모든 개수에서 고정 할인 `D`를 빼고, 미관찰 항목에 재분배합니다.
6. **Kneser-Ney.** 절대 할인에 하위 차수 모델에 대한 영리한 선택을 더합니다: *연속 확률*(단어가 나타나는 컨텍스트 수)을 원시 빈도 대신 사용합니다.

Kneser-Ney의 통찰은 깊습니다. "San Francisco"는 흔한 바이그램입니다. 유니그램 "Francisco"는 대부분 "San" 뒤에 나타납니다. 단순한 절대 할인은 "Francisco"에 높은 유니그램 확률을 부여합니다(개수가 높기 때문). Kneser-Ney는 "Francisco"가 단 하나의 컨텍스트에서만 나타난다는 것을 인식하고, 이에 따라 연속 확률을 낮춥니다. 결과: "Francisco"로 끝나는 새로운 바이그램은 적절한 낮은 확률을 얻습니다.

**평가: 퍼플렉시티.** 홀드아웃 테스트 세트에서 단어당 평균 음의 로그 우도 지수입니다. 낮을수록 좋습니다. 퍼플렉시티가 100이라는 것은 모델이 100개 단어 중 균일하게 선택하는 것과 똑같이 혼란스럽다는 의미입니다.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 구현하기

### 1단계: 트라이그램 개수

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

입력은 토큰화된 문장 목록입니다. 출력은 n-gram 개수와 컨텍스트 개수입니다. `<s>`와 `</s>`는 문장 경계입니다.

### 2단계: 라플라스 스무딩

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

모든 개수에 1을 더합니다. 스무딩을 적용하지만, 미관찰 이벤트에 질량을 과다 배분하여 희귀한 알려진 이벤트에도 해를 끼칩니다.

### 3단계: Kneser-Ney (바이그램, 보간)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

세 가지 이동 부분. `continuation_prob`는 "이 단어가 몇 개의 서로 다른 컨텍스트에 나타나는가?"를 포착합니다(Kneser-Ney의 혁신). `lambda_prev`은 할인으로 해제된 질량으로, 백오프 가중치에 사용됩니다. 최종 확률은 할인된 주요 항에 가중치 적용된 연속 항을 더한 값입니다.

### 4단계: 샘플링으로 텍스트 생성

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

확률에 비례하여 샘플링합니다. 시드마다 항상 다른 출력을 제공합니다. 빔 검색과 유사한 출력을 원한다면, 각 단계에서 argmax를 선택(탐욕적)하고 작은 무작위성 조절 장치(온도)를 추가합니다.

### 5단계: 퍼플렉시티

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

낮을수록 좋습니다. Brown 코퍼스의 경우, 잘 튜닝된 4-gram KN 모델은 퍼플렉시티가 약 140입니다. 같은 테스트 세트에서 트랜스포머 LM은 15-30을 기록합니다. 격차는 약 10배입니다. 이 격차 때문에 분야가 발전했습니다.

## 사용하기

- **고전적 NLP 교육.** 스무딩, MLE, 퍼플렉시티(Perplexity)를 가장 명확하게 경험할 수 있습니다.
- **KenLM.** 프로덕션 n-gram 라이브러리. 낮은 지연이 중요한 음성 및 MT 시스템에서 리스코어(rescorer)로 사용됩니다.
- **온디바이스 자동완성.** 키보드의 트라이그램 모델. 지금도 사용되고 있습니다.
- **베이스라인.** 신경망 LM가 좋다고 선언하기 전에 항상 n-gram LM의 퍼플렉시티를 계산하세요. 트랜스포머가 KN(Kneser-Ney)을 큰 차이로 이기지 못한다면, 무언가 잘못된 것입니다.

## 출시하기

`outputs/prompt-lm-baseline.md`로 저장하세요:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 연습 문제

1. **쉬움.** 1,000문장 셰익스피어 코퍼스로 트라이그램 LM를 학습하세요. 20문장을 생성하세요. 지역적으로 타당하지만 전역적으로는 비일관적일 것입니다. 이는 표준적인 데모입니다.
2. **중간.** 홀드아웃(held-out) 셰익스피어 분할에 대해 KN 모델의 퍼플렉시티를 구현하세요. 라플라스(Laplace)와 비교하세요. KN이 퍼플렉시티를 30-50% 낮추는 것을 확인해야 합니다.
3. **어려움.** 트라이그램 철자 교정기를 구축하세요: 오타가 있는 단어와 그 문맥이 주어지면, LM 하의 문맥 확률로 교정 후보를 생성하고 순위를 매기세요. 공개된 Birkbeck 철자 코퍼스로 평가하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| N-gram | 단어 순서 | `n`개의 연속된 토큰 순서. |
| 스무딩 | 0 피하기 | 확률 질량을 재배분하여 보지 못한 이벤트에 비-0 확률을 부여하는 것. |
| 퍼플렉시티 | LM 품질 지표 | 홀드아웃 데이터에서의 `exp(-average log-prob)`. 낮을수록 좋습니다. |
| 백오프 | 더 짧은 문맥으로 폴백 | 트라이그램 카운트가 0이면, 바이그램을 사용하세요. Katz 백오프는 이를 형식화합니다. |
| Kneser-Ney | n-gram에 대한 최상의 스무딩 | 절대 할인 + 하위 차수 모델에 대한 연속 확률. |
| 연속 확률 | KN 특유 | `P(w)`는 `w`이 나타나는 문맥 수로 가중치가 매겨지며, 원시 카운트로 매겨지지 않습니다. |
| 텍스트 엔트로피 | 심볼당 정보량 | 문맥이 주어졌을 때 다음 심볼을 인코딩하는 데 필요한 평균 비트 수. 100자까지의 문맥을 가진 인쇄된 영어에 대한 Shannon의 1951년 추정치: 문자당 0.6-1.3 비트, 모델이 존재하기 전에 측정되었습니다. |

## 추가 읽기

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — 모든 언어 모델이 최적화하는 목표를 정의한 추측 게임 실험입니다.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM과 스무딩에 대한 표준적인 처리 방식입니다.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Ney가 최상의 n-gram 스무더임을 확립한 논문입니다.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — KN의 원 논문입니다.
- [KenLM](https://kheafield.com/code/kenlm/) — 빠른 프로덕션 n-gram LM으로, 2026년에도 지연에 민감한 애플리케이션에서 여전히 사용되고 있습니다.
