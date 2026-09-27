# 트랜스포머 이전의 텍스트 생성 — N-gram 언어 모델 (Text Generation Before Transformers — N-gram Language Models)

> 단어가 놀랍다면(surprising) 모델이 좋지 않은 것입니다. 퍼플렉서티(Perplexity)는 놀라움을 숫자로 나타냅니다. 스무딩(Smoothing)은 이를 유한하게 유지합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제 (The Problem)

트랜스포머(transformers), RNN, 워드 임베딩(word embeddings)이 등장하기 전에는 언어 모델이 이전 `n-1`개의 단어 뒤에 특정 단어가 얼마나 자주 나타나는지 세는 방식으로 다음 단어를 예측했습니다. "the cat" → "sat"가 47번, "the cat" → "jumped"가 12번, "the cat" → "refrigerator"가 0번 나왔다고 카운트하고, 이를 정규화(normalize)하여 확률 분포를 얻습니다.

이것이 바로 n-gram 언어 모델입니다. 이 모델은 1980년부터 2015년까지 모든 음성 인식기, 맞춤법 검사기, 그리고 구문 기반 기계 번역(phrase-based machine translation) 시스템을 구동했습니다. 저비용의 온디바이스(on-device) 언어 모델링이 필요한 경우 오늘날에도 여전히 사용됩니다.

흥미로운 문제는 훈련 데이터에서 보지 못한(unseen) n-gram을 어떻게 처리하느냐입니다. 단순 빈도 기반(raw count-based) 모델은 보지 못한 모든 것에 0의 확률을 할당하는데, 문장은 길고 거의 모든 긴 문장에는 적어도 하나 이상의 보지 못한 시퀀스가 포함되어 있기 때문에 이는 치명적입니다. 50년간의 스무딩(smoothing) 연구가 이 문제를 해결했습니다. 그 결과물이 Kneser-Ney 스무딩이며, 현대의 딥러닝은 그 경험적 전통을 계승했습니다.

## 개념 (The Concept)

![N-gram 모델: 카운트, 스무딩, 생성](../assets/ngram.svg)

### 예측 게임 (The prediction game)

이 모든 메커니즘이 존재하기 전, 하나의 실험이 언어 모델이 무엇인지 정의했습니다. 영어 문장의 다음 글자를 가리세요. 누군가에게 한 번에 하나씩 맞힐 때까지 추측해 보라고 요청하세요. 추측 횟수를 기록하세요. 수백 개의 글자에 대해 이 과정을 반복합니다.

추측 횟수는 단순한 잡학 지식이 아닙니다. 이는 텍스트의 손실 없는 재인코딩(lossless re-encoding)입니다. 이 횟수 시퀀스를 동일한 두 번째 추측자에게 전달하면, 각 위치에서 어떤 추측이 먼저 나오는지 정확히 알 수 있기 때문에 모든 글자를 재구성할 수 있습니다. 더 적은 기호로 재인코딩할 수 있는 메시지는 기호당 정보량이 적으므로, 추측 횟수 통계는 영어의 엔트로피(entropy)에 상한선을 설정합니다.

섀넌(Shannon)은 1951년에 이를 실행하여 오늘날까지 이 분야를 지배하는 수치를 얻었습니다. 27개 기호 알파벳(알파벳 26자와 공백 1자)은 글자당 `log2(27) ≈ 4.75` 비트를 전달할 수 있습니다. 100글자의 문맥을 가진 인간 추측자는 글자당 0.6에서 1.3 비트 사이를 기록했습니다. 영어는 대략 4분의 3이 강제된 움직임(forced moves)입니다. 모델이 학습해야 할 구조는 모델이 학습할 수 있기 훨씬 전부터 이미 측정되었습니다.

이후의 모든 언어 모델은 이 게임의 기계적 플레이어이며, 이 레슨의 모든 평가 수치는 게임의 점수입니다:

- **교차 엔트로피 손실(Cross-entropy loss)**은 모델이 기호당 필요로 하는 평균 비트 수입니다. 언어 모델(LM)을 학습시키는 것은 말 그대로 추측 게임에서의 점수를 최소화하는 것입니다.
- **퍼플렉서티(Perplexity)**는 `2^bits`(또는 `e^nats`)로, 추측을 마친 후 모델이 여전히 직면하게 되는 분기 계수(branching factor)입니다. 27개 기호에 대해 균등하게 추측하면 퍼플렉서티는 27이며, 글자당 1비트를 사용하는 플레이어의 퍼플렉서티는 2입니다.
- **문맥 길이(Context length)는 플레이어의 기억력입니다.** 트라이그램(trigram) 모델은 두 개의 토큰만큼의 기억력을 가지고 게임을 합니다. 트랜스포머는 10만 개의 토큰을 가지고 동일한 게임을 수행합니다. 규칙은 변하지 않았으며, 플레이어가 더 향상되었을 뿐입니다.

주의해야 할 단위 차이: 게임 점수는 글자당 비트(`log2`)로 표시되는 반면, 아래의 n-gram 공식은 단어 토큰당 내트(nats, 자연로그)로 점수를 매깁니다. 퍼플렉서티 `e^H`(nats)는 `2^H`(bits)와 같으므로, 두 관점은 서로 다른 단위를 사용하는 동일한 측정값입니다.

```figure
prediction-game
```

**N-gram 확률(N-gram probability):** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. `n`을 고정합니다(일반적으로 트라이그램은 3, 4-gram은 4). 카운트로부터 계산합니다:

```text
P(w | context) = count(context, w) / count(context)
```

**제로 카운트 문제(The zero-count problem).** 학습 과정에서 보지 못한 n-gram은 확률이 0이 됩니다. Brown 코퍼스에 대한 2007년 연구에 따르면, 4-gram 모델조차 홀드아웃(held-out)된 4-gram의 30%가 학습 데이터에 없었습니다. 스무딩(smoothing) 없이는 실제 텍스트를 평가할 수 없습니다.

**스무딩 접근 방식(Smoothing approaches), 정교함 순서대로:**

1. **라플라스(Laplace, add-one).** 모든 카운트에 1을 더합니다. 단순하지만 희귀한 이벤트에는 매우 취약합니다.
2. **굿-튜링(Good-Turing).** 빈도의 빈도(frequency-of-frequencies)를 기반으로 고빈도 이벤트의 확률 질량을 보지 못한 이벤트로 재할당합니다.
3. **보간법(Interpolation).** n-gram, (n-1)-gram 등의 추정치를 조정 가능한 가중치로 결합합니다.
4. **백오프(Backoff).** n-gram의 카운트가 0이면 (n-1)-gram으로 물러납니다(fallback). Katz 백오프가 이를 정규화합니다.
5. **절대 할인(Absolute discounting).** 모든 카운트에서 고정된 할인 값 `D`를 빼고, 이를 보지 못한 이벤트에 재분배합니다.
6. **크네저-네이(Kneser-Ney).** 절대 할인에 더해 하위 차수 모델을 위한 영리한 선택을 추가합니다. 원시 빈도 대신 *연속 확률(continuation probability)*(단어가 나타나는 서로 다른 문맥의 수)을 사용합니다.

Kneser-Ney의 통찰은 깊이가 있습니다. "San Francisco"는 흔한 바이그램(bigram)입니다. 유니그램(unigram) "Francisco"는 주로 "San" 뒤에 나타납니다. 단순 절대 할인을 적용하면 "Francisco"는 높은 유니그램 확률을 갖게 됩니다(카운트가 높기 때문). Kneser-Ney는 "Francisco"가 오직 한 가지 문맥에서만 나타난다는 점을 포착하고 그에 따라 연속 확률을 낮춥니다. 결과적으로 "Francisco"로 끝나는 새로운 바이그램은 적절하게 낮은 확률을 얻게 됩니다.

**평가: 퍼플렉서티(Evaluation: perplexity).** 홀드아웃 테스트 세트에서 단어당 평균 음의 로그 가능도(negative log-likelihood)의 지수(exponent)입니다. 낮을수록 좋습니다. 퍼플렉서티가 100이라는 것은 모델이 100개의 단어 중에서 균등하게 선택할 때만큼 혼란스러워한다는 것을 의미합니다.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 직접 구현하기 (Build It)

### 1단계: 트라이그램 카운트 (Trigram counts)

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

입력은 토큰화된 문장 리스트입니다. 출력은 n-gram 카운트와 문맥(context) 카운트입니다. `<s>`와 `</s>`는 문장 경계입니다.

### 2단계: 라플라스 스무딩 (Laplace smoothing)

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

모든 카운트에 1을 더합니다. 스무딩을 수행하지만 보지 못한 이벤트에 확률 질량을 과도하게 할당하여 드물게 나타나는 이미 알려진 이벤트에도 손해를 끼칩니다.

### 3단계: Kneser-Ney (바이그램, 보간법) (Kneser-Ney (bigram, interpolated))

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

세 가지 구성 요소가 있습니다. `continuation_prob`는 "이 단어가 얼마나 많은 서로 다른 문맥에서 나타나는가?"를 포착합니다(Kneser-Ney의 핵심 혁신). `lambda_prev`는 할인(discount)을 통해 확보되어 백오프의 가중치를 매기는 데 사용되는 질량입니다. 최종 확률은 할인된 주요 항에 가중치가 적용된 연속 항을 더한 값입니다.

### 4단계: 샘플링을 통한 텍스트 생성 (Generating text with sampling)

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

확률에 비례하여 샘플링합니다. 시드(seed)에 따라 항상 다른 출력을 제공합니다. 빔 서치(beam search)와 유사한 출력을 원한다면 각 단계에서 argmax를 선택(greedy)하고 약간의 무작위성 조절 장치(temperature)를 추가하세요.

### 5단계: 퍼플렉서티 (Perplexity)

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

낮을수록 좋습니다. Brown 코퍼스의 경우 잘 튜닝된 4-gram KN 모델은 약 140의 퍼플렉서티를 기록합니다. 트랜스포머 LM은 동일한 테스트 세트에서 15~30을 기록합니다. 약 10배의 격차가 발생합니다. 이 격차로 인해 연구 분야가 발전하게 되었습니다.

## 활용하기 (Use It)

- **고전적 NLP 교육.** 스무딩, MLE, 퍼플렉서티를 가장 명확하게 배울 수 있는 방법입니다.
- **KenLM.** 프로덕션용 n-gram 라이브러리입니다. 짧은 지연 시간이 중요한 음성 및 기계 번역 시스템에서 리스코어러(rescorer)로 사용됩니다.
- **온디바이스 자동 완성.** 키보드의 트라이그램 모델로 여전히 활용됩니다.
- **베이스라인.** 신경망 LM이 우수하다고 선언하기 전에 항상 n-gram LM 퍼플렉서티를 계산해 보세요. 트랜스포머가 KN을 큰 차이로 능가하지 못한다면 어딘가 문제가 있는 것입니다.

## 완성하기 (Ship It)

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
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn the math.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 연습 문제 (Exercises)

1. **쉬움(Easy).** 1,000개 문장으로 구성된 셰익스피어 코퍼스로 트라이그램 LM을 훈련하세요. 20개의 문장을 생성해 보세요. 국소적으로는 그럴듯하지만 전체적으로는 일관성이 없을 것입니다. 이것이 표준적인 데모입니다.
2. **보통(Medium).** 홀드아웃된 셰익스피어 분할 데이터에 대해 KN 모델의 퍼플렉서티를 구현해 보세요. Laplace 방식과 비교해 보세요. KN이 퍼플렉서티를 30~50% 낮추는 것을 확인할 수 있습니다.
3. **어려움(Hard).** 트라이그램 철자 교정기를 구축해 보세요. 철자가 틀린 단어와 그 문맥이 주어졌을 때 교정 후보를 생성하고 LM 하에서의 문맥 확률에 따라 순위를 매깁니다. 공개된 Birkbeck 철자 코퍼스를 사용하여 평가하세요.

## 핵심 용어 (Key Terms)

| 용어 | 일반적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| N-gram | 단어 시퀀스 | 연속된 `n`개 토큰의 시퀀스. |
| 스무딩 (Smoothing) | 0 방지하기 | 보지 못한 이벤트가 0이 아닌 확률을 얻도록 확률 질량을 재할당하는 기법. |
| 퍼플렉서티 (Perplexity) | LM 품질 지표 | 홀드아웃 데이터에 대한 `exp(-평균 log-prob)`. 낮을수록 우수함. |
| 백오프 (Backoff) | 더 짧은 문맥으로 후퇴 | 트라이그램 카운트가 0이면 바이그램을 사용. Katz 백오프가 이를 공식화함. |
| Kneser-Ney | n-gram을 위한 최적의 스무딩 | 절대 할인 + 하위 차수 모델을 위한 연속 확률. |
| 연속 확률 (Continuation probability) | KN 특화 방식 | 원시 카운트가 아니라 `w`가 나타나는 서로 다른 문맥의 수로 가중치를 부여한 `P(w)`. |
| 텍스트의 엔트로피 (Entropy of text) | 기호당 정보량 | 문맥이 주어졌을 때 다음 기호를 인코딩하는 데 필요한 평균 비트 수. 섀넌(Shannon)의 1951년 추정에 따르면 최대 100자의 문맥을 가진 인쇄된 영어는 0.6~1.3 bits/letter이며, 이는 어떠한 모델도 존재하기 전에 측정되었습니다. |

## 더 읽을거리 (Further Reading)

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — 모든 언어 모델이 여전히 최적화하고 있는 목표를 정의한 추측 게임 실험.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM과 스무딩에 관한 표준 참고 자료.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Ney가 최적의 n-gram 스무딩 기법임을 규명한 논문.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — Kneser-Ney 원저 논문.
- [KenLM](https://kheafield.com/code/kenlm/) — 지연 시간에 민감한 애플리케이션을 위해 2026년에도 여전히 사용되는 빠른 프로덕션용 n-gram LM.
