# 감정 분석

> NLP의 표준적인 작업입니다. 고전적인 텍스트 분류에 대해 알아야 할 대부분의 내용이 여기에 나타납니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 2단계 · 14강 (Naive Bayes)
**시간:** 약 75분

## 문제점

"음식이 그다지 좋지 않았습니다." 긍정일까요, 부정일까요?

감정 분석은 단순해 보입니다. 리뷰어가 무언가를 좋아했거나 좋아하지 않았다고 말했습니다. 문장에 라벨을 붙이면 됩니다. 이 작업이 NLP의 표준적인 작업이 된 이유는, 단순해 보이는 모든 사례가 어려운 사례를 숨기고 있기 때문입니다. 부정어가 의미를 뒤집습니다. 아이러니는 의미를 반전시킵니다. "전혀 나쁘지 않다"는 두 개의 부정적인 단어가 있음에도 긍정적입니다. 이모지는 주변 텍스트보다 더 많은 신호를 전달합니다. 도메인별 어휘가 중요합니다 (`tight`은 음악 리뷰에서, `tight`은 패션 리뷰에서).

감정 분석은 고전 NLP를 위한 실험실입니다. 모든 단순한 기준 모델이 특정 실패 양상을 보이는 이유를 이해하면, 더 복잡한 모델이 왜 발명되었는지도 이해하게 됩니다. 이 강의에서는 Naive Bayes 기준 모델을 처음부터 구축하고, 로지스틱 회귀를 추가하며, 프로덕션 환경의 감정 분석을 컴플라이언스 수준의 문제로 만드는 함정들을 짚어 봅니다.

## 개념

고전적인 감정 분석은 두 단계의 레시피입니다.

1. **표현.** 텍스트를 특징 벡터로 변환합니다. BoW, TF-IDF, 또는 n-gram을 사용합니다.
2. **분류.** 라벨이 붙은 예제에 선형 모델(Naive Bayes, 로지스틱 회귀, SVM)을 적용합니다.

Naive Bayes는 작동하는 가장 단순한 모델입니다. 라벨이 주어졌을 때 모든 특징이 독립적이라고 가정합니다. `P(word | positive)`과 `P(word | negative)`을 카운트로부터 추정합니다. 추론 시 확률을 곱합니다. "순진한" 독립성 가정은 터무니없이 틀렸지만, 결과는 놀라울 정도로 강력합니다. 그 이유는 희소한 텍스트 특징과 중간 규모의 데이터에서는 분류기가 각 단어가 어느 쪽으로 기울어지는지보다 얼마나 기울어지는지에 더 집중하기 때문입니다.

로지스틱 회귀는 독립성 가정을 수정합니다. 음의 가중치를 포함하여 각 특징에 대한 가중치를 학습합니다. `not good`은 bigram 특징으로 음의 가중치를 받습니다. Naive Bayes는 라벨이 붙지 않은 bigram에 대해 그렇게 할 수 없습니다.

```figure
sentiment-logits
```

## 구현하기

### 1단계: 실제 미니 데이터셋

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

의도적으로 작게 설계했습니다. 실제 작업에서는 수만 개의 예제(IMDb, SST-2, Yelp polarity)를 사용합니다. 수학적 원리는 동일합니다.

### 2단계: 멀티노미얼 나이브 베이즈(Naive Bayes)를 처음부터 구현하기

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

가산 스무딩(additive smoothing, alpha=1.0)은 라플라스 스무딩(Laplace smoothing)입니다. 이 스무딩이 없으면 특정 클래스에서 한 번도 등장하지 않은 단어의 확률이 0이 되어 로그 값이 폭발합니다. `alpha=0.01`은 실무에서 자주 사용됩니다. `alpha=1.0`은 교육용 기본값입니다.

### 3단계: 로지스틱 회귀(logistic regression)를 처음부터 구현하기

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

여기서는 L2 정규화(L2 regularization)가 중요합니다. 텍스트 특징은 희소(sparse)하므로, L2 정규화가 없으면 모델은 학습 예제를 암기하게 됩니다. `0.01`에서 시작하여 튜닝해 보세요.

### 4단계: 부정어 처리 (실패 모드)

"not good"과 "not bad"를 고려해 보세요. BoW 분류기는 `{not, good}`과 `{not, bad}`을 보고, 학습 데이터에서 더 많이 등장한 쪽을 학습합니다. 빅램(bigram) 분류기는 `not_good`과 `not_bad`을 보고 이를 서로 다른 특징으로 학습합니다. 보통은 이것으로 충분합니다.

빅램을 사용하지 않을 때 작동하는 더 단순한 수정 방법은 **부정어 범위 지정(negation scoping)**입니다. 부정어 뒤에 오는 토큰에 `NOT_`을 접두어로 붙여 다음 문장 부호까지 적용합니다.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

이제 `good`과 `NOT_good`은 서로 다른 특징이 됩니다. 분류기는 이 두 특징에 반대되는 가중치를 부여할 수 있습니다. 전처리 세 줄로 감정 분석 벤치마크에서 측정 가능한 정확도 향상 효과를 얻을 수 있습니다.

### 5단계: 중요한 평가 지표

클래스가 불균형한 경우 정확도만으로는 오해를 불러일으킬 수 있습니다. 실제 감정 분석 코퍼스는 보통 70-80%가 긍정적이거나 70-80%가 부정적입니다. 다수 클래스를 항상 예측하는 분류기는 80%의 정확도를 얻지만, 쓸모가 없습니다. 다음 지표들을 모두 보고하세요:

- **클래스별 정밀도 및 재현율(Precision & Recall).** 클래스마다 한 쌍씩 계산합니다. 클래스 균형을 반영하는 단일 값을 얻기 위해 매크로 평균(macro-average)을 사용하세요.
- **매크로-F1 (불균형 데이터의 주요 지표).** 클래스별 F1 점수의 평균이며, 모든 클래스에 동일한 가중치를 부여합니다. 클래스가 불균형할 때는 정확도 대신 이 지표를 사용하세요.
- **가중치-F1 (Weighted-F1, 대안).** 매크로-F01강 동일하지만 클래스 빈도에 따라 가중치를 부여합니다. 불균형 자체에 비즈니스적 의미가 있을 때 매크로-F01강 함께 보고하세요.
- **혼동 행렬(Confusion matrix).** 원시 개수(raw counts)를 나타냅니다. 스칼라 지표를 신뢰하기 전에 항상 검토하세요. 모델이 어떤 클래스 쌍을 혼동하는지 드러냅니다.
- **클래스별 오류 샘플.** 클래스당 잘못된 예측을 5개 추출하세요. 이를 읽어 보세요. 실제 오류를 읽는 것보다 더 좋은 방법은 없습니다.

심각한 불균형 데이터(95:5 비율 이상)에서는 정확도 대신 **AUROC**와 **AUPRC**를 보고하세요. AUPRC는 소수 클래스에 더 민감하며, 이는 보통 스팸, 사기, 희귀 감정 등 당신이 관심 있는 부분입니다.

**피해야 할 흔한 버그.** 불균형 데이터에서 매크로-F1 대신 마이크로-F1을 보고하면 다수 클래스에 의해 지배되어 높은 값이 나오게 됩니다. 매크로-F1은 소수 클래스의 성능을 확인하도록 강제합니다.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 사용하기

scikit-learn은 이를 6줄로 정확하게 처리합니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

세 가지에 주목하세요. `stop_words=None`은 부정어(negation)를 유지합니다. `ngram_range=(1, 2)`은 바이그램(bigram)을 추가하여 `not_good`가 특징(feature)이 됩니다. `sublinear_tf=True`은 반복된 단어를 감쇠합니다. 이 세 가지 플래그는 SST-2에서 75% 정확도 기준선과 85% 정확도 기준선 사이의 차이입니다.

### 트랜스포머를 사용할 때

- 아이러니(sarcasm) 감지. 고전적 모델은 여기서 실패합니다. 끝.
- 문서 중간에 감정이 변하는 긴 리뷰.
- 측면 기반(sentiment) 감정 분석. "카메라는 훌륭했지만 배터리는 형편없었습니다." 측면(aspect)에 감정을 귀속해야 합니다. 트랜스포머나 구조화된 출력 모델만 가능합니다.
- 비영어, 저자원 언어. Multilingual BERT는 제로샷 기준선을 무료로 제공합니다.

위 사항 중 하나가 필요하다면 7단계(트랜스포머 심층 분석)로 건너뛰세요. 그렇지 않다면, TF-IDF에 바이그램과 부정어 처리를 더한 Naive Bayes나 로지스틱 회귀가 2026년 생산 기준선입니다.

### 재현성 함정 (다시)

감정 모델의 재학습은 일상적입니다. 재평가하는 것은 그렇지 않습니다. 논문에서 보고된 정확도 값은 특정 분할, 특정 전처리, 특정 토크나이저를 사용합니다. 동일한 파이프라인을 사용하지 않고 새 모델을 기준선과 비교하면 오해의 소지가 있는 차이(deltas)가 발생합니다. 항상 당신의 파이프라인에서 기준선을 재생성하세요, 논문의 값이 아니라.

## 출시하기

`outputs/prompt-sentiment-baseline.md`로 저장하세요:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 연습 문제

1. **쉬움.** scikit-learn 파이프라인의 전처리 단계로 `apply_negation`을 추가하고, 작은 감정 데이터셋에서 F1 차이(delta)를 측정하세요.
2. **중간 난이도.** 클래스 가중치 로지스틱 회귀를 구현하세요 (scikit-learn에 `class_weight="balanced"`를 전달하거나, 기울기를 직접 유도해 보세요). 합성 데이터의 90-10 클래스 불균형에 대한 영향을 측정하세요.
3. **어려운 난이도.** 감정 모델의 잔차(residual)에 두 번째 분류기를 학습하여 아이러니(sarcasm) 감지기를 구축하세요. 실험 설정을 문서화하세요. 정확도가 우연 수준(chance level)보다 낮을 때 독자에게 경고하세요 (2클래스 아이러니 감지의 우연 수준은 약 50%이며, 대부분의 첫 시도는 이 수준에 머무릅니다).

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 극성(Polarity) | 긍정 또는 부정 | 이진 레이블; 때로는 중립이나 세분화된 등급(5-스타)으로 확장되기도 합니다. |
| 측면 기반 감정 분석(Aspect-based sentiment) | 측면별 극성 | 텍스트에 언급된 특정 엔티티나 속성에 감성을 할당합니다. |
| 부정어 범위 지정(Negation scoping) | 근처 토큰의 극성 반전 | "not" 이후의 접두어 토큰에 `NOT_`를 붙여 문장 부호까지 적용합니다. |
| 라플라스 스무딩(Laplace smoothing) | 카운트에 1을 더하기 | 나이브 베이즈(Naive Bayes)에서 확률이 0인 특성을 방지합니다. |
| L2 정규화(L2 regularization) | 가중치 축소 | 손실 함수에 `lambda * sum(w^2)`를 더합니다. 희소한 텍스트 특성에 필수적입니다. |

## 추가 읽기

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 기초적인 서베이입니다. 길지만, 첫 네 섹션은 고전적인 모든 내용을 다룹니다.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 바이그램(bigram)과 나이브 베이즈(Naive Bayes)가 짧은 텍스트에서 이기기 어렵다는 것을 보여준 논문입니다.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`, `TfidfVectorizer` 및 튜닝할 모든 매개변수에 대한 참고 자료입니다.
