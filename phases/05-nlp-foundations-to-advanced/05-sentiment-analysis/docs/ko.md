# 감성 분석 (Sentiment Analysis)

> 대표적인(canonical) NLP 과제입니다. 고전적 텍스트 분류에 대해 알아야 할 대부분의 내용이 여기에 나옵니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## 문제 (The Problem)

"The food was not great." 긍정일까요, 부정일까요?

감성은 단순해 보입니다. 리뷰어가 무언가를 좋아했는지 싫어했는지 말하고, 문장에 라벨을 붙이면 됩니다. 이것이 대표적인 NLP 과제가 된 이유는, 쉬워 보이는 사례마다 어려운 사례가 숨겨져 있기 때문입니다. 부정(negation)은 의미를 뒤집습니다. 풍자(sarcasm)는 의미를 반전합니다. "Not bad at all"은 부정적으로 코딩된 단어가 둘인데도 긍정입니다. 이모지는 주변 텍스트보다 신호가 더 큽니다. 도메인 어휘가 중요합니다(음악 리뷰의 `tight` 대 패션 리뷰의 `tight`).

감성은 고전 NLP의 실습 실험실입니다. 모든 나이브 베이스라인이 왜 특정 실패 모드를 갖는지 이해하면, 더 풍부한 모델이 왜 발명되었는지도 이해합니다. 이 레슨은 Naive Bayes 베이스라인을 처음부터 만들고, 로지스틱 회귀를 더하며, 프로덕션 감성을 컴플라이언스급 문제로 만드는 함정들에 이름을 붙입니다.

## 개념 (The Concept)

고전적 감성은 두 단계 레시피입니다.

1. **표현(Represent).** 텍스트를 특징 벡터로 바꿉니다. BoW, TF-IDF, 또는 n-그램.
2. **분류(Classify).** 라벨이 있는 예제에 선형 모델(Naive Bayes, 로지스틱 회귀, SVM)을 맞춥니다.

Naive Bayes는 동작하는 가장 단순한 모델입니다. 라벨이 주어졌을 때 모든 특징이 독립이라고 가정합니다. 카운트에서 `P(word | positive)`와 `P(word | negative)`를 추정합니다. 추론 시 확률을 곱합니다. "나이브" 독립 가정은 우스울 정도로 틀렸지만, 결과는 놀랍도록 강합니다. 이유: 희소한 텍스트 특징과 적당한 데이터에서는, 분류기가 정확한 결합 확률보다 각 단어가 어느 쪽으로 기울었는지를 더 신경 씁니다.

로지스틱 회귀는 독립 가정을 고칩니다. 특징마다 가중치를 학습하며, 음수 가중치도 포함합니다. 바이그램 특징으로서 `not good`은 음수 가중치를 얻습니다. Naive Bayes는 한 번도 라벨링하지 않은 바이그램에 대해 그렇게 할 수 없습니다.

```figure
sentiment-logits
```

## 직접 만들기 (Build It)

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

의도적으로 작습니다. 실제 작업은 수만 개의 예제(IMDb, SST-2, Yelp polarity)를 씁니다. 수학은 동일합니다.

### 2단계: 처음부터 만드는 다항 Naive Bayes

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

가산 스무딩(alpha=1.0)은 Laplace 스무딩입니다. 없으면 어떤 클래스에서 본 적 없는 단어의 확률이 0이 되어 로그가 폭발합니다. 실무에서는 `alpha=0.01`이 흔합니다. `alpha=1.0`은 교육용 기본값입니다.

### 3단계: 처음부터 만드는 로지스틱 회귀

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

여기서 L2 정규화가 중요합니다. 텍스트 특징은 희소하고, L2 없으면 모델이 학습 예제를 암기합니다. `0.01`에서 시작해 튜닝하세요.

### 4단계: 부정 처리 (실패 모드)

"not good"과 "not bad"를 생각해 보세요. BoW 분류기는 `{not, good}`과 `{not, bad}`를 보고, 학습에서 더 많이 나온 쪽에 맞춰 학습합니다. 바이그램 분류기는 `not_good`과 `not_bad`를 보고 별개 특징으로 학습합니다. 보통 그것으로 충분합니다.

바이그램이 없을 때 통하는 더 거친 수정: **부정 범위(negation scoping)**. 부정어 다음 토큰에, 다음 구두점까지 `NOT_` 접두사를 붙입니다.

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

이제 `good`과 `NOT_good`은 다른 특징입니다. 분류기가 반대 방향으로 가중치를 줄 수 있습니다. 전처리 세 줄로, 감성 벤치마크에서 측정 가능한 정확도 상승이 납니다.

### 5단계: 중요한 평가 지표

클래스가 불균형이면 정확도만으로는 오해를 부릅니다. 실제 감성 코퍼스는 보통 70–80%가 긍정이거나 70–80%가 부정입니다. 다수 클래스만 항상 예측하는 분류기가 정확도 80%를 얻고도 쓸모없습니다. 아래를 모두 보고하세요.

- **클래스별 정밀도(precision)와 재현율(recall).** 클래스마다 한 쌍. 매크로 평균해 클래스 균형을 존중하는 단일 숫자를 얻습니다.
- **매크로-F1 (불균형 데이터의 주 지표).** 클래스별 F1의 평균, 동일 가중. 클래스가 불균형할 때 정확도 대신 이것을 쓰세요.
- **가중-F1 (대안).** 매크로와 같지만 클래스 빈도로 가중. 불균형 자체가 비즈니스 의미가 있을 때 매크로-F1과 함께 보고하세요.
- **혼동 행렬(confusion matrix).** 원시 카운트. 스칼라 지표를 믿기 전에 항상 확인하세요. 모델이 어느 클래스 쌍을 혼동하는지 드러납니다.
- **클래스별 오류 샘플.** 클래스마다 틀린 예측 5개를 뽑고 읽으세요. 실제 오류를 읽는 것을 대체할 수 없습니다.

심하게 불균형한 데이터(> 95-5 비율)에서는 정확도 대신 **AUROC**와 **AUPRC**를 보고하세요. AUPRC는 소수 클래스에 더 민감하며, 보통 그게 관심사입니다(스팸, 사기, 희귀 감성).

**피할 흔한 버그.** 불균형 데이터에서 매크로-F1 대신 마이크로-F1을 보고하면, 다수 클래스가 지배해 높아 보이는 숫자가 나옵니다. 매크로-F1은 소수 클래스 성능을 보게 강제합니다.

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

## 활용하기 (Use It)

scikit-learn은 여섯 줄로, 올바르게 합니다.

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

주목할 점 세 가지. `stop_words=None`은 부정어를 유지합니다. `ngram_range=(1, 2)`는 바이그램을 더해 `not_good`이 특징이 되게 합니다. `sublinear_tf=True`는 반복 단어를 완화합니다. 이 세 플래그가 SST-2에서 75% 정확도 베이스라인과 85% 정확도 베이스라인의 차이입니다.

### 언제 트랜스포머로 갈까

- 풍자 탐지. 고전 모델은 여기서 실패합니다. 끝.
- 문서 중간에 감성이 바뀌는 긴 리뷰.
- 측면 기반 감성(aspect-based sentiment). "Camera was great but battery was terrible." 측면에 감성을 귀속해야 합니다. 트랜스포머 또는 구조화 출력 모델만 해당합니다.
- 비영어, 저자원 언어. Multilingual BERT가 무료로 제로샷 베이스라인을 줍니다.

위가 필요하면 phase 7(transformers deep dive)로 건너뛰세요. 그렇지 않으면 TF-IDF + 바이그램 + 부정 처리 위의 Naive Bayes 또는 로지스틱 회귀가 2026년 프로덕션 베이스라인입니다.

### 재현성 함정 (또 다시)

감성 모델을 재학습하는 것은 일상입니다. 재평가하는 것은 아닙니다. 논문에 보고된 정확도 숫자는 특정 분할, 특정 전처리, 특정 토크나이저를 씁니다. 동일 파이프라인 없이 새 모델을 베이스라인과 비교하면 오해의 델타가 나옵니다. 항상 논문 숫자가 아니라 자신의 파이프라인에서 베이스라인을 다시 생성하세요.

## 산출물 (Ship It)

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

## 연습 문제 (Exercises)

1. **쉬움.** scikit-learn 파이프라인에 전처리 단계로 `apply_negation`을 넣고, 작은 감성 데이터셋에서 F1 델타를 측정하세요.
2. **보통.** 클래스 가중 로지스틱 회귀를 구현하세요(scikit-learn에 `class_weight="balanced"`를 넘기거나, 직접 그래디언트를 유도). 합성 90-10 클래스 불균형에서 효과를 측정하세요.
3. **어려움.** 감성 모델의 잔차(residuals)로 두 번째 분류기를 학습해 풍자 탐지기를 만드세요. 실험 설정을 문서화하세요. 정확도가 우연 수준 아래일 때 독자에게 경고하세요(2클래스 풍자에서 우연 수준은 ~50%이고, 대부분의 첫 시도가 거기에 착지합니다).

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|----------|
| 극성(Polarity) | 긍정 또는 부정 | 이진 라벨. 때로 중립이나 세분(5점 척도)으로 확장. |
| 측면 기반 감성(Aspect-based sentiment) | 측면별 극성 | 텍스트에 언급된 특정 엔티티나 속성에 감성을 귀속. |
| 부정 범위(Negation scoping) | 근처 토큰 반전 | "not" 다음 토큰에 구두점까지 `NOT_` 접두사. |
| Laplace 스무딩(Laplace smoothing) | 카운트에 1 더하기 | Naive Bayes에서 영확률 특징을 막음. |
| L2 정규화(L2 regularization) | 가중치 축소 | 손실에 `lambda * sum(w^2)`를 더함. 희소 텍스트 특징에 필수. |

## 더 읽기 (Further Reading)

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 기초 서베이. 길지만 앞의 네 섹션이 고전적 내용을 모두 다룹니다.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 짧은 텍스트에서 바이그램 + Naive Bayes가 이기기 어렵다는 것을 보여 준 논문.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`, `TfidfVectorizer`, 튜닝할 모든 노브의 참고 문서.
