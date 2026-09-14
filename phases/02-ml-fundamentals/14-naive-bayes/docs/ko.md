# 나이브 베이즈 (Naive Bayes)

> "나이브"한 가정은 틀렸는데도 잘 됩니다. 그게 아름다움입니다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- 텍스트 분류를 위해 라플라스 스무딩(Laplace smoothing)이 있는 다항 나이브 베이즈(Multinomial Naive Bayes)를 처음부터 구현합니다
- 나이브한 독립성 가정이 수학적으로는 틀렸지만 실무에서 올바른 클래스 순위를 왜 내는지 설명합니다
- Multinomial, Bernoulli, Gaussian 나이브 베이즈 변형을 비교하고 주어진 특성 타입에 맞는 것을 고릅니다
- 고차원 희소 데이터에서 나이브 베이즈를 로지스틱 회귀와 비교 평가하고, 작동하는 편향-분산 트레이드오프를 설명합니다

## 문제 상황 (The Problem)

텍스트를 분류해야 합니다. 이메일을 스팸/비스팸으로. 고객 리뷰를 긍정/부정으로. 지원 티켓을 카테고리로. 특성은 수천 개(단어당 하나)이고 학습 데이터는 제한적입니다.

대부분의 분류기는 여기서 막힙니다. 로지스틱 회귀는 수천 개 가중치를 안정적으로 추정할 샘플이 필요합니다. 결정 나무는 한 번에 한 단어로 분할하고 심하게 과적합합니다. 10,000차원의 KNN은 모든 점이 서로 비슷하게 멀어 의미가 없습니다.

나이브 베이즈는 이걸 처리합니다. 수학적으로 틀린 가정(클래스가 주어지면 모든 특성이 서로 독립)을 하지만, 특히 작은 학습 세트에서 텍스트 분류에 "더 똑똑한" 모델보다 자주 앞섭니다. 데이터를 한 번만 훑어 학습합니다. 수백만 특성으로 확장됩니다. 확률 추정치도 내지만(독립성 가정 때문에 보정이 나쁜 경우가 많음).

틀린 가정이 좋은 예측으로 이어지는 이유를 이해하면 머신러닝의 근본을 배웁니다: 최고의 모델은 가장 올바른 모델이 아니라, 데이터에 대해 편향-분산 트레이드오프가 가장 좋은 모델입니다.

## 핵심 개념 (The Concept)

### 베이즈 정리 빠른 복습 (Bayes' Theorem)

베이즈 정리는 조건부 확률을 뒤집습니다:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

원하는 것은 `P(class | features)` -- 문서의 단어가 주어졌을 때 해당 클래스에 속할 확률입니다. 다음으로부터 계산할 수 있습니다:
- `P(features | class)` -- 이 클래스의 문서에서 이 단어들을 볼 가능도(likelihood)
- `P(class)` -- 클래스의 사전 확률(prior) (일반적으로 스팸이 얼마나 흔한가?)
- `P(features)` -- 증거(evidence). 모든 클래스에 같아 비교할 때 무시할 수 있음

`P(class | features)`가 가장 높은 클래스가 이깁니다.

### 나이브한 독립성 가정 (The Naive Independence Assumption)

`P(features | class)`를 정확히 계산하려면 모든 특성의 결합 확률을 추정해야 합니다. 어휘가 10,000단어이면 2^10,000 가지 조합에 대한 분포를 추정해야 합니다. 불가능합니다.

나이브 가정: 클래스가 주어지면 모든 특성은 조건부 독립입니다.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

불가능한 하나의 결합 분포 대신, 특성당 n개의 단순한 분포를 추정합니다. 각각은 카운트만 필요합니다.

이 가정은 분명히 틀렸습니다. "machine"과 "learning"은 어떤 문서에서도 독립이 아닙니다. 하지만 분류기는 정확한 확률 추정치가 필요하지 않습니다. 올바른 순위 -- 어느 클래스의 확률이 가장 높은지 -- 가 필요합니다. 독립성 가정은 체계적 오차를 넣지만, 그 오차가 모든 클래스에 비슷하게 영향을 주어 순위는 맞게 유지됩니다.

### 그래도 잘 되는 이유 (Why It Still Works)

세 가지 이유:

1. **보정(calibration)보다 순위(ranking).** 분류는 최상위 클래스만 맞으면 됩니다. 진짜 확률이 0.7인데 P(spam) = 0.99999여도 분류기는 스팸을 올바르게 고릅니다. 정확한 확률이 아니라 올바른 승자가 필요합니다.

2. **높은 편향, 낮은 분산.** 독립성 가정은 강한 사전입니다. 모델을 강하게 제약해 과적합을 막습니다. 학습 데이터가 제한적일 때, 약간 틀렸지만 안정적인 모델이 이론적으로 맞지만 심하게 불안정한 모델을 이깁니다. 이것이 편향-분산 트레이드오프입니다.

3. **특성 중복이 상쇄됨.** 상관된 특성은 중복 증거를 제공합니다. 분류기는 이 증거를 이중 계산하지만, 올바른 클래스에 대해서도 이중 계산합니다. "machine"과 "learning"이 항상 함께 나타나면 둘 다 "tech" 클래스의 증거입니다. NB는 두 번 세지만, 올바른 클래스에 대해 두 번 셉니다.

네 번째 실용적 이유: 나이브 베이즈는 매우 빠릅니다. 학습은 빈도를 세며 데이터를 한 번 훑는 것입니다. 예측은 행렬 곱셈입니다. 백만 문서를 초 단위로 학습할 수 있습니다. 이 속도 덕분에 더 빠르게 반복하고, 더 많은 특성 세트를 시도하며, 느린 모델보다 더 많은 실험을 돌릴 수 있습니다.

### 수학을 단계별로 (The Math Step by Step)

구체적 예시를 따라가 봅시다. 클래스는 스팸과 비스팸 두 개. 어휘는 "free", "money", "meeting" 세 단어.

학습 데이터:
- 스팸 이메일: "free" 80회, "money" 60회, "meeting" 10회 (총 150단어)
- 비스팸 이메일: "free" 5회, "money" 10회, "meeting" 100회 (총 115단어)
- 이메일의 40%가 스팸, 60%가 비스팸

라플라스 스무딩(alpha=1)으로:

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

새 이메일 내용: "free" (2회), "money" (1회), "meeting" (0회).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

스팸이 큰 차이로 이깁니다. "free"가 두 번 나온 것이 스팸의 강한 증거입니다. "meeting"이 나타나지 않으면 양쪽 log 합에 0을 더합니다(0 * log(P)) -- Multinomial NB에서 없는 단어는 효과가 없습니다. 단어 부재를 명시적으로 모델링하는 것은 Bernoulli NB입니다.

### 세 가지 변형 (Three Variants)

나이브 베이즈는 세 가지 맛이 있습니다. 각각 `P(feature | class)`를 다르게 모델링합니다.

#### Multinomial Naive Bayes

각 특성을 카운트로 모델링합니다. 특성이 단어 빈도나 TF-IDF 값인 텍스트 데이터에 가장 적합합니다.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha`는 라플라스 스무딩입니다(아래에서 설명). 이 변형이 텍스트 분류의 주력입니다.

#### Gaussian Naive Bayes

각 특성을 정규 분포로 모델링합니다. 연속형 특성에 가장 적합합니다.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

클래스마다 특성별 평균과 분산을 가집니다. 클래스 안에서 특성이 실제로 종 모양을 따를 때 잘 작동합니다.

#### Bernoulli Naive Bayes

각 특성을 이진(존재 또는 부재)으로 모델링합니다. 짧은 텍스트나 이진 특성 벡터에 가장 적합합니다.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Multinomial과 달리 Bernoulli는 단어 부재를 명시적으로 페널티합니다. "free"가 보통 스팸에 나타나는데 이 이메일에 없으면, Bernoulli는 그걸 스팸에 반하는 증거로 셉니다.

### 각 변형을 언제 쓸까 (When to Use Each Variant)

| 변형 | 특성 타입 | 가장 적합한 경우 | 예시 |
|---------|-------------|----------|---------|
| Multinomial | 카운트 또는 빈도 | 텍스트 분류, bag-of-words | 이메일 스팸, 토픽 분류 |
| Gaussian | 연속값 | 정규에 가까운 표 형식 데이터 | Iris 분류, 센서 데이터 |
| Bernoulli | 이진 (0/1) | 짧은 텍스트, 이진 특성 벡터 | SMS 스팸, 존재/부재 특성 |

### 라플라스 스무딩 (Laplace Smoothing)

테스트 데이터에 어떤 단어가 나타났는데, 특정 클래스의 학습 데이터에는 한 번도 안 나왔다면?

스무딩 없이: `P(word | class) = 0/N = 0`. 하나의 0이 전체 곱을 통해 곱해지면, 다른 모든 증거와 무관하게 `P(class | features) = 0`이 됩니다. 미지 단어 하나가 전체 예측을 망칩니다.

라플라스 스무딩은 모든 특성 카운트에 작은 카운트 `alpha`(보통 1)를 더합니다:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

alpha=1이면 모든 단어가 최소한 아주 작은 확률을 갖습니다. 테스트 이메일에 "discombobulate"가 나와도 스팸 확률을 죽이지 않습니다. 스무딩에는 베이즈적 해석이 있습니다: 단어 분포에 균일 Dirichlet 사전을 두는 것과 같습니다.

alpha가 높을수록 스무딩이 강합니다(더 균일한 분포). alpha가 낮을수록 모델이 데이터를 더 믿습니다. Alpha는 튜닝하는 하이퍼파라미터입니다.

alpha의 효과:

| Alpha | 효과 | 언제 쓸까 |
|-------|--------|-------------|
| 0.001 | 거의 스무딩 없음, 데이터를 믿음 | 매우 큰 학습 세트, 미지 특성 예상 없음 |
| 0.1 | 가벼운 스무딩 | 큰 학습 세트 |
| 1.0 | 표준 라플라스 스무딩 | 기본 시작점 |
| 10.0 | 강한 스무딩, 분포를 평평하게 | 매우 작은 학습 세트, 미지 특성 많이 예상 |

### 로그 공간 계산 (Log-Space Computation)

각각 1보다 작은 확률 수백 개를 곱하면 부동소수점 언더플로가 납니다. 진짜 값은 아주 작은 양수인데 곱이 부동소수점에서는 0이 됩니다.

해결: 로그 공간에서 작업합니다. 확률을 곱하는 대신 로그를 더합니다:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

예측이 내적으로 바뀝니다:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

행렬 곱셈입니다. 나이브 베이즈 예측이 이렇게 빠른 이유입니다 -- 단층 선형 모델과 같은 연산입니다.

### 나이브 베이즈 vs 로지스틱 회귀 (Naive Bayes vs Logistic Regression)

둘 다 텍스트용 선형 분류기입니다. 차이는 무엇을 모델링하느냐입니다.

| 측면 | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| 유형 | 생성(generative) (P(X\|Y) 모델링) | 판별(discriminative) (P(Y\|X) 모델링) |
| 학습 | 빈도 카운트 | 손실 함수 최적화 |
| 작은 데이터 | 더 나음 (강한 사전이 도움) | 더 나쁨 (가중치 추정에 데이터 부족) |
| 큰 데이터 | 더 나쁨 (틀린 가정이 해로움) | 더 나음 (유연한 경계) |
| 특성 | 독립성 가정 | 상관을 처리 |
| 속도 | 한 번 통과, 매우 빠름 | 반복 최적화 |
| 보정 | 나쁜 확률 | 더 나은 확률 |

경험 법칙: 나이브 베이즈로 시작하세요. 데이터가 충분하고 NB가 정체되면 로지스틱 회귀로 바꾸세요.

### 분류 파이프라인 (Classification Pipeline)

```mermaid
flowchart LR
    A[원시 텍스트] --> B[토큰화]
    B --> C[어휘 구축]
    C --> D[단어 빈도 카운트]
    D --> E[스무딩 적용]
    E --> F[로그 확률 계산]
    F --> G[예측: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

실무에서는 부동소수점 언더플로를 피하려고 로그 공간에서 작업합니다. 작은 확률을 많이 곱하는 대신 로그를 더합니다:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 직접 구현하기 (Build It)

`code/naive_bayes.py`의 코드는 MultinomialNB와 GaussianNB를 처음부터 구현합니다.

### MultinomialNB

처음부터의 구현:

1. **fit(X, y)**: 클래스마다 각 특성의 빈도를 셉니다. 라플라스 스무딩을 더합니다. 로그 확률을 계산합니다. 클래스 사전(클래스 빈도의 로그)을 저장합니다.

2. **predict_log_proba(X)**: 샘플마다 모든 클래스에 대해 log P(class) + log P(feature_i | class)의 합을 계산합니다. 이는 행렬 곱셈입니다: X @ log_probs.T + log_priors.

3. **predict(X)**: 로그 확률이 가장 높은 클래스를 반환합니다.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

핵심 통찰: 적합 후 예측은 행렬 곱셈에 편향을 더한 것뿐입니다. 나이브 베이즈가 이렇게 빠른 이유입니다.

### GaussianNB

연속형 특성에서는 클래스·특성마다 평균과 분산을 추정합니다:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

예측은 특성마다 가우시안 PDF를 쓰고, 특성들에 걸쳐 곱합니다(로그 공간에서는 더함).

### 데모: 텍스트 분류 (Text Classification)

코드는 두 클래스(기술 기사 vs 스포츠 기사)를 시뮬레이션하는 합성 bag-of-words 데이터를 생성합니다. 클래스마다 단어 빈도 분포가 다릅니다. MultinomialNB가 단어 카운트로 분류합니다.

합성 데이터는 이렇게 동작합니다: "단어" 200개(특성 열)를 만듭니다. 단어 0-39는 기술 기사에서 빈도가 높고 스포츠에서는 낮습니다. 단어 80-119는 스포츠에서 높고 기술에서는 낮습니다. 단어 40-79는 둘 다에서 중간 빈도입니다. 일부 단어는 강한 클래스 지표이고 나머지는 노이즈인 현실적인 시나리오입니다.

### 데모: 연속형 특성 (Continuous Features)

코드는 Iris와 비슷한 데이터(3클래스, 4특성, 가우시안 클러스터)를 생성합니다. GaussianNB가 클래스별 평균과 분산으로 분류합니다. 클래스마다 중심(평균 벡터)과 퍼짐(분산)이 달라, 측정값이 범주마다 체계적으로 다른 실제 데이터를 흉내 냅니다.

코드가 또한 보여주는 것:
- **스무딩 비교:** 서로 다른 alpha로 MultinomialNB를 학습해 스무딩 강도가 정확도에 미치는 효과를 보입니다.
- **학습 크기 실험:** 학습 데이터가 20에서 1600 샘플로 늘 때 NB 정확도가 어떻게 좋아지는지. NB는 아주 적은 샘플로도 괜찮은 정확도에 도달합니다 -- 이것이 주요 장점입니다.
- **혼동 행렬:** 클래스별 precision, recall, F1로 NB가 어디서 틀리는지 보입니다.

### 예측 속도 (Prediction Speed)

나이브 베이즈 예측은 행렬 곱셈입니다. 특성이 d개, 클래스가 k개인 샘플 n개에 대해:
- MultinomialNB: 한 번의 행렬 곱 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k번의 가우시안 PDF 평가, 각각 d특성 = O(n * d * k)

둘 다 모든 차원에서 선형입니다. 모든 학습 점과의 거리 계산이 필요한 KNN이나, 모든 서포트 벡터에 대한 커널 평가가 필요한 RBF 커널 SVM과 비교하세요. NB는 예측 시점에 자릿수 단위로 더 빠릅니다.

## 라이브러리로 쓰기 (Use It)

sklearn에서는 두 변형 모두 한 줄입니다:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

sklearn으로 텍스트 분류:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py`의 코드는 같은 데이터에서 처음부터 구현과 sklearn을 비교해 정확성을 검증합니다.

### TF-IDF와 나이브 베이즈 (TF-IDF with Naive Bayes)

원시 단어 카운트는 출현마다 모든 단어에 같은 가중치를 줍니다. 하지만 "the", "is" 같은 흔한 단어는 모든 클래스에 자주 나타나 정보가 없습니다. TF-IDF(Term Frequency - Inverse Document Frequency)는 흔한 단어의 가중치를 낮추고 드물고 변별력 있는 단어의 가중치를 올립니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 값은 음이 아니므로 MultinomialNB와 함께 쓸 수 있습니다. TF-IDF + MultinomialNB 조합은 텍스트 분류의 가장 강한 베이스라인 중 하나입니다. 학습 샘플이 10,000개 미만인 데이터셋에서 더 복잡한 모델을 자주 이깁니다.

### 짧은 텍스트용 BernoulliNB (BernoulliNB for Short Text)

짧은 텍스트(트윗, SMS, 채팅)에서는 BernoulliNB가 MultinomialNB를 이길 수 있습니다. 짧은 텍스트는 단어 카운트가 낮아 MultinomialNB가 의존하는 빈도 정보가 노이즈입니다. BernoulliNB는 존재/부재만 보므로 짧은 텍스트에서 더 신뢰할 수 있습니다.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer의 `binary=True` 플래그는 모든 카운트를 0/1로 바꿉니다. 없어도 BernoulliNB는 동작하지만, 설계되지 않은 카운트를 보게 됩니다.

### NB 확률 보정 (Calibrating NB Probabilities)

NB 확률은 보정이 나쁩니다. NB가 P(spam) = 0.95라고 해도 진짜 확률은 0.7일 수 있습니다. 신뢰할 수 있는 확률 추정치가 필요하면(예: 임계값 설정이나 다른 모델과 결합), sklearn의 CalibratedClassifierCV를 쓰세요:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

교차 검증으로 NB의 원시 점수 위에 로지스틱 회귀를 적합합니다. 결과 확률은 진짜 클래스 빈도에 훨씬 가깝습니다.

### 흔한 함정 (Common Gotchas)

1. **음수 특성 값.** MultinomialNB는 음이 아닌 특성이 필요합니다. 음수 값(특정 설정의 TF-IDF나 표준화된 특성)이 있으면 GaussianNB를 쓰거나 특성을 양수로 이동하세요.

2. **분산이 0인 특성.** GaussianNB는 분산으로 나눕니다. 어떤 클래스에서 특성이 분산 0(모든 값이 동일)이면 확률 계산이 깨집니다. 코드는 모든 분산에 작은 스무딩 항(1e-9)을 더해 이를 막습니다.

3. **클래스 불균형.** 이메일의 99%가 비스팸이면 사전 P(not-spam) = 0.99가 너무 강해 가능도 증거를 압도합니다. 클래스 사전을 수동으로 설정하거나 sklearn의 class_prior 파라미터를 쓸 수 있습니다.

4. **특성 스케일링.** MultinomialNB는 스케일링이 필요 없습니다(카운트에서 동작). GaussianNB도 필요 없습니다(특성별 통계량을 추정). 특성 스케일에 민감한 로지스틱 회귀·SVM 대비 장점입니다.

## 산출물 (Ship It)

이 레슨이 만드는 것:
- `outputs/skill-naive-bayes-chooser.md` -- 올바른 NB 변형을 고르는 결정 스킬
- `code/naive_bayes.py` -- MultinomialNB와 GaussianNB를 처음부터, sklearn 비교 포함

### 나이브 베이즈가 실패할 때 (When Naive Bayes Fails)

NB는 독립성 가정이 잘못된 확률만이 아니라 잘못된 순위를 만들 때 실패합니다. 이런 경우입니다:

1. **강한 특성 상호작용.** 클래스가 두 특성의 조합에 의존하고 각각만으로는 의존하지 않으면(XOR류 패턴), NB는 완전히 놓칩니다. 각 특성만으로는 증거가 없고, NB는 비선형으로 결합할 수 없습니다.

2. **반대 증거를 가진 고상관 특성.** 특성 A는 "스팸", 특성 B는 "비스팸"이라고 하는데 A와 B가 완전 상관(실제로는 항상 일치)이면, NB는 없는 충돌 증거를 봅니다.

3. **매우 큰 학습 세트.** 데이터가 충분하면 로지스틱 회귀 같은 판별 모델이 진짜 결정 경계를 배워 NB를 이깁니다. 작은 데이터에서 도움이 되던 독립성 가정이 이제 모델을 붙잡습니다.

실무에서 이런 실패 모드는 텍스트 분류에서 드뭅니다. 텍스트 특성은 많고 개별적으로 약하며, 독립성 가정의 오차는 상쇄되는 경향이 있습니다. 강하게 상관된 특성이 적은 표 형식 데이터에서는 로지스틱 회귀나 트리 기반 모델을 먼저 고려하세요.

## 연습 문제 (Exercises)

1. **스무딩 실험.** 텍스트 데이터에서 alpha 0.01, 0.1, 1.0, 10.0, 100.0으로 MultinomialNB를 학습하세요. 정확도 vs alpha를 그리세요. 성능이 어디서 정점을 찍나요? 아주 높은 alpha가 왜 해로운가요?

2. **특성 독립성 테스트.** 실제 텍스트 데이터셋을 가져오세요. 분명히 상관된 두 단어("machine"과 "learning")를 고르세요. P(word1 | class) * P(word2 | class)를 계산하고 P(word1 AND word2 | class)와 비교하세요. 독립성 가정이 얼마나 틀렸나요? 분류 정확도에 영향을 주나요?

3. **Bernoulli 구현.** 코드에 BernoulliNB 클래스를 추가하세요. bag-of-words를 이진(존재/부재)으로 바꾸고 텍스트 데이터에서 MultinomialNB와 정확도를 비교하세요. Bernoulli가 언제 이기나요?

4. **NB vs 로지스틱 회귀.** 텍스트 데이터에서 둘 다 학습하세요. 학습 샘플 100개부터 10,000개까지 늘리세요. 둘의 정확도 vs 학습 세트 크기를 그리세요. 어느 지점에서 로지스틱 회귀가 나이브 베이즈를 앞지르나요?

5. **스팸 필터.** 완전한 스팸 분류기를 만드세요: 원시 이메일 텍스트 토큰화, 어휘 구축, bag-of-words 특성 생성, MultinomialNB 학습, precision과 recall로 평가(정확도만이 아님 -- 왜?).

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| Naive Bayes | "간단한 확률 분류기" | 클래스가 주어지면 특성이 조건부 독립이라는 가정으로 베이즈 정리를 적용하는 분류기 |
| Conditional independence | "특성이 서로 영향을 안 줌" | P(A, B \| C) = P(A \| C) * P(B \| C) -- C를 알면 B를 알아도 A에 새 정보가 없음 |
| Laplace smoothing | "add-one 스무딩" | 0 확률이 예측을 지배하지 않도록 모든 특성에 작은 카운트를 더함 |
| Prior | "데이터를 보기 전 믿음" | P(class) -- 특성을 관측하기 전 각 클래스의 확률 |
| Likelihood | "데이터가 얼마나 맞는지" | P(features \| class) -- 클래스를 알 때 이 특성을 관측할 확률 |
| Posterior | "데이터를 본 뒤 믿음" | P(class \| features) -- 특성을 관측한 뒤 갱신된 클래스 확률 |
| Generative model | "데이터가 어떻게 생성되는지 모델링" | P(X \| Y)와 P(Y)를 학습한 뒤 베이즈 정리로 P(Y \| X)를 구하는 모델 |
| Discriminative model | "결정 경계를 모델링" | X가 어떻게 생성되는지는 모델링하지 않고 P(Y \| X)를 직접 학습하는 모델 |
| Log probability | "언더플로 방지" | 작은 수의 곱이 부동소수점에서 0이 되지 않도록 P 대신 log P로 작업 |

## 더 읽어보기 (Further Reading)

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 세 변형과 수학적 세부사항
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 텍스트용 Multinomial vs Bernoulli의 고전적 비교
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 텍스트용 NB 개선
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- NB가 데이터가 적을 때 LR보다 더 빨리 수렴함을 증명
