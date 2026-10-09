# 나이브 베이즈

> "나이브" 가정이 틀렸음에도 작동합니다. 이것이 바로 그 매력입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 2단계, 01-07강 (분류, 베이즈 정리)
**시간:** 약 75분

## 학습 목표

- 텍스트 분류를 위해 라플라스 스무딩을 적용한 다항 나이브 베이즈를 처음부터 구현해 보세요
- 나이브 독립 가정이 수학적으로 틀렸음에도 실무에서 올바른 클래스 순위를 산출하는 이유를 설명해 보세요
- 다항, 베르누이, 가우시안 나이브 베이즈 변형을 비교하고 주어진 특징 유형에 적합한 것을 선택해 보세요
- 고차원 희소 데이터에서 나이브 베이즈를 로지스틱 회귀와 비교하고, 작동하는 편향-분산 트레이드오프를 설명해 보세요

## 문제점

텍스트를 분류해야 합니다. 이메일을 스팸 또는 스팸이 아닌 것으로 분류하고, 고객 리뷰를 긍정적 또는 부정적으로 분류하며, 지원 티켓을 범주로 분류해야 합니다. 수천 개의 특징(단어당 하나)이 있고 훈련 데이터는 제한적입니다.

대부분의 분류기는 여기서 막혀버립니다. 로지스틱 회귀는 수천 개의 가중치를 신뢰할 수 있게 추정하기 위해 충분한 샘플이 필요합니다. 결정 트리는 한 번에 하나의 단어로 분할하며 극심한 과적합을 일으킵니다. 10,000 차원의 KNN은 모든 점이 서로 동일한 거리로 떨어져 있어 의미가 없습니다.

나이브 베이즈는 이를 처리합니다. 수학적으로 틀린 가정(클래스가 주어졌을 때 모든 특징이 서로 독립적이라는 가정)을 내세우지만, 텍스트 분류, 특히 작은 훈련 세트에서는 "더 똑똑한" 모델보다 성능이 뛰어납니다. 데이터를 한 번만 훑어 훈련합니다. 수백만 개의 특징으로 확장됩니다. 확률 추정치를 산출합니다(독립 가정 때문에 보정이 잘 안 되는 경우가 많지만).

틀린 가정이 좋은 예측으로 이어지는 이유를 이해하면 머신러닝의 근본적인 원리를 깨닫게 됩니다. 최선의 모델은 가장 정확한 모델이 아니라, 데이터에 대해 편향-분산 트레이드오프가 가장 좋은 모델입니다.

## 개념

### 베이즈 정리 (빠른 복습)

베이즈 정리는 조건부 확률을 뒤집습니다:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

우리는 `P(class | features)`를 원합니다. 이는 문서에 포함된 단어를 고려할 때, 해당 문서가 특정 클래스에 속할 확률입니다. 이 값은 다음을 통해 계산할 수 있습니다:
- `P(features | class)` -- 이 클래스의 문서에서 이러한 단어를 볼 가능성
- `P(class)` -- 클래스의 사전 확률 (스팸이 일반적으로 얼마나 흔한가요?)
- `P(features)` -- 증거(evidence)로, 모든 클래스에 동일하므로 비교 시 무시할 수 있습니다

`P(class | features)`가 가장 높은 클래스가 승리합니다.

### 소박한 독립성 가정

`P(features | class)`를 정확히 계산하려면 모든 특징을 함께 고려한 결합 확률을 추정해야 합니다. 어휘가 10,000 단어라면 2^10,000개의 가능한 조합에 대한 분포를 추정해야 합니다. 불가능합니다.

소박한 가정: 모든 특징은 클래스가 주어졌을 때 조건부 독립입니다.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

하나의 불가능한 결합 분포 대신, n개의 단순한 특징별 분포를 추정합니다. 각각은 단지 개수(count)만 필요합니다.

이 가정은 명백히 틀렸습니다. "machine"과 "learning"이라는 단어는 어떤 문서에서도 독립적이지 않습니다. 하지만 분류기는 정확한 확률 추정치가 필요하지 않습니다. 정확한 순위(ranking)가 필요합니다. 즉, 어떤 클래스가 가장 높은 확률을 가지는지 알면 됩니다. 독립성 가정은 체계적인 오차를 도입하지만, 그 오차는 모든 클래스에 유사하게 영향을 미치므로 순위는 정확히 유지됩니다.

### 왜 여전히 효과가 있는가

세 가지 이유:

1. **순위 vs. 보정.** 분류는 상위 순위 클래스가 정확하기만 하면 됩니다. P(spam) = 0.99999이고 실제 확률이 0.7인 경우에도, 분류기는 스팸을 올바르게 선택합니다. 우리는 정확한 확률이 필요하지 않습니다. 올바른 승자(winner)가 필요합니다.

2. **높은 편향, 낮은 분산.** 독립성 가정은 강한 사전(prior)입니다. 이는 모델을 강하게 제약하여 과적합을 방지합니다. 제한된 학습 데이터에서는 약간 틀리지만 안정적인 모델이, 이론적으로 정확하지만 극도로 불안정한 모델보다 더 잘 작동합니다. 이는 편향-분산 트레이드오프(bias-variance tradeoff)가 작동하는 방식입니다.

3. **특징 중복은 상쇄됩니다.** 상관된 특징은 중복된 증거를 제공합니다. 분류기는 이 증거를 이중으로 계산하지만, 올바른 클래스에 대해서도 이중으로 계산합니다. "machine"과 "learning"이 항상 함께 나타나면, 둘 다 "tech" 클래스에 대한 증거를 제공합니다. NB는 이들을 두 번 계산하지만, 올바른 클래스에 대해서도 두 번 계산합니다.

네 번째, 실용적인 이유: Naive Bayes는 매우 빠릅니다. 학습은 데이터의 빈도를 세는 단일 패스입니다. 예측은 행렬 곱셈입니다. 백만 개의 문서를 몇 초 안에 학습할 수 있습니다. 이 속도 덕분에 더 빠른 반복, 더 많은 특징 세트 시도, 더 많은 실험을 느린 모델보다 많이 수행할 수 있습니다.

### 수식을 단계별로 따라가 보기

구체적인 예시를 따라가 보겠습니다. 두 클래스가 있다고 가정합니다: spam과 not-spam. 어휘는 세 단어로 구성됩니다: "free", "money", "meeting".

학습 데이터:
- Spam 이메일은 "free"를 80번, "money"를 60번, "meeting"을 10번 언급합니다 (총 150 단어)
- Not-spam 이메일은 "free"를 5번, "money"를 10번, "meeting"을 100번 언급합니다 (총 115 단어)
- 이메일의 40%는 spam이고, 60%는 not-spam입니다

Laplace smoothing (alpha=1)을 적용하면:

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

새 이메일에는 "free" (2번), "money" (1번), "meeting" (0번)이 포함되어 있습니다.

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam이 큰 차이로 승리합니다. "free"가 두 번 나타나는 것은 spam에 대한 강력한 증거입니다. "meeting"이 나타나지 않는 것은 두 로그 합에 모두 0을 기여합니다 (0 * log(P)) -- Multinomial NB에서는 없는 단어는 효과가 없습니다. 단어의 부재를 명시적으로 모델링하는 것은 Bernoulli NB입니다.

### 세 가지 변형

Naive Bayes는 세 가지 종류가 있습니다. 각각 `P(feature | class)`을 다르게 모델링합니다.

#### Multinomial Naive Bayes

각 특징을 개수로 모델링합니다. 특징이 단어 빈도나 TF-IDF 값인 텍스트 데이터에 가장 적합합니다.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha`은 Laplace smoothing입니다 (아래 설명 참조). 이 변형은 텍스트 분류의 주력입니다.

#### Gaussian Naive Bayes

각 특징을 정규 분포로 모델링합니다. 연속적인 특징에 가장 적합합니다.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

각 클래스는 특징별로 자체 평균과 분산을 가집니다. 특징이 각 클래스 내에서 진정으로 종 모양 곡선을 따를 때 잘 작동합니다.

#### 베르누이 나이브 베이즈

각 특징을 이진 값(존재 또는 부재)으로 모델링합니다. 짧은 텍스트나 이진 특징 벡터에 가장 적합합니다.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

멀티노미얼과 달리, 베르누이는 단어의 부재를 명시적으로 페널티로 계산합니다. "free"가 스팸에는 일반적으로 나타나지만 이 이메일에는 없다면, 베르누이는 이를 스팸에 대한 반증으로 간주합니다.

### 각 변형의 사용 시점

| 변형 | 특징 유형 | 최적 용도 | 예시 |
|---------|-------------|----------|---------|
| 멀티노미얼 | 개수 또는 빈도 | 텍스트 분류, 단어 주머니(bag-of-words) | 이메일 스팸, 주제 분류 |
| 가우시안 | 연속 값 | 정규 분포에 가까운 특징을 가진 표 데이터 | Iris 분류, 센서 데이터 |
| 베르누이 | 이진 (0/1) | 짧은 텍스트, 이진 특징 벡터 | SMS 스팸, 존재/부재 특징 |

### 라플라스 스무딩

테스트 데이터에出现的 단어가 특정 클래스의 훈련 데이터에는 한 번도 등장하지 않은 경우 어떻게 될까요?

스무딩이 없으면: `P(word | class) = 0/N = 0`. 하나의 0이 전체 곱셈에 포함되면 `P(class | features) = 0`이 되며, 다른 모든 증거는 무시됩니다. 하나의 unseen 단어가 전체 예측을 파괴합니다. 다른 증거가 얼마나 많이 이를 지지하든 상관없습니다.

라플라스 스무딩은 모든 특징 개수에 작은 개수 `alpha` (보통 1)를 더합니다:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

alpha=1이면 모든 단어가 최소한 아주 작은 확률을 가집니다. 테스트 이메일에 "discombobulate"가 나타나도 스팸 확률을 죽이지 않습니다. 이 스무딩은 베이지안 해석을 가집니다: 단어 분포에 균일한 디리클레 사전 분포를 놓는 것과 동일합니다.

alpha가 높으면 스무딩이 강해집니다(분포가 더 균일해짐). alpha가 낮으면 모델이 데이터를 더 신뢰합니다. Alpha는 당신이 튜닝하는 하이퍼파라미터입니다.

alpha의 효과:

| Alpha | 효과 | 사용 시점 |
|-------|--------|-------------|
| 0.001 | 거의 스무딩 없음, 데이터 신뢰 | 매우 큰 훈련 세트, unseen 특징이 예상되지 않음 |
| 0.1 | 가벼운 스무딩 | 큰 훈련 세트 |
| 1.0 | 표준 라플라스 스무딩 | 기본 시작점 |
| 10.0 | 강한 스무딩, 분포를 평평하게 만듦 | 매우 작은 훈련 세트, 많은 unseen 특징이 예상됨 |

### 로그 공간 연산

수백 개의 확률(각각 1보다 작은 값)을 곱하면 부동 소수점 언더플로가 발생합니다. 실제 값은 매우 작은 양수임에도 불구하고, 부동 소수점에서는 곱의 결과가 0이 됩니다.

해결 방법: 로그 공간에서 연산합니다. 확률을 곱하는 대신 로그를 더합니다:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

이렇게 하면 예측이 내적(dot product)이 됩니다:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

행렬 곱셈입니다. 이것이 나이브 베이즈 예측이 매우 빠른 이유입니다. 단일층 선형 모델과 동일한 연산이기 때문입니다.

### 나이브 베이즈 vs 로지스틱 회귀

둘 다 텍스트를 위한 선형 분류기입니다. 차이점은 무엇을 모델링하느냐에 있습니다.

| 측면 | 나이브 베이즈 | 로지스틱 회귀 |
|--------|------------|-------------------|
| 유형 | 생성형 (P(X\|Y) 모델링) | 판별형 (P(Y\|X) 모델링) |
| 학습 | 빈도 계산 | 손실 함수 최적화 |
| 작은 데이터 | 더 좋음 (강한 사전 확률이 도움) | 더 나쁨 (가중치를 추정하기에는 데이터가 부족) |
| 큰 데이터 | 더 나쁨 (잘못된 가정이 해로움) | 더 좋음 (유연한 경계) |
| 특징 | 독립성 가정 | 상관관계 처리 |
| 속도 | 단일 패스, 매우 빠름 | 반복적 최적화 |
| 보정 | 확률이 부정확함 | 확률이 더 정확함 |

경험칙: 나이브 베이즈로 시작하세요. 충분한 데이터가 있고 NB가 평탄화(plateau)되면 로지스틱 회귀로 전환하세요.

### 분류 파이프라인

```mermaid
flowchart LR
    A["원시 텍스트"] --> B["Tokenize"]
    B --> C["어휘 구축"]
    C --> D["단어 빈도 계산"]
    D --> E["스무딩 적용"]
    E --> F["로그 확률 계산"]
    F --> G["예측: 단어가 주어졌을 때 argmax P 클래스"]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

실제로는 부동 소수점 언더플로를 피하기 위해 로그 공간에서 연산합니다. 많은 작은 확률을 곱하는 대신, 그 로그를 더합니다:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 구현하기

`code/naive_bayes.py`의 코드는 MultinomialNB와 GaussianNB를 처음부터(scratch) 구현합니다.

### MultinomialNB

처음부터(scratch) 구현한 내용:

1. **fit(X, y)**: 각 클래스에 대해 각 특징의 빈도를 계산합니다. 라플라스 스무딩을 추가합니다. 로그 확률을 계산합니다. 클래스 사전 확률(클래스 빈도의 로그)을 저장합니다.

2. **predict_log_proba(X)**: 각 샘플에 대해 모든 클래스에 대해 log P(class) + 모든 기능에 대한 log P(feature_i | class)의 합을 계산합니다. 이는 행렬 곱셈입니다: X @ log_probs.T + log_priors.

3. **predict(X)**: 가장 높은 로그 확률을 가진 클래스를 반환합니다.

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

핵심 통찰: 피팅 후 예측은 행렬 곱셈과 편향(bias)만 추가하는 것입니다. 이것이 Naive Bayes가 매우 빠른 이유입니다.

### GaussianNB

연속 기능의 경우, 각 클래스와 각 기능에 대해 평균과 분산을 추정합니다:

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

예측은 각 기능에 대해 가우시안 PDF를 사용하며, 기능 전체에 걸쳐 곱해집니다 (로그 공간에서는 더해집니다).

### 데모: 텍스트 분류

코드는 두 클래스(기술 기사 vs 스포츠 기사)를 시뮬레이션하는 합성 bag-of-words 데이터를 생성합니다. 각 클래스는 다른 단어 빈도 분포를 가집니다. MultinomialNB는 단어 수를 사용하여 분류합니다.

합성 데이터는 다음과 같이 작동합니다: 200개의 "단어"(기능 열)를 생성합니다. 단어 0-39는 기술 기사에서 높은 빈도를, 스포츠 기사에서 낮은 빈도를 가집니다. 단어 80-119는 스포츠 기사에서 높은 빈도를, 기술 기사에서 낮은 빈도를 가집니다. 단어 40-79는 두 기사 모두에서 중간 빈도를 가집니다. 이는 일부 단어는 강력한 클래스 지표이고 나머지는 잡음인 현실적인 시나리오를 만듭니다.

### 데모: 연속 기능

코드는 Iris 유사 데이터(3개 클래스, 4개 기능, 가우시안 클러스터)를 생성합니다. GaussianNB는 클래스별 평균과 분산을 사용하여 분류합니다. 각 클래스는 다른 중심(평균 벡터)과 다른 퍼짐(분산)을 가지며, 범주 간에 측정값이 체계적으로 다른 실제 데이터를 모방합니다.

코드는 또한 다음을 시연합니다:
- **스무딩 비교:** MultinomialNB를 다른 alpha 값으로 훈련하여 스무딩 강도가 정확도에 미치는 영향을 보여줍니다.
- **훈련 크기 실험:** 훈련 데이터가 20에서 1600 샘플로 증가함에 따라 NB 정확도가 향상되는 방식. NB는 매우 적은 샘플로도 decent한 정확도에 도달합니다 -- 이것이 NB의 주요 장점입니다.
- **혼동 행렬:** NB가 실수하는 부분을 보여주기 위한 클래스별 정밀도, 재현율 및 F1 점수.

### 예측 속도

Naive Bayes 예측은 행렬 곱셈입니다. d개의 기능과 k개의 클래스를 가진 n개의 샘플에 대해:
- MultinomialNB: 하나의 행렬 곱 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k개의 가우시안 PDF 평가, 각각 d개의 특징에 대해 = O(n * d * k)

두 모델 모두 모든 차원에서 선형입니다. 이는 KNN(모든 학습 포인트에 대한 거리 계산이 필요함)이나 RBF 커널 SVM(모든 서포트 벡터에 대한 커널 평가가 필요함)과 비교해 보세요. NB는 예측 시간에서 몇 자릿수(order of magnitude)만큼 더 빠릅니다.

## 사용하기

sklearn을 사용하면 두 변형 모두 한 줄 코드로 구현할 수 있습니다:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

sklearn을 사용한 텍스트 분류의 경우:

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

`naive_bayes.py`의 코드는 같은 데이터에서 처음부터 구현한 것과 sklearn을 비교하여 정확성을 검증합니다.

### TF-IDF와 Naive Bayes

원시 단어 개수는 모든 단어가 출현할 때마다 동일한 가중치를 부여합니다. 하지만 "the"나 "is" 같은 흔한 단어는 모든 클래스에서 자주 나타나며, 정보를 담고 있지 않습니다. TF-IDF (Term Frequency - Inverse Document Frequency)는 흔한 단어의 가중치를 낮추고 희소하고 판별적인 단어의 가중치를 높입니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 값은 음수가 아니므로 MultinomialNB와 함께 사용할 수 있습니다. TF-IDF + MultinomialNB 조합은 텍스트 분류에서 가장 강력한 기준선(baseline) 중 하나입니다. 학습 샘플이 10,000개 미만인 데이터셋에서는 더 복잡한 모델을 자주 능가합니다.

### 짧은 텍스트를 위한 BernoulliNB

짧은 텍스트(트윗, SMS, 채팅 메시지)의 경우 BernoulliNB가 MultinomialNB보다 성능이 더 좋을 수 있습니다. 짧은 텍스트는 단어 개수가 적어 MultinomialNB가 의존하는 빈도 정보가 잡음이 많습니다. BernoulliNB는 존재 여부에만 집중하므로 짧은 텍스트에서 더 신뢰할 수 있습니다.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer의 `binary=True` 플래그는 모든 개수를 0/1로 변환합니다. 이 플래그가 없으면 BernoulliNB는 여전히 작동하지만, 설계되지 않은 개수를 보게 됩니다.

### NB 확률 보정하기

NB 확률은 보정이 잘 되어 있지 않습니다. NB가 P(spam) = 0.95라고 말할 때, 실제 확률은 0.7일 수 있습니다. 신뢰할 수 있는 확률 추정치가 필요한 경우(예: 임계값 설정이나 다른 모델과 결합할 때) sklearn의 CalibratedClassifierCV를 사용하세요:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

이는 교차 검증을 사용하여 NB의 원시 점수 위에 로지스틱 회귀를 적합합니다. resulting probabilities는 실제 클래스 빈도에 훨씬 더 가깝습니다.

### 흔한 함정

1. **음수 특성 값.** MultinomialNB는 비음수 특성을 요구합니다. 음수 값이 있는 경우 (예: 특정 설정의 TF-IDF나 표준화된 특성), GaussianNB를 사용하거나 특성을 양수로 이동하세요.

2. **분산이 0인 특성.** GaussianNB는 분산으로 나눕니다. 특정 클래스에 대해 특성의 분산이 0인 경우 (모든 값이 동일), 확률 계산이 깨집니다. 이 코드는 이를 방지하기 위해 모든 분산에 작은 스무딩 항 (1e-9)을 더합니다.

3. **클래스 불균형.** 이메일의 99%가 스팸이 아닌 경우, 사전 확률 P(not-spam) = 0.99가 매우 강해서 우도 증거를 압도합니다. 클래스 사전 확률을 수동으로 설정하거나 sklearn의 class_prior 매개변수를 사용할 수 있습니다.

4. **특성 스케일링.** MultinomialNB는 스케일링이 필요하지 않습니다 (카운트 기반). GaussianNB도 스케일링이 필요하지 않습니다 (특성별 통계를 추정). 이는 특성 스케일에 민감한 로지스틱 회귀 및 SVM에 대한 장점입니다.

## 출시하기

이 강의는 다음을 생성합니다:
- `outputs/skill-naive-bayes-chooser.md` -- 올바른 NB 변형을 선택하는 결정 스킬
- `code/naive_bayes.py` -- MultinomialNB와 GaussianNB를 처음부터 구현하고, sklearn과 비교

### Naive Bayes가 실패하는 경우

NB는 독립성 가정이 잘못된 순위를 초래할 때 실패합니다 (단순히 확률이 틀린 것이 아님). 이는 다음 상황에서 발생합니다:

1. **강한 특성 상호작용.** 클래스가 두 특성의 조합에 의존하지만 각각의 특성만으로는 의존하지 않는 경우 (XOR와 유사한 패턴), NB는 이를 완전히 놓칩니다. 각 특성만으로는 증거가 제공되지 않으며, NB는 이를 비선형적으로 결합할 수 없습니다.

2. **상반된 증거를 가진 고도로 상관된 특성.** 특성 A가 "스팸"이라고 말하고 특성 B가 "스팸이 아님"이라고 말하지만, A와 B가 완벽하게 상관되어 있다면 (현실에서 항상 일치), NB는 실제로 존재하지 않는 상반된 증거를 보게 됩니다.

3. **매우 큰 훈련 세트.** 충분한 데이터가 있으면 로지스틱 회귀와 같은 판별 모델이 실제 결정 경계를 학습하여 NB보다 성능이 뛰어납니다. 작은 데이터에서는 도움이 되었던 독립성 가정이 이제 모델의 성능을 저해합니다.

실제로 텍스트 분류에서는 이러한 실패 모드가 드뭅니다. 텍스트 특징은 많고, 각각은 약하며, 독립성 가정의 오류는 서로 상쇄되는 경향이 있습니다. 강하게 상관된 특징이 몇 개뿐인 표 데이터의 경우, 로지스틱 회귀나 트리 기반 모델을 먼저 고려해 보세요.

## 연습 문제

1. **스무딩 실험.** 텍스트 데이터에 대해 alpha 값이 0.01, 0.1, 1.0, 10.0, 100.0인 MultinomialNB를 학습하세요. alpha에 따른 정확도를 그래프로 그려 보세요. 성능이 최고조에 달하는 지점은 어디인가요? alpha가 매우 높으면 성능이 왜 떨어지나요?

2. **특징 독립성 테스트.** 실제 텍스트 데이터셋을 가져오세요. 명백히 상관된 두 단어("machine"과 "learning")를 선택하세요. P(word1 | class) * P(word2 | class)를 계산하고 P(word1 AND word2 | class)와 비교하세요. 독립성 가정이 얼마나 틀렸나요? 분류 정확도에 영향을 미치나요?

3. **Bernoulli 구현.** BernoulliNB 클래스를 추가하여 코드를 확장하세요. bag-of-words를 binary (present/absent)로 변환하고 텍스트 데이터에서 MultinomialNB와 정확도를 비교하세요. Bernoulli가 이기는 경우는 언제인가요?

4. **NB vs 로지스틱 회귀.** 텍스트 데이터에 대해 두 모델을 모두 학습하세요. 학습 샘플을 100개부터 시작하여 10,000개까지 늘려가세요. 두 모델 모두에 대해 학습 세트 크기에 따른 정확도를 그래프로 그려 보세요. 로지스틱 회귀가 Naive Bayes를 추월하는 지점은 어디인가요?

5. **스팸 필터.** 완전한 스팸 분류기를 구축하세요. 원시 이메일 텍스트를 토큰화하고, 어휘를 구축하고, bag-of-words 특징을 생성하고, MultinomialNB를 학습하고, precision과 recall로 평가하세요 (정확도만으로는 왜 안 될까요?).

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| Naive Bayes | "단순한 확률 분류기" | 특징이 클래스가 주어졌을 때 조건부 독립이라는 가정을 적용하여 Bayes 정리를 적용하는 분류기 |
| 조건부 독립 | "특징이 서로 영향을 주지 않는다" | P(A, B \| C) = P(A \| C) * P(B \| C) -- C를 알면 B를 아는 것이 A에 대해 새로운 정보를 주지 못함 |
| Laplace 스무딩 | "add-one 스무딩" | 예측을 지배하는 0 확률을 방지하기 위해 모든 특징에 작은 개수를 더하는 것 |
| 사전 확률 | "데이터를 보기 전에 믿었던 것" | P(class) -- 특징을 관측하기 전 각 클래스의 확률 |
| 우도 | "데이터가 얼마나 잘 맞는지" | P(features \| class) -- 클래스가 알려져 있을 때 이러한 특징을 관측할 확률 |
| 사후 확률 | "데이터를 본 후의 믿음" | P(class \| features) -- 특징을 관측한 후 업데이트된 클래스의 확률 |
| 생성 모델 | "데이터가 생성되는 방식을 모델링" | P(X \| Y)와 P(Y)를 학습한 후, 베이즈 정리를 사용하여 P(Y \| X)를 얻는 모델 |
| 판별 모델 | "결정 경계를 모델링" | X가 생성되는 방식을 모델링하지 않고 P(Y \| X)를 직접 학습하는 모델 |
| 로그 확률 | "언더플로우 방지" | P 대신 log P를 사용하여, 작은 수들의 곱이 부동 소수점에서 0이 되는 것을 방지 |

## 추가 읽기

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 수학적 세부 사항이 포함된 세 가지 변형 모두
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 텍스트에 대한 Multinomial과 Bernoulli의 고전적 비교
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 텍스트에 대한 NB 개선 사항
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- NB가 LR보다 적은 데이터로 더 빠르게 수렴함을 증명
