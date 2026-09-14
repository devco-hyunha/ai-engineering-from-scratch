# 특성 공학과 선택 (Feature Engineering & Selection)

> 좋은 특성 하나가 데이터 포인트 천 개 값입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Statistics for ML, Linear Algebra), Phase 2 Lessons 1-7
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- 수치 변환(표준화, min-max 스케일링, 로그 변환, 구간화)을 구현하고 각각이 언제 적절한지 설명합니다
- 범주형 특성에 one-hot·라벨·타깃 인코딩을 만들고, 타깃 인코딩의 데이터 누수 위험을 식별합니다
- TF-IDF 벡터라이저를 처음부터 구현하고, 텍스트 분류에서 원시 단어 카운트보다 나은 이유를 설명합니다
- 필터 기반 특성 선택(분산 임계값, 상관, 상호정보)을 적용해 차원을 줄입니다

## 문제 상황 (The Problem)

데이터셋이 있습니다. 알고리즘을 고릅니다. 학습합니다. 결과가 그저 그렇습니다. 더 화려한 알고리즘을 시도합니다. 여전히 그저 그렇습니다. 하이퍼파라미터 튜닝에 일주일을 씁니다. 개선은 미미합니다.

그러다 누군가가 원시 데이터를 더 나은 특성으로 바꾸고, 단순 로지스틱 회귀가 튜닝된 그래디언트 부스팅 앙상블을 이깁니다.

이런 일은 끊임없이 일어납니다. 고전 ML에서는 알고리즘 선택보다 데이터 표현이 더 중요합니다. "면적"과 "침실 수"를 쓰는 집값 모델은, 학습기가 아무리 정교해도 "주소 원시 문자열"을 쓰는 모델을 이깁니다. 알고리즘은 당신이 준 것만으로 일할 수 있습니다.

특성 공학은 원시 데이터를 모델이 패턴을 찾기 쉬운 표현으로 바꾸는 과정입니다. 특성 선택은 신호 없이 노이즈만 더하는 특성을 버리는 과정입니다. 둘이 합쳐져 고전 ML에서 레버리지가 가장 큰 활동이 됩니다.

## 핵심 개념 (The Concept)

### 특성 파이프라인

```mermaid
flowchart LR
    A[원시 데이터] --> B[결측값 처리]
    B --> C[수치 변환]
    B --> D[범주형 인코딩]
    B --> E[텍스트 특성]
    C --> F[특성 상호작용]
    D --> F
    E --> F
    F --> G[특성 선택]
    G --> H[모델 준비 데이터]
```

### 수치 특성

원시 숫자는 거의 모델에 바로 쓸 수 없습니다. 흔한 변환:

**스케일링:** 특성을 같은 범위에 두어, 거리 기반 알고리즘(K-Means, KNN, SVM)이 모든 특성을 동등하게 다루게 합니다. Min-max 스케일링은 [0, 1]로 매핑합니다. 표준화(z-score)는 평균=0, 표준편차=1로 매핑합니다.

**로그 변환:** 오른쪽 치우친 분포(소득, 인구, 단어 수)를 압축합니다. 곱셈 관계를 덧셈 관계로 바꿉니다.

**구간화 (Binning):** 연속값을 범주로 바꿉니다. 특성과 타깃의 관계가 비선형이지만 계단식일 때 유용합니다(예: 연령대).

**다항 특성:** x^2, x^3, x1*x2 항을 만듭니다. 특성 수가 늘어나는 대신 선형 모델이 비선형 관계를 잡을 수 있게 합니다.

### 범주형 특성

모델은 숫자가 필요합니다. 범주는 인코딩이 필요합니다.

**원-핫 인코딩 (One-hot encoding):** 각 범주에 이진 열을 만듭니다. "color = red/blue/green"이 is_red, is_blue, is_green 세 열이 됩니다. 카디널리티가 낮은 특성에는 잘 작동하지만, 범주가 많으면 폭발합니다.

**라벨 인코딩 (Label encoding):** 각 범주를 정수로 매핑합니다: red=0, blue=1, green=2. 거짓 순서를 도입합니다(모델이 green > blue > red라고 생각할 수 있음). 개별 값으로 분할하는 트리 기반 모델에만 적절합니다.

**타깃 인코딩 (Target encoding):** 각 범주를 그 범주의 타깃 변수 평균으로 바꿉니다. 강력하지만 위험합니다: 데이터 누수 위험이 큽니다. 학습 데이터에서만 계산하고 테스트 데이터에 적용해야 합니다.

### 텍스트 특성

**카운트 벡터라이저:** 문서에서 각 단어가 몇 번 나타나는지 셉니다. "the cat sat on the mat"은 {the: 2, cat: 1, sat: 1, on: 1, mat: 1}이 됩니다.

**TF-IDF:** Term Frequency-Inverse Document Frequency. 문서 전반에서 얼마나 독특한지로 단어에 가중치를 줍니다. "the" 같은 흔한 단어는 낮은 가중치. 드물고 구별되는 단어는 높은 가중치.

```
TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF * IDF
```

### 결측값

실제 데이터에는 구멍이 있습니다. 전략:

- **행 삭제:** 결측이 드물고 무작위일 때만
- **평균/중앙값 대체:** 단순하고 분포 형태를 유지(중앙값이 이상치에 더 견고)
- **최빈값 대체:** 범주형 특성용
- **지시자 열:** 대체하기 전에 "was_this_missing" 이진 열을 추가. 데이터가 빠진다는 사실 자체가 정보가 될 수 있음
- **전방/후방 채우기:** 시계열 데이터용

### 특성 상호작용

때로는 관계가 조합에 있습니다. "키"와 "몸무게"만으로는 "BMI = weight / height^2"보다 예측력이 약합니다. 특성 상호작용은 특성 공간을 곱으로 늘리므로, 도메인 지식으로 맞는 것을 고르세요.

### 특성 선택

특성이 많을수록 항상 나은 것은 아닙니다. 관련 없는 특성은 노이즈를 더하고, 학습 시간을 늘리며, 과적합을 유발할 수 있습니다.

**필터 방법 (모델 전):**
- 상관: 서로 상관이 높은 특성 제거(중복)
- 상호정보: 특성을 알면 타깃에 대한 불확실성이 얼마나 줄어드는지 측정
- 분산 임계값: 거의 변하지 않는 특성 제거

**래퍼 방법 (모델 기반):**
- L1 정규화 (Lasso): 관련 없는 특성 가중치를 정확히 0으로 밀어냄
- 재귀적 특성 제거: 학습 → 가장 덜 중요한 특성 제거 → 반복

**선택이 중요한 이유:** 좋은 특성 10개 모델이, 좋은 특성 10개 + 노이즈 90개 모델보다 보통 낫습니다. 노이즈 특성은 일반화되지 않는 학습 데이터 패턴에 과적합할 기회를 줍니다.

```figure
feature-scaling
```

## 직접 만들기 (Build It)

### 1단계: 수치 변환을 처음부터

```python
import math


def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result
```

### 2단계: 범주형 인코딩을 처음부터

```python
def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding
```

### 3단계: 텍스트 특성을 처음부터

```python
def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab
```

### 4단계: 결측값 대체를 처음부터

```python
def impute_mean(values):
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]
```

### 5단계: 특성 선택을 처음부터

```python
def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]
```

### 6단계: 전체 파이프라인과 데모

```python
import random


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== Raw Data Sample ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== Missing Value Handling ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  sqft missing: {sqft_missing}/{len(sqft_raw)}")
    print(f"  age missing: {age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  sqft filled with median: {sqft_fill:.0f}")
    print(f"  age filled with mean: {age_fill:.1f}")

    print("\n=== Numerical Transforms ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  sqft standardized: mean={sum(sqft_scaled)/len(sqft_scaled):.4f}, std={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  age bins: {sorted(set(age_binned))}")

    print("\n=== Categorical Encoding ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  One-hot categories: {ohe_cats}")
    print(f"  Sample encoding: {neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  Label encoding map: {le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  Target encoding: {({k: round(v) for k, v in te_map.items()})}")

    print("\n=== Text Features ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  Vocabulary size: {len(cv_vocab)}")
    print(f"  Doc 0 non-zero features: {sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF vocabulary size: {len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  Doc 0 top TF-IDF words: {top_words}")

    print("\n=== Polynomial Features ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  Input: {[round(v, 4) for v in sample_row]}")
    print(f"  Polynomial: {[round(v, 4) for v in poly]}")
    print(f"  Features: [x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== Feature Selection ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  Total features: {len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  After variance threshold (0.01): {len(surviving_var)} features kept")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  After correlation filter (0.9): {len(surviving_corr)} features kept")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  Mutual information with target:")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}: MI={mi:.4f}")

    print("\n  Correlation with price:")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}: r={corr:.4f}")
```

## 활용하기 (Use It)

scikit-learn에서는 이 변환들을 조합 가능한 파이프라인으로 씁니다:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

처음부터 구현한 버전은 각 변환 안에서 정확히 무엇이 일어나는지 보여 줍니다. 라이브러리 버전은 엣지 케이스 처리, 희소 행렬 지원, 파이프라인 조합을 더하지만, 수학은 같습니다.

## 산출물 (Ship It)

이 레슨은 다음을 만듭니다:
- `outputs/prompt-feature-engineer.md` — 원시 데이터에서 체계적으로 특성을 공학하는 프롬프트

## 연습 문제 (Exercises)

1. 수치 변환에 견고 스케일링(평균·표준편차 대신 중앙값과 사분위수 범위 사용)을 추가하세요. 극단 이상치가 있는 데이터에서 표준 스케일링과 비교하세요.
2. leave-one-out 타깃 인코딩을 구현하세요: 각 행에 대해 그 행의 타깃 값을 제외한 타깃 평균을 계산합니다. 나이브 타깃 인코딩보다 과적합을 어떻게 줄이는지 보이세요.
3. 분산 임계값, 상관 필터, 상호정보 랭킹을 결합한 자동 특성 선택 파이프라인을 만드세요. 주택 데이터셋에 적용하고, 전체 특성 vs 선택된 특성으로 모델 성능(단순 선형 회귀)을 비교하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Feature engineering | "새 열 만들기" | 원시 데이터를 모델에 패턴이 드러나는 표현으로 변환 |
| Standardization | "정규화하기" | 평균을 빼고 표준편차로 나눠 특성을 mean=0, std=1로 만듦 |
| One-hot encoding | "더미 변수 만들기" | 범주마다 이진 열 하나. 각 행에서 정확히 한 열만 1 |
| Target encoding | "정답으로 인코딩" | 각 범주를 그 범주의 평균 타깃 값으로 대체. 과적합 방지용 스무딩 포함 |
| TF-IDF | "있어 보이는 단어 세기" | Term Frequency × Inverse Document Frequency: 코퍼스에서 얼마나 구별되는지로 단어에 가중 |
| Imputation | "빈칸 채우기" | 결측값을 추정값(평균, 중앙값, 최빈값, 또는 모델 예측)으로 대체 |
| Feature selection | "나쁜 열 버리기" | 노이즈나 중복을 더하는 특성을 제거하고, 타깃에 신호가 있는 것만 유지 |
| Mutual information | "하나가 다른 것에 대해 얼마나 알려주는지" | 변수 X를 관측해 변수 Y에 대한 불확실성이 얼마나 줄어드는지의 측도 |
| Data leakage | "우연히 부정행위" | 예측 시점에 쓸 수 없는 정보를 학습에 써서, 성능을 거짓으로 낙관적으로 만듦 |

## 더 읽을거리 (Further Reading)

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/) — 특성 공학 전 지형을 다루는 무료 온라인 책
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html) — 표준 변환의 실무 참고서
- [Target Encoding Done Right (Micci-Barreca, 2001)](https://dl.acm.org/doi/10.1145/507533.507538) — 스무딩이 있는 타깃 인코딩의 원논문
