# 노름과 거리 (Norms and Distances)

> 거리 함수가 "비슷하다"의 의미를 정의합니다. 잘못 고르면 하류의 모든 것이 깨집니다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- L1, L2, cosine, Mahalanobis, Jaccard, edit distance 함수를 처음부터 직접 구현합니다
- 주어진 ML 작업에 맞는 거리 측도를 선택하고, 대안이 실패하는 이유를 설명합니다
- L1·L2 노름을 LASSO·Ridge 정규화와 그 기하적 제약 영역과 연결합니다
- 같은 데이터셋이 측도에 따라 다른 최근접 이웃을 만드는 것을 시연합니다

## 문제 상황 (The Problem)

벡터가 두 개 있습니다. 단어 임베딩일 수도, 사용자 프로필일 수도, 픽셀 배열일 수도 있습니다. 알아야 할 것: 얼마나 가까운가?

답은 어떤 거리 함수를 고르느냐에 전적으로 달렸습니다. 두 데이터 포인트는 한 측도에서는 최근접 이웃이고 다른 측도에서는 멀리 떨어져 있을 수 있습니다. KNN 분류기, 추천 엔진, 벡터 데이터베이스, 클러스터링 알고리즘, 손실 함수 — 전부 이 선택에 의존합니다. 틀리면 모델이 잘못된 것을 최적화합니다.

보편적으로 최선인 거리는 없습니다. L2는 공간 데이터에 잘 맞습니다. Cosine similarity는 NLP를 지배합니다. Jaccard는 집합을, edit distance는 문자열을 다룹니다. Mahalanobis는 상관을 반영합니다. Wasserstein은 확률 질량을 옮깁니다. 각각 "비슷하다"에 대한 다른 가정을 인코딩합니다.

이 레슨은 주요 거리 함수를 처음부터 만들고, 각각이 언제 맞는 도구인지 보여 주며, 같은 데이터가 측도에 따라 완전히 다른 최근접 이웃을 만드는 것을 시연합니다.

## 핵심 개념 (The Concept)

### 노름: 벡터 크기 측정

노름은 벡터의 "크기"를 잽니다. 두 벡터 사이의 모든 거리 함수는 차이의 노름으로 쓸 수 있습니다: d(a, b) = ||a - b||. 따라서 노름을 이해하는 것은 거리를 이해하는 것입니다.

### L1 노름 (맨해튼 거리)

L1 노름은 모든 성분의 절댓값을 합합니다.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

맨해튼 거리라고 불리는 이유는, 축을 따라서만 움직일 수 있는(대각선 불가) 도시 격자에서 얼마나 걷는지 재기 때문입니다.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1을 쓸 때:
- 고차원 희소 데이터 (텍스트 피처, 원-핫 인코딩)
- 이상치에 강건하고 싶을 때 (하나의 큰 차이가 지배하지 않음)
- 피처 선택 문제 (L1 정규화가 희소성을 촉진)

L1 정규화(Lasso)와의 연결: 손실에 ||w||_1을 더하면 절대 가중치 합에 페널티를 줍니다. 작은 가중치를 정확히 0으로 밀어 자동 피처 선택을 합니다. L1 페널티는 가중치 공간에 다이아몬드 모양 제약 영역을 만들고, 다이아몬드 꼭짓점은 일부 가중치가 0인 축 위에 있습니다.

손실 함수와의 연결: 평균 절대 오차(MAE)는 예측과 타깃 사이 평균 L1 거리입니다. 모든 오차를 선형으로 페널티하여 MSE보다 이상치에 강건합니다.

### L2 노름 (유클리드 거리)

L2 노름은 직선 거리입니다. 제곱 성분의 합의 제곱근.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

기하 시간에 배운 거리입니다. n차원의 피타고라스.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2를 쓸 때:
- 저~중차원 연속 데이터
- 피처 스케일이 비슷할 때
- 물리적 거리 (공간 데이터, 센서 측정)
- 픽셀 수준의 이미지 유사도

L2 정규화(Ridge)와의 연결: 손실에 ||w||_2^2를 더하면 큰 가중치에 페널티를 줍니다. L1과 달리 가중치를 0으로 밀지 않습니다. 모든 가중치를 비례적으로 0 쪽으로 수축합니다. L2 페널티는 원형 제약 영역을 만들어 축 위의 꼭짓점이 없습니다. 가중치는 작아지지만 정확히 0이 되는 경우는 드뭅니다.

손실 함수와의 연결: 평균 제곱 오차(MSE)는 L2 거리 제곱의 평균입니다. 제곱은 큰 오차를 작은 오차보다 더 강하게 페널티합니다.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 노름: 일반 가족

L1과 L2는 Lp 노름의 특수 경우입니다:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p 값에 따라 다른 모양의 "단위 공"(원점에서 거리 1인 모든 점의 집합)이 생깁니다:

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 노름 (체비쇼프 거리)

p가 무한대로 가면 Lp 노름은 절댓값 성분의 최댓값으로 수렴합니다.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

두 점 사이 거리는 가장 크게 다른 한 차원으로 결정됩니다. 다른 차원은 무시됩니다.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-infinity를 쓸 때:
- 어느 한 차원의 최악 편차가 중요할 때
- 보드 게임 (체스의 왕은 L-infinity로 이동: 어떤 방향이든 한 칸 비용 1)
- 제조 공차 (모든 차원이 스펙 안이어야 함)

### Cosine Similarity와 Cosine Distance

Cosine similarity는 크기를 무시하고 두 벡터 사이 각도를 잽니다.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

범위는 -1(반대 방향)부터 +1(같은 방향)까지입니다. 수직 벡터의 cosine similarity는 0입니다.

Cosine distance는 거리로 변환합니다: cosine_distance = 1 - cosine_similarity. 범위는 0(동일 방향)부터 2(반대 방향)까지입니다.

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

cosine이 NLP와 임베딩을 지배하는 이유: 텍스트에서 문서 길이는 유사도에 영향을 주면 안 됩니다. 고양이에 관한 문서가 다른 고양이에 관한 문서의 두 배 길이여도 여전히 "비슷"해야 합니다. Cosine similarity는 크기(길이)를 무시하고 방향만 봅니다. 같은 단어 분포이지만 길이가 다른 두 문서는 같은 방향을 가리켜 cosine similarity 1.0을 얻습니다.

cosine similarity를 쓸 때:
- 텍스트 유사도 (TF-IDF 벡터, 단어 임베딩, 문장 임베딩)
- 크기는 잡음이고 방향이 신호인 모든 도메인
- 추천 시스템 (사용자 선호도 벡터)
- 임베딩 검색 (벡터 데이터베이스는 거의 항상 cosine 또는 내적 사용)

### 내적 유사도 vs Cosine Similarity

두 벡터의 내적은:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity는 두 크기로 정규화한 내적입니다. 두 벡터가 이미 단위 정규화(크기 = 1)되어 있으면 내적과 cosine similarity는 동일합니다.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

다를 때: 내적은 크기 정보를 포함합니다. 크기가 큰 벡터가 더 높은 내적 점수를 받습니다. "인기" 아이템을 더 높게 두고 싶은 일부 검색 시스템에서 중요합니다. 크기가 암묵적 품질·중요도 신호로 작용합니다.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

실무에서:
- 순수 방향 유사도가 필요하면 cosine similarity
- 크기가 의미 있는 정보를 담으면 내적
- 많은 벡터 데이터베이스(Pinecone, Weaviate, Qdrant)가 둘 중 선택을 허용
- 임베딩이 L2 정규화되어 있으면 선택은 중요하지 않음

### Mahalanobis 거리

유클리드 거리는 모든 차원을 동등하게 취급합니다. 하지만 피처가 상관되어 있거나 스케일이 다르면 L2는 오해를 줍니다.

Mahalanobis 거리는 데이터의 공분산 구조를 반영합니다.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

여기서 S는 데이터의 공분산 행렬입니다.

직관적으로: Mahalanobis 거리는 먼저 데이터를 비상관화·정규화(whitening)한 뒤, 그 변환된 공간에서 L2를 계산합니다. S가 단위행렬(비상관, 단위 분산 피처)이면 Mahalanobis는 유클리드 거리로 환원됩니다.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Mahalanobis 거리를 쓸 때:
- 이상치 탐지 (평균에서 Mahalanobis 거리가 큰 점이 이상치)
- 피처 스케일과 상관이 다른 분류
- 신뢰할 수 있는 공분산 행렬을 추정할 데이터가 충분할 때
- 제조 품질 관리 (다변량 공정 모니터링)

### Jaccard Similarity (집합용)

Jaccard similarity는 두 집합의 겹침을 잽니다.

```
J(A, B) = |A intersect B| / |A union B|
```

범위는 0(겹침 없음)부터 1(동일 집합)까지입니다. Jaccard distance = 1 - Jaccard similarity.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Jaccard를 쓸 때:
- 태그, 카테고리, 피처 집합 비교
- (빈도가 아닌) 단어 존재 기반 문서 유사도
- 거의 중복 탐지 (Jaccard의 MinHash 근사)
- 이진 피처 벡터 비교 (존재/부재 데이터)
- 세그멘테이션 모델 평가 (Intersection over Union = Jaccard)

### Edit Distance (Levenshtein Distance)

Edit distance는 한 문자열을 다른 문자열로 바꾸는 데 필요한 최소 단일 문자 연산 수를 셉니다. 연산은 삽입, 삭제, 치환입니다.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

동적 프로그래밍으로 계산합니다. 항목 (i, j)가 문자열 A의 앞 i자와 문자열 B의 앞 j자 사이 edit distance인 행렬을 채웁니다.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

edit distance를 쓸 때:
- 맞춤법 검사와 교정
- DNA 시퀀스 정렬 (가중 연산 포함)
- 퍼지 문자열 매칭
- 지저분한 텍스트 데이터의 중복 제거

### KL Divergence (거리는 아니지만 거리처럼 쓰임)

KL 발산은 한 확률 분포가 다른 분포와 얼마나 다른지 잽니다. Lesson 09에서 다루지만, 거리가 아님에도 사람들이 "거리"처럼 쓰기 때문에 이 논의에 속합니다.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

핵심 성질: KL 발산은 대칭이 아닙니다.

```
D_KL(P || Q) != D_KL(Q || P)
```

거리 측도의 기본 요구를 만족하지 않습니다. 삼각형 부등식도 만족하지 않습니다. 거리(distance)가 아니라 발산(divergence)입니다.

Forward KL (D_KL(P || Q))는 "평균 추구(mean-seeking)": Q가 P의 모든 모드를 덮으려 합니다.
Reverse KL (D_KL(Q || P))는 "모드 추구(mode-seeking)": Q가 P의 단일 모드에 집중합니다.

KL 발산을 볼 때:
- VAE (ELBO의 KL 항이 잠재 분포를 prior 쪽으로 밀어냄)
- 지식 증류 (학생이 교사 분포를 맞추려 함)
- RLHF (KL 페널티가 미세조정 모델을 베이스 모델 가까이에 유지)
- 정책 그래디언트 방법 (정책 업데이트 제약)

### Wasserstein Distance (Earth Mover's Distance)

Wasserstein 거리는 한 확률 분포를 다른 분포로 바꾸는 데 필요한 최소 "일"을 잽니다. 한 분포가 흙더미이고 다른 분포가 구덩이라면, 흙을 얼마나 멀리 얼마나 많이 옮겨야 하는가?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

1D 분포에서는 누적 분포 함수의 절댓값 차이의 적분으로 단순해집니다:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein이 중요한 이유:
- 진짜 측도입니다 (대칭, 삼각형 부등식 만족)
- 분포가 겹치지 않아도 그래디언트를 제공합니다 (KL 발산은 무한대로 감)
- 이 성질 때문에 원래 GAN의 학습 불안정성을 해결한 Wasserstein GAN(WGAN)의 핵심이 되었습니다

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Wasserstein을 쓸 때:
- GAN 학습 (WGAN, WGAN-GP)
- 겹치지 않을 수 있는 분포 비교
- 최적 수송 문제
- 이미지 검색 (색 히스토그램 비교)

### 왜 작업마다 다른 거리가 필요한가

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### 손실 함수와의 연결

손실 함수는 예측 대 타깃에 적용된 거리 함수입니다.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 정규화와의 연결

정규화는 손실 함수에 가중치 노름 페널티를 더합니다.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

L1은 희소성을 만들고 L2는 만들지 않는 이유: 2D 가중치 공간의 제약 영역을 그리세요. L1은 다이아몬드, L2는 원입니다. 손실 함수의 등고선(타원)은 다이아몬드와 꼭짓점(한 가중치가 0)에서 만날 가능성이 가장 큽니다. 원과는 매끄러운 점에서 만나 두 가중치가 모두 0이 아닙니다.

### 최근접 이웃 검색

모든 거리 함수는 최근접 이웃 검색 문제를 함의합니다: 질의점이 주어지면 데이터셋에서 가장 가까운 점을 찾습니다.

정확한 최근접 이웃 검색은 n개 점, d차원 데이터셋에서 질의당 O(n * d)입니다. 큰 데이터셋에서는 너무 느립니다.

근사 최근접 이웃(ANN) 알고리즘은 약간의 정확도를 희생해 막대한 속도 이득을 얻습니다:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW(Hierarchical Navigable Small World)는 현대 벡터 데이터베이스의 지배적 알고리즘입니다. 각 노드가 근사 최근접 이웃과 연결되는 다층 그래프를 만듭니다. 검색은 최상층(희소, 긴 점프)에서 시작해 최하층(밀집, 짧은 점프)으로 내려갑니다.

```figure
norm-unit-balls
```

## 구현하기 (Build It)

### Step 1: 모든 노름과 거리 함수

완전한 구현은 `code/distances.py`를 보세요. 모든 함수는 기본 Python 수학만으로 처음부터 만들어집니다.

### Step 2: 같은 데이터, 다른 거리, 다른 이웃

`distances.py`의 데모는 데이터셋을 만들고, 질의점을 고른 뒤, 거리 측도에 따라 최근접 이웃이 어떻게 바뀌는지 보여 줍니다. L1에서 "가장 가까운" 점이 L2나 cosine에서는 가장 가깝지 않을 수 있습니다.

### Step 3: 임베딩 유사도 검색

코드에는 cosine similarity 대 L2 거리로 질의와 가장 비슷한 "문서"를 찾는 목 임베딩 유사도 검색이 포함되어, 순위가 달라질 수 있음을 보여 줍니다.

## 실용 활용 (Use It)

가장 흔한 실전 용도: 벡터 데이터베이스에서 비슷한 항목 찾기.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

`model.encode(text)`를 호출한 뒤 벡터 데이터베이스를 검색할 때, 내부에서 일어나는 일입니다. 임베딩 모델이 텍스트를 벡터로 매핑합니다. 벡터 데이터베이스는 질의 벡터와 저장된 모든 벡터 사이 cosine similarity(또는 내적)를 계산하며, 전부를 검사하지 않도록 ANN 알고리즘을 사용합니다.

## 연습 문제 (Exercises)

1. (1, 2, 3)과 (4, 0, 6) 사이 L1, L2, L-infinity 거리를 계산하세요. 임의의 점 쌍에 대해 항상 L-inf <= L2 <= L1이 성립함을 검증하고, 이 순서가 보장되는 이유를 증명하세요.

2. cosine similarity는 높고(> 0.9) L2 거리는 큰(> 10) 벡터 쌍을 만드세요. 기하적으로 무엇이 일어나는지 설명하세요. 그다음 cosine similarity는 낮고(< 0.3) L2 거리는 작은(< 0.5) 벡터 쌍을 만드세요.

3. 데이터셋과 질의점을 받아 L1, L2, cosine, Mahalanobis 거리 아래 최근접 이웃을 반환하는 함수를 구현하세요. 네 측도가 어느 점이 최근접인지에 모두 의견이 갈리는 데이터셋을 찾으세요.

4. CDF 방법으로 [0.5, 0.5, 0, 0]과 [0, 0, 0.5, 0.5] 사이 Wasserstein 거리를 손으로 계산하세요. 그다음 [0.25, 0.25, 0.25, 0.25]와 [0, 0, 0.5, 0.5] 사이를 계산하세요. 어느 쪽이 더 크고 왜인가요?

5. 근사 Jaccard similarity용 MinHash를 구현하세요. 랜덤 집합 100개를 만들고, 모든 쌍의 exact Jaccard를 계산한 뒤, 해시 함수 50·100·200개를 쓴 MinHash 근사와 비교하세요. 근사 오차를 플롯하세요.

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "벡터의 크기" | 벡터를 음이 아닌 스칼라로 매핑하는 함수. 삼각형 부등식, 절대 동차성, 영벡터에서만 0을 만족합니다 |
| L1 norm | "맨해튼 거리" | 절댓값 성분의 합. 최적화에서 희소성을 만듭니다. 이상치에 강건합니다 |
| L2 norm | "유클리드 거리" | 제곱 성분 합의 제곱근. 유클리드 공간의 직선 거리 |
| Lp norm | "일반화된 노름" | 절댓값 성분의 p제곱 합의 p제곱근. L1과 L2가 특수 경우 |
| L-infinity norm | "최대 노름" 또는 "체비쇼프 거리" | 절댓값 성분의 최댓값. p가 무한대로 갈 때 Lp의 극한 |
| Cosine similarity | "벡터 사이 각도" | 두 크기로 정규화한 내적. -1부터 +1. 벡터 길이를 무시합니다 |
| Cosine distance | "1 빼기 cosine similarity" | cosine similarity를 거리로 변환. 0부터 2 |
| Dot product | "정규화되지 않은 cosine" | 성분별 곱의 합. cosine similarity에 두 크기를 곱한 것과 같음 |
| Mahalanobis distance | "상관을 반영한 거리" | 데이터 공분산으로 whitening(비상관화·정규화)한 공간에서의 L2 거리 |
| Jaccard similarity | "집합 겹침" | 교집합 크기를 합집합 크기로 나눔. 벡터가 아니라 집합용 |
| Edit distance | "Levenshtein distance" | 한 문자열을 다른 문자열로 바꾸는 최소 삽입·삭제·치환 수 |
| KL divergence | "분포 사이 거리" | 진짜 거리가 아님(비대칭). Q로 P를 인코딩할 때 추가 비트를 잽니다 |
| Wasserstein distance | "흙 옮기기 거리" | 한 분포에서 다른 분포로 질량을 옮기는 최소 일. 진짜 측도 |
| Approximate nearest neighbor | "ANN 검색" | exact 검색보다 훨씬 빠르게 대략 가장 가까운 점을 찾는 알고리즘(HNSW, LSH, IVF) |
| HNSW | "벡터 DB 알고리즘" | Hierarchical Navigable Small World 그래프. 빠른 근사 최근접 이웃 검색용 다층 그래프 |
| L1 regularization | "Lasso" | 손실에 가중치 L1 노름을 더함. 가중치를 0으로 밀어 희소성 |
| L2 regularization | "Ridge" 또는 "weight decay" | 손실에 가중치 L2 노름 제곱을 더함. 희소성 없이 0 쪽으로 수축 |
| Elastic Net | "L1 + L2" | L1과 L2 정규화를 결합. 상관된 피처 그룹을 단독보다 잘 다룸 |

## 더 읽을거리 (Further Reading)

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - 십억 규모 ANN 검색을 위한 Meta 라이브러리
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - GAN에 Earth Mover's distance를 도입한 논문
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 기초적인 ANN 알고리즘
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec. cosine similarity가 임베딩 기본값이 된 계기
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn의 거리 측도와 이웃 알고리즘 실전 가이드
