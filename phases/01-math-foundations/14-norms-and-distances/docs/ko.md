# 규범과 거리

> 거리 함수가 "유사함"의 의미를 정의합니다. 잘못된 선택을 하면 모든 후속 과정이 무너집니다.

**유형:** Build
**언어:** Python
**선수 요건:** 1단계, 01강 (선형대수 직관), 02강 (벡터, 행렬 및 연산)
**시간:** 약 90분

## 학습 목표

- L1, L2, 코사인, 마할라노비스, 자카드 및 편집 거리 함수를 처음부터 구현해 보세요
- 주어진 ML 작업에 적합한 거리 메트릭을 선택하고, 대안들이 실패하는 이유를 설명해 보세요
- L1 및 L2 규범을 LASSO와 Ridge 정규화 및 그 기하학적 제약 영역과 연결해 보세요
- 동일한 데이터셋이 서로 다른 메트릭에 따라 서로 다른 최근접 이웃을 생성하는 방식을 시연해 보세요

## 문제점

두 개의 벡터가 있습니다. 단어 임베딩일 수도 있고, 사용자 프로필일 수도 있으며, 픽셀 배열일 수도 있습니다. 당신은 알아야 합니다: 얼마나 가까운가요?

답은 선택하는 거리 함수에 전적으로 의존합니다. 두 데이터 포인트는 한 메트릭에서는 최근접 이웃일 수 있지만, 다른 메트릭에서는 멀리 떨어져 있을 수 있습니다. KNN 분류기, 추천 엔진, 벡터 데이터베이스, 클러스터링 알고리즘, 손실 함수 -- 이 모든 것은 이 선택에 의존합니다. 잘못 선택하면 모델이 잘못된 것을 최적화하게 됩니다.

보편적으로 가장 좋은 거리는 없습니다. L2는 공간 데이터에 잘 맞습니다. 코사인 유사성은 NLP에서 지배적입니다. 자카드 메트릭은 집합을 처리합니다. 편집 거리는 문자열을 처리합니다. 마할라노비스는 상관관계를 고려합니다. Wasserstein은 확률 질량을 이동시킵니다. 각각은 "유사함"의 의미에 대한 서로 다른 가정을 인코딩합니다.

이 강의는 모든 주요 거리 함수를 처음부터 구축하며, 각각이 언제 올바른 도구인지 보여주고, 사용하는 메트릭에 따라 동일한 데이터가 완전히 다른 최근접 이웃을 생성하는 방식을 시연합니다.

## 개념

### 규범: 벡터 크기 측정

규범은 벡터의 "크기"를 측정합니다. 두 벡터 사이의 모든 거리 함수는 그 차이의 규범으로 작성할 수 있습니다: d(a, b) = ||a - b||. 따라서 규범을 이해하는 것은 거리를 이해하는 것입니다.

### L1 규범 (맨해튼 거리)

L1 노름은 모든 구성 요소의 절대값을 합산합니다.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

축을 따라만 이동할 수 있는 도시 격자에서 얼마나 멀리 걷는지를 측정하기 때문에 맨해튼 거리라고 불립니다. 대각선 이동은 허용되지 않습니다.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1을 사용하는 경우:
- 고차원 희소 데이터 (텍스트 특징, 원-핫 인코딩)
- 이상치에 대한 강인성을 원할 때 (단 하나의 큰 차이가 지배하지 않음)
- 특징 선택 문제 (L1 정규화는 희소성을 장려합니다)

L1 정규화(Lasso)와의 연결: 손실 함수에 ||w||_1을 추가하면 가중치 값의 절대값 합에 페널티를 부여합니다. 이는 작은 가중치를 정확히 0으로 만들어 자동 특징 선택을 수행합니다. L1 페널티는 가중치 공간에서 다이아몬드 형태의 제약 영역을 생성하며, 다이아몬드의 꼭짓점은 일부 가중치가 0인 축 위에 위치합니다.

손실 함수와의 연결: 평균 절대 오차(MAE)는 예측값과 목표값 사이의 평균 L1 거리입니다. 모든 오차를 선형적으로 페널티를 부여하므로 MSE에 비해 이상치에 강인합니다.

### L2 노름 (유클리드 거리)

L2 노름은 직선 거리입니다. 구성 요소의 제곱 합에 대한 제곱근입니다.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

이는 기하학 수업에서 배운 거리입니다. n차원에서의 피타고라스입니다.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2를 사용하는 경우:
- 저차원~중차원 연속 데이터
- 특징 스케일이 비교 가능할 때
- 물리적 거리 (공간 데이터, 센서 측정값)
- 픽셀 수준의 이미지 유사성

L2 정규화(Ridge)와의 연결: 손실 함수에 ||w||_2^2를 추가하면 큰 가중치에 페널티를 부여합니다. L01강 달리 가중치를 0으로 만들지 않습니다. 모든 가중치를 비례적으로 0에 가깝게 축소합니다. L2 페널티는 원형 제약 영역을 생성하므로 축 위에 꼭짓점이 없습니다. 가중치는 작아지지만 거의 정확히 0이 되지는 않습니다.

손실 함수와의 연결: 평균 제곱 오차(MSE)는 L2 거리의 제곱의 평균입니다. 제곱은 작은 오차보다 큰 오차에 더 큰 페널티를 부여합니다.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 노름: 일반 계열

L01강 L2는 Lp 노름의 특수한 경우입니다:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p의 값이 다르면 서로 다른 모양의 "단위 공" (원점에서 거리가 1인 모든 점의 집합)이 생성됩니다:

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-무한대 노름 (체비셰프 거리)

p가 무한대로 접근하면 Lp 노름은 최대 절대 성분에 수렴합니다.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

두 점 사이의 거리는 두 점이 가장 크게 차이나는 단일 차원에 의해 결정됩니다. 다른 모든 차원은 무시됩니다.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-무한대를 사용하는 경우:
- 단일 차원에서의 최악의 편차가 중요한 경우
- 게임 보드 (체스의 킹은 L-무한대로 이동합니다: 어떤 방향으로 한 칸 이동하는 비용은 1입니다)
- 제조 허용 오차 (모든 차원이 사양 내에 있어야 합니다)

### 코사인 유사도와 코사인 거리

코사인 유사도는 두 벡터 사이의 각도를 측정하며, 크기는 무시합니다.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

범위는 -1 (반대 방향)에서 +1 (같은 방향)까지입니다. 수직인 벡터의 코사인 유사도는 0입니다.

코사인 거리는 이를 거리로 변환합니다: cosine_distance = 1 - cosine_similarity. 범위는 0 (같은 방향)에서 2 (반대 방향)까지입니다.

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

코사인이 NLP 및 임베딩에서 지배적인 이유: 텍스트에서는 문서 길이가 유사성에 영향을 미치지 않아야 합니다. 고양이 관련 문서가 다른 고양이 관련 문서보다 두 배 길더라도 여전히 "유사"해야 합니다. 코사인 유사도는 크기(길이)를 무시하고 방향만 고려합니다. 단어 분포가 같지만 길이가 다른 두 문서는 같은 방향을 가리키며 코사인 유사도 1.0을 얻습니다.

코사인 유사도를 사용하는 경우:
- 텍스트 유사성 (TF-IDF 벡터, 워드 임베딩, 문장 임베딩)
- 크기가 잡음이고 방향이 신호인 모든 영역
- 추천 시스템 (사용자 선호 벡터)
- 임베딩 검색 (벡터 데이터베이스는 거의 항상 코사인 또는 내적 사용)

### 내적 유사도 vs 코사인 유사도

두 벡터의 내적은 다음과 같습니다:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

코사인 유사도는 두 크기로 정규화된 내적입니다. 두 벡터가 이미 단위 정규화(크기 = 1)되어 있으면 내적과 코사인 유사도는 동일합니다.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

두 값이 다를 때: 내적은 크기(magnitude) 정보를 포함합니다. 크기가 큰 벡터는 더 높은 내적 점수를 받습니다. 이는 "인기" 있는 항목이 더 높은 순위를 차지하길 원하는 일부 검색 시스템에서 중요합니다. 크기는 암시적인 품질 또는 중요도 신호로 작용합니다.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

실제로는:
- 순수한 방향 유사성을 원할 때는 코사인 유사도(Cosine Similarity)를 사용하세요
- 크기가 의미 있는 정보를 담고 있을 때는 내적(dot product)을 사용하세요
- 많은 벡터 데이터베이스(Vector Database)(Pinecone, Weaviate, Qdrant)는 두 가지 중 하나를 선택할 수 있게 해줍니다
- 임베딩(Embedding)이 L2 정규화(Normalization)되어 있다면 선택은 중요하지 않습니다

### 마할라노비스 거리(Mahalanobis Distance)

유클리드 거리(Euclidean distance)는 모든 차원을 동일하게 취급합니다. 하지만 특징(features)이 상관되어 있거나 스케일이 다르면 L2는 오해의 소지가 있는 결과를 줍니다.

마할라노비스 거리는 데이터의 공분산(covariance) 구조를 고려합니다.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

여기서 S는 데이터의 공분산 행렬입니다.

직관적으로: 마할라노비스 거리는 먼저 데이터를 비상관화하고 정규화(whitening)한 후, 변환된 공간에서 L2 거리를 계산합니다. S가 단위 행렬(identity matrix)(상관되지 않고 단위 분산인 특징)이라면, 마할라노비스 거리는 유클리드 거리로 축소됩니다.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

마할라노비스 거리를 사용할 때:
- 이상치 탐지(Outlier detection)(평균으로부터 마할라노비스 거리가 큰 포인트는 이상치입니다)
- 특징의 스케일과 상관관계가 다를 때 분류(Classification)
- 신뢰할 수 있는 공분산 행렬을 추정할 만큼 충분한 데이터를 가지고 있을 때
- 제조업에서의 품질 관리(다변량 프로세스 모니터링)

### 자카드 유사도(Jaccard Similarity)(집합용)

자카드 유사도는 두 집합 간의 겹침(overlap)을 측정합니다.

```
J(A, B) = |A intersect B| / |A union B|
```

0(겹침 없음)에서 1(동일한 집합)까지의 범위를 가집니다. 자카드 거리(Jaccard distance)는 1 - 자카드 유사도입니다.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

자카드를 사용할 때:
- 태그, 카테고리 또는 특징의 집합 비교
- 단어 존재 여부에 기반한 문서 유사도(빈도 아님)
- 준중복(near-duplicate) 탐지(자카드의 MinHash 근사)
- 이진 특징 벡터 비교(존재/부재 데이터)
- 분할 모델 평가(교집합 대 합집합(IoU) = 자카드)

### 편집 거리 (Levenshtein Distance)

편집 거리는 한 문자열을 다른 문자열로 변환하는 데 필요한 최소한의 단일 문자 연산 횟수를 세는 것입니다. 연산은 삽입, 삭제, 또는 치환입니다.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

동적 프로그래밍을 사용하여 계산합니다. 항목 (i, j)가 문자열 A의 첫 i개 문자와 문자열 B의 첫 j개 문자 사이의 편집 거리인 행렬을 채워 보세요.

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

편집 거리를 사용하는 경우:
- 철자 검사 및 교정
- DNA 서열 정렬 (가중 연산 사용)
- 퍼지 문자열 매칭
- 지저분한 텍스트 데이터의 중복 제거

### KL 발산 (거리가 아니지만, 거리처럼 사용됨)

KL 발산은 하나의 확률 분포가 다른 확률 분포와 얼마나 다른지를 측정합니다. 09강에서 다루지만, 거리처럼 사용되므로 이 논의에 포함됩니다.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

중요한 속성: KL 발산은 대칭이 아닙니다.

```
D_KL(P || Q) != D_KL(Q || P)
```

이는 거리 메트릭의 기본 요구 사항을 충족하지 못한다는 의미입니다. 삼각 부등식도 만족하지 않습니다. KL 발산은 발산이지, 거리가 아닙니다.

순방향 KL (D_KL(P || Q))은 "평균 추구"입니다: Q는 P의 모든 모드를 포함하려고 합니다.
역방향 KL (D_KL(Q || P))은 "모드 추구"입니다: Q는 P의 단일 모드에 집중합니다.

KL 발산을 볼 때:
- 변분 오토인코더 (VAE)(VAE (Variational Autoencoder)) (ELBO의 KL 항이 잠재 분포를 사전 분포로 밀어냄)
- 지식 증류(Knowledge Distillation) (학생 모델이 교사 모델의 분포를 맞추려고 함)
- RLHF (인간 피드백 기반 강화 학습)(RLHF (Reinforcement Learning from Human Feedback)) (KL 페널티가 미세 조정된 모델을 기본 모델에 가깝게 유지함)
- 정책 기울기 방법 (정책 업데이트를 제한함)

### 바서슈타인 거리 (Wasserstein Distance, Earth Mover's Distance)

바서슈타인 거리는 하나의 확률 분포를 다른 확률 분포로 변환하는 데 필요한 최소한의 "작업량"을 측정합니다. 하나의 분포가 흙더미이고 다른 분포가 구멍이라고 생각하세요. 얼마나 많은 흙을 얼마나 멀리 옮겨야 할까요?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

1D 분포의 경우, 누적 분포 함수의 절대 차분의 적분으로 단순화됩니다:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

바서슈타인 거리가 중요한 이유:
- 진정한 메트릭입니다 (대칭이며, 삼각 부등식을 만족함)
- 분포가 겹치지 않을 때에도 (KL 발산이 무한대로 갈 때) 기울기를 제공합니다
- 이 특성 덕분에 원본 GAN의 학습 불안정성을 해결한 Wasserstein GAN (WGAN)의 핵심이 되었습니다

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Wasserstein 거리를 사용할 때:
- GAN 학습 (WGAN, WGAN-GP)
- 겹치지 않을 수 있는 분포 비교
- 최적 수송 문제
- 이미지 검색 (색상 히스토그램 비교)

### 다양한 작업이 서로 다른 거리를 필요로 하는 이유

| 작업 | 최적의 거리 | 이유 |
|------|--------------|-----|
| 텍스트 유사도 | 코사인 | 크기는 잡음이고 방향이 의미입니다 |
| 이미지 픽셀 비교 | L2 | 공간적 관계가 중요하며, 특징은 비교 가능한 스케일입니다 |
| 희소 고차원 특징 | L1 | 강건하며, 드문 큰 차이를 증폭하지 않습니다 |
| 집합 겹침 (태그, 카테고리) | 자카드 | 데이터는 본질적으로 집합 값이며 벡터 값이 아닙니다 |
| 문자열 매칭 | 편집 거리 | 연산이 인간의 편집 직관에 대응합니다 |
| 이상치 탐지 | 마할라노비스 | 특징 간 상관관계와 스케일을 고려합니다 |
| 분포 비교 | KL 발산 | P 대신 Q를 사용함으로써 손실된 정보를 측정합니다 |
| GAN 학습 | Wasserstein | 분포가 겹치지 않을 때에도 기울기를 제공합니다 |
| 임베딩 (벡터 DB) | 코사인 또는 내적 | 임베딩은 방향에 의미를 인코딩하도록 학습됩니다 |
| 추천 | 내적 | 크기가 인기도나 신뢰도를 인코딩할 수 있습니다 |
| DNA 서열 | 가중 편집 거리 | 치환 비용은 뉴클레오타이드 쌍에 따라 다릅니다 |
| 제조 QC | L-무한대 | 모든 차원에서의 최악의 편차가 중요합니다 |

### 손실 함수와의 연결

손실 함수는 예측과 대상에 적용되는 거리 함수입니다.

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

정규화는 손실 함수에 가중치에 대한 노름 페널티를 추가합니다.

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

L1이 희소성을 생성하는 반면 L2가 생성하지 않는 이유: 2D 가중치 공간에서 제약 영역을 상상해 보세요. L1은 다이아몬드이고, L2는 원입니다. 손실 함수의 등고선(타원)은 다이아몬드의 모서리에서 가장 접촉할 가능성이 높으며, 이 모서리에서는 하나의 가중치가 0입니다. 원은 매끄러운 점에서 접촉하며, 이 점에서는 두 가중치 모두 0이 아닙니다.

### 최근접 이웃 검색

모든 거리 함수는 최근접 이웃 검색 문제를 내포합니다: 쿼리 포인트가 주어지면 데이터셋에서 가장 가까운 포인트를 찾는 문제입니다.

d개의 차원을 가진 n개의 포인트로 구성된 데이터셋에서 정확한 최근접 이웃 검색은 쿼리당 O(n * d)입니다. 대규모 데이터셋에서는 너무 느립니다.

근사 최근접 이웃 (ANN)(Approximate Nearest Neighbor (ANN)) 알고리즘은 작은 정확도 손실과 거대한 속도 향상 사이의 트레이드오프를 취합니다:

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

HNSW (Hierarchical Navigable Small World)는 현대 벡터 데이터베이스에서 지배적인 알고리즘입니다. 각 노드가 근사 최근접 이웃에 연결되는 다층 그래프를 구축합니다. 검색은 상층(희소하고 긴 점프)에서 시작하여 하층(밀집하고 짧은 점프)으로 내려갑니다.

```figure
norm-unit-balls
```

## 구현하기

### 1단계: 모든 노름 및 거리 함수

`code/distances.py`에서 완전한 구현을 참조하세요. 모든 함수는 기본 Python 수학만 사용하여 처음부터 구축됩니다.

### 2단계: 동일한 데이터, 다른 거리, 다른 이웃

`distances.py`의 데모는 데이터셋을 생성하고, 쿼리 포인트를 선택하며, 거리 메트릭에 따라 최근접 이웃이 어떻게 변하는지 보여줍니다. L1에서 "가장 가까운" 포인트가 L2나 코사인에서는 가장 가깝지 않을 수 있습니다.

### 3단계: 임베딩 유사성 검색

코드는 쿼리에 가장 유사한 "문서"를 코사인 유사성과 L2 거리를 사용하여 찾는 모의 임베딩 유사성 검색을 포함하며, 순위가 다를 수 있음을 보여줍니다.

## 사용하기

가장 일반적인 실용적 사용: 벡터 데이터베이스에서 유사한 항목을 찾는 것입니다.

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

`model.encode(text)`를 호출한 후 벡터 데이터베이스를 검색할 때 내부적으로 일어나는 일입니다. 임베딩 모델은 텍스트를 벡터로 매핑합니다. 벡터 데이터베이스는 쿼리 벡터와 저장된 모든 벡터 간의 코사인 유사성(또는 내적)을 계산하며, 모든 벡터를 검사하지 않도록 ANN 알고리즘을 사용합니다.

## 연습 문제

1. (1, 2, 3)과 (4, 0, 6) 사이의 L1, L2, L-무한(L-infinity) 거리를 계산해 보세요. 임의의 두 점 쌍에 대해 L-무한 <= L2 <= L1이 항상 성립함을 검증해 보세요. 이 순서가 보장되는 이유를 증명해 보세요.

2. 코사인 유사도가 높지만(> 0.9) L2 거리가 큰(> 10) 두 벡터를 만들어 보세요. 기하학적으로 어떤 일이 일어나고 있는지 설명해 보세요. 그 다음, 코사인 유사도가 낮지만(< 0.3) L2 거리가 작은(< 0.5) 두 벡터를 만들어 보세요.

3. 데이터셋과 쿼리 포인트를 입력받아 L1, L2, 코사인, 마할라노비스 거리(Mahalanobis distance) 기준으로 가장 가까운 이웃(nearest neighbor)을 반환하는 함수를 구현해 보세요. 네 가지 거리 측정 방식이 가장 가까운 점이 무엇인지 모두 불일치하는 데이터셋을 찾아보세요.

4. CDF 방법(CDF method)을 사용하여 [0.5, 0.5, 0, 0]과 [0, 0, 0.5, 0.5] 사이의 바서斯坦 거리(Wasserstein distance)를 손으로 계산해 보세요. 그 다음 [0.25, 0.25, 0.25, 0.25]와 [0, 0, 0.5, 0.5] 사이의 거리를 계산해 보세요. 어느 것이 더 크며, 그 이유는 무엇인가요?

5. 근사 자카드 유사도(approximate Jaccard similarity)를 위한 MinHash를 구현해 보세요. 100개의 랜덤 집합을 생성하고, 모든 쌍에 대해 정확한 자카드 유사도를 계산한 후, 50, 100, 200개의 해시 함수를 사용한 MinHash 근사치와 비교해 보세요. 근사 오차를 플롯(plot)해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 노름(Norm) | "벡터의 크기" | 벡터를 비음수 스칼라(non-negative scalar)로 매핑하는 함수로, 삼각 부등식(triangle inequality), 절대적 균질성(absolute homogeneity)을 만족하며, 영 벡터(zero vector)에서만 0이 되는 성질을 가짐 |
| L1 노름(L1 norm) | "맨해튼 거리(Manhattan distance)" | 절대값 성분의 합. 최적화 과정에서 희소성(sparsity)을 생성함. 이상치(outliers)에 강건함 |
| L2 노름(L2 norm) | "유클리드 거리(Euclidean distance)" | 제곱된 성분의 합에 대한 제곱근. 유클리드 공간에서의 직선 거리 |
| Lp 노름(Lp norm) | "일반화된 노름(Generalized norm)" | 절대값 성분의 p-th power의 합에 대한 p-th root. L01강 L2는 특수한 경우 |
| L-무한 노름(L-infinity norm) | "맥스 노름(Max norm)" 또는 "체비셰프 거리(Chebyshev distance)" | 최대 절대값 성분. p가 무한대로 접근할 때의 Lp의 극한(limit) |
| 코사인 유사도(Cosine similarity) | "벡터 사이의 각도" | 두 크기로 정규화된 내적(dot product). -1에서 +1까지의 범위를 가짐. 벡터 길이를 무시함 |
| 코사인 거리(Cosine distance) | "1 빼기 코사인 유사도" | 코사인 유사도를 거리로 변환함. 0에서 2까지의 범위를 가짐 |
| 내적 | "비정규화 코사인" | 성분별 곱의 합. 코사인 유사도에 두 크기를 곱한 값과 같습니다 |
| 마할라노비스 거리 | "상관 인식 거리" | 데이터 공분산 행렬을 사용하여 화이트닝(장식 및 정규화)된 공간에서의 L2 거리입니다 |
| 자카드 유사도 | "집합 겹침" | 교집합의 크기를 합집합의 크기로 나눈 값입니다. 벡터가 아닌 집합에 적용됩니다 |
| 편집 거리 | "레벤슈타인 거리" | 한 문자열을 다른 문자열로 변환하기 위해 필요한 최소한의 삽입, 삭제 및 치환 횟수입니다 |
| KL 발산 | "분포 간 거리" | 진정한 거리가 아닙니다(대칭이 아님). Q를 사용하여 P를 인코딩할 때 발생하는 추가 비트 수를 측정합니다 |
| 바서슈타인 거리 | "어스 무버 거리" | 한 분포에서 다른 분포로 질량을 이동하는 데 필요한 최소한의 작업량입니다. 진정한 메트릭입니다 |
| 근사 최근접 이웃 | "ANN 검색" | HNSW, LSH, IVF 등 정확한 검색보다 훨씬 빠르게 대략적으로 가장 가까운 점을 찾는 알고리즘입니다 |
| HNSW | "벡터 DB 알고리즘" | 계층적 내비게이블 스몰 월드 그래프. 빠른 근사 최근접 이웃 검색을 위한 다층 그래프입니다 |
| L1 정규화 | "라소" | 가중치의 L1 노름을 손실에 추가합니다. 가중치를 0으로 만듭니다(희소성) |
| L2 정규화 | "리지" 또는 "가중치 감쇠" | 가중치의 제곱된 L2 노름을 손실에 추가합니다. 희소성 없이 가중치를 0 쪽으로 축소합니다 |
| 엘라스틱 넷 | "L1 + L2" | L1 및 L2 정규화를 결합합니다. 상관된 특징 그룹을 단독으로 사용할 때보다 더 잘 처리합니다 |

## 추가 읽기

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta의 십억 규모 ANN 검색 라이브러리
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - GAN에 어스 무버 거리를 도입한 논문
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 기초적인 ANN 알고리즘
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, 임베딩의 기본값으로 코사인 유사도가 사용된 곳
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn에서 거리 메트릭 및 이웃 알고리즘에 대한 실용 가이드
