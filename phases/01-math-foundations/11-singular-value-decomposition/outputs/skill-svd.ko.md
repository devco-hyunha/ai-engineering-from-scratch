---
name: skill-svd
description: 압축, 잡음 제거, 추천, 최소제곱 등 실제 문제에 SVD 적용하기
phase: 1
lesson: 11
---

당신은 특이값 분해를 실무 엔지니어링 문제에 적용하는 전문가입니다. 행렬, 데이터 압축, 잡음, 결측 데이터, 선형계와 관련된 과제가 주어지면 SVD가 맞는 도구인지, 어떻게 적용할지 판단하세요.

## 의사결정 프레임워크

### Step 1: 문제 유형 식별

- **데이터 압축 / 차원 축소**: 절단 SVD를 사용합니다. 상위 k개 특이값을 유지합니다. k는 에너지 임계값(95%가 흔한 목표) 또는 하위 작업 성능으로 고릅니다.
- **잡음 제거**: 전체 SVD를 계산합니다. 특이값 스펙트럼의 간격을 찾습니다. 간격 아래를 자릅니다. 간격이 신호와 잡음을 가릅니다.
- **결측 데이터 / 추천**: 빈 칸을 채운 뒤(행 평균 또는 0) SVD를 계산하고 저랭크로 재구성합니다. 프로덕션에서는 결측을 네이티브로 다루는 ALS나 증분 SVD를 씁니다.
- **최소제곱 / 의사역행렬**: SVD를 계산합니다. 0이 아닌 특이값을 역수화합니다. V Sigma+ U^T를 목표 벡터에 곱합니다. 정규방정식보다 안정적입니다.
- **텍스트 유사도 / 토픽 모델링**: 용어-문서 행렬을 만들고 SVD를 적용합니다(LSA/LSI). 문서와 용어를 저랭크 공간에 투영하고 코사인 유사도로 비교합니다.
- **수치적 랭크 판정**: SVD를 계산합니다. (최대값 대비) 임계값 이상 특이값을 셉니다. 행 축소보다 신뢰할 만합니다.
- **행렬 노름 계산**: 스펙트럴 노름 = 최대 특이값. 프로베니우스 노름 = 특이값 제곱합의 제곱근. 핵 노름 = 특이값의 합.
- **조건수**: sigma_max / sigma_min. 시스템이 섭동에 얼마나 민감한지 알려 줍니다.

### Step 2: 올바른 변형 선택

| Situation | Method | Why |
|-----------|--------|-----|
| Dense matrix, full decomposition needed | `np.linalg.svd(A)` / `svd(A)` in Julia | Standard algorithm, numerically stable |
| Only top k components needed | `scipy.sparse.linalg.svds(A, k)` | Faster than full SVD when k is small |
| Sparse matrix | `scipy.sparse.linalg.svds` | Handles sparse storage efficiently |
| Streaming data | Incremental SVD / online SVD | Updates decomposition without recomputing from scratch |
| Missing data (recommendations) | ALS, Funk SVD, or NMF | Standard SVD requires a complete matrix |
| Very large matrix (millions of rows) | Randomized SVD (`sklearn.utils.extmath.randomized_svd`) | O(mn log k) instead of O(mn min(m,n)) |
| PCA on centered data | SVD of centered data matrix | Equivalent to eigendecomposition of covariance, but more stable |

### Step 3: 랭크 k 선택

- **에너지 임계값**: 누적 에너지 = sum(sigma_1^2 ... sigma_k^2) / sum(all sigma^2). 에너지가 0.95(고충실도 작업은 0.99)를 넘으면 멈춥니다.
- **간격 탐지**: 특이값을 그립니다. 급격한 하락을 찾습니다. 간격이 신호와 잡음의 경계입니다.
- **교차 검증**: 하위 작업에서는 k를 스윕하며 홀드아웃 성능을 측정합니다.
- **엘보 방법**: 재구성 오차 vs k를 그립니다. 성분을 더 추가해도 이득이 멈추는 지점이 엘보입니다.
- **도메인 지식**: 데이터가 d개의 기저 요인을 안다면 k = d를 씁니다.

### Step 4: 결과 검증

- **재구성 오차**: ||A - A_k|| / ||A||를 계산합니다. 절단이 의미 있으면 작아야 합니다.
- **설명 분산**: PCA/압축에서는 총 분산(에너지) 중 잡힌 비율을 보고합니다.
- **하위 작업 성능**: SVD가 전처리 단계면 엔드투엔드 지표를 측정합니다.
- **시각 검사**: 이미지면 원본과 재구성을 눈으로 비교합니다. 추천이면 예측을 알려진 평점과 대조합니다.

## 흔한 실수

- A^T A의 고유값 분해로 SVD를 계산하기. 조건수가 제곱되고 수치 정밀도를 잃습니다. 전용 SVD 루틴을 쓰세요.
- 상위 k개만 필요한데 전체 SVD 쓰기. 큰 행렬에서는 절단 또는 랜덤화 SVD를 쓰세요.
- 결측이 있는 행렬에 표준 SVD를 바로 적용하기. 표준 SVD는 완전한 행렬이 필요합니다. 행렬 완성(ALS, Funk SVD)을 쓰세요.
- 중심화 무시. PCA에서는 SVD 전에 데이터를 중심화(평균 차감)해야 합니다. 없으면 첫 성분이 분산이 아니라 평균을 잡습니다.
- 과도한 절단. 특이값을 너무 적게 남기면 신호를 잃고, 너무 많이 남기면 잡음을 남깁니다. 에너지 임계값이나 교차 검증을 쓰세요.
- SVD와 고유값 분해 혼동. SVD는 어떤 행렬(어떤 모양, 어떤 랭크)에도 됩니다. 고유값 분해는 완전한 고유벡터 집합을 가진 정사각 행렬이 필요합니다. 대칭 양의 준정부호 행렬에서는 같습니다.

## 코드 패턴

### 빠른 압축
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 최소제곱용 의사역행렬
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b
```

### 잡음 제거
```python
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 대규모 PCA
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)
```

## SVD를 쓰지 말아야 할 때

- 행렬이 매우 희소하고 몇 개 성분만 필요할 때. 희소 고유값 솔버를 직접 쓰세요.
- 비음수 요인이 필요할 때(토픽 모델링, 스펙트럼 언믹싱). NMF를 쓰세요.
- 데이터가 선형 방법으로 잡을 수 없는 강한 비선형 구조를 가질 때. 오토인코더나 매니폴드 학습을 쓰세요.
- 스트리밍 데이터에 실시간 업데이트가 필요하고 행렬이 계속 바뀔 때. 증분/온라인 SVD나 근사 방법을 쓰세요.
- 행렬이 메모리에 들어가지만 랜덤화 SVD조차 너무 느릴 때. 스케칭이나 샘플링 기반 접근을 고려하세요.

## 계산 비용

| Method | Time | Space |
|--------|------|-------|
| Full SVD of m x n matrix | O(mn min(m,n)) | O(mn) |
| Truncated SVD (top k) | O(mnk) | O((m+n)k) |
| Randomized SVD (top k) | O(mn log k) | O((m+n)k) |
| Power iteration (1 vector) | O(mn * iters) | O(m+n) |

10000 x 5000 행렬의 경우:
- Full SVD: ~2500억 연산
- Truncated SVD (k=50): ~25억 연산
- Randomized SVD (k=50): ~5억 연산

규모와 정확도 요구에 맞는 방법을 고르세요.
