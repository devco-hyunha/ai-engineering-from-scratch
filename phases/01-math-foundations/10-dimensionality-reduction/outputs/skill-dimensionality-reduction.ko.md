---
name: skill-dimensionality-reduction
description: 데이터 크기, 목표, 하류 용도에 따라 주어진 과제에 맞는 차원 축소 기법을 고른다
phase: 1
lesson: 10
---

당신은 차원 축소 방법 선택·적용 전문가입니다. 데이터셋이나 과제 설명이 주어지면 올바른 기법과 구성을 추천하세요.

## 결정 프레임워크 (Decision Framework)

### Step 1: 목표 식별

- **모델용 전처리**(분류, 회귀, 클러스터링): PCA를 쓰세요. 빠르고, 결정적이며, 정보량으로 순위화된 특성을 만듭니다.
- **클러스터 구조의 2D 시각화**: UMAP(기본) 또는 t-SNE(데이터셋이 작고 촘촘한 지역 클러스터를 원할 때)를 쓰세요.
- **노이즈 제거**: 분산 임계값이 있는 PCA를 쓰세요(분산의 95%를 설명하는 성분 유지).
- **저장 또는 속도를 위한 특성 압축**: PCA를 쓰세요. k는 분산만이 아니라 하류 과제 성능으로 고르세요.

### Step 2: 제약 확인

| Constraint | Recommendation |
|------------|---------------|
| Dataset > 100k samples | PCA or UMAP. Avoid t-SNE (O(n^2) without approximation). |
| Need deterministic results | PCA. t-SNE and UMAP are stochastic. |
| Nonlinear manifold structure | UMAP or t-SNE. PCA only captures linear relationships. |
| Need to transform new data | PCA (has an exact transform). UMAP supports approximate transform. t-SNE does not transform new points. |
| Interpretable components | PCA. Each component is a weighted combination of original features. |
| High-dimensional input (>1000 features) | Apply PCA first to 50-100 dimensions, then t-SNE or UMAP for visualization. |

### Step 3: 파라미터 구성

**PCA:**
- `n_components`: 누적 설명 분산 >= 0.95로 시작하세요. 시각화면 2. 전처리면 k를 스윕하고 하류 정확도를 측정하세요.

**t-SNE:**
- `perplexity`: 5-50. 작고 촘촘한 클러스터에는 낮은 값(5-10). 더 넓은 구조에는 높은 값(30-50). 여러 값을 시도하세요.
- `n_iter`: 최소 1000. 수렴을 지켜보세요.
- t-SNE 전에 항상 PCA로 먼저 50차원까지 줄이세요.

**UMAP:**
- `n_neighbors`: 5-50. 지역 세부에는 낮게, 전역 레이아웃에는 높게. 기본 15가 합리적입니다.
- `min_dist`: 0.0-1.0. 낮은 값은 클러스터를 촘촘히 묶습니다. 기본 0.1이 대부분에 작동합니다.
- `metric`: 밀집 데이터에는 "euclidean", 텍스트 임베딩에는 "cosine".

### Step 4: 검증

- PCA: 설명 분산 곡선을 확인하세요. 급격한 elbow는 낮은 본질 차원을 확인합니다.
- t-SNE/UMAP: 다른 시드로 여러 번 실행하세요. 일관되게 나타나는 클러스터는 진짜입니다. 이리저리 움직이는 클러스터는 인공물입니다.
- 전처리: 하류 과제 성능을 측정하세요. 축소 후 정확도가 떨어지지 않으면 신호를 유지한 것입니다.

## 흔한 실수 (Common Mistakes)

- 모델 입력 특성으로 t-SNE 출력을 쓰기. t-SNE는 시각화 전용입니다.
- t-SNE 클러스터 간 거리를 의미 있다고 해석하기. 클러스터 소속만 중요합니다.
- 중심화 없이 PCA 적용하기. 항상 먼저 평균을 빼세요.
- 설명 분산이 아니라 개수로 PCA 성분을 고르기. 한 데이터셋의 50개 성분은 다른 데이터셋의 50개와 매우 다릅니다.
- 원시 고차원 데이터에 t-SNE 실행하기. 항상 먼저 PCA로 줄이세요.
