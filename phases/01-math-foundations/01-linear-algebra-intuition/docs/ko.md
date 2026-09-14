# 선형대수 직관 (Linear Algebra Intuition)

> 모든 AI 모델은 결국 행렬 연산을 멋진 모자로 가린 것에 불과합니다.

**Type:** Learn
**Languages:** Python, Julia
**Prerequisites:** Phase 0
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- Python으로 벡터·행렬 연산(덧셈, 내적, 행렬 곱)을 처음부터 직접 구현합니다
- 내적, 사영, 그람-슈미트(Gram-Schmidt) 과정이 기하적으로 무엇을 하는지 설명합니다
- 행 연산을 이용해 벡터 집합의 선형 독립성, 계수(rank), 기저(basis)를 판별합니다
- 선형대수 개념을 AI 응용(임베딩, 어텐션 점수, LoRA)과 연결합니다

## 문제 상황 (The Problem)

ML 논문을 아무거나 펼쳐 보세요. 첫 페이지 안에 벡터, 행렬, 내적, 변환이 나옵니다. 선형대수 직관이 없으면 이것들은 그냥 기호입니다. 직관이 있으면 신경망이 실제로 무엇을 하는지 — 공간에서 점들을 어떻게 옮기는지 — 보이게 됩니다.

수학자가 될 필요는 없습니다. 이 연산들이 기하적으로 무엇을 의미하는지 보고, 직접 코드로 짜면 됩니다.

## 핵심 개념 (The Concept)

### 벡터는 점이다 (그리고 방향이다)

벡터는 그냥 숫자의 목록입니다. 하지만 그 숫자들에는 의미가 있습니다 — 공간에서의 좌표입니다.

**2D 벡터 [3, 2]:**

| x | y | 점 |
|---|---|-------|
| 3 | 2 | 원점 (0,0)에서 평면의 (3, 2)를 가리키는 벡터 |

이 벡터의 크기는 sqrt(3^2 + 2^2) = sqrt(13)이며, 오른쪽 위 방향을 가리킵니다.

AI에서 벡터는 모든 것을 표현합니다:
- 단어 → 768개 숫자로 된 벡터 (임베딩 공간에서의 "의미")
- 이미지 → 수백만 픽셀 값의 벡터
- 사용자 → 선호도를 담은 벡터

### 행렬은 변환이다

행렬은 한 벡터를 다른 벡터로 바꿉니다. 회전, 스케일, 늘리기, 사영이 가능합니다.

```mermaid
graph LR
    subgraph Before["변환 전"]
        A["점 A"]
        B["점 B"]
    end
    subgraph Matrix["행렬 곱셈"]
        M["M (변환)"]
    end
    subgraph After["변환 후"]
        A2["점 A'"]
        B2["점 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

AI에서 행렬은 곧 모델입니다:
- 신경망 가중치 → 입력을 출력으로 바꾸는 행렬
- 어텐션 점수 → 어디에 집중할지 결정하는 행렬
- 임베딩 → 단어를 벡터로 매핑하는 행렬

### 내적은 유사도를 측정한다

두 벡터의 내적은 서로 얼마나 비슷한지를 알려줍니다.

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

검색 엔진, 추천 시스템, RAG가 실제로 하는 일이 이것입니다 — 내적이 큰 벡터를 찾습니다.

### 선형 독립성 (Linear Independence)

집합 안의 어떤 벡터도 다른 벡터들의 조합으로 쓸 수 없을 때, 그 벡터들은 선형 독립입니다. v1, v2, v3가 독립이면 3차원 공간을 생성합니다. 하나가 나머지의 조합이면 평면만 생성합니다.

AI에서 중요한 이유: 특성 행렬의 열은 선형 독립이어야 합니다. 두 특성이 완전히 상관되어 있으면(선형 종속), 모델은 각각의 효과를 구분할 수 없습니다. 회귀에서 다중공선성(multicollinearity)이 생기고 — 가중치 행렬이 불안정해져 작은 입력 변화에도 출력이 크게 흔들립니다.

**구체적 예:**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1과 v2는 독립입니다 — 어느 쪽도 서로의 스칼라 배수나 조합이 아닙니다. 하지만 v3 = 2*v1 + v2이므로 {v1, v2, v3}는 종속 집합입니다. 세 벡터는 모두 xy-평면에 있습니다. 어떻게 조합해도 [0, 0, 1]에 도달할 수 없습니다. 벡터는 세 개지만 자유도는 2차원뿐입니다.

데이터셋에서: feature_3 = 2*feature_1 + feature_2라면, feature_3를 추가해도 모델에 새 정보는 없습니다. 더 나쁘게는 정규방정식이 특이(singular)해져 — 가중치의 유일한 해가 없습니다.

### 기저와 계수 (Basis and Rank)

기저(basis)는 전체 공간을 생성하는 선형 독립 벡터의 최소 집합입니다. 기저 벡터의 개수가 공간의 차원입니다.

3D 공간의 표준 기저는 {[1,0,0], [0,1,0], [0,0,1]}입니다. 하지만 3D에서 독립인 세 벡터라면 어떤 것이든 유효한 기저입니다. 기저 선택은 좌표계 선택입니다.

행렬의 계수(rank) = 선형 독립인 열의 개수 = 선형 독립인 행의 개수. rank < min(rows, cols)이면 행렬은 계수 부족(rank-deficient)입니다. 이는 다음을 의미합니다:
- 해가 무한히 많거나(또는 없음)
- 변환에서 정보가 손실됨
- 행렬을 역행렬로 만들 수 없음

| 상황 | 계수 | ML에서의 의미 |
|-----------|------|---------------------|
| Full rank (rank = min(m, n)) | 가능한 최댓값 | 유일한 최소제곱해가 존재. 모델이 잘 조건화됨. |
| Rank deficient (rank < min(m, n)) | 최댓값 미만 | 특성이 중복. 가중치 해가 무한. 정규화 필요. |
| Rank 1 | 1 | 모든 열이 한 벡터의 스케일 복사. 데이터가 직선 위에 있음. |
| Near rank-deficient (작은 특이값) | 수치적으로 낮음 | 행렬이 불량 조건. 작은 입력 잡음이 큰 출력 변화를 유발. SVD 절단 또는 ridge 회귀 사용. |

### 사영 (Projection)

벡터 **a**를 벡터 **b**에 사영하면, **a** 중 **b** 방향 성분을 얻습니다:

```
proj_b(a) = (a dot b / b dot b) * b
```

잔차(a - proj_b(a))는 b에 수직입니다. 이 직교 분해가 최소제곱 적합의 기초입니다.

사영은 ML 어디에나 있습니다:
- 선형 회귀는 관측에서 열공간까지의 거리를 최소화합니다 — 해가 곧 사영입니다
- PCA는 데이터를 분산이 최대인 방향으로 사영합니다
- 트랜스포머의 어텐션은 쿼리를 키에 사영한 결과를 계산합니다

```mermaid
graph LR
    subgraph Projection["a를 b에 사영"]
        direction TB
        O["원점"] --> |"b (방향)"| B["b"]
        O --> |"a (원래)"| A["a"]
        O --> |"proj_b(a)"| P["사영"]
        A -.-> |"잔차 (수직)"| P
    end
```

**예:** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

사영은 y성분을 버립니다. 가장 단순한 형태의 차원 축소입니다 — 관심 없는 방향을 버리는 것.

### 그람-슈미트 과정 (Gram-Schmidt Process)

독립인 벡터 집합을 정규직교 기저(orthonormal basis)로 바꿉니다. 정규직교란 모든 벡터의 길이가 1이고, 모든 쌍이 서로 수직임을 뜻합니다.

알고리즘:
1. 첫 벡터를 취해 정규화합니다
2. 두 번째 벡터를 취해 첫 벡터로의 사영을 빼고 정규화합니다
3. 세 번째 벡터를 취해 이전 모든 벡터로의 사영을 빼고 정규화합니다
4. 나머지 벡터에 대해 반복합니다

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

이것이 QR 분해가 내부적으로 동작하는 방식입니다. Q는 정규직교 기저, R은 사영 계수를 담습니다. QR 분해는 다음에 쓰입니다:
- 선형 시스템 풀기 (가우스 소거보다 안정적)
- 고유값 계산 (QR 알고리즘)
- 최소제곱 회귀 (표준 수치 방법)

```figure
eigen-directions
```

## 구현하기 (Build It)

### Step 1: 처음부터 벡터 만들기 (Python)

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### Step 2: 처음부터 행렬 만들기 (Python)

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### Step 3: AI에서 왜 중요한가

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### Step 4: Julia 버전

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### Step 5: 선형 독립성과 사영을 처음부터 (Python)

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 실용 활용 (Use It)

이제 NumPy로 같은 일을 합니다 — 실무에서 실제로 쓰는 방식입니다:

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### NumPy로 계수, 사영, QR

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch — 텐서는 자동미분을 가진 벡터

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

x에 대한 내적의 기울기는 그냥 y입니다. PyTorch가 이를 자동으로 계산했습니다. 신경망의 모든 연산은 이런 연산 — 행렬 곱, 내적, 사영 — 으로 구성되며, 자동미분이 그 전부의 기울기를 추적합니다.

방금 NumPy가 한 줄로 하는 일을 처음부터 만들었습니다. 이제 내부에서 무엇이 일어나는지 압니다.

## 배포할 산출물 (Ship It)

이 레슨이 만드는 것:
- `outputs/prompt-linear-algebra-tutor.md` — 기하적 직관으로 선형대수를 가르치는 AI 어시스턴트용 프롬프트

## 연결 (Connections)

이 레슨의 모든 개념은 현대 AI의 특정 부분과 연결됩니다:

| 개념 | 어디서 등장하는가 |
|---------|------------------|
| 내적 (Dot product) | 트랜스포머의 어텐션 점수, RAG의 코사인 유사도 |
| 행렬 곱 | 모든 신경망 레이어, 모든 선형 변환 |
| 선형 독립성 | 특성 선택, 다중공선성 회피 |
| 계수 (Rank) | 시스템 가해 여부 판별, LoRA (저계수 적응) |
| 사영 (Projection) | 선형 회귀 (열공간으로의 사영), PCA |
| Gram-Schmidt / QR | 수치 솔버, 고유값 계산 |
| 정규직교 기저 | 안정적인 수치 계산, whitening 변환 |

LoRA는 특별히 언급할 가치가 있습니다. 큰 언어 모델을 미세 조정할 때 가중치 업데이트를 저계수 행렬로 분해합니다. 4096x4096 가중치 행렬(1,600만 파라미터)을 갱신하는 대신, LoRA는 4096x16과 16x4096 두 행렬(13만 1천 파라미터)을 갱신합니다. rank-16 제약은 가중치 업데이트가 전체 4096차원 공간의 16차원 부분공간에 산다고 가정합니다. 선형대수가 실제로 일을 하는 모습입니다.

## 연습 문제 (Exercises)

1. 두 벡터 사이의 각도를 도(degree)로 반환하는 `Vector.angle_between(other)`를 구현하세요
2. x좌표를 2배, y좌표를 3배로 만드는 2D 스케일 행렬을 만들고, 벡터 [1, 1]에 적용하세요
3. 차원 50인 단어처럼 생긴 랜덤 벡터 5개를 주고, 코사인 유사도로 가장 비슷한 두 개를 찾으세요
4. Gram-Schmidt 출력이 정말로 정규직교인지 검증하세요: 모든 쌍의 내적이 0이고 모든 벡터의 크기가 1인지 확인
5. 계수가 2인 3x3 행렬을 만드세요. `rank()` 메서드로 검증한 뒤, 열들이 생성하는 기하적 대상이 무엇인지 설명하세요
6. 벡터 [1, 2, 3]을 [1, 1, 1]에 사영하세요. 결과가 기하적으로 무엇을 의미하나요?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 벡터 (Vector) | "화살표" | n차원 공간의 점 또는 방향을 나타내는 숫자 목록 |
| 행렬 (Matrix) | "숫자 표" | 한 공간의 벡터를 다른 공간으로 매핑하는 변환 |
| 내적 (Dot product) | "곱해서 더하기" | 두 벡터가 얼마나 정렬되었는지 측정 — 유사도 검색의 핵심 |
| 임베딩 (Embedding) | "AI 마법" | 무언가(단어, 이미지, 사용자)의 의미를 나타내는 벡터 |
| 선형 독립성 | "겹치지 않는다" | 집합의 어떤 벡터도 나머지의 조합으로 쓸 수 없음 |
| 계수 (Rank) | "차원이 몇인가" | 행렬에서 선형 독립인 열(또는 행)의 개수 |
| 사영 (Projection) | "그림자" | 한 벡터가 다른 벡터 방향에 갖는 성분 |
| 기저 (Basis) | "좌표축" | 공간을 생성하는 독립 벡터의 최소 집합 |
| 정규직교 (Orthonormal) | "서로 수직인 단위 벡터" | 서로 수직이고 각각 길이가 1인 벡터들 |
