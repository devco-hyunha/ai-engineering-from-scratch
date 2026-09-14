# 벡터, 행렬과 연산 (Vectors, Matrices & Operations)

> 모든 신경망은 결국 행렬 곱셈에 몇 단계를 더한 것입니다.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- 원소별 연산, 행렬 곱, 전치, 행렬식, 역행렬을 갖춘 Matrix 클래스를 만듭니다
- 원소별 곱과 행렬 곱을 구분하고 각각이 언제 쓰이는지 설명합니다
- 처음부터 만든 Matrix 클래스만으로 단일 dense 신경망 레이어(`relu(W @ x + b)`)를 구현합니다
- 브로드캐스팅 규칙과 신경망 프레임워크에서 편향 덧셈이 어떻게 동작하는지 설명합니다

## 문제 상황 (The Problem)

신경망을 만들고 싶습니다. 코드를 읽으면 이런 줄이 나옵니다:

```
output = activation(weights @ input + bias)
```

그 `@`가 행렬 곱셈입니다. `weights`는 행렬이고 `input`은 벡터입니다. 이 연산들이 무엇을 하는지 모르면 이 줄은 마법입니다. 알면 레이어의 전체 순전파가 세 연산으로 보입니다.

모델이 처리하는 모든 이미지는 픽셀 값의 행렬입니다. 모든 단어 임베딩은 벡터입니다. 모든 신경망의 모든 레이어는 행렬 변환입니다. 변수를 이해하지 않고 코드를 쓸 수 없듯, 행렬 연산에 능숙하지 않으면 AI 시스템을 만들 수 없습니다.

이 레슨은 그 능숙함을 처음부터 만듭니다.

## 핵심 개념 (The Concept)

### 벡터: 순서가 있는 숫자 목록

벡터는 방향과 크기를 가진 숫자 목록입니다. AI에서 벡터는 데이터 포인트, 특성, 또는 파라미터를 나타냅니다.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 벡터 `[3, 4]`는 평면의 좌표 (3, 4)를 가리킵니다. 길이(크기)는 5입니다 (3-4-5 삼각형).

### 행렬: 숫자의 격자

행렬은 2D 격자입니다. 행과 열. m x n 행렬은 행 m개, 열 n개입니다.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

신경망에서 가중치 행렬은 입력 벡터를 출력 벡터로 바꿉니다. 입력이 784개, 출력이 128개인 레이어는 128x784 가중치 행렬을 씁니다.

### 왜 shape가 중요한가

행렬 곱에는 엄격한 규칙이 있습니다: `(m x n) @ (n x p) = (m x p)`. 안쪽 차원이 일치해야 합니다.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch에서 shape mismatch 오류가 나면 이유가 이것입니다.

### 연산 지도

| 연산 | 하는 일 | 신경망에서의 용도 |
|-----------|-------------|-------------------|
| 덧셈 | 원소별 결합 | 출력에 편향 더하기 |
| 스칼라 곱 | 모든 원소 스케일 | 학습률 * 기울기 |
| 행렬 곱 | 벡터 변환 | 레이어 순전파 |
| 전치 | 행과 열 뒤집기 | 역전파 |
| 행렬식 | 단일 숫자 요약 | 가역성 확인 |
| 역행렬 | 변환 되돌리기 | 선형 시스템 풀기 |
| 단위행렬 | 아무것도 안 하는 행렬 | 초기화, residual 연결 |

### 원소별 곱 vs 행렬 곱

이 구분은 초보자를 끊임없이 헷갈리게 합니다.

원소별: 같은 위치끼리 곱합니다. 두 행렬의 shape가 같아야 합니다.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

행렬 곱: 행과 열의 내적입니다. 안쪽 차원이 일치해야 합니다.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

다른 연산, 다른 결과, 다른 규칙입니다.

### 브로드캐스팅 (Broadcasting)

출력 행렬에 편향 벡터를 더할 때 shape가 맞지 않습니다. 브로드캐스팅이 더 작은 배열을 늘려 맞춥니다.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

모든 현대 프레임워크가 이를 자동으로 합니다. 이해하면 shape가 틀려 보이는데 코드가 돌아가는 상황이 덜 헷갈립니다.

```figure
vector-projection
```

## 구현하기 (Build It)

### Step 1: Vector 클래스

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: 핵심 연산을 갖춘 Matrix 클래스

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: 동작 확인

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: 신경망과 연결하기

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

이것이 단일 dense 레이어입니다: `output = relu(W @ x + b)`. 모든 신경망의 모든 dense 레이어가 정확히 이것을 합니다.

## 실용 활용 (Use It)

NumPy는 위의 모든 것을 더 짧은 줄로, 훨씬 빠르게 합니다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python의 `@` 연산자는 `__matmul__`을 호출합니다. NumPy는 C와 Fortran으로 작성된 최적화된 BLAS 루틴으로 구현합니다. 같은 수학, 100배 빠릅니다.

NumPy의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1D 편향을 두 행에 자동으로 브로드캐스트합니다. 모든 신경망 프레임워크에서 편향 덧셈이 이렇게 동작합니다.

## 배포할 산출물 (Ship It)

이 레슨은 기하적 직관으로 행렬 연산을 가르치는 프롬프트를 만듭니다. `outputs/prompt-matrix-operations.md`를 보세요.

여기서 만든 Matrix 클래스는 Phase 3, Lesson 10에서 만드는 미니 신경망 프레임워크의 기초입니다.

## 연습 문제 (Exercises)

1. **역행렬 검증.** `A @ A.inverse_2x2()`를 곱해 단위행렬이 나오는지 확인하세요. 서로 다른 2x2 행렬 세 개로 시도하세요. 행렬식이 0이면 어떻게 되나요?

2. **3x3 역행렬 구현.** Matrix 클래스를 확장해 수반행렬(adjugate) 방법으로 3x3 역행렬을 계산하세요. NumPy의 `np.linalg.inv`와 비교해 테스트하세요.

3. **두 레이어 네트워크 만들기.** Matrix 클래스만 사용해(NumPy 없이) 두 레이어 신경망을 만드세요: input (3) -> hidden (4) -> output (2). 랜덤 가중치를 초기화하고 순전파를 실행한 뒤 모든 shape가 맞는지 검증하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 벡터 (Vector) | "화살표" | 순서가 있는 숫자 목록. AI에서는 고차원 공간의 점. |
| 행렬 (Matrix) | "숫자 표" | 선형 변환. 한 공간의 벡터를 다른 공간으로 매핑. |
| 행렬 곱 | "그냥 숫자를 곱하면 됨" | 첫 행렬의 모든 행과 둘째 행렬의 모든 열 사이 내적. 순서가 중요함. |
| 전치 (Transpose) | "뒤집기" | 행과 열을 바꿈. m x n을 n x m으로. 역전파에서 핵심. |
| 행렬식 (Determinant) | "행렬에서 나온 어떤 숫자" | 행렬이 면적(2D) 또는 부피(3D)를 얼마나 스케일하는지. 0이면 변환이 한 차원을 뭉갬. |
| 역행렬 (Inverse) | "행렬 되돌리기" | 변환을 반대로 하는 행렬. 행렬식이 0이 아닐 때만 존재. |
| 단위행렬 (Identity) | "지루한 행렬" | 1을 곱하는 것과 같음. residual 연결(ResNets)에 사용. |
| 브로드캐스팅 | "마법처럼 shape 맞추기" | 더 작은 배열을 빠진 차원을 따라 반복해 더 큰 배열에 맞춤. |
| 원소별 (Element-wise) | "일반 곱셈" | 같은 위치끼리 곱함. 두 배열의 shape가 같거나 브로드캐스트 가능해야 함. |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 여기서 다룬 모든 연산에 대한 시각적 직관
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 규칙
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 특화 선형대수 요약 참고서
