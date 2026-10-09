# 벡터, 행렬 및 연산

> 모든 신경망은 추가 단계가 있는 행렬 곱셈일 뿐입니다.

**유형:** Build
**언어:** Python, Julia
**선수 요건:** 1단계, 01강 (선형 대수 직관)
**시간:** 약 60분

## 학습 목표

- 요소별 연산, 행렬 곱셈, 전치, 행렬식, 역행렬을 수행하는 Matrix 클래스를 구축해 보세요
- 요소별 곱셈과 행렬 곱셈을 구분하고, 각각이 적용되는 시점을 설명해 보세요
- 처음부터 작성한 Matrix 클래스만 사용하여 단일 밀집 신경망 레이어(`relu(W @ x + b)`)를 구현해 보세요
- 브로드캐스팅 규칙과 신경망 프레임워크에서 편향(bias) 추가가 작동하는 방식을 설명해 보세요

## 문제점

신경망을 구축하고 싶다고 가정해 보세요. 코드를 읽으면 다음과 같은 내용이 보입니다:

```
output = activation(weights @ input + bias)
```

`@`는 행렬 곱셈입니다. `weights`는 행렬이며, `input`는 벡터입니다. 이 연산들이 무엇을 하는지 모른다면, 이 줄은 마법처럼 보일 것입니다. 만약 알고 있다면, 이는 세 가지 연산으로 이루어진 레이어의 전체 순방향 전파(forward pass)입니다.

모델이 처리하는 모든 이미지는 픽셀 값의 행렬입니다. 모든 워드 임베딩은 벡터입니다. 모든 신경망의 모든 레이어는 행렬 변환입니다. 변수를 이해하지 못하면 코드를 작성할 수 없는 것과 마찬가지로, 행렬 연산에 능숙하지 못하면 AI 시스템을 구축할 수 없습니다.

이 강의는 그 능숙함을 처음부터 구축합니다.

## 개념

### 벡터: 순서가 있는 숫자 목록

벡터는 방향과 크기를 가진 숫자의 목록입니다. AI에서는 벡터가 데이터 포인트, 특징(features), 또는 매개변수를 나타냅니다.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 벡터 `[3, 4]`는 평면 위의 좌표 (3, 4)를 가리킵니다. 그 길이(크기)는 5입니다 (3-4-5 삼각형).

### 행렬: 숫자의 격자

행렬은 2D 격자입니다. 행과 열로 구성됩니다. m x n 행렬은 m개의 행과 n개의 열을 가집니다.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

신경망에서는 가중치 행렬이 입력 벡터를 출력 벡터로 변환합니다. 입력이 784개이고 출력이 128개인 레이어는 128x784 크기의 가중치 행렬을 사용합니다.

### 형상(shape)이 중요한 이유

행렬 곱셈에는 엄격한 규칙이 있습니다: `(m x n) @ (n x p) = (m x p)`. 내차원(inner dimensions)이 일치해야 합니다.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch에서 shape mismatch 오류가 발생한다면, 이것이 그 이유입니다.

### 연산 매핑

| 연산 | 수행 내용 | 신경망 사용처 |
|-----------|-------------|-------------------|
| 덧셈 | 요소별 결합 | 출력에 편향(bias) 더하기 |
| 스칼라 곱 | 모든 요소 스케일링 | 학습률 * 기울기 |
| 행렬 곱 | 벡터 변환 | 레이어 순전파 |
| 전치(Transpose) | 행과 열 뒤집기 | 역전파(Backpropagation) |
| 행렬식(Determinant) | 단일 숫자 요약 | 가역성(invertibility) 확인 |
| 역행렬(Inverse) | 변환 되돌리기 | 선형 시스템 풀기 |
| 단위 행렬(Identity) | 아무것도 하지 않는 행렬 | 초기화, 잔여 연결(residual connections) |

### 요소별 곱 vs 행렬 곱

이 구분은 초보자를 항상 헷갈리게 합니다.

요소별 곱: 일치하는 위치를 곱합니다. 두 행렬의 shape가 동일해야 합니다.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

행렬 곱: 행과 열의 내적(dot product)입니다. 내차원이 일치해야 합니다.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

연산이 다르면, 결과가 다르고, 규칙도 다릅니다.

### 브로드캐스팅(Broadcasting)

출력 행렬에 편향 벡터를 더할 때, shape가 일치하지 않습니다. 브로드캐스팅은 더 작은 배열을 늘려 맞춰줍니다.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

모든 최신 프레임워크는 이를 자동으로 수행합니다. 이를 이해하면 shape가 맞지 않는 것 같아도 코드가 실행되는 상황에서 혼란을 방지할 수 있습니다.

```figure
vector-projection
```

## 구현하기

### 1단계: Vector 클래스

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

### 2단계: 핵심 연산이 포함된 Matrix 클래스

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

### 3단계: 동작 확인

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

### 4단계: 신경망과 연결

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

이것은 단일 dense layer입니다: `output = relu(W @ x + b)`. 모든 신경망의 모든 dense layer는 정확히 이 작업을 수행합니다.

## 사용하기

NumPy는 위 모든 작업을 더 짧은 코드로, 몇 배 더 빠르게 수행합니다.

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

Python의 `@` 연산자는 `__matmul__`을 호출합니다. NumPy는 C와 Fortran으로 작성된 최적화된 BLAS 루틴으로 이를 구현합니다. 같은 연산이 100배 더 빠릅니다.

NumPy에서의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1D 편향을 두 행에 걸쳐 자동으로 브로드캐스트합니다. 모든 신경망 프레임워크에서 편향 추가가 작동하는 방식입니다.

## 출시하기

이 강의는 기하학적 직관을 통해 행렬 연산을 가르치는 프롬프트를 생성합니다. `outputs/prompt-matrix-operations.md`를 참조하세요.

여기서 구축한 Matrix 클래스는 3단계, 10강에서 구축하는 미니 신경망 프레임워크의 기초가 됩니다.

## 연습 문제

1. **역행렬 검증.** `A @ A.inverse_2x2()`를 곱하여 단위 행렬(identity matrix)이 나오는지 확인하세요. 세 가지 서로 다른 2x2 행렬로 시도해 보세요. 행렬식이 0일 때는 어떻게 되나요?

2. **3x3 역행렬 구현.** Matrix 클래스를 확장하여 수반 행렬(adjugate) 방법을 사용해 3x3 행렬의 역행렬을 계산하세요. NumPy의 `np.linalg.inv`와 비교하여 테스트하세요.

3. **두 층 네트워크 구축.** Matrix 클래스만 사용(NumPy 사용 금지)하여 두 층 신경망을 만드세요: 입력 (3) -> 은닉층 (4) -> 출력 (2). 랜덤 가중치를 초기화하고, 순전파를 실행하여 모든 형태(shape)가 올바른지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 벡터(Vector) | "화살표" | 숫자의 순서 있는 목록. AI에서는 고차원 공간의 점입니다. |
| 행렬(Matrix) | "숫자의 표" | 선형 변환입니다. 벡터를 한 공간에서 다른 공간으로 매핑합니다. |
| 행렬 곱(Matrix multiply) | "숫자를 그냥 곱하는 것" | 첫 번째 행렬의 모든 행과 두 번째 행렬의 모든 열 사이의 내적(dot product)입니다. 순서가 중요합니다. |
| 전치(Transpose) | "뒤집기" | 행과 열을 교환합니다. m x n 행렬을 n x m으로 만듭니다. 역전파(backpropagation)에서 중요합니다. |
| 행렬식(Determinant) | "행렬에서 나온 어떤 숫자" | 행렬이 면적(2D)이나 부피(3D)를 얼마나 확장하는지 측정합니다. 0이면 변환이 차원을 압축합니다. |
| 역행렬(Inverse) | "행렬을 되돌리기" | 변환을 되돌리는 행렬입니다. 행렬식이 0이 아닐 때만 존재합니다. |
| 단위 행렬(Identity matrix) | "지루한 행렬" | 1을 곱하는 것과 동일한 행렬입니다. 잔여 연결(residual connections, ResNets)에서 사용됩니다. |
| 브로드캐스트(Broadcasting) | "마법 같은 형태 수정" | 더 작은 배열을 반복하여 더 큰 배열의 누락된 차원에 맞춰 늘리는 것입니다. |
| 요소별 | "일반적인 곱셈" | 일치하는 위치를 곱합니다. 두 배열 모두 동일한 모양이어야 하거나 브로드캐스팅이 가능해야 합니다. |

## 추가 읽기

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 여기서 다루는 모든 연산에 대한 시각적 직관
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 규칙
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 특화 선형 대수를 위한 간결한 참고 자료
