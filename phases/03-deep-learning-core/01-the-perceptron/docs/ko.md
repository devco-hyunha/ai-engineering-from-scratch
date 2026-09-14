# 퍼셉트론 (The Perceptron)

> 퍼셉트론은 신경망의 원자입니다. 쪼개 보면 가중치, 편향, 그리고 결정이 나옵니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- Python으로 퍼셉트론을 처음부터 구현하고, 가중치 갱신 규칙과 계단 활성화 함수를 포함합니다
- 단일 퍼셉트론이 선형 분리 가능한 문제만 풀 수 있는 이유를 설명하고, XOR 실패 사례를 시연합니다
- OR, NAND, AND 게이트를 조합해 다층 퍼셉트론을 구성하고 XOR을 풉니다
- 시그모이드 활성화와 역전파로 2층 네트워크를 학습시켜 XOR을 자동으로 학습합니다

## 문제 상황 (The Problem)

벡터와 내적을 알고 있습니다. 행렬이 입력을 출력으로 변환한다는 것도 압니다. 그런데 기계는 *어떤* 변환을 쓸지 어떻게 *학습*할까요?

퍼셉트론이 그 답을 줍니다. 가장 단순한 학습 기계입니다. 입력을 받아 가중치를 곱하고, 편향을 더한 뒤 이진 결정을 내립니다. 그다음 조정합니다. 그게 전부입니다. 지금까지 만든 모든 신경망은 이 아이디어를 층으로 쌓은 것입니다.

퍼셉트론을 이해한다는 것은 코드에서 "학습"이 실제로 무엇을 뜻하는지 이해하는 것입니다. 출력이 현실과 맞을 때까지 숫자를 조정하는 일입니다.

## 핵심 개념 (The Concept)

### 뉴런 하나, 결정 하나

퍼셉트론은 n개의 입력을 받아 각각에 가중치를 곱하고, 합한 뒤 편향을 더하고, 결과를 활성화 함수에 통과시킵니다.

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["편향"] --> sum
    sum --> step["step(z)"]
    step --> out["출력 (0 또는 1)"]
```

계단 함수는 단호합니다. 가중 합에 편향을 더한 값이 >= 0이면 출력 1, 아니면 출력 0입니다.

```
step(z) = 1  if z >= 0
           0  if z < 0
```

이것은 선형 분류기입니다. 가중치와 편향이 입력 공간을 두 영역으로 나누는 직선(고차원에서는 초평면)을 정의합니다.

### 결정 경계 (The Decision Boundary)

입력이 두 개일 때, 퍼셉트론은 2D 공간에 직선을 그립니다.

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

직선 한쪽은 출력 0, 다른 쪽은 출력 1입니다. 학습은 클래스를 올바르게 분리할 때까지 이 직선을 움직입니다.

### 학습 규칙 (The Learning Rule)

퍼셉트론 학습 규칙은 단순합니다.

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

예측이 맞으면 error = 0이라 아무 것도 바뀌지 않습니다. 0을 예측했는데 1이어야 하면 가중치가 커집니다. 1을 예측했는데 0이어야 하면 가중치가 작아집니다. 학습률은 각 조정의 크기를 제어합니다.

### XOR 문제 (The XOR Problem)

여기서 깨집니다. 논리 게이트를 보세요.

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND와 OR는 선형 분리 가능합니다. 0과 1을 가르는 직선을 하나 그을 수 있습니다. XOR는 아닙니다. [0,1]과 [1,0]을 [0,0]과 [1,1]에서 가르는 단일 직선은 없습니다.

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

이것은 근본적 한계입니다. 단일 퍼셉트론은 선형 분리 가능한 문제만 풀 수 있습니다. Minsky와 Papert가 1969년에 이를 증명했고, 신경망 연구가 거의 10년 동안 멈출 뻔했습니다.

해결책: 퍼셉트론을 층으로 쌓습니다. 다층 퍼셉트론은 두 선형 결정을 비선형 하나로 합쳐 XOR을 풀 수 있습니다.

```figure
perceptron-boundary
```

## 직접 만들기 (Build It)

### Step 1: Perceptron 클래스

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### Step 2: 논리 게이트로 학습

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### Step 3: XOR이 실패하는 모습 보기

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

절대 수렴하지 않습니다. 단일 퍼셉트론이 XOR을 학습할 수 없다는 단단한 증명입니다.

### Step 4: 두 층으로 XOR 풀기

요령: XOR = (x1 OR x2) AND NOT (x1 AND x2). 퍼셉트론 세 개를 합칩니다.

```mermaid
graph LR
    x1["x1"] --> OR["OR 뉴런"]
    x1 --> NAND["NAND 뉴런"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND 뉴런"]
    NAND --> AND
    AND --> out["출력"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

네 경우 모두 맞습니다. 퍼셉트론을 층으로 쌓으면 단일 퍼셉트론이 만들 수 없는 결정 경계가 생깁니다.

### Step 5: 2층 네트워크 학습시키기

Step 4는 가중치를 손으로 고정했습니다. XOR에는 되지만, 미리 맞는 가중치를 모르는 실제 문제에는 안 됩니다. 해결책: 계단 함수를 시그모이드로 바꾸고, 역전파 (backpropagation)로 가중치를 자동 학습합니다.

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

Step 4와의 핵심 차이 두 가지입니다. 첫째, 시그모이드가 계단 함수를 대체합니다. 매끄러워서 기울기가 존재합니다. 둘째, `train` 메서드가 출력층에서 은닉층으로 오차를 뒤로 전파하며, 오차에 대한 기여도에 비례해 모든 가중치를 조정합니다. 그게 20줄짜리 역전파입니다.

이것이 Lesson 03으로 가는 다리입니다. `d_output`과 `hidden_deltas` 뒤의 수학은 네트워크 그래프에 적용한 연쇄법칙입니다. 거기서 제대로 유도합니다.

## 활용하기 (Use It)

방금 처음부터 만든 모든 것이 import 한 줄에 있습니다.

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

다섯 줄입니다. 30줄짜리 `Perceptron` 클래스와 같은 일을 합니다. sklearn 버전은 수렴 검사, 여러 손실 함수, 희소 입력 지원을 더하지만, 핵심 루프는 동일합니다. 가중 합, 계단 함수, 오차에 따른 가중치 갱신.

진짜 차이는 규모에서 드러납니다. 프로덕션 네트워크에서 바뀌는 것:

- 계단 함수가 시그모이드, ReLU, 또는 다른 매끄러운 활성화가 됩니다
- 가중치는 역전파로 자동 학습됩니다 (Lesson 03)
- 층이 깊어집니다: 3, 10, 100+ 층
- 원리는 같습니다. 각 층이 이전 층 출력으로부터 새 특성을 만듭니다

단일 퍼셉트론은 직선만 그릴 수 있습니다. 쌓으면 어떤 모양도 그릴 수 있습니다.

## 산출물 (Ship It)

이 레슨이 만드는 것:
- `outputs/skill-perceptron.md` - 단일층 vs 다층 아키텍처가 언제 필요한지 다루는 스킬

## 연습 문제 (Exercises)

1. NAND 게이트(만능 게이트 — 어떤 논리 회로든 NAND로 만들 수 있음)로 퍼셉트론을 학습시키세요. 가중치와 편향이 유효한 결정 경계를 이루는지 확인하세요.
2. Perceptron 클래스를 수정해 매 에폭마다 결정 경계(w1*x1 + w2*x2 + b = 0)를 추적하세요. AND 게이트 학습 중 직선이 어떻게 이동하는지 출력하세요.
3. 입력 3개 중 최소 2개가 1일 때만 1을 출력하는 퍼셉트론(다수결 함수)을 만드세요. 선형 분리 가능한가요? 이유는?

## 핵심 용어 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Perceptron (퍼셉트론) | "가짜 뉴런" | 선형 분류기: 입력과 가중치의 내적에 편향을 더한 뒤 계단 함수를 통과 |
| Weight (가중치) | "입력이 얼마나 중요한지" | 각 입력이 결정에 기여하는 정도를 스케일하는 곱셈 계수 |
| Bias (편향) | "임계값" | 결정 경계를 이동시키는 상수. 입력이 전부 0이어도 퍼셉트론이 발화할 수 있게 함 |
| Activation function (활성화 함수) | "값을 찌그러뜨리는 것" | 가중 합 뒤에 적용하는 함수. 퍼셉트론은 계단, 현대 네트워크는 시그모이드/ReLU |
| Linearly separable (선형 분리 가능) | "그 사이에 직선을 그을 수 있다" | 단일 초평면으로 클래스를 완벽히 나눌 수 있는 데이터셋 |
| XOR problem (XOR 문제) | "퍼셉트론이 못 하는 것" | 단일층 네트워크가 비선형 분리 불가 함수를 학습할 수 없다는 증명 |
| Decision boundary (결정 경계) | "분류기가 바뀌는 지점" | 입력 공간을 두 클래스로 나누는 초평면 w*x + b = 0 |
| Multi-layer perceptron (다층 퍼셉트론) | "진짜 신경망" | 층으로 쌓인 퍼셉트론. 각 층의 출력이 다음 층의 입력이 됨 |

## 더 읽을거리 (Further Reading)

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- 이 모든 것을 시작한 원논문
- Minsky & Papert, "Perceptrons" (1969) -- 단일층 네트워크로 XOR이 풀리지 않음을 증명하고 퍼셉트론 연구를 10년 가까이 멈춘 책
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- 무료 온라인, 퍼셉트론이 네트워크로 조합되는 방식을 가장 잘 시각적으로 설명
