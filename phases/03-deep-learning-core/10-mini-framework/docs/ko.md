# 나만의 미니 프레임워크 만들기 (Build Your Own Mini Framework)

> 뉴런, 레이어, 네트워크, 역전파, 활성화, 손실 함수, 옵티마이저, 정규화, 초기화, 학습률 스케줄까지 모두 만들었고, 전부 따로 떨어져 있었습니다. 이제 프레임워크로 한곳에 묶습니다. PyTorch도 TensorFlow도 아닙니다. 당신 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** All of Phase 03 (Lessons 01-09)
**Time:** ~120 minutes

## 학습 목표 (Learning Objectives)

- Module, Linear, ReLU, Sigmoid, Dropout, BatchNorm, Sequential, 손실 함수, 옵티마이저, DataLoader가 포함된 완전한 딥러닝 프레임워크(~500줄)를 구축합니다
- Module 추상화(forward, backward, parameters)를 설명하고, train/eval 모드 전환이 왜 필요한지 설명합니다
- 모든 구성 요소를 연결해 원 분류(circle classification)에서 4층 네트워크를 학습하는 동작하는 학습 루프를 만듭니다
- 프레임워크의 각 구성 요소를 PyTorch 대응물(nn.Module, nn.Sequential, optim.Adam, DataLoader)에 매핑합니다

## 문제 상황 (The Problem)

열 개 레슨의 빌딩 블록이 서로 다른 파일에 흩어져 있습니다. 여기엔 `Value` 클래스, 저기엔 학습 루프, 다른 파일엔 가중치 초기화, 또 다른 파일엔 학습률 스케줄. 네트워크를 학습하려면 다섯 개 레슨에서 복사해 붙여넣고 손으로 연결해야 합니다.

프레임워크가 해결하는 문제가 바로 이것입니다. PyTorch는 `nn.Module`, `nn.Sequential`, `optim.Adam`, `DataLoader`, 그리고 이들을 묶는 학습 루프 패턴을 줍니다. TensorFlow는 `keras.Layer`, `keras.Sequential`, `keras.optimizers.Adam`을 줍니다. 마법이 아닙니다. 배관을 매번 다시 만들지 않고도 네트워크를 정의·학습·평가할 수 있게 하는 조직화 패턴입니다.

약 500줄의 Python으로 같은 것을 만듭니다. numpy 없음. 외부 의존성 없음. 어떤 피드포워드 네트워크든 정의하고, SGD 또는 Adam으로 학습하고, 데이터를 배치하고, 드롭아웃과 배치 정규화를 적용하고, 어떤 활성화든 쓰고, 학습률을 스케줄할 수 있는 프레임워크입니다.

끝나면 `model = nn.Sequential(...)`를 PyTorch에서 쓸 때 무슨 일이 일어나는지 정확히 알게 됩니다. `model.train()`과 `model.eval()`이 왜 있는지, `optimizer.zero_grad()`가 왜 별도 호출인지. 전부 직접 만들었기 때문에 이해합니다.

## 핵심 개념 (The Concept)

### Module 추상화 (The Module Abstraction)

PyTorch의 모든 레이어는 `nn.Module`을 상속합니다. Module에는 세 가지 책임이 있습니다.

1. **forward()** -- 입력이 주어졌을 때 출력을 계산
2. **parameters()** -- 학습 가능한 모든 가중치를 반환
3. **backward()** -- 기울기 계산 (PyTorch에서는 autograd가 처리, 우리 구현에서는 명시적)

Linear 레이어는 Module입니다. ReLU 활성화도 Module입니다. Dropout도, BatchNorm도 Module입니다. 모두 같은 인터페이스를 가집니다.

### Sequential 컨테이너 (Sequential Container)

`nn.Sequential`은 Module을 체인으로 연결합니다. 순전파: Module 1 → Module 2 → Module 3. 역전파: 체인을 반대로. 컨테이너 자체도 Module입니다 — forward(), parameters(), backward()를 가집니다. 이것이 합성 패턴(composite pattern)입니다. Module 시퀀스 자체가 Module입니다.

### 학습 모드 vs 평가 모드 (Training vs Evaluation Mode)

Dropout은 학습 중에는 뉴런을 무작위로 끄지만, 평가 중에는 모두 통과시킵니다. BatchNorm은 학습 중에는 배치 통계를, 평가 중에는 이동 평균을 씁니다. `train()`과 `eval()`이 이 동작을 전환합니다. 모든 Module에는 `training` 플래그가 있습니다.

### 옵티마이저 (Optimizer)

옵티마이저는 기울기를 사용해 파라미터를 갱신합니다. SGD: `param -= lr * grad`. Adam: 모멘텀과 분산 추정치를 유지한 뒤 갱신합니다. 옵티마이저는 네트워크 구조를 모릅니다 — 평평한 파라미터 목록과 그 기울기만 봅니다.

### DataLoader

배칭이 중요한 이유는 두 가지입니다. 첫째, 큰 문제에서는 전체 데이터셋을 메모리에 올릴 수 없습니다. 둘째, 미니배치 경사하강법의 노이즈가 지역 최솟값 탈출에 도움이 됩니다. DataLoader는 데이터를 배치로 나누고, 선택적으로 에포크마다 섞습니다.

### 프레임워크 아키텍처 (Framework Architecture)

```mermaid
graph TD
    subgraph "모듈 (Modules)"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>무작위 제로 마스크"]
        BatchNorm["BatchNorm<br/>활성화 정규화"]
    end

    subgraph "컨테이너 (Containers)"
        Sequential["Sequential<br/>모듈 체인"]
    end

    subgraph "손실 함수 (Loss Functions)"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>이진 교차 엔트로피"]
    end

    subgraph "옵티마이저 (Optimizers)"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>적응적 모멘트"]
    end

    subgraph "데이터 (Data)"
        DataLoader["DataLoader<br/>배칭 + 셔플"]
    end

    Sequential --> |"포함"| Linear
    Sequential --> |"포함"| ReLU
    Sequential --> |"순전파/역전파"| MSE
    SGD --> |"갱신"| Sequential
    DataLoader --> |"공급"| Sequential
```

### 학습 루프 (Training Loop)

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant M as 모델
    participant L as 손실
    participant O as 옵티마이저

    loop 각 에포크
        DL->>M: 입력 배치
        M->>M: 순전파 (레이어별)
        M->>L: 예측
        L->>L: 손실 계산
        L->>M: 역전파 (기울기)
        M->>O: 파라미터 + 기울기
        O->>M: 갱신된 파라미터
        O->>O: 기울기 초기화
    end
```

### Module 계층 (Module Hierarchy)

```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

```figure
gradient-clipping
```

## 직접 만들기 (Build It)

### Step 1: Module 베이스 클래스

모든 레이어가 구현하는 추상 인터페이스입니다.

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### Step 2: Linear 레이어

기본 빌딩 블록입니다. 가중치와 편향을 저장하고, 순전파에서 Wx + b를, 역전파에서 가중치/입력 기울기를 계산합니다.

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### Step 3: 활성화 Module

ReLU, Sigmoid, Tanh를 Module로. 각각 역전파에 필요한 값을 캐시합니다.

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### Step 4: Dropout Module

학습 중에는 요소를 무작위로 0으로 만듭니다. 남은 요소는 1/(1-p)로 스케일해 기댓값을 유지합니다. eval에서는 아무것도 하지 않습니다.

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### Step 5: BatchNorm Module

배치에 걸쳐 특성별로 활성화를 평균 0, 분산 1로 정규화합니다. eval 모드용 이동 통계를 유지합니다.

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### Step 6: Sequential 컨테이너

모듈을 체인으로 연결합니다. 순전파는 왼쪽→오른쪽, 역전파는 오른쪽→왼쪽입니다.

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### Step 7: 손실 함수

MSE와 이진 교차 엔트로피. 각각 손실 값을 반환하고, 기울기를 반환하는 backward()를 제공합니다.

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### Step 8: SGD와 Adam 옵티마이저

둘 다 파라미터 목록을 받아 기울기로 가중치를 갱신합니다.

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### Step 9: DataLoader

데이터를 배치로 나누고, 선택적으로 에포크마다 섞습니다.

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### Step 10: 원 분류에서 4층 네트워크 학습

모두 연결합니다. 모델을 정의하고, 손실과 옵티마이저를 고르고, 학습 루프를 돌립니다.

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## 활용하기 (Use It)

방금 만든 것의 PyTorch 대응입니다.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

구조는 동일합니다. `Sequential`, `Linear`, `ReLU`, `Sigmoid`, `BCELoss`, `Adam`, `zero_grad`, `backward`, `step`, `train`, `eval`. 모든 개념이 일대일로 대응합니다. 차이는 PyTorch가 autograd를 자동으로 처리하고(각 모듈에서 backward()를 구현할 필요 없음), GPU에서 돌며, 수년간 최적화됐다는 점입니다. 뼈대는 같습니다.

이제 PyTorch 코드를 보면 매 줄에서 무슨 일이 일어나는지 정확히 압니다. 그 이해가 이 레슨의 전부입니다.

## 산출물 (Ship It)

이 레슨이 만드는 것:
- `outputs/prompt-framework-architect.md` -- 프레임워크 추상화를 사용해 신경망 아키텍처를 설계하는 프롬프트

## 연습 문제 (Exercises)

1. 다중 클래스 분류용 `SoftmaxCrossEntropyLoss` 클래스를 추가하세요. 예측에 Softmax를 적용하고, 교차 엔트로피 손실을 계산하고, 결합된 역전파를 처리하세요. 3클래스 나선 데이터셋에서 테스트하세요.

2. 옵티마이저에 학습률 스케줄링을 구현하세요. `set_lr()` 메서드를 추가하고 Lesson 09의 cosine 스케줄을 연결하세요. warmup + cosine으로 원 분류기를 학습하고 상수 LR과 비교하세요.

3. Sequential에 `save()`와 `load()` 메서드를 추가해 모든 가중치를 JSON 파일로 직렬화하고 다시 로드하세요. 로드한 모델이 원본과 같은 예측을 내는지 검증하세요.

4. Adam 옵티마이저에 weight decay(L2 정규화)를 구현하세요. 매 step마다 가중치를 0 쪽으로 줄이는 `weight_decay` 파라미터를 추가하세요. decay=0과 decay=0.01로 학습을 비교하세요.

5. 샘플별 학습 루프를 올바른 미니배치 기울기 누적으로 바꾸세요. 배치의 모든 샘플에 걸쳐 기울기를 누적한 뒤, 배치 크기로 나누고 옵티마이저 step을 한 번만 하세요. 수렴 속도가 바뀌는지 측정하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|----------------|----------------------|
| Module | "레이어" | 프레임워크의 기본 추상화 — forward(), backward(), parameters()를 가진 모든 것 |
| Sequential | "레이어를 순서대로 쌓기" | 모듈을 체인으로 연결하는 컨테이너. 순전파는 순서대로, 역전파는 역순으로 |
| Forward pass | "네트워크 실행" | 입력을 각 모듈에 순서대로 통과시켜 출력을 계산하는 것 |
| Backward pass | "기울기 계산" | 손실 기울기를 각 모듈에 역순으로 전파해 파라미터 기울기를 계산하는 것 |
| Parameters | "학습 가능한 가중치" | 옵티마이저가 갱신할 수 있는 네트워크의 모든 값 — 가중치와 편향 |
| Optimizer | "가중치를 갱신하는 것" | 기울기로 파라미터를 갱신하는 알고리즘. SGD, Adam 등 |
| DataLoader | "데이터를 공급하는 것" | 데이터셋을 배치로 나누고, 선택적으로 에포크마다 섞는 이터레이터 |
| Training mode | "model.train()" | 드롭아웃·배치 통계 기반 BatchNorm 같은 확률적 동작을 켜는 플래그 |
| Evaluation mode | "model.eval()" | 드롭아웃을 끄고 BatchNorm에 이동 통계를 쓰는 플래그 |
| Zero grad | "기울기 지우기" | 다음 배치의 기울기를 계산하기 전에 모든 파라미터 기울기를 0으로 리셋 |

## 더 읽을거리 (Further Reading)

- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019) -- PyTorch 설계 결정을 설명하는 논문
- Chollet, "Deep Learning with Python, Second Edition" (2021) -- 3장에서 같은 module/layer 추상화로 Keras 내부를 다룸
- Johnson, "Tiny-DNN" (https://github.com/tiny-dnn/tiny-dnn) -- 프레임워크 내부를 이해하기 위한 헤더 온리 C++ 딥러닝 프레임워크
