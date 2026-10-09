# JAX 소개

> PyTorch는 텐서를 변경합니다. TensorFlow는 그래프를 구축합니다. JAX는 순수 함수를 컴파일합니다. 마지막 특징이 딥러닝을 생각하는 방식을 바꿉니다.

**유형:** Build
**언어:** Python
**선수 요건:** 03단계 01-10강, NumPy 기초
**시간:** 약 90분

## 학습 목표

- JAX의 함수형 API(jax.numpy, jax.grad, jax.jit, jax.vmap)를 사용하여 순수 함수 기반 신경망 코드를 작성해 보세요
- PyTorch의 즉시 변경(eager mutation)과 JAX의 함수형 컴파일 모델 간의 주요 설계 차이를 설명해 보세요
- 단순한 Python 코드와 비교하여 학습 루프를 가속화하기 위해 jit 컴파일과 vmap 벡터화를 적용해 보세요
- JAX에서 간단한 네트워크를 학습하고, PyTorch의 객체 지향적 접근 방식과 대비하여 명시적인 상태 관리를 비교해 보세요

## 문제점

PyTorch로 신경망을 구축하는 방법을 알고 있습니다. `nn.Module`을 정의하고, `.backward()`을 호출하며, 옵티마이저를 한 단계 진행합니다. 잘 작동합니다. 수백만 명이 이 방식을 사용합니다.

하지만 PyTorch에는 DNA에 내장된 제약이 있습니다. 연산을 즉시(eager) 방식으로 하나씩 Python에서 추적합니다. 모든 `tensor + tensor`은 별도의 커널 실행입니다. 모든 학습 단계는 동일한 Python 코드를 다시 해석합니다. 5,400억 매개변수 모델을 2,048개의 TPU에서 학습해야 할 때까지는 잘 작동합니다. 그때 오버헤드가 문제를 일으킵니다.

Google DeepMind는 Gemini를 JAX로 학습합니다. Anthropic은 Claude를 JAX로 학습했습니다. 이는 작은 작업이 아닙니다. 지구상에서 가장 큰 신경망 학습 실행입니다. JAX를 선택한 이유는 학습 루프를 Python 호출의 연속이 아닌, 컴파일 가능한 프로그램으로 취급하기 때문입니다.

JAX는 세 가지 초능력을 가진 NumPy입니다. 자동 미분, XLA로의 JIT 컴파일, 그리고 자동 벡터화입니다. 하나의 예제를 처리하는 함수를 작성합니다. JAX는 배치 처리, 그래디언트 계산, 기계어 컴파일, 다중 장치 실행을 수행하는 함수를 제공합니다. 원본 함수를 변경하지 않고도 가능합니다.

## 개념

### JAX 철학

JAX는 함수형 프레임워크입니다. 클래스가 없고, 변경 가능한 상태가 없으며, `.backward()` 메서드가 없습니다. 대신:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 클래스는 상태를 가짐 | 순수 함수: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 즉시 실행 | XLA를 통한 JIT 컴파일 |
| `for x in batch:` 수동 루프 | `jax.vmap(f)` 자동 벡터화 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 자동 병렬화 |
| 변경 가능한 `model.parameters()` | 변경 불가능한 배열의 pytree |

이것은 스타일 선호도가 아닙니다. 컴파일러 제약 조건입니다. JIT 컴파일은 순수 함수를 요구합니다 -- 동일한 입력은 항상 동일한 출력을 생성해야 하며, 사이드 이펙트가 없어야 합니다. 이 제약이 100배 속도 향상을 가능하게 합니다.

### jax.numpy: 익숙한 인터페이스

JAX는 NumPy API를 가속기에서 재구현합니다:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

동일한 함수 이름. 동일한 브로드캐스팅 규칙. 동일한 슬라이싱 의미론. 하지만 배열은 GPU/TPU에 위치하며, 모든 연산은 컴파일러가 추적할 수 있습니다.

중요한 차이점 하나: JAX 배열은 변경 불가능합니다. `a[0] = 5`는 없습니다. 대신: `a = a.at[0].set(5)`를 사용합니다. 처음 일주일 동안은 어색하게 느껴지지만, 곧 이해가 됩니다 -- 변경 불가능성은 `grad`, `jit`, `vmap`와 같은 변환을 합성 가능하게 만드는 핵심입니다.

### jax.grad: 함수적 자동 미분

PyTorch는 텐서에 기울기를 연결합니다 (`.grad`). JAX는 함수에 기울기를 연결합니다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 기울기를 계산하는 새로운 함수를 반환합니다. `.backward()` 호출이 필요 없습니다. 텐서에 저장된 계산 그래프도 없습니다. 기울기는 단순히 호출, 합성, JIT 컴파일할 수 있는 또 다른 함수일 뿐입니다.

이것은 임의로 합성됩니다:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2차 미분. 3차 미분. 야코비안. 헤시안. 모두 `grad`를 합성하여 수행합니다. PyTorch도 이를 수행할 수 있습니다 (`torch.autograd.functional.hessian`) 하지만 이는 부속적으로 추가된 기능입니다. JAX에서는 이것이 기본입니다.

제약 조건: `grad`는 순수 함수에서만 작동합니다. 내부에 print 문이 있으면 안 됩니다 (추적 중에 실행되며, 실행 중에는 실행되지 않습니다). 외부 상태 변경이 없어야 합니다. 명시적인 키 관리 없이는 난수 생성이 불가능합니다.

### jit: XLA로 컴파일

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 번째 호출 시 JAX는 함수를 추적(trace)합니다 -- 연산이 기록되지만, 실행되지는 않습니다. 이후 그 추적(trace)을 XLA (Accelerated Linear Algebra), 즉 TPU와 GPU용 Google의 컴파일러에 전달합니다. XLA는 연산을 융합(fuse)하고, 중복된 메모리 복사를 제거하며, 최적화된 기계어 코드를 생성합니다.

이후 호출은 Python을 완전히 건너뜁니다. 컴파일된 코드는 C++ 속도로 가속기에서 실행됩니다.

JIT가 도움이 되는 경우:
- 학습 단계 (수천 번 반복되는 동일한 연산)
- 추론 (동일한 모델, 서로 다른 입력)
- 유사한 형태의 입력으로 한 번 이상 호출되는 모든 함수

JIT가 해가 되는 경우:
- 값에 의존하는 Python 제어 흐름이 있는 함수 (`if x > 0`, 여기서 x는 추적된 배열)
- 단일 실행 연산 (컴파일 오버헤드가 실행 시간을 초과함)
- 디버깅 (추적(trace)이 실제 실행을 숨김)

제어 흐름 제한은 실제적입니다. `jax.lax.cond`는 `if/else`을 대체합니다. `jax.lax.scan`는 `for` 루프를 대체합니다. 이는 선택 사항이 아닙니다 -- 컴파일의 대가입니다.

### vmap: 자동 벡터화

하나의 예제를 처리하는 함수를 작성합니다:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`는 이를 배치(batch)를 처리하도록 승격합니다:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`는 다음을 의미합니다: `params`에 대해 배치하지 마세요 (공유됨), `x`의 축 0에 대해 배치하세요. 수동 `for` 루프가 없습니다. 리셰이핑이 없습니다. 배치 차원 스레딩이 없습니다. JAX는 배치 차원을 파악하고 전체 연산을 벡터화합니다.

이는 구문적 설탕(syntactic sugar)이 아닙니다. `vmap`는 Python 루프보다 10-100배 빠르게 실행되는 융합된 벡터화 코드를 생성합니다. 그리고 `jit` 및 `grad`과 결합됩니다:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 기울기. 한 줄로. PyTorch에서는 해킹(hacks) 없이 거의 불가능합니다.

### pmap: 장치 간 데이터 병렬화

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`는 함수를 모든 가용 장치(GPU/TPU)에 복제하고 배치를 분할합니다. 함수 내부에서 `jax.lax.pmean` 및 `jax.lax.psum`는 장치 간 기울기를 동기화합니다.

Google은 `pmap` (및 그 후속 `shard_map`)를 사용하여 수천 개의 TPU v5e 칩에 걸쳐 Gemini를 학습합니다. 프로그래밍 모델: 단일 장치 버전을 작성하고, `pmap`로 감싸면 끝입니다.

### Pytree: 범용 데이터 구조

JAX는 "pytree"를 다룹니다. pytree는 리스트, 튜플, dict, 배열의 중첩 조합입니다. 모델 매개변수는 pytree입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

모든 JAX 변환 -- `grad`, `jit`, `vmap` -- 은 pytree를 순회하는 방법을 알고 있습니다. `jax.tree.map(f, tree)`은 모든 리프에 `f`을 적용합니다. 옵티마이저가 모든 매개변수를 한 번에 업데이트하는 방식입니다:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드가 없습니다. 매개변수 등록도 없습니다. 트리 구조가 곧 모델입니다.

### 함수형 vs 객체 지향

PyTorch는 객체 내부에 상태를 저장합니다:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 상태를 사용하는 순수 함수를 사용합니다:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

매개변수는 인자로 전달됩니다. 아무것도 저장되지 않고, 아무것도 변경되지 않습니다. 덕분에 모든 함수가 테스트 가능하고, 조합 가능하며, 컴파일 가능합니다. 또한 매개변수를 직접 관리하거나 Flax나 Equinox 같은 라이브러리를 사용해야 합니다.

### JAX 생태계

JAX는 기본 요소(primitives)를 제공합니다. 라이브러리는 사용 편의성(ergonomics)을 제공합니다:

| 라이브러리 | 역할 | 스타일 |
|---------|------|-------|
| **Flax** (Google) | 신경망 레이어 | `nn.Module` + 명시적 상태 |
| **Equinox** (Patrick Kidger) | 신경망 레이어 | Pytree 기반, Python 스타일 |
| **Optax** (DeepMind) | 옵티마이저 + 학습률 스케줄 | 조합 가능한 기울기 변환 |
| **Orbax** (Google) | 체크포인팅 | pytree 저장/복원 |
| **CLU** (Google) | 지표 + 로깅 | 학습 루프 유틸리티 |

Optax는 표준 옵티마이저 라이브러리입니다. 기울기 변환(Adam, SGD, 클리핑)과 매개변수 업데이트를 분리하여 조합이 쉽습니다:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorch 사용 시점

| 요소 | JAX | PyTorch |
|--------|-----|---------|
| TPU 지원 | 일급 지원 (Google이 둘 다 개발) | 커뮤니티 유지보수 (torch_xla) |
| GPU 지원 | 좋음 (XLA를 통한 CUDA) | 최고 수준 (네이티브 CUDA) |
| 디버깅 | 어려움 (추적 + 컴파일) | 쉬움 (eager, 라인별) |
| 생태계 | 연구 중심 (Flax, Equinox) | 방대 (HuggingFace, torchvision 등) |
| 채용 | 니치 (Google/DeepMind/Anthropic) | 주류 (전 분야) |
| 대규모 학습 | 우수 (XLA, pmap, mesh) | 좋음 (FSDP, DeepSpeed) |
| 프로토타이핑 속도 | 느림 (함수적 오버헤드) | 빠름 (변경 후 실행) |
| 프로덕션 추론 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 사용처 | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

솔직한 답변: JAX를 사용할 특별한 이유가 없는 한 PyTorch를 사용하세요. 그 이유는 -- TPU 접근, 예제별 기울기 필요, 대규모 다중 장치 학습, Google/DeepMind/Anthropic에서의 작업입니다.

### JAX의 난수

JAX는 전역 난수 상태를 가지고 있지 않습니다. 모든 난수 연산은 명시적인 PRNG 키가 필요합니다:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음에는 번거로울 수 있습니다. 하지만 이는 장치 및 컴파일 간 재현성을 보장합니다 -- PyTorch의 `torch.manual_seed`이 다중 GPU 환경에서 보장할 수 없는 속성입니다.

```figure
batchnorm-effect
```

## 구현하기

### 1단계: 설정 및 데이터

JAX와 Optax를 사용하여 MNIST에 3층 MLP를 학습합니다. 입력은 784개, 은닉층은 256개와 128개 뉴런을 가진 두 층, 출력은 10개 클래스입니다.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 2단계: 매개변수 초기화

클래스가 없습니다. pytree를 반환하는 함수만 있습니다:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 초기화를 수동으로 수행합니다. 하나의 시드에서 분할된 세 개의 PRNG 키를 사용합니다. 모든 가중치는 중첩된 dict 내의 불변 배열입니다.

### 3단계: 순전파

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

순수 함수입니다. 매개변수를 입력하고 예측을 출력합니다. `self`이 없고, 저장된 상태도 없습니다. `loss_fn`은 교차 엔트로피를 처음부터 계산합니다 -- softmax, log, 음의 평균.

### 4단계: JIT 컴파일된 학습 단계

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`은 손실 값과 기울기를 한 번의 패스로 반환합니다. `@jax.jit` 데코레이터는 두 함수를 XLA로 컴파일합니다. 첫 번째 호출 이후, 각 학습 단계는 Python을 건드리지 않고 실행됩니다.

### 5단계: 학습 루프

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 에포크. 테스트 정확도는 약 97%입니다. 첫 번째 에포크는 느립니다 (JIT 컴파일). 2-10 에포크는 빠릅니다.

누락된 것을 주목하세요: `.zero_grad()`이 없고, `.backward()`이 없고, `.step()`이 없습니다. 전체 업데이트는 하나의 합성된 함수 호출입니다. 기울기는 계산되고, Adam에 의해 변환되며, 매개변수에 적용됩니다 -- 모두 `train_step` 내부에서 이루어집니다.

## 사용하기

### Flax: Google 표준

Flax는 가장 일반적인 JAX 신경망 라이브러리입니다. `nn.Module`을 다시 추가하지만, 명시적인 상태 관리를 사용합니다:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch와 동일한 구조이지만, `params`는 모델과 분리되어 있습니다. `model.init()`이 매개변수를 생성합니다. `model.apply(params, x)`가 순방향 전파를 실행합니다. 모델 객체에는 상태가 없습니다.

### Equinox: Python다운 대안

Equinox (Patrick Kidger 제작)는 모델을 pytree로 표현합니다:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree입니다. `.apply()`이 필요하지 않습니다. 매개변수는 모델의 리프(leaf)일 뿐입니다. 이는 JAX가 생각하는 방식에 더 가깝습니다.

### Optax: 조합 가능한 옵티마이저

Optax는 기울기 변환을 업데이트와 분리합니다:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

기울기 클리핑, 학습률 워밍업, 가중치 감쇠 -- 모두 변환의 체인으로 조합됩니다. 각 변환은 기울기를 보고, 수정하며, 다음 변환으로 전달합니다. 거대한 단일 옵티마이저 클래스가 없습니다.

## 출시하기

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원을 위해:

```bash
pip install jax[cuda12]
```

TPU (Google Cloud)를 위해:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 주의 사항:**

- 첫 JIT 호출은 느립니다 (컴파일). 벤치마킹 전에 워밍업하세요.
- JIT 내부에서 JAX 배열에 대한 Python 루프를 피하세요. `jax.lax.scan` 또는 `jax.lax.fori_loop`을 사용하세요.
- `jax.debug.print()`은 JIT 내부에서 작동합니다. 일반적인 `print()`은 작동하지 않습니다.
- `jax.profiler` 또는 TensorBoard로 프로파일링하세요. XLA 컴파일은 병목 현상을 숨길 수 있습니다.
- JAX는 기본적으로 GPU 메모리의 75%를 사전 할당합니다. `XLA_PYTHON_CLIENT_PREALLOCATE=false`을 설정하여 비활성화하세요.

**체크포인팅:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이 강이 생성하는 것:**
- `outputs/prompt-jax-optimizer.md` -- 올바른 JAX 옵티마이저 구성을 선택하기 위한 프롬프트
- `outputs/skill-jax-patterns.md` -- JAX의 함수형 패턴을 다루는 스킬

## 연습 문제

1. MLP에 드롭아웃을 추가하세요. JAX에서 드롭아웃은 PRNG 키가 필요합니다 -- 순방향 전파를 통해 키를 전달하고 각 드롭아웃 레이어마다 분할하세요. 테스트 정확도를 드롭아웃이 있을 때와 없을 때 비교하세요.

2. `jax.vmap`을 사용하여 32개의 MNIST 이미지 배치에 대한 예제별 기울기를 계산하세요. 각 예제의 기울기 노름을 계산하세요. 어떤 예제가 가장 큰 기울기를 가지며, 그 이유는 무엇인가요?

3. 수동 forward 함수를 임의의 레이어 수에 대해 작동하는 범용 `mlp_forward(params, x)`으로 교체하세요. `jax.tree.leaves`을 사용하여 깊이를 자동으로 결정하세요.

4. `@jax.jit`이 있을 때와 없을 때의 학습 스텝을 벤치마킹하세요. 각각 100 스텝을 측정하세요. 하드웨어에서 속도 향상은 얼마나 큰가요? 첫 호출 시 컴파일 오버헤드는 얼마인가요?

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`을 조합하여 기울기 클리핑을 구현하세요. 클리핑을 적용한 경우와 적용하지 않은 경우로 학습하세요. 학습 중 기울기 노름(norm)을 플롯하여 효과를 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만드는 것" | Accelerated Linear Algebra -- 연산 그래프에서 연산을 융합하고 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "Just-in-time 컴파일" | JAX는 첫 호출 시 함수를 추적(tracing)하여 XLA로 컴파일하고, 이후 호출에서는 컴파일된 버전을 실행합니다 |
| 순수 함수 | "사이드 이펙트 없음" | 출력은 입력에만 의존하는 함수 -- 전역 상태 없음, 변경(mutation) 없음, 명시적 키 없이는 랜덤성 없음 |
| vmap | "자동 배치" | 하나의 예제를 처리하는 함수를 재작성하지 않고도 배치를 처리하는 함수로 변환합니다 |
| pmap | "자동 병렬화" | 함수를 여러 디바이스에 복제하고 입력 배치를 분할합니다 |
| Pytree | "배열의 중첩 dict" | JAX가 순회하고 변환할 수 있는 리스트, 튜플, dict, 배열의 중첩된 모든 구조 |
| Tracing | "연산 기록" | JAX는 실제 결과를 계산하지 않고 추상값으로 함수를 실행하여 연산 그래프를 구축합니다 |
| 함수적 오토디프 | "함수의 grad" | 텐서에 기울기 저장소를 부착하는 것이 아니라 함수를 변환하여 미분값을 계산하는 것 |
| Optax | "JAX의 옵티마이저 라이브러리" | Adam, SGD, 클리핑, 스케줄링 등 기울기 변환을 체이닝하는 조합 가능한 라이브러리 |
| Flax | "JAX의 nn.Module" | 상태를 명시적으로 유지하면서 레이어 추상화를 추가하는 Google의 JAX용 신경망 라이브러리 |

## 추가 읽기

- JAX 문서: https://jax.readthedocs.io/ -- 공식 문서로, grad, jit, vmap에 대한 훌륭한 튜토리얼이 포함되어 있습니다
- "JAX: Python+NumPy 프로그램의 조합 가능한 변환" (Bradbury et al., 2018) -- 설계 철학을 설명하는 원 논문
- Flax 문서: https://flax.readthedocs.io/ -- JAX용 Google 신경망 라이브러리
- Patrick Kidger, "Equinox: 호출 가능한 PyTree와 필터링된 변환을 통한 JAX의 신경망" (2021) -- Flax의 Python다운 대안
- DeepMind, "Optax: 조합 가능한 기울기 변환 및 최적화" -- 표준 옵티마이저 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 저자 중 한 명이 작성한 JAX의 함정과 패턴에 대한 실용 가이드
