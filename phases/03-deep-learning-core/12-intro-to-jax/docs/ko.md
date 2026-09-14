# JAX 입문 (Introduction to JAX)

> PyTorch는 텐서를 변이시킵니다. TensorFlow는 그래프를 만듭니다. JAX는 순수 함수를 컴파일합니다. 마지막 것이 딥러닝을 생각하는 방식을 바꿉니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- JAX의 함수형 API(jax.numpy, jax.grad, jax.jit, jax.vmap)로 순수 함수 신경망 코드를 작성합니다
- PyTorch의 즉시 변이(eager mutation)와 JAX의 함수형 컴파일 모델 사이의 핵심 설계 차이를 설명합니다
- jit 컴파일과 vmap 벡터화를 적용해, 단순한 Python보다 학습 루프를 가속합니다
- JAX로 간단한 네트워크를 학습하고, 명시적 상태 관리를 PyTorch의 객체지향 접근과 대비합니다

## 문제 상황 (The Problem)

PyTorch로 신경망을 만드는 법을 압니다. `nn.Module`을 정의하고, `.backward()`를 호출하고, 옵티마이저를 step합니다. 동작합니다. 수백만 명이 씁니다.

하지만 PyTorch DNA에는 제약이 있습니다. 연산을 Python에서 하나씩, 즉시(eager) 추적합니다. `tensor + tensor`마다 별도 커널 런치입니다. 학습 step마다 같은 Python 코드를 다시 해석합니다. 2,048개 TPU에 걸쳐 5,400억 파라미터 모델을 학습해야 하기 전까지는 괜찮습니다. 그때는 오버헤드가 죽입니다.

Google DeepMind는 Gemini를 JAX로 학습합니다. Anthropic은 Claude를 JAX로 학습했습니다. 작은 작업이 아닙니다 — 지구에서 가장 큰 신경망 학습 런입니다. 학습 루프를 Python 호출 시퀀스가 아니라 컴파일 가능한 프로그램으로 다루기 때문에 JAX를 골랐습니다.

JAX는 세 가지 초능력을 가진 NumPy입니다. 자동 미분, XLA로의 JIT 컴파일, 자동 벡터화. 예제 하나를 처리하는 함수를 쓰면, JAX가 배치를 처리하고, 기울기를 계산하고, 기계어로 컴파일하고, 여러 디바이스에서 도는 함수를 줍니다. 원본 함수를 바꾸지 않고요.

## 핵심 개념 (The Concept)

### JAX 철학 (The JAX Philosophy)

JAX는 함수형 프레임워크입니다. 클래스 없음, 가변 상태 없음, `.backward()` 메서드 없음. 대신:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

스타일 취향이 아닙니다. 컴파일러 제약입니다. JIT 컴파일은 순수 함수를 요구합니다 — 같은 입력은 항상 같은 출력, 부수 효과 없음. 그 제약이 100배 속도 향상을 가능하게 합니다.

### jax.numpy: 익숙한 표면

JAX는 가속기에서 NumPy API를 다시 구현합니다:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

같은 함수 이름. 같은 브로드캐스팅 규칙. 같은 슬라이싱 의미. 하지만 배열은 GPU/TPU에 살고, 모든 연산은 컴파일러가 추적할 수 있습니다.

핵심 차이 하나: JAX 배열은 불변입니다. `a[0] = 5`는 안 됩니다. 대신: `a = a.at[0].set(5)`. 일주일은 어색하다가, 그다음 클릭됩니다 — 불변성이 `grad`, `jit`, `vmap` 같은 변환을 합성 가능하게 만듭니다.

### jax.grad: 함수형 자동 미분

PyTorch는 기울기를 텐서에 붙입니다(`.grad`). JAX는 기울기를 함수에 붙입니다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 기울기를 계산하는 새 함수를 반환합니다. `.backward()` 호출 없음. 텐서에 저장된 계산 그래프 없음. 기울기는 호출·합성·JIT 컴파일할 수 있는 또 다른 함수일 뿐입니다.

임의로 합성됩니다:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2차 도함수. 3차 도함수. Jacobian. Hessian. 모두 `grad`를 합성하면 됩니다. PyTorch도 할 수 있습니다(`torch.autograd.functional.hessian`). 하지만 덧붙여진 것입니다. JAX에서는 기반입니다.

제약: `grad`는 순수 함수에만 동작합니다. 안쪽에 print 없음(추적 중에 실행되고, 실제 실행 중이 아님). 외부 상태 변이 없음. 명시적 키 관리 없는 난수 생성 없음.

### jit: XLA로 컴파일

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 호출에서 JAX는 함수를 추적합니다 — 실행하지 않고 어떤 연산이 일어나는지 기록합니다. 그 추적을 Google의 TPU·GPU용 컴파일러인 XLA(Accelerated Linear Algebra)에 넘깁니다. XLA는 연산을 퓨즈하고, 불필요한 메모리 복사를 없애고, 최적화된 기계어를 생성합니다.

이후 호출은 Python을 완전히 건너뜁니다. 컴파일된 코드가 C++ 속도로 가속기에서 돕니다.

JIT가 도움이 될 때:
- 학습 step (같은 계산을 수천 번 반복)
- 추론 (같은 모델, 다른 입력)
- 비슷한 shape의 입력으로 두 번 이상 호출되는 함수

JIT가 해로울 때:
- 값에 의존하는 Python 제어 흐름이 있는 함수 (`x > 0`처럼 x가 추적된 배열인 경우)
- 일회성 계산 (컴파일 오버헤드가 런타임보다 큼)
- 디버깅 (추적이 실제 실행을 가림)

제어 흐름 제약은 진짜입니다. `jax.lax.cond`가 `if/else`를 대체합니다. `jax.lax.scan`이 `for` 루프를 대체합니다. 선택 사항이 아닙니다 — 컴파일의 대가입니다.

### vmap: 자동 벡터화

예제 하나를 처리하는 함수를 씁니다:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`이 배치를 처리하도록 올립니다:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`은: `params`는 배치하지 않음(공유), `x`의 축 0에 걸쳐 배치. 수동 `for` 루프 없음. reshape 없음. 배치 차원을 손으로 꿰뚫을 필요 없음. JAX가 배치 차원을 파악하고 전체 계산을 벡터화합니다.

문법 설탕이 아닙니다. `vmap`은 Python 루프보다 10-100배 빠른 퓨즈된 벡터화 코드를 생성합니다. 그리고 `jit`, `grad`와 합성됩니다:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 기울기. 한 줄. PyTorch에서는 해킹 없이 거의 불가능합니다.

### pmap: 디바이스 간 데이터 병렬

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`은 사용 가능한 모든 디바이스(GPU/TPU)에 함수를 복제하고 배치를 나눕니다. 함수 안에서 `jax.lax.pmean`과 `jax.lax.psum`이 디바이스 간 기울기를 동기화합니다.

Google은 `pmap`(및 후속 `shard_map`)으로 수천 개의 TPU v5e 칩에 걸쳐 Gemini를 학습합니다. 프로그래밍 모델: 단일 디바이스 버전을 쓰고, `pmap`으로 감싸고, 끝.

### Pytree: 보편 데이터 구조

JAX는 "pytree" — 리스트, 튜플, dict, 배열의 중첩 조합 — 위에서 동작합니다. 모델 파라미터가 pytree입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

모든 JAX 변환 — `grad`, `jit`, `vmap` — 은 pytree를 순회하는 법을 압니다. `jax.tree.map(f, tree)`는 모든 리프에 `f`를 적용합니다. 옵티마이저가 모든 파라미터를 한 번에 갱신하는 방식입니다:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드 없음. 파라미터 등록 없음. 트리 구조가 모델입니다.

### 함수형 vs 객체지향 (Functional vs Object-Oriented)

PyTorch는 상태를 객체 안에 저장합니다:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 상태를 가진 순수 함수를 씁니다:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params가 전달됩니다. 저장된 것 없음. 변이된 것 없음. 모든 함수가 테스트·합성·컴파일 가능합니다. 대신 params를 직접 관리하거나 — Flax나 Equinox 같은 라이브러리를 씁니다.

### JAX 생태계 (The JAX Ecosystem)

JAX는 원시 요소를 줍니다. 라이브러리가 편의성을 줍니다:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax는 표준 옵티마이저 라이브러리입니다. 기울기 변환(Adam, SGD, clipping)을 파라미터 갱신과 분리해, 합성이 쉬워집니다:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorch를 언제 쓸까 (When to Use JAX vs PyTorch)

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

솔직한 답: JAX를 쓸 구체적 이유가 없으면 PyTorch를 쓰세요. 그 이유는 — TPU 접근, 예제별 기울기 필요, 대규모 다중 디바이스 학습, 또는 Google/DeepMind/Anthropic에서 일할 때입니다.

### JAX의 난수 (Random Numbers in JAX)

JAX에는 전역 난수 상태가 없습니다. 모든 난수 연산에 명시적 PRNG 키가 필요합니다:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음엔 짜증납니다. 하지만 디바이스와 컴파일에 걸쳐 재현성을 보장합니다 — 다중 GPU에서 PyTorch의 `torch.manual_seed`가 보장하지 못하는 속성입니다.

```figure
batchnorm-effect
```

## 직접 만들기 (Build It)

### Step 1: 설정과 데이터

JAX와 Optax로 MNIST에서 3층 MLP를 학습합니다. 입력 784, 은닉층 256·128, 출력 클래스 10.

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

### Step 2: 파라미터 초기화

클래스 없음. pytree를 반환하는 함수만:

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

He 초기화를 수동으로. 시드 하나에서 PRNG 키 세 개로 분할. 모든 가중치는 중첩 dict 안의 불변 배열입니다.

### Step 3: 순전파

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

순수 함수. Params 들어가고, 예측 나옴. `self` 없음, 저장된 상태 없음. `loss_fn`은 교차 엔트로피를 처음부터 계산합니다 — softmax, log, 음의 평균.

### Step 4: JIT 컴파일된 학습 Step

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

`jax.value_and_grad`는 한 패스에서 손실 값과 기울기를 모두 반환합니다. `@jax.jit` 데코레이터가 두 함수를 XLA로 컴파일합니다. 첫 호출 이후 각 학습 step은 Python을 건드리지 않고 돕니다.

### Step 5: 학습 루프

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

10 에포크. 테스트 정확도 ~97%. 첫 에포크는 느립니다(JIT 컴파일). 에포크 2-10은 빠릅니다.

빠진 것에 주목하세요. `.zero_grad()` 없음, `.backward()` 없음, `.step()` 없음. 전체 갱신이 한 번의 합성된 함수 호출입니다. 기울기 계산, Adam 변환, 파라미터 적용이 모두 `train_step` 안에서 일어납니다.

## 활용하기 (Use It)

### Flax: Google 표준

Flax는 가장 흔한 JAX 신경망 라이브러리입니다. `nn.Module`을 되살리되, 명시적 상태 관리를 유지합니다:

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

PyTorch와 같은 구조이지만, `params`는 모델과 분리됩니다. `model.init()`이 params를 만듭니다. `model.apply(params, x)`가 순전파를 돌립니다. 모델 객체에는 상태가 없습니다.

### Equinox: Python스러운 대안

Equinox(Patrick Kidger)는 모델을 pytree로 표현합니다:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree입니다. `.apply()`가 필요 없습니다. 파라미터는 모델의 리프일 뿐입니다. JAX가 생각하는 방식에 더 가깝습니다.

### Optax: 합성 가능한 옵티마이저

Optax는 기울기 변환을 갱신과 분리합니다:

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

기울기 클리핑, 학습률 warmup, weight decay — 모두 변환 체인으로 합성됩니다. 각 변환이 기울기를 보고, 수정하고, 다음으로 넘깁니다. 단일체 옵티마이저 클래스 없음.

## 산출물 (Ship It)

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원:

```bash
pip install jax[cuda12]
```

TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 주의사항:**

- 첫 JIT 호출은 느립니다(컴파일). 벤치마크 전에 warmup하세요.
- JIT 안에서 JAX 배열에 대한 Python 루프를 피하세요. `jax.lax.scan` 또는 `jax.lax.fori_loop`를 쓰세요.
- `jax.debug.print()`는 JIT 안에서 동작합니다. 일반 `print()`는 안 됩니다.
- `jax.profiler` 또는 TensorBoard로 프로파일하세요. XLA 컴파일이 병목을 가릴 수 있습니다.
- JAX는 기본적으로 GPU 메모리의 75%를 미리 할당합니다. `XLA_PYTHON_CLIENT_PREALLOCATE=false`로 끌 수 있습니다.

**체크포인팅:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이 레슨이 만드는 것:**
- `outputs/prompt-jax-optimizer.md` -- 올바른 JAX 옵티마이저 설정을 고르는 프롬프트
- `outputs/skill-jax-patterns.md` -- JAX의 함수형 패턴을 다루는 스킬

## 연습 문제 (Exercises)

1. MLP에 dropout을 추가하세요. JAX에서 dropout은 PRNG 키가 필요합니다 — 순전파에 키를 꿰고 각 dropout 레이어마다 분할하세요. 유무에 따른 테스트 정확도를 비교하세요.

2. `jax.vmap`으로 MNIST 이미지 32개 배치의 예제별 기울기를 계산하세요. 각 예제의 기울기 노름을 계산하세요. 어떤 예제의 기울기가 가장 크고, 왜인가요?

3. 수동 순전파 함수를 임의의 레이어 수에 동작하는 일반 `mlp_forward(params, x)`로 바꾸세요. `jax.tree.leaves`로 깊이를 자동으로 결정하세요.

4. `@jax.jit` 유무에 따라 학습 step을 벤치마크하세요. 각각 100 step을 재세요. 하드웨어에서 속도 향상은 얼마나 큰가요? 첫 호출의 컴파일 오버헤드는?

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`를 합성해 기울기 클리핑을 구현하세요. 클리핑 유무로 학습하세요. 학습 중 기울기 노름을 그려 효과를 보세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만드는 것" | Accelerated Linear Algebra — 연산을 퓨즈하고 계산 그래프에서 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "Just-in-time 컴파일" | JAX가 첫 호출에서 함수를 추적해 XLA로 컴파일한 뒤, 이후 호출에서 컴파일된 버전을 실행 |
| Pure function | "부수 효과 없음" | 출력이 입력에만 의존하는 함수 — 전역 상태·변이·명시적 키 없는 난수 없음 |
| vmap | "자동 배칭" | 예제 하나를 처리하는 함수를, 다시 쓰지 않고 배치를 처리하는 함수로 변환 |
| pmap | "자동 병렬화" | 여러 디바이스에 함수를 복제하고 입력 배치를 분할 |
| Pytree | "배열의 중첩 dict" | JAX가 순회·변환할 수 있는 리스트·튜플·dict·배열의 중첩 구조 |
| Tracing | "계산 기록" | JAX가 추상 값으로 함수를 실행해 실제 결과 없이 계산 그래프를 만듦 |
| Functional autodiff | "함수의 grad" | 텐서에 기울기 저장소를 붙이지 않고, 함수를 변환해 도함수를 계산 |
| Optax | "JAX 옵티마이저 라이브러리" | Adam, SGD, clipping, scheduling을 체인으로 합성하는 기울기 변환 라이브러리 |
| Flax | "JAX의 nn.Module" | 상태를 명시적으로 유지하면서 레이어 추상화를 추가하는 Google의 JAX 신경망 라이브러리 |

## 더 읽을거리 (Further Reading)

- JAX documentation: https://jax.readthedocs.io/ -- 공식 문서. grad, jit, vmap 튜토리얼이 우수함
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 설계 철학을 설명하는 원 논문
- Flax documentation: https://flax.readthedocs.io/ -- Google의 JAX 신경망 라이브러리
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax의 Python스러운 대안
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 표준 옵티마이저 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 저자 중 한 명의 JAX 함정과 패턴 실용 가이드
