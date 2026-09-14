---
name: skill-jax-patterns
description: JAX의 함수형 프로그래밍 패턴 — grad, jit, vmap, pmap을 언제·어떻게 쓰는가
version: 1.0.0
phase: 3
lesson: 12
tags: [jax, functional-programming, autodiff, compilation, vectorization]
---

# JAX 함수형 패턴 (JAX Functional Patterns)

JAX는 순수 함수를 변환합니다. 아래 모든 패턴은 한 규칙을 따릅니다. 입력을 받아 출력을 반환하고 부수 효과가 없는 함수를 쓴 뒤, 변환합니다.

## 네 가지 변환 (The Four Transforms)

### grad -- 함수를 미분

```python
grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```

사용할 때: 최적화용 기울기가 필요할 때.
제약: 함수는 스칼라를 반환해야 합니다. 비스칼라 출력에는 `jax.jacobian`을 쓰세요.

### jit -- 함수를 컴파일

```python
fast_fn = jax.jit(f)
```

사용할 때: 같은 shape의 입력으로 두 번 이상 호출될 함수.
제약: 추적된 값에 의존하는 Python 제어 흐름 없음. 조건문은 `jax.lax.cond`, 루프는 `jax.lax.scan`.

### vmap -- 함수를 벡터화

```python
batch_fn = jax.vmap(f, in_axes=(None, 0))
```

사용할 때: 예제 하나용 함수를 배치에도 써야 할 때.
`in_axes`는 어느 인자 축에 걸쳐 배치할지 지정합니다. `None`은 배치하지 않음(브로드캐스트).

### pmap -- 디바이스에 걸쳐 병렬화

```python
parallel_fn = jax.pmap(f, axis_name='devices')
```

사용할 때: GPU/TPU가 여러 개이고 데이터 병렬이 필요할 때.
함수 안에서 `jax.lax.pmean(x, 'devices')`가 디바이스에 걸쳐 평균합니다.

## 합성 규칙 (Composition Rules)

변환은 합성됩니다. 순서가 중요합니다:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))
```

오른쪽에서 왼쪽으로 읽기: loss_fn의 기울기를 취하고, 예제에 걸쳐 벡터화하고, 결과를 컴파일.

유효한 합성:
- `jit(grad(f))` -- 컴파일된 기울기 계산
- `jit(vmap(f))` -- 컴파일된 배치 계산
- `vmap(grad(f))` -- 예제별 기울기
- `pmap(jit(f))` -- 병렬 컴파일 계산
- `grad(jit(f))` -- 컴파일된 함수의 기울기 (jit(grad(f))와 동일)

## 파라미터 관리 패턴 (Parameter Management Pattern)

JAX 파라미터는 pytree(배열의 중첩 dict)입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}
```

모든 파라미터를 한 번에 갱신:
```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

파라미터 수 세기:
```python
n_params = sum(p.size for p in jax.tree.leaves(params))
```

## PRNG 키 관리 (PRNG Key Management)

JAX는 명시적 난수 키가 필요합니다:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```

여러 난수 연산에는 한 번에 분할:
```python
keys = jax.random.split(key, n)
```

키를 재사용하지 마세요. 쓰기 전에 항상 분할하세요.

## 흔한 실수 (Common Mistakes)

1. **jit 안에서 배열 변이**: JAX 배열은 불변입니다. `x[i] = v` 대신 `x.at[i].set(v)`를 쓰세요.

2. **jit 안에서 Python print**: `print`는 추적 중에 실행되고, 실제 실행 중이 아닙니다. `jax.debug.print("{}", x)`를 쓰세요.

3. **추적된 값에 대한 jit 안의 Python if/for**: `jax.lax.cond`, `jax.lax.switch`, `jax.lax.scan`, `jax.lax.fori_loop`를 쓰세요.

4. **`.block_until_ready()` 잊기**: JAX는 비동기 디스패치를 씁니다. 벤치마크 시 `.block_until_ready()`로 실제 완료를 기다리세요.

5. **PRNG 키 재사용**: 같은 키로 두 연산을 하면 같은 "난수" 값이 나옵니다. 항상 분할하세요.

6. **jitted 함수의 전역 상태**: 전역 변수는 추적 시점에 캡처됩니다. 추적 후 변경은 보이지 않습니다. 모든 것을 인자로 넘기세요.

## 결정 체크리스트 (Decision Checklist)

1. 함수가 두 번 이상 호출되나? `@jax.jit` 추가.
2. 기울기가 필요하나? `jax.grad` 또는 `jax.value_and_grad`로 감싸기.
3. 예제 하나를 처리하는데 배치가 있나? `jax.vmap`으로 감싸기.
4. 디바이스가 여러 개인가? `jax.pmap`으로 감싸기.
5. 난수를 쓰나? PRNG 키를 명시적으로 꿰기.
6. 배열 값에 대한 Python 제어 흐름이 있나? `jax.lax` 원시 요소로 교체.

## JAX를 언제 쓸까 (When to Use JAX)

JAX를 쓸 때:
- 예제별 기울기가 필요할 때 (차분 프라이버시, Fisher 정보)
- TPU에서 학습할 때 (JAX가 네이티브 프레임워크)
- 고차 도함수가 필요할 때 (Hessian, Jacobian)
- 전체 학습 step을 단일 커널로 컴파일하고 싶을 때
- 팀이 Google DeepMind 또는 Anthropic일 때

PyTorch를 쓸 때:
- 가장 큰 생태계가 필요할 때 (HuggingFace, torchvision, Lightning)
- 날것의 속도보다 디버깅 용이성을 우선할 때
- TorchServe/Triton으로 NVIDIA GPU에 배포할 때
- 채용할 때 (PyTorch 개발자가 더 많음)
- 새 아키텍처를 빠르게 반복하고 싶을 때
