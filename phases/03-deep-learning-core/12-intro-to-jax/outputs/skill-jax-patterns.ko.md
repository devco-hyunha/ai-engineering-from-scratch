---
name: skill-jax-patterns
description: JAX의 함수형 프로그래밍 패턴 -- grad, jit, vmap, pmap을 언제, 어떻게 사용하는지
version: 1.0.0
phase: 3단계
lesson: 12강
tags: [jax, functional-programming, autodiff, compilation, vectorization]
---

# JAX 함수형 패턴

JAX는 순수 함수를 변환합니다. 아래 모든 패턴은 하나의 규칙을 따릅니다: 입력을 받아 출력으로 반환하는 함수를 작성하고, 사이드 이펙트(side effects)가 없도록 합니다. 그런 다음 변환합니다.

## 네 가지 변환

### grad -- 함수의 미분

```python
grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```

사용 시점: 최적화를 위해 기울기가 필요할 때.
제약 조건: 함수는 스칼라를 반환해야 합니다. 스칼라가 아닌 출력에는 `jax.jacobian`를 사용하세요.

### jit -- 함수 컴파일

```python
fast_fn = jax.jit(f)
```

사용 시점: 동일한 형태의 입력으로 함수가 두 번 이상 호출될 때.
제약 조건: 추적된(traced) 값에 의존하는 Python 제어 흐름은 사용할 수 없습니다. 조건문에는 `jax.lax.cond`를, 루프에는 `jax.lax.scan`를 사용하세요.

### vmap -- 함수 벡터화

```python
batch_fn = jax.vmap(f, in_axes=(None, 0))
```

사용 시점: 하나의 예제에 대해 함수를 작성했고, 이를 배치(batch)에 대해 작동하게 해야 할 때.
`in_axes`는 배치할 인수의 축(axis)을 지정합니다. `None`은 배치하지 않음(브로드캐스트)을 의미합니다.

### pmap -- 장치 간 병렬화

```python
parallel_fn = jax.pmap(f, axis_name='devices')
```

사용 시점: 여러 GPU/TPU가 있고 데이터 병렬화를 원할 때.
함수 내부에서 `jax.lax.pmean(x, 'devices')`는 장치 간 평균을 계산합니다.

## 조합 규칙

변환은 조합됩니다. 순서가 중요합니다:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))
```

오른쪽에서 왼쪽으로 읽기: loss_fn의 기울기를 구하고, 예제에 대해 벡터화하며, 결과를 컴파일합니다.

유효한 조합:
- `jit(grad(f))` -- 컴파일된 기울기 계산
- `jit(vmap(f))` -- 컴파일된 배치 계산
- `vmap(grad(f))` -- 예제별 기울기
- `pmap(jit(f))` -- 병렬 컴파일된 계산
- `grad(jit(f))` -- 컴파일된 함수의 기울기 (jit(grad(f))와 동일)

## 매개변수 관리 패턴

JAX 매개변수는 pytree (배열의 중첩된 dict)입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}
```

모든 매개변수를 한 번에 업데이트합니다:
```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

매개변수 개수를 셉니다:
```python
n_params = sum(p.size for p in jax.tree.leaves(params))
```

## PRNG 키 관리

JAX는 명시적인 랜덤 키를 요구합니다:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```

여러 개의 무작위 연산이 있는 경우, 한 번만 분할하세요:
```python
keys = jax.random.split(key, n)
```

키를 재사용하지 마세요. 사용 전에 항상 분할하세요.

## 공통 실수

1. **jit 내부에서 배열 변경**: JAX 배열은 불변입니다. `x[i] = v` 대신 `x.at[i].set(v)`를 사용하세요.

2. **jit 내부에서 Python print 사용**: `print`는 실행이 아닌 추적(tracing) 중에 실행됩니다. `jax.debug.print("{}", x)`를 사용하세요.

3. **jit 내부에서 추적된 값에 대한 Python if/for 사용**: `jax.lax.cond`, `jax.lax.switch`, `jax.lax.scan`, `jax.lax.fori_loop`를 사용하세요.

4. **`.block_until_ready()` 잊기**: JAX는 비동기 디스패치를 사용합니다. 벤치마킹 시 실제 완료를 기다리려면 `.block_until_ready()`를 호출하세요.

5. **PRNG 키 재사용**: 같은 키를 사용하는 두 연산은 동일한 "무작위" 값을 생성합니다. 항상 분할하세요.

6. **jitted 함수 내 전역 상태**: 전역 변수는 추적 시점에 캡처됩니다. 추적 이후의 변경은 반영되지 않습니다. 모든 것을 인자로 전달하세요.

## 의사 결정 체크리스트

1. 함수가 한 번 이상 호출되나요? `@jax.jit`를 추가하세요.
2. 기울기가 필요한가요? `jax.grad` 또는 `jax.value_and_grad`로 감싸세요.
3. 하나의 예제를 처리하는데 배치가 있나요? `jax.vmap`로 감싸세요.
4. 여러 장치가 있나요? `jax.pmap`로 감싸세요.
5. 무작위성을 사용하나요? PRNG 키를 명시적으로 전달하세요.
6. 배열 값에 대한 Python 제어 흐름이 있나요? `jax.lax` 원시 연산으로 대체하세요.

## JAX를 사용할 때

다음과 같은 경우 JAX를 사용하세요:
- 예제별 기울기가 필요할 때 (차분 프라이버시, 피셔 정보)
- TPU에서 학습할 때 (JAX가 네이티브 프레임워크)
- 고계 미분(Hessian, Jacobian)이 필요할 때
- 전체 학습 단계를 단일 커널로 컴파일하고 싶을 때
- 팀이 Google DeepMind나 Anthropic에 있을 때

PyTorch를 사용할 때:
- 가장 큰 생태계(HuggingFace, torchvision, Lightning)를 원할 때
- 순수 속도보다 디버깅 편의성을 우선시할 때
- NVIDIA GPU에 TorchServe/Triton으로 배포할 때
- 채용할 때 (PyTorch 개발자가 더 많음)
- 새로운 아키텍처를 빠르게 반복하고 싶을 때
