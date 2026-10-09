---
name: distributed-fsdp-ddp
description: gloo 또는 nccl 백엔드에서 처음부터 만든 DDP 래퍼와 FSDP 매개변수 샤딩 스케치를 사용하여 다중 랭크 훈련을 시작합니다.
version: 1.0.0
phase: 19단계
lesson: 48강
tags: [distributed, ddp, fsdp, collectives]
---

## 언제 사용해야 하는가

모델이 하나의 장치에 fits 되지만 더 높은 처리량이 필요한 경우 (DDP). 모델이 하나의 장치에 fits 되지 않는 경우 (FSDP). 두 경우 모두: 동일한 코드 경로를 사용하는 다중 랭크 훈련 설정입니다.

## 프로세스 그룹 시작하기

```python
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
```

`gloo`는 CPU 백엔드이며, `nccl`는 GPU 백엔드입니다. 둘 다 동일한 collective 인터페이스를 구현합니다.

## 모델 래핑하기

1. 랭크 0에서 시드(seed)를 사용하여 모델을 빌드합니다.
2. DDP 셸로 래핑합니다.
3. 셸의 `__init__`는 모든 매개변수와 버퍼에 대해 `dist.broadcast(p.data, src=0)`를 호출합니다.
4. 매 `loss.backward()` 이후, 트레이너는 `sync_grads()`를 호출합니다.
5. `sync_grads()`는 `dist.all_reduce(p.grad, op=SUM)`와 `p.grad.div_(world_size)`를 호출합니다.
6. 모든 랭크에서 동일한 평균화된 기울기로 옵티마이저 스텝을 수행합니다.

## 매개변수 샤딩 (FSDP 스케치)

1. 각 매개변수를 평탄화(flatten)하고 `world_size`의 배수로 패딩합니다.
2. 로컬 샤드를 유지하고 나머지는 해제합니다.
3. forward 전에, `dist.all_gather(...)`를 사용하여 모든 랭크에서 전체 텐서를 재구성합니다.
4. forward 후, 전체 텐서를 버립니다.

## 실패 모드

- 브로드캐스트를 건너뛰는 경우: 랭크들이 서로 다른 초기값에서 시작하여 조용히 발산(diverge)합니다.
- 합산 후 나누는 것을 잊는 경우: 기울기가 world_size로 스케일링되어 옵티마이저 스텝이 너무 커집니다.
- 체크포인트에 cross-device rename을 사용하는 경우: atomic하지 않으며, 47강의 함정과 같습니다.
- 동일한 collective에서 CPU 및 CUDA 텐서를 혼합하는 경우: 백엔드 불일치로 인해 실행이 멈춥니다.
