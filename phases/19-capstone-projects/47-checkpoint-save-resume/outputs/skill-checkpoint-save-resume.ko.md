---
name: checkpoint-save-resume
description: 원자적, 샤드화된 체크포인트로 전체 RNG를 캡처하여, 종료된 실행이 동일한 손실 궤적으로 에포크 중간에 재개할 수 있습니다.
version: 1.0.0
phase: 19단계
lesson: 47강
tags: [training, durability, resume, sharded-state]
---

## 언제 사용해야 하는가

클러스터의 벽시계(wallclock) 상한보다 긴 모든 훈련 실행, 노드 재부팅을 견뎌야 하는 모든 실행, 단일 페이로드에 너무 큰 모든 모델에 사용하세요.

## 페이로드 형태

```python
{
  "schema": "ckpt.v1",
  "model": model.state_dict(),
  "optimizer": opt.state_dict(),
  "scheduler": sched.state_dict(),
  "state": {"step": int, "epoch": int, "batch_in_epoch": int, "losses": [float, ...]},
  "rng": {"python": ..., "numpy": ..., "torch_cpu": ..., "torch_cuda": ...},
  "wall_saved_at": time.time(),
}
```

## 원자적 저장

1. 페이로드를 대상 파일과 같은 디렉토리의 고유한 임시 파일에 쓰세요.
2. `os.replace(tmp, target)`를 사용하여 원자적으로 교체하세요.
3. 대상 이름에 직접 쓰지 마세요.

## 샤드화 레이아웃

- 샤드당 `model.shard-NNN.pt`를 사용하며, 키에 따라 라운드 로빈으로 처리하거나 파라미터 그룹별로 분할하세요.
- `meta.pt`는 옵티마이저, 스케줄러, 훈련 상태, RNG, 그리고 샤드 매니페스트를 포함합니다.
- `index.json`는 모든 샤드와 `meta.pt`에 대한 `sha256`를 포함합니다.
- 로더는 병합하기 전에 모든 해시를 검증하며, 체크포인트 디렉토리 외부의 샤드 경로를 거부합니다.
- 모든 파일을 `torch.load(path, map_location="cpu", weights_only=True)`로 로드하세요. RNG 상태를 평범한 리스트로 유지하여 가중치 전용 로더에서도 살아남도록 하세요.

## 에포크 중간 재개

- `step` 옆에 `(epoch, batch_in_epoch)`를 저장하세요.
- 재개된 에포크의 첫 번째 배치 전에 RNG 상태를 복원하세요.
- 소비된 배치를 넘겨 생성기를 빠르게 진행하세요.

## 실패 모드

- 교차 장치 이름 변경: 원자적이지 않으며, 이전 파일을 잃습니다. 임시 파일을 같은 디렉토리에 두세요.
- RNG를 잊는 경우: 재개된 손실이 기준선과 갈라집니다. 데모의 어서션을 실행하세요.
- 옵티마이저 상태를 잊는 경우: 다음 단계가 급변합니다. 같은 차이가 폭발합니다.
- 잘못된 체크포인트를 제거하는 경우: 마지막 K개와 최상의 것을 유지하세요.
- `weights_only=False`로 로드하는 경우: `.pt` 파일은 피클이므로, 신뢰할 수 없는 체크포인트는 로드 시 코드를 실행합니다.
