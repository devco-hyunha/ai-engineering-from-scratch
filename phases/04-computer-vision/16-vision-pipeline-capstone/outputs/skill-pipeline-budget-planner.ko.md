---
name: skill-pipeline-budget-planner
description: 목표 지연 시간 및 처리량에 따라 모든 파이프라인 단계에 시간 예산을 할당하고, 예산을 먼저 초과하는 단계를 표시합니다
version: 1.0.0
phase: 4
lesson: 16
tags: [vision, pipeline, performance, deployment]
---

# 파이프라인 예산 계획자

지연 시간/처리량 목표를 단계별 예산으로 변환하여, 모든 팀원이 어떤 수치를 목표로 개발해야 하는지 알 수 있도록 합니다.

## 사용 시점

- 새로운 비전 서비스를 구축하기 전에, 각 단계의 기대치를 설정할 때 사용합니다.
- 첫 번째 벤치마크 후, 예산에서 가장 멀리 떨어진 단계를 확인할 때 사용합니다.
- SLA가 변경되어 예산을 재협상해야 할 때 사용합니다.

## 입력

- `p95_latency_target_ms`: 요청당 예산.
- `target_qps`: 복제본(replica)당 처리량.
- `stages`: `{ name: str, current_ms: float }` 목록.

## 할당 규칙

현재 측정값이 제공되지 않은 경우, 7가지 표준 단계에 대한 기본 할당:

| 단계 | 비중 |
|-------|-------|
| 디코딩 + 전처리 | 15% |
| 디텍터 forward | 55% |
| 탐지 후처리 (NMS, clamp) | 5% |
| 분류기용 크롭 + 리사이즈 | 5% |
| 분류기 forward | 15% |
| 스키마 검증 | <1% |
| 응답 직렬화 | 4% |

GPU 바운드 파이프라인(클라우드)에서는 디텍터 비중이 종종 70%까지 상승합니다. CPU에서는 전처리 및 분류기 배치가 더 많은 시간을 소모합니다.

## 보고서

```
[budget plan]
  p95 target:  <ms>
  throughput:  <qps per replica>

| stage               | target_ms | current_ms | headroom | gate |
|---------------------|-----------|------------|----------|------|
| decode+preprocess   | ...       | ...        | ...      | ok|X |
| detector            | ...       | ...        | ...      | ok|X |
| ...                 | ...       | ...        | ...      |      |

[bottleneck]
  stage:  <name>
  miss:   <ms over budget>
  lever:  <specific action>

[levers]
  decode+preprocess:   Pillow-SIMD, libjpeg-turbo, decode on GPU via NVJPEG
  detector:            smaller backbone, lower input resolution, INT8, TensorRT
  postprocess:         GPU-side NMS (torchvision.ops), fused masks
  crop+resize:         GPU crop with grid_sample, batched interpolate
  classifier:          smaller backbone, INT8, warm cache, batch
  schema:              skip validation in hot path, validate at boundaries only
  response:            orjson, stream protobuf
```

## 규칙

- 프로덕션 경로에서 스키마 검증을 제거하는 것을 절대 권장하지 마세요. 대신 경계(boundary)로 이동하는 것을 제안하세요.
- 전처리가 예산을 초과하는 경우, 모델을 변경하기 전에 항상 Pillow-SIMD 또는 NVJPEG를 시도해 보세요.
- 디텍터의 초과분이 목표의 30% 이상인 경우, 현재 모델을 최적화하는 대신 모델을 교체하세요.
- current_ms > 1.1 * target_ms일 때 게이트를 `X`으로 표시하고, 예산의 10% 이내일 경우 `ok`으로 표시하세요.
