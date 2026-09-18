---
name: skill-pipeline-budget-planner
description: 목표 지연과 처리량이 주어지면 모든 파이프라인 단계에 시간 예산을 할당하고 어떤 단계가 예산을 먼저 못 지킬지 표시
version: 1.0.0
phase: 4
lesson: 16
tags: [vision, pipeline, performance, deployment]
---

# Pipeline Budget Planner

지연/처리량 목표를 단계별 예산으로 바꿔, 모든 팀원이 어떤 숫자를 향해 엔지니어링하는지 알게 합니다.

## When to use

- 새 비전 서비스를 만들기 전에, 각 단계의 기대치를 정할 때.
- 첫 벤치마크 후, 어떤 단계가 예산에서 가장 멀리 있는지 볼 때.
- SLA가 바뀌어 예산을 재협상해야 할 때.

## Inputs

- `p95_latency_target_ms`: 요청당 예산.
- `target_qps`: 레플리카당 처리량.
- `stages`: `{ name: str, current_ms: float }` 리스트.

## Allocation rules

현재 측정이 없을 때 일곱 표준 단계에 대한 기본 할당:

| Stage | Share |
|-------|-------|
| decode + preprocess | 15% |
| detector forward | 55% |
| postprocess detections (NMS, clamp) | 5% |
| crop + resize for classifier | 5% |
| classifier forward | 15% |
| schema validation | <1% |
| response serialisation | 4% |

GPU 바운드 파이프라인(클라우드)에서는 탐지기 비중이 종종 70%까지 올라갑니다. CPU에서는 전처리와 분류기 배칭이 더 많이 먹습니다.

## Report

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

## Rules

- 프로덕션 경로에서 스키마 검증을 제거하라고 추천하지 마세요. 대신 경계로 옮기라고 제안하세요.
- 전처리가 예산을 못 지키면, 모델을 바꾸기 전에 항상 Pillow-SIMD 또는 NVJPEG를 먼저 시도하세요.
- 탐지기 미스가 목표의 30%를 넘으면, 현재 모델을 최적화하기보다 모델을 바꾸세요.
- current_ms > 1.1 * target_ms이면 게이트를 `X`로, 예산의 10% 이내면 `ok`로 표시하세요.
