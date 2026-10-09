---
name: skill-inference-optimization
description: LLM 추론 서빙의 처리량, 지연 시간 및 비용을 진단하고 최적화합니다
version: 1.0.0
phase: 10단계
lesson: 12강
tags: [추론, kv-cache, 배치, 추론적 디코딩, vllm, 최적화]
---

# LLM 추론 최적화 패턴

두 단계로 구성됩니다: 프리필(연산 집중, 병렬)과 디코딩(메모리 집중, 순차).
모든 최적화는 이 두 단계 중 하나 또는 둘 모두를 목표로 합니다.

```
Request -> Prefill (process prompt) -> Decode (generate tokens) -> Response
              |                            |
         Compute-bound               Memory-bound
         Optimize: fusion,           Optimize: batching,
         prefix caching              quantization, speculation
```

## 의사 결정 프레임워크

### 1단계: 병목 현상 식별

작업 부하에 대한 연산:바이트 비율을 측정하세요:

| 연산:바이트 | 병목 유형 | 최적화 대상 |
|----------|-------|-----------------|
| < 50 | 메모리 | KV 캐시 양자화, 배치 크기 증가 |
| 50-200 | 전환 구간 | 둘 다 중요하며, 배치부터 시작하세요 |
| > 200 | 연산 | 커널 융합, 텐서 병렬화, FP8 |

### 2단계: 엔진 선택

- **기본값**: vLLM (가장 넓은 모델 지원, PagedAttention, OpenAI 호환 API)
- **다중 턴 / 구조화된 출력**: SGLang (RadixAttention 접두어 캐싱, 제약 디코딩)
- **NVIDIA 최대 처리량**: TensorRT-LLM (커널 융합, H100에서의 FP8)

### 3단계: 순서대로 최적화 적용

1. **KV 캐시** -- 항상 켜 두세요, 단점이 없습니다
2. **연속 배치** -- 항상 켜 두세요, 단점이 없습니다 (vLLM/SGLang은 기본적으로 이를 수행합니다)
3. **접두어 캐싱** -- 공유 시스템 프롬프트가 있는 경우 활성화하세요 (대부분의 챗봇이 해당됩니다)
4. **양자화** -- KV 캐시 INT8/FP8은 품질 손실이 거의 없이 메모리를 2-4배 줄입니다
5. **추론적 디코딩** -- 처리량보다 지연 시간이 더 중요한 경우 추가하세요
6. **텐서 병렬화** -- 모델이 하나의 GPU에 맞지 않을 경우 GPU 간에 분할하세요

## KV 캐시 메모리 공식

```
per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
total = per_token * sequence_length * num_concurrent_users
```

일반적인 모델에 대한 빠른 참조 (BF16):

| 모델 | 토큰당 | 100명 사용자 @ 4K |
|-------|-----------|----------------|
| Llama 3 8B | 32 KB | 12.5 GB |
| Llama 3 70B | 320 KB | 125 GB |
| Llama 3 405B | 504 KB | 197 GB |

## 추론적 디코딩 체크리스트

- 초안 모델은 대상 모델보다 5-10배 작아야 합니다 (예: 70B에 대해 8B 초안 모델 사용)
- 의미 있는 속도 향상을 위해 수용률 > 70%
- 예측 가능한 텍스트(코드, 구조화된 출력, 자연어)에서 가장 효과적
- 창작/샘플링이 많은 작업에서 가장 효과적이지 않음(낮은 온도가 도움이 됨)
- 대부분의 워크로드에서 EAGLE > draft-target > n-gram

## 공통 실수

- 배치=1로 디코딩 실행(메모리 바운드, 연산에서 GPU가 95% 유휴 상태)
- 연속 KV 캐시 블록 할당(페이지드 KV 캐시(Paged KV Cache)를 사용하여 거의 제로 낭비)
- 요청의 80%가 동일한 시스템 프롬프트(System Prompt)를 공유할 때 접두어 캐싱(Prefix Caching)을 무시
- 모델 가중치(Weight)에 GPU 메모리를 과잉 할당하여 KV 캐시(KV Cache)에 남는 메모리가 없음
- 지연 시간을 측정하지 않고 처리량 측정(10초의 첫 토큰까지의 시간 (TTFT)(Time to First Token (TTFT))에서의 높은 처리량은 무용지물)
- 높은 온도(Temperature)로 추론적 디코딩(Speculative Decoding) 사용(수용률이 50% 미만으로 떨어짐)

## 모니터링 체크리스트

- 첫 토큰까지의 시간 (TTFT)(Time to First Token (TTFT)): 프리필(Prefill) 지연, 인터랙티브 사용 시 목표 < 500ms
- 토큰 간 지연 (ITL)(Inter-Token Latency (ITL)): 디코딩 속도, 스트리밍(Streaming) 시 목표 < 50ms
- 처리량(초당 토큰 수 (TPS)(Tokens per Second (TPS))): 모든 동시 사용자의 총합
- KV 캐시(KV Cache) 활용률: 할당된 캐시 중 사용 중인 비율
- 배치 활용률: 반복(iteration)마다 채워진 배치 슬롯의 비율
- 큐 깊이: 배치 슬롯을 기다리는 요청 수
