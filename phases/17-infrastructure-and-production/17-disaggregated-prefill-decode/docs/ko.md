# 분산 서빙 프리필/디코딩 — NVIDIA Dynamo 및 llm-d

> 프리필은 연산 집약적(compute-bound)이고, 디코딩은 메모리 집약적(memory-bound)입니다. 두 작업을 같은 GPU에서 실행하면 한쪽 자원이 낭비됩니다. 분산 서빙은 이를 별도의 풀로 분리하고, NIXL (RDMA/InfiniBand 또는 TCP 폴백)을 통해 KV 캐시를 전송합니다. NVIDIA Dynamo (GTC 2025 발표, 1.0 GA)는 vLLM/SGLang/TRT-LLM 위에 위치하며, Planner Profiler + SLA Planner가 SLO를 충족하기 위해 프리필:디코딩 비율을 자동으로 매칭합니다. NVIDIA는 이 범위의 처리량 향상분을 공개하고 있습니다. developer.nvidia.com (2025-06)에 따르면 GB200 NVL72 + Dynamo 환경에서 DeepSeek-R1 MoE의 중간 지연(medium-latency) 영역에서 약 6배 개선이 확인되며, Dynamo 제품 페이지 (developer.nvidia.com, 날짜 미기재)는 GB300 NVL72 + Dynamo가 Hopper 대비 MoE 처리량을 최대 50배까지 향상시킨다고 광고합니다. "30x" 수치는 전체 스택 Blackwell + Dynamo + DeepSeek-R1 보고서에 대한 커뮤니티 집계이며, 정확히 30x라고 명시한 단일 원천 자료는 발견하지 못했으므로 방향성 있는 주장으로 취급해야 합니다. llm-d (Red Hat + AWS)는 Kubernetes 네이티브로, 프리필 / 디코딩 / 라우터를 독립적인 Service로 구성하며 역할별 HPA를 적용합니다. llm-d 0.5는 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, scale-to-zero를 추가합니다. 경제성: 여러 고객 공개 자료의 내부 집계에 따르면, 상수 SLA에서 colocated serving에서 Dynamo를 사용한 disaggregated serving으로 전환할 경우 $2M-class inference spend (i.e., $600-800K/year) 비용의 30–40% 절감이 예상됩니다. 구체적인 $2M→$600-800K 수치는 단일 공개 사례 연구가 아닌 내부 복합 추정치이므로, 참고 인용이 아닌 규모 추정 앵커로 사용해야 합니다. 짧은 프롬프트 (<512 토큰, 짧은 출력)는 전송 비용을 정당화하지 못합니다.

**유형:** Learn
**언어:** Python (stdlib, 분산 서빙 vs colocated 시뮬레이터)
**선수 요건:** 17단계 · 04강 (서빙 엔진 내부 구조), 17단계 · 08강 (추론 지표)
**시간:** 약 75분

## 학습 목표

- 프리필과 디코딩이 최적 GPU 할당량이 다른 이유를 설명하고, colocated 환경에서의 낭비를 정량화합니다.
- 분산 서빙 아키텍처를 다이어그램으로 표현합니다: 프리필 풀, 디코딩 풀, NIXL을 통한 KV 전송, 라우터.
- 분산 서빙이 효과가 없는 조건을 명시합니다 (짧은 프롬프트, 짧은 출력).
- NVIDIA Dynamo (스택 상층)와 llm-d (Kubernetes 네이티브)를 구분하고, 각각을 운영 환경에 매칭해 보세요.

## 문제점

8개의 H100 GPU에서 Llama 3.3 70B를 실행합니다. 혼합 워크로드(긴 프롬프트 + 짧은 출력)에서는 대부분의 연산이 프리필에 사용되어 디코딩 중에 GPU가 유휴 상태가 됩니다. 다른 워크로드(짧은 프롬프트 + 긴 출력)에서는 그 반대 상황이 발생합니다. 프리필과 디코딩을 한곳에 배치하면 두 작업 모두 과잉 프로비저닝하게 됩니다.

예산 영향: GPU 시간의 20-40%가 잘못된 자원에 낭비됩니다. 메모리 바운드 디코딩을 위해 H100 연산력을 구매하거나, 연산 바운드 프리필을 위해 H100 HBM 대역폭을 구매하는 것은 모두 비싼 낭비입니다.

분산 서빙(Disaggregated Serving)은 프리필(Prefill)과 디코딩을 각각의 병목에 맞춰 크기를 조정한 별도 풀로 분리합니다. KV 캐시(KV Cache)는 프리필 풀에서 디코딩 풀로 고대역폭 인터커넥트를 통해 전송됩니다.

## 개념

### 병목 현상이 다른 이유

**프리필(Prefill)** — 전체 입력 프롬프트를 한 번의 순전파로 트랜스포머를 실행합니다. 행렬 곱셈이 지배적이며, 연산에 제한을 받습니다. H100 FP8는 약 2000 TFLOPS의 유용한 처리량을 제공합니다. 배치 효율이 좋습니다 — 한 번의 순전파로 많은 토큰을 처리합니다.

**디코딩** — 한 번에 하나의 토큰을 생성하며, 각 반복에서 전체 가중치를 읽습니다. 메모리 대역폭에 의해 제한됩니다. HBM3는 약 3 TB/s를 제공합니다. 배치 효율은 높은 동시성에서만 좋습니다 — 가중치 읽기는 배치 전체에 걸쳐 상각됩니다.

함께 배치하기: 두 용도에 모두 최적화된 GPU를 구매합니다. H100은 두 용도 모두에 적합하지만, 어느 용도로 사용하든 비용은 동일합니다. 대규모 환경에서는 프리필(prefill) 풀을 H100 / 연산 집약적 구성으로, 디코딩(decode) 풀을 H200 / 메모리 집약적 구성으로 배치하거나, 공격적인 양자화(quantization)를 적용하는 것이 좋습니다.

### 아키텍처

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL은 NVIDIA의 노드 간 전송 기술입니다. RDMA/InfiniBand가 사용 가능하면 이를 활용하고, 그렇지 않으면 TCP로 폴백합니다. 전송 지연은 실제로 존재하며, 70B FP8 모델에서 4K 토큰 프롬프트의 KV 캐시 전송에는 일반적으로 20-80 ms가 소요됩니다. 짧은 프롬프트는 분산 서빙(Disaggregated Serving)을 정당화하지 못하는 이유가 바로 이것입니다. 전송 비용이 절감 효과를 초과하기 때문입니다.

### Dynamo vs llm-d

**NVIDIA Dynamo** (GTC 2025 발표, 1.0 GA):
- vLLM, SGLang, TRT-LLM 위에 오케스트레이터(Orchestration)로 위치합니다.
- Planner Profiler가 워크로드를 측정하고, SLA Planner가 프리필:디코딩 비율을 자동 구성합니다.
- Rust 코어, Python 확장성.
- 처리량 향상: NVIDIA는 GB200 NVL72 + Dynamo 환경에서 DeepSeek-R1 MoE의 중간 지연 영역에서 6배 향상된 처리량을 보고했습니다 (developer.nvidia.com, 2025-06). 전체 Blackwell + Dynamo + DeepSeek-R1 스택에서 "최대 30배"라는 커뮤니티 보고서는 단일 주요 출처가 없으므로 방향성 지표로만 취급해야 합니다.
- GB300 NVL72 + Dynamo: Dynamo 제품 페이지에 따르면 Hopper 대비 MoE 처리량이 최대 50배 향상됩니다 (developer.nvidia.com, 날짜 미기재).

**llm-d** (Red Hat + AWS, Kubernetes 네이티브):
- 프리필 / 디코딩 / 라우터를 독립적인 Kubernetes 서비스로 구성합니다.
- 각 역할별 HPA는 프리필의 큐 깊이 / 디코딩의 KV 활용률 신호를 사용합니다.
- `topologyConstraint packDomain: rack`는 고대역폭 KV 전송을 위해 프리필+디코딩 클리크를 동일한 랙에 배치합니다.
- llm-d 0.5 (2026): 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, 스케일 투 제로(scale-to-zero) 지원.

관리형 스택 상위 오케스트레이터를 원한다면 Dynamo를 사용하세요. Kubernetes 네이티브 프리미티브를 원하고 CNCF 생태계에 전념한다면 llm-d를 사용하세요.

### 경제성

내부 복합 지표 (단일 공개 사례 연구가 아님 — 자릿수 기준 앵커):

- 컬로케이티드(colocated) 서빙에서 연 200만 달러의 추론 비용이 발생합니다.
- Dynamo를 사용하여 분산(disaggregated) 방식으로 전환했습니다.
- 동일한 요청 볼륨, 동일한 P99 지연 SLA를 유지합니다.
- 보고된 절감액: $600K–$ 연 80만 달러 (30–40% 감소).
- 새로운 하드웨어는 필요하지 않습니다.

이 수치는 단일 인용 가능한 사례 연구가 아니라 여러 고객 공개 자료로부터 합성된 것입니다. 가장 가까운 공개 데이터 포인트는 Dynamo KV 라우팅을 통해 TTFT가 2배 빨라지고 처리량이 61% 증가한 Baseten의 사례 (baseten.co, 2025-10)이며, VAST + CoreWeave는 KV 적중률 40–60%에서 토큰/$가 60–130% 더 많다고 예측합니다 (vastdata.com, 2025-12). 절감 효과는 각 풀을 적정 크기로 조정(right-sizing)하는 데서 나오며, 프리필 중심 워크로드(8K+ 접두어를 가진 RAG)는 균형 잡힌 워크로드보다 더 큰 혜택을 받습니다.

### 분산(disaggregation)을 하지 말아야 할 경우

- 프롬프트가 512 토큰 미만이고 출력이 200 토큰 미만인 경우: 전송 비용이 이득을 압도합니다.
- 소규모 클러스터 (GPU 4개 미만): 풀 다양성이 충분하지 않습니다.
- 팀이 역할별 스케일링으로 두 개의 GPU 풀을 운영할 수 없는 경우: Dynamo가 도움이 되지만 간단하지는 않습니다.
- RDMA 패브릭이 없는 경우: TCP 전송 비용이 더 무겁습니다.

### 라우터는 17단계 · 11강과 통합됩니다

분산 라우터는 KV 캐시를 인식합니다(17단계 · 11강). 요청은 해당 접두어를 보유한 디코딩 풀로 전달되며, 매칭이 없으면 프리필 → 디코딩 순서로 처리됩니다. 적중률과 분산은 상호 보완적입니다. 캐시 인식 라우터는 새로운 프리필이 필요한지 여부를 결정합니다.

### Blackwell에서의 MoE는 실제 성능 수치가 나타나는 영역입니다

GB300 NVL72 + Dynamo는 Hopper 기준선 대비 MoE 처리량이 50배 향상됨을 보여줍니다. MoE 전문가 라우팅은 프리필 시 연산 집약적이지만 디코딩 시 메모리 집약적(전문가 캐시)이므로, 분산은 이중의 이점을 제공합니다. 2026년 프론티어 모델 서빙은 MoE 중심입니다(DeepSeek-V3, 미래의 GPT-5 변형).

### 기억해야 할 수치

벤치마크 수치는 변동합니다. NVIDIA와 추론 스택은 매 분기마다 업데이트된 결과를 게시합니다. 인용하기 전에 재확인하세요.

- GB200 NVL72 + Dynamo에서의 DeepSeek-R1: 중간 지연 레짐에서 기준선 대비 약 6배 처리량(developer.nvidia.com, 2025-06). 전체 Blackwell + Dynamo 스택에 대한 커뮤니티의 "최대 30배" 주장은 단일 주요 출처가 없는 방향성 집계입니다.
- GB300 NVL72 + Dynamo: Hopper 대비 MoE 처리량 최대 50배(developer.nvidia.com, 날짜 미기재).
- 절감 기준(내부 복합 지표, 단일 사례 연구 아님): $600-800K/year off a $ SLA 유지 시 연 200만 달러 지출.
- 분산 임계값: 프롬프트 >512 토큰 + 출력 >200 토큰.
- NIXL을 통한 KV 전송: 70B FP8의 4K 프롬프트 KV에 대해 20-80 ms.

```figure
prefill-decode-split
```

## 사용하기

`code/main.py`은 콜로케이티드(colocated) 서빙과 분산 서빙을 시뮬레이션합니다. 처리량, 요청당 비용, 프롬프트 길이 교차점을 보고합니다.

## 출시하기

이 강의는 `outputs/skill-disaggregation-decider.md`을 생성합니다. 워크로드와 클러스터를 고려하여 분산 여부를 결정합니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 어떤 프롬프트 길이에서 분산이 콜로케이티드(colocated) 방식보다 우월합니까?
2. P99 접두어 길이 8K, 출력 300인 RAG 서비스용 프리필 풀과 디코딩 풀을 설계하세요.
3. Python 런타임 선호도가 없는 순수 Kubernetes 환경에서 Dynamo와 llm-d 중 하나를 선택하세요.
4. KV 전송 비용을 계산하세요: 70B FP8의 4K 프리필은 약 500 MB KV입니다. RDMA 100 GB/s에서는 전송 시간이 5 ms입니다. TCP 10 GB/s에서는 50 ms입니다. SLA에 어떤 것이 더 중요합니까?
5. MoE 전문가 라우팅은 KV 접근 패턴을 변경합니다. 토큰마다 다른 전문가를 활성화하는 MoE 환경에서 분산 서빙은 어떻게 동작하나요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 분산 서빙(Disaggregated Serving) | "프리필/디코딩 분리" | 각 단계별로 분리된 GPU 풀 |
| NIXL | "NVIDIA 전송" | Dynamo의 노드 간 KV 전송 (RDMA/TCP) |
| NVIDIA Dynamo | "오케스트레이터" | vLLM/SGLang/TRT-LLM을 위한 스택 상위 조정자 |
| llm-d | "Kubernetes 네이티브" | Red Hat + AWS K8s 분산 서빙 스택 |
| Planner Profiler | "Dynamo 자동 구성" | 워크로드를 측정하고 풀 비율을 구성 |
| SLA Planner | "Dynamo 정책" | SLO를 충족하기 위해 프리필:디코딩 비율을 자동으로 매칭 |
| `packDomain: rack` | "llm-d 토폴로지" | 빠른 KV를 위해 프리필+디코딩을 같은 랙에 배치 |
| UCCL | "통합 콜렉티브" | scale-to-zero를 위한 llm-d 0.5 네트워킹 계층 |
| MoE 전문가 라우팅 | "토큰별 전문가" | DeepSeek-V3 패턴; 분산 서빙이 도움이 됨 |

## 추가 읽기

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
