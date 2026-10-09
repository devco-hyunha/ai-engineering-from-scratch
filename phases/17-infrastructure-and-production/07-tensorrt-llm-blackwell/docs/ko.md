# 하드웨어 특화 추론 컴파일 — Blackwell에서의 FP8 및 NVFP4

> 하드웨어 특화 추론 컴파일은 이식성을 희생하고 처리량을 얻는 트레이드오프이며, TensorRT-LLM — NVIDIA 전용으로 Blackwell에 최적화됨 — 이 트레이드오프가 성공한 가장 명확한 예입니다. GB200 NVL72에서 Dynamo 오케스트레이션과 함께 SemiAnalysis InferenceX는 H100 + vLLM 대비 $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $0.09/M를 측정했습니다 — 이는 7배의 경제적 격차입니다. 이 스택은 세 가지 부동 소수점 체제가 결합된 형태입니다: FP8는 KV 캐시와 어텐션 커널에 필요한 동적 범위를 제공하므로 여전히 중요합니다; NVFP4 (4비트 마이크로 스케일링)는 가중치와 활성화를 처리합니다; 다중 토큰 예측 (MTP)과 분리된 프리필/디코딩은 추가로 2-3배의 성능을 더합니다. Day-0 모델 지원은 학습 후 변환 없이 FP4 가중치를 직접 로드합니다. 2026년 엔지니어링 팀을 위한 주의점: TRT-LLM은 오픈소스이지만 NVIDIA 전용 — CUDA 및 Blackwell 특화 — 이므로, 이를 채택하면 이식성을 희생하고 처리량을 얻게 됩니다. 모델과 하드웨어 구성에 대한 계산을 수행한 후 결정하세요.

**유형:** Learn
**언어:** Python (표준 라이브러리, FP8/NVFP4 메모리 및 비용 계산기 토이)
**선수 요건:** 17단계 · 04강 (서빙 엔진 내부 구조), 10단계 · 13강 (양자화)
**시간:** 약 75분

## 학습 목표

- 가중치가 NVFP4로 변환된 경우에도 KV 캐시와 어텐션에 FP8가 여전히 중요한 이유를 설명해 보세요.
- 프론티어 모델의 HBM 점유 공간을 BF16, FP8, NVFP4에서 계산하고, 절약 효과가 어디서 오는지 추론해 보세요.
- TRT-LLM이 활용하는 Blackwell 특화 기능 (Day-0 FP4, MTP, 분리된 서빙, all-to-all 프리미티브)을 나열해 보세요.
- Hopper에서의 vLLM 대비 7배 비용 격차에 대해 TRT-LLM의 NVIDIA 락이 가치 있는 경우를 결정해 보세요.

## 문제점

2026년 추론 경제학의 프론티어는 "달러당 토큰 수"입니다. 답은 네 가지 중첩된 선택에 따라 달라집니다: 하드웨어 세대 (Hopper H100/H200 vs Blackwell B200/GB200), 정밀도 (BF16 → FP8 → NVFP4), 서빙 엔진 (vLLM vs SGLang vs TRT-LLM), 그리고 오케스트레이션 (단순 vs 분리형 vs Dynamo).

Hopper에서 vLLM을 사용할 경우, 120B MoE는 약 $0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at ~$0.012로 실행되며, 이는 7배 더 저렴합니다. 이 격차의 일부는 하드웨어에 기인합니다 (Blackwell은 Hopper 대비 GPU당 LLM 처리량이 11-15배 높습니다). 나머지 일부는 스택에 기인합니다: FP4 가중치, MTP 초안, 분산 서빙(prefill/decode), 그리고 MoE 전문가 통신을 위한 NVLink 5 all-to-all.

NVIDIA의 스택 외부에서는 이를 복제할 수 없습니다. 이것이 트레이드오프입니다 — 이식성과 경제성의 교환입니다. 어떤 스택 선택이 격차의 어느 부분을 차지하는지 이해하는 것이 이 강의의 핵심입니다.

## 개념

### KV 캐시에서 FP8이 여전히 하한선인 이유

2026년의 흔한 실수: NVFP4가 모든 곳에 적용된다고 가정하는 것입니다. 그렇지 않습니다. KV 캐시는 넓은 동적 범위를 가진 어텐션 키와 값을 저장하므로 FP8 (8비트 부동 소수점)이 필요합니다. KV를 FP4로 양자화하면 치명적인 정확도 손실이 발생합니다 — 분포의 꼬리가 사라지고 어텐션 점수가 붕괴됩니다. FP8의 지수 비트는 KV 캐시에 필요한 범위를 제공합니다.

NVFP4 (2025-2026)는 가중치와 활성화에 적용됩니다. 마이크로 스케일링: 각 가중치 블록은 자체 스케일 팩터를 가지므로, 작은 블록들은 텐서 단위 스케일 손실 없이 서로 다른 동적 범위를 포괄할 수 있습니다. 활성화의 경우, FP4가 잘 작동합니다. 왜냐하면 활성화는 레이어 내에서 작은 범위를 가지기 때문입니다.

일반적인 Blackwell 구성:

- 가중치: NVFP4 (4비트 마이크로 스케일링).
- 활성화: NVFP4.
- KV 캐시: FP8.
- 어텐션 어큐뮬레이터: FP32 (소프트맥스 안정성).

### TRT-LLM이 사용하는 Blackwell 전용 프리미티브

- **Day-0 FP4 가중치**: 모델 제공업체가 FP4 가중치를 직접 출시합니다; TRT-LLM은 학습 후 변환 없이 로드합니다. FP4에 대한 AWQ / GPTQ 단계가 없습니다.
- **다중 토큰 예측 (MTP)**: EAGLE (17단계 · 05강)와 동일한 개념이지만, TRT-LLM 빌드에 통합되어 있습니다.
- **분산 서빙**: prefill과 decode를 분리된 GPU 풀에서 수행하며, KV 캐시는 NVLink 또는 InfiniBand를 통해 전송됩니다. Dynamo (17단계 · 20강)와 동일한 개념입니다.
- **All-to-all 통신 프리미티브**: NVLink 5는 Hopper 대비 MoE 전문가 통신 지연을 3배 감소시킵니다. TRT-LLM의 MoE 커널은 이를 위해 튜닝되었습니다.
- **NVFP4 + MXFP8 마이크로 스케일링**: Blackwell Tensor Cores에서 하드웨어 가속화된 스케일 팩터 처리.

### 암기해야 할 수치

- GPT-OSS-120B를 TRT-LLM으로 실행할 때 HGX B200의 비용은 토큰당 $0.02/M입니다.
- Dynamo (TRT-LLM 오케스트레이션)를 통해 GB200 NVL72의 비용은 토큰당 $0.012/M입니다.
- 비교 가능한 작업 부하에서 H100 + vLLM은 토큰당 약 $0.09/M입니다.
- 2026년 TRT-LLM 업데이트를 통해 3개월 동안 처리량이 2.8배 증가했습니다.
- Blackwell은 Hopper 대비 GPU당 LLM 처리량이 11-15배 높습니다.
- MLPerf Inference v6.0 (2026년 4월): Blackwell은 제출된 모든 작업에서 우위를 점했습니다.

### FP4가 품질에 실제로 미치는 비용

NVFP4는 공격적입니다. 추론 중심 작업 부하 (사고의 연쇄, 수학, 긴 컨텍스트 코드 생성)에서는 FP4 가중치가 눈에 띄게 저하됩니다. 블록별 보정은 완화하지만 제거하지는 않습니다. 추론 모델을 출시하는 팀은 종종 FP8 가중치 + FP4 활성화를 절충안으로 사용하거나, 전체적으로 H200에서 FP8을 유지합니다.

규칙: NVFP4 가중치를 확정하기 전에 평가 세트에서 작업 품질을 항상 검증하세요.

### NVIDIA 락(Lock) 결정인 이유

TRT-LLM은 C++ + CUDA + 비공개 소스 커널입니다. 모델은 특정 GPU SKU에 맞춰 컴파일해야 합니다. AMD, Intel, ARM은 지원되지 않습니다. 인프라 전략이 멀티 벤더라면, TRT-LLM이 제공하는 계층에서는 TRT-LLM이 시작점이 될 수 없습니다. 혼합 하드웨어에서 vLLM으로 서빙할 수는 있습니다. NVIDIA 전용이라면, 7배의 격차가 락(Lock) 비용을 상쇄합니다.

### 2026년 실용적인 레시피

연간 추론 비용이 $100M 이상인 경우, Hopper + vLLM으로 실행하면 7-10배의 잠재력을 놓치게 됩니다. 비용 지배적인 작업 부하를 Blackwell + TRT-LLM + Dynamo로 마이그레이션하세요. 모델 반복 속도 향상을 위해 실험 계층은 H100 + vLLM으로 유지하세요. 프로덕션 전에 각 NVFP4 변환 모델의 품질을 검증하세요.

### 분산 서빙의 보너스

TRT-LLM의 분산 서빙 (프리필 및 디코딩 풀 분리)은 17단계 · 20강에서 상세히 다루고 있습니다. Blackwell에서는 배수가 누적됩니다: FP4 가중치 × MTP 가속 × 분산 배치 × 캐시 인식 라우팅. 7배 수치는 이 전체 스택을 가정합니다.

```figure
pipeline-parallel
```

## 사용하기

`code/main.py`는 세 가지 스택 (H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM)에 걸쳐 모델의 HBM 점유량, 디코딩 처리량 (메모리 바운드 레짐), 토큰당 $/M를 계산합니다. 이를 실행하여 누적 효과와 각 변경이 격차에 기여하는 비중을 확인해 보세요.

## 출시하기

이 강의는 `outputs/skill-trtllm-blackwell-advisor.md`를 생성합니다. 워크로드, 모델 크기, 연간 토큰 볼륨을 고려하여 Blackwell + TRT-LLM 스택이 NVIDIA 락인(NVIDIA-lock)을 정당화하는지 결정합니다.

## 연습 문제

1. `code/main.py`를 실행해 보세요. 활성 파라미터가 30%인 120B MoE 모델에서 H100 BF16, H100 FP8, B200 NVFP4/FP8의 메모리 대역폭 제한 디코딩 처리량을 계산해 보세요. 가장 큰 성능 향상은 어디에서 발생하나요?
2. 고객이 H100 + vLLM에 연 200만 달러를 지출합니다. 7배의 경제적 격차를 고려할 때, TRT-LLM으로의 마이그레이션을 12개월 내에 상각(amortize)하기 위해 필요한 Blackwell GPU의 손익분기점(break-even) 개수는 몇 개인가요?
3. NVFP4 가중치 변환 후 MATH 벤치마크에서 정확도가 3포인트 하락했습니다. 두 가지 복구 경로를 제시하세요: 하나는 품질 우선(FP8 가중치 유지), 다른 하나는 비용 우선(도메인 내 데이터로 보정(calibrate)).
4. MLPerf v6.0 추론 결과를 읽어 보세요. Blackwell과 Hopper 간 격차가 가장 작은 작업은 무엇이며, 그 이유는 무엇인가요?
5. 128k 컨텍스트에서 NVFP4 가중치 + FP8 KV 캐시를 사용하는 405B 모델에 필요한 HBM을 계산해 보세요. 단일 GB200 NVL72 노드에서 수용 가능한가요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| FP8 | "8비트 부동소수점" | 8비트 부동소수점; 동적 범위(dynamic range)로 인해 KV 캐시와 어텐션에 사용됨 |
| NVFP4 | "4비트 마이크로" | NVIDIA의 4비트 마이크로 스케일링(microscaling) FP 형식; Blackwell에서 가중치와 활성화에 사용 |
| MXFP8 | "MX 8비트" | 마이크로 스케일링(microscaling) FP8 변형; Blackwell Tensor Cores에서 하드웨어 가속됨 |
| Day-0 FP4 | "FP4 가중치 출시" | 모델 제공자가 이미 FP4 형식의 가중치를 출시함; 사후 학습 변환 단계가 없음 |
| MTP | "다중 토큰 예측" | TRT-LLM의 통합 추론적 디코딩(Speculative Decoding) 초안(draft) (17단계 · 05강) |
| 분산 서빙(Disaggregated Serving) | "프리필/디코딩 분리" | 프리필(Prefill)과 디코딩(Decode)을 분리된 GPU 풀에서 수행; KV는 NVLink/IB를 통해 전송 |
| All-to-all | "MoE 전문가 통신" | 토큰을 전문가 GPU로 라우팅하는 통신 패턴; NVLink 5로 3배 감소 |
| InferenceX | "SemiAnalysis 추론 벤치마크" | 2026년 업계에서 인정하는 토큰당 비용(cost-per-token) 벤치마크 |

## 추가 읽기

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026년 4월 MLPerf 결과.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 및 MoE 커널(kernel).
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 공식 엔진 문서입니다.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 상위의 분산 서빙 오케스트레이션입니다.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Blackwell 성능 수치를 공개하는 벤치마크 스위트입니다.
