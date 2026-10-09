# 추론 지표 — TTFT, TPOT, ITL, Goodput, P99

> 네 가지 지표가 추론 배포가 정상적으로 작동하는지 결정합니다. TTFT는 프리필, 큐잉, 네트워크의 합입니다. TPOT (ITL과 동일)는 토큰당 메모리 바운드 디코딩 비용입니다. 엔드투엔드 지연은 TTFT에 TPOT을 출력 길이로 곱한 값입니다. 처리량은 전체 플릿(fleet)에 걸쳐 초당 토큰 수로 집계됩니다. 제품 측면에서 중요한 것은 goodput입니다. 모든 SLO를 동시에 충족한 요청의 비율입니다. 높은 처리량과 낮은 goodput은 사용자가 시간에 맞춰 도달하지 못하는 토큰을 처리하고 있다는 의미입니다. 2026년 TRT-LLM에서 Llama-3.1-8B-Instruct의 참고 수치: 평균 TTFT 162 ms, 평균 TPOT 7.33 ms, 평균 E2E 1,093 ms. 항상 P50, P90, P99를 보고하세요. 평균만 보고하지 마세요. 측정 함정에 주의하세요. GenAI-Perf는 ITL 계산에서 TTFT를 제외하지만, LLMPerf는 포함합니다. 두 도구가 동일한 실행에서 TPOT에 대해 의견이 다릅니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 장난감 백분율 계산기 및 goodput 리포터)
**선수 요건:** 17단계 · 04강 (서빙 엔진 내부 구조)
**시간:** 약 60분

## 학습 목표

- TTFT, TPOT, ITL, E2E, 처리량 및 goodput을 정확히 정의하고, 각 지표가 측정하는 구성 요소를 지정해 보세요.
- LLM 서빙에서 평균이 잘못된 통계인 이유와 P50/P90/P99를 읽는 방법을 설명해 보세요.
- SLO 다중 제약 조건(예: TTFT<500 ms AND TPOT<15 ms AND E2E<2 s)을 구성하고, 이를 기준으로 goodput을 계산해 보세요.
- 동일한 실행에서 TPOT에 대해 의견이 다른 두 벤치마킹 도구를 지정하고, 그 이유를 설명해 보세요.

## 문제점

"우리 처리량은 초당 15,000 토큰입니다." 그래서 뭐가 달라지나요? 요청의 40%가 엔드투엔드 2초를 초과했다면, 사용자는 세션을 포기했습니다. 처리량만으로는 제품이 작동하는지 알 수 없습니다.

추론에는 여러 축의 지연이 있으며, 각각은 서로 다른 방식으로 실패합니다. 프리필은 컴퓨팅 바운드이며 프롬프트 길이에 따라 확장됩니다. 디코딩은 메모리 바운드이며 배치 크기에 따라 확장됩니다. 큐잉 지연은 운영 문제입니다. 네트워크는 물리적 거리 문제입니다. 각각에 대해 고유한 지표가 필요하며, 백분위수가 필요하고, "사용자가 기대한 것을 얻었는지"를 나타내는 단일 복합 지표가 필요합니다. 이것이 바로 goodput입니다.

## 개념

### TTFT — 첫 토큰까지의 시간

`TTFT = queue_time + network_request + prefill_time`

프롬프트가 길면 프리필(Prefill)이 지배적입니다. H100에서 Llama-3.3-70B FP8의 경우, 32k 프롬프트는 순수 프리필에 약 800 ms가 소요됩니다. 큐 시간은 부하 하에서의 스케줄러 동작입니다. 네트워크 요청은 TLS를 포함한 전송 시간입니다. TTFT는 사용자가 스트리밍 시작 전에 체감하는 지연 시간입니다.

### TPOT / ITL — 토큰 간 지연 (ITL)(Inter-Token Latency (ITL))

하나의 양에 대한 여러 이름입니다. `TPOT` (출력 토큰당 시간 (TPOT)(Time per Output Token (TPOT))), `ITL` (토큰 간 지연 (ITL)(Inter-Token Latency (ITL))), `decode latency per token` — 모두 동일합니다. 첫 번째 토큰 이후 연속적으로 스트리밍되는 토큰 사이의 시간입니다.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

청크 프리필(Chunked Prefill)을 사용하는 동일한 Llama-3.3-70B H100 스택에서 TPOT 평균은 약 7 ms입니다. 청크 프리필이 없으면, 인접한 시퀀스의 긴 프리필 동안 TPOT가 50 ms까지 급증할 수 있습니다. 평균이 아닌 P99를 모니터링하세요.

### E2E 지연

`E2E = TTFT + TPOT * output_tokens + network_response`

긴 출력(500 토큰 초과)의 경우, E2E는 TPOT에 의해 지배됩니다. 긴 프롬프트를 가진 짧은 출력의 경우, E2E는 TTFT에 의해 지배됩니다. 출력 길이에 따른 조건부 E2E를 보고하세요.

### 처리량

`throughput = total_output_tokens / elapsed_time`

집계 지표입니다. 플릿(fleet) 효율성을 알려줍니다. 개별 요청의 건강 상태는 알려주지 않습니다.

### 순수 처리량(Goodput) — 실제로 중요하게 여기는 지표

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO는 다중 제약 조건입니다. 모든 제약 조건이 충족될 때만 요청이 "순수(good)"합니다. 순수 처리량은 그 비율입니다. 순수 처리량이 60%인 높은 처리량은 실패입니다. 순수 처리량이 99%인 낮은 처리량이 목표입니다.

2026년, 순수 처리량은 MLPerf Inference v6.0 제출 및 AI 플랫폼 제공업체의 내부 SLA 추적에 사용되는 지표입니다.

### 평균이 잘못된 통계인 이유

LLM 지연 분포는 오른쪽으로 치우쳐 있습니다. 긴 프리필을 가진 인접한 디코딩 배치에서는 TPOT가 약 7 ms인 500 토큰과 TPOT가 약 60 ms인 20 토큰을 전송할 수 있습니다. 평균 TPOT는 9 ms입니다. P99 TPOT는 65 ms입니다. 사용자는 P99에 정기적으로 직면합니다 — 이것이 그들이 이탈하는 이유입니다.

항상 세 쌍(P50, P90, P99)을 보고하세요. 사용자 경험 측면에서는 P99가 최적화해야 할 지표입니다.

### 참고 수치 — TRT-LLM의 Llama-3.1-8B-Instruct, 2026

- 평균 TTFT: 162 ms
- 평균 TPOT: 7.33 ms
- 평균 E2E: 1,093 ms
- P99 TPOT: 청크 프리필(Chunked Prefill) 구성에 따라 10-25 ms가 변동합니다.

이 값은 NVIDIA가 공개한 기준 포인트입니다. 모델 크기(70B는 3-5배), 하드웨어(H100 vs B200는 약 3배), 부하에 따라 달라집니다.

### 측정 함정

2026년 가장 많이 사용되는 벤치마크 도구 두 개가 동일한 실행에 대해 TPOT 값이 다릅니다:

- **NVIDIA GenAI-Perf**: TTFT를 ITL 계산에서 제외합니다. ITL은 토큰 2부터 시작합니다.
- **LLMPerf**: TTFT를 포함합니다. ITL은 토큰 1부터 시작합니다.

TTFT가 500 ms이고 총 디코딩 시간 700 ms에 출력 토큰 100개인 요청의 경우, GenAI-Perf는 `ITL = 700/99 = 7.07 ms`을 보고하고, LLMPerf는 `ITL = 1200/100 = 12.00 ms`을 보고합니다. 도구 선택에 따라 값이 달라집니다.

항상 사용 도구를 명시하세요. 항상 정의를 공개하세요.

### SLO 구성하기

2026년 70B 채팅 모델에 대한 합리적인 소비자 대상 SLO는 다음과 같습니다:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- 300 토큰 미만 출력에 대해 E2E P99 <= 3 s.
- 순수 처리량(Goodput) 목표 >= 99%.

엔터프라이즈 SLO는 TTFT를 더 엄격하게(200-400 ms) 설정하고 E2E는 완화합니다. 핵심은 이들을 문서화하고 세 가지 모두를 측정하며, 순수 처리량(Goodput)을 단일 복합 지표로 추적하는 것입니다.

### 측정 방법

- 실제 트래픽이나 현실적인 합성 트래픽(LLMPerf와 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`)을 실행하세요.
- 벤치마크 실행을 위해 피크 동시성의 2배를 목표로 하세요.
- 30-50번 반복 실행하고, 통합 샘플의 백분위수를 취하세요.
- 도구 이름, 도구 버전, 모델, 하드웨어, 동시성, 프롬프트 분포와 함께 공개하세요.

```figure
throughput-latency
```

## 사용하기

`code/main.py`은 순수 처리량(Goodput) 계산 도구입니다. 합성 지연 분포를 생성하고, SLO를 적용하여 순수 처리량(Goodput)을 계산합니다. 또한 동일한 추적(Trace)에서 GenAI-Perf와 LLMPerf의 TPOT 차이를 보여줍니다.

## 출시하기

이 강의는 `outputs/skill-slo-goodput-gate.md`을 생성합니다. 워크로드와 SLO가 주어지면, 처리량(Throughput)이 아닌 순수 처리량(Goodput)으로 배포를 게이트하는 CI/CD용 벤치마크 레시피를 생성합니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 꼬리 스파이크가 1%인 분포를 생성하세요. P99 TPOT를 30 ms에서 15 ms로 강화하면 순수 처리량(Goodput)이 어떻게 변합니까?
2. 벤더가 "Llama 3.3 70B H100에서 15,000 tok/s"라고 견적을 제시했습니다. 이를 신뢰하기 전에 물어봐야 할 세 가지 질문을 나열해 보세요.
3. 청킹 프리필이 P99 TPOT를 보호하는 이유는 무엇이며, 평균 TPOT에는 영향을 미치지 않는 이유는 무엇인가요?
4. 음성 어시스턴트(첫 토큰이 읽히는 것이 아니라 들리는 경우)에 대한 소비자 SLO를 구성해 보세요. 가장 사용자 친숙한 지표는 무엇인가요?
5. LLMPerf README와 GenAI-Perf 문서를 읽어 보세요. 두 도구가 서로 다른 세 가지 지표를 식별해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| TTFT | "첫 토큰까지의 시간" | 큐 + 네트워크 + 프리필; 긴 프롬프트에서는 프리필이 지배적 |
| TPOT | "출력 토큰당 시간" | 첫 토큰 이후의 토큰별 메모리 바운드 디코딩 비용 |
| ITL | "토큰 간 지연" | 대부분의 도구에서는 TPOT와 동일 (모든 도구는 아님 — GenAI-Perf 참조) |
| E2E | "엔드 투 엔드" | TTFT + TPOT * output_len; 응답 측 네트워크가 추가됨 |
| Throughput | "tok/s" | 플릿 효율성; 레이턴시 백분위수 없이는 무용지물 |
| Goodput | "SLO 충족률" | 모든 SLO 제약 조건을 동시에 충족하는 요청의 비율 |
| P99 | "꼬리" | 100분의 1 최악의 레이턴시; 사용자 경험 지표 |
| SLO 다중 제약 | "결합 조건" | 세 가지 레이턴시 상한의 AND; 하나라도 위반되면 요청이 실패 |
| GenAI-Perf vs LLMPerf | "도구 함정" | ITL에 TTFT가 포함되는지 여부에 대해 도구들이 의견이 다름 |

## 추가 읽기

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT, ITL, TPOT의 표준 정의.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 대안적 정의 및 측정 레시피.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 실제 배포 환경에서의 적용 측정.
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray 기반 오픈소스 벤치마크.
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) — NVIDIA의 벤치마크 도구.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 업계에서 인정하는 goodput 기반 벤치마크.
