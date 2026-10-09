# 캡스톤 14 — 추론적 디코딩 추론 서버

> 추론적 디코딩(Speculative Decoding) — 값싼 초안 모델이 토큰을 제안하고, 대상 모델이 이를 한 번의 패스로 검증하는 방식 — 은 이제 연구용 트릭이 아닌 프로덕션 준비가 완료된 최적화입니다. vLLM 0.7의 EAGLE-3는 실제 트래픽에서 2.5-3배의 처리량을 제공합니다. P-EAGLE (AWS 2026)는 병렬 추론을 한층 더 발전시켰습니다. SGLang의 SpecForge는 대규모로 초안 헤드를 훈련했습니다. Red Hat의 Speculators 허브는 일반적인 오픈 모델에 대한 정렬된 초안을 공개했습니다. TensorRT-LLM은 NVIDIA에서 추론적 디코딩을 일급 시민으로 만들었습니다. 2026년 프로덕션 서빙 스택은 vLLM 또는 SGLang에 EAGLE 계열 초안, FP8 또는 INT4 양자화, 그리고 큐 대기 기반 HPA를 사용하는 구성입니다. 이 캡스톤의 목표는 두 개의 오픈 모델을 2.5배 이상의 기본 처리량으로 서빙하고, 완전한 꼬리 지연(Tail Latency) 보고서를 작성하는 것입니다.

**유형:** Capstone
**언어:** Python (서빙), C++ / CUDA (커널 검사), YAML (설정)
**선수 요건:** 3단계 (딥러닝), 7단계 (트랜스포머), 10단계 (LLM从零부터 구축), 17단계 (인프라)

**활용 단계:** P3 · P7 · P10 · P17
**시간:** 30시간

## 문제점

추론적 디코딩은 2026년에 상품화되었습니다. EAGLE-3 초안 헤드는 대상 모델의 은닉 상태(hidden states)에서 훈련되며 N개의 토큰을 미리 예측합니다. 대상 모델은 이를 한 번의 패스로 검증합니다. 60-80%의 수용률(acceptance rate)은 2-3배의 엔드투엔드 처리량으로 이어집니다. vLLM 0.7은 이를 네이티브로 통합했습니다. SGLang + SpecForge는 훈련 파이프라인을 제공합니다. Red Hat의 Speculators는 Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B에 대한 정렬된 초안을 공개합니다.

핵심은 모델이 아닌 서빙 운영에 있습니다. 수용률은 트래픽 분포(ShareGPT vs 코드 vs 도메인 데이터)에 따라 변동합니다. 거부(rejection) 발생 시 꼬리 지연은 추론적 디코딩을 사용하지 않는 경우보다 더 나빠질 수 있습니다 — 따라서 steady-state tokens/sec뿐만 아니라 여러 배치 크기에서의 p99를 보고해야 합니다. Anthropic / OpenAI API 대비 100만 토큰당 비용은 신뢰성을 확보하는 레버입니다.

## 개념

추론적 디코딩은 두 개의 계층으로 구성됩니다. **초안(draft)** 모델(EAGLE-3 헤드, ngram, 또는 더 작은 대상 정렬 모델)은 단계당 k개의 후보 토큰을 제안합니다. **대상(target)** 모델은 모든 k를 한 번의 패스로 검증하며, 수용된 접두어는 greedy 경로를 대체합니다. 수용률은 초안-대상 정렬 및 입력 분포에 따라 달라집니다.

EAGLE-3는 대부분의 트래픽에서 ngram 초안보다 우월합니다. P-EAGLE는 더 깊은 초안 트리를 위해 병렬 추론을 수행합니다. 트레이드오프는 거부(rejection) 시 P99 지연이 더 높다는 점입니다. 이는 검증 패스가 더 크기 때문입니다. 서빙 구성은 이 문제를 드러내기 위해 배치 크기별 버킷화된 지연 시간을 보고해야 합니다.

배포는 Kubernetes입니다. vLLM 0.7은 GPU당 하나의 레플리카 또는 텐서 병렬화(Tensor Parallelism) 샤드에서 실행됩니다. HPA는 CPU가 아닌 큐 대기(queue-wait) 지표에 따라 자동 확장합니다. FP8 (Marlin) 및 INT4 (AWQ) 양자화(Quantization)는 GPU 메모리를 H100 / H200 범위 내에 유지합니다. 엔드투엔드(end-to-end) 보고서는 처리량, 수용률, 배치 1/8/32에서의 p50/p99 지연, 그리고 토큰 100만 개당($) 비용입니다.

## 아키텍처

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 스택

- 서빙: vLLM 0.7 또는 SGLang 0.4
- 추론적 디코딩(Speculative Decoding) 방법: EAGLE-3 초안 헤드, P-EAGLE 병렬 추론, ngram 폴백
- 초안 학습: SpecForge (SGLang) 또는 Red Hat Speculators
- 타겟 모델: Llama 3.3 70B, Qwen3-Coder-30B MoE (혼합 전문가)(MoE (Mixture of Experts)), GPT-OSS-120B
- 양자화(Quantization): FP8 (Marlin), INT4 AWQ
- 배포: Kubernetes + NVIDIA 디바이스 플러그인; 큐 대기 지표 기반 HPA
- 평가(Evaluation (Eval)): ShareGPT, MT-Bench-v2, GSM8K, HumanEval을 사용하여 도메인별 수용률 측정
- 참고: 벤더 기준선(vendor baseline)을 위한 TensorRT-LLM 추론적 디코딩(Speculative Decoding)

```figure
cf-spec-decode
```

## 구현하기

1. **타겟 모델 준비.** Llama 3.3 70B를 선택하세요. Marlin을 통해 FP8로 양자화(Quantization)하세요. vLLM 0.7에서 1xH100 (또는 2x 텐서 병렬화(Tensor Parallelism))으로 배포하세요.

2. **초안 소스.** Red Hat Speculators에서 정렬된 EAGLE-3 초안 헤드를 가져오세요 (또는 SpecForge를 통해 하나를 학습하세요). vLLM의 추론적 디코딩(Speculative Decoding) 구성에 로드하세요.

3. **기준 수치.** 추론적 디코딩(Speculative Decoding) 전에: 배치 1/8/32에서의 토큰/초, p50/p99 지연, GPU 사용률. 이를 공개하세요.

4. **EAGLE-3 활성화.** 구성을 전환하세요. 동일한 벤치마크를 다시 실행하세요. 속도 향상, 수용률, p99 꼬리 지연(tail-latency) 델타를 보고하세요.

5. **P-EAGLE.** 병렬 추론을 활성화하세요. 직렬 EAGLE-03강 비교하여 더 깊은 초안 트리를 측정하세요. P-EAGLE가 도움이 되는 지점과 해가 되는 지점의 변곡점을 보고하세요.

6. **도메인 트래픽.** ShareGPT, HumanEval, 도메인 특화 트래픽을 동일한 서버를 통해 실행하세요. 분포별 수용률을 측정하세요. 초안이 드리프트(drift)하는 시점을 식별하세요.

7. **두 번째 대상 모델.** Qwen3-Coder-30B MoE에 동일한 파이프라인을 실행하세요. 초안 생성이 더 어렵습니다(MoE 라우팅 잡음). 보고서를 작성하세요.

8. **K8s HPA.** `queue_wait_ms`을 추적하는 HPA를 사용하여 K8s에 배포하세요. 부하가 3배 증가할 때 스케일아웃을 시연하세요.

9. **비용 비교.** 동일한 평가에서 Anthropic Claude Sonnet 4.7 및 OpenAI GPT-5.4와 비교하여 토큰 100만 개당($) 비용을 계산하세요. 공개하세요.

## 사용하기

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 출시하기

`outputs/skill-inference-server.md`은 산출물을 설명합니다. 추론적 디코딩(Speculative Decoding)을 포함한 측정된 서빙 스택, 전체 벤치마크 보고서, K8s 배포가 포함됩니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 기준선 대비 측정된 속도 향상 | 두 모델에서 품질이 동일한 상태에서 2.5배 이상의 처리량 |
| 20 | 실제 트래픽에서의 수용률 | 분포별 수용률 보고서 |
| 20 | P99 꼬리 지연(Tail Latency) 관리 | 배치 크기 1/8/32에서 추론적 디코딩 적용 여부에 따른 p99 |
| 20 | 운영 | K8s 배포, 큐 대기 시간 기반 HPA, 매끄러운 롤아웃 |
| 15 | 문서화 및 방법론 | 변경된 내용과 이유에 대한 명확한 설명 |
| **100** | | |

## 연습 문제

1. 초안 모델이 대상 모델보다 한 버전 뒤처질 때(예: Llama 3.3 -> 3.4 드리프트) 수용률 저하를 측정하세요. 모니터링 경보를 구축하세요.

2. ngram 폴백을 구현하세요: EAGLE-3 수용률이 임계값 아래로 떨어지면 ngram 초안으로 전환합니다. 신뢰성 개선 사항을 보고하세요.

3. 제어된 MoE 실험을 실행하세요: 동일한 Qwen3-Coder-30B에 라우팅 잡음을 주입한 경우와 주입하지 않은 경우를 비교합니다. 초안 수용 민감도를 측정하세요.

4. H200(141 GB)으로 확장하세요. 레플리카당 모델 크기 여유(headroom)가 얼마나 증가했는지, 그리고 양자화되지 않은 Llama 3.3 70B를 서빙할 수 있는지 보고하세요.

5. 동일한 H100 하드웨어에서 TensorRT-LLM 추론적 디코딩(Speculative Decoding)을 벤치마킹하세요. vLLM 대비 우월한 부분을 보고하세요.

## 핵심 용어

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 초안 모델 | "Speculator" | 대상 모델이 검증할 N개의 토큰을 제안하는 소형 모델 |
| EAGLE-3 | "2026 초안 아키텍처" | 대상 모델의 은닉 상태(hidden states)에 대해 학습된 초안 헤드; 약 75% 수용률 |
| P-EAGLE | "병렬 추측" | 하나의 타겟 패스에서 검증된 초안 가지의 트리 |
| 수용률 | "히트율" | 재샘플링 없이 수용된 초안 토큰의 비율 |
| 양자화 | "FP8 / INT4" | GPU 메모리에 더 많은 모델을 수용하기 위한 저정밀도 가중치 |
| 큐 대기 | "HPA 지표" | 추론이 시작되기 전 대기열에서 요청이 대기하는 시간 |
| 추측기 허브 | "정렬된 초안" | 일반적인 오픈 모델을 위한 EAGLE 초안 모음인 Red Hat Neural Magic 허브 |

## 추가 읽기

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 참조 서빙 스택
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 병렬 추론적 디코딩 논문 + 통합
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 초안 헤드 학습 파이프라인
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 정렬된 초안 허브
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 벤더 대안
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 상용 참조
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 방법론 논문
- [vLLM repository](https://github.com/vllm-project/vllm) — 코드 및 벤치마크
