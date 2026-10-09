# LLM API 부하 테스트 — k6와 Locust가 거짓말하는 이유

> 전통적인 부하 테스트 도구는 스트리밍 응답, 가변 출력 길이, 토큰 단위 지표, GPU 포화 상태를 고려하지 않고 설계되었습니다. 대부분의 팀이 두 가지 함정에 빠집니다. GIL 함정: Locust의 토큰 단위 측정 기능은 Python GIL 하에서 토큰화를 수행하며, 이는 높은 동시성 환경에서 요청 생성과 경쟁합니다. 토큰화 지연이 누적되면 보고된 토큰 간 지연(ITL)이 부풀려지는데, 이는 서버가 아니라 클라이언트가 병목 현상을 일으키기 때문입니다. 프롬프트 균일성 함정: 반복되는 동일한 프롬프트는 토큰 분포의 한 지점만 테스트합니다. 실제 트래픽은 길이가 다양하고 접두어 매칭이 다양합니다. LLMPerf는 `--mean-input-tokens` + `--stddev-input-tokens`을 사용하여 이를 해결합니다. 2026년 도구 매핑: 토큰 단위 정확도를 위해 LLM 특화 도구(GenAI-Perf, LLMPerf, LLM-Locust, guidellm)를 사용하세요. **k6 v2026.1.0** + **k6 Operator 1.0 GA (2025년 9월)**는 스트리밍을 인식하며, TestRun/PrivateLoadZone CRD를 통해 Kubernetes 네이티브 분산 테스트를 지원하므로 CI/CD 게이트에 가장 적합합니다. Vegeta는 Go의 일정 속도 포화 테스트에 적합합니다. Locust 2.43.3는 스트리밍을 위해 LLM-Locust 확장 프로그램과 함께 사용할 때만 적합합니다. 부하 패턴: 정상 상태(steady-state), 램프(ramp), 스파이크(spike, 자동 확장 테스트), 소크(soak, 메모리 누수).

**유형:** Build
**언어:** Python (표준 라이브러리, 사실적인 프롬프트 생성기 + 지연 수집기)
**선수 요건:** 17단계 · 08강 (추론 지표), 17단계 · 03강 (GPU 자동 확장)
**시간:** 약 75분

## 학습 목표

- LLM API에 대해 범용 부하 테스트 도구가 거짓말을 하게 만드는 두 가지 안티 패턴(GIL 함정, 프롬프트 균일성 함정)을 설명해 보세요.
- 목적에 맞는 도구를 선택해 보세요: LLMPerf (벤치마크 실행), k6 + 스트리밍 확장(CI 게이트), guidellm (대규모 합성), GenAI-Perf (NVIDIA 참조).
- 네 가지 부하 패턴(정상 상태, 램프, 스파이크, 소크)을 설계하고, 각 패턴이 포착하는 실패 모드를 지정해 보세요.
- 입력 토큰의 평균 + 표준 편차를 사용하여 고정 길이가 아닌 사실적인 프롬프트 분포를 구축해 보세요.

## 문제점

LLM 엔드포인트를 500명의 동시 사용자로 k6 테스트했습니다. 서비스가 버텼습니다. 출시했습니다. 실제 사용자는 200명뿐인 프로덕션 환경에서 서비스가 무너졌습니다. P99 TTFT가 폭발하고 GPU가 고정되었습니다.

두 가지 일이 발생했습니다. 첫째, k6는 500개의 동일한 프롬프트를 보냈습니다. 요청 병합(request-coalescing)과 접두어 캐싱(prefix caching) 덕분에 실제로는 하나의 요청만 처리하고 있었음에도, 마치 500개의 동시 디코딩을 처리하는 것처럼 보였습니다. 둘째, k6는 스트리밍 응답에서 눈이 경험하는 것과 같은 방식으로 토큰 간 지연(inter-token latency)을 추적하지 않습니다. k6는 하나의 HTTP 연결만 보고, 다양한 간격으로 도착하는 500개의 토큰을 보지 못합니다.

LLM을 위한 부하 테스트는 그 자체로 하나의 독립적인 분야입니다.

## 개념

### GIL 함정 (Locust)

Locust는 Python을 사용하며, GIL(GIL) 하에서 클라이언트 측 토큰화를 실행합니다. 높은 동시성 환경에서는 토크나이저가 요청 생성 뒤에 대기하게 됩니다. 보고되는 토큰 간 지연(inter-token latency)에는 클라이언트 측 토큰화 백로그가 포함됩니다. 서버가 느리다고 생각하지만, 실제로는 테스트 하네스(harness)가 느린 것입니다.

해결책: LLM-Locust 확장 프로그램은 토큰화를 별도 프로세스로 이동시키거나, 컴파일된 언어 하네스(k6, tokenizers.rs를 사용하는 LLMPerf)를 사용하세요.

### 프롬프트 균일성 함정

알려진 모든 부하 테스트 도구는 하나의 프롬프트만 설정할 수 있습니다. 10,000번 반복하는 루프 테스트에서는 매번 정확히 동일한 프롬프트가 전송됩니다. 서버는 매번 동일한 접두어를 보게 되며, 접두어 캐싱(prefix caching) 적중률이 100%에 가까워져 처리량(throughput)이 매우 좋아 보입니다.

해결책: 프롬프트 분포에서 샘플링하세요. LLMPerf는 `--mean-input-tokens 500 --stddev-input-tokens 150`를 사용합니다 — 다양한 길이와 다양한 내용으로 구성됩니다.

### 네 가지 부하 패턴

1. **정상 상태(Steady-state)** — 30~60분 동안 일정한 RPS. 포착 대상: 기준 성능 저하(baseline performance regressions).
2. **램프(Ramp)** — 15분 동안 RPS를 0에서 목표까지 선형적으로 증가시킴. 포착 대상: 용량 한계점(capacity breakpoint), 워밍업(warm-up) 이상 현상.
3. **스파이크(Spike)** — 2분 동안 RPS를 갑자기 3~10배로 증가시킨 후 원래대로 복귀. 포착 대상: 자동 확장(autoscaling) 지연, 큐 포화(queue saturation), 콜드 스타트(cold-start) 영향.
4. **소크(Soak)** — 4~8시간 동안 정상 상태 유지. 포착 대상: 메모리 누수(memory leaks), 연결 풀(connection-pool) 드리프트, 관측 가능성(observability) 오버플로.

### 2026년 도구 매핑

**LLMPerf** (Anyscale) — Python 기반이지만 Rust 백업 토큰화 사용. 평균/표준 편차 프롬프트. 스트리밍 인식. 성능 실행에 가장 적합한 기본 도구.

**NVIDIA GenAI-Perf** — NVIDIA의 참조 도구. Triton 클라이언트를 사용하며 포괄적인 지표 커버리지 제공. ITL이 TTFT를 제외하는 점에 유의하세요. LLMPerf의 ITL은 TTFT를 포함합니다. 두 도구는 동일한 서버에 대해 서로 다른 TPOT를 산출합니다.

**LLM-Locust** (TrueFoundry) — GIL 함정을 해결하는 Locust 확장 프로그램입니다. 익숙한 Locust DSL과 스트리밍 메트릭을 제공합니다.

**guidellm** — 대규모 합성 벤치마킹을 수행합니다.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025년 9월)**:
- k6 자체(Go, 컴파일됨, GIL 없음)는 스트리밍 인식 메트릭을 추가했습니다.
- k6 Operator는 Kubernetes 네이티브 분산 테스트를 위해 TestRun / PrivateLoadZone CRD를 사용합니다.
- CI/CD 게이트 및 SLA 테스트에 가장 적합합니다.

**Vegeta** — Go이며 k6보다 단순합니다. 일정 속도의 HTTP 포화 테스트를 수행합니다. LLM 인식 기능은 없지만 게이트웨이 / 속도 제한 테스트에 좋습니다.

**Locust 2.43.3 기본 버전** — LLM의 경우 GIL 함정이 있습니다. LLM-Locust 확장 프로그램과 함께 사용할 때만 유용합니다.

### CI에서의 SLA 게이트

PR에 대해 k6를 실행합니다:

- 기본 RPS에서 각각 30-50번 반복합니다.
- 게이트: P50/P95 TTFT, 5xx < 5%, TPOT가 임계값 미만이어야 합니다.
- 임계값을 초과하면 빌드를 중단합니다.

### 현실적인 프롬프트 분포

실제 트래픽 샘플(있는 경우)이나 공개된 분포(예: 채팅용 ShareGPT 프롬프트, 코드용 HumanEval)에서 구축하세요. 평균과 표준 편차를 LLMPerf에 입력하세요. 어떤 비용이 들더라도 단일 프롬프트 반복 루프는 피해야 합니다.

### 기억해야 할 수치

- k6 Operator 1.0 GA: 2025년 9월.
- k6 v2026.1.0: 스트리밍 인식 메트릭.
- 일반적인 LLMPerf 실행: 동시성 X에서 100-1000개 요청.
- 일반적인 CI 게이트: PR당 30-50번 반복.
- 네 가지 패턴: steady, ramp, spike, soak.

```figure
load-pattern-waves
```

## 사용하기

`code/main.py`는 현실적인 프롬프트 분포로 부하 테스트를 시뮬레이션하고, 유효 TPOT를 측정하며, 균일한 프롬프트 함정을 시연합니다.

## 출시하기

이 강의는 `outputs/skill-load-test-plan.md`를 생성합니다. 워크로드와 SLA가 주어지면 도구를 선택하고 네 가지 부하 패턴을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 균일한 분포와 현실적인 분포를 비교해 보세요 — 격차는 어디에 있나요?
2. CI 게이트용 k6 스크립트를 작성하세요: 동시성 100에서 TTFT P95 < 800 ms, 실행 시간 5분.
3. soak 테스트에서 메모리가 시간당 50 MB 증가합니다. 세 가지 원인을 나열하고, 그중 하나를 선택하기 위한 계측 장치를 지정하세요.
4. 10 RPS에서 100 RPS로 스파이크 테스트를 수행합니다. Karpenter + vLLM production-stack이 적용된 상태(17단계 · 03 + 18)에서 예상 복구 시간은 얼마입니까?
5. GenAI-Perf는 TPOT=6ms를 보고하고, LLMPerf는 같은 서버에서 TPOT=11ms를 보고합니다. 그 이유를 설명해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| LLMPerf | "the LLM harness" | Anyscale 벤치마크 도구, 스트리밍 인식 |
| GenAI-Perf | "NVIDIA tool" | NVIDIA 참조 하네스 |
| LLM-Locust | "Locust for LLMs" | GIL 함정을 해결한 Locust 확장 |
| guidellm | "synthetic benchmark" | 대규모 합성 도구 |
| k6 Operator | "K8s k6" | CRD 기반 분산 k6 |
| GIL trap | "Python client overhead" | 토큰화 백로그가 보고된 지연 시간을 부풀림 |
| Prompt-uniformity trap | "single-prompt lie" | 동일한 프롬프트로 반복하면 캐시에 적중하여 처리량이 부풀려짐 |
| Steady-state | "constant load" | N분 동안 평평한 RPS |
| Ramp | "linear up" | 기간 동안 0에서 목표까지 선형 증가 |
| Spike | "burst test" | 갑작스러운 배수 증가 후 복귀 |
| Soak | "long test" | 누수 감지를 위한 수 시간 테스트 |

## 추가 읽기

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
