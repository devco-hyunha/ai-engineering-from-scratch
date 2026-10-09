# 다중 지역 LLM 서빙 및 KV 캐시 지역성

> 라운드 로빈 로드 밸런싱은 캐시된 LLM 추론에 actively harmful( actively 해로운)입니다. 캐시된 접두어를 가진 요청이 해당 노드에 도착하지 않으면 전체 프리필 비용을 지불하게 되며, 긴 프롬프트의 경우 P50 기준 약 800ms가 소요되는 반면 캐시 적중 시에는 약 80ms가 소요됩니다. 2026년 생산 환경에서는 KV 캐시 이벤트를 소비하고 접두어 해시 일치에 따라 라우팅하는 캐시 인식 라우터(vLLM Router in Rust, llm-d router)가 표준 패턴입니다. 최근 연구(GORGO)는 라우팅 목표에 지역 간 네트워크 지연을 명시적인 항으로 포함합니다. 상용 "지역 간 추론(cross-region inference)" 제품(Bedrock cross-region inference, GKE multi-cluster gateways)은 추론을 불투명하게 처리하며, 가용성을 관리할 뿐 TTFT는 관리하지 않습니다. JPMorgan과 Mayo Clinic은 2024년 11월에 us-east-1 페일오버를 약 22분 동안 수행했습니다. DR(재해 복구)의 현실은 다음과 같습니다: LLM DR 실패의 32%는 팀이 가중치를 백업했지만 토크나이저 파일이나 양자화 구성을 잊어버렸기 때문에 발생합니다.

**유형:** Learn
**언어:** Python (stdlib, toy prefix-cache-aware router simulator)
**선수 요건:** 17단계 · 04강 (vLLM Serving), 17단계 · 06강 (SGLang RadixAttention)
**시간:** 약 60분

## 학습 목표

- 라운드 로빈 로드 밸런싱이 캐시된 추론을 왜 깨뜨리는지 설명하고 TTFT 페널티를 정량화하세요.
- 캐시 인식 라우터를 다이어그램으로 표현하세요: 입력(KV 캐시 이벤트), 알고리즘(접두어 해시 일치), 타이브레이커(GPU 사용률).
- LLM의 DR 실패 원인 32%(누락된 토크나이저 파일 / 양자화 구성)를 명시하고 3개 파일 DR 체크리스트를 제시하세요.
- 상용 지역 간 추론 제품(Bedrock CRI, GKE Multi-Cluster Gateway)과 KV 인식 라우팅을 구분하세요.

## 문제점

서비스가 us-east-1, us-west-2, eu-west-1에서 실행됩니다. ALB를 앞에 두고 라운드 로빈을 적용했습니다. 생산 환경에서 접두어 캐시 적중률이 8%로 떨어졌습니다. TTFT P50이 3배 증가했습니다. vLLM 로그는 모든 요청이 전체 프리필 비용을 지불하고 있음을 보여줍니다.

라운드 로빈은 상태 비저장(stateless) 서비스에 최적입니다. LLM 추론은 설계상 상태가 있습니다 — KV 캐시는 모델이 본 모든 것을 인코딩합니다. 블라인드 라우팅은 잘못된 캐시로 라우팅하는 것입니다.

별도로, 팀에는 DR 계획이 있습니다. 모델 가중치를 S3에 지역 간 백업합니다. 지역 장애가 발생하면 페일오버를 시도하지만, 복제본이 시작을 거부합니다. tokenizer.json, 양자화 설정, RoPE 스케일링 설정이 동기화하지 않은 별도 버킷에 있었다는 것을 잊었습니다.

다중 지역 LLM 서빙은 캐시 문제, 라우팅 문제, DR 위생 문제이며, 로드 밸런서 문제가 아닙니다.

## 개념

### 캐시 인식 라우팅

프롬프트가 포함된 요청이 도착합니다. 라우터는 접두어(예: 첫 512개 토큰)를 해싱하고, 각 복제본에 "이 접두어가 캐시되어 있나요?"라고 묻습니다. 복제본은 블록을 할당하고 제거할 때 pub/sub 채널에 KV 캐시 이벤트를 게시합니다. 라우터는 일치하는 복제본을 선택하며, 일치하는 복제본이 없으면 GPU 사용량 기반의 타이브레이커로 폴스루합니다.

**vLLM Router** (Rust, 2026 프로덕션 스택): `kv.cache.block_added` 이벤트를 구독하고, 접두어 해시 → 복제본 인덱스를 유지하며, O(1) 조회로 라우팅합니다. 일치하는 항목이 없으면 최소 큐 깊이로 폴스루합니다.

**llm-d router**: 동일한 패턴, Kubernetes 네이티브. ControlPlane API를 통해 이벤트를 게시합니다.

**SGLang RadixAttention** (17단계 · 06)은 복제본 내부의 동등한 기능입니다. 복제본 간 라우팅은 엄격히 상류(upstream)입니다.

### 수치

2K 토큰 프롬프트, Llama 3.3 70B FP8, H100에서의 TTFT P50:
- 캐시 적중 (동일 복제본, 접두어 상주): ~80 ms.
- 캐시 미스 (콜드 프리필): ~800 ms.

10배 차이. 라우터가 복제본 간 접두어 캐시를 60-80% 적중시키면, N개 복제본 용량으로 단일 복제본 성능에 근사합니다. 10% 적중률에서는 단순한 스케일링에 근사합니다.

### 지역 간에는 새로운 제약이 있습니다 — 네트워크 지연

지역 간 RTT:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

us-east-1에서 ap-southeast-1의 핫 접두어로 요청을 라우팅하면, 절약된 프리필 (800 → 80 ms)은 440 ms 왕복 지연에 의해 압도됩니다. GORGO (2026 연구)는 이를 명시합니다 — 프리필만 고려하지 말고 `prefill_time + network_latency`를 공동으로 최소화하세요. 종종 해답은 대규모 다중 MB 접두어처럼 프리필이 지배적인 경우를 제외하고 라우팅을 지역적으로 유지하는 것입니다.

### 상용 "지역 간 추론"은 이 상황에서 도움이 되지 않습니다

AWS Bedrock의 지역 간 추론은 용량 압박이 있을 때 요청을 다른 지역으로 자동으로 라우팅합니다. 이는 가용성을 최적화하며 TTFT를 최적화하지는 않으며, 추론을 불투명하게 처리합니다. GKE Multi-Cluster Gateway도 동일합니다 — 서비스 수준 페일오버이며 KV 캐시를 인식하지 못합니다.

이러한 기능을 사용하더라도 여전히 앱 계층의 캐시 인식 라우터가 필요합니다. 이들은 "us-east-1이 다운된" 경우를 처리합니다. 캐시 인식 라우팅은 TTFT 경우를 처리합니다.

### DR 위생 — 32% 파일 누락 문제

2026년 광범위하게 인용된 통계: LLM DR 실패의 32%는 팀이 가중치를 백업했지만 다음을 잊어버렸기 때문에 발생합니다:

- `tokenizer.json` 또는 `tokenizer.model`
- 양자화 구성 (`quantize_config.json`, AWQ 스케일, GPTQ 제로 포인트)
- 모델별 구성 (RoPE 스케일링, 어텐션 마스크, 채팅 템플릿)
- 엔진 구성 (`vllm_config.yaml`, 샘플링 기본값, LoRA 어댑터 매니페스트)

해결책은 최소 3개 파일의 DR 매니페스트입니다:

1. HF 모델 저장소 하의 모든 파일 (가중치 + 구성 + 토크나이저).
2. 엔진별 서빙 구성.
3. 배포 매니페스트 (K8s YAML, Dockerfile, 의존성 잠금).

추가로: 분기마다 DR 훈련을 수행하세요. JPMorgan의 us-east-1 훈련은 플레이북이 반복 연습되었기 때문에 2024년 11월에 22분 복구 시간을 달성했습니다.

### 데이터 주권은 직교합니다

EU 고객의 PHI는 EU를 떠날 수 없습니다. 캐시 인식 라우터가 파리에서 발생한 요청을 접두어 매칭을 위해 us-east-1로 보내면 TTFT 이득과 관계없이 GDPR을 위반한 것입니다. 캐시를 최적화하기 전에 주권 경계별로 라우터를 분할하세요.

### 기억해야 할 수치

- 캐시 적중 vs 미스 TTFT 격차: 약 10배 (2K 프롬프트에서 80 ms vs 800 ms).
- 지역 간 RTT US-EU: 약 75 ms.
- DR 실패: 32%가 토크나이저/양자화 구성을 누락합니다.
- JPMorgan us-east-1 페일오버 2024년 11월: 22분 (30분 SLA).

```figure
cache-aware-router
```

## 사용하기

`code/main.py`는 다중 지역 워크로드에서 세 가지 라우팅 전략 (라운드 로빈, 캐시 인식 지역별, 캐시 인식 전역)을 시뮬레이션합니다. 캐시 적중률, TTFT P50/P99 및 지역 간 청구서를 보고합니다.

## 출시하기

이 강의는 `outputs/skill-multi-region-router.md`를 생성합니다. 지역, 거주 요건, SLA가 주어지면 라우팅 계획을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행해 보세요. RTT가 75 ms일 때, 어떤 프롬프트 길이에서 지역 간 라우팅이 지역 전용 라우팅보다 우월합니까?
2. 캐시 적중률이 70%에서 12%로 떨어졌습니다. 세 가지 가능한 원인을 진단하고, 각각을 확인할 수 있는 관측 지표는 무엇입니까?
3. vLLM에서 5개의 LoRA 어댑터와 함께 서빙되는 70B AWQ 양자화 모델에 대한 DR 매니페스트를 설계하세요. 모든 파일과 구성을 나열합니다.
4. 엄격한 TTFT SLO를 가진 핀테크 기업에 Bedrock 지역 간 추론이 "충분한지" 논증하세요. 구체적인 동작을 인용하세요.
5. 파리에서 발생한 요청이 us-east-1의 접두어와 일치합니다. 라우팅하시겠습니까? 정책을 작성하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 캐시 인식 라우팅 | "스마트 LB" | KV 캐시를 보유한 복제본의 접두어 해시 일치에 따라 라우팅 |
| KV 캐시 이벤트 | "캐시 발행-구독" | 복제본이 블록 추가/제거를 발행하고 라우터가 인덱싱 |
| 접두어 해시 | "캐시 키" | 라우터 조회에 사용되는 첫 N개 토큰의 해시 |
| GORGO | "지역 간 라우팅 연구" | arXiv 2602.11688; 네트워크 지연을 명시적 항으로 사용 |
| 지역 간 추론 | "Bedrock CRI" | AWS 제품; 가용성 페일오버이며 TTFT 인식은 아님 |
| DR 매니페스트 | "백업 목록" | 복구에 필요한 모든 파일 — 가중치만 포함하는 것이 아님 |
| 데이터 거주 요건 | "GDPR 경계" | 사용자 데이터를 어떤 지역이 볼 수 있는지에 대한 법적 제약 |
| RTT | "왕복 시간" | 네트워크 지연; US-EU는 75 ms, US-APAC는 220 ms |
| LLM 인식 LB | "캐시 적중 LB" | 캐시 인식 라우터를 제품 범주로 정의 |

## 추가 읽기

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 네트워크 지연 항을 포함한 지역 간 KV 캐시 재사용.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 가용성 페일오버 문서.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 캐시 인식 라우터 소스.
