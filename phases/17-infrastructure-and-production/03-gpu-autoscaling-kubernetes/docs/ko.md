# Kubernetes에서의 GPU 자동 확장 — Karpenter, KAI Scheduler, Gang Scheduling

> 세 개의 계층, 하나의 계층이 아닙니다. Karpenter는 노드를 동적으로 프로비저닝합니다 (1분 미만, Cluster Autoscaler보다 40% 빠름). KAI Scheduler는 gang scheduling, 토폴로지 인식, 계층적 큐를 처리합니다 — 7개의 노드가 대기하며 하나의 GPU가 부족해 연소되는 7-of-8 부분 할당 함정을 방지합니다. 애플리케이션 수준의 자동 확장기(NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler)는 추론 특화 신호 — 큐 깊이, KV 캐시 활용률 — 에 따라 확장하며, CPU/DCGM duty cycle에는 반응하지 않습니다. 고전적인 HPA 함정은 `DCGM_FI_DEV_GPU_UTIL`가 duty-cycle 측정값이라는 점입니다: 100%는 10개 요청일 수도, 100개 요청일 수도 있습니다. vLLM은 KV 캐시 메모리를 사전 할당하므로, 메모리는 scale-down을 트리거하지 않습니다. 이 강의는 세 계층을 조합하고, 실행 중인 GPU 작업을 추론 중에 종료하는 기본 Karpenter `WhenEmptyOrUnderutilized` 정책을 피하는 방법을 가르쳐 줍니다.

**유형:** Learn
**언어:** Python (stdlib, toy queue-depth autoscaler simulator)
**선수 요건:** 17단계 · 02강 (추론 플랫폼 경제학), 17단계 · 04강 (서빙 엔진 내부 구조)
**시간:** 약 75분

## 학습 목표

- 세 개의 자동 확장 계층(노드 프로비저닝, gang scheduling, 애플리케이션 수준)을 다이어그램으로 표현하고, 각 계층에서 사용되는 도구를 명시해 보세요.
- vLLM에 대해 `DCGM_FI_DEV_GPU_UTIL`가 HPA 신호로 부적합한 이유를 설명하고, 두 가지 대체 신호(큐 깊이, KV 캐시 활용률)를 명시해 보세요.
- Gang scheduling과 KAI Scheduler가 방지하는 부분 할당 실패 모드(8개 GPU 중 7개가 유휴 상태)를 설명해 보세요.
- 실행 중인 GPU 작업을 종료하는 Karpenter 통합 정책(`WhenEmptyOrUnderutilized`)을 명시하고, 2026년 안전한 대안을 서술해 보세요.

## 문제점

팀이 Kubernetes에서 LLM 서빙 서비스를 출시합니다. HPA를 `DCGM_FI_DEV_GPU_UTIL` 신호로 설정합니다. 서비스는 업무 시간 동안 100% 활용률에 고정됩니다. HPA는 절대 scale-up하지 않습니다 — 이미 포화 상태라고 판단하기 때문입니다. 수동으로 레플리카를 추가하면 TTFT가 감소합니다. HPA는 여전히 scale-up하지 않습니다. 신호가 거짓말을 하고 있습니다.

별도로, 노드 프로비저닝에 Cluster Autoscaler를 사용합니다. 새벽 2시에 1M 토큰 프롬프트가 도착합니다; 클러스터는 노드를 프로비저닝하는 데 3분이 소요되며, 요청은 타임아웃됩니다.

또한, 2개 노드에 걸쳐 8개의 GPU가 필요한 70B 모델을 배포한다고 가정해 보세요. 클러스터에는 7개의 GPU가 비어 있고, 나머지 1개는 3개 노드에 분산되어 있습니다. Cluster Autoscaler는 부족한 GPU 1개를 위해 노드를 프로비저닝합니다. Kubernetes가 마지막 GPU를 준비하는 동안 7개 노드가 4분간 대기하며 비용을 낭비합니다.

세 가지 계층, 세 가지 다른 실패 모드. 2026년의 GPU 인식 오토스케일링은 단순히 "HPA를 켜는 것"이 아닙니다. 노드 프로비저닝, 갱 스케줄링, 애플리케이션 신호 기반 오토스케일링을 조합하는 것입니다.

## 개념

### 계층 1 — 노드 프로비저닝 (Karpenter)

Karpenter는 대기 중인(pending) 파드를 감시하며 약 45~60초 내에 노드를 프로비저닝합니다 (Cluster Autoscaler는 GPU 노드의 경우 일반적으로 90~120초가 소요됩니다). Karpenter는 `NodePool` 제약 조건에 따라 인스턴스 유형을 동적으로 선택합니다. 파드가 8개의 H100을 필요로 하는데 클러스터에 일치하는 노드가 없다면, Karpenter는 기존 그룹을 확장하는 대신 직접 노드를 프로비저닝합니다.

**통합(consolidation) 함정**: Karpenter의 기본 `consolidationPolicy: WhenEmptyOrUnderutilized` 설정은 GPU 풀에 위험합니다. 실행 중인 GPU 노드를 종료하고 파드를 더 저렴하고 크기가 적절한(right-sized) 인스턴스로 마이그레이션합니다. 추론 워크로드의 경우, 이는 실행 중인 요청을 강제 종료(evict)하고 새 노드에서 70B 모델을 다시 로드해야 함을 의미합니다. 손실은 몇 분간의 용량 손실과 요청 실패로 이어집니다.

GPU 풀을 위한 안전한 설정:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Karpenter가 완전히 비어 있는 노드를 1시간 후에 통합하도록 허용하되, 실행 중인 작업은 절대 강제 종료하지 않도록 설정합니다.

### 계층 2 — 갱 스케줄링 (KAI Scheduler)

KAI Scheduler (프로젝트 "Karp"에서 이름이 변경됨)는 기본 kube-scheduler가 처리하지 못하는 부분을 담당합니다:

**갱 스케줄링** — 전부를 함께 실행하거나 전부 실행하지 않는(all-or-nothing) 스케줄링. 8개의 GPU가 필요한 분산 추론 파드는 8개 모두 함께 시작하거나, 하나도 시작하지 않습니다. 이 기능이 없으면 부분 할당 함정에 빠집니다: 8개 파드 중 7개만 시작하고, 무한정 대기하며 비용을 낭비합니다.

**토폴로지 인식** — 어떤 GPU가 NVLink를 공유하는지, 어떤 GPU가 같은 랙에 있는지, 어떤 GPU 사이에 InfiniBand가 연결되어 있는지 파악합니다. 이에 따라 파드를 배치합니다. DeepSeek-V3 67B 텐서 병렬화(Tensor Parallelism) 워크로드는 하나의 NVLink 도메인에 머물러야 하며, KAI Scheduler는 이를 존중합니다.

**계층형 큐** — 여러 팀이 동일한 GPU 풀을 우선순위와 할당량(quota)을 두고 경쟁합니다. 팀 A의 프로덕션(pinch)은 우선순위 규칙이 허용하는 경우에만 팀 B의 학습(training) 작업에 의해 선점(preempt)됩니다.

KAI는 kube-scheduler와 함께 보조 스케줄러로 배포되며, 워크로드에 이를 사용하도록 주석(annotate)을 추가합니다. Ray와 vLLM production-stack 모두 통합되어 있습니다.

### 3계층 — 애플리케이션 수준 신호

**HPA의 함정**: `DCGM_FI_DEV_GPU_UTIL`는 듀티 사이클(duty-cycle) 지표입니다. 이는 각 샘플링 간격에서 GPU가 작업을 수행했는지를 측정합니다. 100% 사용률은 동시 요청이 10개일 수도, 100개일 수도 있습니다. GPU는 어느 경우에도 바빴습니다. 듀티 사이클에 기반한 스케일링은 맹목적인 스케일링입니다.

더 나쁜 점은, vLLM 및 유사한 엔진이 KV 캐시 메모리를 사전 할당(최대 `--gpu-memory-utilization`)한다는 것입니다. 요청이 하나뿐이어도 메모리 사용률은 90%에 가깝게 유지됩니다. 메모리 기반 HPA는 절대 축소되지 않습니다.

**2026년 대체 신호**:

- 큐 깊이(prefill을 기다리는 요청 수).
- KV 캐시 사용률(활성 시퀀스에 할당된 블록의 비율).
- 레플리카별 P99 TTFT (SLA 신호).
- 순수 처리량(Goodput)(초당 모든 SLO를 충족하는 요청 수).

NVIDIA Dynamo Planner와 llm-d Workload Variant Autoscaler는 이러한 신호를 소비하고 레플리카를 스케일링합니다. LLM 서빙에서는 HPA를 완전히 대체합니다.

### 무엇을 언제 사용할지

| 스케일링 결정 | 도구 |
|----------------|------|
| 노드 추가/제거 | Karpenter |
| 다중 GPU 작업 스케줄링 | KAI Scheduler |
| 레플리카 추가/제거 | Dynamo Planner / llm-d WVA (또는 큐 깊이 기반 커스텀 HPA) |
| GPU 유형 선택 | Karpenter NodePool |
| 낮은 우선순위 선점 | KAI Scheduler 큐 |

### 분리된 prefill/decode는 모든 것을 복잡하게 만듭니다

분리된 prefill/decode (17단계 · 17강)를 실행하면, 서로 다른 스케일링 트리거를 가진 두 가지 pod 클래스가 존재합니다. prefill pod는 큐 깊이에 따라 스케일링되고, decode pod는 KV 캐시 압력에 따라 스케일링됩니다. llm-d는 이를 `Services`로 노출하며 역할별 HPA를 제공합니다. 두 pod에 단일 HPA를 적용하려고 시도하지 마세요.

### 콜드 스타트도 여기서 중요합니다

콜드 스타트 완화 (17단계 · 10강)는 노드 프로비저닝 시간이 사용자에게 가시적으로 되는 지점입니다. Karpenter의 45-60초 워밍업, 20GB 모델 로드, 엔진 초기화 때문에 제로 상태에서의 요청은 2-5분이 소요됩니다. SLO가 중요한 경로에는 워밍 풀(`min_workers=1`)을 유지하거나, 애플리케이션 레이어에서 Modal 스타일의 체크포인팅을 사용하세요.

### 기억해야 할 수치

- Karpenter 노드 프로비저닝: 약 45-60초 vs Cluster Autoscaler 약 90-120초 (GPU 노드).
- KAI Scheduler는 부분 할당 낭비를 방지합니다 — 7-of-8 함정.
- HPA 신호로 `DCGM_FI_DEV_GPU_UTIL` 사용: 깨진 상태; 큐 깊이 또는 KV 활용률을 사용하세요.
- Karpenter `WhenEmptyOrUnderutilized`: 실행 중인 GPU 작업을 종료합니다. 추론에는 `WhenEmpty + consolidateAfter: 1h`를 사용하세요.

```figure
autoscaling
```

## 사용하기

`code/main.py`는 버스트형 GPU 워크로드에서 3계층 오토스케일러를 시뮬레이션합니다. 단순 HPA (듀티 사이클), 큐 깊이 HPA, KAI 갱(gang) 스케줄링 스케일링을 비교합니다. 충족되지 않은 요청 수, 유휴 GPU 분, 복합 점수를 보고합니다.

## 출시하기

이 강의는 `outputs/skill-gpu-autoscaler-plan.md`를 생성합니다. 클러스터 토폴로지, 워크로드 형태, SLO가 주어지면 3계층 오토스케일링 계획을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 버스트형 워크로드에서 단순 듀티 사이클 HPA가 놓친 요청 중 큐 깊이 HPA가 잡는 요청은 몇 개입니까? 차이점은 어디에서 발생합니까?
2. H100 SXM5에서 Llama 3.3 70B FP8를 서빙하는 클러스터용 Karpenter NodePool을 설계하세요. `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter` 및 GPU가 아닌 워크로드가 이 노드에서 실행되지 않도록 하는 테인트(taint)를 지정하세요.
3. 팀이 "GPU는 가용하지만 포드가 스케줄링되지 않아" 배포가 Pending 상태에 갇혀 있다고 보고합니다. 진단하세요 — 이것이 Karpenter, kube-scheduler, KAI Scheduler 중 어느 쪽 문제입니까? 어떤 지표가 이를 확인합니까?
4. 분산 프리필(prefill) 포드를 오토스케일링할 신호와 디코딩(decode) 포드를 오토스케일링할 다른 신호를 선택하세요. 두 신호 모두에 대한 근거를 제시하세요.
5. 하루 평균 60건의 요청 드롭(drop) 이벤트가 발생하고 P99 TTFT가 10초를 초과하는 24x7 프로덕션 서비스에서 `WhenEmptyOrUnderutilized` 통합(integration) 함정의 비용을 계산하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Karpenter | "노드 프로비저너" | Kubernetes 노드 오토스케일러; 1분 미만 프로비저닝 |
| Cluster Autoscaler | "구형 스케일러" | Kubernetes 노드 오토스케일러 전신; 느리고 그룹 기반 |
| KAI Scheduler | "GPU 스케일러" | 갱(gang) + 토폴로지 + 큐를 위한 보조 스케일러 |
| 갱(gang) 스케줄링 | "전부 또는 전부 없음" | N개의 포드를 원자적으로 스케줄링하거나 모두 지연 |
| 토폴로지 인식 | "rack-aware" | NVLink/IB/랙 배치에 따라 팟을 배치합니다 |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | 듀티 사이클 지표; LLM용 스케일링 신호가 아닙니다 |
| 큐 깊이 | "waiting requests" | 프리필 바운드 스케일링을 위한 올바른 HPA 신호입니다 |
| KV 캐시 활용률 | "memory pressure" | 디코딩 바운드 스케일링을 위한 올바른 HPA 신호입니다 |
| 통합 | "Karpenter consolidation" | 더 저렴한 인스턴스 유형으로 노드를 종료합니다 |
| `WhenEmpty + 1h` | "safe consolidation" | 실행 중인 GPU 작업을 강제로 제거하지 않는 정책입니다 |

## 추가 읽기

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 설계 문서 및 구성 예시입니다.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 통합 정책 의미론 및 GPU 안전 기본값입니다.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 스케일링 신호입니다.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 통합 패턴입니다.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 관리형 Kubernetes 전용 가이드입니다.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 워크로드 변형 오토스케일러 설계입니다.
