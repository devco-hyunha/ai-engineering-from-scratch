---
name: gpu-autoscaler-plan
description: Kubernetes 기반 LLM 서빙 클러스터용 3계층 GPU 자동 확장 계획(Karpenter + KAI Scheduler + 애플리케이션 신호)을 설계합니다. DCGM_FI_DEV_GPU_UTIL 함정과 부분 할당 실패를 진단합니다.
version: 1.0.0
phase: 17
lesson: 03
tags: [kubernetes, gpu, autoscaling, karpenter, kai-scheduler, hpa, dynamo-planner, llm-d]
---

클러스터 토폴로지(노드, GPU 유형, NVLink 도메인), 워크로드 형태(TP/PP 구성, 평균 동시성, 버스트 팩터), SLO(TTFT P99, 순수 처리량)가 주어지면 3계층 자동 확장 계획을 생성합니다.

생성할 내용:

1. 계층 1 — Karpenter NodePool. `instance-type`, `capacity-type` (on-demand / spot / reserved), `consolidationPolicy` (GPU 풀은 `WhenEmpty`와 `consolidateAfter: 1h`이어야 함), 비-GPU 워크로드를 제외하는 테인트(taint), KAI Scheduler 선택을 위한 레이블을 지정합니다.
2. 계층 2 — KAI Scheduler 정책. 갱(gang) 스케줄링이 필요한지 명시합니다(TP/PP > 1인 경우 예). 토폴로지 제약(NVLink 도메인, 랙, 존)을 정의합니다. 프로덕션 및 학습 테넌트 간의 큐 계층 구조와 선점(preemption) 규칙을 지정합니다.
3. 계층 3 — 애플리케이션 자동 확장기. 신호를 선택합니다: 프리필(prefill) 중심 워크로드의 경우 큐 깊이, 디코딩(decode) 중심의 경우 KV 캐시 활용률, 혼합 워크로드의 경우 복합 순수 처리량. `DCGM_FI_DEV_GPU_UTIL`를 금지하고 그 이유를 설명합니다.
4. 분리(disaggregated) 분할. 17단계 · 17강의 분리된 프리필/디코딩을 사용하는 경우, 프리필 풀용 큐 깊이 신호와 디코딩 풀용 KV 활용률 신호로 분리된 HPA를 지정합니다.
5. 워밍(warm) 풀 크기 조정. P99 TTFT 제약과 관측된 콜드 스타트 시간(노드 프로비저닝 + 모델 로드)을 기반으로 SLO에 중요한 경로에 대한 최소 준비된(replica) 수를 설정합니다.
6. 모니터링. 대시보드에 표시할 지표: 레플리카별 큐 깊이, 레플리카별 KV 활용률, 노드 프로비저닝 대기 시간, 갱 스케줄링 지연 횟수, Karpenter 통합(consolidation) 이벤트.

거부 조건:
- `DCGM_FI_DEV_GPU_UTIL`에 HPA를 권장하는 경우. 거부하고 큐 깊이와 KV 활용률이 올바른 신호임을 명시합니다.
- GPU 풀에 `consolidationPolicy: WhenEmptyOrUnderutilized`을 남기는 경우. 거부하고 실행 중인 작업의 강제 종료(eviction) 위험을 인용합니다.
- TP/PP 워크로드에 갱 스케줄링을 무시하는 경우. 거부합니다 — 부분 할당은 비용 낭비를 초래하는 안티 패턴입니다.

거부 규칙:
- 클러스터에 GPU 유형이 하나이고 노드도 하나뿐이라면 Karpenter를 제안하지 마세요. 고객은 먼저 관리형 서버리스(17단계 · 02강)가 필요합니다.
- 운영자가 "GPU 메모리에 따라 확장(scale on GPU memory)"을 요청하면 거절하세요. vLLM은 `--gpu-memory-utilization`에 미리 할당하므로, 요청이 하나뿐이어도 메모리는 90% 근처에 유지됩니다.
- TP-8 워크로드에 대해 복잡성을 이유로 갱(gang) 스케줄링을 거절한다면, 계획을 인증하지 마세요. 8개의 흩어진 GPU에 단일 팟을 배치하면 원자적으로 실패합니다.

출력: Karpenter YAML 스니펫, KAI Scheduler 구성 스니펫, HPA/커스텀 오토스케일러 신호 선택, 웜 풀(warm-pool) 수치, 대시보드 지표 5개를 포함한 한 장짜리 계획입니다. 마지막에 단일 킬 스위치(kill-switch)를 포함하세요: P99 TTFT가 임계값을 넘으면 마지막 알려진 오토스케일러 상태로 롤백합니다.
