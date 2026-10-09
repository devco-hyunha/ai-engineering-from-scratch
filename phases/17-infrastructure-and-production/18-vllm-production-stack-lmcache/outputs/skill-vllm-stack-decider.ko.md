---
name: vllm-stack-decider
description: vLLM 배포 레이아웃 결정 — production-stack Helm 차트, KV 오프로드(네이티브 CPU 또는 LMCache), 라우터/관측성 통합 — 워크로드 및 플릿 규모에 따라.
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, production-stack, lmcache, kv-offload, connector-api]
---

워크로드(프롬프트 형태, 동시성, 접두어 재사용 패턴), 플릿(엔진, GPU 유형), 운영 컨텍스트(Kubernetes 네이티브, 멀티 테넌트, 예산)를 고려하여 vLLM 스택 계획을 작성하세요.

다음 내용을 작성하세요:

1. 스택. vLLM production-stack Helm 차트(새 배포에 권장)를 사용하거나 직접 구축하세요. 적용되는 연산자(operator)/CRD를 명시하세요.
2. KV 오프로드. 선택하세요:
   - 없음(짧은 프롬프트, 낮은 동시성 — 오버헤드가 이점을 초과).
   - 네이티브 vLLM CPU 오프로드(단일 엔진 HBM 압력, 단순).
   - LMCache 커넥터(다중 엔진 접두어 재사용, 선점(preemption)이 많거나, 멀티 테넌트 공유 프롬프트).
3. HBM 사용량 모니터링. `--gpu-memory-utilization`를 여유(headroom)를 두고 설정하세요. 92%+ 지속 시 선점(pre-preemption) 신호로 알람을 설정하세요.
4. 라우터 통합. 캐시 인식 라우터(17단계 · 11강). KV 이벤트 채널이 구성되었는지 확인하세요.
5. 관측성. 엔진별 Prometheus 스크레이프, OTel GenAI 속성(17단계 · 13강), production-stack의 Grafana 대시보드 템플릿.
6. 예상 영향. 현재 대비 예상 처리량 증가분을 정량화하세요 — 16x H100 벤치마크 형태를 참조하세요(LMCache는 KV 발자국이 HBM을 초과할 때 도움이 됩니다).

하드 거부(Hard rejects):
- 공유 접두어 또는 선점(preemption) 없이 LMCache 배포. 거부하세요 — 오버헤드, 이점 없음.
- HBM 압력 모니터링 없이 vLLM 실행. 거부하세요 — 첫 선점(preemption)은 예상치 못한 일이 될 것입니다.
- Helm 차트가 사용 사례를 커버할 때 production-stack을 직접 구축. 거부하세요 — 재발명 비용.

거부 규칙:
- 플릿에 엔진이 2개 미만이면 LMCache를 거부하세요 — 엔진 간 재사용이 핵심입니다. 단일 엔진은 네이티브를 사용하세요.
- 워크로드의 프롬프트가 1K 토큰 미만이고 동시성이 100 미만이면 모든 종류의 오프로드를 거부하세요 — HBM 여유가 충분합니다.
- 팀에 K8s 역량이 없으면 production-stack을 거부하세요 — 단일 엔진 vLLM + 단순 프록시로 시작하세요.

출력: 스택, KV 오프로드 선택, HBM 모니터링, 라우터 통합, 관측 가능성, 예상 영향을 명시한 한 페이지 계획입니다. 마지막에는 단일 게이트로 최근 24시간의 HBM 사용량 P99를 포함하세요.
