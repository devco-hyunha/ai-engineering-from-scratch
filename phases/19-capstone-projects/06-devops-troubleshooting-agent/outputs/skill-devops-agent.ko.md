---
name: devops-agent
description: 클러스터 지식 그래프를 탐색하고, 근본 원인을 순위를 매기며, 모든 복구 조치를 Slack을 통해 게이트하는 Kubernetes 문제 해결 에이전트를 구축하세요.
version: 1.0.0
phase: 19단계
lesson: 06강
tags: [capstone, devops, sre, kubernetes, langgraph, fastmcp, aiops]
---

K8s 클러스터와 알림 소스(PagerDuty 또는 Alertmanager)가 주어졌을 때, 5분 이내에 순위가 매겨진 근본 원인 가설을 생성하고 모든 복구 조치를 Slack 승인 카드를 통해 게이트하는 에이전트를 구축하세요.

구축 계획:

1. 30초마다 kube-state-metrics를 Neo4j 또는 kuzu에 수집하세요. Pods, Deployments, Services, Nodes, PVCs, HPAs의 그래프를 구축하고 Prometheus, Loki, Tempo 소스에 대한 텔레메트리 오버레이 엣지를 추가하세요.
2. PagerDuty와 Alertmanager용 FastAPI 웹훅 수신기를 구축하세요.
3. StreamableHTTP 전송을 통해 FastMCP로 읽기 전용 도구(kubectl get/describe, promql, logql, traceql)를 노출하세요.
4. 세 개의 노드를 가진 LangGraph 근본 원인 에이전트를 구축하세요: `sample` (15분 텔레메트리 가져오기), `walk` (그래프 이웃 탐색), `hypothesize` (신뢰도 × 특이성 × 인용 횟수로 후보 순위 매기기).
5. 상위 3개 순위 가설을 그래프 경로 시각화와 함께 승인 버튼이 포함된 Slack 메시지로 게시하세요.
6. 파괴적 도구(scale, rollback, delete)는 Slack 승인 후 에이전트가 획득한 승인 토큰 뒤에 있는 별도의 FastMCP 서버에 배치하세요.
7. 추적 전용 감사 로그를 유지하세요: 모든 *고려된* 명령, 승인 여부, 실행 여부, 승인자를 기록하세요.
8. 20개의 합성 사고 시나리오(OOMKill, DNS flap, HPA thrash, PVC fill, noisy neighbor, faulty sidecar, ConfigMap bad rollout, cert rotation, image-pull backoff, probe failure 및 10개 추가)를 구축하세요. RCA 정확도와 가설 도달 시간으로 에이전트를 평가하세요.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 시나리오 스위트에 대한 RCA 정확도 | 20개 합성 사고 중 80% 이상에서 올바른 근본 원인 |
| 20 | 안전성 | 감사 로그에서 Slack 승인 없이 파괴적 작업 가드가 발동되지 않음 |
| 20 | 가설 도달 시간 | 알림에서 Slack 브리프까지 p50이 5분 미만 |
| 20 | 설명 가능성 | 모든 가설이 그래프 경로와 텔레메트리 인용을 포함합니다 |
| 15 | 통합 완성도 | PagerDuty, Slack, ArgoCD, Prometheus가 엔드투엔드로 작동합니다 |

하드 리젝트:

- 읽기 전용 도구와 파괴적 도구를 혼합하는 단일 MCP 서버를 사용하는 에이전트.
- 텔레메트리 인용 없이 생성된 모든 RCA. 인용되지 않은 가설은 반드시 거부해야 합니다.
- 실행만 기록하는 감사 로그. 모든 고려된 명령을 기록해야 합니다.
- 시드를 포함하는 20개 시나리오 스위트에 대해 에이전트를 실행하지 않은 채 정확성을 주장하는 경우.

거부 규칙:

- 인간 온콜러의 Slack 승인 없이 복구 작업을 거부합니다. 가설이 명백하더라도 거부해야 합니다.
- 읽기 전용 MCP를 통해 `kubectl exec`, `kubectl port-forward` 또는 모든 인터랙티브 도구를 노출하는 것을 거부합니다. 이들은 실질적으로 파괴적입니다.
- 배포별 승인 카드 없이 여러 배포에 걸쳐 복구 작업을 일괄 적용하는 것을 거부합니다.

출력: FastAPI 수신기, LangGraph 에이전트, 읽기 전용 및 파괴적 MCP 서버, Slack 통합, 20개 시나리오 테스트 스위트, 세 개의 공유된 인시던트에 대해 AWS DevOps Agent와 나란히 비교한 결과, 그리고 일주일 관측 기간 동안 에이전트가 *고려*했으나 실행하지 않은 아슬아슬한 명령에 대한 보고서가 포함된 저장소입니다.
