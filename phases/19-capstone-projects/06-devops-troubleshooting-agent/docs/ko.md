# 캡스톤 06 — Kubernetes용 DevOps 문제 해결 에이전트

> AWS의 DevOps 에이전트가 GA가 되었고, Resolve AI는 K8s 플레이북을 공개했으며, NeuBird는 시맨틱 모니터링을 시연했고, Metoro는 AI SRE를 서비스별 SLO에 연결했습니다. 프로덕션 형태는 확정되었습니다: 알람 웹훅이 발동되면, 에이전트가 텔레메트리를 읽고, K8s 객체 그래프를 탐색하며, 근본 원인 가설을 순위를 매겨, 승인 버튼이 포함된 Slack 브리핑을 게시합니다. 기본적으로 읽기 전용입니다. 모든 복구 작업은 인간이 게이트합니다. 이 캡스톤은 20개의 합성된 인시던트로 평가되고, AWS의 에이전트와 세 개의 공유된 사례로 비교되는 그 에이전트입니다.

**유형:** Capstone
**언어:** Python (에이전트), TypeScript (Slack 통합)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구 및 MCP), 14단계 (에이전트), 15단계 (자율), 17단계 (인프라), 18단계 (안전)

**활용 단계:** P11 · P13 · P14 · P15 · P17 · P18
**시간:** 30시간

## 문제점

2025-2026 SRE 서사는 "AI 에이전트가 인시던트를 분류(triage)하고, 인간이 복구를 승인한다"가 되었습니다. AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps는 모두 이 형태를 프로덕션에 출시합니다. 에이전트는 Prometheus 지표, Loki 로그, Tempo 추적, kube-state-metrics, 그리고 K8s 객체의 지식 그래프를 읽습니다. 5분 이내에 텔레메트리 인용이 포함된 순위가 매겨진 근본 원인 가설을 생성합니다. Slack를 통한 명시적인 인간 승인 없이는 파괴적인 명령을 절대 실행하지 않습니다.

가장 어려운 작업은 추론이 아니라 범위 설정과 안전입니다. 에이전트는 기본적으로 읽기 전용인 RBAC 표면, 강화된 MCP 도구 서버, 그리고 고려된 모든 명령과 실행된 모든 명령의 감사 로그가 필요합니다. 깊이를 벗어났을 때를 알고 에스컬레이션해야 합니다. 그리고 OOM-kill 연쇄가 $5k 에이전트 청구서를 생성하지 않을 만큼 충분히 저렴하게 실행되어야 합니다.

## 개념

에이전트는 지식 그래프에서 작동합니다. 노드는 K8s 객체(Pods, Deployments, Services, Nodes, HPAs, PVCs)와 텔레메트리 소스(Prometheus 시리즈, Loki 스트림, Tempo 추적)입니다. 엣지는 소유권(Pod -> ReplicaSet -> Deployment), 스케줄링(Pod -> Node), 그리고 관찰(Pod -> Prometheus 시리즈)을 인코딩합니다. 그래프는 kube-state-metrics 동기화로 최신 상태를 유지하며, 모든 알람 시에 재샘플링됩니다.

알림이 발생하면 에이전트(Agent)는 영향을 받은 객체에서 근본 원인을 파악합니다. 에이전트(Agent)는 엣지를 따라 이동하며 관련 텔레메트리 슬라이스(최근 15분)를 가져오고 가설을 작성합니다. 가설은 증거에 따라 순위가 매겨집니다: 몇 개의 텔레메트리 인용이 이를 지원하는지, 얼마나 최신인지, 얼마나 구체적인지. 상위 3개 가설은 그래프 경로 시각화와 함께 Slack으로 전송되며, 복구 조치에 대한 승인 버튼이 포함됩니다.

복구 조치는 게이트로 통제됩니다. 허용된 기본 조치는 읽기 전용입니다. 파괴적인 조치(축소, 롤백(Rollback), Pod 삭제)는 Slack 승인이 필요하며, ArgoCD 롤백(Rollback) 훅은 에이전트(Agent)가 절대 보유하지 않는 인증 토큰을 요구합니다. 감사 로그(Audit Log)는 에이전트(Agent)가 *고려한* 모든 명령을 기록합니다. 실행된 것뿐만 아니라, 리뷰 프로세스가 아슬아슬한 실수를 포착할 수 있도록 합니다.

## 아키텍처

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 스택

- 관측 가능성(Observability) 소스: Prometheus, Loki, Tempo, kube-state-metrics
- 지식 그래프: K8s 객체 + 텔레메트리 엣지의 Neo4j (관리형) 또는 kuzu (임베디드)
- 에이전트(Agent): 도구별 허용 목록이 있는 LangGraph, 기본값은 읽기 전용
- 도구 전송: StreamableHTTP를 통한 FastMCP; 승인 게이트(Approval Gate) 뒤에 파괴적 도구를 위한 별도 서버
- 모델: 근본 원인 추론용 Claude Sonnet 4.7, 로그 요약용 Gemini 2.5 Flash
- 복구: ArgoCD 롤백(Rollback) 웹훅, PagerDuty 에스컬레이션, Slack 승인 카드
- 감사: 추가 전용 구조화 로그 (고려, 실행, 승인, 결과)
- 배포: 자체 좁은 RBAC 역할이 있는 K8s 배포; 별도 네임스페이스

```figure
ce-rootcause-walk
```

## 구현하기

1. **그래프 인제스트.** kube-state-metrics를 Neo4j/kuzu에 30초마다 동기화합니다. 노드: Pod, Deployment, Node, Service, PVC, HPA. 엣지: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. 텔레메트리 오버레이 엣지: OBSERVED_BY (Pod는 Prometheus 시계열에 의해 관측됨).

2. **알림 수신기.** PagerDuty 또는 Alertmanager 웹훅을 허용하는 FastAPI 엔드포인트. 영향을 받은 객체와 SLO 위반을 추출합니다.

3. **읽기 전용 도구 표면.** FastMCP를 통해 kubectl, Prometheus 쿼리, Loki logql, Tempo traceql을 래핑합니다. 모든 도구는 좁은 RBAC 동사("get", "list", "describe")를 가집니다. 기본 서버에는 "delete", "exec", "scale"이 없습니다.

4. **원인 분석 에이전트.** LangGraph의 세 노드: `sample`은 최근 15분간의 텔레메트리 슬라이스를 가져오며, `walk`은 그래프에서 인접한 객체를 조회하고, `hypothesize`는 텔레메트리 인용이 포함된 순위가 매겨진 원인 후보를 작성합니다.

5. **증거 점수화.** 각 가설의 점수는 최근성 * 구체성 * 그래프 경로 길이 역수 * 인용 횟수로 계산됩니다. 상위 3개를 반환합니다.

6. **Slack 브리핑.** 가설, 그래프 경로 시각화(서버 측에서 렌더링된 하위 그래프 이미지), 그리고 최대 하나의 복구 조치에 대한 승인 버튼을 포함한 첨부 파일을 게시합니다.

7. **복구 게이트.** 파괴적인 도구(축소, 롤백, 삭제)는 승인 토큰 뒤에 있는 두 번째 MCP 서버에 위치합니다. 에이전트는 Slack 카드가 인간에 의해 승인된 후에만 이 도구들을 호출할 수 있습니다.

8. **감사 로그.** 추가 전용 JSONL: 모든 후보 명령에 대해 고려 여부, 실행 여부, 승인자를 기록합니다. 매일 S3로 전송합니다.

9. **합성 인시던트 스위트.** 20개 시나리오를 구축합니다: OOMKill 연쇄, DNS 플랩, HPA 스래시, PVC 채움, 잡음 많은 이웃, 결함 있는 사이드카, 잘못된 ConfigMap 롤아웃, 인증서 회전, 이미지 풀 백오프 등. 에이전트를 원인 분석 정확도와 가설 도달 시간으로 평가합니다.

## 사용하기

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 출시하기

`outputs/skill-devops-agent.md`는 산출물입니다. K8s 클러스터와 알림 소스가 주어지면, 에이전트는 순위가 매겨진 원인 가설과 Slack 게이트가 적용된 복구 흐름을 생성합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 시나리오 스위트에서의 RCA 정확도 | 20개 합성 인시던트 전반에 걸쳐 ≥80%의 정확한 원인 |
| 20 | 안전성 | 감사 로그에서 Slack 승인 없이 파괴적 조치 가드가 발동되지 않음 |
| 20 | 가설 도달 시간 | 알림에서 Slack 브리핑까지 p50가 5분 미만 |
| 20 | 설명 가능성 | 모든 가설이 그래프 경로와 텔레메트리 인용을 포함 |
| 15 | 통합 완성도 | PagerDuty, Slack, ArgoCD, Prometheus가 엔드투엔드로 작동 |
| **100** | | |

## 연습 문제

1. AWS의 DevOps Agent가 시연되는 동일한 세 인시던트에서 에이전트를 실행하세요. 나란히 비교한 결과를 게시하세요. 에이전트가 갈리는 부분을 보고하세요.

2. 에이전트가 *고려한* 승인 없는 파괴적 명령을 플래그하는 "아슬아슬한 실수(near-miss)" 감사 기능을 추가하세요. 1주 동안 아슬아슬한 실수 발생률을 측정하세요.

3. 가설 모델을 Claude Sonnet 4.7에서 자체 호스팅된 Llama 3.3 70B로 교체하세요. RCA 정확도 차이와 사고당 비용을 측정하세요.

4. 인과 필터를 구축하세요: 상관된 텔레메트리 급증과 진정한 근본 원인을 구분하세요. 20개 시나리오 레이블로 작은 분류기를 학습하세요.

5. 롤백 드라이 런을 추가하세요: 동일한 매니페스트로 스테이징 클러스터에 ArgoCD 롤백을 실행하세요. Slack 승인 버튼 전에 라이브 클러스터에서 롤백 계획을 검증하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| K8s 지식 그래프 | "클러스터 그래프" | 노드 = K8s 객체 + 텔레메트리 시계열; 엣지 = 소유권, 스케줄링, 관측 |
| 기본 읽기 전용 | "범위 지정 RBAC" | 에이전트의 서비스 계정은 get/list/describe 동사만 가짐; 파괴적 동사는 승인 뒤에 있는 별도 서버에 위치 |
| 감사 로그 | "고려 vs 실행" | 모든 후보 명령, 실행 여부, 승인자를 기록하는 추가 전용 레코드 |
| 가설 순위 매기기 | "증거 점수" | 최신성 × 특이성 × 그래프 경로 길이 역수 × 인용 횟수 |
| Slack 승인 카드 | "HITL 게이트" | 복구 버튼이 있는 인터랙티브 Slack 메시지; 사람이 클릭할 때까지 에이전트가 진행할 수 없음 |
| 텔레메트리 인용 | "증거 포인터" | 주장을 뒷받침하는 Prometheus 쿼리, Loki 셀렉터, 또는 Tempo 추적 URL |
| MTTR | "해결까지의 시간" | 알림 발생부터 SLO 복구까지의 실시간 |

## 추가 읽기

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026년 표준 참조
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 경쟁사 참조
- [NeuBird semantic monitoring](https://www.neubird.ai) — 시맨틱 그래프 접근법
- [Metoro AI SRE](https://metoro.io) — SLO 우선 프로덕션 프레임
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 클러스터 상태 소스
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 참조 에이전트 오케스트레이터
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 서버 프레임워크
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 게이트된 복구 대상
