# LLM 프로덕션을 위한 카오스 엔지니어링

> 2026년 현재, LLM을 위한 카오스 엔지니어링은 그 자체로 하나의 독립적인 분야입니다. 프로덕션 환경에서 실험을 수행하기 전에 충족해야 할 선수 요건은 다음과 같습니다: 정의된 SLI/SLO, 추적(trace)+메트릭+로그 관측 가능성, 자동 롤백, 런북, 온콜(on-call) 체계. 아키텍처는 네 개의 평면(plane)으로 구성됩니다: 제어(control, 실험 스케줄러), 대상(target, 서비스, 인프라, 데이터 스토어), 안전(safety, 가드 + 중단 + 트래픽 필터), 관측 가능성(observability, 메트릭 + 추적 + 로그), 피드백(feedback, SLO 조정으로 연결). 가드레일(guardrails)은 필수입니다: 일일 오류 예산 소모량이 예상치의 2배를 초과하면 버레이트(burn-rate) 경보가 실험을 일시 중단합니다. 억제 윈도우(suppression windows)와 추적 ID(trace-ID) 상관관계를 통해 경보 노이즈를 중복 제거합니다. 주기(cadence): 주간에는 작은 카나리 릴리스(canary release)와 SLO 리뷰, 월간에는 게임 데이(game day)와 사후 분석(postmortem), 분기별에는 팀 간 복원력(resilience) 감사와 의존성 매핑을 수행합니다. LLM 특유의 실험: 메모리 과부하, 네트워크 장애, 제공자(provider) 중단, 잘못된 프롬프트, KV 캐시 eviction storm. 도구: Harness Chaos Engineering (LLM 기반 권장 사항, blast-radius 축소, MCP 도구 통합); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**유형:** Learn
**언어:** Python (stdlib, toy chaos experiment runner)
**선수 요건:** 17단계 · 23강 (AI를 위한 SRE), 17단계 · 13강 (관측 가능성)
**시간:** 약 60분

## 학습 목표

- 카오스 엔지니어링의 다섯 가지 선수 요건(SLI/SLO, 관측 가능성, 롤백, 런북, 온콜)을 나열하고, 이 중 하나라도 생략하면 실무가 왜 무너지는지 설명해 보세요.
- 네 개의 평면(제어, 대상, 안전, 관측 가능성)과 SLO로 연결되는 피드백 루프를 다이어그램으로 그려 보세요.
- LLM 특유의 다섯 가지 실험(메모리 과부하, 네트워크 장애, 제공자 중단, 잘못된 프롬프트, KV eviction storm)을 나열해 보세요.
- 스택에 따라 Harness, LitmusChaos, Chaos Mesh 중 하나의 도구를 선택해 보세요.

## 문제점

전통적인 스택에서의 카오스 테스트는 이미 확립되어 있습니다. LLM 스택은 새로운 장애 모드(failure modes)를 추가합니다. 독성 문자(poison character)가 포함된 4K 토큰 프롬프트는 토크나이저(tokenizer)를 12초 동안 멈추게 합니다. 상류 제공자(upstream provider)가 429 오류를 반환하면, 게이트웨이(gateway)가 재시도(retry)하고, 서비스는 재시도로 인한 동시성 증가로 OOM(Out Of Memory)이 발생합니다. 버스트(burst) 부하 하에서의 KV 캐시 eviction storm은 재프리필(re-prefill) 캐스케이드(cascade)를 유발하여 컴퓨팅 자원을 포화(saturate)시킵니다.

이러한 문제들은 단위 테스트(unit tests)에서는 드러나지 않습니다. 카오스 엔지니어링은 사용자가 발견하기 전에 이러한 문제를 발견하는 방법입니다.

## 개념

### 선수 요건

다음 항목이 없다면 프로덕션에서 카오스 테스트를 실행하지 마세요:

1. **SLI/SLO** — 정의된 서비스 수준 지표 및 목표.
2. **관측 가능성(Observability)** — 트레이스, 메트릭, 로그가 대시보드에 연결되어 있습니다.
3. **자동 롤백(Rollback)** — 17단계 · 20강의 정책 플래그 롤백.
4. **런북(Runbooks)** — 구조화된 문서, 17단계 · 23강.
5. **온콜(On-call)** — 대응할 사람이 있습니다.

이 중 하나라도 빠지면 카오스 테스트가 실제 인시던트로 변합니다.

### 네 개의 평면 + 피드백

**제어 평면(Control plane)** — 실험 스케줄러 (Litmus 워크플로, Chaos Mesh 스케줄, Harness UI).

**타겟 평면(Target plane)** — 서비스, 포드, 노드, 로드 밸런서, 데이터 스토어.

**안전 평면(Safety plane)** — 킬 스위치, 억제 윈도우, 폭발 반경 제한, 오류 예산 게이트.

**관측 평면(Observability plane)** — 정상 메트릭 + 트레이스 ID 상관관계를 통해 카오스 유발 실패와 자연 발생 실패를 구분합니다.

**피드백 루프(Feedback loop)** — 발견된 사항이 SLO 조정, 런북 업데이트, 코드 수정으로 반영됩니다.

### 가드레일(Guardrails)은 필수입니다

- **소진율(Burn-rate) 알림**: 일일 오류 예산 소진량이 예상치의 2배를 초과하면 실험을 일시 중지합니다.
- **억제 윈도우(Suppression windows)**: 실험 중 폭발 반경 내의 비실험 알림을 침묵시킵니다.
- **트레이스 ID 상관관계(Trace-ID correlation)**: 실험으로 유발된 모든 오류에 태그를 부여하여 온콜 담당자가 중복을 제거할 수 있도록 합니다.

### LLM 특화 실험 5가지

1. **메모리 과부하(Memory overload)** — 높은 동시성으로 긴 컨텍스트 요청을 보내 KV 캐시 선점 폭풍을 유발합니다. 관찰: 서비스가 우아한 저하(Graceful Degradation)를 수행하는지, 아니면 크래시하는지 확인하세요.

2. **네트워크 장애(Network failure)** — 추론 게이트웨이와 제공자 간의 연결을 끊습니다. 관찰: SLA 내에서 폴백(Fallback)이 작동하는지 확인하세요. (17단계 · 19강)

3. **제공자 장애 시뮬레이션(Provider outage simulation)** — OpenAI에서 100% 429 응답을 시뮬레이션합니다. 관찰: Anthropic으로 라우팅 페일오버(Failover)가 발생하는지 확인하세요. (17단계 · 16, 19강)

4. **잘못된 프롬프트(Malformed prompt)** — 토크나이저를 멈추게 하는 페이로드 (예: deeply nested unicode, huge UTF-8 codepoint)를 주입합니다. 관찰: 단일 요청이 워커를 잠그는지 확인하세요.

5. **KV eviction storm** — vLLM 블록 예산을 포화(Saturation) 상태로 만들어 강제 eviction을 유발합니다. 관찰: LMCache가 복구하는지, 서비스가 저하되는지 확인하세요.

### 주기

- **주간** — 스테이징 환경에서 작은 카나리 실험, 프로덕션의 경우 최대 5%.
- **월간** — 특정 시나리오에 대한 정기적인 게임 데이(game day); 팀 간 참여; 사후 분석(Postmortem).
- **분기별** — 팀 간 복원력 감사; 의존성 맵 업데이트.

### 도구

- **Harness Chaos Engineering** — 상용 도구; AI 기반 실험 추천; 폭발 반경(blast-radius) 축소; MCP 도구 통합.
- **LitmusChaos** — CNCF 졸업 프로젝트; Kubernetes 워크플로 기반.
- **Chaos Mesh** — CNCF 샌드박스 프로젝트; Kubernetes 네이티브 CRD 스타일.
- **Gremlin** — 상용 도구; 폭넓은 지원.
- **AWS FIS** / **Azure Chaos Studio** — 관리형 클라우드 서비스.

### 작게 시작하기

첫 번째 실험: 안정적인 트래픽 하에서 디코딩(replica) 하나를 pod-kill 합니다. 라우팅 재배치 및 복구를 관찰하세요. 이 실험이 성공적이고 안전해 보이면, 네트워크 카오스로 전환하세요.

첫 번째 LLM 특화 실험: 한 제공자(provider)에 대해 5분간 429 오류를 주입하세요. 폴백(fallback)을 관찰하세요. 대부분의 팀은 폴백이 완전히 테스트되지 않았다는 사실을 발견합니다.

### 기억해야 할 숫자

- 네 가지 평면(plane): 제어(control), 대상(target), 안전(safety), 관측 가능성(observability).
- 소진율(burn-rate) 일시중지: 예상 일일 예산 소진의 2배.
- 주기: 주간 카나리, 월간 게임 데이, 분기별 감사.
- 다섯 가지 LLM 실험: 메모리, 네트워크, 제공자(provider), 잘못된 프롬프트(malformed prompt), KV 폭풍(storm).

```figure
i4-chaos-guard
```

## 사용하기

`code/main.py`는 안전 평면(safety plane) 게이트가 있는 세 가지 카오스 실험을 시뮬레이션합니다. 어떤 실험이 소진율(burn-rate) 중단 조건을 트리거하는지 보고합니다.

## 출시하기

이 강의는 `outputs/skill-chaos-plan.md`를 생성합니다. 스택과 성숙도를 고려하여, 첫 번째 세 가지 실험과 도구를 선택합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 어떤 실험이 소진율(burn-rate) 게이트를 트리거하며, 그 이유는 무엇인가요?
2. vLLM 기반 RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) 서비스용 첫 번째 다섯 가지 카오스 실험을 설계하세요. 성공 기준을 포함하세요.
3. 소진율(burn-rate) 경보가 실험을 일시중지했습니다. 카오스 실험의 원인인지, 자연적인 원인인지 어떻게 결정하나요?
4. 카오스 실험을 프로덕션 환경에서 실행해야 하는지, 스테이징 환경에서만 실행해야 하는지 논증하세요. 프로덕션이 정답인 경우는 언제인가요?
5. 일반적인 네트워크 카오스로는 재현할 수 없는 LLM 특유의 실패 모드 세 가지를 나열해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| SLI / SLO | "서비스 목표" | 지표 + 목표; 필수 선수 요건 |
| Blast radius | "범위" | 실험의 영향을 받는 서비스/사용자 집합 |
| Burn-rate alert | "예산 게이트" | 오류 예산 소진율이 예상의 2배를 초과할 때 발생 |
| Game day | "월간 훈련" | 정기적으로 진행되는 팀 간 카오스 연습 |
| LitmusChaos | "CNCF 워크플로" | CNCF 졸업(Kubernetes) 카오스 도구 |
| Chaos Mesh | "CNCF CRD" | CNCF 샌드박스 Kubernetes 네이티브 카오스 |
| Harness CE | "상업용 AI 지원" | AI 추천이 포함된 Harness 카오스 |
| Malformed prompt | "토크나이저 폭탄" | 토큰화를 지연시키는 입력 |
| KV eviction storm | "선점 캐스케이드" | 대량 퇴거로 재프리필이 트리거되는 현상 |

## 추가 읽기

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
