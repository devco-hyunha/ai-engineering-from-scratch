# 오케스트레이션 패턴: 슈퍼바이저, 스웜, 계층적

> 2026년 프레임워크 전반에 걸쳐 네 가지 오케스트레이션 패턴이 반복됩니다: 슈퍼바이저-워커, 스웜 / 피어 투 피어, 계층적, 디베이트. Anthropic의 가이드라인: "필요에 맞는 올바른 시스템을 구축하는 것이 중요합니다." 단순하게 시작하세요. 단일 에이전트와 다섯 가지 워크플로우 패턴이 충분하지 않을 때만 토폴로지를 추가하세요.

**유형:** 학습 + 빌드
**언어:** Python (표준 라이브러리)
**선수 요건:** 14단계 · 12강 (워크플로우 패턴), 14단계 · 25강 (멀티 에이전트 디베이트)
**시간:** 약 60분

## 학습 목표

- 네 가지 반복되는 오케스트레이션 패턴을 나열하고 각각이 적합한 시점을 설명해 보세요.
- 도구 호출 기반 감독 vs 슈퍼바이저 라이브러리라는 2026년 LangChain 권장 사항을 설명해 보세요.
- Anthropic의 "올바른 시스템을 구축하라"는 규칙과 이것이 토폴로지 선택을 어떻게 제한하는지 설명해 보세요.
- 공통 스크립트 LLM을 사용하여 표준 라이브러리에서 네 가지 패턴을 모두 구현해 보세요.

## 문제점

팀들은 필요하기 전에 "멀티 에이전트"를 도입하려는 경향이 있습니다. 네 가지 패턴이 프레임워크 전반에 걸쳐 반복됩니다. 패턴을 식별할 수 있다면 올바른 것을 선택하거나, 토폴로지를 완전히 생략할 수 있습니다.

## 개념

### 슈퍼바이저-워커

- 중앙 라우팅 LLM이 전문가 에이전트로 작업을 분배합니다.
- 결정 사항: 자기 자신에게 루프 백, 전문가에게 핸드오프, 종료.
- 전문가들은 서로 대화하지 않으며, 모든 라우팅은 슈퍼바이저를 통해 이루어집니다.

프레임워크: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026 LangChain 권장 사항:** `create_supervisor`가 아닌 직접 도구 호출을 통해 감독을 수행하세요. 더 세밀한 컨텍스트 엔지니어링 제어 권한을 제공합니다. 각 전문가가 정확히 무엇을 보는지 결정할 수 있습니다.

### 스웜 / 피어 투 피어

- 에이전트들이 공유 도구 표면을 통해 직접 핸드오프합니다.
- 중앙 라우터가 없습니다.
- 슈퍼바이저보다 지연 시간이 낮습니다 (호프가 적음).
- 추론하기가 더 어렵습니다 (단일 제어 지점이 없음).

프레임워크: LangGraph swarm topology, OpenAI Agents SDK handoffs (모든 에이전트가 다른 모든 에이전트로 핸드오프할 수 있는 경우).

### 계층적

- 하위 감독자가 워커를 관리하고, 감독자가 하위 감독자를 관리하는 구조입니다.
- LangGraph에서는 중첩된 서브그래프, CrewAI에서는 중첩된 크루로 구현됩니다.
- 운영 복잡도를 감수하고 대규모 에이전트 인원으로 확장할 수 있습니다.

언제 필요한가요: 단일 감독자의 컨텍스트 예산이 모든 전문가의 설명을 담을 수 없을 때입니다.

### 토론

- 병렬 제안자 + 반복적 상호 비판 (25강).
- 엄밀히 말하면 오케스트레이션이라기보다 검증에 가깝지만, 프레임워크에서는 토폴로지 선택지로 나타납니다.

### 자율 크루 vs 결정적 흐름

CrewAI는 두 가지 배포 모드를 공식화합니다:

- **Flow**는 결정적 이벤트 기반 자동화를 위한 것으로, 프로덕션의 권장 시작점입니다.
- **Crew**는 자율적인 역할 기반 협업을 위한 것입니다.

이 개념은 위의 네 가지 패턴과 직교하지만 토폴로지와 매핑됩니다: Flow는 일반적으로 감독자 또는 계층형이고, Crew는 일반적으로 LLM 라우터를 갖춘 감독자형입니다.

### Anthropic의 가이드라인

"LLM 분야의 성공은 가장 정교한 시스템을 구축하는 것이 아닙니다. 필요에 맞는 시스템을 구축하는 것입니다."

결정 순서:

1. 단일 에이전트 + 워크플로우 패턴 (12강) — 여기서 시작하세요.
2. 감독자-워커 — 전문가가 2~4명일 때.
3. 스웜 — 지연 시간이 추론 명확성보다 중요할 때.
4. 계층형 — 감독자 컨텍스트 예산이 실패할 때만.
5. 토론 — 정확도가 비용보다 중요할 때.

### 이 패턴이 잘못되는 지점

- **토폴로지 우선 사고.** 다중 에이전트가 해결할 문제를 식별하기 전에 "다중 에이전트가 필요하다"고 결정하는 것.
- **스웜에서의 반복적인 핸드오프.** A -> B -> A -> B. 호프 카운터를 사용하세요.
- **가짜 계층.** "엔터프라이즈"라는 이유로 세 계층을 만들지만, 실제로는 두 팀뿐입니다. 축약하세요.

```figure
orchestration-pattern
```

## 구현하기

`code/main.py`는 스크립트된 LLM을 사용하여 표준 라이브러리에서 네 가지 패턴을 모두 구현합니다:

- `Supervisor` — 중앙 라우터.
- `Swarm` — 직접 핸드오프를 사용하는 피어 투 피어.
- `Hierarchical` — 감독자의 감독자.
- `Debate` — 병렬 제안자 + 비평.

각 패턴은 동일한 3가지 의도 작업(환불 / 버그 / 판매)을 처리합니다. 추적(trace) 형태는 다릅니다.

실행해 보세요:

```
python3 code/main.py
```

출력: 패턴별 추적 + 연산 횟수. Supervisor가 가장 깔끔합니다; swarm가 가장 짧습니다; hierarchical가 가장 깊습니다; debate가 가장 비쌉니다.

## 사용하기

- **LangGraph**는 supervisor와 hierarchical(중첩된 subgraph)에 사용합니다.
- **OpenAI Agents SDK**는 handoffs-as-tools(supervisor 형태)에 사용합니다.
- **CrewAI Flow**는 생산 환경의 결정적(deterministic) 작업에 사용합니다.
- **Custom**은 debate나 정확한 제어가 필요할 때 사용합니다.

## 출시하기

`outputs/skill-orchestration-picker.md`가 토폴로지를 선택하고 구현합니다.

## 연습 문제

1. supervisor-worker를 router를 제거하여 swarm으로 변환해 보세요. 무엇이 깨집니까? 무엇이 개선됩니까?
2. swarm에 hop 카운터를 추가해 보세요: 3번의 핸드오프 이후에 거부합니다. A->B->A 왕복을 잡을 수 있습니까?
3. 12개 전문가 도메인을 위한 2단계 hierarchical 시스템을 구축해 보세요. 중첩(nesting)이 없으면 컨텍스트 예산이 어디에서 실패합니까?
4. 생산 환경 형태의 워크로드에서 4가지 패턴을 프로파일링해 보세요. 어떤 패턴이 어떤 지표(레이턴시, 비용, 정확도, 디버깅 용이성)에서 이깁니까?
5. Anthropic의 "Building Effective Agents" 게시물을 읽어 보세요. 생산 환경의 흐름 각각을 4가지 중 하나로 매핑해 보세요. 깔끔하게 매핑되지 않는 것이 있습니까?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | 중앙 LLM이 전문가에게 분배(dispatch)합니다; 전문가끼리는 서로 대화하지 않습니다 |
| Swarm | "Peer-to-peer" | 공유 도구를 통한 직접 핸드오프; 중앙 router 없음 |
| Hierarchical | "Supervisors of supervisors" | 대규모 인원을 위한 중첩된 subgraph |
| Debate | "Proposer + critique" | 병렬 제안자, 상호 비평 (25강) |
| Tool-call-based supervision | "Supervisor without a library" | 컨텍스트 제어를 위해 supervisor를 직접 도구 호출로 구현 |
| Crew | "Autonomous team" | CrewAI의 역할 기반 협업 모드 |
| Flow | "Deterministic workflow" | CrewAI의 이벤트 기반 생산 모드 |

## 추가 읽기

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 5가지 패턴 + 에이전트 대 워크플로
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 슈퍼바이저, 스웜, 계층형
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew 대 Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 토론 패턴
