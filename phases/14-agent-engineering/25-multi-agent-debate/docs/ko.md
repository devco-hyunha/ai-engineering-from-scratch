# 멀티 에이전트 토론 및 협업

> Du et al. (ICML 2024, "Society of Minds")는 N개의 모델 인스턴스를 실행하여 독립적으로 답변을 제안한 후, R라운드에 걸쳐 서로를 반복적으로 비판하여 수렴합니다. 이는 사실성, 규칙 준수, 추론을 개선합니다. 희소(topology) 구조는 토큰 비용 측면에서 완전 메쉬(full mesh)보다 우수합니다.

**유형:** 학습 + 빌드
**언어:** Python (표준 라이브러리)
**선수 요건:** 14단계 · 12강 (워크플로우 패턴), 14단계 · 05강 (Self-Refine 및 CRITIC)
**시간:** 약 60분

## 학습 목표

- 토론 프로토콜을 설명하세요: N명의 제안자, R라운드, 공유된 답변으로 수렴.
- 토론이 사실성, 규칙 준수, 추론을 개선하는 이유를 설명하세요.
- 희소(topology) 구조를 설명하세요: 모든 토론자가 서로를 모두 볼 필요는 없습니다.
- 스크립트된 LLM을 사용하여 완전 메쉬(full-mesh) 및 희소(sparse) 변형으로 표준 라이브러리 기반 토론을 구현하고, 토큰 비용과 정확도를 측정하세요.

## 문제점

Self-Refine (05강)은 하나의 모델이 스스로를 비판하는 방식이며, 이는 집단 사고(groupthink) 위험을 내포합니다. CRITIC (05강)은 비판을 외부 도구에 기반하지만, 항상 이용 가능하지는 않습니다. 토론은 세 번째 방식을 도입합니다: 여러 인스턴스, 상호 비판, 불일치를 통한 수렴.

## 개념

### Society of Minds (Du et al., ICML 2024)

- N개의 모델 인스턴스가 동일한 질문에 대해 독립적으로 답변을 제안합니다.
- R라운드에 걸쳐 각 모델은 다른 모델의 제안을 읽고 비판합니다.
- 모델은 비판에 기반하여 답변을 업데이트합니다.
- R라운드 후, 수렴된 답변을 반환합니다.

원래 실험에서는 비용 때문에 N=3, R=2를 사용했습니다. 어려운 문제(MMLU, GSM8K, Chess Move Validity, 전기 생성)에서는 더 많은 에이전트와 더 많은 라운드를 사용할수록 정확도가 향상됩니다.

모델 간 조합은 단일 모델 토론보다 우수합니다: ChatGPT + Bard가 함께 작동할 때 각각 단독으로 작동할 때보다 더 좋습니다.

### 희소(topology) 구조

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025)는 완전 메쉬(full-mesh) 토론이 항상 최적은 아니라는 것을 보여줍니다. 희소(topology) 구조(스타, 링, 허브-앤-스포크)는 더 낮은 토큰 비용으로 정확도를 일치시킬 수 있습니다. 각 토론자는 동료의 하위 집합만 봅니다.

시사점:

- 풀 메시(full mesh) N=5, R=3 = 5 × 3 = 15개의 제안, 각 제안이 4개의 피어를 읽음 = 60개의 비평 연산.
- 스타(star) N=5, R=3 (하나의 허브 + 4개의 스포크) = 15개의 제안, 스포크는 허브만 읽음 = 12개의 비평 연산.

### 디베이트(debate)가 도움이 되는 경우

- **사실성(Factuality).** N개의 독립적인 제안, 교차 검증이 환각(Hallucination)을 줄임.
- **규칙 준수(Rule-following).** 체스 수의 유효성 — 한 모델이 규칙을 놓치면 다른 모델이 잡아냄.
- **개방형 추론(Open-ended reasoning).** 여러 가지 프레이밍이 올바른 답으로 좁혀감.

### 디베이트(debate)가 해가 되는 경우

- **레이턴시(latency) 민감한 UX.** N × R의 직렬 라운드는 가질 수 없는 레이턴시(latency)임.
- **비용 민감한 규모.** 질문당 N × R 토큰(Token).
- **단순한 사실 조회(Simple factual lookups).** 한 번의 조회가 다섯 번의 디베이트(debate)보다 저렴함.

### 2026년 실용적 구현

- **Anthropic 오케스트레이터-워커(orchestrator-workers)** (12강) — 합성 단계가 있는 디베이트(debate)의 한 변형.
- **LangGraph 슈퍼바이저(supervisor)** (13강) — 중앙 라우터(router) + 전문가 에이전트(Agent)가 디베이트(debate)를 노드(node)로 구현할 수 있음.
- **OpenAI Agents SDK** (16강) — 에이전트(Agent)가 반복적인 비평을 위해 서로 핸드오프(Handoff)함.
- **멀티 에이전트(multi-agent) 평가(Evaluation (Eval))** — 디베이트(debate) + 평가자-최적화자(evaluator-optimizer)를 짝지어 평가 신호를 생성함.

### 이 패턴이 잘못된 경우

- **수렴 붕괴(Convergence collapse).** 모든 에이전트(Agent)가 첫 번째 잘못된 답으로 수렴함. 필수 불일치 라운드로 완화하세요.
- **허브(Hub) 실패.** 스타(star) 토폴로지(topology)에서 나쁜 허브(Hub)가 모두를 오염시킴. 허브(Hub)를 회전시키거나 여러 허브(Hub)를 사용하세요.
- **프롬프트(Prompt) 균일화(Prompt homogenization).** 모든 에이전트(Agent)가 동일한 프롬프트(Prompt)를 사용하므로 동일한 답을 생성함. 다양한 프롬프트(Prompt) 및/또는 모델을 사용하세요.

```figure
debate-converge
```

## 구현하기

`code/main.py`이 스탄다드 라이브러리(stdlib) 디베이트(debate)를 구현함:

- `Debater` 클래스(디베이터(debater)별 의견 드리프트(drift)가 있는 스크립트된 LLM (대규모 언어 모델)).
- `FullMeshDebate` 및 `SparseDebate` 러너(runner).
- 세 가지 질문: 하나는 사실적, 하나는 규칙 기반, 하나는 추론.
- 메트릭: 수렴된 답, 수렴까지의 라운드 수, 총 비평 연산 수.

실행하세요:

```
python3 code/main.py
```

출력: 프로토콜별 정확도와 비용; 희소 매치(sparse matches)가 풀 메시(full mesh)의 2/3 질문에 대해 더 낮은 비용으로 일치함.

## 사용하기

- **Anthropic orchestrator-workers**는 단순한 2-3명 에이전트 토론에 적합합니다.
- **LangGraph**는 체크포인팅을 지원하는 상태 기반 다중 라운드 토론에 적합합니다.
- **Custom**은 연구나 특수한 정확성 보장이 필요한 경우 적합합니다.

## 출시하기

`outputs/skill-debate.md`는 구성 가능한 토폴로지, N, R, 수렴 규칙을 가진 다중 에이전트 토론을 스캐폴딩합니다.

## 연습 문제

1. "강제 불일치" 규칙을 구현해 보세요: 1라운드에서 모든 토론자는 서로 다른 제안을 제시해야 합니다. 수렴 속도에 미치는 영향을 측정해 보세요.
2. 신뢰도 가중 집계 기능을 추가해 보세요: 토론자는 (답변, 신뢰도)를 반환하고, 집계기는 신뢰도에 따라 가중치를 적용합니다. 효과가 있는지 확인해 보세요.
3. "에이전트" 중 하나를 다른 의견을 가진 스크립트된 LLM으로 교체해 보세요. 이질성이 정확도를 개선하는지 확인해 보세요.
4. 3가지 질문에 대해 완전 메시(full mesh)와 희소(sparse) 토폴로지의 토큰 비용을 측정해 보세요. 비용 대비 정확도를 그래프로 그려 보세요.
5. Society of Minds 논문을 읽어 보세요. 토이 프로젝트를 N=5, R=3으로 포팅해 보세요. 무엇이 깨지고, 무엇이 개선되나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Debate | "다중 에이전트 비판" | N명의 제안자, R라운드의 상호 비판, 수렴 |
| Full mesh | "모두가 모두를 읽음" | 각 라운드에서 모든 토론자가 모든 동료의 답변을 읽음 |
| Sparse topology | "제한된 동료 시야" | 토론자가 동료의 일부만 읽음 |
| Hub-and-spoke | "스타 토폴로지" | 중앙 토론자 1명, N-1명의 스포크는 중앙만 읽음 |
| Convergence | "합의" | 토론자가 공유된 답변으로 수렴 |
| Society of Minds | "Du et al.의 토론 논문" | ICML 2024 다중 에이전트 토론 방법 |

## 추가 읽기

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 표준 다중 에이전트 토론
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — 희소 토폴로지 결과
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 토론 변형으로서의 orchestrator-workers
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 단일 모델 자기 비판 대응물
