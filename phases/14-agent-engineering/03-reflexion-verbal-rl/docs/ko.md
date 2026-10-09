# Reflexion: 언어적 강화 학습

> 기반 강화 학습은 실패 모드를 수정하기 위해 수천 번의 시도와 GPU 클러스터가 필요합니다. Reflexion (Shinn et al., NeurIPS 2023)은 자연어로 이를 수행합니다: 각 실패한 시도 후, 에이전트(Agent)는 반성을 작성하고 이를 일화 기억(episodic memory)에 저장하며, 다음 시도를 그 기억에 조건화합니다. 이는 Letta의 수면 시간 계산(sleep-time compute), Claude Code의 CLAUDE.md 학습 내용, pro-workflow의 learn-rule 뒤에 있는 패턴입니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 02강 (ReWOO)
**시간:** 약 60분

## 학습 목표

- Reflexion의 세 구성 요소(Actor, Evaluator, Self-Reflector)와 일화 기억(episodic memory)의 역할을 나열해 보세요.
- 이진 평가기, 반성 버퍼, 새로운 재시도를 포함하는 stdlib Reflexion 루프를 구현해 보세요.
- 주어진 작업에 대해 스칼라, 휴리스틱, 자기 평가 피드백 소스 중 하나를 선택해 보세요.
- 언어적 강화 학습이 왜 기반 강화 학습이 수천 번의 시도를 필요로 하는 오류를 포착하는지 설명해 보세요.

## 문제점

에이전트(Agent)가 작업을 실패합니다. 표준 강화 학습에서는 수천 번의 시도를 더 실행하고, 기울기를 계산하고, 가중치를 업데이트합니다. 이는 비용이 많이 들고 느리며, 대부분의 프로덕션 에이전트(Agent)는 모든 실패에 대한 훈련 예산이 없습니다.

Reflexion (Shinn et al., arXiv:2303.11366)은 다른 질문을 던집니다: 에이전트(Agent)가 왜 실패했는지 생각하고, 그 생각을 프롬프트에 포함하여 다시 시도한다면 어떨까요? 가중치 업데이트는 없습니다. 기울기도 없습니다. 시도 사이에 저장된 자연어만 있습니다.

결과: ALFWorld에서는 ReAct 및 기타 비미세 조정(non-fine-tuned) 기준선을 능가합니다. HotpotQA에서는 ReAct보다 개선됩니다. 코드 생성(HumanEval/MBPP)에서는 당시 최신 기술(state of the art)을 설정합니다. 단 한 번의 기울기 단계도 없이 모든 것을 달성합니다.

## 개념

### 세 구성 요소

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

거기에 하나의 데이터 구조가 추가됩니다:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

한 시도는 Actor를 실행합니다. Evaluator가 점수를 매깁니다. 점수가 낮으면 Self-Reflector가 반성("X에 대해 묻는 것으로 질문을 잘못 읽어서 Y에 대해 묻는 것이 아니라 X에 대해 묻는 것으로 오해하여 잘못된 도구를 선택했다")을 생성합니다. 반성은 일화 기억(episodic memory)에 들어갑니다. 다음 시도는 새로 시작하지만 반성을 봅니다.

### 세 가지 평가자 유형

1. **스칼라** — 외부의 이진 신호입니다. ALFWorld는 성공하거나 실패합니다. HumanEval 테스트는 통과하거나 실패합니다. 가장 단순하며 신호가 가장 강합니다.
2. **휴리스틱** — 사전 정의된 실패 시그니처입니다. "에이전트가 같은 행동을 연속으로 두 번 수행하면 막힘(stuck)으로 표시합니다." "트래젝토리가 50 단계를 초과하면 비효율적으로 표시합니다."
3. **자기 평가** — LLM이 자신의 트래젝토리를 채점합니다. 정답(ground truth)이 없을 때 필요합니다. 신호가 더 약합니다. 도구 기반 검증(05강 — CRITIC)과 잘 어울립니다.

2026년 기본 설정은 혼합입니다: 스칼라가 있으면 스칼라를, 없으면 자기 평가를, 휴리스틱은 안전 레일(safety rails)로 사용합니다.

### 이것이 일반화되는 이유

Reflexion은 새로운 알고리즘이라기보다 이름 붙여진 패턴입니다. 거의 모든 프로덕션 "자가 치유(self-healing)" 에이전트는 몇 가지 변형을 실행합니다:

- Letta의 수면 시간 계산(08강): 별도의 에이전트가 과거 대화를 반성하고 메모리 블록에 기록합니다.
- Claude Code의 `CLAUDE.md` / "메모리 저장" 패턴: 반성이 학습 내용으로 캡처되어 향후 세션에 선행됩니다.
- pro-workflow의 `/learn-rule` 명령: 수정 사항이 명시적인 규칙으로 캡처됩니다.
- LangGraph의 반성 노드: 출력을 채점하고 필요하면 refine으로 라우팅하는 노드입니다.

모두 동일한 통찰에서 유래합니다: 자연어는 실행 간 "실패에서 무엇을 배웠는지"를 전달하기에 충분히 풍부한 매체입니다.

### 작동할 때와 작동하지 않을 때

Reflexion이 작동하는 경우:

- 명확한 실패 신호가 있습니다(테스트 실패, 도구 오류, 잘못된 답변).
- 작업 클래스가 재현 가능합니다(유형의 질문을 다시 요청할 수 있습니다).
- 반성이 트래젝토리를 개선할 여지가 있습니다(충분한 행동 예산).

Reflexion이 도움이 되지 않는 경우:

- 에이전트가 이미 첫 시도에서 성공합니다.
- 실패가 외부적입니다(네트워크 다운, 도구 고장) — "네트워크가 다운되었다"에 대한 반성은 향후 실행에 도움이 되지 않습니다.
- 반성이 미신으로 변합니다 — 일회성 불안정한 실행에 대한 서사를 저장합니다.

2026년 함정: 메모리 부패. 반성이 누적되면 일부는 낡거나 부정확해지고, 일화 버퍼가 커질수록 재실행이 느려집니다. 완화책: 주기적 압축(06강), 반성에 TTL 적용, 또는 별도의 수면 시간 정리 에이전트(Letta) 사용.

```figure
react-trace
```

## 구현하기

`code/main.py`는 장난감 퍼즐에 Reflexion을 구현합니다: 합이 목표값이 되는 3개 요소의 리스트를 생성합니다. Actor가 후보 리스트를 방출하고, Evaluator가 합을 확인하며, Self-Reflector가 무엇이 잘못되었는지에 대한 한 줄을 작성합니다. 반성은 다음 시도를 위해 일화 메모리에 저장됩니다.

구성 요소:

- `Actor` — 반성을 볼 때 개선되는 스크립트 정책.
- `Evaluator.binary()` — 목표 합에 대한 통과/실패 판정.
- `SelfReflector` — 실패에 대한 한 줄 진단을 생성합니다.
- `EpisodicMemory` — TTL semantics를 가진 제한된 리스트.

실행해 보세요:

```
python3 code/main.py
```

추적은 세 번의 시도를 보여줍니다. 시도 1은 실패하고, 반성이 저장되며, 시도 2는 반성을 보고 개선되지만 여전히 실패하고, 시도 3은 성공합니다. 반성이 없는 기본 실행과 비교해 보세요 — 시도 1의 답에 계속 갇혀 있습니다.

## 사용하기

LangGraph는 반성을 노드 패턴으로 제공합니다. Claude Code의 `/memory` 명령과 pro-workflow의 `/learn-rule`은 일화 버퍼를 markdown 파일로 외부화합니다. Letta의 수면 시간 계산은 downtime 동안 Self-Reflector를 실행하여 주 에이전트가 지연에 묶인 상태로 유지되도록 합니다. OpenAI Agents SDK는 Reflexion을 직접 제공하지 않습니다; 점수에 따라 궤적을 거부하는 커스텀 Guardrail과 실행 간에 유지되는 메모리 `Session`를 사용하여 직접 구축해야 합니다.

## 출시하기

`outputs/skill-reflexion-buffer.md`는 반성 캡처, TTL 및 중복 제거를 포함하는 일화 버퍼를 생성하고 유지합니다. 작업 클래스와 실패가 주어지면, 다음 시도에 실제로 도움이 되는 반성(일반적인 "더 조심해"가 아닌)을 방출합니다.

## 연습 문제

1. 이진 평가기를 목표값에서 얼마나 멀리 떨어져 있는지를 반환하는 스칼라 평가기(거리 메트릭)로 전환해 보세요. 더 빠르게 수렴하나요?
2. 반성에 10번 시도의 TTL을 추가해 보세요. 그 시점 이후에 오래된 반성이 해를 끼치나요, 아니면 도움이 되나요?
3. 동일한 행동이 반복되면 시도를 '고착(stuck)' 상태로 표시하는 휴리스틱 평가기를 구현해 보세요. 이것이 Self-Reflector와 어떻게 상호작용하나요?
4. 반성(reflection)을 무시하는 적대적 Actor로 Reflexion을 실행해 보세요. Actor가 반성을 인식하도록 강제하는 최소한의 반성 프롬프트 엔지니어링은 무엇인가요?
5. Reflexion 논문에서 AlfWorld에 대한 4절을 읽어 보세요. 130%의 성공률 향상 개념을 재현해 보세요: vanilla ReAct 대비 핵심적인 차이(delta)는 무엇인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Reflexion | "자기 교정" | Shinn et al. 2023 — Actor, Evaluator, Self-Reflector 및 에피소드 메모리 |
| 언어적 강화 | "기울기 없는 학습" | 자연어 반성을 다음 시도의 프롬프트 앞에 붙이는 것 |
| 에피소드 메모리 | "작업별 반성" | 하나의 작업 클래스에 대한 이전 반성들의 유한 버퍼 |
| 스칼라 평가자 | "이진 성공 신호" | 정답(ground truth)에서 얻은 통과/실패 또는 수치 점수 |
| 휴리스틱 평가자 | "패턴 기반 탐지자" | 사전 정의된 실패 시그니처 (예: 루프에 갇힘, 너무 많은 단계) |
| 자기 평가자 | "자신의 추적에 대한 LLM-as-judge" | 정답이 없을 때의 낮은 신호 폴백 — 도구 기반 검증과 함께 사용 |
| 메모리 부패 | "낡은 반성" | 에피소드 버퍼가 오래된 항목으로 채워짐; 압축/TTL로 수정 |
| 수면 시간 반성 | "비동기 자기 반성" | Self-Reflector를 핫 경로(hot path) 밖에서 실행하여 주 에이전트가 빠르게 유지되도록 함 |

## 추가 읽기

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 표준 논문
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 프로덕션에서의 비동기 반성
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 컨텍스트의 일부로서 에피소드 버퍼 관리
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 반성 노드 패턴
