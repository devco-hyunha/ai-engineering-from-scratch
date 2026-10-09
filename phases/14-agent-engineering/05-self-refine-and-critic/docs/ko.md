# Self-Refine 및 CRITIC: 반복적 출력 개선

> Self-Refine (Madaan et al., 2023)은 하나의 LLM을 생성, 피드백, 개선이라는 세 가지 역할로 반복 루프에서 사용합니다. 평균 성능 향상: 7개 작업에서 절대값 +20. CRITIC (Gou et al., 2023)은 검증 단계를 외부 도구로 라우팅하여 피드백 단계를 강화합니다. 2026년에는 이 패턴이 모든 프레임워크에서 "evaluator-optimizer" (Anthropic) 또는 가드레일 루프 (OpenAI Agents SDK) 형태로 출시됩니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 03강 (Reflexion)
**시간:** 약 60분

## 학습 목표

- Self-Refine의 세 가지 프롬프트 (생성, 피드백, 개선)를 나열하고, 개선 프롬프트에서 히스토리가 중요한 이유를 설명해 보세요.
- CRITIC의 핵심 통찰을 설명해 보세요: LLM은 외부 근거(grounding) 없이 자체 검증에 신뢰성이 낮습니다.
- 히스토리를 포함하고 선택적으로 외부 검증기를 사용하는 stdlib Self-Refine 루프를 구현해 보세요.
- 이 패턴을 Anthropic의 "evaluator-optimizer" 워크플로우와 OpenAI Agents SDK의 출력 가드레일에 매핑해 보세요.

## 문제점

에이전트가 거의 정확한 답변을 생성합니다. 코드 한 줄에 구문 오류가 있거나, 요약이 너무 길거나, 계획이 엣지 케이스를 놓칠 수 있습니다. 원하는 것은 에이전트가 자신의 출력을 비판하고, 이를 수정하는 것입니다.

Self-Refine은 단일 모델로, 학습 데이터 없이, RL 없이 이 방법이 작동함을 보여줍니다. 하지만 함정이 있습니다: LLM은 어려운 사실에 대한 자체 검증에 능숙하지 못합니다. CRITIC은 해결책을 제시합니다 — 검증 단계를 외부 도구 (검색, 코드 인터프리터, 계산기, 테스트 러너)를 통해 라우팅하는 것입니다.

이 두 논문은 반복적 개선의 2026년 기본 패턴을 정의합니다: 생성, (가능하면 외부적으로) 검증, 개선, 검증기가 통과하면 중지.

## 개념

### Self-Refine (Madaan et al., NeurIPS 2023)

하나의 LLM, 세 가지 역할:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

핵심 세부 사항: `refine`는 전체 히스토리 — 모든 이전 출력과 비판 —를 보므로 실수를 반복하지 않습니다. 논문은 이를 제거하는 실험(ablation)을 수행했습니다: 히스토리를 제거하면 품질이 급격히 떨어집니다.

헤드라인: 7개 작업(수학, 코드, 약어, 대화)에 걸쳐 GPT-4를 포함하여 평균 +20의 절대적 개선. 학습 없음, 외부 도구 없음, 단일 모델.

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Self-Refine의 약점: 피드백 단계는 LLM이 스스로 점수를 매기는 것입니다. 사실적 주장에 대해서는 신뢰할 수 없습니다(환각은 생성한 모델에게 종종 설득력 있어 보입니다). CRITIC은 `feedback(task, output)`을 `verify(task, output, tools)`로 대체하며, `tools`에는 다음이 포함됩니다:

- 사실적 주장을 위한 검색 엔진.
- 코드 정확성을 위한 코드 인터프리터.
- 산술 연산을 위한 계산기.
- 도메인별 검증기(단위 테스트, 타입 체커, 린터).

검증기는 도구 결과에 기반한 구조화된 비평을 생성합니다. 리파이너는 이 비평을 조건으로 삼습니다.

헤드라인: CRITIC은 비평이 근거에 기반하기 때문에 사실적 작업에서 Self-Refine보다 성능이 우수합니다. 외부 검증기가 없는 작업(창작 글쓰기, 형식화)에서는 CRITIC이 Self-Refine으로 축소됩니다.

### 중단 조건

두 가지 일반적인 형태:

1. **검증기 통과.** 외부 테스트가 성공을 반환합니다. 사용 가능할 때 선호됩니다(단위 테스트, 타입 체커, 가드레일 어서션).
2. **피드백 미발행.** 모델이 "출력이 괜찮습니다"라고 말합니다. 더 저렴하지만 신뢰할 수 없으며, 최대 반복 횟수 상한과 함께 사용하세요.

2026년 기본값: 두 가지를 결합합니다. "검증기가 통과하거나 모델이 괜찮다고 말하고 반복 횟수 >= 2이거나 반복 횟수 >= max_iterations이면 중단하세요."

### 평가자-최적화자 (Anthropic, 2024)

Anthropic의 2024년 12월 게시물은 이를 5가지 워크플로우 패턴 중 하나로 지정합니다. 두 가지 역할:

- 평가자: 출력을 점수화하고 비평을 생성합니다.
- 최적화자: 비평을 받아 출력을 수정합니다.

평가자가 통과할 때까지 반복하세요. 이는 Anthropic의 관점에서 Self-Refine/CRITIC입니다. Anthropic이 추가한 중요한 엔지니어링 세부 사항: 평가자와 최적화자 프롬프트는 모델이 단순히 승인 도장만 찍지 않도록 충분히 달라야 합니다.

### OpenAI Agents SDK 출력 가드레일

OpenAI Agents SDK는 이 패턴을 "출력 가드레일(output guardrails)"로 제공합니다. 가드레일은 에이전트의 최종 출력에 대해 실행되는 검증기입니다. 가드레일이 트리거되면 (`OutputGuardrailTripwireTriggered`를 발생시키면) 출력이 거부되고 에이전트는 재시도할 수 있습니다. 가드레일은 도구를 호출할 수 있으며 (CRITIC 스타일) 순수 함수일 수도 있습니다 (Self-Refine 스타일).

### 2026년 함정

- **도장 찍기 루프.** 동일한 모델이 동일한 프롬프트 스타일로 생성과 비판을 수행하면 "나쁘지 않다"로 수렴합니다. 구조적으로 다른 프롬프트를 사용하거나, 비판에는 작고 저렴한 모델을 사용하세요.
- **과도한 정제.** 각 정제 패스는 지연 시간과 토큰을 추가합니다. 1~3 패스로 예산을 제한하세요. 그 이후에는 인간 검토로 에스컬레이션하세요.
- **사소한 작업에 CRITIC 사용.** 외부 검증기가 없으면 CRITIC은 Self-Refine으로 퇴화합니다. 스텁(stub) 검증기를 위해 지연 시간을 지불하지 마세요.

```figure
self-refine
```

## 구현하기

`code/main.py`는 장난감 작업에 Self-Refine과 CRITIC을 구현합니다: 주어진 주제에 대해 짧은 불릿 리스트를 생성합니다. 검증기는 형식을 확인합니다 (3개의 불릿, 각각 60자 미만). CRITIC은 알려진 환각(hallucination)에 페널티를 부여하는 외부 "사실 검증기"를 추가합니다.

구성 요소:

- `generate` — 스크립트화된 생성자.
- `feedback` — LLM 스타일 자기 비판.
- `verify_external` — CRITIC 스타일 그라운딩된 검증기.
- `refine` — 기록(history)을 고려하여 출력을 재작성합니다.
- 중단 조건 — 검증기가 통과하거나 최대 4번 반복합니다.

실행해 보세요:

```
python3 code/main.py
```

Self-Refine과 CRITIC 실행을 비교하세요. CRITIC은 Self-Refine이 놓친 사실적 오류를 포착합니다. 외부 검증기는 자기 비판자가 가지지 못한 그라운딩을 가지고 있기 때문입니다.

## 사용하기

Anthropic의 평가자-최적화자(evaluator-optimizer)는 Claude 친화적 언어로 표현된 이 패턴입니다. OpenAI Agents SDK의 출력 가드레일은 CRITIC 형태입니다 (가드레일이 도구를 호출할 수 있음). LangGraph는 Self-Refine처럼 읽히는 리플렉션 노드를 제공합니다. Google의 Gemini 2.5 Computer Use는 CRITIC 변형인 단계별 안전 평가기를 추가합니다: 모든 행동은 커밋 전에 검증됩니다.

## 출시하기

`outputs/skill-refine-loop.md`는 작업 형태, 검증기 가용성, 반복 예산을 고려하여 평가자-최적화자 루프를 구성합니다. 생성자, 평가자/검증기, 최적화자를 위한 프롬프트와 중단 정책을 생성합니다.

## 연습 문제

1. max_iterations=1로 장난감(toy)을 실행해 보세요. CRITIC이 여전히 도움이 되나요?
2. 외부 검증기를 노이즈가 있는 것으로 교체해 보세요 (무작위로 30%의 오탐(false positives) 발생). 루프는 어떻게 작동하나요? 이것이 2026년 대부분의 가드레일 스택의 현실입니다.
3. "다른 모델에서의 생성자-비평가(generator-critic on different models)" 변형을 구현해 보세요: 큰 모델이 생성하고, 작은 모델이 비평합니다. 같은 모델 방식보다 더 나은 결과를 내나요?
4. CRITIC 논문 섹션 3 (arXiv:2305.11738 v4)을 읽어 보세요. 세 가지 검증 도구 범주의 이름을 나열하고 각각에 대한 예시를 제시해 보세요.
5. OpenAI Agents SDK의 `output_guardrails`을 CRITIC의 검증기 역할에 매핑해 보세요. SDK는 무엇을 잘못하고 있으며, 무엇을 잘하고 있나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Self-Refine | "스스로 고치는 LLM" | 하나의 모델 내에서 생성 -> 피드백 -> 정제 루프를 수행하며, 히스토리를 유지 |
| CRITIC | "도구 기반 검증" | 피드백을 외부 검증기(검색, 코드, 계산, 테스트)로 대체 |
| Evaluator-Optimizer | "Anthropic 워크플로우 패턴" | 두 가지 역할 — 평가자가 점수를 매기고, 최적화자가 수정 — 수렴할 때까지 반복 |
| Output guardrail | "사후 검사(Post-hoc check)" | 에이전트가 출력을 생성한 후 실행되는 OpenAI Agents SDK 검증기 |
| Verify step | "비평 단계(Critique phase)" | 핵심적인 결정: 근거 기반(grounded)인지 자기 평가(self-rated)인지 |
| Refine history | "모델이 이미 시도한 것" | 정제 프롬프트 앞에 이전 출력과 비평을 추가; 이를 제거하면 품질이 급격히 저하됨 |
| Rubber-stamp loop | "자기 합의 실패(Self-agreement failure)" | 동일한 프롬프트의 비평이 "좋아 보여요"라고 반환; 구조적으로 다른 프롬프트로 수정 |
| Stop condition | "수렴 테스트(Convergence test)" | 검증기가 통과하거나 피드백이 없고 반복 상한에 도달; 단일 조건만으로는 절대 사용하지 않음 |

## 추가 읽기

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 표준 논문
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 도구 기반 검증
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 평가자-최적화자 워크플로우 패턴
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — CRITIC 형태의 검증기로서의 출력 가드레일
