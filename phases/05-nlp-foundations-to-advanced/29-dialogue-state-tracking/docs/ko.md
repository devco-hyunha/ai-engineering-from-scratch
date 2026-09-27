# 대화 상태 추적 (Dialogue State Tracking)

> "북쪽에 있는 저렴한 식당을 원해요... 사실 중간 가격대로 해주고... 이탈리안 요리도 추가해 주세요." 세 번의 턴, 세 번의 상태 업데이트. DST는 예약이 원활하게 진행되도록 슬롯-값(slot-value) 딕셔너리를 동기화합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## 문제점 (The Problem)

태스크 지향 대화 시스템(task-oriented dialogue system)에서 사용자의 목표는 슬롯-값 쌍(slot-value pairs)의 집합으로 인코딩됩니다: `{cuisine: italian, area: north, price: moderate}`. 사용자는 매 턴마다 슬롯을 추가, 변경 또는 삭제할 수 있습니다. 시스템은 대화 전체를 파악하여 현재 상태를 정확하게 출력해야 합니다.

단 하나의 슬롯이라도 잘못 파악하면 시스템은 잘못된 레스토랑을 예약하거나, 잘못된 항공편을 일정에 잡거나, 잘못된 카드로 결제하게 됩니다. 대화 상태 추적(DST, Dialogue State Tracking)은 사용자가 말한 내용과 백엔드가 실행하는 내용 사이를 잇는 핵심 연결 고리입니다.

LLM(대규모 언어 모델) 시대인 2026년에도 이것이 여전히 중요한 이유:

- 규제 준수가 중요한 도메인(금융, 의료, 항공 예약)은 자유 형식의 생성(free-form generation)이 아닌 결정론적인 슬롯 값을 필요로 합니다.
- 도구 사용 에이전트(Tool-use agents)는 API를 호출하기 전에 여전히 슬롯 해소(slot resolution) 과정이 필요합니다.
- 다회차 수정(Multi-turn correction)은 보기보다 까다롭습니다: "아니 사실, 목요일로 바꿔줘."

현대의 파이프라인: 고전적인 DST 개념 + LLM 추출기(extractors) + 구조화된 출력 가드레일(structured-output guardrails).

## 개념 (The Concept)

![DST: dialog history → slot-value state](../assets/dst.svg)

**태스크 구조 (Task structure).** 스키마는 도메인(음식점, 호텔, 택시)과 그에 따른 슬롯(요리 종류, 지역, 가격, 인원수)을 정의합니다. 각 슬롯은 비어 있거나, 폐쇄된 집합 내의 값(가격: {저렴함, 보통, 비쌈})으로 채워지거나, 자유 형식의 값(이름: "The Copper Kettle")으로 채워질 수 있습니다.

**두 가지 DST 정식화 (Two DST formulations).**

- **분류 (Classification).** 각 `(slot, candidate_value)` 쌍에 대해 yes/no를 예측합니다. 폐쇄형 어휘(closed-vocab) 슬롯에 적합합니다. 2020년 이전의 표준 방식입니다.
- **생성 (Generation).** 대화 내용을 바탕으로 슬롯 값을 자유 텍스트로 생성합니다. 개방형 어휘(open-vocab) 슬롯에 적합합니다. 현대적인 기본 방식입니다.

**평가 지표 (Metric).** 공동 목표 정확도 (Joint Goal Accuracy, JGA) — *모든* 슬롯이 정확한 턴(turn)의 비율을 의미합니다. 전부 아니면 전무(All-or-nothing) 방식입니다. MultiWOZ 2.4 리더보드의 상위권은 2026년 기준 약 83%에 달합니다.

**아키텍처 (Architectures).**

1. **규칙 기반 (Rule-based; 슬롯 정규표현식 + 키워드).** 좁은 도메인에 대한 강력한 베이스라인입니다. 디버깅이 용이합니다.
2. **TripPy / BERT-DST.** BERT 인코딩을 사용한 복사 기반 생성 방식입니다. LLM 이전의 표준입니다.
3. **LDST (LLaMA + LoRA).** 도메인-슬롯 프롬프팅을 사용하는 지시어 튜닝(Instruction-tuned)된 LLM입니다. MultiWOZ 2.4에서 ChatGPT 수준의 품질에 도달합니다.
4. **온톨로지 프리 (Ontology-free; 2024–26).** 스키마를 건너뛰고 슬롯 이름과 값을 직접 생성합니다. 개방형 도메인을 처리할 수 있습니다.
5. **프롬프트 + 구조화된 출력 (Prompt + structured output; 2024–26).** Pydantic 스키마와 제약 디코딩(constrained decoding)을 결합한 LLM 방식입니다. 단 5줄의 코드로 구현 가능하며, 프로덕션 환경에 즉시 적용할 수 있습니다.

### 전형적인 실패 사례 (The classic failure modes)

- **대화 맥락 간 상호 참조 (Co-reference across turns).** "첫 번째 옵션으로 할게요."라고 말할 때, 어떤 옵션인지 식별해야 합니다.
- **덮어쓰기 vs 추가 (Over-write vs append).** 사용자가 "이탈리아 요리 추가해 줘"라고 말했을 때, 기존 요리 종류를 교체해야 할까요, 아니면 추가해야 할까요?
- **암시적 확인 (Implicit confirmations).** "오, 좋아요" — 이 말이 제안된 예약을 수락한 것일까요?
- **수정 (Correction).** "사실 저녁 7시로 바꿔 줘." 다른 슬롯을 지우지 않고 시간만 업데이트해야 합니다.
- **이전 시스템 발화에 대한 상호 참조 (Coreference to previous system utterance).** "네, 그걸로 할게요." 여기서 "그것"은 무엇일까요?

```figure
n5-slot-tracker
```

## 직접 구현해 보기 (Build It)

### 1단계: 규칙 기반 슬롯 추출기 (rule-based slot extractor)

`code/main.py`를 참조하세요. 정규 표현식(Regex)과 유의어 사전(synonym dictionaries)을 사용하면 좁은 도메인 내의 표준 발화(canonical utterances) 70%를 처리할 수 있습니다.

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

표준 어휘 범위를 벗어나면 취약해집니다. 결정론적인 슬롯 확인(deterministic slot confirmations) 작업에는 적합합니다.

### 2단계: 상태 업데이트 루프 (state update loop)

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

세 가지 불변 조건(invariants):

- 사용자가 건드리지 않은 슬롯은 절대 초기화하지 마세요.
- 명시적 부정("음식 종류는 신경 쓰지 마세요")은 값을 삭제(clear)해야 합니다.
- 사용자의 수정("사실은...")은 값을 추가(append)하는 것이 아니라 덮어쓰기(overwrite)해야 합니다.

### 3단계: 구조화된 출력을 활용한 LLM 기반 DST (LLM-driven DST with structured output)

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor와 Pydantic을 사용하면 유효한 상태 객체(state object)를 보장할 수 있습니다. 정규 표현식(regex) 오류, 스키마 불일치, 혹은 환각된 슬롯(hallucinated slots) 문제가 발생하지 않습니다.

### 4단계: JGA 평가 (JGA evaluation)

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

교정(Calibrate): 시스템이 모든 슬롯(slot)을 정확하게 맞춘 턴(turn)의 비율은 얼마인가요? MultiWOZ 2.4의 2026년 상위 시스템들의 경우 80~83%를 기록했습니다. 여러분의 도메인 특화(in-domain) 시스템은 제한된 어휘(narrow vocabulary) 환경에서 이 수치를 상회해야 하며, 그렇지 않으면 LLM 베이스라인에 뒤처지게 됩니다.

### 5단계: 수정 처리 (Handling Correction)

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

수정이 감지되면, 기존 슬롯에 내용을 추가(append)하는 대신 마지막으로 업데이트된 슬롯을 덮어쓰기(overwrite)해야 합니다. LLM의 도움 없이 이를 정확하게 구현하기는 어렵습니다. 최신 패턴은 다음과 같습니다: 슬롯을 점진적으로 업데이트하는 대신, 항상 LLM이 전체 대화 기록(history)을 바탕으로 전체 상태(state)를 다시 생성하도록 하는 것입니다. 이 방식은 수정 사항을 자연스럽게 처리합니다.

## 주의 사항 (Pitfalls)

- **전체 이력 재생성 비용 (Full-history regeneration cost).** 매 턴마다 LLM이 상태를 다시 생성하게 하면 총 토큰 비용이 $O(n^2)$로 증가합니다. 이력의 길이를 제한하거나 오래된 대화 내용을 요약하세요.
- **스키마 드리프트 (Schema drift).** 사후에 새로운 슬롯(slot)을 추가하면 기존 학습 데이터가 손상될 수 있습니다. 스키마의 버전을 관리하세요.
- **대소문자 구분 (Case sensitivity).** "Italian" vs "italian" vs "ITALIAN" — 모든 곳에서 정규화(normalize)를 수행하세요.
- **암시적 상속 (Implicit inheritance).** 사용자가 이전에 "4명용"이라고 지정했다면, 다른 시간에 대한 새로운 요청이 들어오더라도 인원수 정보가 지워져서는 안 됩니다. 항상 전체 이력을 전달하세요.
- **자유 형식 vs 폐쇄형 세트 (Free-form vs closed-set).** 이름, 시간, 주소는 자유 형식 슬롯이 필요하며, 요리 종류나 지역은 폐쇄형 세트가 적합합니다. 스키마에 이 두 가지를 혼합하여 사용하세요.

## 사용 방법 (Use It)

2026년 기술 스택:

| 상황 (Situation) | 접근 방식 (Approach) |
|-----------|----------|
| 좁은 도메인 (한두 개의 의도) | 규칙 기반(Rule-based) + 정규표현식(regex) |
| 넓은 도메인, 라벨링된 데이터가 있는 경우 | LDST (MultiWOZ 스타일 데이터 기반의 LLaMA + LoRA) |
| 넓은 도메인, 라벨 없음, 프로덕션 준비 완료 | LLM + Instructor + Pydantic 스키마 |
| 음성 / 보이스 (Spoken / voice) | ASR + 정규화기(normalizer) + LLM-DST |
| 멀티 도메인 예약 흐름 (Multi-domain booking flow) | 도메인별 Pydantic 모델을 사용하는 스키마 가이드형 LLM |
| 컴플라이언스 민감 (Compliance-sensitive) | 규칙 기반을 우선 적용하고, 확인 절차를 포함한 LLM 폴백(fallback) 사용 |

## Ship It

`outputs/skill-dst-designer.md`로 저장하세요:

```markdown
---
name: dst-designer
description: 대화 상태 추적기(Dialogue State Tracker) 설계 — 스키마, 추출기, 업데이트 정책, 평가.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

사용 사례(도메인, 언어, 어휘 개방성, 컴플라이언스 요구사항)가 주어지면 다음을 출력하세요:

1. 스키마(Schema). 도메인 목록, 도메인별 슬롯(slot), 슬롯별 개방형(open) vs 폐쇄형(closed) 어휘.
2. 추출기(Extractor). 규칙 기반(Rule-based) / seq2seq / Pydantic을 활용한 LLM 방식 중 선택 및 근거 제시.
3. 업데이트 정책(Update policy). 전체 상태 재생성(Regenerate-whole-state) / 증분 업데이트(incremental); 수정 처리(correction handling); 부정 표현 처리(negation handling).
4. 평가(Evaluation). 별도의 대화 데이터셋에 대한 공동 목표 정확도(Joint Goal Accuracy), 슬롯 수준의 정밀도/재현율(precision/recall), 가장 어려운 슬롯에서의 혼동(confusion).
5. 확인 흐름(Confirmation flow). 사용자에게 명시적 확인을 요청해야 하는 시점(파괴적 작업, 신뢰도가 낮은 추출 결과).

컴플라이언스에 민감한 슬롯에 대해 규칙 기반의 2차 검증이 없는 LLM 전용 DST는 거부하세요. 사용자의 수정 시 슬롯을 롤백(roll back)할 수 없는 DST는 거부하세요. 버전 태그가 없는 스키마는 플래그를 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에 3개의 슬롯(`cuisine`, `area`, `price`)을 처리하는 규칙 기반 상태 추적기(rule-based state tracker)를 구축해 보세요. 직접 작성한 10개의 대화 데이터셋으로 테스트하고, JGA를 측정해 보세요.
2. **중간 (Medium).** 동일한 데이터셋에 Instructor + Pydantic + 소형 LLM을 사용하여 구현해 보세요. JGA를 비교하고, 가장 난이도가 높았던 턴(turn)들을 분석해 보세요.
3. **어려움 (Hard).** 두 방식을 모두 구현하고 라우팅(routing) 로직을 적용해 보세요: 규칙 기반 방식을 기본(primary)으로 사용하되, 규칙 기반 방식이 신뢰도(confidence)와 함께 2개 미만의 슬롯을 출력할 경우 LLM을 폴백(fallback)으로 사용합니다. 결합된 JGA와 턴당 추론 비용(inference cost)을 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking (대화 상태 추적) | 대화 턴이 진행되는 동안 슬롯-값(slot-value) 딕셔너리를 유지하는 것. |
| Slot | 사용자 의도의 단위 | 백엔드에서 필요한 명명된 파라미터 (예: 요리 종류, 날짜). |
| Domain | 작업 영역 | 레스토랑, 호텔, 택시 등 — 슬롯들의 집합. |
| JGA | Joint Goal Accuracy (공동 목표 정확도) | 모든 슬롯이 정확한 턴의 비율. 전부 아니면 전무(All-or-nothing) 방식. |
| MultiWOZ | 벤치마크 | 멀티 도메인 WOZ 데이터셋; 표준 DST 평가 방식. |
| Ontology-free DST | 스키마 없음 | 고정된 목록 없이 슬롯 이름과 값을 직접 생성하는 방식. |
| Correction | "사실은..." | 이전에 채워진 슬롯을 덮어쓰는 대화 턴. |

## 추가 읽을거리 (Further Reading)

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 표준 벤치마크입니다.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — DST를 위한 LLaMA + LoRA 지시어 튜닝(instruction tuning)에 관한 연구입니다.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 복사 기반(copy-based) DST의 핵심 연구입니다.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM 기반의 비지도 학습 TOD 연구입니다.
- [MultiWOZ 리더보드(leaderboard)](https://github.com/budzianowski/multiwoz) — 표준 DST 결과들을 확인할 수 있습니다.
