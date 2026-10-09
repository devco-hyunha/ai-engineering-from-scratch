# 대화 상태 추적

> "북쪽의 저렴한 식당을 원합니다... 실제로는 중급으로... 그리고 이탈리아 음식을 추가하세요." 세 번의 발화, 세 번의 상태 업데이트. DST는 슬롯-값 사전(slot-value dict)을 동기화하여 예약이 정상적으로 작동하도록 합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 17강 (챗봇), 5단계 · 20강 (구조화된 출력)
**시간:** 약 75분

## 문제점

작업 지향 대화 시스템에서 사용자의 목표는 슬롯-값 쌍(slot-value pairs)의 집합으로 인코딩됩니다: `{cuisine: italian, area: north, price: moderate}`. 모든 사용자 발화는 슬롯을 추가, 변경 또는 삭제할 수 있습니다. 시스템은 전체 대화를 읽고 현재 상태를 정확하게 출력해야 합니다.

단 하나의 슬롯이라도 잘못되면 시스템은 잘못된 식당을 예약하거나, 잘못된 항공편을 스케줄링하거나, 잘못된 카드를 청구합니다. DST는 사용자가 말한 것과 백엔드가 실행하는 것 사이의 연결 고리입니다.

LLM이 있음에도 2026년에도 여전히 중요한 이유:

- 규제 민감 도메인(은행, 헬스케어, 항공 예약)은 자유 형식 생성이 아닌 결정론적 슬롯 값을 요구합니다.
- 도구 사용 에이전트도 API를 호출하기 전에 슬롯 해석이 필요합니다.
- 다중 발화 수정은 보이는 것보다 어렵습니다: "사실은 아니요, 목요일로 해주세요."

현대적 파이프라인: 고전적 DST 개념 + LLM 추출기 + 구조화된 출력 가드레일.

## 개념

![DST: dialog history → slot-value state](../assets/dst.svg)

**작업 구조.** 스키마는 도메인(식당, 호텔, 택시)과 그 슬롯(음식 종류, 지역, 가격, 인원)을 정의합니다. 각 슬롯은 비어 있거나, 닫힌 집합의 값(price: {cheap, moderate, expensive})으로 채워지거나, 자유 형식 값(name: "The Copper Kettle")일 수 있습니다.

**두 가지 DST 공식화.**

- **분류.** 각 (슬롯, 후보 값) 쌍에 대해 예/아니오를 예측합니다. 어휘가 닫혀 있는 슬롯에 작동합니다. 2020년 이전의 표준입니다.
- **생성.** 대화를 입력으로 받아 슬롯 값을 자유 텍스트로 생성합니다. 어휘가 열려 있는 슬롯에 작동합니다. 현대의 기본값입니다.

**지표.** Joint Goal Accuracy (JGA) — *모든* 슬롯이 정확한 발화의 비율입니다. 전부 맞거나 전부 틀린 방식입니다. 2026년 MultiWOZ 2.4 리더보드 상위는 약 83%입니다.

**아키텍처.**

1. **규칙 기반 (슬롯 정규식 + 키워드).** 좁은 도메인에 대한 강력한 기준선. 디버깅이 가능합니다.
2. **TripPy / BERT-DST.** BERT 인코딩을 사용하는 복사 기반 생성. LLM 이전의 표준입니다.
3. **LDST (LLaMA + LoRA).** 도메인 슬롯 프롬프트로 지시문 미세 조정된 LLM. MultiWOZ 2.4에서 ChatGPT 수준의 품질에 도달합니다.
4. **온톨로지 프리 (2024–26).** 스키마를 건너뛰고 슬롯 이름과 값을 직접 생성합니다. 개방형 도메인을 처리합니다.
5. **프롬프트 + 구조화된 출력 (2024–26).** Pydantic 스키마 + 제약 디코딩을 사용하는 LLM. 5줄의 코드로 프로덕션 준비가 완료됩니다.

### 고전적인 실패 모드

- **턴 간 참조(Co-reference).** "첫 번째 옵션으로 유지합시다." 어떤 옵션인지 해결해야 합니다.
- **덮어쓰기 vs 추가.** 사용자가 "이탈리아 요리 추가"라고 말합니다. 요리 유형을 교체합니까, 추가합니까?
- **암묵적 확인.** "좋아요, 멋지네요" — 제안된 예약을 수락한 것입니까?
- **수정.** "사실은 7시로 해주세요." 다른 슬롯을 지우지 않고 시간을 업데이트해야 합니다.
- **이전 시스템 발화 참조.** "네, 그거요." 어떤 "그거"를 말하는 것입니까?

```figure
n5-slot-tracker
```

## 구현하기

### 1단계: 규칙 기반 슬롯 추출기

`code/main.py`를 참조하세요. 정규식 + 동의어 사전은 좁은 도메인의 표준 발화 중 70%를 커버합니다:

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

표준 어휘 밖에서는 취약합니다. 결정적인 슬롯 확인에는 잘 작동합니다.

### 2단계: 상태 업데이트 루프

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

세 가지 불변 조건:

- 사용자가 건드리지 않은 슬롯은 절대 리셋하지 마세요.
- 명시적 부정 ("요리 유형은 신경 쓰지 마세요")은 반드시 지워야 합니다.
- 사용자 수정 ("사실은...")은 추가가 아닌 덮어쓰기를 해야 합니다.

### 3단계: 구조화된 출력으로 LLM 기반 DST

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

Instructor + Pydantic은 유효한 상태 객체를 보장합니다. 정규식, 스키마 불일치, 환각된 슬롯이 없습니다.

### 4단계: JGA 평가

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

보정하세요: 시스템이 모든 슬롯을 정확히 맞춘 턴의 비율은 얼마입니까? MultiWOZ 2.4의 경우, 2026년 상위 시스템은 80-83%입니다. 좁은 어휘를 사용하는 도메인 시스템은 이보다 높아야 하며, 그렇지 않으면 LLM 기준선이 당신을 이깁니다.

### 5단계: 수정 처리

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

교정이 감지되면, 추가하지 않고 마지막에 업데이트된 슬롯을 덮어쓰세요. LLM의 도움 없이 이를 정확히 구현하기는 어렵습니다. 현대적인 패턴은: 전체 히스토리에서 LLM이 상태를 재생성하도록 항상 허용하는 것입니다. 증분 업데이트를 하지 마세요. 이 방식은 교정을 자연스럽게 처리합니다.

## 함정

- **전체 히스토리 재생성 비용.** 매 턴마다 LLM이 상태를 재생성하면 총 토큰 수가 O(n²)가 됩니다. 히스토리를 제한하거나 이전 턴을 요약하세요.
- **스키마 드리프트.** 사후에 새로운 슬롯을 추가하면 이전 학습 데이터가 깨집니다. 스키마를 버전 관리하세요.
- **대소문자 민감도.** "Italian", "italian", "ITALIAN" — 모든 곳에서 정규화하세요.
- **암시적 상속.** 사용자가 이전에 "4명"을 지정했다면, 다른 시간에 대한 새로운 요청이 인원 수를 초기화하지 않아야 합니다. 항상 전체 히스토리를 전달하세요.
- **자유 형식 vs 폐쇄 집합.** 이름, 시간, 주소는 자유 형식 슬롯이 필요하며, 요리 종류와 지역은 폐쇄 집합입니다. 스키마에 두 가지를 모두 포함하세요.

## 사용하기

2026년 스택:

| 상황 | 접근 방식 |
|-----------|----------|
| 좁은 도메인 (하나 또는 두 개의 의도) | 규칙 기반 + 정규식 |
| 넓은 도메인, 레이블된 데이터 가용 | LDST (MultiWOZ 스타일 데이터에 LLaMA + LoRA) |
| 넓은 도메인, 레이블 없음, 프로덕션 준비 | LLM + Instructor + Pydantic 스키마 |
| 음성 / 보이스 | ASR + 정규화기 + LLM-DST |
| 다중 도메인 예약 흐름 | 도메인별 Pydantic 모델이 있는 스키마 가이드 LLM |
| 컴플라이언스 민감 | 규칙 기반 주력, 확인 흐름이 있는 LLM 폴백 |

## 출시하기

`outputs/skill-dst-designer.md`로 저장하세요:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 3개 슬롯(요리 종류, 지역, 가격)에 대한 규칙 기반 상태 추적기를 구축하세요. 10개의 수작업 대화로 테스트하세요. JGA를 측정하세요.
2. **중간.** Instructor + Pydantic + 작은 LLM을 사용하여 동일한 데이터셋을 처리하세요. JGA를 비교하세요. 가장 어려운 턴을 검사하세요.
3. **어려움.** 두 가지를 모두 구현하고 라우팅하세요: 규칙 기반 주력, 규칙 기반이 신뢰도와 함께 2개 미만 슬롯을 방출할 때 LLM 폴백. 결합된 JGA와 턴당 추론 비용을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| DST | 대화 상태 추적 | 대화 턴에 걸쳐 슬롯-값 사전(slot-value dict)을 유지합니다. |
| Slot | 사용자 의도의 단위 | 백엔드가 필요로 하는 명명된 매개변수(요리 종류, 날짜 등)입니다. |
| Domain | 작업 영역 | 레스토랑, 호텔, 택시 — 슬롯의 집합입니다. |
| JGA | Joint Goal Accuracy | 모든 슬롯이 정확한 턴의 비율입니다. 전부 맞아야 합니다. |
| MultiWOZ | 벤치마크 | 다중 도메인 WOZ 데이터셋; 표준 DST 평가입니다. |
| Ontology-free DST | 스키마 없음 | 고정된 목록 없이 슬롯 이름과 값을 직접 생성합니다. |
| Correction | "사실은..." | 이전에 채워진 슬롯을 덮어쓰는 턴입니다. |

## 추가 읽기

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 표준 벤치마크입니다.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — DST를 위한 LLaMA + LoRA 지시 미세 조정입니다.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 복사 기반 DST의 핵심 작업입니다.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM 기반의 비지도 TOD입니다.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 표준 DST 결과입니다.
