# 챗봇 — 규칙 기반에서 신경망, LLM 에이전트로

> ELIZA는 패턴 매칭으로 응답했습니다. DialogFlow는 의도를 매핑했습니다. GPT는 가중치에서 답변했습니다. Claude는 도구를 실행하고 검증합니다. 각 시대는 이전 시대의 가장 큰 실패를 해결했습니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 13강 (질문 답변), 5단계 · 14강 (정보 검색)
**시간:** 약 75분

## 문제점

사용자가 "비행기를 변경하고 싶어요"라고 말합니다. 시스템은 사용자가 무엇을 원하는지, 어떤 정보가 누락되었는지, 어떻게 그 정보를 얻는지, 그리고 어떻게 작업을 완료하는지 파악해야 합니다. 그런 다음 사용자가 "잠깐, 취소하는 건 어때요?"라고 말하면, 시스템은 컨텍스트를 기억하고, 작업을 전환하며, 상태를 유지해야 합니다.

대화는 ML 시스템에게 어렵습니다. 입력은 열려 있습니다. 출력은 여러 턴에 걸쳐 일관되어야 합니다. 시스템은 세계에 행동해야 할 수도 있습니다 (비행기를 변경하거나, 카드를 청구하는 등). 모든 잘못된 단계는 사용자에게 보입니다.

챗봇 아키텍처는 네 가지 패러다임을 순환해 왔으며, 각각은 이전 패러다임이 너무 눈에 띄게 실패했기 때문에 도입되었습니다. 이 강의는 순서대로 이들을 다룹니다. 2026년 생산 환경은 마지막 두 패러다임의 하이브리드입니다.

## 개념

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### 스크립트된 반세기, 1950-2001

첫 번째 패러다임은 5년 동안 지속되지 않았습니다. 50년 동안 지속되었습니다. 그 흐름을 아는 것이 중요한 이유는 그 안의 모든 시스템이 같은 기계이기 때문입니다 — 입력을 매칭하고, 미리 준비된 응답을 방출하고, 약간의 상태를 업데이트합니다. 그 기계에 규칙을 추가한 50년은 일반 케이스를 만들어내지 못했습니다. 그 한계는 두 번째부터 네 번째 패러다임이 존재하는 이유입니다.

**1950.** 튜링은 "기계가 생각할 수 있는가?"라는 질문을 우회하여, 텔레타이프를 통해 기계가 사람과 구별되지 않는다면 철학적 질문은 무의미하다는 운영적 대체안을 제안합니다. 대화가 필드의 벤치마크가 되지만, 필드에는 아직 이름이 없습니다.

**1956.** 이름이 등장합니다. 다트머스에서의 여름 워크숍이 "인공지능"이라는 용어를 만들며, 지능의 모든 특징이 "원리적으로 매우 정밀하게 설명될 수 있어서 기계가 이를 시뮬레이션할 수 있다"는 가설을 세웁니다. 제안은 상당한 진전을 위해 두 달의 예산을 배정합니다.

**1966.** ELIZA는 1단계에서 구축하는 반사(trick) 기법을 출시했습니다. 분해 규칙은 입력에서 조각을 추출하고, 재조립 규칙은 이를 질문으로 되돌려줍니다. 총 약 200개의 패턴, 상태 없음, 이해 없음 — 그런데도 사용자는 ELIZA에게 마음을 터놓았습니다. Weizenbaum은 이토록 적은 기계적 장치로 가능한 것에 대해 남은 커리어 동안 경각심을 가졌습니다.

**1972.** Stanford에서 편집증(paranoia)을 모델링하기 위해 구축된 PARRY는 ELIZA가 결여했던 요소, 즉 내부 상태를 추가했습니다. 두려움, 분노, 불신에 대한 수치 변수가 매 턴마다 업데이트되며 다음에 실행될 스크립트를 게이트합니다. 따라서 동일한 입력이라도 지금까지의 대화에 따라 다른 응답이 생성됩니다. 블라인드 테스트에서 정신과 의사는 PARRY와 인간 환자 간에 우연한 수준의 구별만 해냈습니다. 이는 페르소나 조건부(persona conditioning)의 직접적인 조상입니다. 세 개의 float로 구현된 시스템 프롬프트인 셈입니다. 같은 해, 두 봇이 ARPANET을 통해 서로를 향하도록 지정되었습니다. 치료사 스크립트가 편집증 상태 기계와 인터뷰하는, 네트워크에서의 첫 봇 간 대화였습니다.

**1995.** ALICE는 패턴-템플릿 쌍을 위한 XML 방언인 AIML로 ELIZA 레시피를 확장했습니다. 약 40,000개의 수작업 작성된 카테고리, Loebner Prize 3회 수상. 이는 규칙 기반 시스템의 확장 법칙을 증명했습니다. 더 많은 규칙은 커버리지를 구매하지만, 범용성은 구매하지 못합니다. 모든 규칙은 누군가가 유지해야 하는 부채입니다.

**2001.** SmarterChild는 이 레시피를 3,000만 인스턴트 메신저 사용자 앞에 배치하고, 백엔드 조회(weather, stocks, movie times)를 템플릿에 접합하는 기능을 추가했습니다. 자세히 보면 2001년 의상을 입은 도구 호출(tool calling)입니다. 의도를 파싱하고, 서비스를 호출하고, 결과를 응답에 렌더링합니다.

50년, 하나의 메커니즘, 증가하는 규칙 수. 이 패러다임은 누군가가 이를 반증해서 끝난 것이 아닙니다. 수작업 작성된 상태 기계의 유지보수 비용이 커버리지에 따라 선형적으로 증가하는 반면, 사용자 기대는 그들이 지난 주에 본 것에 따라 증가하기 때문에 끝났습니다.

```figure
chatbot-lineage
```

**규칙 기반 (ELIZA, AIML, DialogFlow).** 수작업 작성된 패턴이 사용자 입력과 매칭되어 응답을 생성합니다. 의도 분류기가 사전 정의된 플로우로 라우팅합니다. 슬롯 채우기 상태 기계가 필요한 정보를 수집합니다. 설계된 좁은 범위 내에서는 훌륭하게 작동합니다. 그 범위 밖에서는 즉시 실패합니다. 환각이 허용되지 않는 안전 크리티컬 도메인(은행 인증, 항공 예약)에서는 여전히 출시되고 있습니다.

**검색 기반.** FAQ 스타일 시스템입니다. 모든 (발화, 응답) 쌍을 인코딩합니다. 런타임 시 사용자의 메시지를 인코딩하고 가장 가까운 저장된 응답을 검색합니다. Zendesk의 고전적인 "유사 기사" 기능을 생각해 보세요. 규칙보다 패러프레이징(paraphrasing)을 더 잘 처리합니다. 생성이 없으므로 환각(Hallucination)이 없습니다.

**신경망 (seq2seq).** 대화 로그로 학습된 인코더-디코더입니다. 응답을 처음부터 생성합니다. 유창하지만 일반적인 출력("모르겠어요")과 사실적 드리프트(factual drift)에 취약합니다. 주제에 항상 신뢰할 수 있게 집중하지는 못합니다. Google, Facebook, Microsoft가 2016-2019년에 실망스러운 챗봇을 내놓은 이유입니다.

**LLM 에이전트.** 계획, 도구 호출, 결과 검증 루프에 감싸진 언어 모델입니다. 긴 프롬프트를 가진 챗봇이 아닙니다. 에이전트 루프: 계획 → 도구 호출 → 결과 관찰 → 다음 단계 결정. 검색 우선 그라운딩(RAG)이 환각(Hallucination)을 방지합니다. 도구 호출을 통해 실제로 작업을 수행할 수 있습니다. 이것이 2026년 아키텍처입니다.

네 가지 패러다임은 순차적인 대체 관계가 아닙니다. 2026년 프로덕션 챗봇은 네 가지를 모두 거칩니다: 인증 및 파괴적 작업에는 규칙 기반, FAQ에는 검색 기반, 자연스러운 표현에는 신경망 생성, 모호한 개방형 쿼리에는 LLM 에이전트를 사용합니다.

## 구현하기

### 1단계: 규칙 기반 패턴 매칭

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20줄로 작성한 ELIZA. 반사 기법("슬퍼요" → "왜 슬퍼하나요")은 Weizenbaum의 1966년 정신치료사 데모의 정석입니다. 여전히 교육적 가치가 있습니다.

### 2단계: 검색 기반 (FAQ)

이 예시 스니펫은 `pip install sentence-transformers`이 필요합니다 (torch를 가져옵니다). 이 강의의 실행 가능한 `code/main.py`은 대신 표준 라이브러리 Jaccard 유사도를 사용하므로 외부 의존성 없이 강의를 실행할 수 있습니다.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

임계값 기반 거절이 핵심 설계 선택입니다. 가장 좋은 매칭이 충분히 가깝지 않으면 `None`을 반환하고 시스템이 에스컬레이션하도록 하세요.

### 3단계: 신경망 생성 (기본)

소규모 지시 미세 조정된 인코더-디코더(FLAN-T5)나 미세 조정된 대화 모델을 사용하세요. 2026년 기준 단독으로는 프로덕션에 사용할 수 없습니다(모순, 주제 이탈, 사실적 무의미함) 하지만, 자연스러운 표현을 위해 하이브리드 시스템 내부에 포함됩니다. DialoGPT 스타일의 디코더 전용 모델은 일관된 응답을 생성하기 위해 명시적인 턴 구분자 및 EOS 처리가 필요합니다. FLAN-T5 text2text 파이프라인은 교육용 예시로 바로 사용할 수 있습니다.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 4단계: LLM 에이전트 루프

2026년 프로덕션 형태:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

세 가지 요소를 지정해 보세요. 도구는 LLM이 호출할 수 있는 함수입니다. 루프는 LLM이 도구 호출 대신 최종 답변을 반환할 때 종료됩니다. 단계 예산은 모호한 작업에서 무한 루프를 방지합니다.

실제 프로덕션에서는 다음이 추가됩니다: 검색 우선 그라운딩(각 LLM 호출 전에 관련 문서를 주입), 가드레일(확인 없이 파괴적인 작업을 거부), 관측 가능성(모든 단계를 기록), 평가(에이전트 동작이 사양에 유지되는지 자동 검사).

### 5단계: 하이브리드 라우팅

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

패턴: 파괴적인 작업에는 결정론적 규칙, canned FAQ에는 검색, 그 외에는 LLM 에이전트를 사용합니다. 이것이 2026년 고객 지원 시스템에 출시되는 형태입니다.

## 사용하기

2026년 스택:

| 사용 사례 | 아키텍처 |
|---------|---------------|
| 예약, 결제, 인증 | 규칙 기반 상태 기계 + 슬롯 채우기 |
| 고객 지원 FAQ | 큐레이션된 답변에 대한 검색 |
| 개방형 도움 채팅 | RAG + 도구 호출을 사용하는 LLM 에이전트 |
| 내부 도구 / IDE 어시스턴트 | 도구 호출(검색, 읽기, 쓰기)을 사용하는 LLM 에이전트 |
| 동반자 / 캐릭터 채팅봇 | 페르소나 시스템 프롬프트로 튜닝된 LLM, 지식에 대한 검색 |

프로덕션에서는 항상 하이브리드 라우팅을 사용하세요. 단일 아키텍처는 모든 요청을 잘 처리하지 못합니다. 라우팅 계층은 일반적으로 작은 의도 분류기입니다.

## 아직 출시되는 실패 모드

- **확신 있는 조작.** LLM 에이전트가 수행하지 않은 작업을 완료했다고 주장합니다. 완화: 결과를 검증하고, 도구 호출을 기록하며, 성공적인 도구 반환 없이 LLM이 무언가를 수행했다고 주장하지 못하게 하세요.
- **프롬프트 인젝션.** 사용자가 시스템 프롬프트를 덮어쓰는 텍스트를 삽입합니다. 2025년 OWASP LLM 애플리케이션 Top 10에서 LLM01로 순위가 매겨졌습니다. 두 가지 유형: 직접 인젝션(채팅에 붙여넣기)과 간접 인젝션(에이전트가 읽는 문서, 이메일, 도구 출력에 숨겨짐).

공격률은 시나리오에 따라 달라집니다. 프론티어 모델의 일반적인 도구 사용 및 코딩 벤치마크에서 측정된 성공률은 약 0.5-8.5% 범위입니다. 특정 고위험 설정(AI 코딩 에이전트에 대한 적응형 공격, 취약한 오케스트레이션)은 약 84%까지 도달했습니다. 프로덕션 CVE에는 EchoLeak (CVE-2025-32711, CVSS 9.3)가 포함됩니다. 이는 공격자가 제어하는 이메일에 의해 트리거되는 Microsoft 365 Copilot의 제로 클릭 데이터 유출 결함입니다.

완화 조치: 루프 전체에서 사용자 입력을 신뢰할 수 없는 것으로 취급하세요. 도구 호출 전에 정제하세요. 도구 출력과 메인 프롬프트를 격리하세요. 에이전트가 먼저 계획을 세우고, 실행하기 전에 각 행동을 그 계획에 대해 검증하는 Plan-Verify-Execute (PVE) 패턴을 사용하세요(이렇게 하면 도구 결과가 계획되지 않은 새로운 행동을 주입하는 것을 방지합니다). 파괴적인 행동에 대해 사용자 확인을 요구하세요. 도구 범위에 최소 권한을 적용하세요.

프롬프트 엔지니어링으로는 이 위험을 완전히 제거할 수 없습니다. 외부 런타임 방어 계층(LLM Guard, 허용 목록 검증, 시맨틱 이상 탐지)이 필요합니다.
- **범위 확대.** 도구 호출이 간접적으로 관련된 정보를 반환하여 에이전트가 작업에서 벗어납니다. 완화 조치: 도구 계약을 좁히세요. 시스템 프롬프트를 집중된 상태로 유지하세요. 작업 이탈률에 대한 평가를 추가하세요.
- **무한 루프.** 에이전트가 같은 도구를 계속 호출합니다. 완화 조치: 단계 예산, 도구 호출 중복 제거, "진전이 이루어지고 있는지"에 대한 LLM 판정.
- **컨텍스트 윈도우 고갈.** 긴 대화로 인해 가장 초기 턴이 컨텍스트에서 밀려납니다. 완화 조치: 이전 턴을 요약하세요. 유사성에 따라 관련 과거 턴을 검색하세요. 또는 긴 컨텍스트 모델을 사용하세요.

## 출시하기

`outputs/skill-chatbot-architect.md`로 저장하세요:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see 14강), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 연습 문제

1. **쉬움.** 위 규칙 기반 응답을 커피숍 주문 봇용 패턴 10개로 구현하세요. 엣지 케이스를 테스트하세요: 이중 주문, 수정, 취소, 불명확한 의도.
2. **중간.** 하이브리드 FAQ + LLM 폴백을 구축하세요. SaaS 제품용 50개 canned FAQ 항목, 문서 사이트에 대한 검색을 통한 LLM 폴백. 100개 실제 지원 질문에 대해 거절률과 정확도를 측정하세요.
3. **어려움.** 위 에이전트 루프를 세 도구(search, read-user-data, send-email)로 구현하세요. 프롬프트 인젝션 시도를 포함하는 50개 테스트 시나리오로 평가를 실행하세요. 작업 이탈률, 실패 작업률, 그리고 모든 인젝션 성공을 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 의도(Intent) | 사용자가 원하는 것 | 범주형 레이블(book_flight, reset_password). 핸들러로 라우팅됩니다. |
| 슬롯(Slot) | 정보의 한 조각 | 봇이 필요로 하는 매개변수(date, destination). 슬롯 채우기는 질문의 순서입니다. |
| RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) | 검색 및 생성 | 관련 문서를 검색한 후 LLM의 응답을 그라운딩합니다. |
| 도구 호출(Function Calling) | 함수 호출 | LLM이 이름 + 인수를 포함한 구조화된 호출을 생성합니다. 런타임이 실행하고 결과를 반환합니다. |
| 에이전트 루프(Agent loop) | 계획, 실행, 검증 | 작업이 완료될 때까지 LLM 호출과 도구 호출을 교대로 실행하는 컨트롤러입니다. |
| 프롬프트 인젝션(Prompt Injection) | 사용자가 프롬프트를 공격함 | 시스템 프롬프트를 덮어쓰려고 시도하는 악성 입력입니다. |

## 추가 읽기

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — 대화가 해당 분야의 벤치마크가 되게 한 논문입니다.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 최초의 규칙 기반 챗봇 논문입니다.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — PARRY의 감정 변수 아키텍처, 최초의 상태 유지 챗봇입니다.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — LLM 에이전트가 주도권을 잡기 직전, Google의 후기 신경망 챗봇 논문입니다.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 에이전트 루프 패턴을 명명한 논문입니다.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2026년에도 유효한 2024년 프로덕션 가이드라인입니다.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 프롬프트 인젝션 논문입니다.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 프롬프트 인젝션을 가장 중요한 보안 문제로 만든 순위입니다.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Execute 및 사용자 확인 흐름을 포함한 실용적인 오케스트레이션 레이어 방어입니다.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 간접 프롬프트 주입(Indirect Prompt Injection)에서 비롯된 표준적인 제로 클릭 데이터 유출(Data Exfiltration) CVE입니다. 쓰기 권한을 가진 에이전트가 런타임 방어를 필요로 하는 이유에 대한 참조 사례입니다.
