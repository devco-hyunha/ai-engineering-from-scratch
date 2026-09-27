# 챗봇 — 규칙 기반에서 신경망을 거쳐 LLM 에이전트까지 (Chatbots — Rule-Based to Neural to LLM Agents)

> ELIZA는 패턴 매칭으로 응답했습니다. DialogFlow는 의도(intents)를 매핑했습니다. GPT는 가중치(weights)를 통해 답변했습니다. Claude는 도구를 실행하고 검증합니다. 각 시대는 이전 시대의 가장 큰 결함을 해결하며 발전했습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 문제점 (The Problem)

사용자가 "비행편을 변경하고 싶어요"라고 말합니다. 시스템은 사용자가 무엇을 원하는지, 어떤 정보가 누락되었는지, 어떻게 그 정보를 얻을지, 그리고 어떻게 동작을 완료할지를 파악해야 합니다. 그 후 사용자가 "잠시만요, 대신 취소하면 어떻게 되나요?"라고 말하면, 시스템은 문맥(context)을 기억하고, 작업을 전환하며, 상태(state)를 유지해야 합니다.

대화는 머신러닝(ML) 시스템에게 어려운 과제입니다. 입력은 개방형(open-ended)이며, 출력은 여러 턴(turn)에 걸쳐 일관성을 유지해야 합니다. 또한 시스템은 세상에 직접적인 동작을 수행해야 할 수도 있습니다(비행편 변경, 카드 결제 등). 모든 잘못된 단계는 사용자에게 그대로 드러납니다.

챗봇 아키텍처는 네 가지 패러다임을 거쳐 순환해 왔으며, 각 패러다임은 이전 패러다임이 너무 눈에 띄게 실패했기 때문에 도입되었습니다. 이 레슨에서는 이를 순서대로 살펴봅니다. 2026년의 프로덕션 환경은 마지막 두 패러다임이 결합된 하이브리드 형태입니다.

## 개념 (The Concept)

![챗봇의 진화: 규칙 기반(rule-based) → 검색(retrieval) → 신경망(neural) → 에이전트(agent)](../assets/chatbot.svg)

### 스크립트 기반의 반세기, 1950-2001 (The scripted half-century, 1950-2001)

첫 번째 패러다임은 5년도 채 지속되지 않았습니다. 하지만 50년 동안 이어졌습니다. 그 궤적을 아는 것이 중요한 이유는 그 안의 모든 시스템이 동일한 메커니즘을 사용하기 때문입니다. 즉, 입력을 매칭하고, 정해진 응답을 내보내며, 약간의 상태(state)를 업데이트하는 방식입니다. 50년 동안 이 기계에 규칙을 추가해 왔지만, 결코 일반적인 사례(general case)를 만들어내지는 못했습니다. 이러한 한계 때문에 두 번째부터 네 번째 패러다임이 존재하게 되었습니다.

**1950년.** 튜링(Turing)은 "기계가 생각할 수 있는가?"라는 질문을 우회하여 운영적인 대체안을 제시했습니다. 만약 질문자가 텔레타이프를 통해 기계와 사람을 구분할 수 없다면, 철학적 질문은 무의미해진다는 것입니다. 대화는 이 분야의 이름이 정해지기도 전에 이 분야의 벤치마크가 되었습니다.

**1956년.** 이름이 등장합니다. 다트머스(Dartmouth)의 여름 워크숍에서 "지능의 모든 특징은 원칙적으로 매우 정밀하게 묘사될 수 있어 기계가 이를 시뮬레이션하도록 만들 수 있다"는 가설 하에 "인공지능(artificial intelligence)"이라는 용어가 만들어졌습니다. 이 제안서는 상당한 진전을 위해 두 달의 예산을 책정했습니다.

**1966년.** ELIZA가 1단계에서 구축한 반사 기법(reflection trick)을 선보입니다. 분해 규칙(decomposition rules)이 입력에서 파편을 추출하면, 재조립 규칙(reassembly rules)이 이를 질문 형태로 다시 되돌려줍니다. 총 200여 개의 패턴, 상태(state) 없음, 이해도 없음 — 그럼에도 사용자들은 ELIZA에 속마음을 털어놓았습니다. 바이젠바움(Weizenbaum)은 이 시스템이 얼마나 적은 메커니즘만으로 작동하는지를 보며 남은 생애 동안 경계심을 늦추지 않았습니다.

**1972년.** 편집증(paranoia)을 모델링하기 위해 스탠퍼드에서 구축된 PARRY는 ELIZA에 부족했던 요소인 내부 상태(internal state)를 추가했습니다. 공포, 분노, 불신을 나타내는 수치형 변수가 매 턴마다 업데이트되며 다음에 실행될 스크립트를 결정하므로, 동일한 입력이라도 지금까지의 대화 내용에 따라 다른 응답을 생성합니다. 눈을 가린 채 진행된 전사 기록 테스트에서 정신과 의사들은 PARRY를 인간 환자와 구분하는 확률이 우연의 일치 수준에 불과했습니다. 이는 페르소나 컨디셔닝(persona conditioning)의 직접적인 조상 격으로, 세 개의 부동 소수점(float) 값으로 구현된 시스템 프롬프트와 같습니다. 같은 해, 두 봇은 ARPANET을 통해 서로 마주하게 되었습니다. 상담가 스크립트가 편집증 상태 머신을 인터뷰하는, 네트워크상에서 이루어진 최초의 봇 간 대화였습니다.

**1995년.** ALICE는 패턴-템플릿 쌍을 위한 XML 방언인 AIML을 사용하여 ELIZA의 방식을 확장했습니다. 약 40,000개의 수동 작성 카테고리를 보유했으며, 로브너 상(Loebner Prize)을 세 번 수상했습니다. 이는 규칙 기반 시스템의 확장 법칙(scaling law)을 증명했습니다. 규칙이 많아질수록 커버리지는 늘어나지만, 일반성은 결코 늘어나지 않는다는 것입니다. 모든 규칙은 누군가가 유지 관리해야 하는 부채(liability)입니다.

**2001년.** SmarterChild는 이 방식을 3,000만 명의 인스턴트 메신저 사용자에게 선보였으며, 템플릿에 삽입된 날씨, 주식, 영화 시간 등의 백엔드 조회 기능을 추가했습니다. 자세히 들여다보면 이는 2001년식 복장을 입은 도구 호출(tool calling)과 같습니다. 의도(intent)를 파싱하고, 서비스를 호출하며, 결과를 응답에 렌더링하는 방식입니다.

50년 동안 하나의 메커니즘과 계속해서 늘어나는 규칙의 수. 이 패러다임이 끝난 것은 누군가 그것이 틀렸음을 증명해서가 아니라, 수동으로 작성된 상태 머신의 유지 관리 비용은 커버리지에 따라 선형적으로 증가하는 반면, 사용자의 기대치는 지난주에 본 것에 따라 급격히 성장하기 때문입니다.

```figure
chatbot-lineage
```

**규칙 기반 (Rule-based; ELIZA, AIML, DialogFlow).** 수동으로 작성된 패턴이 사용자 입력을 매칭하여 응답을 생성합니다. 의도 분류기(Intent classifier)가 미리 정의된 흐름으로 경로를 지정합니다. 슬롯 필링(Slot-filling) 상태 머신이 필요한 정보를 수집합니다. 설계된 좁은 범위 내에서는 훌륭하게 작동하지만, 범위를 벗어나면 즉시 실패합니다. 환각(hallucination)이 허용되지 않는 안전 필수 도메인(은행 인증, 항공권 예약 등)에서는 여전히 사용됩니다.

**검색 기반 (Retrieval-based).** FAQ 스타일의 시스템입니다. (발화, 응답)의 모든 쌍을 인코딩합니다. 실행 시점에 사용자의 메시지를 인코딩하고 가장 유사한 저장된 응답을 검색합니다. Zendesk의 고전적인 "유사한 문서" 기능을 생각하면 됩니다. 규칙보다 패러프레이징(paraphrase)을 더 잘 처리합니다. 생성 과정이 없으므로 환각이 발생하지 않습니다.

**신경망 기반 (Neural; seq2seq).** 대화 로그로 학습된 인코더-디코더(Encoder-decoder) 모델입니다. 응답을 처음부터 생성합니다. 유창하지만 일반적인 출력("잘 모르겠습니다")이나 사실 관계 이탈(factual drift)이 발생하기 쉽습니다. 주제를 안정적으로 유지하지 못합니다. Google, Facebook, Microsoft가 2016-2019년 사이에 실망스러운 챗봇을 내놓았던 이유입니다.

**LLM 에이전트 (LLM agents).** 계획을 세우고, 도구를 호출하며, 결과를 검증하는 루프에 감싸인 언어 모델입니다. 긴 프롬프트를 가진 챗봇이 아닙니다. 에이전트 루프는 다음과 같습니다: 계획 $\rightarrow$ 도구 호출 $\rightarrow$ 결과 관찰 $\rightarrow$ 다음 단계 결정. 검색 우선 접지(Retrieval-first grounding, RAG)를 통해 환각을 방지합니다. 도구 호출을 통해 실제로 작업을 수행할 수 있습니다. 이것이 2026년의 아키텍처입니다.

이 네 가지 패러다임은 순차적인 교체 관계가 아닙니다. 2026년의 프로덕션 챗봇은 이 네 가지를 모두 거쳐 경로를 지정합니다. 인증 및 파괴적인 작업에는 규칙 기반을, FAQ에는 검색 기반을, 자연스러운 문구 생성에는 신경망 생성을, 모호하고 개방적인 질의에는 LLM 에이전트를 사용합니다.

## 구현해 보기 (Build It)

### 1단계: 규칙 기반 패턴 매칭 (Rule-based pattern matching)

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

20줄로 구현한 ELIZA입니다. "I feel sad" → "Why do you feel sad"와 같은 반영 기법(reflection trick)은 Weizenbaum(1966)의 전형적인 심리 치료사 데모입니다. 여전히 교육적 가치가 있습니다.

### 2단계: 검색 기반 (FAQ) (Retrieval-based)

이 예시 스니펫을 실행하려면 `pip install sentence-transformers`(torch 포함)가 필요합니다. 이 레슨의 실행 가능한 `code/main.py`는 대신 표준 라이브러리의 Jaccard 유사도를 사용하므로, 외부 의존성 없이 레슨을 진행할 수 있습니다.

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

임계값 기반 거부(Threshold-based refusal)가 핵심 설계 선택 사항입니다. 가장 유사한 매칭이 충분히 가깝지 않으면 `None`을 반환하여 시스템이 에스컬레이션(escalate)하도록 합니다.

### 3단계: 신경망 생성 (기준선) (Neural generation, baseline)

소형 지시어 튜닝된 인코더-디코더(FLAN-T5) 또는 미세 조정된 대화형 모델을 사용해 보세요. 2026년 기준으로 단독으로는 실무에 사용하기 어렵지만(모순, 대상 이탈 생성(off-target generation), 사실 관계 오류), 자연스러운 문구 생성을 위해 하이브리드 시스템 내부에 포함되어 배포됩니다. DialoGPT 스타일의 디코더 전용(decoder-only) 모델은 일관된 답변을 생성하기 위해 명시적인 턴 구분자(turn separators)와 EOS 처리가 필요합니다. 반면, FLAN-T5 text2text 파이프라인은 교육용 예제로 즉시 사용하기에 적합합니다.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 4단계: LLM 에이전트 루프 (LLM agent loop)

2026년형 프로덕션 형태:

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

세 가지 핵심 요소가 있습니다. 도구(Tools)는 LLM이 호출할 수 있는 실행 가능한 함수입니다. 루프는 LLM이 도구 호출 대신 최종 답변을 반환할 때 종료됩니다. 단계 예산(step budget)은 모호한 작업에서 무한 루프가 발생하는 것을 방지합니다.

실제 프로덕션 환경에서는 다음 요소들이 추가됩니다: 검색 우선 그라운딩(retrieval-first grounding, 각 LLM 호출 전에 관련 문서를 주입), 가드레일(guardrails, 확인 없이 파괴적인 동작을 수행하는 것을 거부), 관측 가능성(observability, 모든 단계를 로그로 기록), 그리고 평가(evaluations, 에이전트의 동작이 사양을 준수하는지 확인하는 자동화된 점검).

### 5단계: 하이브리드 라우팅 (Hybrid routing)

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

이 패턴은 파괴적인 작업에는 결정론적 규칙(deterministic rules)을, 정형화된 FAQ에는 검색(retrieval)을, 그 외의 모든 것에는 LLM 에이전트를 사용합니다. 이것이 2026년 고객 지원 시스템에 실제로 적용될 방식입니다.

## 활용하기 (Use It)

2026년 스택:

| 사용 사례 | 아키텍처 |
|---------|---------------|
| 예약, 결제, 인증 | 규칙 기반 상태 머신 + 슬롯 필링 |
| 고객 지원 FAQ | 큐레이션된 답변에 대한 검색 |
| 개방형 도움말 채팅 | RAG + 도구 호출을 사용하는 LLM 에이전트 |
| 내부 도구 / IDE 어시스턴트 | 도구 호출(검색, 읽기, 쓰기)을 사용하는 LLM 에이전트 |
| 컴패니언 / 캐릭터 챗봇 | 페르소나 시스템 프롬프트 및 지식 검색을 적용한 튜닝된 LLM |

프로덕션 환경에서는 항상 하이브리드 라우팅을 사용하세요. 단일 아키텍처가 모든 요청을 잘 처리할 수는 없습니다. 라우팅 레이어 자체는 일반적으로 작은 의도 분류기입니다.

## 여전히 발생하는 실패 모드 (Failure modes that still ship)

- **확신에 찬 허위 사실 생성(Confident fabrication).** LLM 에이전트가 수행하지 않은 작업을 완료했다고 주장합니다. 완화 방법: 결과를 검증하고, 도구 호출(tool calls)을 기록하며, 도구의 성공적인 반환 없이 LLM이 작업을 완료했다고 주장하게 두지 마세요.
- **프롬프트 인젝션(Prompt injection).** 사용자가 시스템 프롬프트를 무력화하는 텍스트를 삽입합니다. 2025년 OWASP LLM 애플리케이션 Top 10에서 LLM01 순위를 차지했습니다. 두 가지 유형이 있습니다: 직접 인젝션(채팅에 붙여넣기)과 간접 인젝션(에이전트가 읽는 문서, 이메일 또는 도구 출력에 숨겨짐)입니다.

  공격률은 시나리오에 따라 다릅니다. 일반적인 도구 사용 및 코딩 벤치마크에서 프런티어 모델(frontier models)의 측정된 성공률은 약 0.5~8.5% 범위입니다. 특정 고위험 설정(AI 코딩 에이전트에 대한 적응형 공격, 취약한 오케스트레이션)에서는 약 84%에 달하기도 합니다. 실제 운영 환경의 CVE 사례로는 EchoLeak(CVE-2025-32711, CVSS 9.3)이 있습니다. 이는 공격자가 제어하는 이메일에 의해 트리거되는 Microsoft 365 Copilot의 제로 클릭 데이터 유출 결함입니다.

  완화 방법: 루프 전반에 걸쳐 사용자 입력을 신뢰할 수 없는 것으로 취급하세요; 도구 호출 전에 데이터를 정제(sanitize)하세요; 도구 출력을 메인 프롬프트로부터 격리하세요; 에이전트가 먼저 계획을 세운 뒤, 실행하기 전에 해당 계획에 따라 각 작업을 검증하는 Plan-Verify-Execute (PVE) 패턴을 사용하세요(이는 도구 결과가 계획되지 않은 새로운 작업을 주입하는 것을 방지합니다); 파괴적인 작업에 대해서는 사용자의 확인을 요구하세요; 도구 범위에 최소 권한 원칙(least-privilege)을 적용하세요.

  아무리 많은 프롬프트 엔지니어링을 하더라도 이 위험을 완전히 제거할 수는 없습니다. 외부 런타임 방어 계층(LLM Guard, 허용 목록 검증, 의미론적 이상 탐지)이 필요합니다.
- **범위 이탈(Scope creep).** 도구 호출이 부수적으로 관련된 정보를 반환하여 에이전트가 작업에서 벗어납니다. 완화 방법: 도구 계약(tool contracts)을 좁게 정의하고, 시스템 프롬프트의 초점을 유지하며, 작업 이탈률(off-task rate)에 대한 평가를 추가하세요.
- **무한 루프(Infinite loops).** 에이전트가 동일한 도구를 계속 호출합니다. 완화 방법: 단계 예산(step budget) 설정, 도구 호출 중복 제거, "진전이 있는가"를 판단하는 LLM 판사(LLM judge) 활용.
- **컨텍스트 창 고갈(Context window exhaustion).** 긴 대화는 초기 대화 내용을 컨텍스트 밖으로 밀어냅니다. 완화 방법: 이전 대화 내용을 요약하거나, 유사도에 따라 관련 있는 과거 대화를 검색하거나, 롱 컨텍스트(long-context) 모델을 사용하세요.

## Ship It

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
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 연습 문제 (Exercises)

1. **쉬움.** 커피숍 주문 봇을 위해 10가지 패턴을 사용하여 위에서 다룬 규칙 기반 응답(rule-based respond)을 구현해 보세요. 엣지 케이스(edge cases)를 테스트해 보세요: 중복 주문, 수정, 취소, 불분명한 의도.
2. **중간.** 하이브리드 FAQ + LLM 폴백(fallback) 시스템을 구축해 보세요. SaaS 제품을 위한 50개의 정형화된(canned) FAQ 항목을 만들고, 문서 사이트의 검색(retrieval)을 활용한 LLM 폴백을 구현하세요. 100개의 실제 고객 지원 질문을 대상으로 거절률(refusal rate)과 정확도를 측정해 보세요.
3. **어려움.** 세 가지 도구(`search`, `read-user-data`, `send-email`)를 사용하여 위에서 다룬 에이전트 루프(agent loop)를 구현해 보세요. 프롬프트 인젝션(prompt injection) 시도를 포함한 50개의 테스트 시나리오로 평가를 수행하세요. 업무 이탈률(off-task rate), 작업 실패율(failed task rate), 그리고 인젝션 성공 여부를 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Intent | 사용자가 원하는 것 | 범주형 레이블(`book_flight`, `reset_password`). 핸들러로 라우팅됩니다. |
| Slot | 정보의 조각 | 봇이 필요한 파라미터(`date`, `destination`). 슬롯 필링(Slot filling)은 질문을 이어가는 과정입니다. |
| RAG | 검색 및 생성 | 관련 문서를 검색한 후, LLM의 응답에 근거를 제공합니다. |
| Tool call | 함수 호출 | LLM이 이름과 인자(`args`)가 포함된 구조화된 호출을 생성합니다. 런타임에서 이를 실행하고 결과를 반환합니다. |
| Agent loop | 계획, 실행, 검증 | 작업이 완료될 때까지 LLM 호출과 도구 호출을 교차하며 실행하는 컨트롤러입니다. |
| Prompt injection | 사용자의 프롬프트 공격 | 시스템 프롬프트를 무력화하려는 악의적인 입력입니다. |

## 추가 읽을거리 (Further Reading)

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — 대화를 이 분야의 벤치마크로 만든 논문입니다.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 초기 규칙 기반 챗봇에 관한 논문입니다.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — 최초의 상태 유지(stateful) 챗봇인 PARRY의 정동 변수(affect-variable) 아키텍처를 다룹니다.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — LLM 에이전트가 주류가 되기 직전, 구글이 발표한 후기 신경망 챗봇 논문입니다.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 에이전트 루프(agent loop) 패턴의 명칭을 정립한 논문입니다.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2026년에도 여전히 유효한 2024년 기준 프로덕션 가이드입니다.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 프롬프트 인젝션(prompt-injection)에 관한 논문입니다.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 프롬프트 인젝션을 최고의 보안 위협으로 만든 순위입니다.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Execute 및 사용자 확인(user-confirmation) 흐름을 포함한 실무적인 오케스트레이션 계층 방어 전략을 다룹니다.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 간접 프롬프트 인젝션으로 인한 전형적인 제로 클릭(zero-click) 데이터 유출 CVE 사례입니다. 쓰기 권한을 가진 에이전트에게 왜 런타임 방어가 필요한지 보여주는 참조 사례입니다.
