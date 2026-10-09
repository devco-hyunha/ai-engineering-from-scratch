# FIPA-ACL 및 화행의 유산

> MCP, A2A 이전에는 FIPA-ACL이 있었습니다. 2000년 IEEE 지능형 물리 에이전트 재단은 20개의 수행어(performatives), 2개의 콘텐츠 언어, 그리고 계약 네트워크(contract net), 구독/통지(subscribe/notify), 요청 시(request-when) 등 일련의 상호작용 프로토콜을 포함하는 에이전트 통신 언어를 비준했습니다. 웹에 대한 온톨로지 오버헤드가 너무 무거워 산업계에서는 사라졌지만, LLM의 부흥과 함께 다중 에이전트 시스템은 형식적 의미론(formal semantics) 없이 동일한 아이디어를 조용히 재구현하고 있습니다. JSON 계약이 수행어를 대체하고, 자연어가 온톨로지를 대체합니다. 이 강의를 통해 FIPA-ACL을 진지하게 읽어보시면, 2026년 프로토콜 결정 중 어떤 것이 재발명이고 어떤 것이 새로운지, 그리고 현재 흐름이 2000년대에 이미 해결된 문제들을 어떻게 재발견할지 파악할 수 있습니다.

**유형:** Learn
**언어:** Python (stdlib)
**선수 요건:** 16단계 · 01강 (왜 다중 에이전트인가)
**시간:** 약 60분

## 문제점

2026년 에이전트 프로토콜 생태계는 활발합니다. 도구를 위한 MCP, 에이전트를 위한 A2A, 기업 감사용 ACP, 분산 신뢰용 ANP, 자연어 콘텐츠용 NLIP, 그리고 CA-MCP와 수십 개의 연구 제안이 있습니다. 각 사양은 스스로를 기초적이라고 선언합니다.

솔직히 말하면, 그 대부분은 20년 전의 매우 구체적인 의사결정 트리를 재발견하고 있습니다. Austin(1962)과 Searle(1969)의 화행 이론은 "발화는 행동이다"라는 개념을 제공했습니다. KQML(1993)은 이를 와이어 프로토콜로 전환했습니다. FIPA-ACL(2000년 비준)은 참조 표준화를 산출했습니다. 20개의 수행어, 콘텐츠 언어 SL0/SL1, 계약 네트워크 및 구독-통지를 위한 상호작용 프로토콜이 포함되었습니다. JADE와 JACK은 Java 참조 플랫폼이었습니다. 온톨로지 오버헤드가 너무 무거워 웹이 승리하면서 2010년경 이 노력은 사라졌습니다.

MCP의 `tools/call`, A2A의 작업 수명주기, CA-MCP의 공유 컨텍스트 저장소를 살펴보면, FIPA 결정 사항의 더 부드럽고 JSON 네이티브한 재구성임을 알 수 있습니다. 유산을 알면 두 가지를 알 수 있습니다. 새로운 "혁신" 중 실제로는 재발명인 것과, 새로운 사양이 재발견할 오래된 실패 모드입니다.

## 개념

### 화행, 한 단락으로 요약

오스틴은 어떤 문장들은 세계를 묘사하지 않고, 세계를 바꾼다는 것을 발견했습니다. "저는 약속합니다." "저는 요청합니다." "저는 선언합니다." 그는 이것을 수행적 발화(performative utterances)라고 불렀습니다. 설(Searle)은 이를 다섯 범주로 공식화했습니다: 단정(assertive), 지시(directive), 약속(commissive), 표현(expressive), 선언(declarative). KQML (Finin et al., 1993)은 이를 소프트웨어 에이전트(Agent)에 대해 실용적으로 만들었습니다: 메시지는 수행적 행위(the action)와 내용(what the action is about)으로 구성됩니다. FIPA-ACL은 KQML의 공백을 정리하고 약 20개의 수행적 행위(performatives)를 표준화했습니다.

### FIPA의 20개 수행적 행위 (부분 목록)

| 수행적 행위 | 의도 |
|---|---|
| `inform` | "P가 참이라고 알려드립니다" |
| `request` | "X를 해 주시기를 요청합니다" |
| `query-if` | "P가 참인가요?" |
| `query-ref` | "X의 값은 무엇인가요?" |
| `propose` | "X를 하자고 제안합니다" |
| `accept-proposal` | "제안을 수락합니다" |
| `reject-proposal` | "제안을 거절합니다" |
| `agree` | "X를 하기로 동의합니다" |
| `refuse` | "X를 하는 것을 거부합니다" |
| `confirm` | "P가 참임을 확인합니다" |
| `disconfirm` | "P를 부인합니다" |
| `not-understood` | "메시지가 파싱되지 않았습니다" |
| `cfp` | "X에 대한 제안 요청" |
| `subscribe` | "X가 변경되면 알려주세요" |
| `cancel` | "진행 중인 X를 취소합니다" |
| `failure` | "X를 시도했으나 실패했습니다" |

전체 목록은 `fipa00037.pdf` (FIPA ACL 메시지 구조)에 있습니다. 이 목록을 암기하는 것이 목적이 아닙니다. 핵심은 이 모든 항목이 LLM 프로토콜이 결국 다시 추가하게 되는 원시적 요소(primitive)에 대응한다는 점입니다.

### 표준 FIPA-ACL 메시지

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

7개의 필드가 프로토콜 엔벨로프를 담당하며, 하나의 필드(`content`)가 페이로드를 담당합니다. 나머지 필드들은 JSON 프로토콜에 재시도(retry), 스레딩(threading), 온톨로지(ontology)를 붙일 때마다 매번 다시 발명하는 것들입니다.

### 두 개의 레거시 플랫폼

**JADE** (Java Agent DEvelopment framework, 1999–2020년대)는 가장 많이 사용된 FIPA 호환 런타임이었습니다. 에이전트(Agent)는 기본 클래스를 확장하고, ACL 메시지를 교환하며, 컨테이너 내부에서 실행되고, "행동(behaviors)"을 사용하여 조정했습니다. 상호작용 프로토콜 라이브러리에는 contract-net, subscribe-notify, request-when, propose-accept가 포함되어 있었습니다.

**JACK** (Agent Oriented Software, 상용)은 FIPA 메시지 위에 BDI (Belief-Desire-Intention) 추론을 강조했습니다. 더 형식적이지만, 채택률은 낮았습니다.

웹 스택이 멀티 에이전트(Multi-Agent) 사용 사례를 흡수하면서 두 프레임워크 모두 쇠퇴했습니다. MCP와 A2A는 2026년의 런타임 "컨테이너"입니다.

### FIPA가 쇠퇴한 이유

- **온톨로지 오버헤드.** FIPA는 `content`을 파싱하기 위해 공유 온톨로지를 요구했습니다. 온톨로지에 합의하는 것은 수년간의 표준화 과정입니다. 웹은 HTTP + JSON을 사용했습니다.
- **아무도 사용하지 않는 형식적 의미론.** SL (Semantic Language)은 엄격한 진리 조건을 제공했지만, 대부분의 프로덕션 시스템은 자유 형식 콘텐츠를 사용하며 형식주의를 무시했습니다.
- **도구 잠금(Lock-in).** JADE는 Java 전용이었고, JACK는 상용이었습니다. 폴리글롯(Polyglot) 팀은 두 프레임워크를 우회했습니다.
- **인터넷이 스택을 승리했습니다.** REST, 그 다음 JSON-RPC, 그 다음 gRPC가 ACL의 전송을 대체했습니다.

### LLM의 부활은 FIPA-lite입니다

FIPA `request`와 MCP `tools/call`를 비교해 보세요:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

동일한 엔벨로프, 다른 구문. 둘 다 다음을 포함합니다: 발신자, 수신자, 의도, 페이로드, 상관 ID. 둘 중 하나가 다른 것에 대한 혁명은 아닙니다 — 동일한 설계에 대한 다른 트레이드오프일 뿐입니다.

Liu et al.의 2025년 설문조사("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279)는 이 계보를 명확히 합니다: MCP는 도구 사용 발화 행위에, A2A는 에이전트 간 발화 행위에, ACP는 감사 추적 발화 행위에, ANP는 분산 신원 확장 기능에 대응합니다. 새로운 사양은 JSON 구문과 느슨한 의미론을 가진 ACL의 후손입니다.

### 명확하게 진술된 트레이드오프

**FIPA가 제공한 것과 현대 사양이 버린 것:**

- 형식적 의미론 — `inform`이 발신자가 내용을 믿음을 함을 증명할 수 있습니다.
- 퍼포마티브의 표준 카탈로그 — "`cancel`을 가져야 하는가?"를 다시 논쟁할 필요가 없습니다.
- 수십 년간의 상호작용 프로토콜 패턴 — 계약 네트워크(contract-net), 구독-알림(subscribe-notify), 제안-수락(propose-accept) — 잘 알려진 정확성 속성을 갖추고 있습니다.

**현대 사양이 제공하는 것과 FIPA가 제공하지 못한 것:**

- 모든 현대 도구와 호환되는 JSON 네이티브 페이로드입니다.
- 수동으로 코딩된 온톨로지 없이 LLM이 해석할 수 있는 자연어 콘텐츠입니다.
- 웹 스택 기반 전송(HTTP, SSE, WebSocket)입니다.
- 실시간 MCP `server/discover` 및 A2A 에이전트 카드를 통한 기능 발견입니다.

구현이 쉬운 더 느슨한 의도(semantics)입니다. 이것이 정확히 트레이드오프입니다.

### 이식할 가치가 있는 상호작용 프로토콜

FIPA는 약 15개의 상호작용 프로토콜을 출시했습니다. LLM 멀티 에이전트 시스템으로 이어갈 가치가 있는 것은 세 가지입니다:

1. **계약 네트워크 프로토콜(CNP).** 관리자가 `cfp` (제안 요청)를 발행하고, 입찰자가 `propose`로 응답하며, 관리자가 수락/거부합니다. 이는 표준적인 작업 시장 패턴입니다(16단계 · 16 협상).
2. **구독/알림.** 구독자가 `subscribe`를 보내면, 게시자는 주제(topic)가 변경될 때마다 `inform`를 보냅니다. 이는 2026년의 모든 이벤트 버스입니다.
3. **요청-조건.** "조건 Y가 성립할 때 X를 수행." 사전 조건을 가진 지연 실행입니다. 2026년 아날로그는 내구성 있는 워크플로우 엔진의 지연 작업입니다(16단계 · 22 프로덕션 확장).

각 프로토콜은 현대 메시지 큐, HTTP + 폴링, 또는 SSE 스트리밍에 깔끔하게 매핑됩니다.

### 온톨로지를 제거하면 무엇이 깨지는가

공유 온톨로지가 없으면, 에이전트는 자연어 콘텐츠에서 의미를 추론합니다. 문서화된 2026년 실패 모드인 **의미적 드리프트(semantic drift)**는 두 에이전트가 같은 단어(`"customer"`)를 미묘하게 다른 개념에 사용하는 경우입니다. 수신자 에이전트가 잘못된 해석에 따라 행동하며, 스키마 검증기가 이를 잡아내지 못합니다. FIPA의 온톨로지 요구사항은 파싱 시점에 메시지를 거부했을 것입니다.

완전한 온톨로지로 가지 않으면서 취할 수 있는 완화책:

- `content`에 대한 JSON Schema — 전송(wire) 단계에서 구조적 오류를 거부합니다.
- 타입 지정된 아티팩트(A2A) — 잘못된 모달리티를 거부합니다.
- 봉투(envelope)에 명시적인 수행 행위(performative) 포함 — 콘텐츠가 자연어일지라도 의도를 모호하지 않게 만듭니다.

### 화행(speech-act) 전통에 매핑된 2026년 사양

| 현대 사양 | FIPA 유사체 | 유지하는 것 | 버리는 것 |
|---|---|---|---|
| MCP `tools/call` | `request` | 명시적 의도, 상관 ID | 형식적 의미론, 온톨로지 |
| MCP `resources/read` | `query-ref` | 명시적 의도, 상관 ID | 형식적 의미론 |
| A2A 작업 수명주기 | 계약망 + 요청 시 | 비동기 수명주기, 상태 전환 | 형식적 완전성 보장 |
| A2A 스트리밍 이벤트 | 구독/통지 | 비동기 푸시 | 타입 지정된 술어 구독 |
| CA-MCP 공유 컨텍스트 | 블랙보드 (Hayes-Roth 1985) | 다중 작성자 공유 메모리 | 논리적 일관성 모델 |
| NLIP | 자연어 콘텐츠 | LLM 네이티브 | 스키마 |

표 위쪽에서 아래쪽으로 읽으면 패턴은 다음과 같습니다: 구조적 원시 요소를 유지하고, 형식주의를 버리며, 모호성은 LLM이 해결하도록 맡깁니다.

```figure
sw-contract-net
```

## 구현하기

`code/main.py`은 순수 표준 라이브러리 기반의 FIPA-ACL 번역기를 구현합니다. 이는 표준 ACL 엔벨로프를 인코딩 및 디코딩하며, 모든 MCP / A2A 메시지 형태가 동일한 7개 필드로 축소되는 방식을 보여줍니다. 데모는:

- 5개의 MCP 스타일 및 A2A 스타일 메시지를 FIPA-ACL로 인코딩합니다.
- FIPA-ACL을 현대적 등가물로 디코딩합니다.
- `cfp`, `propose`, `accept-proposal`, `reject-proposal`을 사용하여 한 명의 관리자와 세 명의 입찰자 간의 장난감 계약망 협상을 실행합니다.

실행:

```
python3 code/main.py
```

출력은 각 현대적 메시지를 2026년 JSON 형태와 FIPA-ACL 형태로 나란히 표시하는 추적 기록이며, 그 다음 계약망 입찰의 왕복이 진행됩니다. 동일한 프로토콜 원시 요소는 왕복을 통해 유지되며, 구문만 다릅니다.

## 사용하기

`outputs/skill-fipa-mapper.md`은 모든 에이전트 프로토콜 사양을 읽고 FIPA-ACL 매핑을 생성하는 스킬입니다. 새로운 프로토콜을 채택하기 전에 "이것이 진정으로 새로운 것인가요, 아니면 JSON 구문으로 된 `inform`인가요?"라는 질문에 답하기 위해 사용해 보세요.

## 출시하기

FIPA-ACL을 가져오지 마세요. 대신 그 체크리스트를 가져오세요:

- 각 메시지의 의도 원시 요소(퍼포마티브)는 무엇인가요?
- 요청-응답 및 취소에 대한 상관 ID가 있나요?
- 명시적인 콘텐츠 언어(JSON-RPC, 평문 텍스트, 구조화된 타입 지정 아티팩트)가 있나요?
- 인터랙션 프로토콜이 일급 객체로 취급되나요, 아니면 계약망(contract-net)을 처음부터 다시 구현하고 있나요?
- 두 에이전트(Agent)가 콘텐츠 의미에 대해 의견이 불일치할 때 (의미 드리프트(Semantic Drift)) 어떻게 처리하나요?

새로운 프로토콜을 프로덕션에 출시하기 전에 이 다섯 가지 질문을 문서화해 보세요.

## 연습 문제

1. `code/main.py`을 실행하세요. 왕복 인코딩을 관찰하세요. `tools/call`, `resources/read` 및 A2A 작업 생성에 해당하는 FIPA 퍼포마티브(Performative)가 무엇인지 식별해 보세요.
2. 계약망(contract-net) 데모를 확장하여, 매니저가 입찰 중도에 작업을 철회할 수 있도록 하는 `cancel` 퍼포마티브(Performative)를 추가하세요. `cancel`은 단순한 재시도(retry)로는 해결할 수 없는 어떤 실패 케이스를 해결하나요?
3. FIPA ACL 메시지 구조(http://www.fipa.org/specs/fipa00037/)의 4.1–4.3 섹션을 읽어 보세요. 이 강의에서 다루지 않은 퍼포마티브(Performative) 하나를 선택하여, 그 현대적인 JSON-RPC 유사 기능을 설명해 보세요.
4. Liu et al., arXiv:2505.02279를 읽어 보세요. MCP, A2A, ACP, ANP 각각에 대해, 유지하는 FIPA 퍼포마티브(Performative) 계열과 폐기하는 계열을 나열해 보세요.
5. 자체 시스템에서 `request` 퍼포마티브(Performative)의 `content` 필드에 대한 최소한의 JSON-Schema를 설계해 보세요. 그 스키마가 순수 자연어로는 제공하지 못하는 것은 무엇이며, 비용은 얼마인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 발화 행위(Speech act) | "무언가를 수행하는 발화" | Austin/Searle: 발화를 행동으로 봄. ACL의 이론적 부모. |
| FIPA | "그 오래된 XML 것" | 지능형 물리 에이전트 IEEE 재단. 2000년에 ACL을 표준화. |
| ACL | "에이전트 통신 언어" | FIPA의 인velope 형식: 퍼포마티브(Performative) + 콘텐츠 + 메타데이터. |
| 퍼포마티브(Performative) | "동사" | 메시지의 의도 클래스: `inform`, `request`, `propose`, `cfp` 등. |
| KQML | "FIPA의 전신" | 지식 질의 및 조작 언어(Knowledge Query and Manipulation Language) (1993). 더 단순하고 좁음. |
| 온톨로지(Ontology) | "공유 어휘" | 콘텐츠 언어가 다루는 개념에 대한 형식적 정의. |
| SL0 / SL1 | "FIPA 콘텐츠 언어" | 시맨틱 언어(Semantic Language) 레벨 0 및 1 — 형식적 콘텐츠 언어 계열. |
| 계약망(Contract Net) | "작업 시장" | 매니저가 cfp를 발행; 입찰자가 제안; 매니저가 수락. 표준적인 인터랙션 프로토콜. |
| 상호작용 프로토콜 | "메시지 패턴" | request-when, subscribe-notify 등 알려진 정확성을 가진 퍼포마티브(performative)의 시퀀스 |

## 추가 읽기

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 현대 사양을 FIPA 유산과 연결하는 2025년 표준 조사
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000년 비준된 봉투(envelope) 형식
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 전체 퍼포마티브(performative) 카탈로그
- [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — `request`/`query-ref`의 현재 상태 비저장 도구 사용에 해당하는 것
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net 및 subscribe-notify의 현대적 에이전트 피어(peer)에 해당하는 것
