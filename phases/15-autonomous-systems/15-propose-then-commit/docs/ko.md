# 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL)): 제안 후 커밋

> 2026년 HITL에 대한 합의는 구체적입니다. "에이전트가 묻고, 사용자가 승인 버튼을 클릭하는" 방식이 아닙니다. 제안 후 커밋(propose-then-commit) 방식입니다. 제안된 행동은 멱등성 키(Idempotency)와 함께 내구성 있는 스토어(Durable Store)에 저장되며, 의도, 데이터 계보(Data Lineage), 접근 권한, 영향 범위, 롤백(Rollback) 계획과 함께 리뷰어에게 제시됩니다. 긍정적 확인이 이루어진 후에만 커밋되며, 실행 후 실제 부작용이 발생했는지 확인합니다. LangGraph의 `interrupt()` 및 PostgreSQL 체크포인팅(Checkpointing), Microsoft Agent Framework의 `RequestInfoEvent`, Cloudflare의 `waitForApproval()`는 모두 동일한 형태를 구현합니다. 표준적인 실패 모드는 rubber-stamp approval(무비자 승인)입니다. "승인하시겠습니까?"를 검토 없이 클릭하는 경우입니다. 문서화된 완화책은 명시적인 체크리스트를 활용한 challenge-and-response(요청 및 응답) 방식입니다.

**유형:** Learn
**언어:** Python (stdlib, 멱등성 키를 활용한 제안 후 커밋 상태 머신)
**선수 요건:** 15단계 · 12강 (내구성 있는 실행(Durable Execution)), 15단계 · 14강 (트립와이어(Tripwires))
**시간:** 약 60분

## 문제점

에이전트가 행동을 취합니다. 사용자는 승인 여부를 결정해야 합니다. 결정이 즉각적이라면, 그것은 아마도 리뷰가 아닐 것입니다. 결정이 구조화되어 있다면, 느리지만 신뢰할 수 있습니다. 엔지니어링의 질문은 구조화된 리뷰가 가장 저항이 적은 경로가 되도록 하는 방법입니다.

2023년대의 HITL 패턴은 동기적 프롬프트였습니다. "에이전트가 X에게 Y 내용으로 이메일을 보내려 합니다. 승인하시겠습니까?" 사용자가 승인 버튼을 클릭합니다. 모든 사람은 시스템이 안전하다고 느낍니다. 실제로 이 표면은 rubber-stamp approval(무비자 승인)가 만연합니다. 사용자는 빠르게 승인하고, 승인은 예측력이 낮으며, 에이전트가 잘못될 때 감사 로그(Audit Log)는 사용자가 기억할 수 없는 긴 승인 기록을 보여줍니다.

2026년 패턴인 제안 후 커밋(propose-then-commit)은 HITL을 내구성 있는 기반(Durable Substrate)으로 옮기고, 구조화된 메타데이터를 첨부하며, 긍정적 커밋을 요구합니다. 모든 관리형 에이전트 SDK는 버전을 제공합니다. LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. API 이름은 다르지만, 형태는 동일합니다.

## 개념

### 제안 후 커밋(propose-then-commit) 상태 머신

1. **제안.** 에이전트가 제안된 행동을 생성합니다. 내구성 있는 스토어(PostgreSQL, Redis, Durable Object)에 저장됩니다. 포함 내용:
   - 의도 (에이전트가 이 작업을 수행하는 이유)
   - 데이터 계보(Data Lineage) (이 제안이 어떤 출처에서 유래했는지)
   - 접촉한 권한 (어떤 범위 / 파일 / 엔드포인트)
   - 폭발 반경 (최악의 경우)
   - 롤백(Rollback) 계획 (커밋된 경우, 어떻게 되돌리는지)
   - 멱등성 키(Idempotency Key) (제안마다 고유; 재제출 시 동일한 레코드를 반환)
2. **표면.** 리뷰어는 모든 메타데이터가 포함된 제안을 봅니다. 리뷰어는 사람입니다 (에이전트(Agent)가 스스로를 리뷰하는 것이 아닙니다).
3. **커밋.** 긍정적 승인. 작업이 실행됩니다.
4. **검증.** 실행 후, 사이드 이펙트(Side Effect)가 다시 읽혀지고 확인됩니다. 검증 단계가 실패하면, 시스템은 알려진 불량 상태가 되며 알람이 작동합니다.

### 멱등성 키(Idempotency Key)

멱등성 키가 없으면, 일시적 실패 후 재시도가 승인된 작업을 이중 실행할 수 있습니다. 구체적인 예: 사용자가 "A에서 B로 $100 이체"를 승인합니다. 네트워크가 불안정합니다. 워크플로우가 재시도합니다. 사용자는 한 번 승인했지만 이체가 두 번 실행됩니다. 멱등성 키는 승인을 단일하고 고유한 사이드 이펙트(Side Effect)에 연결합니다; 두 번째 실행은 노오프(No-op)입니다.

이것은 Stripe와 AWS API가 사용하는 동일한 멱등성 패턴입니다. 에이전트(Agent) 승인에 이를 재사용하는 것은 Microsoft Agent Framework 문서에 명시되어 있습니다.

### 내구성: 왜 승인(Durable Execution)이 프로세스보다 오래 지속되는가

승인 대기실은 에이전트(Agent)가 소유하지 않는 상태의 일부입니다. 워크플로우가 일시 중지됩니다 (12강). 승인이 도착하면, 워크플로우가 정확히 그 지점에서 재개됩니다. LangGraph가 `interrupt()`를 PostgreSQL 체크포인팅(Checkpointing)과 짝을 이루는 이유이며, 메모리 내 상태만으로는 충분하지 않습니다 — 이틀 후의 승인도 워크플로우가 온전한 상태로 남아 있도록 합니다.

### 무비자 승인(Rubber-stamp approvals)과 도전-응답(challenge-and-response) 완화 전략

HITL (인간 개입 루프)(Human-in-the-Loop (HITL))의 기본 UI ("승인" / "거부" 버튼)는 진정한 리뷰 없이 빠른 승인을 생성합니다. 문서화된 완화 전략: 승인 버튼이 활성화되기 전에 특정 질문에 긍정적 답변을 요구하는 도전-응답 체크리스트입니다. 구체적인 형태:

- "이 자원이 무엇을 접촉하는지 이해하십니까? [ ]"
- "폭발 반경이 허용 가능한지 검증하셨습니까? [ ]"
- "이 실패할 경우 롤백(Rollback) 계획이 있습니까? [ ]"

그 자체가 목적인 관료주의가 아닙니다 — 강제 기능입니다. 체크박스를 채울 수 없는 리뷰어는 명확한 설명을 요청(상향식 에스컬레이션)하거나 거절(안전한 기본값)합니다. Anthropic의 에이전트 안전성 연구는 도장 찍기식 승인 패턴에 대한 완화책으로 체크리스트 기반 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL))를 명시적으로 인용합니다.

### 중대한 조치의 기준

모든 작업이 제안 후 커밋(propose-then-commit)을 필요로 하지는 않습니다. 2026년 가이드라인:

- **중대한 조치** (항상 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL)) 필요): 되돌릴 수 없는 쓰기, 금융 거래, 외부 통신, 프로덕션 데이터베이스 변경, 파괴적인 파일 시스템 작업.
- **되돌릴 수 있는 조치** (때때로 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL)) 필요): 로컬 파일 편집, 스테이징 환경 변경, 명확한 롤백(Rollback)이 가능한 되돌릴 수 있는 쓰기.
- **읽기 및 검사** (인간 개입 루프 (HITL)(Human-in-the-Loop (HITL)) 불필요): 파일 읽기, 리소스 나열, 읽기 전용 API 호출.

### 작업 후 검증

"커밋이 실행되었다"는 "사이드 이펙트(side effect)가 발생했다"와 같은 뜻이 아닙니다. 네트워크 분할 및 경쟁 조건(race condition)은 백엔드가 실제로 데이터를 저장하지 않았는데도 성공했다고 생각하는 워크플로우를 만들 수 있습니다. 검증 단계는 커밋 후 대상 리소스를 다시 읽어 확인합니다. 이는 `RETURNING` 절이 있는 데이터베이스 트랜잭션이나 `PutObject` 이후의 AWS `GetObject`와 동일한 패턴입니다.

### EU AI Act 제14조

EU AI Act 제14조는 EU 내 고위험 AI 시스템에 대해 효과적인 인간 감독을 의무화합니다. "효과적인" 것은 장식적인 것이 아닙니다. 규제 용어는 도장 찍기식 패턴을 구체적으로 배제합니다. 도전 및 응답(challenge-and-response)이 포함된 제안 후 커밋(propose-then-commit)은 Microsoft Agent Governance Toolkit 컴플라이언스 문서에서 제14조 심사를 통과하는 형태입니다.

```figure
mx-propose-then-commit
```

## 사용하기

`code/main.py`은 stdlib Python에서 제안 후 커밋(propose-then-commit) 상태 머신을 구현합니다. 내구성 있는 스토어는 JSON 파일입니다. 멱등성 키는 (thread_id, action_signature)의 해시입니다. 드라이버는 세 가지 경우를 시뮬레이션합니다: 깨끗한 승인 흐름, 일시적 실패 후 재시도(중복 실행이 되어서는 안 됨), 그리고 도장 찍기식 기본값과 도전 및 응답(challenge-and-response) 흐름의 비교.

## 출시하기

`outputs/skill-hitl-design.md`는 제안 후 커밋(propose-then-commit) 형태를 위해 제안된 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL)) 워크플로우를 검토하고, 누락된 메타데이터, 멱등성, 검증, 또는 도전 및 응답(challenge-and-response) 계층을 플래그합니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 승인된 제안의 재시도가 내구성 있는 레코드를 사용하며 재실행되지 않음을 확인하세요. 이제 멱등성 키에 타임스탬프를 포함하도록 변경하고, 재시도가 이중 실행되는 것을 보여주세요.

2. 제안 레코드를 `rollback` 필드로 확장하세요. 검증 단계가 실패하는 실행을 시뮬레이션하세요. 롤백이 자동으로 실행되는 것을 보여주세요.

3. Microsoft Agent Framework의 `RequestInfoEvent` 문서를 읽어보세요. API가 포함하고 있는 메타데이터 필드 중 토이 엔진이 누락한 것을 하나 식별하세요. 이를 추가하고, 이 필드가 무엇을 보호하는지 설명하세요.

4. 특정 작업(예: "공개 Twitter 계정에 게시")에 대한 도전-응답 체크리스트를 설계하세요. 리뷰어가 답해야 하는 세 가지 질문은 무엇이며, 왜 그 세 가지인가요?

5. 동기식 "승인?" 프롬프트가 충분하고 내구성 있는 저장소가 필요하지 않은 경우를 하나 선택하세요. 이유를 설명하고, 수용하는 위험 클래스를 명시하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|---|---|---|
| 제안 후 커밋 | "2단계 승인" | 저장된 제안 + 긍정적 커밋 + 검증 |
| 멱등성 키 | "재시도 안전 토큰" | 제안마다 고유; 두 번째 실행은 동작하지 않음 |
| 데이터 계보 | "출처가 어디인지" | 제안으로 이어진 특정 출처 콘텐츠 |
| 폭발 반경 | "최악의 경우" | 작업이 잘못될 경우의 영향 범위 |
| 도장 찍기 | "빠른 승인" | 진정한 검토 없이 "승인"을 클릭한 것 |
| 도전-응답 | "체크리스트 강제" | 리뷰어가 특정 질문에 긍정적으로 답해야 함 |
| RequestInfoEvent | "MS Agent Framework 프리미티브" | 구조화된 메타데이터를 가진 내구성 있는 HITL 요청 |
| `interrupt()` / `waitForApproval()` | "프레임워크 프리미티브" | LangGraph / Cloudflare의 동일한 형태 구현 |

## 추가 읽기

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, 내구성 있는 승인.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 및 Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 장기적 위험에 대한 완화책으로서의 HITL.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 고위험 시스템에 대한 규제 기준선.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 감독에 관한 헌법적 프레임.
