---
name: hitl-design
description: 제안된 인간 개입 루프 (HITL)(Human-in-the-Loop) 워크플로우를 제안 후 커밋(propose-then-commit) 형태에 대해 검토하고, 누락된 메타데이터, 멱등성(Idempotency), 검증, 또는 도전-응답(challenge-and-response) 계층을 식별합니다.
version: 1.0.0
phase: 15단계
lesson: 15강
tags: [hitl, propose-then-commit, idempotency, langgraph, cloudflare, agent-framework, eu-ai-act]
---

제안된 인간 개입 루프 (HITL)(Human-in-the-Loop) 워크플로우가 주어지면, 제안 후 커밋(propose-then-commit) 참조 표준에 대해 감사하고, 누락된 부분, 불명확한 부분, 또는 규제와 호환되지 않는 부분을 식별합니다.

다음 내용을 생성합니다:

1. **제안 메타데이터.** 모든 제안이 다음을 포함하는지 확인합니다: 의도 (왜), 데이터 계보(Data Lineage)(source content), 접근 권한, 폭발 반경 (최악의 경우), 롤백(Rollback) 계획. 누락된 필드는 차단 요인입니다; "에이전트(Agent)가 X를 원한다"는 제안이 아닙니다.
2. **멱등성(Idempotency).** 멱등성 키 구성을 명시합니다. 재시도가 동일한 레코드를 반환하도록 제안 내용에서 파생될 수 있어야 합니다. 벽시계 시간(wall-clock time)을 포함하는 키는 멱등성 키가 아니며, 로깅 타임스탬프입니다.
3. **내구성(Durability).** 저장소(PostgreSQL, Redis, Durable Object, 무결성 검사가 있는 객체 저장소)를 명시합니다. 승인 내역이 에이전트(Agent) 재시작, 호스트 재시작, 배포를 통해 유지되는지 확인합니다. 메모리 내 큐는 자격이 없습니다.
4. **승인 인터페이스.** 단순 승인 (단일 Approve 버튼)은 이 감사에 실패합니다. 요구 사항: 의도 이해, 폭발 반경 검증, 롤백(Rollback) 준비 상태에 대한 긍정적 확인을 포함하는 도전-응답(challenge-and-response) 체크리스트. 체크리스트가 특정 작업 클래스에 맞춰져 있는지, 일반적이지 않은지 확인합니다.
5. **커밋 후 검증.** 워크플로우가 실행 후 대상 자원을 다시 읽고 검증 실패 시 알림을 보내는지 확인합니다. "도구가 200을 반환했다"는 검증이 아닙니다.

하드 거부(Hard rejects):
- 제안을 내구성 있게 지속하지 않는 인간 개입 루프 (HITL)(Human-in-the-Loop) 인터페이스.
- 리뷰어가 에이전트(Agent) 자체인 승인 흐름.
- 도전-응답(challenge-and-response)이 없는 되돌릴 수 없는 프로덕션 작업.
- 벽시계 시간(wall-clock) 구성 요소를 포함하는 멱등성 키.
- 중요한 작업에 커밋 후 검증이 없는 워크플로우.

거부 규칙:
- 사용자가 승인 UI를 지정하지만 그 뒤의 내구성 있는 저장소를 지정할 수 없다면, 거부하고 먼저 저장소를 요구합니다.
- 사용자가 "max_budget_usd와 확인 대화상자"를 충분한 인간 개입 루프(HITL)로 간주한다면, 거부하세요. 예산은 비용을 제한할 뿐, 정확성을 보장하지 않습니다.
- 배포가 고위험 EU 범위를 다루며 rubber-stamp 패턴이 남아 있다면, 제14조 근거로 거부하세요.

출력 형식:

제안 후 커밋 감사(propose-then-commit audit)를 다음 내용과 함께 반환하세요:
- **제안 필드 표** (의도 / 계보 / 영향 범위 / 롤백 / 권한 — 다섯 가지 모두 필수)
- **멱등성(Idempotency) 노트** (키 구성, 재시도 테스트 결과)
- **내구성(Durability) 라인** (저장소, 재시작 후 생존 여부 y/n)
- **승인 표면(Approval surface)** (rubber-stamp / 체크리스트; 체크리스트인 경우 질문 목록 나열)
- **커밋 후 검증(Post-commit verify)** (존재 여부 y/n, 다시 읽는 내용)
- **준비 상태(Readiness)** (production / staging / research-only)
