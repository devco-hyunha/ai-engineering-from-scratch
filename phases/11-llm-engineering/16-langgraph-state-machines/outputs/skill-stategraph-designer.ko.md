---
name: stategraph-designer
description: 에이전트 작업을 LangGraph StateGraph로 변환합니다. 명명된 노드, 타입이 지정된 상태, 리듀서, 체크포인터, 인간 개입 루프(HITL)를 포함합니다.
version: 1.0.0
phase: 11단계
lesson: 16강
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

에이전트 작업(사용자-facing 목표, 사용 가능한 도구, 예상 턴 수, 안전 영향 범위를 가진 부작용, 내구성 있는 실행(Durable Execution) 요구 사항, 목표 지연 시간 예산)이 주어지면 다음을 출력합니다:

1. 노드 목록. 모든 개별 단계를 이름으로 지정합니다: LLM 사고자, 각 도구 실행자, 모든 인간 검토 단계, 요약자나 비평가, 검색자. 하나의 노드가 여러 관심사를 다루는 경우 설계를 거부하고 분리합니다.
2. 상태 스키마. 모든 리스트에 리듀서가 있는 TypedDict(또는 Pydantic) 필드. 메시지 로그에는 항상 Annotated[list, add_messages]를 사용합니다. 계획, 예산 카운터, 검색된 문서 목록 등 작업별 리스트를 messages에서 분리하여 병렬 업데이트 시 리듀서가 올바르게 작동하도록 합니다.
3. 엣지 맵. 다음 단계가 결정적인 경우 정적 엣지를 사용합니다. 모델이 다음 단계를 선택하는 경우에만 명명된 라우터 함수를 사용하는 조건부 엣지를 사용합니다. 라우터 함수가 이전 노드에서 이미 수행하지 않은 새로운 LLM 호출에 의존하는 그래프는 거부합니다.
4. 인터럽트 배치. 되돌릴 수 없는 부작용(쓰기, 삭제, 결제, 비용이 발생하는 외부 API 호출)이 있는 모든 노드에 interrupt_before를 사용합니다. 출력 검증이 별도 프로세스에서 실행될 때 모델 노드에 interrupt_after를 사용합니다. 부작용이 있는 노드에는 interrupt_after를 거부합니다; 그 시점에는 부작용이 이미 발생했습니다.
5. 체크포인터. 테스트 전용으로 MemorySaver를 사용합니다. 재시작을 견뎌야 하는 환경에서는 PostgresSaver, SQLiteSaver, RedisSaver 중 하나를 선택합니다. thread_id 전략(사용자별, 세션별, 대화별)과 체크포인트 TTL을 확인합니다.

체크포인터가 없는 LangGraph는 출시하지 마세요. 체크포인터가 없으면 재개, 시간 여행, 인간 개입 루프(HITL) 재생이 불가능합니다. add_messages가 없는 messages 필드를 출시하지 마세요; 두 번째 쓰기가 첫 번째 쓰기를 조용히 덮어써서 대화의 절반이 사라집니다. 모든 전이가 플래너 LLM이 라우팅하는 조건부 엣지인 그래프는 거부하세요; 이는 추가 단계가 있는 AutoGen이며 턴마다 토큰을 소모합니다.

예시 입력: "Anthropic Claude 기반의 환불 처리 에이전트. 세 개의 도구(lookup_order, issue_refund, send_email)를 사용하며, 100달러를 초과하는 모든 환불에 대해 인간 개입 루프(Human-in-the-Loop (HITL))를 통해 인간 검토를 위해 일시 중지해야 하고, 서버 재시작 후 재개해야 하며, p95 지연 시간 예산은 8초입니다."

예시 출력:
- 노드: agent (LLM (대규모 언어 모델)(LLM (Large Language Model)) 호출), lookup_tool, refund_tool, email_tool, human_review.
- 상태: add_messages를 사용하는 messages, order_context (overwrite), refund_amount (overwrite), reviewer_decision (overwrite).
- 엣지: agent에서 should_continue 라우터로 연결되며, lookup_tool, refund_tool, email_tool, human_review, END로 분기됩니다. 도구 노드들은 agent로 돌아갑니다.
- 인터럽트: refund_amount가 100을 초과할 때 refund_tool에 interrupt_before를 적용합니다. lookup_tool이나 email_tool에는 인터럽트를 적용하지 않습니다.
- 체크포인터: thread_id가 "user:{user_id}:case:{case_id}"이고 TTL이 30일인 PostgresSaver를 사용합니다.
