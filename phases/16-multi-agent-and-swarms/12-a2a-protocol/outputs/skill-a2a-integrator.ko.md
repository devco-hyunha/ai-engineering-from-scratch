---
name: a2a-integrator
description: 두 에이전트 간 A2A 통합 설계 — 에이전트 카드, 작업 스키마, 인증, 스트리밍 또는 폴링.
version: 1.0.0
phase: 16단계
lesson: 12강
tags: [multi-agent, a2a, protocol, interoperability, google]
---

상호 운용이 필요한 두 에이전트 시스템이 주어졌을 때, A2A 통합 계획을 작성해 보세요: 에이전트 카드 내용, 작업 스키마, 인증, 전송 모드.

다음 내용을 작성하세요:

1. **에이전트 카드.** 이름, 설명, 버전, 스킬 (각 스킬에 `id`, `name`, `description`, `tags` 포함), `supportedInterfaces` (각 항목에 `url`, `protocolBinding`, `protocolVersion` 포함), `defaultInputModes` 및 `defaultOutputModes`를 미디어 타입(텍스트, 구조화된 JSON, 이미지, 오디오, 비디오)으로 지정하고, `securitySchemes` 및 `securityRequirements`에 인증 선언을 포함하세요.
2. **스킬별 작업 스키마.** 입력 JSON 스키마 + 산출물 JSON 스키마. 명확하게 작성하세요 — 클라이언트가 검증합니다.
3. **인증 선택.** Bearer 토큰 (OAuth2 또는 불투명 토큰), mTLS, 또는 API 키를 각각 `securitySchemes` 항목으로 선언하세요. 위협 모델(공용 인터넷, VPC, 혼합)을 고려하여 정당화하세요.
4. **전송 모드.** 폴링 (`GetTask`) vs SSE 스트리밍 (`SendStreamingMessage`, `SubscribeToTask`) vs 푸시 알림 (`CreateTaskPushNotificationConfig`). 장기 실행 작업이나 진행 상황 중심 작업에는 스트리밍을, 짧은 작업에는 폴링을 사용하세요.
5. **속도 제한.** 클라이언트별 및 작업별 제한. 남용 방지.
6. **멱등성(Idempotency).** 중복 `POST /message:send` 요청에 대한 전략 (안정적인 클라이언트 `messageId`, 서버 측 중복 제거 적용).
7. **실패 처리.** `TASK_STATE_FAILED` 이후의 작업 상태 (재시도 가능 vs 치명적), 데드 레터 정책, 오류 산출물 스키마.
8. **MCP vs A2A 분리.** 원격 에이전트가 내부적으로 MCP를 사용하는 경우, 노출된 도구와 내부 유지 도구를 명시하세요.

하드 리젝트(Hard rejects):

- `supportedInterfaces` 항목이 `protocolVersion`를 선언하지 않은 에이전트 카드.
- 사용 사례가 구조를 요구하는데 자유 형식 텍스트인 작업 스키마.
- 공용 인터넷 배포에서 `securityRequirements`가 비어 있는 경우.

거부 규칙:

- 두 에이전트가 동일한 프로세스에서 실행되면 A2A를 거부하고 직접 Python/JS 호출을 권장하세요. A2A는 시스템 간 경계를 위한 것입니다.
- 레이턴시 요구 사항이 왕복 100ms 미만이면 A2A를 거부하고 공유 스키마를 사용하는 직접 RPC를 권장하세요.
- 원격 에이전트가 에이전트 카드(Agent Card)를 선언하지 않으면 통합을 거부하고, 먼저 에이전트 카드를 게시할 것을 권장합니다.

출력: 한 페이지 분량의 통합 브리프. 엔지니어링이 `/.well-known/agent-card.json`에 붙여넣을 수 있도록 에이전트 카드 JSON을 인라인으로 붙여넣어 마무리합니다.
