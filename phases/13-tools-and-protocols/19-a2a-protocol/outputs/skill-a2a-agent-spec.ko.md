---
name: a2a-agent-spec
description: A2A로 호출 가능한 에이전트의 에이전트 카드 및 스킬 스키마를 생성합니다.
version: 1.0.0
phase: 13단계
lesson: 18강
tags: [a2a, agent-card, task-lifecycle, delegation]
---

에이전트의 기능과 의도된 협력자를 고려하여, 해당 에이전트의 A2A 에이전트 카드와 스킬 정의를 생성해 보세요.

생성 대상:

1. 에이전트 카드. `name`, `description`, `version`, `supportedInterfaces[]` (각각 `url`, `protocolBinding`, `protocolVersion` 포함), `capabilities` (스트리밍, 푸시 알림), `defaultInputModes`, `defaultOutputModes`, `securitySchemes` 및 `securityRequirements`, `skills[]`. `/.well-known/agent-card.json`에서 서빙하세요.
2. 스킬 목록. 각각 `id`, `name`, `description`, `tags` 및 선택적 `inputModes` / `outputModes` 미디어 타입을 포함합니다. 설명에는 "X일 때 사용하세요. Y에는 사용하지 마세요." 패턴을 사용하세요.
3. 작업 상태 계획. 각 스킬에 대해 예상되는 상태 전환과 `TASK_STATE_INPUT_REQUIRED` 경로를 정의하세요.
4. 서명 계획. `signatures`에 JWS 항목을 포함하여 카드를 서명할지 여부 (외부에서 호출 가능한 에이전트의 경우 권장됨).
5. 프로토콜 바인딩. `JSONRPC`, `GRPC` 또는 `HTTP+JSON`. 각각 `protocolVersion` `1.0`를 가진 `supportedInterfaces` 항목으로 선언됩니다. 클라이언트는 `A2A-Version: 1.0`를 전송합니다. v1.0과의 하위 호환성을 고려하세요.

거부 조건:
- 안정적인 `supportedInterfaces` URL이 없는 에이전트 카드는 거부합니다. 발견(discovery)이 불가능해집니다.
- `tags`가 없거나 입력 및 출력 모드(스킬 자체 또는 카드 기본값)가 없는 스킬은 거부합니다. 호출자가 호환성을 판단할 수 없게 됩니다.
- 외부에서 호출 가능한 에이전트인데 카드 서명 계획이 없는 경우 거부합니다. 사칭 벡터가 됩니다.

거부 규칙:
- 에이전트의 사용 사례가 단일 도구 호출인 경우, A2A 스캐폴딩을 거부하고 MCP를 권장하세요.
- 에이전트가 노출하지 말아야 할 내부 정보(도구 호출 추적, 사고의 연쇄)를 노출하는 경우, 거부하고 불투명성을 강제하세요.
- 에이전트가 결제에 A2A가 필요한 경우(AP2 사용 사례), AP2 확장 버전을 확인하고 AP2가 코어 A2A와 분리되어 있음을 명시하세요.

출력: 한 페이지 분량의 에이전트 카드 JSON, 각 작업에 대한 스킬 스키마, 상태 전환 계획, 서명 및 전송 선택 사항. 에이전트가 약속하는 최소한의 v1.0 하위 호환성 보장을 마지막에 포함하세요.
