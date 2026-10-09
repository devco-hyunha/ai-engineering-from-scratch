---
name: task-store-designer
description: 현재 Tasks 확장 기능을 사용하여 내구성 있는 MCP 작업을 설계합니다. 상태 비저장 요청, 명시적 소유권, 폴링, 입력 업데이트 및 취소를 포함합니다.
version: 2.0.0
phase: 13단계
lesson: 13강
tags: [mcp, tasks, extension, durable-state, stateless]
---

`io.modelcontextprotocol/tasks` 확장 기능을 사용하여 장기 실행 MCP 작업을 설계해 보세요.

다음 내용을 생성합니다:

1. 적격성 결정. 동기 `tools/call`가 아닌 작업(task)이 필요한 이유를 설명합니다.
2. 기능 계약. `server/discover`에서 정확한 `supportedVersions`, 기능(capabilities), `ttlMs` 및 `cacheScope`를 표시하고, 요청별 클라이언트 기능에 Tasks 확장 기능을 포함합니다. 도구가 광고되는 경우, 유효한 객체 `inputSchema`, 서버 식별자 메타데이터 및 캐시 힌트를 포함하는 필수 결정론적 `tools/list` 설명자를 포함합니다. 확장 기능이 없는 경우 `requiredCapabilities` 객체와 함께 `-32021`를 사용하고, 지원되지 않는 버전의 경우 정확한 `supported` 및 `requested` 데이터와 함께 `-32022`를 사용합니다.
3. 생성 트랜잭션. `tasks/get`가 작업을 해결할 수 있을 때까지 작업을 지속하고, 서버가 지정한 `resultType: "task"`를 반환합니다.
4. 상태 형식. `taskId`, `status`, `statusMessage`, ISO 타임스탬프, `ttlMs`, `pollIntervalMs`, 권한 있는 소유자, 원본 작업 참조, 결과 또는 오류, 미완료 입력 요청 및 발급된 모든 입력 키를 포함합니다. 완료된 작업의 중첩된 `CallToolResult`에는 필수 `resultType: "complete"`가 있으며, 자체 `io.modelcontextprotocol/serverInfo` 메타데이터를 포함해야(SHOULD) 합니다.
5. 현재 메서드. `tasks/get`, `tasks/update` 및 `tasks/cancel`를 정의합니다. Streamable HTTP의 경우 각 요청은 `Mcp-Name`를 `params.taskId`로 설정합니다. `tasks/status`, `tasks/result` 또는 `tasks/list`를 도입하지 마세요.
6. 입력 연속. 생성 전 MRTR과 생성 후 `tasks/get` 및 `tasks/update`를 분리합니다. 수명 동안 유일한 입력 키와 부분 응답 처리를 요구합니다.
7. 내구성 계획. 원자적 파일 시스템 저장소, 트랜잭션 데이터베이스 또는 공유 큐 및 저장소를 선택합니다. 워커 리스 및 재시작 동작을 포함합니다.
8. 소유권 정책. 모든 작업 메서드와 구독을 테넌트 및 주체(principal)별로 승인합니다. 작업 ID 지식은 절대 권한으로 취급하지 마세요.
9. 취소 계약. 승인(acknowledgement)은 협력적이며 `cancelled`로 이어지지 않을 수 있음을 명시합니다.
10. 알림 옵션. POST 응답 SSE 스트림에 `subscriptions/listen`을 사용하고 `notifications/tasks`을 사용하며, 폴링을 기본으로 설정하세요. `io.modelcontextprotocol/subscriptionId`을 알림 요청 ID와 동일하게 설정하여 승인 및 모든 작업 알림에 포함하세요. ID가 없는 알림은 JSON-RPC 응답을 받지 않으며, 승인된 HTTP 알림은 본문 없이 `202`을 받습니다.
11. 만료 정책. 생성 시점부터 `ttlMs`을 해석하고, 소거 동작을 정의하며, 다른 테넌트의 작업이 존재하는지 여부를 노출하지 않도록 하세요.
12. 마이그레이션 맵. 클라이언트가 요청한 작업 플래그와 제거된 실험적 메서드를 현재 확장 플로우로 교체하세요.

하드 거부:

- 영속적 읽기 가시성 확보 전에 작업 핸들 반환.
- 확장 기능을 광고하지 않은 요청에 `resultType: "task"`을 반환.
- `params._meta.task.required`, `tasks/status`, `tasks/result`, `tasks/list`을 현재 API로 사용.
- `initialize`, `Mcp-Session-Id`, 스티키 라우팅, 또는 숨겨진 전송 세션 상태를 작업 저장소로 사용.
- `tasks/cancel` 승인을 워커가 중지된 증거로 취급.
- 하나의 작업 수명 동안 `inputRequests` 키를 재사용.
- 작업의 권위 있는 소유자가 아닌 호출자에게 작업 반환.
- 독립적인 GET, 세션 SSE, 또는 `Last-Event-ID` 재생을 통해 알림 전달 구현.

거부 규칙:

- 호출자가 구체적인 내구성 요구 사항을 제시하지 않는 한, 빠른 결정적 조회를 위한 작업을 거부하세요.
- 작업이 프로세스 재시작을 생존해야 할 때, 메모리 전용 프로덕션 저장소를 거부하세요.
- 무제한 결과 페이로드를 거부하세요. 대형 아티팩트는 외부에 저장하고 승인된 리소스 핸들을 반환하세요.
- 명시적인 테넌트 소유권, 필터링, 페이지네이션, 보존 정책이 없는 히스토리 엔드포인트를 거부하세요.

수명 주기 표, 와이어 메서드, 영속성 트랜잭션, 소유권 규칙, 입력 플로우, 폴링 주기, 취소 의미론, 구독 옵션, 만료 정리, 실패 모델, 레거시 마이그레이션 맵을 포함한 한 페이지 디자인을 출력하세요.
