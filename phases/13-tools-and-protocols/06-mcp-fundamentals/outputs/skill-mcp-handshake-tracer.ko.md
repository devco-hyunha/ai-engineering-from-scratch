---
name: mcp-request-tracer
description: 현대의 상태 비저장 및 명시적 레거시 프로토콜 시대에 걸쳐 MCP 트랜스크립트를 메시지 단위로 감사합니다.
version: 2.0.0
phase: 13단계
lesson: 06강
tags: [mcp, json-rpc, stateless, metadata, compatibility]
---

MCP JSON-RPC 엔벨로프 시퀀스가 주어지면, 각 메시지를 MCP `2026-07-28`에 대해 독립적으로 감사합니다. 레거시 트래픽을 감지하되, 핸드셰이크나 프로토콜 세션이 존재한다고 가정하지 마세요.

다음 내용을 생성합니다:

1. 메시지 주석. 방향, JSON-RPC 유형, 메서드, 원시(primitive), 요청 ID, 감지된 시대(state)를 명시합니다.
2. 현대적 메타데이터 확인. 모든 요청에 대해 `params._meta.io.modelcontextprotocol/protocolVersion`과 `params._meta.io.modelcontextprotocol/clientCapabilities`을 검증합니다. 권장되는 `clientInfo`가 존재하는지 기록합니다.
3. 결과 확인. 모든 현대적 성공 응답에 `resultType: "complete"` 또는 지정된 다른 결과 유형이 있으며, 결과 `_meta`에 권장되는 서버 식별자가 포함되는지 검증합니다.
4. 발견 및 버전 확인. 현대적 서버가 `server/discover`을 구현하는지 검증합니다. `-32022`을 현대적 증거로 해석하고 `data.requested` 및 `data.supported`을 확인합니다.
5. 캐시 확인. `server/discover`의 경우 메서드를 나열하고, `resources/read`의 경우 `ttlMs` 및 `cacheScope`을 요구합니다. 비결정적 목록 순서를 플래그합니다.
6. 방향 확인. 현대적 트래픽에서 서버가 개시한 JSON-RPC 요청을 거부합니다. 요청 관련 알림 및 클라이언트가 연 `subscriptions/listen` 스트림은 허용합니다.
7. 호환성 확인. `initialize`과 `notifications/initialized`을 레거시로만 분류합니다. 현대적 트래픽에서는 이를 요구하지 마세요.

하드 거부(Hard rejects):

- stdio 프로세스, HTTP 연결, 또는 `Mcp-Session-Id`를 현대적 프로토콜 상태로 취급하는 것.
- 이전 요청에서 클라이언트 기능을 추론하는 것.
- `-32020`, `-32021`, `-32022`와 같은 인식된 현대적 오류 이후 레거시로 폴백하는 것.
- `resultType` 없이 현대적 성공 응답을 허용하는 것.

거부 규칙:

- 트랜스크립트가 JSON-RPC 2.0이 아닌 경우, 중단하고 호환되지 않는 엔벨로프를 식별합니다.
- 증거를 조용히 재작성하라는 요청을 받으면 거부합니다. 원본 트랜스크립트를 보존하고 별도의 수정된 예제를 생성합니다.

도착 순서대로 메시지당 한 줄을 출력합니다:

```text
[request/modern/tools] id=7 tools/list metadata=valid
```

현대적, 레거시, 유효하지 않은, 모호한 메시지의 개수를 나열하고, 첫 번째 시정 조치를 따릅니다.
