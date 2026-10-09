---
name: mcp-server-designer
description: 명시적 발견, 상태, 전송 및 안전 계약이 포함된 상태 비저장(stateless) MCP 2026-07-28 서버를 설계합니다.
version: 2.0.0
phase: 11단계
lesson: 14강
tags: [llm-engineering, mcp, stateless, tool-use]
---

도메인(내부 API, 데이터베이스, 파일 소스)과 서버를 마운트할 호스트가 주어지면 다음을 출력합니다:

1. 원시(primitive) 맵. 어떤 기능이 `tools` (action)이 되는지, 어떤 기능이 `resources` (read-only data)이 되는지, 어떤 기능이 `prompts` (user-invoked templates)이 되는지 결정합니다. 원시마다 한 줄씩 작성합니다.
2. 발견(discovery) 계약. 구현이 지원하는 정확한 버전, 기능, 서버 식별자, 지침, `ttlMs`, `cacheScope`를 포함하여 `server/discover`를 작성합니다.
3. 요청(request) 계약. 모든 요청의 `params._meta`에서 문자열 프로토콜 버전과 객체 형식의 클라이언트 기능을 요구합니다. 클라이언트 식별자를 권장합니다. 필수 메타데이터가 누락되거나 타입이 올바르지 않은 경우 Invalid Params (`-32602`)를 반환합니다. 서버가 구현하지 않은 버전 문자열이 제공된 경우에만 `UnsupportedProtocolVersionError` (`-32022`)를 `data.supported` 및 `data.requested`와 함께 반환합니다.
4. 결과(result) 계약. 모든 해당 결과에 `resultType`, 서버 식별자 메타데이터, 결정론적 목록 순서, 캐시 정책을 추가합니다.
5. MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR)) 계획. `input_required`는 `tools/call`, `resources/read`, `prompts/get`에만 사용합니다. `inputRequests` 또는 불투명한(opaque) `requestState` 중 하나 이상을 포함합니다. 요청된 경우 입력 응답, 존재하는 경우 정확한 상태 값과 함께 새로운 JSON-RPC ID로 원시 메서드를 재시도합니다.
6. 상태(state) 계획. 모든 다중 호출 워크플로우에 대해, 일반적인 도구 인자로 전달되는 서버가 생성한 불투명한 핸들(opaque handle)을 정의합니다. 연결이나 프로토콜 세션 뒤에 상태를 숨기지 마세요.
7. 전송 및 인증 계획. stdio 또는 2026-07-28 Streamable HTTP POST 엔드포인트를 선택합니다. HTTP의 경우 Origin 검증과 요청별 인증을 정의합니다. POST 요청에는 `MCP-Protocol-Version`, JSON-RPC 요청에는 `Mcp-Method`, `tools/call`, `resources/read`, `prompts/get`에만 `Mcp-Name`를 요구합니다. 승인된 알림(notification) POST는 본문(body) 없이 HTTP 202를 반환합니다.
8. 스키마(schema) 초안. 모든 도구 매개변수에 대해 JSON Schema를 작성하며, 모델 선택을 위해 조정된 설명과 신뢰할 수 없는 입력에 대한 명시적 경계를 포함합니다.
9. 파괴적 작업 목록. 모든 변경(mutating) 도구에 `destructiveHint: true`를 표시하고 인간 승인을 요구합니다.
10. 검증 계획. JSON-RPC 응답을 생성하지 않는 알림, 형식이 잘못된 인벨로프 및 요청 ID, 메타데이터 거부, 발견, 결정론적 목록, 버전 불일치, 캐시 필드, 헤더-본문 불일치, 인증, 승인 및 프롬프트 주입 사례 하나를 포함하세요.

`initialize`, `notifications/initialized`, `Mcp-Session-Id`, 독립적인 HTTP GET, HTTP DELETE 또는 `Last-Event-ID`를 현대적 경로로 사용하는 설계를 거부하세요. 2025-11-25까지의 프로토콜 버전에 대해 명확히 격리된 어댑터 내에서만 이러한 메커니즘을 허용하세요. 새로운 구현에 폐기된 Roots, Sampling 또는 Logging을 추가하지 마세요. 호환성 지원은 라벨이 붙어야 하며 Roots 또는 Sampling 입력은 MRTR를 사용해야 합니다. 인증, 검증 및 승인 경로 없이 디스크에 쓰거나 외부 API를 호출하는 서버는 거부하세요.
