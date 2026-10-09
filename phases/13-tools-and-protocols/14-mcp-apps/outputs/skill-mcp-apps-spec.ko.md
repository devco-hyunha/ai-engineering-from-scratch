---
name: mcp-apps-spec
description: 상태 비저장 2026-07-28 프로토콜에서 MCP App 계약을 설계하고 검토합니다.
version: 2.0.0
phase: 13단계
lesson: 14강
tags: [mcp, apps, stateless, ui-resources, csp, sandbox]
---

인터랙티브 뷰가 필요할 수 있는 MCP 도구가 주어지면, 프레임워크 중립적인 계약을 생성해 보세요.

## 필수 입력

- 도구 이름, 인자, 일반 텍스트 결과 및 구조화된 결과.
- 뷰가 지원해야 하는 사용자 상호작용.
- 데이터 민감도 및 응답이 인증 컨텍스트에 따라 달라지는지 여부.
- 뷰가 필요로 하는 브라우저 권한 및 외부 출처(origin).
- Apps 지원이 없는 호스트에서의 텍스트 전용 동작.

## 생성

1. 현재 핵심(envelope) 구조. `2026-07-28`, 요청별 `protocolVersion`, `clientCapabilities`, 권장되는 `clientInfo`, 일치하는 `Mcp-Method` 및 `Mcp-Name` 헤더, 그리고 `resultType` 응답을 표시해 보세요.
2. 발견(discovery) 항목. `server/discover`에서 `io.modelcontextprotocol/ui`을 광고하며, 보수적인 `ttlMs` 및 `cacheScope`을 포함합니다.
3. 도구 선언. `tools/list`이 반환하는 도구에 중첩된 `_meta.ui.resourceUri`을 배치합니다. `tools/call`가 UI를 드러낼 때까지 기다리지 마세요.
4. 리소스 계약. `resources/read` 전에 결정적인 `resources/list` 메타데이터를 포함합니다. 하나의 표준 `ui://` URI, 안정적인 이름 및 설명, `text/html;profile=mcp-app`, 캐시 힌트, CSP 도메인 목록(`connectDomains`, `resourceDomains`, `frameDomains`, `baseUriDomains`) 및 최소 권한 객체를 제공합니다.
5. 결과 계약. 호스트가 App을 렌더링하는지 여부에 관계없이 유용한 텍스트와 구조화된 데이터를 반환합니다.
6. 브리지 계약. 모든 Apps `ui/*` 또는 프록시된 메서드, 정확한 메시지 출처(origin), 인자 스키마, 결과 스키마 및 호스트 측 동의 확인을 나열합니다.
7. 폴백(fallback). 클라이언트가 Apps 확장 기능을 생략했을 때의 도구 및 결과를 설명합니다.
8. 검증 표. 라우팅 전 HTTP 400 `-32020` 헤더 불일치, 지원 및 요청된 버전 데이터가 포함된 HTTP 400 `-32022`, `data.requiredCapabilities`이 포함된 HTTP 400 `-32021`, HTTP 404 `-32601`, 202 빈 본문 알림, CSP 위반, 신뢰할 수 없는 콘텐츠, 무단 브리지 호출 및 텍스트 폴백을 다루세요.
9. 전송 경계. 구현이 파싱된 요청과 헤더를 수신한다면, 이를 프로세스 내 프로토콜 모델로 표시하고 09강의 완전한 Streamable HTTP 어댑터에 연결하세요. 실제 어댑터는 JSON Content-Type과 JSON 및 SSE를 포함하는 Accept 값을 요구해야 합니다.

## 하드 거부

- 현재 MCP로 제시된 핵심 `initialize`, `notifications/initialized`, `Mcp-Session-Id` 경로.
- 와일드카드 `postMessage` 대상 오리진 또는 `event.origin` 검증을 건너뛰는 수신자.
- 도구 실행 후에만 드러나는 UI 바인딩.
- 와일드카드 CSP 도메인 목록, 무제한 네트워크 오리진, 또는 가시적인 기능 없는 권한.
- 정의된 정화(sanitization) 경계 없이 삽입된 사용자 제어 HTML.
- iframe 클릭을 호스트 인증으로 취급하는 중요한 UI 작업.
- 자원을 광고하지만 `resources/list`를 생략하는 서버.
- `id` 없이 알림에 대한 모든 JSON-RPC 응답 본문.

## 호환성 경계

레거시 평면 UI 메타데이터는 폴백으로 읽힐 수 있지만, 새로운 출력은 중첩된 `_meta.ui.resourceUri`를 사용합니다. `ui/initialize`는 Apps postMessage 핸드셰이크로 식별된 경우에만 허용됩니다. 제거된 MCP 코어 초기화를 대체하는 용도로는 절대 사용되지 않습니다.

## 출력 형식

다음 헤딩을 포함하는 간결한 디자인을 반환하세요: 코어 와이어, 발견, 도구, 자원, 결과, 브리지, 보안, 폴백, 검증. 가장 위험한 오리진, 권한, 또는 동의 가정을 단일 항목으로 마무리하세요.
