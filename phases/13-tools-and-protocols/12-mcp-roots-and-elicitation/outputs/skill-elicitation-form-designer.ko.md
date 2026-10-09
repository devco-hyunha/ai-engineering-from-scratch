---
name: elicitation-form-designer
description: 권한 부여, 안전한 양식, 서명된 재시도 상태를 사용하여 명시적 리소스 범위 및 상태 비저장 MCP 2026-07-28 elicitation을 설계합니다.
version: 2.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, mrtr, scope, authorization]
---

프로토콜 개정 `2026-07-28`을 목표로 하는 MCP 작업에 대한 사용자 입력 단계를 설계해 보세요.

다음 항목을 생성합니다:

1. 범위 계약(Scope Contract). 작업 공간, 디렉터리, 또는 리소스 URI를 가시적인 도구 인자나 서버 구성에 포함합니다. 이를 사용할 수 있는 인증된 주체를 명시합니다.
2. 경계 포인트(경계 포인트). URI 정규화, 경로 구성 요소 포함, 심볼릭 링크 정책, 운영체제 샌드박스(Sandbox)를 정의합니다.
3. 트리거 조건. 사용자 입력이 필요한 정확한 모호성, 확인, 또는 외부 상호작용을 지정합니다.
4. 발견 및 기능 게이트. `server/discover`에서 정확한 `supportedVersions`, 기능, `ttlMs`, `cacheScope`를 반환합니다. 도구가 광고되는 경우, 유효한 객체 `inputSchema`, 서버 식별자 메타데이터, 캐시 힌트를 포함하는 필수적인 결정론적 `tools/list` 설명자를 포함합니다. `elicitation: {}`과 명시적인 `elicitation.form`은 양식 지원으로 취급합니다. 누락되거나 URL 전용 지원은 `-32021`과 `data.requiredCapabilities.elicitation.form`로 거부하며, 지원되지 않는 버전에는 정확한 `supported` 및 `requested` 데이터를 사용하여 `-32022`을 사용합니다.
5. MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR)) 결과. 안정적인 `inputRequests` 키와 `elicitation/create` 요청을 포함하는 `resultType: "input_required"`를 반환합니다.
6. 상호작용 설계. 양식 모드에서는 평문 메시지와 제한된 평면 스키마를 제공합니다. URL 모드에서는 HTTPS 목적지와 대역 외 완료 규칙을 표시합니다.
7. 재시도 계약. 새로운 JSON-RPC id, 원본 메서드 및 인자, 현재 `inputResponses`, 요청별 `_meta`, 정확한 `requestState` 에코를 요구합니다.
id가 없는 알림은 JSON-RPC 결과나 오류를 받지 않으며, 승인된 Streamable HTTP 알림은 본문이 없는 `202`를 받습니다.
8. 분기 처리. `accept`, `decline`, `cancel`를 서로 다른 안전한 결과로 매핑합니다.
9. 상태 보호. HMAC 또는 인증된 암호화를 인증된 주체, 원본 인자 다이제스트, 후보 집합, 작업 단계, 만료 시간, 일회용 논스에 바인딩합니다. 모든 핸들러 인스턴스가 공유하는 제한된 TTL 제거 재생 저장소에서 논스를 원자적으로 소비합니다.
10. 최종 재검증. 변경(mutation) 직전에 권한 부여, 실시간 레코드 상태, 격리(containment)를 즉시 재확인하세요.

하드 거부(Hard rejects):

- 폐기된 Roots를 권한 부여, 격리(containment), 샌드박스(sandboxing)로 취급하는 것.
- 새로운 2026-07-28 설계에서 `roots/list` 또는 `notifications/roots/list_changed`를 사용하는 것.
- MRTR를 통해 반환하는 대신 역방향 `elicitation/create` 요청을 보내는 것.
- 양식(form) 모드에서 비밀번호, API 키, 접근 토큰, 결제 자격 증명을 수집하는 것.
- 현재 요청별 capabilities에 없는 elicitation 모드를 보내는 것.
- `clientInfo`를 인증된 사용자 식별자로 취급하는 것.
- 검증된 수락(validated acceptance) 및 최종 권한 부여 체크를 수행하기 전에 파괴적인 동작을 수행하는 것.
- 후보(candidates) 또는 권한 관련 데이터를 포함하는 서명되지 않은 `requestState`.

거부 규칙:

- 명시적 거절(explicit decline) 후 반복적인 프롬프트를 거부하세요.
- 서버가 사용자 없이 파생하거나 검증할 수 있는 값에 대한 elicitation을 거부하세요.
- 자격 증명, 사용자 비밀, 사전 인증된 bearer 값이 포함된 URL을 거부하세요.
- 숨겨진 프로토콜 세션 상태, `initialize`, `Mcp-Session-Id`를 사용하는 요청을 거부하세요.

범위(scope), 권한 부여, 격리(containment), 상호작용 모드, 스키마 또는 URL, MRTR 와이어 형식, 상태 필드, 응답 분기, 재생(replay) 정책, 최종 재검증 체크리스트를 포함한 한 페이지 설계를 출력하세요.
