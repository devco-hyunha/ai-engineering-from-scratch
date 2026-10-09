---
name: oauth-scope-planner
description: CIMD, 발급자 격리, 리소스 지표, 단계적 승격 스코프를 사용하여 MCP 2026-07-28 인증을 설계합니다.
version: 2.0.0
phase: 13단계
lesson: 16강
tags: [mcp, oauth, cimd, pkce, issuer, resource-indicators]
---

원격 HTTP MCP 서버와 그 도구 목록이 주어지면, 완전한 인증 경계를 설계해 보세요.

## 필수 입력

- 정규화된 MCP 리소스 URI 및 보호된 리소스 메타데이터 위치.
- 허용된 인증 서버 발급자.
- 클라이언트 런타임: 네이티브 또는 웹이며, 정확한 리다이렉트 URI가 필요합니다.
- 도구-스코프 매핑 및 중대한 연쇄 작업.
- 토큰, 갱신 및 자격 증명 저장 제약 조건.
- CIMD가 없는 레거시 인증 서버가 있다면 포함합니다.

## 산출물

1. 리소스 메타데이터. RFC 9728 `resource`, `authorization_servers`, `scopes_supported`를 초안 작성합니다. well-known 세그먼트 이후의 리소스 경로를 유지하며, `https://notes.example.com/mcp`에 대해 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`과 같이 처리합니다.
2. 발급자 정책. 허용된 발급자를 명시하고, 메타데이터 검증, 변경 처리 및 RFC 9207 `iss` 비교를 정의합니다.
3. 등록. 사전 등록이 가능한 경우 이를 사용하며, 그렇지 않으면 클라이언트 ID 메타데이터 문서(Client ID Metadata Document)를 우선합니다. 경로가 포함된 HTTPS URL이 `client_id`입니다. 정확한 리다이렉트 URI를 요구하고, 표시 메타데이터는 신뢰할 수 없는 것으로 취급합니다. `application_type`은 여기서 선택 사항입니다.
4. DCR 폴백. 필요하다면 이를 폐기된 것으로 표기하고, `application_type`을 선언하며, 폴백을 허용하는 정확한 조건을 정의합니다. 일반적인 CIMD 보안 실패 후에는 다운그레이드하지 마세요.
5. 자격 증명 키. 사전 등록 및 DCR 자격 증명은 발급자별로 저장하고, 토큰은 `(issuer, resource)` 아래에 저장합니다. 발급자 간 재사용을 금지합니다. 자체 호스팅 CIMD URL은 이식 가능하며, 신뢰할 수 있는 발급자가 변경될 때 DCR 재등록이 필요하지 않음을 명시합니다.
6. PKCE 흐름. S256, 정확한 리다이렉트 URI, 인증 응답 발급자 검증, 그리고 인증 및 토큰 요청에서 동일한 리소스를 요구합니다.
7. 스코프 모델. 모든 도구를 최소 스코프에 매핑합니다. 현재 `WWW-Authenticate` 스코프 챌린지를 권위 있는 것으로 취급합니다.
8. 단계적 승격 경험. 추가 스코프, 사용자 설명, 동의 지점, 새로운 인증 및 새로운 MCP 요청 ID를 사용한 재시도를 식별합니다.
9. 리소스 서버 점검. 유효한 객체 루트 스키마, 결정적 순서, 결과 유형, 서버 식별자 및 캐시 힌트를 포함하는 광고된 `tools/list`을 구현하세요. 도구 디스패치 전에 발급자, 대상, 만료일, 범위, 현재 MCP 헤더 및 요청 메타데이터를 검증하세요.
10. 토큰 관리. Bearer 헤더만 사용, 쿼리 토큰 금지, 토큰 패스스루 금지, 기밀성 있는 리프레시 토큰 저장 및 로테이션 계획.
11. 오류 계약. OAuth 실패를 포함하여 모든 요청 ID를 JSON-RPC 오류 엔벨로프에 보존하세요. 헤더 불일치 시 HTTP 400 `-32020`을 HTTP 400 `-32022` 버전 지원 체크보다 먼저 요구하며, 지원되는 데이터와 요청된 데이터를 정확히 지정하고, 알 수 없는 메서드에 대해 HTTP 404 `-32601`을 반환하며, 승인된 알림에 대해 빈 본문으로 202를 반환하세요.
12. 전송 경계. 파싱된 본문 예제를 프로세스 내 프로토콜 모델로 라벨링하고, JSON Content-Type 및 JSON plus SSE Accept 검증을 위해 09강의 완전한 Streamable HTTP 어댑터에 연결하세요.

## 하드 거부

- DCR을 선호되는 신규 등록 메커니즘으로 제시하세요.
- `application_type` 없이 DCR을 사용하세요.
- 발급자가 변경된 후 발급자가 생성한 등록 자격 증명, 액세스 토큰 또는 리프레시 토큰을 재사용하지 마세요. 자체 호스팅 CIMD URL은 발급자가 생성한 비밀이 아닌 이식 가능한 예외입니다.
- 비교 전에 `iss`을 사용하여 인증 응답을 정규화하지 마세요.
- 인증 및 토큰 요청에서 PKCE S256가 누락되거나 `resource`가 누락되지 마세요.
- 다른 대상을 위한 토큰을 수락하거나 MCP 토큰을 다운스트림으로 전달하지 마세요.
- `clientInfo`, `serverInfo`, 기능(capabilities) 또는 제거된 프로토콜 세션을 인증으로 사용하지 마세요.
- 원격 HTTP을 모방하기 위해 로컬 stdio에만 OAuth를 추가하지 마세요.
- RFC 9728 메타데이터 URL을 구성할 때 보호된 리소스 경로를 삭제하지 마세요.
- MCP 요청 오류에 대해 JSON-RPC 엔벨로프 대신 평문 텍스트나 임시 객체를 반환하지 마세요. 같은 id를 가진 JSON-RPC 엔벨로프를 반환하세요.

## 출력 형식

Resource, Issuers, Enrollment, Credential Store, PKCE Flow, Scope Matrix, Step-Up, Server Validation, Token Hygiene 및 Compatibility라는 이름의 섹션을 반환하세요. 발급자 검토를 강제하는 정확한 이벤트와, 발급자가 생성한 클라이언트의 경우 재등록으로 마무리하세요.
