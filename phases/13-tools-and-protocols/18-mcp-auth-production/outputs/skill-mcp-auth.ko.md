---
name: mcp-auth-wiring
description: 발급자 바인딩 등록, CIMD, 보호 리소스 메타데이터, JWKS 갱신, 오디언스 고정, 요청별 검증을 포함하는 MCP 2026-07-28 인증을 설계합니다.
version: 2.0.0
phase: 13단계
lesson: 18강
tags: [mcp, oauth, cimd, dcr, jwks, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728, rfc9207]
---

MCP 서버 구성과 IdP 기능 집합이 주어지면, 프로덕션 MCP 인증 계층을 구성하는 인증 표면 및 거부 규칙을 생성합니다.

입력:

- `mcp_resource_url` — 표준 리소스 URL (가장 구체적인 식별자; 공동 호스팅 서버를 구분할 때만 경로 유지), `aud` 및 보호 리소스 메타데이터 `resource` 값으로 사용.
- `idp_metadata_url` — IdP의 `/.well-known/oauth-authorization-server` (또는 OpenID Connect Discovery) URL.
- `idp_capabilities`: `issuer`, `code_challenge_methods_supported`, `grant_types_supported`, `client_id_metadata_document_supported`, 폐기된 `registration_endpoint`, `response_types_supported`, `authorization_response_iss_parameter_supported`의 관찰된 값.
- `pre_registered_client_ids`: 인증 서버 운영자가 제공하는 발급자별 클라이언트 ID 매핑. CIMD보다 먼저 이 발급자 범위 식별자를 선호하며, 폐기된 DCR은 최종 호환성 경로로만 사용.
- `application_type`: `native` 또는 `web`, 폐기된 DCR 호환성을 선택한 경우 필수.
- `credential_store`: 인증 서버 발급자를 키로 하는 클라이언트 ID 및 등록 자격 증명을 포함하며, `(issuer, mcp_resource_url)`을 키로 하는 접근 토큰을 포함.
- `tools`: 각 도구마다 필요한 스코프가 포함된 MCP 도구 목록.

생성:

1. **거부 게이트.** 하드 조건이 하나라도 실패하면 연결을 거부하고 중단합니다:
   - `code_challenge_methods_supported`에 `S256`이 없습니다 (PKCE는 저하 모드 없음).
   - `grant_types_supported`에 `authorization_code`이 없습니다.
   - `response_types_supported`가 정확히 `["code"]`가 아닌 경우.
   - 등록 경로가 없습니다: 사전 등록된 `client_id`, `client_id_metadata_document_supported: true`, 폐기된 DCR 호환성 엔드포인트 중 아무것도 사용할 수 없습니다.
   - CIMD가 선택되었지만 `client_id`가 경로가 포함된 절대 HTTPS 문서 URL이 아니거나, 문서 URL과 일치하지 않거나, 문서에 비어 있지 않은 `client_name` 또는 `redirect_uris` 배열이 없습니다. `application_type`은 CIMD에 선택 사항입니다.
   - 반환된 RFC 9207 `iss`이 리디렉션 전에 기록된 발급자와 다르거나, 서버가 이를 지원한다고 광고했는데도 생략된 경우입니다.
   - 폐기된 DCR에 `application_type`이 없거나, 리디렉션 URI 정책이 `native` 또는 `web`와 충돌합니다.

2. MCP 서버용 **보호된 리소스 메타데이터 문서**(RFC 9728)입니다. 경로가 있는 리소스의 경우, 해당 경로 앞에 well-known 세그먼트를 삽입하세요: `https://host/team/mcp`은 `https://host/.well-known/oauth-protected-resource/team/mcp`에 매핑됩니다. `resource`, `authorization_servers`(발급자 허용 목록), `scopes_supported`, `bearer_methods_supported: ["header"]`를 포함하세요.

3. **HTTP 엔드포인트.**
   - `GET /.well-known/oauth-protected-resource` — (2)의 문서를 반환합니다.
   - `POST /mcp`(상태 비저장 MCP 전송): 도구 실행 전에 이 요청의 베어러 토큰을 검증합니다.
   - DCR 호환성 전용: `POST /register`, 그 앞에 애플리케이션 유형 체크와 속도 제한 체크가 있습니다.

4. **백그라운드 작업 + 루틴.**
   - `jwks_uri`을 캐시 `{keys, fetched_at}`에 다시 가져오는 예약된 JWKS 갱신입니다. 멱등적이며, 키를 생성하지 않습니다. AS는 키를 회전시키지만, 리소스 서버는 갱신만 수행합니다. 기본값은 `0 */6 * * *`이며, 키 회전 빈도가 높은 IdP의 경우 `*/15 * * * *`으로 강화하세요.
   - `validate` 루틴 — `iss` 허용 목록, 캐시된 JWKS에 대한 서명, `aud == mcp_resource_url`, `exp`, 필수 스코프를 확인합니다.
   - 단계적 발급 경로 — 도구 목록에 사용자가 처음에 부여하지 않은 스코프로 제한된 작업이 포함된 경우에만 사용합니다.

5. **캐시 계획.** `issuer`을 키로 하여 승인된 발급자당 하나의 항목을 저장하고, `{keys, fetched_at}`을 보관합니다. 읽기 패턴을 문서화하세요: 검증기는 캐시를 읽고 `kid` 미스 시 단일 동기식 갱신으로 폴백합니다(재가져오기, 회전 아님 — 재가져오기는 멱등적이며 키 생성 DoS로 전환될 수 없습니다).

6. **스코프 매핑.** 모든 도구를 필요한 스코프에 매핑하세요. 표를 출력하세요:
`| tool | required_scope | rationale |`. 파괴적 도구를 자체 스코프에 그룹화하세요. 쓰기 도구에는 읽기 스코프를 절대 재사용하지 마세요.

7. **런타임 거부 규칙**(검증기가 이를 인코딩해야 합니다):
   - `aud != mcp_resource_url`일 경우 거부 → 401 `Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="<prm_url>"`.
   - `iss not in authorization_servers`일 경우 거부.
   - 단일 재가져오기 폴백 후 캐시된 JWKS에 `kid`이 없을 경우 거부.
   - 필수 스코프가 없을 경우 거부 → 403 `Bearer error="insufficient_scope", scope="<required>", resource_metadata="<prm_url>"`.
   - S256 `code_challenge`이 없는 인증 요청은 거부하고, `code_verifier`, 클라이언트, 리다이렉트 URI, 또는 `resource`이 일회용 인증 코드 기록과 일치하지 않는 토큰 요청은 거부하세요.
   - 발행자(issuer)가 자격 증명 저장소 키와 일치하지 않는 자격 증명이나 토큰은 거부하세요. 발행자 변경은 새로운 등록(enrollment)이 필요합니다.

하드 거부 항목 (이 중 어떤 것도 연결하지 마세요 — 요청을 거부하고 이유를 문서화하세요):

- `client_secret`을 평문으로 저장하는 것. 공개 클라이언트는 `token_endpoint_auth_method: none`을, 기밀 클라이언트는 `private_key_jwt`을 사용하세요. 저장 시나 등록 응답 로그에 평문 공유 비밀이 있어서는 안 됩니다.
- 검증자(validator)에서 `aud` 검사를 건너뛰는 것. 오디언스 바인딩(액세스 토큰 권한 제한)은 RFC 8707 + RFC 9728의 존재 이유 그 자체입니다.
- JWKS 캐시 미스 시 폴백(fall-back)을 재-fetch가 아닌 rotate-and-mint로 연결하는 것. 이는 누락된 `kid`을 생성하지 못하며, 공격자가 제어하는 `kid` 값이 무제한 키 생성을 유발하게 합니다. 폴백은 반드시 멱등적인(idempotent) refresh여야 합니다.
- PKCE가 없는 인증 코드 요청을 허용하는 것. OAuth 2.1은 이를 금지합니다; 검증자는 저장된 인증 코드 기록에 `code_challenge`이 없는 `/token` 교환을 거부해야 합니다.
- refresh 작업 없이 JWKS를 캐싱하는 것. 스케줄된 refresh가 배포되지 않으면, 인증 인터페이스는 배포되지 않아야 합니다.
- allow-list 없이 `iss` 클레임을 신뢰하는 것. 임의의 `iss`에서 토큰을 허용하는 검증자는 공격자가 자체 IdP를 구축하고 토큰을 위조할 수 있게 합니다.
- 수신된 MCP 토큰을 업스트림 API로 전달하는 것 (토큰 패스스루). MCP 서버가 업스트림 API를 호출한다면 자체적인 별도 토큰을 획득해야 합니다; 패스스루는 confused-deputy 문제를 만듭니다.
- `registration_access_token`을 평문으로 저장하는 것. 저장 시 해싱(Hash-at-rest)을 적용하고, 모든 업데이트 시 평문을 요구하세요.
- MCP 요청 메타데이터나 제거된 프로토콜 세션을 인증 상태로 취급하는 것. 2026-07-28 트랜스포트는 상태 비저장(stateless)입니다; 모든 요청에 대해 인증하고 승인하세요.

출력: 보호된 자원 문서, 발행자 키 기반 등록 레이아웃, 발행자 및 자원 토큰 레이아웃, 선택된 등록 경로, HTTP 엔드포인트, JWKS refresh 작업, 스코프 매핑, 런타임 거부 규칙을 포함한 한 페이지 계획. 인증 서버의 실제 메타데이터에서 발견된 첫 번째 미충족 배포 게이트로 마무리하세요.
