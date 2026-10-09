# 프로덕션에서의 MCP 인증: 발급자 바인딩 등록 및 토큰

> 16강에서는 OAuth 2.1 상태 머신을 구축했습니다. 이 강의는 MCP 2026-07-28을 위해 프로덕션 경계를 강화합니다: Client ID Metadata Documents를 우선하고, 호환성을 위해 폐기된 동적 등록만 사용하며, 인증 응답의 발급자 검증, 발급자 키 기반 클라이언트 자격 증명, JWKS 갱신, 그리고 모든 상태 비저장 요청에 대해 청중 고정 토큰을 사용합니다.
>
> **규격 노트 (2026-07-28):** 동적 클라이언트 등록(Dynamic Client Registration)은 Client ID Metadata Documents를 선호하여 폐기되었습니다. DCR은 호환성 메커니즘으로 남아 있습니다. 이를 사용할 때 클라이언트는 올바른 `application_type`을 선언합니다. 클라이언트는 존재하는 RFC 9207 `iss` 값을 검증하며, 인증 서버 발급자 간에 자격 증명을 재사용하지 않습니다.

**유형:** Build
**언어:** Python (stdlib)
**선수 요건:** 13단계 · 16강 (OAuth 2.1 상태 머신), 13단계 · 17강 (게이트웨이)
**시간:** 약 90분

## 학습 목표

- RFC 8414 메타데이터를 통해 인증 서버를 발견하고 계약을 검증해 보세요.
- Client ID Metadata Documents를 통해 등록하고, 폐기된 DCR을 폴백으로 격리해 보세요.
- RFC 9207 `iss`을 검증하고, 인증 서버 발급자별로 등록을 키화하며, 발급자 및 리소스별로 리소스 바인딩 토큰을 키화해 보세요.
- JWKS 키를 스케줄에 따라 캐시하고 갱신하여 키 롤오버에도 서명 검증이 유지되도록 해 보세요.
- RFC 8707 리소스 지표(RFC 8707 resource indicators)를 사용하여 토큰을 단일 MCP 리소스에 고정하고, 혼란스러운 대리인(confused-deputy) 재사용을 거부해 보세요.
- JWT 검증 또는 토큰 인스펙션을 선택하고, 취소 신선도를 정의하며, 신원 의존성이 사용 불가능할 때 안전하게 실패하도록 해 보세요.
- 인증 서버, 리소스 서버, 클라이언트를 분리하여 각각이 자체 검사만 강제하도록 해 보세요.
- 배포 체크리스트에 대해 인증 서버를 감사하고, 안전하지 않은 등록이나 토큰 재사용을 거부해 보세요.

## 문제점

16강 시뮬레이터는 메모리에서 OAuth 2.1을 실행합니다. 프로덕션에는 메모리 전용 시뮬레이터가 볼 수 없는 세 가지 운영적 공백이 있습니다.

첫 번째 공백은 등록 및 자격 증명 격리입니다. 실제 조직은 수백 개의 MCP 서버와 수천 개의 MCP 클라이언트를 운영할 수 있습니다. 2026-07-28 개정판은 **클라이언트 ID 메타데이터 문서(Client ID Metadata Document)**를 선호합니다. 클라이언트는 본인이 제어하는 HTTPS URL의 경로를 식별자로 사용하며, 인증 서버는 이 메타데이터를 가져옵니다. RFC 7591 동적 등록(DCR)은 더 이상 권장되지 않는 호환 경로로만 남아 있습니다. DCR이 불가피한 경우, 요청은 올바른 `application_type`을 선언합니다. 클라이언트는 인증 서버의 발급자(issuer) 아래에 등록 정보를 저장하고, `(issuer, resource)` 쌍 아래에 접근 토큰을 저장합니다. 발급자가 변경되면 새로운 등록이 필요하며, 다른 리소스는 별도로 청중(audience)에 바인딩된 토큰을 의미합니다.

두 번째 공백은 키 회전입니다. JWT 검증은 인증 서버의 서명 키에 의존하며, 이 키는 JSON 웹 키 세트(JWKS)로 게시됩니다. 인증 서버는 이 키들을 주기적으로 회전합니다(보통 매시간, 인시던트 대응 시에는 더 빠르게). 부팅 시 JWKS를 한 번만 가져오는 MCP 서버는 회전 윈도우 전까지는 정상적으로 검증되지만, 이후에는 재시작할 때까지 모든 요청이 실패합니다. 프로덕션 환경에서는 JWKS를 캐시된 값으로 연결하고, 이전 키가 만료되기 전에 캐시를 덮어쓰는 갱신 작업을 설정하며, 캐시보다 최신 키로 서명된 토큰이 도착하는 경우 캐시 미스 시 폴백(fall-back) 가져오기를 수행합니다.

세 번째 공백은 청중 바인딩입니다. 16강에서 RFC 8707 리소스 지표(resource indicators)를 소개했습니다. 프로덕션 환경에서는 이 지표가 모든 요청에 대한 하드 클레임(hard claim) 체크가 됩니다. MCP 서버는 `token.aud`을 자신의 표준(canonical) 리소스 URL과 비교하고, 불일치하는 경우 HTTP 401로 거부합니다. 이는 같은 신뢰 메쉬(trust mesh) 내의 다른 서버에 토큰을 재사용(replay)하려는 상류(upstream) MCP 서버(또는 한 서버용 토큰을 가진 악성 클라이언트)에 대한 유일한 방어책입니다.

이 강의에서는 각 공백을 표면(surface)의 구체적인 부분으로 매핑합니다. 메타데이터 문서는 HTTP 엔드포인트입니다. JWKS 캐시 갱신은 예약된 작업과 키-값 캐시입니다. JWT 검증은 리소스 서버가 도구를 디스패치(dispatch)하기 전에 실행하는 루틴입니다. 세 가지 역할을 분리하고, 각 역할이 소유한 체크만 강제하도록 유지하세요. 인증 서버는 키를 발급하고 회전하며, 리소스 서버는 캐시하고 검증하며, 클라이언트는 발견하고 등록합니다.

## 범위: 16강 이후의 프로덕션 강제 적용

[16강: MCP Security with OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md)는 인가 코드 상태 머신, PKCE, 보호된 리소스 검색, 리소스 지표 및 범위 결정을 담당합니다. 이 강의는 두 번째 OAuth 흐름을 정의하지 않습니다. 이러한 계약이 존재한 후 시작하며, 배포된 리소스 서버가 키 회전, 불투명 토큰 검증, 취소, 의존성 장애, 롤아웃 및 인시던트 대응 동안에도 이러한 계약을 계속 강제하는 방법을 묻습니다.

프로덕션 경계는 더 좁고 운영적입니다:

- JWT 경로는 매 요청마다 고정된 발급자, 알고리즘, 서명 키, 청중(audience), 시간 클레임 및 범위를 검증하며 JWKS를 안전하게 갱신합니다.
- 불투명 토큰 경로는 발급자의 인증된 인트로스펙션(introspection) 엔드포인트를 호출하여 반환된 활성 상태, 청중 또는 리소스, 만료, 주체 및 범위를 검증합니다.
- 취소 정책은 자격 증명이 얼마나 빨리 작동을 멈춰야 하는지, 그리고 어떤 캐시가 그 사실을 지연시킬 수 있는지를 정의합니다.
- 실패 정책은 검색, JWKS, 인트로스펙션 또는 취소 인프라가 사용 불가능할 때 어떤 일이 일어나는지 결정합니다.
- 증거는 토큰을 저장하지 않으면서, 어떤 발급자 메타데이터, 키 세트 또는 인트로스펙션 응답, 토큰 클레임, 정책 버전 및 거부 이유가 결과를 주도했는지 기록합니다.

이 구분은 강의를 조합 가능하게 유지합니다. 16강은 흐름을 증명합니다. 18강은 토큰이 실제 MCP 요청 경로에 도달한 후에도 신뢰할 수 있거나, 거부되는지 증명합니다.

## 개념

### RFC 8414 — OAuth 인가 서버 메타데이터

`/.well-known/oauth-authorization-server`에 있는 문서는 클라이언트가 필요로 하는 모든 것을 설명합니다:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCP 리소스 URL을 받은 클라이언트는 검색을 체이닝합니다: RFC 9728의 `oauth-protected-resource`(리소스 서버의 문서)가 발급자를 지정하고, `oauth-authorization-server`(이 RFC)가 모든 엔드포인트를 지정합니다. 클라이언트는 인가 URL을 하드 코딩하지 않습니다.

경로가 있는 리소스 식별자의 경우, 그 경로 앞에 well-known 세그먼트를 삽입합니다. 예를 들어, `https://mcp.example.com/team/server`는 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`에서 보호된 리소스 메타데이터를 해석합니다. 리소스 경로 뒤에 `/.well-known/...`를 붙이는 것은 올바르지 않습니다.

MCP용 IdP를 신뢰하기 전에 검증하는 계약은 다음과 같습니다:

- `code_challenge_methods_supported`에 `S256`가 포함됩니다 (PKCE는 RFC 7636에 따름). 사양은 명확합니다: 이 필드가 **없으면**, 인증 서버는 PKCE를 지원하지 않으며 클라이언트는 진행을 **거부해야(MUST)** 합니다.
- `grant_types_supported`에 `authorization_code`가 포함되며 `password`와 `implicit`를 거부합니다.
- 최소 하나의 등록 경로가 이용 가능합니다: `client_id_metadata_document_supported: true` (CIMD, 권장), 사전 등록된 클라이언트, 또는 `registration_endpoint` (폐기된 RFC 7591 호환성).
- `authorization_response_iss_parameter_supported`가 true이면, 클라이언트는 반환된 RFC 9207 `iss`을 요구하며, 리다이렉트 전에 기록된 발급자(issuer)와 정확히 비교합니다.
- OAuth 2.1에서 `response_types_supported`는 정확히 `["code"]`입니다.

`S256`가 없으면, MCP 서버는 이 IdP에 대해 배포를 거부합니다 — PKCE에 대한 저하 모드(degraded mode)는 없습니다. *두* 등록 경로 모두 광고되지 않고 사전 등록된 `client_id`가 없다면, 등록할 수도 없습니다; 배포 매니페스트가 잘못된 것이지, 코드가 잘못된 것이 아닙니다.

### RFC 9728 (요약) — 보호된 리소스 메타데이터

16강에서 RFC 9728을 다루었습니다. 프로덕션에서의 차이점: 이 문서는 클라이언트가 *이* MCP 서버가 신뢰하는 인증 서버를 찾는 유일한 위치입니다. 단일 MCP 서버는 여러 IdP의 토큰을 허용할 수 있습니다 (하나는 직원용, 하나는 파트너용). RFC 9728는 그 집합을 선언합니다; RFC 8414는 각 IdP가 무엇을 지원하는지 문서화합니다.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### 클라이언트 ID 메타데이터 문서 (권장 기본값)

CIMD는 등록을 *푸시(push)*에서 *풀(pull)*로 뒤집습니다. 클라이언트는 인증 서버가 `client_id`를 발급하도록 요청하는 대신, 자신이 제어하는 HTTPS URL을 `client_id`으로 **사용합니다**. 이 URL은 JSON 메타데이터 문서로 해석되며, 인증 서버는 OAuth 흐름 중 필요할 때 이를 가져옵니다. 신뢰는 DNS에 뿌리를 두고 있습니다: 서버 운영자가 `app.example.com`를 신뢰한다면, `https://app.example.com/client.json`에서 제공되는 클라이언트를 신뢰합니다. 등록 왕복(round-trip)이 없고, 고갈될 `client_id` 네임스페이스가 없으며, 동기화해야 할 서버별 상태가 없습니다.

클라이언트가 호스팅하는 메타데이터 문서:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

문서 내 `client_id` 값은 **반드시(MUST)** 제공된 URL과 같아야 합니다 (인증 서버가 이를 검증하며, 불일치는 거부됩니다). 인증 서버는 RFC 8414 메타데이터에서 `client_id_metadata_document_supported: true`로 지원 여부를 광고합니다.

현재 CIMD 계약에서는 `client_id`, `client_name` 및 비어 있지 않은 `redirect_uris` 배열이 필요합니다. 클라이언트 식별자는 경로가 포함된 절대 HTTPS URL입니다. `application_type`은 포함될 수 있지만 필수 CIMD 필드는 아닙니다. `application_type`에 대한 DCR 요구 사항을 선호되는 CIMD 경로에 복사하지 마세요.

명세가 명확하게 언급하는 두 가지 보안 사실은 다음과 같습니다:

- **SSRF.** 인증 서버는 공격자가 제공한 URL을 가져옵니다. 서버 측 요청 위조(SSRF)에 대해 방어해야 합니다(내부/관리 엔드포인트에 대한 가져오기 금지).
- **localhost 사칭.** CIMD만으로는 로컬 공격자가 합법적인 클라이언트의 메타데이터 URL을 주장하고 임의의 `localhost` 리디렉션을 바인딩하는 것을 막을 수 없습니다. 인증 서버는 동의 과정에서 리디렉션 URI 호스트 이름을 **MUST** 명확하게 표시해야 하며, `localhost` 전용 리디렉션에 대해 **SHOULD** 경고해야 합니다.

CIMD는 서버 측 상태가 필요하지 않으므로, DCR이 요구하는 방식으로 등록 기관(registrar)을 구축할 필요가 없습니다. 클라이언트 측에서는 읽기 전용으로 동작합니다: 정적 HTTPS 엔드포인트에서 메타데이터 문서를 제공하고 인증 서버가 이를 가져가도록 하세요.

인증 서버 운영자가 이미 클라이언트 식별자를 프로비저닝한 경우, 자동 등록을 시도하기 전에 해당 발급자 범위(issuer-scoped) 등록을 사용하세요. 그렇지 않은 경우 CIMD를 선호하세요. 발급자가 사전 등록이나 CIMD를 사용할 수 없는 경우에만 폐기된 DCR을 사용하세요.

### RFC 7591: 폐기된 호환성 등록

DCR은 2026-07-28 개정안에서 폐기되었습니다. CIMD를 소비할 수 없고 사전 등록이 실용적이지 않은 인증 서버에만 유지하세요. 호환성 클라이언트는 다음을 POST합니다:

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

서버는 `client_id`과 이후 업데이트용 `registration_access_token`로 응답합니다:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type`는 장식적인 요소가 아닙니다. 루프백 데스크톱 클라이언트는 `native`을 선언합니다. 서버 호스팅 클라이언트는 `web`을 선언하고 HTTPS 리디렉션 URI를 사용합니다. `token_endpoint_auth_method: none`은 공개 네이티브 클라이언트에 대한 올바른 기본값입니다. 이는 `client_id`만 받으며, PKCE가 소유 증명(proof-of-possession)을 제공합니다.

세 가지 프로덕션 함정:

- 등록 엔드포인트는 소스 IP별로 속도 제한(rate-limit)을 적용해야 합니다. 이를 적용하지 않으면 적대적인 행위자가 수백만 개의 가짜 등록을 스크립트로 생성하여 `client_id` 네임스페이스를 고갈시킬 수 있습니다. 등록 기관이 요청을 처리하기 전에 속도 제한 검사를 실행하세요.
- `software_statement` (클라이언트를 보증하는 서명된 JWT)는 일부 엔터프라이즈 IdP에서 요구됩니다. 이 강의의 목업(mock)은 이를 생략하지만, 프로덕션 환경에서는 localhost 리다이렉트 URI가 아닌 곳에서 온 서명되지 않은 등록을 거부하는 검증 단계를 구현합니다.
- `registration_access_token`는 평문이 아닌 해시로 저장해야 합니다. 이 토큰이 유출되면 공격자는 클라이언트의 리다이렉트 URI를 재작성할 수 있습니다.

### RFC 8707 (요약) — 리소스 지표(Resource Indicators)

16강에서 형태를 정의했습니다. 프로덕션 규칙은 다음과 같습니다: 모든 토큰 요청에 `resource=<canonical-mcp-url>`가 포함되며, MCP 서버는 모든 호출에서 `token.aud`이 자신의 리소스 URL과 일치하는지 검증합니다. 표준 URI(canonical URI)는 서버에 대한 *가장 구체적인* 식별자입니다: 소문자 스킴과 호스트를 사용하며, 프래그먼트는 없고, 관례적으로 후행 슬래시도 없습니다. 경로 구성 요소는 규칙에 의해 **제거되지** 않습니다 — 스펙은 개별 MCP 서버를 식별하는 데 필요할 경우 이를 유지합니다. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, `https://mcp.example.com/server/mcp`는 모두 유효한 표준 URI입니다. 서버마다 하나를 선택하고 `aud`를 정확히 그 값으로 고정하세요. (이 강의의 목업은 간결성을 위해 `https://notes.example.com`와 같은 맨 호스트(bare-host) 오디언스를 사용합니다; 하나의 오리진 아래 여러 MCP 서버를 공동 호스팅하는 배포 환경은 경로로 이를 구분합니다.)

### RFC 7636 (요약) — PKCE

PKCE는 OAuth 2.1에서 필수입니다. 이 강의에서의 인가 코드(authorization-code) 흐름은 항상 `code_challenge`와 `code_verifier`를 포함합니다. 서버는 검증자(verifier)가 없거나, 저장된 챌린지(challenge)와 해시가 일치하지 않는 검증자를 가진 토큰 요청을 거부합니다.

### MCP 2026-07-28 인가 프로필

현재 MCP 개정판은 OAuth 리소스 서버 경계를 유지하면서 MCP 트랜스포트는 상태 비저장(stateless)으로 만듭니다. 신원 결정을 캐싱할 프로토콜 세션이 없습니다. 따라서 인가 계층은 각 요청을 독립적으로 검증합니다:

- RFC 9728 보호된 리소스 메타데이터를 구현하고, 그 위치를 401 응답의 `WWW-Authenticate: Bearer resource_metadata="..."` 헤더 **또는** 잘 알려진 URI `/.well-known/oauth-protected-resource`를 통해 제공하세요 (SEP-985는 잘 알려진 URI 폴백을 위해 헤더를 선택 사항으로 만들었습니다). 메타데이터의 `authorization_servers` 필드는 **반드시(MUST)** 최소한 하나의 서버를 지정해야 합니다.
- **모든** 요청에서 `Authorization: Bearer ...`를 통해 토큰만 허용하세요 — 쿼리 문자열로는 절대 허용하지 않으며, 세션 시작 시에만 검증하지 마세요.
- 요청마다 `aud`, `iss`, `exp` 및 필수 범위를 검증합니다. 서버는 토큰이 해당 서버를 위해 발급되었는지(오디언스)를 **반드시(MUST)** 검증해야 합니다. `aud`가 없거나 일치하지 않으면 거부되며, 와일드카드(wildcard)로 취급되지 않습니다.
- 401/403 응답 시 `WWW-Authenticate: Bearer`를 반환하며, `error=...`, `resource_metadata="<PRM-URL>"` 매개변수(메타데이터 문서의 URL, *단순한 리소스 URL이 아님*), 그리고 `insufficient_scope`(403)에 `scope="..."`를 포함합니다. 참고: 매개변수는 `resource_metadata`로, 발견(discovery) 포인터입니다. 챌린지에는 `resource` 매개변수가 없습니다.
- 인증 서버 발견은 **둘 중 하나**인 RFC 8414 OAuth 메타데이터 **또는** OpenID Connect Discovery 1.0을 허용합니다. 클라이언트는 우선순위 순서로 두 well-known 접미사를 모두 시도해야 합니다.
- 클라이언트(서버가 아님)는 **믹스업(mix-up) 공격**에 대비합니다. 리다이렉트하기 전에 예상 `issuer`를 기록하고, 코드를 교환하기 전에 실제 인증 응답에서 반환된 `iss` 값을 검증합니다(RFC 9207). PKCE만으로는 믹스업을 막을 수 없습니다. 클라이언트가 유도된 토큰 엔드포인트에 `code_verifier`를 전달하기 때문입니다.
- 클라이언트 자격 증명은 하나의 인증 서버 발급자(issuer)에 속합니다. 발견 결과가 다른 발급자로 해석되면, 클라이언트는 이전 `client_id`, 등록 토큰, 또는 접근 토큰을 제시하는 대신 재등록(re-enroll)합니다.
- CIMD가 선호되는 등록 메커니즘입니다. DCR은 폐기(deprecated)되었습니다. 호환성 DCR 요청은 여전히 올바른 `application_type`를 선언합니다.

OAuth 2.1 초안이 기반(substrate)이며, RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD가 표면(surface)이고, MCP 사양이 프로필(profile)입니다.

### 배포 기능 체크리스트

벤더 기능 표는 빠르게 낡아집니다. 실제로 배포할 인증 서버가 반환하는 메타데이터를 검사하는 것이 좋습니다. 게이트는 기계적입니다:

| 검사 | 필수 결정 |
|---|---|
| 발견된 발급자 | 정책이 예상하는 정확한 HTTPS 발급자 |
| PKCE | `S256`가 광고됨. 그렇지 않으면 중단 |
| 등록 | CIMD 선호, 사전 등록 허용, DCR은 폐기된 호환성으로만 |
| 인증 응답 | 존재하거나 광고될 경우 RFC 9207 `iss` 검증 |
| 리소스 바인딩 | 토큰 요청이 `resource`를 포함하며, 리소스 서버는 일치하는 `aud`를 요구 |
| 자격 증명 저장 | 발급자별로 클라이언트 ID 및 등록 자격 증명을 키로 저장; 발급자 및 리소스별로 접근 토큰을 키로 저장 |
| DCR 호환성 | `native` 또는 `web`을 선언; 선언된 애플리케이션 유형에 맞지 않는 리다이렉트 URI를 거부 |

제품명이나 가격 티어로부터 지원 여부를 추론하지 마세요. 발견된 문서를 배포 증거에 기록하고, 필수 필드가 없으면 안전하게 실패(fail closed) 처리하세요.

### JWKS 갱신 패턴 (인증 서버에서 회전, 리소스 서버에서 갱신)

두 동사를 분리해서 유지하세요. 이 둘을 혼동하는 것은 실제 프로덕션 버그로 이어집니다:

- **회전(Rotate)**은 *인증 서버*가 수행하는 작업입니다: 새로운 서명 키를 생성하고 JWKS에 게시하며, 이후 오래된 키를 폐기합니다. 리소스 서버는 이 과정에 관여하지 않으며 수행할 수도 없습니다. 리소스 서버는 IdP의 개인 키를 보유하지 않기 때문입니다.
- **갱신(Refresh)**은 *리소스 서버*가 수행하는 작업입니다: 게시된 JWKS를 캐시에 다시 `GET`합니다. 리소스 서버가 수행하는 유일한 JWKS 작업이 이것입니다.

프로덕션 실패 모드는 오래된(stale) 캐시입니다. 예약된 갱신 작업과 키-값 캐시를 조합하여 해결하세요. 리소스 서버는 고정된 간격으로 `<issuer>/.well-known/jwks.json`을 가져와 `cache[issuer] = {keys, fetched_at}`을 덮어쓰는 작업(cron, 타이머 등 런타임이 제공하는 것)을 실행합니다. 검증기는 이 캐시에서 읽습니다. 캐시에 `kid`가 없는 토큰은 폴백(fall-back)으로 **하나의** 동기 갱신을 트리거한 후 재검증합니다. 이는 예약된 갱신과, 다음 예약된 갱신 전에 완전히 새로운 키로 서명된 토큰이 도착하는 키 겹침(key-overlap) 윈도우 두 가지 경우를 동시에 처리합니다.

폴백은 **반드시 재-fetch여야 하며, 회전(rotate)이어서는 안 됩니다**. 캐시 미스(cache-miss) 경로를 회전 및 생성(mint)에 연결하면 두 가지가 깨집니다: (1) 새 키를 생성하면 토큰과 *여전히* 일치하지 않는 `kid`이 생성되므로 조회가 실패합니다; (2) 공격자가 임의의 `kid` 값으로 토큰을 뿌리면 키 생성의 무한한 시리즈가 강제되어, 자초한 DoS가 됩니다. 재-fetch는 멱등성(idempotent)이 있으므로, 가짜 `kid`는 최대 한 번의 낭비된 fetch만 비용으로 발생합니다.

캐시 형태:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

두 개의 키가 동시에 존재하는 것이 정상 상태입니다. 인증 서버는 이전 키(`k_2026_03`)를 폐기하기 전에 다음 키(`k_2026_04`)를 도입하여 키를 회전시키므로, 이전 키로 발급된 토큰은 만료될 때까지 유효성을 유지합니다. 캐시는 두 키의 합집합을 저장하며, 검증기는 `kid`를 기준으로 선택합니다.

### 검증 루틴

MCP 서버는 모든 도구를 디스패치하기 전에 검증을 수행합니다. `code/main.py`가 사용하는 형태는 다음과 같습니다:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate`는 JWT를 디코딩하고, JWKS 캐시에서 서명 키를 해석하며(미스 발생 시 한 번 갱신), 서명을 검증한 후 `iss`를 허용 목록과 비교하고, `aud`를 이 서버의 표준 리소스와 비교하며, `exp` 및 필요한 스코프를 확인합니다. 첫 번째 실패 시 `WWW-Authenticate` 챌린지를 반환합니다. 이를 리소스 서버의 단일 루틴으로 유지하면 모든 진입점(모든 도구 호출, 모든 전송)이 동일한 검사를 거치게 되며, 검증을 거치지 않고 도구에 도달하는 경로가 존재하지 않습니다.

### 불투명 토큰은 추측이 아닌 인트로스펙션을 사용

모든 접근 토큰이 JWT인 것은 아닙니다. 발급자가 불투명 토큰을 문서화한 경우, 리소스 서버는 이를 신뢰할 수 있는 클레임으로 디코딩할 수 없습니다. 토큰을 인증된 백채널을 통해 발급자의 RFC 7662 인트로스펙션 엔드포인트로 전송하며, `active: true`, 예상 발급자 컨텍스트, 정확한 MCP 오디언스 또는 리소스, 만료되지 않은 시간 클레임, 그리고 구체적 도구가 요구하는 스코프를 확인해야 합니다.

발급자, 단방향 토큰 다이제스트, MCP 리소스를 기준으로 인트로스펙션을 캐시합니다. 명확한 토큰을 로그나 캐시 레이블로 절대 사용하지 마세요. 토큰 만료, 발급자 캐시 가이드라인, 배포의 폐기 신선도 목표 중 가장 빠른 시점을 기준으로 긍정적 캐시 항목의 유효 기간을 제한합니다. 새로 발급된 토큰이 거짓으로 비활성 상태로 남지 않도록 부정적 캐시 기간은 충분히 짧게 유지하세요. 불투명 토큰 문자열이 동일하더라도 한 리소스에 대한 결과가 다른 리소스를 승인할 수 없습니다.

공격자가 제어하는 토큰 내용에 따라 검증 모드를 선택하지 마세요. JWT와 인트로스펙션 동작을 검증된 발급자 메타데이터 및 배포 구성에 고정하세요. JWT 경로에서는 허용된 알고리즘과 신뢰할 수 있는 `jwks_uri`를 고정하며, 토큰 헤더가 선택한 키 URL이나 알고리즘을 절대 따르지 마세요.

### 폐기는 신선도 계약입니다

RFC 7009는 클라이언트가 인증 서버에 토큰을 폐기하도록 요청할 수 있게 합니다. 이 요청은 모든 리소스 서버가 이미 캐시한 사본을 삭제하지는 않습니다. 허용 가능한 최대 폐기 지연 시간을 정의하고 모든 캐시가 이를 준수하도록 하세요.

불투명 토큰 배포는 고위험 호출마다 인트로스펙션을 수행하거나 짧은 양의 캐시를 사용하여 더 엄격한 폐기를 달성할 수 있습니다. 자체 포함 JWT 배포는 일반적으로 짧은 액세스 토큰 수명, 발급자 전체 사건에 대한 키 은퇴, 그리고 긴급한 지역적 거부를 위한 선택적 주체, 세션, 또는 토큰 ID 거부 목록을 결합합니다. 서명된 JWT는 리소스 서버가 최신 외부 폐기 증거를 가지고 있지 않는 한 만료 전까지 암호화적으로 유효합니다.

로그아웃, 계정 비활성화, 동의 철회, 및 사건 대응은 서로 다른 트리거이지만, 하나의 측정 가능한 진술로 수렴해야 합니다: 선언된 폐기 윈도우가 지난 후 모든 복제본은 자격 증명을 거부합니다. 이 진술을 로드 밸런서를 통해 테스트하세요. 하나의 따뜻한 프로세스에만 대해 테스트하지 마세요.

### 의존성 실패는 선언된 결정이 필요합니다

예외 처리기 내에서 가용성 정책을 즉흥적으로 만들지 마세요.

| 실패 | 안전한 프로덕션 동작 |
|---|---|
| 예약된 JWKS 갱신이 실패하고 알려진 `kid`이 여전히 유효한 제한된 캐시에 남아 있음 | 선언된 stale-on-error 윈도우 내에서만 계속하고 저하된 상태의 건강 증거를 방출하세요 |
| 토큰에 알 수 없는 `kid`이 있고 허용된 한 번의 갱신이 실패함 | 거부하세요. 검증할 수 없는 서명을 절대 수락하지 마세요 |
| 인트로스펙션이 사용 불가능함 | 보호된 호출에 대해 fail closed 하세요. 네트워크 실패를 `active: true`로 변환하지 마세요 |
| 보호된 리소스 또는 발급자 메타데이터가 예상치 못하게 변경됨 | 새로운 등록 및 토큰 획득을 중단하세요. 제한된 사건 정책 하에 명시적으로 고정된 만료되지 않은 구성만 유지하세요 |
| 폐기 엔드포인트가 사용 불가능함 | 로그아웃 또는 폐기를 불완전한 것으로 보고하세요. 가능하면 자격 증명을 로컬에서 사용 불가능한 상태로 유지하고, 전역 폐기가 성공했다고 주장하지 마세요 |
| 시계 소스 또는 클레임 유형이 유효하지 않음 | 토큰이 통과할 때까지 스큐를 넓히지 말고 거부하세요 |

실패는 유효하지 않은 자격 증명과 별도로 분류합니다. 의존성 장애는 상태 및 재시도 정책이 있는 운영 오류입니다. 서명, 발급자, 대상, 만료일 또는 범위가 잘못된 것은 권한 거부입니다. 둘 다 도구 핸들러에 도달하지 않으며, 감사 증거에 토큰 내용이 유출되지 않도록 해야 합니다.

### 대상 재사용 시나리오 (액세스 토큰 권한 제한)

서버 A (`notes.example.com`)와 서버 B (`tasks.example.com`)는 모두 동일한 인증 서버에 등록합니다. 서버 A가 침해되었습니다. 공격자는 사용자의 노트 토큰을 가져와 서버 B에 재사용합니다.

서버 B의 검증기:

1. JWT를 디코딩하고, `kid`로 JWKS를 가져오며, 서명을 검증합니다.
2. `iss`을 보호된 리소스 메타데이터의 `authorization_servers`과 비교합니다. (통과 — 동일한 IdP입니다.)
3. `aud == "https://tasks.example.com"`을 확인합니다. (실패 — 토큰의 `aud`은 `https://notes.example.com`입니다.)
4. `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`와 함께 401을 반환합니다.

대상 클레임은 프로토콜 계층에서 이 공격에 대한 유일한 방어 수단입니다. 성능을 위해 이를 생략하는 것은 가장 흔한 프로덕션 실수입니다. 검증기는 세션 시작 시뿐만 아니라 모든 요청에서 실행되어야 합니다. 사양은 이를 **액세스 토큰 권한 제한**이라고 부릅니다: MCP 서버 `MUST`는 대상에 자신을 포함하지 않는 토큰을 거부해야 합니다.

> **명명 노트.** 사양은 *혼란된 대리인(confused deputy)*이라는 용어를 관련되지만 다른 문제를 위해 예약합니다: MCP 서버가 서명된 클라이언트 ID를 사용하여 서드파티 API의 OAuth **프록시**로 작동하며, 클라이언트별 사용자 동의를 얻지 않고 토큰을 전달하는 경우입니다. 대상 바인딩은 위의 재사용을 해결합니다. 혼란된 대리인 문제는 클라이언트별 동의 **그리고** 들어오는 토큰을 업스트림 API로 전달하지 않는 것(서버 `MUST`는 자체적인 별도의 업스트림 토큰을 가져야 함)으로 해결됩니다.

### 혼동 공격 (서버가 제공할 수 없는 클라이언트 측 방어)

클라이언트는 수명 동안 여러 인증 서버와 통신합니다. 악성 AS는 클라이언트가 정직한 AS의 인증 코드를 공격자의 토큰 엔드포인트에서 사용하도록 시도할 수 있습니다. 대상 바인딩은 여기서 도움이 되지 않습니다 — 공격은 토큰이 존재하기 전에 발생합니다. 방어는 클라이언트(RFC 9207)에 있습니다:

1. 리다이렉트하기 전에, 클라이언트는 검증된 AS 메타데이터에서 예상 `issuer`을 기록합니다.
2. 인증 응답에서 클라이언트는 코드를 어디로 보내기 전에, 기록된 발급자와 반환된 `iss` 매개변수를 단순 문자열 비교(정규화 없음)로 비교합니다.
3. 불일치(또는 AS가 `authorization_response_iss_parameter_supported`을 광고했는데 `iss`이 없는 경우) → 거부하며, `error` 필드를 표시하지도 않습니다.

PKCE만으로는 mix-up을 막을 수 없습니다. 클라이언트가 유도된 토큰 엔드포인트에 `code_verifier`을 전달하기 때문입니다. 이 때문에 사양은 PKCE 검증자와 `state` alongside 요청별로 발급자를 기록합니다.

### 실패 모드

- **오래된 JWKS.** AS가 키를 회전(rotating)하면 검증기가 유효한 토큰을 거부합니다. 해결책은 위의 cron-refresh + cache-miss-refetch 패턴입니다. JWKS는 refresh job 없이 캐시하지 마세요.
- **회전(rotating)을 폴백으로 사용.** cache-miss 경로를 재-fetch 대신 rotate-and-mint에 연결하는 것은 실제 버그입니다. 이는 누락된 `kid`을 생성하지 못하며, 공격자가 제어하는 `kid` 값을 키 생성 DoS로 만듭니다. 폴백은 멱등적인 `refresh-jwks`이어야 합니다.
- **누락된 `aud` 클레임.** 일부 IdP는 토큰 요청에 `resource`이 없으면 `aud`을 생략하는 것이 기본값입니다. 검증기는 `aud`이 누락된 토큰을 거부해야 하며, 누락을 와일드카드로 취급하지 않아야 합니다.
- **누락된 `iss` 체크를 통한 mix-up.** 리다이렉트 전에 기록한 발급자와 RFC 9207 `iss` 인증 응답 매개변수를 검증하지 않는 클라이언트는, 정직한 AS의 코드를 공격자의 토큰 엔드포인트에서 redeem하도록 유도될 수 있습니다. 이는 클라이언트 측 실패이며, 리소스 서버는 이를 보상할 수 없습니다.
- **스코프 업그레이드 경쟁.** 같은 사용자를 위한 두 개의 동시 step-up 플로우가 모두 성공하여 서로 다른 스코프를 가진 두 개의 접근 토큰을 생성할 수 있습니다. 검증기는 요청에 제시된 토큰을 사용해야 하며, "사용자의 현재 스코프"를 조회하지 않아야 합니다. 이는 TOCTOU 윈도우를 만듭니다.
- **등록 토큰 도난.** 유출된 `registration_access_token`은 공격자가 리다이렉트 URI를 재작성할 수 있게 합니다. 저장 시 해시하고, 모든 업데이트에서 클라이언트가 평문(cleartext)을 제시하도록 요구하며, 의심이 있으면 회전(rotating)하세요.
- **`iss`이 고정되지 않았습니다.** 임의의 `iss`을 허용하는 검증기는 공격자가 자체 인증 서버를 구축하고, 대상 청중을 위한 클라이언트를 등록하며, 토큰을 발급할 수 있게 합니다. 보호된 리소스 메타데이터의 `authorization_servers` 목록이 허용 목록입니다. 이를 강제하세요.
- **자격 증명 또는 토큰 캐시 충돌.** 등록을 리소스만으로 키를 설정하는 클라이언트는 한 인증 서버의 신원을 다른 서버에 제시할 수 있습니다. 접근 토큰을 발급자만으로 키를 설정하는 클라이언트는 토큰을 잘못된 청중에서 재생할 수 있습니다. 등록은 검증된 발급자로 키를 설정하고, 접근 토큰은 `(issuer, resource)`으로 키를 설정하며, 발급자가 변경될 때마다 재등록하세요.

```figure
t3-jwks-rotate
```

## 사용하기

`code/main.py`은 스탄다드 라이브러리 Python과 세 가지 역할인 `AuthorizationServer`, `ResourceServer`, `Client`을 사용하여 전체 프로덕션 흐름을 진행합니다. 흐름은 다음과 같습니다:

저장소 루트에서 실행하세요:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

첫 번째 명령은 발급자 바인딩 등록 및 토큰 검증
트랜스크립트를 출력합니다. 두 번째 명령은 18개 통과 검사를 보고합니다. 두 명령 모두
네트워크 리스너를 열지 않으며 자격 증명을 기록하지 않습니다.

1. 인증 서버는 `/.well-known/oauth-authorization-server`에 RFC 8414 메타데이터를 게시합니다.
2. MCP 클라이언트는 메타데이터 엔드포인트를 호출하고 등록 옵션(CIMD용 `client_id_metadata_document_supported`, DCR용 `registration_endpoint`) 및 `S256` PKCE 지원을 확인합니다.
3. 클라이언트는 발급자 범위 사전 등록을 확인하며, 없으면 HTTPS 클라이언트 ID 메타데이터 문서로 등록합니다. 폐기된 DCR은 별도로 테스트 가능한 호환성 방법으로 남아 있습니다.
4. 클라이언트는 검증된 발급자를 기록하고, S256 챌린지를 생성하며, 일회용 인증 코드와 `iss`을 수신하고, 반환된 발급자를 검증하며, 원본 검증자와 RFC 8707 `resource` 지표로 코드를 교환합니다.
5. MCP 클라이언트는 `Authorization: Bearer ...`을 사용하여 MCP 서버의 도구를 호출합니다.
6. MCP 서버는 `validate`을 실행하며, JWKS 캐시에서 서명 키를 해석합니다.
7. IdP가 키를 회전하면, 예약된 새로고침이 JWKS를 캐시로 다시 가져옵니다.
8. 다음 호출은 재시작 없이 새로고침된 키에 대해 검증하며, 이전 토큰은 겹치는 기간 동안에도 여전히 유효합니다.
9. 다른 MCP 리소스에 대한 청중 재생 시도는 `audience mismatch`과 `resource_metadata` 포인터가 포함된 401을 받습니다.

여기서 JWT는 공유 비밀 키를 사용하는 HS256을 사용하므로(이 강의는 표준 라이브러리만 사용하여 실행됩니다). 프로덕션에서는 위 JWKS 패턴을 사용하여 RS256 또는 EdDSA를 사용하며, 검증 로직은 그 외에는 동일합니다. IdP와 리소스 서버가 하나의 프로세스에 존재하므로, `refresh_jwks`는 인증 서버의 키 목록을 직접 읽습니다. 네트워크를 통해 전송될 때는 `jwks_uri`에 대한 HTTP `GET`가 됩니다.

## 출시하기

이 강의는 `outputs/skill-mcp-auth.md`를 생성합니다. MCP 서버 구성과 IdP 기능 집합이 주어지면, 이 스킬은 구축해야 할 인증 표면—보호된 리소스 메타데이터, 사용할 등록 경로(CIMD, 사전 등록 또는 DCR 폴백), JWKS 갱신 스케줄, 범위 매핑, 그리고 IdP가 전체 RFC 프로필을 지원하지 않을 때 적용해야 할 거부 규칙—를 출력합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 흐름을 추적하세요. 6단계에서 IdP가 키를 회전하는 방식, 스케줄된 `refresh_jwks`이 게시된 집합을 다시 가져오는 방식, 그리고 재시작 없이 이전 토큰(중복 기간)과 새 토큰이 모두 검증되는 방식을 확인하세요.

2. 보호된 리소스 메타데이터의 `authorization_servers` 목록에 새 IdP를 추가하세요. 새 IdP가 서명한 토큰을 발급하고 검증기가 이를 허용하는지 확인하세요. 목록에 없는 IdP가 서명한 토큰을 발급하고 검증기가 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`로 거부하는지 확인하세요.

3. `register_client`에 레지스트라가 요청을 수락하기 전에 실행되는 속도 제한 체크를 추가하세요. IP를 키로 사용하는 작은 dict에 소스 IP별 토큰 버킷을 유지하세요.

4. RFC 7591을 읽고 이 강의의 `/register` 핸들러가 검증하지 않는 두 필드를 식별하세요. 검증을 추가하세요. (힌트: `software_statement` 및 `redirect_uris` URI 스키마.)

5. 두 번째 인증 서버를 추가하세요. 클라이언트가 발급자 키 기반의 별도 등록을 저장하고, 첫 번째 발급자의 토큰이나 `client_id`를 재사용하지 않는지 확인하세요.

6. DoS 수정을 증명하세요. 검증기에 임의의 `kid`가 포함된 토큰을 보내고, `refresh_jwks`이 최대 한 번만 실행되며 인증 서버의 키 개수가 증가하지 않는지 확인하세요. 그런 다음 폴백을 의도적으로 회전 및 생성 방식으로 재연결하여 가짜 토큰마다 키 개수가 증가하는 것을 관찰하세요. 이후 재가져오기 방식으로 복원하세요.

7. `native` 및 `web` 클라이언트 모두로 폐기된 DCR을 연습하세요. HTTP 리디렉션 URI를 가진 웹 클라이언트와 정확한 루프백 리디렉션이 없는 네이티브 클라이언트가 거부되는지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ASM | "OAuth 메타데이터 문서" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "클라이언트 메타데이터 URL" | 클라이언트 ID 메타데이터 문서: `client_id`으로 사용되는 HTTPS URL; AS가 JSON을 가져옵니다. MCP 2026-07-28에서 선호되는 등록 방식 |
| DCR | "셀프 서비스 클라이언트 등록" | RFC 7591 `POST /register`; 현재 MCP에서는 폐기되었으며 호환성 유지를 위해만 남겨져 있습니다 |
| JWKS | "JWT 검증을 위한 공개 키" | JSON Web Key Set, `jwks_uri`에서 가져오며 `kid`으로 인덱싱됩니다 |
| 로테이션 vs 갱신 | "키 업데이트" | *로테이션* = AS가 서명 키를 발급/폐기하는 것; *갱신* = 리소스 서버가 게시된 키 집합을 다시 가져오는 것. 리소스 서버는 항상 갱신만 수행합니다 |
| 리소스 지표 | "청중 파라미터" | RFC 8707 `resource` 파라미터로 토큰을 하나의 서버에 고정합니다 |
| `aud` 클레임 | "청중(Audience)" | 검증자가 표준 리소스 URL과 비교하는 JWT 클레임 |
| 청중 재생 | "토큰 재생" | 서버 A를 위해 발급된 토큰을 서버 B에 제시하는 것; 청중 검증으로 방어됩니다 (사양: 접근 토큰 권한 제한) |
| 혼란스러운 대리인 | "프록시 토큰 오용" | 정적 클라이언트 ID를 가진 MCP 프록시가 클라이언트별 동의 없이 토큰을 전달하는 것; 청중 재생과 구별됩니다 |
| 혼동 공격 | "잘못된 토큰 엔드포인트" | 클라이언트가 정직한 AS의 코드를 공격자의 엔드포인트에서 교환하도록 유도되는 것; 클라이언트 측에서 RFC 9207 `iss`으로 방어됩니다 |
| `iss` 허용 목록 | "신뢰할 수 있는 인증 서버" | 보호된 리소스 메타데이터의 `authorization_servers`에 지정된 집합 |
| `resource_metadata` | "PRM 문서를 찾는 위치" | 401/403 응답에서 RFC 9728 메타데이터 URL을 지정하는 `WWW-Authenticate` 파라미터 |
| 공개 클라이언트 | "네이티브 또는 브라우저 클라이언트" | `client_secret`이 없는 OAuth 클라이언트; PKCE가 이를 보완합니다 |
| `WWW-Authenticate` | "401/403 응답 헤더" | 클라이언트 복구를 유도하는 `Bearer error=...` 지시문을 포함합니다 |

## 추가 읽기

- [MCP authorization specification (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - 현재 MCP 인증 프로필
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD, 발급자 검증, DCR 폐기 및 발급자 키 기반 자격 증명 변경
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 발견 계약
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (대체 경로)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 공개 클라이언트 소유 증명
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 청중 고정
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 리소스 서버 발견
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 혼동 공격(mix-up attacks)을 방어하는 `iss` 매개변수
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
