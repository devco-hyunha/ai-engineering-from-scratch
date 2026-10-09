# MCP 인증: CIMD, 발급자 바인딩, PKCE, 그리고 단계적 인증

> 원격 MCP 요청은 상태 비저장(stateless)이지만, 인증은 익명성이 아닙니다. 모든 자격 증명을 생성한 발급자(issuer)에 바인딩하고, 모든 토큰을 수신하는 리소스에 바인딩하세요.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계 · 09 (전송), 13단계 · 15 (보안)
**시간:** 약 90분

## 학습 목표

- 보호된 리소스 메타데이터를 통해 인증 서버를 발견해 보세요.
- 폐기된 동적 클라이언트 등록(Dynamic Client Registration)보다 클라이언트 ID 메타데이터 문서(Client ID Metadata Documents)를 우선적으로 사용하세요.
- DCR 호환 경로가 불가피할 경우 올바른 `application_type`을 선언하세요.
- 인증 응답 `iss`을 검증하고 발급자별로 자격 증명을 격리하세요.
- PKCE, 리소스 지표, 청중(audience) 검증, 점진적 스코프를 사용하세요.
- 프로토콜 세션 없이 승인된 MCP 2026-07-28 요청을 전송하세요.

## 문제점

원격 MCP 서버는 비공개 레코드를 읽거나, 외부 시스템을 쓰거나, 비용이 많이 드는 작업을 트리거할 수 있습니다. 인증(Authentication)은 자격 증명을 제시한 주체가 누구인지 알려줍니다. 인증(Authorization)은 또한 다음 질문에 답해야 합니다:

- 자격 증명을 발급한 인증 서버는 어디인가요?
- 이 토큰은 어떤 MCP 리소스를 위한 것인가요?
- 어떤 클라이언트와 리다이렉트 URI가 플로우를 완료했나요?
- 사용자가 승인한 작업은 무엇인가요?
- 이 정확한 요청이 해당 승인에 여전히 부합하나요?

2026-07-28 인증 프로필은 클라이언트 등록과 발급자 처리를 강화합니다. 클라이언트 ID 메타데이터 문서를 선호하고, 동적 클라이언트 등록을 폐기하며, DCR에서 올바른 `application_type`을 요구하고, RFC 9207 발급자 응답을 검증하며, 발급자 간 자격 증명 재사용을 금지합니다.

이러한 규칙은 상태 비저장 코어를 보완합니다. 코어 핸드셰이크나 `Mcp-Session-Id`을 복원하지는 않습니다.

## 개념

### 세 가지 역할 파악하기

- **MCP 클라이언트:** 리소스 소유자를 대신하여 요청을 전송합니다.
- **MCP 리소스 서버:** 접근 토큰을 수락하고 MCP 엔드포인트를 제공합니다.
- **인증 서버:** 리소스 소유자를 인증하고, 동의를 수집하며, 토큰을 발급합니다.

리소스 서버와 인증 서버는 함께 운영될 수 있지만, 식별자와 검증 책임은 분리해 유지해 주세요.

### 인증은 HTTP에 적용됩니다

MCP 인증 사양은 HTTP 기반 전송에 적용됩니다. 로컬 stdio 서버는 프로세스 및 운영체제 신뢰 경계 내에서 실행됩니다. 단순한 대칭성을 위해 stdio에 가짜 브라우저 OAuth 흐름을 추가하지 마세요.

원격 Streamable HTTP의 경우, 모든 요청에서 `Authorization` 헤더에 베어러 토큰을 전송하세요. URL에 절대 포함하지 마세요.

### 보호된 리소스 메타데이터로 시작하세요

리소스 서버는 RFC 9728 메타데이터를 게시합니다:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

클라이언트는 MCP 리소스 URL에서 시작하여 이 문서를 가져오고, 광고된 인증 서버를 선택한 후 해당 서버의 OAuth 또는 OpenID Connect 메타데이터를 가져옵니다.

RFC 9728 잘 알려진 URL을 구성할 때 리소스 경로를 보존하세요. 리소스 `https://notes.example.com/mcp`의 경우, 이 강의에서는 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`을 사용합니다. `/mcp` 접미사를 제거하면 같은 오리진에서 다른 보호된 리소스의 메타데이터가 선택될 수 있습니다.

호스트 이름으로 인증 서버를 추측하지 마세요. 검증되지 않은 오류 본문에서 발견된 발급자를 따르지 마세요. 클라이언트가 신뢰할 의사가 있는 발급자에 대한 정책을 유지하세요.

### 인증 서버 메타데이터를 검증하세요

메타데이터는 엔드포인트와 지원되는 제어 기능을 노출해야 합니다:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

PKCE에 S256을 요구하세요. 정확한 발급자 문자열을 기록하세요. 그 정확한 값이 등록 및 토큰 저장의 키가 됩니다.

### 등록 우선순위를 따르세요

클라이언트가 선택된 발급자와 명시적인 관계를 이미 가지고 있다면 사전 등록된 클라이언트 정보를 사용하세요. 그렇지 않다면 인증 서버가 지원을 광고할 때 Client ID Metadata Documents를 선호하세요. DCR은 폐기된 호환성 폴백으로만 사용하며, 그 중 어떤 메커니즘도 이용 불가능하다면 클라이언트 정보를 입력하라는 프롬프트를 표시하세요.

### Client ID Metadata Documents를 선호하세요

Client ID Metadata Document는 인증 서버에 클라이언트 식별자이자 그 메타데이터의 위치인 HTTPS URL을 제공합니다:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

인증 서버는 문서를 가져와 검증합니다. `client_id`는 경로를 포함한 HTTPS URL이어야 하며, 문서 내부의 값은 해당 URL과 정확히 일치해야 합니다. 필수 문서 필드는 `client_id`, `client_name`, `redirect_uris`입니다. `application_type`는 이 예시에 나타나지만 CIMD 요구 사항은 아닙니다. 의 새로운 필수 사용은 특히 DCR 경로에 해당합니다.

문서 가져오기를 SSRF에 민감한 작업으로 취급하세요. 목적지를 해석하고 검증하며, 루프백, 프라이빗, 링크 로컬 및 기타 허용되지 않는 주소를 거부하고, 리다이렉트와 DNS 변경 후 재검증하며, 리다이렉트, 바이트, 시간을 제한하고, JSON을 요구하며, 검증된 HTTP 캐시 제어에 따라 캐싱하세요. `client_name` 및 기타 표시 필드는 신뢰할 수 없는 텍스트로 취급하세요.

CIMD는 첫 접촉마다 새로운 동적 식별자를 발급할 필요성을 없애줍니다. 리다이렉트 URI 검증, 발급자 정책, 사용자 동의는 없애지 않습니다.

### DCR은 호환성 경로입니다

동적 클라이언트 등록은 오래된 인증 서버에 여전히 사용 가능하지만, 새로운 MCP 구현에서는 권장되지 않습니다.

DCR을 사용할 때 `application_type`를 선언하세요:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 데스크톱, 모바일, 명령줄 및 루프백 클라이언트는 `native`를 사용합니다.
- 원격 호스팅된 브라우저 애플리케이션은 `web`와 원격 HTTPS 리다이렉트를 사용합니다.

이 필드를 생략하면 OpenID Connect 등록 구현에서 `web`로 기본 설정될 수 있으며, 합법적인 루프백 리다이렉트가 실패할 수 있습니다.

DCR 코드를 명시적인 폴백 결정 뒤에 유지하세요. 임의의 CIMD 검증 실패 후 조용히 폴백하지 마세요. 이는 보안 실패를 더 약한 등록 경로로 바꿀 수 있습니다.

### 자격 증명을 발급자에 바인딩하세요

발급자가 발급한 등록 자료를 정확한 발급자 아래에 저장하세요:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

보호된 자원 발견이 `https://auth-one.example`에서 `https://auth-two.example`로 변경되면 신뢰를 재평가하세요. 첫 발급자의 클라이언트 시크릿, DCR 클라이언트 ID, 등록 접근 토큰, 리프레시 토큰, 접근 토큰을 두 번째 발급자에게 절대 보내지 마세요. 사전 등록 및 DCR 클라이언트는 새로운 발급자를 위해 발급된 자격 증명을 사용해야 합니다.

CIMD 클라이언트 ID는 인증 서버가 발급한 자격 증명이 아니라 자체 호스팅 HTTPS URL이므로 다릅니다. 동일한 CIMD URL은 이식성이 있습니다. 새로운 신뢰할 수 있는 발급자는 DCR 재등록 없이 문서를 가져와 검증합니다. 인증 응답과 토큰은 여전히 새로운 발급자 아래에서 검증되고 저장됩니다.

### PKCE를 사용한 인증 코드

인터랙티브 흐름은 다음과 같습니다:

1. 고엔트로피 `code_verifier`를 생성합니다.
2. S256 `code_challenge`를 파생합니다.
3. 정확한 `client_id`, `redirect_uri`, `scope`, `code_challenge`, `resource`를 포함하여 인증 요청을 전송합니다.
4. `code`와, 제공되는 경우 `iss`를 포함하는 인증 응답을 수신합니다.
5. 응답 필드를 사용하기 전에 기록된 발급자와 정확히 일치하는지 `iss`를 검증합니다.
6. `code_verifier`, 동일한 리다이렉트 URI, 동일한 `resource`를 사용하여 코드를 교환합니다.
7. 결과 토큰을 `(issuer, resource)` 아래에 저장합니다.

RFC 8707의 `resource` 매개변수는 인증 및 토큰 요청 모두에 나타납니다. 이는 표준 MCP 서버 URI를 식별합니다.

### `iss`를 정확히 검증합니다

RFC 9207은 한 발급자의 인증 응답이 다른 발급자의 응답과 혼동되는 것을 방지합니다.

`iss`가 존재하는 경우, 대소문자 병합, 후행 슬래시 변경, 기본 포트 제거, 퍼센트 인코딩 정규화 없이 기록된 발급자와 비교합니다. 불일치 시, 코드에 대해 조치를 취하지 말고 해당 응답에서 공격자가 제어하는 오류 세부 정보를 표시하지도 마세요.

`iss`를 포함하는 인증 서버는 `authorization_response_iss_parameter_supported: true`를 광고합니다. 현재 클라이언트는 해당 광고가 누락된 경우에도 존재하는 `iss`를 여전히 검증합니다.

### MCP 서버에서 청중(audience)을 검증합니다

리소스 서버는 자신에 대해 발급된 토큰만 허용합니다:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

유효하지 않은, 만료된, 잘못된 발급자, 또는 잘못된 청중(audience) 토큰은 401을 받습니다. MCP 서버는 다른 서비스를 위한 토큰을 허용하거나 전달해서는 안 됩니다.

### 현재 필요한 최소한의 스코프를 요청합니다

지금 필요한 스코프부터 시작합니다. 이후 도구가 더 많은 스코프를 요구하면 서버는 권위 있는 스코프 도전(challenge)과 함께 403을 반환합니다:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

클라이언트는 새로운 권한을 설명하고, 동의를 얻고, 결합된 스코프 집합으로 새로운 인증 흐름을 수행하며, 새로운 JSON-RPC id로 MCP 요청을 재시도합니다.

도전받은 스코프가 `scopes_supported`의 하위 집합이라고 가정하지 마세요. 도전은 현재 작업에 대해 권위적입니다.

### 인증과 상태 비저장 MCP 통신

인증된 도구 호출은 여전히 완전한 현재 요청 엔벨로프를 포함합니다:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

토큰은 주체를 인증합니다. 요청 메타데이터는 프로토콜 동작을 협상합니다. 둘 중 하나가 다른 하나를 대체하지 않습니다.

통신을 고정된 순서로 검증하세요: JSON-RPC 및 메타데이터 타입, 헤더와 본문 일치, 그 다음 프로토콜 지원. 라우팅 또는 버전 헤더 불일치는 `-32020`와 함께 HTTP 400을 반환합니다. 헤더와 본문이 지원되지 않는 버전으로 일치하면 `-32022`와 `data`가 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`인 HTTP 400을 반환하세요. 알 수 없는 메서드는 `-32601`와 함께 HTTP 404를 반환합니다.

401 유효하지 않은 토큰 및 403 불충분한 스코프를 포함한 모든 요청 오류는 원본 요청 `id`를 가진 JSON-RPC 오류 엔벨로프입니다. 구조화된 복구 정보는 선택적 오류 `data`에 속하며, `WWW-Authenticate`는 HTTP 응답 헤더로 남습니다. 알림에는 `id`가 없으므로 JSON-RPC 본문을 받지 않습니다. 승인된 HTTP 알림은 빈 본문으로 202를 반환합니다.

서버는 `server/discover`를 구현하고 도구를 광고하므로, 필수 `tools/list` 메서드도 구현합니다. 도구 설명자는 안정적인 이름, 설명 및 객체 루트 `inputSchema` 값을 가집니다. 목록은 결정적이며 `resultType`, 서버 식별 메타데이터, 제한된 `ttlMs` 및 `cacheScope`를 반환합니다. 발견 및 사용자 독립적인 도구 목록은 인증 전에 이용 가능할 수 있습니다. 주체에 따라 변동되는 경우 일반적인 정책과 비공개 캐싱을 적용하세요.

### 토큰 패스스루 없음

MCP 서버는 클라이언트의 MCP 접근 토큰을 다운스트림 API로 전달해서는 안 됩니다. 올바른 오디언스를 가진 별도의 다운스트림 토큰을 획득하거나 명시적인 토큰 교환 설계를 사용하세요. 오디언스 검증은 서비스가 타인을 위해 발행된 토큰을 거부할 때만 작동합니다.

### 리프레시 토큰

리프레시 토큰은 선택 사항입니다. 발급될 경우, 기발자(issuer)와 리소스를 기준으로 키를 지정하여 기밀적으로 저장하세요. 존재한다고 가정하지 마세요. 인증 서버가 로테이션을 지원할 때 로테이션을 수행하고, 무효화된 값의 재사용을 감지하세요.

```figure
t3-scope-stepup
```

## **구현하기**

`code/main.py`는 프로세스 내 프로토콜 및 인증 시뮬레이터입니다. 보호된 리소스 발견, 인증 서버 메타데이터, CIMD 등록, 버전 게이트가 적용된 DCR 폴백, 애플리케이션 유형 검사, PKCE, 기발자 검증, 리소스 바운드 토큰, 스코프 스텝업, `server/discover`, `tools/list`, 그리고 상태 비저장 도구 요청을 구현합니다.

모델은 파싱된 요청 본문과 라우팅 헤더를 받습니다. 완전한 HTTP 어댑터가 아니며 `Content-Type` 또는 `Accept`을 파싱하지 않습니다. `Content-Type: application/json`와 `application/json` 및 `text/event-stream`를 모두 포함하는 `Accept` 값이 필요한 09강의 Streamable HTTP 어댑터에 연결하세요.

실행하세요:

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

출력은 발견(discovery) 우선, CIMD 등록, 일반적인 읽기, 두 개의 분리된 스코프 스텝업, 그리고 기발자 키 기반 자격 증명 저장을 보여줍니다.

## **사용하기**

시뮬레이터 객체를 프로덕션 컴포넌트에 매핑하세요:

- `ResourceServer.protected_resource_metadata`는 RFC 9728 엔드포인트가 됩니다.
- `AuthorizationServer.metadata`는 RFC 8414 또는 OpenID Connect 발견(discovery)이 됩니다.
- `Client.enroll`는 CIMD 해석(resolution)과 명시적인 DCR 호환성 분기가 됩니다.
- 기발자가 발급한 클라이언트 자격 증명과 `tokens_by_issuer_resource`는 암호화된 레코드가 됩니다. CIMD URL은 이식 가능(portable)할 수 있지만, 인증 결과는 기발자 바운드(issuer-bound) 상태로 유지됩니다.
- `ResourceServer.handle`는 모든 요청 오류를 일치하는 JSON-RPC 엔벨로프에 유지하면서, 디스패치 전에 현재 MCP 헤더, 토큰 및 도구 스코프를 검증하는 미들웨어가 됩니다.

## **출시하기**

이 강의는 `outputs/skill-oauth-scope-planner.md`를 출시합니다. 이제 등록 우선순위, 기발자 바운드 자격 증명 저장, 애플리케이션 유형, PKCE, 리소스 지표, 스코프 챌린지, 그리고 현재 상태 비저장 요청 경계를 설계합니다.

## **연습 문제**

1. 리프레시 토큰 로테이션을 추가하고 이전 리프레시 토큰의 재사용을 거부하세요.
2. 기발자 허용 목록(allowlist)을 추가하세요. 기발자 변경 시, 이식 가능한 CIMD URL만 재사용하고 모든 이전 기발자 발급 자격 증명 및 토큰을 거부하세요.
3. 인증 코드에 만료 기간을 추가하고, 늦은 교환이 실패하는지 확인해 보세요.
4. 원격 HTTPS 리디렉션을 사용하는 웹 클라이언트 변형을 구축하고, 그 DCR 메타데이터를 네이티브 클라이언트와 비교해 보세요.
5. 동일한 발급자 아래에 두 번째 리소스를 추가하세요. 첫 번째 리소스에서 두 번째 리소스의 액세스 토큰이 사용될 수 없는지 확인하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| 보호 리소스 메타데이터 | 리소스와 인증 서버를 식별하는 RFC 9728 문서 |
| CIMD | URL이 OAuth 클라이언트 식별자인 HTTPS 메타데이터 문서 |
| DCR | 호환성을 위해 유지되는 폐기된 동적 클라이언트 등록 |
| `application_type` | `native` 또는 `web`, 리디렉션 URI 규칙을 검증하는 데 사용 |
| PKCE | 가로채진 인증 코드를 보호하는 검증자 및 S256 챌린지 |
| `iss` | RFC 9207 인증 응답 발급자 식별자 |
| 리소스 지표 | 토큰 요청을 MCP 리소스에 바인딩하는 RFC 8707 매개변수 |
| 청중(Audience) | 토큰이 유효한 리소스 |
| 스텝업(Step-up) | 추가적인 현재 작업 범위에 대한 새로운 동의 및 토큰 발급 |
| 발급자 바인딩 자격 증명 | 정확한 인증 서버 발급자로 격리된 등록 및 토큰 기록 |

## 추가 읽기

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
