# MCP (모델 컨텍스트 프로토콜)(Model Context Protocol)

> MCP는 AI 호스트가 도구, 리소스, 프롬프트를 발견하고 호출하기 위해 하나의 프로토콜을 사용하도록 합니다. 2026-07-28 개정판은 이 프로토콜을 상태 비저장(stateless) 방식으로 변경했습니다. 기능 및 버전 컨텍스트는 연결에 바인딩된 핸드셰이크가 아니라 모든 요청과 함께 전달됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 11단계 · 09강 (함수 호출), 11단계 · 03강 (구조화된 출력)
**시간:** 약 75분

## 학습 목표

- MCP 호스트, 클라이언트, 서버, 전송, 서버 프리미티브를 구분해 보세요.
- MCP 2026-07-28이 요구하는 메타데이터를 포함하여 JSON-RPC 요청을 구축해 보세요.
- `server/discover`를 사용하여 버전, 식별자, 기능을 확인해 보세요.
- 도구, 리소스, 프롬프트에서 타입이 지정되고 캐시를 인식하는 결과를 반환해 보세요.
- 현대적인 상태 비저장 MCP가 핸드셰이크 시대 서버와 상호 운용되는 방식을 설명해 보세요.
- 서버에 안전한 상태, 전송, 승인 경계를 선택해 보세요.

## 문제점

애플리케이션은 데이터베이스 쿼리, 캘린더 작업, 파일 리더가 필요합니다. 공유 프로토콜이 없으면 모든 AI 호스트는 동일한 기능에 대해 자체적인 발견, 호출, 오류, 전송, 인증 글루 코드를 필요로 합니다.

MCP는 그 통합 매트릭스를 줄여줍니다. 서버는 표준 JSON-RPC 표면을 게시합니다. 호환되는 클라이언트는 서버 전용 어댑터 없이 표면을 발견하고, 모델이나 사용자에게 제시하고, 호출하고, 결과를 해석할 수 있습니다.

중요한 경계는 놓치기 쉽습니다. MCP는 통신을 표준화합니다. 모델이 어떤 도구를 호출해야 하는지 결정하지 않으며, 신뢰할 수 없는 콘텐츠를 안전하게 만들지 않으며, 상태 비저장 요청을 내구성 있는 애플리케이션 상태로 변환하지 않습니다. 호스트와 서버는 여전히 이러한 결정을 소유합니다.

## 개념

![MCP host, stateless request, and server primitives](../assets/mcp-architecture.svg)

### 세 가지 서버 프리미티브

1. **도구**는 호출 가능한 작업입니다. 각 도구에는 이름, 설명, JSON Schema 입력, 핸들러가 있습니다.
2. **리소스**는 클라이언트가 읽을 수 있는 이름이 지정된 URI 주소 콘텐츠입니다.
3. **프롬프트**는 호스트가 사용자에게 노출할 수 있는 재사용 가능한 템플릿입니다.

호스트는 AI 애플리케이션입니다. 호스트 내부의 MCP 클라이언트는 하나의 서버와 통신합니다. 전송 계층은 두 주체 간에 JSON-RPC 메시지를 전달합니다.

### 상태 비저장 요청이 핸드셰이크를 대체합니다

MCP 2026-07-28은 `initialize`과 `notifications/initialized`을 제거합니다. 또한 프로토콜 수준 세션도 제거합니다. 모든 요청은 `params._meta`에서 요청을 해석하는 데 필요한 컨텍스트를 포함합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

프로토콜 버전과 클라이언트 기능은 필수입니다. 클라이언트 식별은 권장됩니다. `_meta`이 없거나, 필수 필드가 없거나, 필수 필드의 타입이 올바르지 않으면 형식이 잘못된 것으로 간주되어 Invalid Params (`-32602`)를 반환합니다. 서버가 지원하지 않는 잘 형성된 버전 문자열은 `UnsupportedProtocolVersionError` (`-32022`)를 반환합니다. 서버는 이전 협상 기록을 복원하지 않아도 유효한 요청을 처리할 수 있습니다.

상태 비저장이 애플리케이션이 절대 상태를 유지할 수 없다는 의미는 아닙니다. 상태가 MCP 연결이나 `Mcp-Session-Id` 뒤에 숨겨져 있지 않다는 의미입니다. 워크플로우에 연속성이 필요한 경우, 서버는 불투명 핸들(opaque handle)을 발급하고 클라이언트는 이후 호출에서 일반 도구 인자로 그 핸들을 전달합니다. 권한은 모든 요청에서 여전히 검증되어야 합니다.

### 발견 및 버전 선택

모든 최신 서버는 `server/discover`을 구현합니다. 결과는 지원되는 버전, 기능 및 서버 식별을 표시합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

클라이언트는 다른 메서드를 직접 호출하고 버전 오류를 처리할 수 있지만, 발견(discovery)은 기능 표시와 버전 선택을 명시적으로 만듭니다. 지원되지 않는 버전은 코드 `-32022`와 함께 `UnsupportedProtocolVersionError`를 반환합니다. 그 데이터에는 `supported`, 서버 개정 배열, `requested`, 거부된 개정이 포함됩니다.

stdio에서는 이중 시대(dual-era) 클라이언트가 `server/discover`으로 탐색합니다. 발견 결과나 `UnsupportedProtocolVersionError`과 같은 인식된 최신 오류는 최신 서버를 식별합니다. 최신으로 인식되지 않는 모든 오류나 타임아웃은 2025-11-25 `initialize` 흐름으로 폴백하는 것을 허용합니다. 레거시 동작은 호환성 코드이며, 최신 기본값이 아닙니다.

### 결과는 명시적입니다

모든 핵심 2026-07-28 결과는 `resultType`를 포함합니다:

- `complete`는 작업이 완료되었음을 의미합니다.
- `input_required`는 서버가 Multi Round-Trip Requests 패턴을 통해 왕복 요청을 한 번 더 필요로 함을 의미합니다. 핵심 서버는 `tools/call`, `resources/read` 또는 `prompts/get`에서만 이를 반환할 수 있습니다.

클라이언트는 `resultType`이 생략된 레거시 결과를 완전한 것으로 취급해야 합니다.

서버는 모든 결과의 `_meta`에 `io.modelcontextprotocol/serverInfo`을 포함해야 합니다. 이 식별자는 자가 보고(self-reported)된 것으로, 표시, 로깅 및 디버깅 용도로만 사용되며 보안 결정에는 사용되지 않습니다.

목록 및 읽기 결과에도 `ttlMs`과 `cacheScope`이 포함됩니다. 결정적인 `tools/list` 순서와 신선도 힌트(freshness hint)를 통해 클라이언트는 발견(discovery)을 안전하게 캐시할 수 있으며 프롬프트 캐시 안정성이 향상됩니다. `cacheScope: public`은 공유 캐싱을 허용하며, `private`은 재사용을 호출 컨텍스트로 제한합니다.

### 전송 형식과 전송

MCP는 stdio 또는 Streamable HTTP를 통해 JSON-RPC 2.0을 사용합니다.

- 요청에는 `jsonrpc`, `id`, `method`, `params`가 있습니다.
- 응답에는 일치하는 `id`와 `result` 또는 `error` 중 하나가 있습니다.
- 알림(notification)에는 `id`이 없으며 응답을 기대하지 않습니다.

최신 Streamable HTTP는 POST를 허용하는 단일 엔드포인트를 노출합니다. 각 JSON-RPC 메시지는 자체 POST를 받습니다. 요청 POST는 하나의 JSON 객체 또는 최종 응답으로 끝나는 요청 범위(request-scoped)의 Server-Sent Events 스트림 중 하나를 받습니다. 승인된 알림 POST는 응답 본문 없이 HTTP 202를 받습니다. 이 핵심 개정판은 Streamable HTTP를 통한 클라이언트-서버 간 알림을 정의하지 않습니다.

2026-07-28에는 독립적인 MCP GET 스트림, DELETE 세션 엔드포인트, `Mcp-Session-Id` 또는 `Last-Event-ID` 재생(replay)이 없습니다. 장기적인 변경 알림은 응답이 SSE 스트림으로 열려 있는 `subscriptions/listen` POST를 사용합니다.

### 서버-initiated 요청이 없는 클라이언트 입력

이전 개정판에서는 서버가 스트림을 통해 `sampling/createMessage`, `roots/list` 또는 `elicitation/create`와 같은 요청을 보낼 수 있었습니다. 현재 프로토콜은 대신 MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR))을 사용합니다. 자격을 갖춘 도구 호출, 리소스 읽기 또는 프롬프트 가져오기는 `inputRequests` 또는 `requestState` 중 하나 이상을 포함하는 `resultType: input_required`을 반환합니다. 클라이언트는 요청된 입력을 수집하고, 새로운 JSON-RPC ID와 대응하는 `inputResponses`를 사용하여 원본 메서드를 재시도하며, 제공된 경우 정확한 `requestState`을 에코합니다. `inputRequests`이 없으면 재시도 시 `inputResponses`를 생략합니다.

Roots, Sampling, Logging은 여전히 기능하지만 더 이상 권장되지(deprecated)하므로, 새로운 구현에서는 채택하지 않아야 합니다. 기존 Roots 또는 Sampling 요청은 MRTR `inputRequests` 내부에서 전송되며, 독립적인 서버-클라이언트 JSON-RPC 요청으로는 전송되지 않습니다. 명시적인 파일 또는 디렉터리 매개변수, 리소스 URI, 서버 구성, 직접적인 모델 제공자 통합을 우선적으로 사용하세요. stdio 진단에는 stderr를, 프로덕션 텔레메트리에는 OpenTelemetry를 사용하세요.

```figure
mcp-nxm-collapse
```

## 구현하기

### 1단계: 서버 표면 등록

요청 계약이 변경되었지만 등록은 단순합니다:

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

`code/main.py`에 포함된 구현은 리소스와 프롬프트도 등록합니다. 프로토콜을 SDK에 위임하지 않고 각 엔벨로프(envelope)를 직접 볼 수 있도록 표준 라이브러리를 의도적으로 사용합니다.

### 2단계: 모든 요청에 메타데이터 첨부

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

이 메타데이터를 연결 객체에만 캐시하지 마세요. 서버는 모든 요청에서 이를 검증합니다.

### 3단계: 선택적으로 목록 나열 전에 발견

`server/discover`를 호출하여 지원되는 버전을 선택한 후 `tools/list`를 호출하세요. 이미 버전을 알고 있고 `-32022`를 처리할 수 있다면, 직접 `tools/list`를 호출하는 것도 유효합니다.

데모는 도구 목록을 이름 순서로 반환하며 `ttlMs`, `cacheScope`, `resultType` 및 서버 식별자를 첨부합니다. 도구 호출은 결과가 현재 상태에 의존할 수 있으므로 캐시 불가능한 완전한 결과를 반환합니다.

### 4단계: 동일한 요청을 HTTP로 매핑

원격 `tools/call` POST는 JSON-RPC 본문과 동일한 헤더를 포함합니다:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

`MCP-Protocol-Version` 헤더는 `_meta`의 버전과 일치해야 합니다. `Mcp-Method`는 모든 JSON-RPC 요청에 필수이며 `method`와 일치해야 합니다. `Mcp-Name`는 `tools/call`, `resources/read`, `prompts/get`에만 필수이며, 이 경우 도구 이름, 리소스 URI 또는 프롬프트 이름과 일치해야 합니다. 필수 헤더가 없거나 불일치하면 `-32020` 코드를 가진 HTTP 400이 `HeaderMismatch`와 함께 반환됩니다.

### 5단계: 프로토콜 상태 밖에서 안전성 강제

- 모든 HTTP 요청에서 권한 및 대상(audience)을 검증하세요.
- 로컬 서버는 localhost에 바인딩하고 Streamable HTTP에서 `Origin`를 검증하세요.
- 변경(mutating) 도구는 `destructiveHint: true`로 표시하고 호스트 승인을 요구하세요.
- Roots는 더 이상 권장되지 않으므로, 디렉터리 및 파일 범위를 명시적으로 전달하세요.
- 리소스와 도구 출력은 신뢰할 수 없는 데이터로 취급하세요.
- stdio에서는 stdout을 JSON-RPC 전용으로 예약하고, 진단 정보는 stderr에 기록하세요.

## 사용하기

해당 강의를 디렉터리에서 실행하세요:

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

첫 번째 줄은 `demo-server`가 `2026-07-28` 프로토콜에서 발견되었음을 보고해야 합니다. 그 후 `MCPClient.request`를 확인하세요: 모든 호출에 대해 `_meta`를 재구성합니다. 한 요청에서 메타데이터를 제거하고 서버가 이를 거부하는지 관찰하세요.

## 출시하기

`outputs/skill-mcp-server-designer.md`는 도메인을 상태 비저장 MCP 설계로 변환합니다. 이 설계의 승인 게이트는 발견 결과, 요청별 메타데이터 정책, 결정적인 캐시 인식 목록, 명시적인 상태 핸들, 전송 헤더, 인증 및 승인 규칙을 요구합니다.

## MCP 심층 탐구 계속하기

이 강의는 프로토콜 모델을 제공합니다. 13단계는 네 가지 생산 경계를 각각 빌드 및 검증 강의로 분리합니다:

1. [MCP Tool Contracts and Content](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md)는 폐쇄형 입력 스키마, 구조화된 콘텐츠, 라우팅 메타데이터, 불투명한 페이지네이션, 완료 인증, 그리고 프로토콜 및 도구 도메인 오류의 차이를 다룹니다.
2. [MCP Reliability, Cancellation, and Flow Control](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md)는 요청 취소, 내구성 있는 작업 취소, 마감 시간, 멱등성, 백프레셔, 프록시 버퍼링 및 재연결 동작을 다룹니다.
3. [MCP Registry Supply Chain, Admission, Drift, and Rollback](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md)는 네임스페이스 증명, 아티팩트 출처, 불변 핀, 라이브 드리프트, 레지스트리 상태, 수용 증거 및 롤백을 다룹니다.
4. [MCP Conformance Engineering](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md)는 골든 및 네거티브 와이어 트랜스크립트, 엄격한 버전 시대, SDK 차이점, 프록시 증거, 편집, 건강 게이트 및 릴리스 롤백을 다룹니다.

서버가 팀 또는 신뢰 경계를 넘을 때 순서대로 따르세요. 이 강의들은 “메서드가 작동한다”에서 “계약이 배포를 통해 안전하고 진단 가능하게 유지된다”로 나아갑니다.

## 연습 문제

1. `subtract` 도구를 추가하고 `tools/list`이 알파벳 순서로 유지되는지 확인하세요.
2. 프로토콜 버전 키를 제거하고 Invalid Params (`-32602`)를 검증하세요. 그 후 잘 형성되었지만 지원되지 않는 버전 `2025-11-25`을 보내고 `-32022`를 검증한 후, `requested`이 해당 개정을 에코하는지 확인하고 `supported`에서 선택하세요.
3. 생성 작업에 서버가 발급한 `draftId`을 추가한 후, 업데이트 시 이를 인자로 요구해 보세요. 이것이 프로토콜 세션이 아닌 애플리케이션 상태인 이유를 설명해 보세요.
4. 사용자 확인이 필요한 도구에서 `input_required`을 반환해 보세요. 서버-클라이언트 JSON-RPC 요청을 새로 만들지 말고, 새로운 ID, `inputResponses` 항목, 그리고 정확한 `requestState`를 사용하여 원본 호출을 재시도해 보세요.
5. 두 시대를 지원하는 stdio 클라이언트를 간략히 설계해 보세요. 결과나 인식된 최신 오류는 최신으로 취급하고, 인식되지 않은 오류나 시간 초과에만 `initialize`으로 폴백하는 것을 허용해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| MCP | "LLM용 도구 프로토콜" | 서버 발견, 도구, 리소스, 프롬프트 및 확장을 위한 JSON-RPC 프로토콜 |
| Host | "AI 앱" | 모델과 UI를 소유하며 하나 이상의 MCP 클라이언트를 마운트함 |
| Client | "연결자" | 호스트를 대신하여 하나의 서버와 MCP로 통신함 |
| 상태 비저장 MCP(Stateless MCP) | "세션 없음" | 모든 요청이 버전과 기능을 포함하며, 연결에 기반한 프로토콜 상태가 없음 |
| `server/discover` | "기능 탐지" | 버전, 기능 및 신원을 광고하는 필수 서버 메서드 |
| `resultType` | "결과 상태" | 결과를 `complete` 또는 `input_required`로 표시함 |
| 상태 핸들 | "워크플로우 ID" | 일반적인 인자로 전달되는 서버가 발급한 애플리케이션 식별자 |
| Streamable HTTP | "원격 전송" | JSON 또는 요청 범위 SSE 응답을 사용하는 하나의 POST 엔드포인트 |
| MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR)) | "요청 및 재시도" | 결과에 포함된 입력 요청, 이후 원본 작업의 재시도 |

## 추가 읽기

- [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
