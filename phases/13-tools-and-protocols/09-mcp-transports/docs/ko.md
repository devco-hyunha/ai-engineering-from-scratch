# MCP 전송: stdio 및 상태 비저장 스트리밍 HTTP

> 전송은 MCP 메시지를 전달합니다. 누락된 프로토콜 상태를 제공하지는 않습니다. `2026-07-28`에서는 로컬 stdio와 원격 스트리밍 HTTP 모두 자기 서술형 요청을 전달합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 13단계, 07강 및 08강
**시간:** 약 65분

## 학습 목표

- 로컬 자식 프로세스에는 stdio를, 네트워크 서비스에는 스트리밍 HTTP를 선택해 보세요.
- 최신 단일 엔드포인트, POST 전용 스트리밍 HTTP 계약을 구현해 보세요.
- JSON-RPC 본문에 대해 MCP 버전, 메서드 및 이름 헤더를 미러링하고 검증해 보세요.
- 요청 범위 SSE와 장기 `subscriptions/listen` 스트림을 올바르게 전달해 보세요.
- 레거시 동작을 최신으로 제시하지 않으면서 세션 기반 및 레거시 HTTP+SSE 배포를 마이그레이션해 보세요.

## 문제점

이전 스트리밍 HTTP 개정판은 프로토콜 협상과 연결 및 세션 동작을 결합했습니다. 서버는 `Mcp-Session-Id`을 생성하고, 독립적인 GET 스트림을 노출하며, 세션 종료에 DELETE를 허용하고, `Last-Event-ID`로 SSE를 재개할 수 있었습니다.

MCP `2026-07-28`은 최신 와이어에서 이러한 메커니즘을 제거합니다. 모든 요청은 프로토콜 버전과 클라이언트 기능이 요청 본문에 담겨 있으므로, 모든 정상 워커에 도달할 수 있습니다. HTTP 헤더는 라우팅 및 정책을 위해 선택된 필드를 미러링하지만, 서버는 실행 전에 본문을 기준으로 헤더를 검증합니다.

그 결과 확장하기가 더 쉬워지고, 추론하기도 더 쉬워집니다. 또한 서버가 2025년 전송을 최신으로 가르친다면, 이는 잘못된 장애 및 보안 모델을 가르치는 것입니다.

## 개념

### stdio

stdio 바인딩은 클라이언트가 실행한 하위 프로세스를 위한 것입니다:

- 클라이언트는 stdin에 한 줄당 하나의 UTF-8 JSON-RPC 메시지를 씁니다.
- 서버는 stdout에 한 줄당 하나의 UTF-8 JSON-RPC 메시지를 씁니다.
- 서버는 stderr에 진단 정보를 씁니다.
- 서버는 stdin EOF 시 즉시 종료합니다.
- 모든 최신 요청은 `params._meta`에 버전과 클라이언트 기능을 담습니다.

이 프로세스는 여러 호출 동안 지속될 수 있지만, 현대적인 프로토콜 세션은 아닙니다. 프로세스가 예상치 못하게 종료되면 진행 중인 요청이 손실됩니다. 프로세스를 재시작하고, 재발견하고, 목록을 다시 나열하고, 구독을 다시 열고, 새로운 요청 ID로 안전한 작업을 재시도해 보세요.

### 2026-07-28의 스트리밍 HTTP

현대적인 서버는 POST를 허용하는 `/mcp`과 같은 하나의 MCP 엔드포인트를 노출합니다.

모든 JSON-RPC 요청 또는 알림은 새로운 HTTP POST입니다. 본문에는 하나의 JSON-RPC 메시지가 포함됩니다. 클라이언트는 서버에 JSON-RPC 응답을 보내지 않습니다.

요청에 대해 서버는 다음 중 하나를 반환합니다:

- 하나의 JSON-RPC 응답을 포함하는 `Content-Type: application/json`; 또는
- 해당 요청과 관련된 알림을 포함하고, 그 뒤에 최종 JSON-RPC 응답이 따라오는 `Content-Type: text/event-stream`.

수락된 알림에 대해 서버는 본문이 없는 `202 Accepted`을 반환합니다.

클라이언트는 두 가지 응답 유형을 모두 광고합니다:

```http
Accept: application/json, text/event-stream
```

### POST 전용은 POST 전용을 의미합니다

현대적인 스트리밍 HTTP에는 독립적인 GET 스트림과 DELETE 세션 엔드포인트가 없습니다.

- `GET /mcp`은 `405 Method Not Allowed`을 반환합니다.
- `DELETE /mcp`은 `405 Method Not Allowed`을 반환합니다.
- `Mcp-Session-Id`은 무시되며, 생성되거나 에코되지 않습니다.
- 현대적인 스트림은 재개할 수 없으므로 `Last-Event-ID`은 무시됩니다.

요청 범위 스트림이 최종 응답 전에 끊어지면, 클라이언트는 해당 진행 중인 요청을 잃게 됩니다. 재시도가 안전할 때 새로운 JSON-RPC ID로 새로운 요청을 발행할 수 있습니다. 스트림 재개를 시도해서는 안 됩니다.

### 출처 검증

서버는 DNS 리바인딩을 방지하기 위해 들어오는 연결에서 `Origin`을 검증합니다. 헤더가 존재하고 명시적으로 허용되지 않은 경우 `403 Forbidden`을 반환합니다. 비브라우저 클라이언트는 `Origin`을 생략할 수 있으며, 공식 전송 규칙이 이를 허용합니다.

로컬 서버는 모든 인터페이스가 아닌 `127.0.0.1`에 바인드해야 합니다. 네트워크 서비스는 모든 요청에 대해 인증 및 권한 부여가 필요합니다. 출처 검증은 인증이 아닙니다.

정규화된 구성 후 정확한 출처 매칭을 사용하세요. `origin.startswith("https://trusted.example")`과 같은 접두어 체크는 공격자가 제어하는 접미어를 허용할 수 있으므로 안전하지 않습니다.

### 필수 HTTP 메타데이터 헤더

모든 현대적인 POST 요청은 다음을 포함합니다:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

헤더 규칙:

- `MCP-Protocol-Version`은 필수이며 `params._meta.io.modelcontextprotocol/protocolVersion`과 동일해야 합니다.
- `Mcp-Method`은 필수이며 JSON-RPC `method`과 동일해야 합니다.
- `Mcp-Name`은 `tools/call`, `resources/read`, `prompts/get`에 필수입니다.
- `Mcp-Name`은 `params.name`과 동일하거나, `resources/read`의 경우 `params.uri`이어야 합니다.
- 헤더 이름은 대소문자를 구분하지 않지만, 헤더 값은 대소문자를 구분합니다.

안전하지 않거나 비ASCII `Mcp-Name` 값은 정확한 UTF-8 Base64 센티넬을 사용합니다:

```text
=?base64?{Base64EncodedValue}?=
```

서버는 본문을 비교하기 전에 해당 값을 디코딩합니다.

미러링된 헤더가 누락, 형식 오류, 불일치인 경우 HTTP `400`과 JSON-RPC 코드 `-32020`을 반환합니다. 헤더와 본문이 서버가 지원하지 않는 버전에 대해 일치하는 경우, HTTP `400`과 `-32022` 및 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`와 같은 정확한 오류 데이터를 반환합니다.

알 수 없는 최신 메서드는 HTTP `404`과 JSON-RPC `-32601`을 반환합니다. JSON-RPC 본문은 듀얼 에라(dual-era) 클라이언트가 최신 오류와 레거시 엔드포인트 미스를 구분하는 데 사용하므로 중요합니다.

### 요청 범위 SSE

서버는 하나의 장기 실행 요청에 대해 SSE를 선택할 수 있습니다:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

서버는 이 스트림에서 독립적인 JSON-RPC 요청을 보내서는 안 됩니다. 샘플링, 유도, 루트 상호작용은 MRTR (다중 왕복 요청)(Multi Round-Trip Request) 결과를 사용합니다. 응답 스트림을 닫으면 해당 요청이 취소됩니다.

재전송을 위해 SSE 이벤트 ID를 추가하지 마세요. `Last-Event-ID` 재개는 최신 개정판의 일부가 아닙니다.

### 장기적인 변경은 구독/청취(subscriptions/listen)를 사용

변경 알림은 독립적인 GET이 아닌, 클라이언트가 연 요청을 사용합니다:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

POST 응답은 장기적인 SSE 스트림입니다. 첫 번째 프로토콜 메시지는 `notifications/subscriptions/acknowledged`입니다. 승인, 모든 변경 알림, 최종 결과는 `_meta`에 `io.modelcontextprotocol/subscriptionId`을 포함하며, 이는 청취 요청 ID와 동일합니다. 서버는 유지 신호(keepalive)로 SSE 주석을 방출할 수 있습니다. 스트림이 끊어지면 클라이언트는 새로운 요청 ID로 `subscriptions/listen`을 재발행하고 영향을 받은 데이터를 다시 가져옵니다.

`resources/subscribe`과 `resources/unsubscribe`은 레거시 에라에 속합니다. 최신 연결에서는 사용하지 마세요.

### 명시적 애플리케이션 상태

프로토콜 세션을 제거해도 상태가 있는 워크플로우를 금지하지는 않습니다. 서버는 불투명한 상태 핸들을 생성하고 일반적인 도구 결과로 반환할 수 있습니다. 클라이언트는 이후 호출 시 그 핸들을 명시적인 인자로 전달합니다.

핸들을 인증된 주체(principal)에 바인딩하고, 추측 불가능하게 만들며, 만료시키고, 모든 사용을 승인합니다. 이는 상태를 전송 계층의 친화성(affinity)에 숨기지 않고 애플리케이션 계층에서 가시화합니다.

숨겨진 복제본(replica) 상태로 인한 실패는 기계적입니다:

1. 요청 A가 복제본 1에 도달하여 해당 프로세스의 메모리에 초안을 생성합니다.
2. 구현이 연결을 초안 식별자로 가정하므로, 응답은 초안 핸들을 반환하지 않습니다.
3. 요청 B는 새로운 POST 요청이며, 레플리카 2에 도달합니다.
4. 복제본 2는 유효한 프로토콜 메타데이터를 가지고 있지만, 초안을 이름으로 지정하거나 로드할 방법이 없으므로 워크플로우가 실패하거나 잘못된 로컬 객체를 읽게 됩니다.
5. 스티키 라우팅은 재시작, 롤아웃, 재스케줄링, 페일오버로 다음 요청이 이동하기 전까지 증상을 해결한 것처럼 보입니다.

올바른 경계는 두 부분으로 구성됩니다. 프로토콜 컨텍스트는 각 요청에 남아 있습니다. 내구성 있는 애플리케이션 상태는 서버가 발급한 핸들 아래에 공유 저장소에 저장되며, 이 핸들은 클라이언트에 반환됩니다. 다음 호출이 해당 핸들을 제공하면, 모든 레플리카가 동일한 레코드를 로드하고, 인증은 레코드를 인증된 주체와 테넌트에 바인딩합니다. 레플리카 메모리는 레코드를 캐시할 수 있지만, 정확성을 위해 필요한 유일한 사본이 될 수는 없습니다.

수명(lifetime)에 따라 상태 메커니즘을 선택하세요. 요청 지역 변수는 단일 호출에 사용할 수 있습니다. 짧은 MRTR (다중 왕복 요청)(Multi Round-Trip Request) 연속 처리는 무결성 보호 `requestState`를 사용할 수 있습니다. 초안이나 내구성 있는 실행(Durable Execution) 작업은 명시적 핸들 및 공유 지속성, 만료, 동시성 제어, 멱등성(Idempotency)이 필요합니다. 이러한 객체 중 어느 것도 MCP (모델 컨텍스트 프로토콜)(MCP (Model Context Protocol)) 프로토콜 세션이 아닙니다.

### HTTP 이중 시대 호환성

현대 및 레거시 서버를 모두 지원하는 클라이언트는 먼저 현대 POST를 시도합니다. HTTP `400`, `404`, `405`를 수신하면 응답 본문을 검사합니다:

- 인정된 최신 JSON-RPC 오류는 서버가 최신임을 증명합니다. 요청을 수정하거나 광고된 버전을 재시도하세요. 다운그레이드하지 마세요.
- 빈 본문이나 인식할 수 없는 응답은 레거시 HTTP+SSE 서버를 나타낼 수 있습니다. 이때만 이전 GET 엔드포인트를 시도하고 레거시 `endpoint` 이벤트를 기대해 보세요.

서버는 마이그레이션 기간 동안 현대적 메타데이터를 현대적 POST 전용 구현으로 라우팅하고, 이전 클라이언트를 위해 별도의 레거시 엔드포인트를 유지함으로써 두 시대를 모두 지원할 수 있습니다. `2026-07-28`의 일부로 레거시 GET, DELETE, 세션 ID, 재생(replay) 동작을 설명하지 마세요.

```figure
tp-transport-handshake
```

## 사용하기

`code/main.py`는 Python 표준 라이브러리를 사용하여 유한한 현대적 Streamable HTTP 서버를 구현합니다. Origin 및 미러링된 헤더를 검증하고, 제거된 세션 헤더는 무시하며, 일반 호출에 대해 JSON을 반환하고, 유한한 `subscriptions/listen` SSE 스트림을 시연합니다.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

프로브는 다음을 확인합니다:

- 유효하지 않은 Origin이 거부됩니다;
- 세션 ID 없이 발견(discovery)이 성공합니다;
- `Mcp-Session-Id` 및 `Last-Event-ID`는 무시됩니다;
- 헤더 불일치는 `-32020`를 반환합니다;
- 지원되지 않는 버전은 정확한 `supported` 및 `requested` 데이터와 함께 `-32022`를 반환합니다;
- 수락된 ID 없는 알림은 본문 없이 HTTP `202`를 반환합니다;
- GET 및 DELETE는 `405`를 반환합니다;
- `subscriptions/listen`는 POST 응답 스트림이며, 그 승인, 알림 및 최종 결과는 구독 ID를 포함합니다.

## 출시하기

이 강의는 `outputs/skill-mcp-transport-migrator.md`를 출시합니다. 현대적 프로토콜 세션을 제거하고, 헤더-본문 검증을 추가하며, 독립적인 GET을 `subscriptions/listen`로 대체하고, 레거시 브리지를 명확하게 분리된 상태로 유지합니다.

## 연습 문제

1. POST에서 `Mcp-Method`를 제거하세요. HTTP `400` 및 오류 `-32020`를 확인하세요.
2. 헤더와 본문 버전 `2027-01-01`가 일치하도록 전송하세요. HTTP `400`, 오류 `-32022` 및 정확한 데이터 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`를 확인하세요.
3. 비 ASCII 리소스 URI에 대해 Base64 센티넬 `Mcp-Name`를 전송하세요. 디코딩된 값이 `params.uri`와 비교되는지 확인하세요.
4. 최종 응답 전에 유한한 listen 스트림을 끊으세요. 새로운 JSON-RPC ID로 재발행하고 도구를 다시 가져오세요.
5. ping 도구에 명시적인 워크플로우 핸들(handle)을 추가하세요. 연결 친화성(connection affinity)을 사용하지 않고 인증 주체(authorization subject)에 바인딩하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| stdio | 클라이언트가 실행한 하위 프로세스를 통해 전송되는 줄바꿈 구분 JSON-RPC |
| Streamable HTTP | 각 최신 메시지가 새 POST인 단일 엔드포인트 |
| 요청 범위 SSE | 관련 알림과 최종 응답을 포함하는 POST 응답 스트림 |
| `subscriptions/listen` | 선택한 변경 알림을 위한 장기 지속 POST 요청 |
| 헤더 불일치 | 미러링된 헤더가 본문과 일치하지 않을 때의 HTTP `400` 및 JSON-RPC `-32020` |
| Origin 검증 | 들어오는 연결에 대한 DNS 리바인딩 방어이며, 인증이 아님 |
| 명시적 상태 핸들 | 숨겨진 세션 상태 대신 일반 인자로 전달되는 애플리케이션 토큰 |
| 레거시 브리지 | 호환성을 위해 유지되는 이전 시대의 별도 동작 |

## 추가 읽기

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
