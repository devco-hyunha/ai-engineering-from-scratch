# 상태 비저장 프로토콜의 MCP Apps

> 인터랙티브 결과도 여전히 MCP 도구 및 리소스 교환입니다. 2026-07-28 코어는 이 교환을 자기 완결적으로 만들며, Apps 확장 기능은 샌드박스화된 브라우저 표면을 추가합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계 · 07강 (MCP 서버), 13단계 · 10강 (리소스)
**시간:** 약 75분

## 학습 목표

- `server/discover` 및 요청별 확장 기능을 통해 MCP Apps를 광고해 보세요.
- 도구가 호출되기 전에 `ui://` 리소스를 도구 위에 선언해 보세요.
- 2026-07-28 상태 비저장 통신에서 완전한 도구 및 리소스 결과를 반환해 보세요.
- Apps `ui/initialize` 브리지 메시지를 제거된 MCP 코어 핸드셰이크와 분리해 보세요.
- 출처 검증, 샌드박스화, CSP 및 최소 권한 권한을 적용해 보세요.

## 문제점

텍스트 결과는 타임라인을 설명할 수 있습니다. 사용자가 필터링, 검사 또는 조작할 수 있는 타임라인을 제공하지는 못합니다.

MCP Apps는 선택적 확장 기능을 통해 표현 문제를 해결합니다. 도구 정의가 `ui://` 리소스를 가리킵니다. 호스트는 도구가 실행되기 전에 해당 리소스를 가져와 검토하고, 샌드박스화된 iframe에 렌더링하며, 모든 앱 동작을 JSON-RPC 브리지를 통해 중개할 수 있습니다.

코어 프로토콜이 2026-07-28에 변경되었습니다. 오래된 연결 수명 주기에 App을 감싸지 마세요:

- 코어 `initialize` 요청이나 `notifications/initialized` 알림이 없습니다.
- `Mcp-Session-Id` 헤더가 없습니다.
- 모든 요청은 `params._meta`에 프로토콜 버전 및 클라이언트 기능을 담습니다.
- 서버는 `server/discover`을 구현하여 클라이언트가 버전, 코어 기능 및 확장 기능을 검사할 수 있도록 합니다.
- 모든 성공적인 결과에는 `resultType` 판별자가 있습니다.
- 스트리밍 HTTP는 요청당 하나의 POST를 사용합니다. 최신 GET 및 DELETE 엔트리포인트는 405를 반환합니다.

Apps 브리지에는 여전히 `ui/initialize`라는 메서드가 있습니다. 이는 iframe postMessage 방언에 속합니다. 코어 MCP 세션을 재생성하지는 않습니다.

## 개념

### 두 프로토콜, 하나의 기능

레이어를 명시적으로 유지하세요:

1. MCP 코어는 `server/discover`, `tools/list`, `tools/call`, `resources/list`, `resources/read`를 포함합니다.
2. MCP Apps 확장 프로그램은 UI를 선언하고 iframe과 호스트 간의 브리지를 정의합니다.
3. 브라우저 샌드박스(Sandbox) 규칙은 UI가 접근할 수 있는 범위를 제한합니다.

확장 식별자는 `io.modelcontextprotocol/ui`입니다. 양쪽 피어(peer)가 모두 참여합니다. 클라이언트는 각 요청의 capabilities 객체 내부에 확장 지원 여부를 전송합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo`는 진단을 위해 권장됩니다. 이는 자가 보고된 데이터이며, 인증된 신분이 아닙니다.

### 렌더링 전에 발견하기

서버의 발견(discovery) 결과는 확장 프로그램을 광고합니다:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

서버는 발견(discovery)을 지원해야 합니다. 각 작업(action)은 자체 capabilities를 포함하므로, 클라이언트가 모든 작업 전에 발견을 호출하도록 강제되지는 않습니다.

### 도구 정의에 UI를 선언하기

최신 Apps 계약은 `tools/list`에서 도구와 UI를 바인딩합니다:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

이는 의도적으로 호출 전(pre-call) 메타데이터입니다. 호스트는 결과가 표시를 요청하기 전에 HTML을 사전 로드, 캐싱 및 보안 검토할 수 있습니다. 호환성 코드는 이전의 평평한(flat) 메타데이터 키를 허용할 수 있지만, 새로운 서버는 중첩된 `_meta.ui.resourceUri` 형식을 생성해야 합니다.

`tools/list`는 현재 코어에서 캐싱이 가능합니다. 결정적인 순서, `ttlMs`, `cacheScope`를 포함하세요. 가시적인 도구가 사용자나 토큰에 따라 달라지는 경우 `private`를 사용하세요.

### 데이터를 반환한 후, 호스트가 뷰를 바인딩하도록 허용하기

도구 호출은 일반적인 콘텐츠와 구조화된 데이터를 반환합니다:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

호스트는 이미 도구에 속한 뷰를 알고 있습니다. URI를 반복하기 위해 새로운 콘텐츠 블록을 발명하지 마세요.

### 앱을 리소스로 제공하기

서버는 발견(discovery)에서 `resources`를 광고하므로, 필수적인 `resources/list` 작업도 구현합니다. 결정적인 목록 항목에는 표준 URI, 안정적인 이름, 설명, MIME 유형이 포함됩니다. 목록 결과에는 결정적인 도구 목록과 마찬가지로 `resultType`, 서버 신원 메타데이터, `ttlMs`, `cacheScope`가 포함됩니다.

호스트는 `resources/read`를 전송합니다. Streamable HTTP에서는 요청이 다음과 같습니다:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

헤더 값과 JSON-RPC 본문은 일치해야 합니다. 불일치는 프로토콜 오류 `-32020`입니다.

결과에는 HTML 리소스와 캐시 힌트가 포함됩니다:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### 실행 가능한 콘텐츠로 UI 리소스를 캐시하기

App 리소스는 일반적인 텍스트와 상호 교환이 불가능합니다. 캐시 항목은 브리지 코드를 실행하고, 도구 데이터를 렌더링하며, 호스트가 중개하는 작업을 요청할 수 있습니다. `ui://` URI, 승인된 서버 식별자 및 버전, 리소스 콘텐츠 다이제스트, `cacheScope`이 비공개일 경우 인증 컨텍스트를 기준으로 키를 지정합니다. URI가 동일하더라도 HTML이나 정책 메타데이터가 다를 수 있으므로, 주체(principal) 간에 비공개 App 리소스를 재사용하지 마세요.

`ttlMs`이 만료되거나, 도구의 `_meta.ui.resourceUri` 바인딩이 변경되거나, 서버 버전 또는 승인된 디스크립터 핀(pin)이 변경되거나, 승인된 리소스 변경 구독이 URI를 지정할 때 캐시 항목을 무효화하세요. 다시 가져오기 전에 CSP 및 권한 검토를 재적용하고 재마운트하세요. 새로운 리소스 버전이 아직 로드되지 않았다는 이유만으로 오래된 iframe이 더 넓은 권한을 유지해서는 안 됩니다.

### 기능 정책(feature policy) 전에 와이어(wire) 모호성을 거부하기

검증에는 의도적인 순서가 있습니다. 먼저 JSON-RPC 형식을 검증하고, 문자열 프로토콜 메타데이터와 객체 클라이언트 기능(capability) 맵을 요구하세요. 그 다음 헤더와 본문을 비교하세요. 그 후에만 매칭된 프로토콜 버전이 지원되는지 결정하세요. 이 순서는 프록시와 서버가 서로 다른 요청을 해석하는 것을 방지합니다.

| 조건 | HTTP | JSON-RPC 오류 |
|-----------|------|----------------|
| 헤더와 본문의 버전, 메서드, 이름이 불일치 | 400 | `-32020` |
| 헤더와 본문이 지원되지 않는 버전으로 일치 | 400 | `-32022`, `data`이 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read`에 Apps 확장 기능이 없음 | 400 | `-32021`, `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| 메서드가 알려지지 않음 | 404 | `-32601` |

JSON-RPC 알림(notification)에는 `id`이 없으므로, 서버는 이에 대한 JSON-RPC 응답을 절대 생성하지 않습니다. 승인된 HTTP 알림은 빈 본문과 함께 202를 반환합니다. 오류는 HTTP 상태를 변경할 수 있지만, 알림에 대해 JSON-RPC 오류 본문을 생성할 수는 없습니다.

### 샌드박스는 경계이지, 신뢰 판정이 아닙니다

호스트가 iframe을 제어합니다. App은 호스트 쿠키, 로컬 저장소, 페이지 DOM을 직접 읽을 수 없습니다. 모든 특권 작업은 브리지를 통해 전달되어야 합니다.

이 기본값을 사용하세요:

- CSP 도메인 목록을 모두 비운 후, 앱이 필요한 출처만 추가하세요. fetch, XHR, WebSocket에는 `connectDomains`을, 스크립트, 스타일, 이미지, 폰트에는 `resourceDomains`을 사용하세요.
- 실용적인 경우 코드와 데이터를 번들링하세요.
- 가시적인 기능이 필요하지 않는 한 카메라, 마이크, 위치 권한을 요청하지 마세요.
- `postMessage`을 정확한 피어 출처에 고정하고, 다른 모든 출처에서 오는 이벤트를 거부하세요.
- 도구 인자, 도구 결과, 리소스 텍스트, 브리지 메시지를 신뢰할 수 없는 입력으로 취급하세요.
- 사용자 동의는 호스트에서 처리하세요. iframe은 자체적으로 중요한 작업을 승인할 수 없습니다.

튜토리얼의 고정된 `sandbox` 속성을 모든 호스트에 복사하지 마세요. 호스트는 앱의 출처 모델과 자체 격리 설계에 따라 플래그를 선택해야 합니다.

허용된 도메인은 여전히 유출 경로입니다. `connectDomains: ["https://api.example.com"]`은 앱 내부에서 실행되는 모든 스크립트가 허용된 데이터를 그곳으로 전송할 수 있음을 의미합니다. 정확한 출처 매칭은 목적지 혼동을 방지하지만, 페이로드가 적절한지 결정하지는 못합니다. 기본적으로 connect 접근을 비우고, iframe에 베어러 토큰을 배치하지 마세요. 실용적인 경우 좁은 연산을 호스트를 통해 프록시하고, 응답 및 요청 크기를 제한하며, 각 아웃바운드 요청을 유발한 사용자 작업을 감사하세요. `resourceDomains`을 `connectDomains`와 별도로 취급하세요. 폰트나 스크립트를 로드하는 권한이 임의의 데이터 업로드를 허용해서는 안 됩니다.

### Apps 브리지에는 자체적인 생명주기가 있습니다

Apps 브리지 `postMessage`를 사용하는 JSON-RPC 방언입니다. `ui/initialize` 및 `ui/*` 알림을 교환할 수 있으며, `tools/call`와 같은 핵심적인 메서드를 프록시할 수 있습니다.

View는 `appInfo`과 `appCapabilities` 객체를 포함하여 `ui/initialize`을 전송합니다. 호스트는 자신의 기능과 호스트 컨텍스트를 반환합니다. 이 응답 후에만 View는 `ui/notifications/initialized`을 전송합니다. 호스트는 View에 메시지를 보내기 전에 이 Apps 알림을 기다려야 합니다.

이 로컬 핸드셰이크는 하나의 iframe과 하나의 호스트 프레임 사이에 브리지를 생성합니다. MCP 프로토콜 버전을 협상하지 않으며, 서버 상태를 생성하지 않고, 전송 세션을 발급하지도 않습니다. 정확한 접두어에 주의하세요. 핵심 `notifications/initialized`은 제거되었지만, Apps `ui/notifications/initialized`은 남아 있습니다. 브리지된 도구 호출에 의해 생성된 핵심 요청은 새로운 JSON-RPC id와 전체 요청 메타데이터를 가진 새로운 독립적인 요청입니다.

### 호스트 컨텍스트, 작업 및 취소

브리지 초기화 후에도 호스트가 권한을 유지합니다. View는 호스트가 광고한 기능(capability)을 통해서만 도구 작업, 탐색, 클립보드 사용 또는 기타 특권 효과를 요청할 수 있습니다. 호스트는 타입이 지정된 요청, 현재 사용자, 대상 및 인수를 검증하고 승인 정책을 적용하며 요청을 거부할 수 있습니다. 버튼 클릭과 유효한 브리지 메시지는 의도를 나타내며, 둘 다 권한을 부여하지 않습니다.

테마, 크기 및 접근성을 일회성 렌더링 입력이 아닌 변경되는 호스트 컨텍스트로 취급하세요:

- 호스트가 제공한 색상 및 타이포그래피 토큰을 적용한 후, 테마 또는 대비 선호도가 변경될 때 반응하세요.
- View가 원하는 차원을 보고하도록 허용하되, 호스트가 iframe 크기를 제한하고 적용하여 콘텐츠가 레이아웃을 벗어나거나 기만적인 오버레이를 생성하지 못하도록 하세요.
- iframe 내에서 키보드 순서, 가시적 포커스, 접근성 이름, 스크린 리더 상태, 충분한 대비, 확대 및 모션 감소 동작을 유지하세요.
- 크기 조정 및 재렌더링 후 호스트 컨트롤과 View 컨트롤 간의 포커스 이동을 다시 테스트하세요.

App이 열려 있는 동안 사용자가 계정을 변경하거나, 정책이 변경되거나, 서버가 격리되거나, 호스트가 동의를 좁히면 기능이 취소될 수 있습니다. `ui/initialize` 동안뿐만 아니라 작업 시점에 기능 및 권한을 확인하세요. 취소 시, 대기 중인 특권 호출을 거부하고, 더 이상 정책에 부합하지 않는 네트워크 활동을 중지하며, 민감한 렌더링된 상태를 지우고, UI 리소스 자체가 더 이상 허용되지 않을 경우 재마운트하거나 텍스트로 폴백하세요. View는 거부를 정상적인 결과로 처리해야 하며, 호스트가 양보할 때까지 재시도하지 않아야 합니다.

### 폴백은 계약의 일부입니다

Apps-aware 서버는 UI 확장을 광고하지 않는 호스트에도 여전히 서비스를 제공할 수 있습니다:

- `tools/list`에서 `_meta.ui` 없이 동일한 도구를 반환하세요.
- `tools/call`에 대해 유용한 텍스트 결과를 유지하세요.
- 기능 누락 오류로 UI에 대한 `resources/read`을 거부하세요.
- 도구가 완료되었는지 결정할 때 iframe이 존재한다고 가정하지 마세요.

```figure
t3-ui-sandbox
```

## 구현하기

`code/main.py`는 SDK 없이 프로세스 내 작은 프로토콜 모델을 구축합니다. 현재 요청 엔벨로프 및 Streamable HTTP 라우팅 값을 검증하고, `server/discover`을 통해 Apps를 광고하며, 도구와 리소스를 나열하고, 도구를 실행하며, 자기 완결형 HTML 리소스를 제공합니다.

이 모델은 이미 파싱된 본문과 라우팅 헤더를 받습니다. 완전한 HTTP 어댑터가 아니며 `Content-Type`이나 `Accept`을 파싱하지 않습니다. `Content-Type: application/json`와 `application/json` 및 `text/event-stream`를 모두 포함하는 `Accept` 값이 필요한 완전한 Streamable HTTP 어댑터는 09강을 사용하세요.

실행해 보세요:

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

출력에서 네 가지를 확인하세요:

1. 모든 호출은 독립적입니다.
2. 모든 요청은 `_meta` 기능을 가집니다.
3. `resources/list`는 리소스 읽기 전에 안정적인 설명자를 반환합니다.
4. 모든 결과는 `resultType` 및 서버 식별자 메타데이터를 포함합니다.
5. 핵심 세션 식별자는 나타나지 않습니다.

## 사용하기

`server/discover`로 시작하세요. 서버 확장 맵에 `io.modelcontextprotocol/ui`이 나타나는지 확인하세요. 그런 다음 `tools/list`를 두 번 호출하세요. 한 번은 Apps 기능과 함께, 한 번은 Apps 기능 없이 호출합니다. 첫 번째 응답은 리소스를 선언합니다. 두 번째 응답은 사용 가능한 텍스트 전용 도구로 남습니다.

`ui://notes/timeline.html`를 읽어 보세요. HTML에서 `hostOrigin`과 `event.origin` 가드를 검색하세요. 이 두 줄은 브리지가 와일드카드 대상을 사용하지 않는다는 최소한의 가시적인 증거입니다.

## 출시하기

이 강의는 `outputs/skill-mcp-apps-spec.md`를 출시합니다. 프레임워크 코드를 작성하기 전에 App 계약을 검토하는 데 사용하세요. 이는 작성자가 현재 핵심 엔벨로프, 확장 협상, 폴백, UI 리소스, 캐시 정책, CSP, 권한, 브리지 메서드 및 동의 경계를 명시하도록 강제합니다.

## 연습 문제

1. 클라이언트 기능을 빈 확장 맵으로 변경하세요. `tools/list`가 도구를 유지하면서 UI 바인딩을 제거하는지 확인하세요.
2. 타임라인을 읽는 본문을 가진 `Mcp-Name: ui://notes/other.html`를 전송하세요. 오류 `-32020`가 발생하는지 확인하세요.
3. 리소스를 `cacheScope: private`로 변경하세요. 이를 정당화하는 사용자별 조건을 설명하세요.
4. 스크립트를 `https://static.example.com/app.js`로 이동하세요. 그 출처를 `resourceDomains`에 추가하고 새로운 공급망 위험을 설명하세요.
5. `notes_open` 도구를 추가하고 버튼 클릭을 호스트를 통해 라우팅하세요. 사용자 승인은 호스트에 유지하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| MCP Apps | MCP 호스트가 렌더링하는 인터랙티브 HTML을 위한 선택적 확장 |
| `io.modelcontextprotocol/ui` | 양쪽 피어가 광고하는 확장 식별자 |
| `ui://` | 앱의 UI 템플릿에 대한 리소스 스킴 |
| `text/html;profile=mcp-app` | MCP 앱 HTML의 MIME 타입 |
| `server/discover` | 프로토콜 및 기능 발견을 위한 현재 RPC |
| `resources/list` | 서버가 리소스를 광고할 때 필수적인 리소스 목록 메서드 |
| `resultType` | 현대적인 성공 결과에 필요한 판별자 |
| `ui/initialize` | 제거된 코어 초기화와 분리된 첫 Apps 브리지 요청 |
| `ui/notifications/initialized` | 호스트가 응답한 후 전송되는 Apps View 준비 완료 알림 |
| CSP | 스크립트, 스타일, 이미지 및 네트워크 출처를 제한하는 브라우저 정책 |
| 텍스트 폴백 | Apps 지원이 없는 호스트에 대해 유지되는 도구 동작 |

## 추가 읽기

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
