# MCP 리소스와 프롬프트: 상태 비저장 서버를 위한 주소 지정 가능한 컨텍스트

> 도구는 연산을 수행합니다. 리소스는 주소 지정 가능한 콘텐츠를 노출합니다. 프롬프트는 사용자가 선택한 메시지 템플릿을 패키징합니다. 좋은 MCP 서버는 이러한 계약을 분리하고 예측 가능하게 유지합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계, 07강 (MCP 서버 구축하기), 13단계, 09강 (MCP 전송)
**시간:** 약 60분

## 학습 목표

- 소비자의 의도에 따라 도구, 리소스, 프롬프트 중 하나를 선택하세요.
- 필수 `server/discover`를 통해 리소스 및 프롬프트 인터페이스를 광고하세요.
- 결정적인 `resources/list` 및 `prompts/list` 결과를 구축하세요.
- 사용자별 데이터를 유출하지 않고 `ttlMs` 및 `cacheScope`를 적용하세요.
- 유효하지 않거나 알 수 없는 리소스 URI에 대해 JSON-RPC 오류 `-32602`를 반환하세요.
- `subscriptions/listen` POST 응답 스트림을 열고 구독 ID로 모든 이벤트를 상관관계 짓세요.
- 리소스 콘텐츠와 프롬프트 템플릿을 신뢰할 수 없는 서버 출력으로 취급하세요.

## 소비자부터 시작하기

MCP를 오용하는 가장 쉬운 방법은 구현 코드부터 시작하는 것입니다. 함수가 익숙하기 때문에 데이터베이스 쿼리가 도구가 됩니다. 파일에 저장되어 있기 때문에 재사용 가능한 워크플로우가 리소스가 됩니다. 호스트가 주입할 수 있기 때문에 프롬프트가 숨겨진 정책이 됩니다.

누가 선택하는지, 그리고 그들이 무엇을 기대하는지부터 시작하세요.

| 원시 요소 | 주요 의도 | 선택 주체 | 일반적인 결과 |
|---|---|---|---|
| 도구 | 연산 수행 | 모델 또는 애플리케이션 | 구조화된 작업 결과 |
| 리소스 | URI에서 콘텐츠 읽기 | 호스트, 애플리케이션 또는 사용자 | 텍스트 또는 바이너리 콘텐츠 |
| 프롬프트 | 재사용 가능한 메시지 워크플로우 시작 | 호스트 UI를 통한 사용자 | 하나 이상의 프롬프트 메시지 |

`notes://note-1`의 메모는 주소 지정 가능한 콘텐츠이므로 리소스입니다. `delete_note`는 상태를 변경하므로 도구입니다. `review_note`는 사용자가 준비된 리뷰 워크플로우를 선택하므로 프롬프트입니다.

완전해 보이려고 하나의 연산을 세 가지 모두로 노출하지 마세요. 각 추가 인터페이스는 발견, 인증, 캐싱, 오류 처리, 테스트 및 문서화가 필요합니다.

## 2026-07-28 상태 비저장 Envelope

이 강의는 MCP 프로토콜 개정 `2026-07-28`을 대상으로 합니다. 이 프로필에는 초기화 핸드셰이크나 프로토콜 세션이 없습니다. 모든 요청은 예약된 `_meta` 키에 프로토콜 버전과 클라이언트 기능을 포함합니다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

서버는 `server/discover`을 구현해야 합니다. 그 결과는 지원되는 버전, 리소스 및 프롬프트 기능, 구현 식별자, 캐시 힌트를 알립니다. 클라이언트는 다른 메서드를 직접 호출할 수 있지만, 발견(discovery)은 UI를 구축하기 전에 하나의 안정적인 스냅샷을 제공합니다.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

정상적인 결과는 `"resultType": "complete"`을 선언합니다. 응답 `_meta`은 `io.modelcontextprotocol/serverInfo`로 서빙하는 구현을 식별합니다. 이 정보는 진단에 유용합니다. 인증 식별자는 아닙니다. 지원되지 않는 개정을 포함하는 요청은 요청된 개정과 서버가 지원하는 개정을 모두 포함하는 `-32022`을 반환합니다.

상태 비저장 계약은 설계 직관을 바꿉니다. 목록은 하나의 연결에서 이전 호출에 의존할 수 없습니다. 자격 증명이 요청 입력이므로 가시적인 세트가 변경될 수 있지만, 연결 기록은 변경되어서는 안 됩니다.

## 리소스는 안정적인 URI 계약입니다

리소스는 URI로 식별되는 콘텐츠입니다. 핸들러보다 URI를 먼저 설계하세요.

좋은 URI 속성:

- 북마크하거나 요청 간에 전달할 수 있을 정도로 안정적입니다.
- 서버의 도메인에 네임스페이스가 지정되어 있습니다.
- 프로세스 ID나 연결과 독립적입니다.
- 저장소 접근 전에 검증됩니다.
- 모든 읽기에서 권한이 부여됩니다.

`notes://note-1`은 네임스페이스가 명시적이므로 `note-1`보다 좋습니다. 파일 서버는 `file://` URI를 사용할 수 있지만, 심링크와 상대 세그먼트를 해석한 후 설정된 디렉터리 경계를 확인해야 합니다.

`resources/list`은 호출자에게 현재 가시적인 리소스를 반환합니다. URI와 같은 안정적인 키로 정렬하세요. 결정적인 순서는 노이즈가 많은 캐시 미스, 변경되는 스냅샷, 새로고침 간에 점프하는 호스트 UI를 방지합니다.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read`은 하나 이상의 콘텐츠 항목을 반환합니다. 알 수 없는 URI는 성공적인 빈 읽기가 아닙니다. 현재 Resources 사양은 유효하지 않거나 알 수 없는 리소스 URI를 JSON-RPC 유효하지 않은 매개변수, 코드 `-32602`에 할당합니다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

이 구별을 통해 클라이언트는 부재와 유효한 빈 문서를 분리할 수 있습니다. 또한 더 넓은 조회로 우연히 폴백되는 것을 방지합니다.

### 리소스 템플릿

리소스 템플릿은 매개변수화된 URI의 계열을 설명합니다. 모든 구체적 항목을 나열하는 것이 비용이 많이 들거나 무한할 때 하나를 사용하세요. 예를 들어, `notes://projects/{project}/decisions/{decision}`는 모든 결정을 반환하지 않고도 클라이언트가 유효한 주소를 형성하는 방법을 알려줍니다.

템플릿은 검증을 약화시키지 않습니다. 변수를 파싱하고, 인증을 적용하며, 길이 및 문자 제한을 강제하고, 타입이 지정된 매개변수로 저장소 쿼리를 구성하세요. 임의의 URI 꼬리를 파일 시스템 경로 또는 데이터베이스 문장에 절대 연결하지 마세요.

### 콘텐츠는 신뢰된 지시가 아닙니다

리소스 텍스트에는 프롬프트 인젝션(Prompt Injection), 비밀, 오해를 불러일으키는 명령, 또는 형식이 잘못된 마크업이 포함될 수 있습니다. 호스트는 출처를 보존하고 리소스 콘텐츠를 데이터로 취급해야 합니다. 서버는 콘텐츠 크기를 제한하고, 정확한 MIME 타입을 반환하며, 호출자가 접근할 수 없는 필드를 편집하고, 관련 없는 레코드를 반환하지 않아야 합니다.

## 프롬프트는 사용자가 제어하는 템플릿입니다

MCP 프롬프트는 명시적인 사용자 선택을 위해 설계되었습니다. 호스트는 이를 슬래시 명령, 메뉴 항목, 또는 워크플로 버튼으로 렌더링할 수 있습니다. 프로토콜은 하나의 UI를 요구하지 않습니다.

`prompts/list`는 동일한 요청 인증에 대해 결정적이어야 합니다. 각 프롬프트는 안정적인 이름, 유용한 설명, 그리고 `prompts/get` 전에 호스트가 입력을 수집할 수 있게 하는 매개변수 선언이 필요합니다.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get`는 매개변수를 메시지로 해석합니다. 이는 호스트의 시스템 프롬프트(System Prompt)를 대체하지 않습니다. 호스트는 반환된 메시지가 모델 컨텍스트에 어떻게 들어가는지 결정하며, 자체 신뢰된 정책을 더 높은 우선순위로 유지합니다.

서버 경계에서 프롬프트 매개변수를 검증하세요. 프롬프트 URI는 직접 리소스 읽기와 동일한 인증 검사를 통과해야 합니다. 프롬프트가 리소스 접근을 우회하는 사이드 채널이 되지 않도록 하세요.

## 캐시 힌트는 정확성의 일부입니다

`ttlMs`는 클라이언트에게 결과가 얼마나 오래 재사용될 수 있는지 알려줍니다. `cacheScope`는 캐시된 값을 누가 공유할 수 있는지 설명합니다.

| 범위 | 의미 | 일반적인 사용 |
|---|---|---|
| `public` | 인증이 허용될 경우 사용자 간에 재사용 가능 | 공개 프롬프트 카탈로그 |
| `private` | 요청한 사용자 또는 자격 증명 컨텍스트에 바인딩 | 사용자 소유의 노트 콘텐츠 |

데이터의 변경 속도와 데이터가 오래되었을 때의 피해에 따라 TTL을 선택해 보세요. 공개 프롬프트 카탈로그에는 5분이 적합할 수 있습니다. 비공개 노트 읽기에는 1분을 사용할 수 있습니다.

MCP는 `public`과 `private`만 `cacheScope` 값으로 정의합니다. 비밀을 포함하거나 빠르게 변경되는 결과를 반환할 때는 `ttlMs: 0`와 함께 `cacheScope: "private"`을 반환한 후, 호스트 캐시 정책에서 더 엄격한 no-store 규칙을 적용해 보세요. `no-store` 자체는 MCP `cacheScope` 값이 아닙니다.

캐시 힌트는 인증을 대체하지 않습니다. 캐시 키는 테넌트, 사용자, 범위, 로케일, 페이지네이션 커서 등 가시성을 변경하는 모든 요청 차원을 포함해야 합니다. 공유 캐시가 이러한 차원을 안전하게 표현할 수 없다면, TTL을 0으로 설정하고 호스트 수준의 no-store 정책과 함께 `private`을 사용해 보세요.

## 구독은 클라이언트가 연 응답 스트림을 사용합니다

현대적인 구독 패턴은 이전 `resources/subscribe` RPC와 오래된 HTTP GET 이벤트 엔드포인트를 대체합니다.

클라이언트는 `subscriptions/listen`을 일반적인 JSON-RPC 요청으로 전송합니다. Streamable HTTP에서는 POST로 전송되며, 응답은 SSE 스트림으로 열려 있는 상태로 유지됩니다. `notifications` 객체는 허용 목록입니다. 서버는 요청되지 않은 알림 유형을 전달해서는 안 됩니다.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

요청 ID는 구독 ID입니다. 요청된 이벤트가 발생하기 전에 서버는 `notifications/subscriptions/acknowledged`을 전송합니다. 이 필터는 서버가 수락한 하위 집합만 포함합니다.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

해당 스트림의 모든 후속 이벤트는 동일한 메타데이터를 포함합니다.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

알림은 리소스가 변경되었음을 알립니다. 클라이언트는 현재 인증을 적용하여 `resources/read`을 통해 다시 읽어 들입니다. 이벤트가 새 문서를 포함한다고 가정하지 않습니다.

여러 구독이 하나의 stdio 채널을 공유할 수 있습니다. 구독 ID를 통해 클라이언트는 이를 복선화(demultiplex)할 수 있습니다. HTTP에서는 응답 스트림을 닫으면 구독이 취소됩니다. 스트림을 정상적으로 종료하는 서버는 원래 요청과 상관된 최종 `resultType: "complete"` 응답을 반환합니다.

구독 스트림을 프로토콜 세션으로 사용하지 마세요. 후속 읽기는 여전히 완전한 요청이며, 건강한 서버 인스턴스에 도달할 수 있습니다.

```figure
t3-primitive-sort
```

## 인터랙티브 랩

프로젝트 트래커에서 다섯 가지 기능(이슈 상세, 이슈 생성, 스프린트 리뷰 템플릿, 프로젝트 정책, 이슈 닫기)을 그림을 사용하여 분류해 보세요. 그런 다음 어떤 목록을 공개 캐싱할 수 있는지, 어떤 읽기 작업은 비공개로 유지해야 하는지, 어떤 자원에 업데이트 알림이 필요한지 결정해 보세요.

각 분류에 대해 선택 주체를 지정해 보세요. 모델이 작업을 수행하면 도구를 사용하세요. 호스트가 URI로 지정된 콘텐츠를 읽으면 자원을 사용하세요. 사용자가 준비된 메시지 워크플로우를 시작하면 프롬프트를 사용하세요.

## 실습 랩

저장소 루트에서 시뮬레이터를 실행하세요:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

트랜스크립트를 다음 순서로 검사하세요:

1. `server/discover`이 현재 리비전과 두 가지 기능을 광고하는지 확인하세요.
2. 두 목록 결과가 정렬되어 있고 `resultType: "complete"`을 사용하는지 확인하세요.
3. 목록 및 읽기 결과가 의도적인 캐시 힌트를 포함하는지 확인하세요.
4. 읽기 URI를 `notes://missing`으로 변경하고 `-32602`을 관찰하세요.
5. 구독 확인이 자원 이벤트보다 먼저 발생하는지 확인하세요.
6. 이벤트와 우아한 종료(Graceful Close)가 모두 구독 ID `5`을 포함하는지 확인하세요.

Python 모델은 실제 HTTP 연결을 열지 않습니다. SDK가 요청 범위 응답 스트림에 배치해야 하는 메시지를 표현합니다. 프로덕션에서는 프레이밍 및 전송을 위해 공식 SDK를 사용하세요.

## 출시된 산출물

`outputs/skill-primitive-splitter.md`은 MCP 원시(primitive) 선택을 위한 재사용 가능한 설계 리뷰입니다. 이제 결정론적 발견, 캐시 범위, 잘못된 URI 동작, 최신 구독 필터를 검사합니다.

이 강의는 원시 및 구독 경계의 정적 버전인 `assets/primitive-split.svg`도 함께 제공하여 오프라인 학습에 활용합니다.

## 검증하기

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

예상 결과: 메인 프로그램이 JSON 트랜스크립트를 출력하고, 테스트 명령이 최소 12개의 통과된 테스트를 보고합니다.

## 캡스톤 연결

캡스톤 서버가 작업(actions) 옆에 주소 지정 가능한 지식을 노출할 때 이 계약을 사용하세요. 결정론적 카탈로그 스냅샷, 승인된 자원 읽기, 프롬프트 해석, 잘못된 URI 케이스, 구독 트랜스크립트를 각각 하나 포함하세요.

증거는 목록이 연결 기록에 의존하지 않으며, 구독 이벤트가 underlying 자원(기저 자원)에 대한 접근 권한을 부여하지 않는다는 것을 보여야 합니다.

## 연습 문제

1. `notes://projects/{project}/notes/{id}` 리소스 템플릿을 추가하고 두 변수를 검증해 보세요.
2. `resources/list`에 결정적 순서를 유지하면서 페이지네이션을 추가해 보세요.
3. 한 리소스를 `cacheScope: "private"`와 `ttlMs: 0`로 변경하고, 호스트 수준 no-store 정책을 추가하며, 두 통제를 정당화하는 위협을 설명해 보세요.
4. 프롬프트 목록 변경 구독을 추가하고, 필터가 `promptsListChanged`를 생략할 때 이벤트가 전송되지 않음을 증명해 보세요.
5. 동시 구독 두 개를 생성하고, 각 이벤트가 올바른 요청 ID를 포함함을 증명해 보세요.
6. 읽기 핸들러에 인증 주체를 추가하고, 캐시 항목이 주체 간에 교차할 수 없음을 증명해 보세요.

## 핵심 용어

- **리소스(Resource):** MCP 서버가 노출하는 URI로 주소 지정된 콘텐츠.
- **프롬프트(Prompt):** MCP 서버가 노출하는 사용자 제어 메시지 템플릿.
- **결정적 목록(Deterministic list):** 동일한 요청 입력에 대해 안정적인 구성원과 순서를 가지는 검색 결과.
- **`ttlMs`:** 밀리초 단위의 캐시 신선도 지속 시간.
- **`cacheScope`:** 캐시된 결과의 공유 경계.
- **`subscriptions/listen`:** 응답 스트림이 명시적으로 필터링된 알림을 전달하는 장기 지속 요청.
- **구독 ID(Subscription ID):** 알림 메타데이터에 반복되는 원본 listen 요청 ID.
- **유효하지 않은 매개변수(Invalid parameters):** JSON-RPC 오류 `-32602`, 유효하지 않거나 알 수 없는 리소스 URI에 사용됨.
- **지원되지 않는 프로토콜 버전(Unsupported protocol version):** `supported` 및 `requested` 개정판이 포함된 JSON-RPC 오류 `-32022`.
- **`server/discover`:** 지원되는 개정판, 기능, 신원 및 선택적 캐시 힌트를 반환하는 필수 서버 메서드.

## 추가 읽기

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
