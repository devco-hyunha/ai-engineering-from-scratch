# 상태 비저장 MCP 게이트웨이와 레지스트리 수용

> 게이트웨이는 모든 라우팅을 명시적으로 처리해야 합니다. 2026-07-28 프로토콜은 전송 세션 없이 메서드, 이름, 버전, 기능, 신원, 캐시 및 추적 경계를 정의합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 13단계 · 15강 (보안), 13단계 · 16강 (권한 부여)
**시간:** 약 75분

## 학습 목표

- 세션 어피니티 없이 하나의 2026-07-28 엔드포인트 뒤에 여러 MCP 서버를 통합해 보세요.
- 정책 적용이나 전달 전에 요청별 메타데이터와 라우팅 헤더를 검증해 보세요.
- 안정적인 네임스페이스, 결정적인 순서, 기술자 고정, RBAC 및 비공개 캐싱을 사용하여 도구를 병합해 보세요.
- 레지스트리 기록은 여전히 수용 정책이 필요한 발견 증거로 취급해 보세요.
- 요청 범위 SSE, `subscriptions/listen`, MRTR 재시도 및 Tasks 확장 호출을 올바르게 라우팅해 보세요.
- 레거시 핸드셰이크 및 세션 지원을 최신 경로와 분리해 보세요.

## 문제점

하나의 클라이언트를 하나의 서버에 직접 연결하는 것은 간단합니다. 더 큰 배포에서는 더 어려운 질문에 대한 일관된 답변이 필요합니다:

- 어떤 서버가 허용됩니까?
- 각 주체가 어떤 도구를 보고 호출할 수 있습니까?
- 두 백엔드가 동일한 이름을 노출하면 어떻게 됩니까?
- 기술자 변경은 어떻게 검토됩니까?
- 속도 제한 및 감사 이벤트는 어디에 적용됩니까?
- 임의의 인스턴스가 다음 요청을 처리할 수 있습니까?

게이트웨이는 클라이언트와 백엔드 MCP 서버 사이에 위치합니다. 하나의 MCP 엔드포인트를 제시하고, 횡단 정책을 적용하며, 승인된 요청을 전달합니다.

이전 게이트웨이 설계는 하나의 클라이언트 세션을 여러 백엔드 세션으로 다중화하고 `Mcp-Session-Id`를 재작성하는 경우가 많았습니다. 이는 레거시 호환성 설계입니다. 2026-07-28 코어에는 프로토콜 세션이 없습니다.

## 개념

### 최신 게이트웨이 경로

각 요청에 대해:

1. 전송 권한 부여에서 주체를 인증합니다.
2. `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name` 및 `params._meta`을 검증합니다.
3. 주체, 리소스, 메서드, 도구 및 인수를 승인합니다.
4. 디스크립터, 레지스트리, 속도 및 데이터 정책을 적용합니다.
5. 선택된 백엔드를 위해 완전히 독립적인 새 요청을 생성합니다.
6. 백엔드 결과를 검증하고 게이트웨이 결과를 반환합니다.
7. 비밀을 기록하지 않고 감사 이벤트를 기록합니다.

어떤 단계도 숨겨진 프로토콜 세션을 필요로 하지 않습니다. 애플리케이션 상태는 데이터베이스, 명시적 핸들, 작업(Tasks) 또는 무결성이 보호되는 MRTR (다중 왕복 요청)(Multi Round-Trip Request) 상태에 존재할 수 있습니다.

### 런타임 정책이 게이트웨이의 주요 결정입니다

수용 제어(Admission Control)는 어떤 백엔드 버전이 게이트웨이에 진입할 수 있는지 결정합니다. 실시간 호출을 승인하지는 않습니다. 모든 요청에 대해 게이트웨이는 인증된 주체, 발급자 및 리소스, 테넌트, 매칭된 메서드와 이름, 정규화된 인수, 수용된 디스크립터 고정(pin), 현재 백엔드 상태, 기능 교집합, 데이터 분류, 속도 상태 및 작업에 바인딩된 승인 등로부터 정책을 재계산합니다.

이 순서가 중요합니다. 사용자의 역할이 취소되어도 레지스트리 기록은 활성 상태로 남을 수 있습니다. 목적지 인수가 테넌트 경계를 넘어서도 디스크립터는 고정된 상태로 남을 수 있습니다. 사고 정책이 상태 변경 호출을 격리하더라도 백엔드는 승인된 상태로 남을 수 있습니다. 따라서 런타임 정책이 주요 허용 또는 거부 결정이며, 레지스트리와 디스크립터 증거는 입력으로 사용됩니다.

연결 또는 제거된 세션 식별자 아래에서 허용 결정을 캐시하지 마세요. 정책이 사용 불가능한 경우, 작업 클래스에 따라 선언된 실패 정책을 따르세요. 안전한 기본값은 상태 변경 및 민감한 읽기에 대해 폐쇄(fail closed)로 실패하는 것이며, 명시적으로 승인된 공개 읽기 경로는 위험 모델이 허용하는 경우에만 짧은 수명의 마지막 알려진 정책을 사용할 수 있습니다. 어떤 정책 버전과 실패 경로가 결정을 내렸는지 기록한 후, 반환하기 전에 백엔드 결과를 검증하세요.

### 단일 POST 엔드포인트

현대적인 Streamable HTTP는 각 JSON-RPC 메시지를 POST를 통해 전송합니다:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

게이트웨이는 해당 POST에 대해 JSON 또는 요청 범위 SSE를 반환할 수 있습니다. GET 및 DELETE는 현대적인 요청에 대해 405를 반환합니다. `Mcp-Session-Id` 및 `Last-Event-ID`는 권한, 친화성, 또는 재생(replay) 동작을 생성하지 않습니다.

헤더와 본문 값이 일치해야 합니다. 백엔드를 조회하기 전에 `-32020`로 불일치를 거부합니다. 이를 통해 로드 밸런서, 게이트웨이 및 속도 제한기가 전체 본문을 파싱하지 않고도 라우팅할 수 있으며, 엔드투엔드 무결성을 유지합니다.

정확한 순서로 검증하세요: JSON-RPC 및 메타데이터 유형, 헤더와 본문의 일치 여부, 그리고 매칭된 버전의 지원 여부. 불일치 시 `-32020`와 함께 HTTP 400을 반환합니다. 헤더와 본문이 지원되지 않는 버전에서 일치하는 경우, `-32022`, `data` 및 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`와 함께 HTTP 400을 반환합니다. 알 수 없는 메서드는 `-32601`와 함께 HTTP 404를 반환합니다.

`ProtocolError`는 선택적인 `data`를 포함하며, 게이트웨이는 이를 JSON-RPC 오류 객체로 직렬화합니다. 알림(notification)에는 `id`가 없으므로 JSON-RPC 성공 또는 오류 응답을 받지 않습니다. 승인된 HTTP 알림은 빈 본문과 함께 202를 반환합니다.

### 모든 계층에서 발견(discovery)을 구현하세요

게이트웨이는 클라이언트를 위해 `server/discover`를 구현합니다. 또한 각 백엔드를 발견하여 프로토콜 버전, 기능 및 확장 기능을 파악합니다.

게이트웨이 결과 예시:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

게이트웨이가 엔드투엔드로 준수할 수 있는 기능 교집합만 광고하세요. 백엔드 기능은 자동으로 노출하기에 안전하지 않습니다. 백엔드 경로가 없는 게이트웨이 기능은 광고할 가치가 없습니다.

`serverInfo`는 자가 보고된 표시 및 진단 데이터입니다. 레지스트리나 게시자 증명으로는 사용하지 마세요.

### 요청별 클라이언트 기능

전달된 모든 요청에는 현재 `_meta` 인벨로프(envelope)가 필요합니다:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

외부 클라이언트 기능을 백엔드에 무비판적으로 복사하지 마세요. 게이트웨이는 백엔드의 클라이언트입니다. 게이트웨이가 올바르게 중재할 기능만 광고하세요.

### 결정적 네임스페이싱

백엔드 도구를 안정적인 공개 이름 아래에 병합하세요:

```text
notes.search
notes.create
issues.list
issues.open
```

공개 이름에서 백엔드 및 원본 도구 이름으로의 매핑을 유지하세요. 충돌 시 첫 번째나 마지막을 선택하지 마세요. 공개 이름은 승인 및 감사 계약의 일부이므로, 이를 변경하는 것은 마이그레이션입니다.

`tools/list`는 결정적이어야 합니다. 주체(principal)에 따라 가시성이 다른 경우 `cacheScope: private`를 반환하세요. 제한된 `ttlMs`는 백엔드 발견 부하를 줄이면서 사용자별 목록이 인증 컨텍스트 간에 유출되는 것을 방지합니다.

노출된 모든 도구 설명자에는 안정된 이름, 설명, 객체 루트 `inputSchema`가 포함됩니다. 네임스페이싱은 필수 설명자 필드를 제거할 수 없습니다. 전체 목록 결과에는 `resultType`, 서버 식별 메타데이터, 캐시 힌트도 포함됩니다.

### 승인된 설명자 고정(Pin)

수용 시점에 전체 설명자를 정규화하고 자격을 갖춘 공개 이름 아래에 그 다이제스트를 저장하세요. 목록 및 호출 시점에 라이브 설명자를 승인된 다이제스트와 비교하세요.

변경된 경우:

- `tools/list`에서 제거하세요.
- 직접 호출을 거부하세요.
- 감사 이벤트를 방출하세요.
- 고정(Pin)을 업데이트하기 전에 정책 또는 인간 재승인을 요구하세요.

게이트웨이는 유용한 중앙 강제 지점이지만, 처음 본 설명자를 안전한 설명자로 만들지는 못합니다. 초기 검토는 여전히 필요합니다.

### 레지스트리는 발견을 돕지, 결정을 내리지는 못합니다

레지스트리 `server.json`는 출판 메타데이터를 제공합니다. 패키지 기반 레코드는 다음과 같이 보일 수 있습니다:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

출판 메타데이터는 게이트웨이의 보안 결정을 담지 않습니다. 검증된 게시자 및 출처 증거는 별도의 수용 상태에 유지하세요:

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

게이트웨이는 `server.json`의 형태를 확인하고 이를 외부 상태와 결합합니다. 게이트웨이에는 여전히 수용 정책이 필요합니다.

수용된 각 백엔드에 대해 다음을 기록하세요:

- 정확한 레지스트리 및 레코드 식별자.
- 검증된 게시자 네임스페이스 또는 도메인 증거.
- 허용된 전송 및 엔드포인트.
- 고정된 버전 또는 승인된 업그레이드 정책.
- 아티팩트 또는 설명자 다이제스트.
- 권한 발급자 및 리소스.
- 리뷰어, 승인 시간, 만료일.

표시 이름이 익숙한 제품과 유사하다는 이유로 서버를 수용하지 마세요. 레지스트리 존재를 운영 보안 검토로 취급하지 마세요. 비공개 서버는 공개 레지스트리에 나타나지 않더라도 동일한 증거 스키마를 통해 수용될 수 있습니다.

이 강의는 게이트웨이 접합(seam)을 구현합니다: 백엔드가 라우팅 가능해지기 전에 게시 증거를 로컬 수용과 결합합니다. [30강: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md)은 정확한 네임스페이스 증명, 아티팩트 출처, 불변 고정(pins), 라이브 디스크립터 드리프트, 레지스트리 상태 조정, 변조 증거가 있는 수용 대장, 증거 기반 롤백을 위한 완전한 제어 평면을 구축합니다. 위의 요청별 런타임 결정과 공급망 상태를 분리하여 유지하세요.

### 자격 증명 중재

게이트웨이는 호출자를 인증하고 백엔드에 대해 별도로 인증합니다. 백엔드 자격 증명은 클라이언트로 전달되지 않습니다.

이 바인딩을 명시적으로 유지하세요:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

외부 게이트웨이 토큰을 백엔드에 전달하지 마세요. 다른 발급자나 리소스에서 백엔드 토큰을 재사용하지 마세요. 도구가 최종 사용자를 대리하는 경우, 공유 서비스 자격 증명으로 사용자를 사칭하지 말고 설계된 교환(exchange) 또는 클레임 모델로 위임을 보존하세요.

### 세션 없는 속도 제한

인증된 주체(principal), 발급자, 리소스, 공개 도구, 비용 클래스, 시간 창(window)에 따라 제한을 설정하세요. 세션 ID는 존재하지 않으며, 존재하더라도 쉽게 회전(rotating)할 수 있습니다.

비싼 작업을 소비하기 전에 저렴한 검증을 적용하세요. 거부된 호출이 남용 제한, 비즈니스 할당량, 또는 둘 다에 포함되는지 결정하세요.

### 결정 체인 감사

호출을 재구성할 수 있도록 충분히 기록하세요:

- 요청 및 추적 식별자.
- 인증된 주체(principal) 및 발급자.
- 공개 도구 및 백엔드 라우트.
- 디스크립터 고정(pin) 버전.
- 정책 결정 및 이유.
- 지연 및 결과 클래스.
- 해당하는 경우 MRTR (다중 왕복 요청)(Multi Round-Trip Request) 라운드 또는 작업 식별자.

베어러(bearer) 토큰, 인증 코드, 리프레시 토큰, 원시 비밀(raw secrets), 불필요한 민감한 인자를 마스킹하세요.

### 요청 범위 SSE

일반 POST는 해당 요청 동안 작업이 스트리밍될 때 요청 범위 SSE를 반환할 수 있습니다. 응답 스트림을 닫으면 해당 진행 중인 최신 HTTP 요청이 취소됩니다.

별도의 GET 스트림을 만들지 마세요. Last-Event-ID 재생(replay)을 약속하지 마세요. 이는 오래된 전송(transports) 가정입니다.

### 장기 지속 변경 알림

목록 및 리소스 변경 알림의 경우, 현재 클라이언트는 POST를 통해 `subscriptions/listen`을 전송하고 SSE 응답을 받습니다. 알림 필터는 `toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`와 같은 정확한 평면 필드를 사용합니다:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

첫 번째 이벤트는 지원되는 하위 집합을 확인합니다. 해당 구독 식별자는 스트림을 연 요청의 JSON-RPC id입니다:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

게이트웨이 확인된 변경 유형만 전달합니다. 해당 스트림의 모든 알림은 `params._meta`에 동일한 `io.modelcontextprotocol/subscriptionId`을 포함합니다. 자동 재생이나 자동 재청취는 없습니다. 재연결 시 클라이언트는 구독을 다시 열고 의존하는 목록을 새로고침합니다. 서버가 시작하는 우아한 종료는 동일한 구독 id로 태그된 최종 완전한 결과를 반환합니다.

현대적 경로는 `resources/subscribe`, `resources/unsubscribe` 및 요청되지 않은 독립적인 GET 스트리밍을 대체합니다. 이러한 기능은 버전이 제한된 이전 경로에서만 유지하세요.

### 게이트웨이를 통한 MRTR

백엔드가 `resultType: input_required`을 반환할 경우, 게이트웨이는 외부 클라이언트가 필요한 입력 요청을 지원할 때만 해당 결과를 전달할 수 있습니다. 게이트웨이가 의도적으로 상호작용을 종료하고 재발행하지 않는 한 `requestState`을 바이트 단위로 보존하세요.

클라이언트는 새로운 JSON-RPC id와 `inputResponses`을 사용하여 원래의 공개 도구를 재시도합니다. 게이트웨이는 재시도를 재승인하고 동일한 공개 경로를 확인한 후 새로운 백엔드 요청을 전달합니다. 이전 라운드가 무제한 승인을 부여했다고 가정해서는 안 됩니다.

### 작업 확장 라우팅

작업은 `io.modelcontextprotocol/tasks`으로 식별되는 공식 확장입니다. 이는 핵심 세션의 대체물이 아닙니다.

클라이언트는 요청별 클라이언트 기능 내에서 확장을 선언하며, 게이트웨이는 전체 수명 주기를 보존할 수 있을 때만 발견(discovery) 과정에서 이를 광고합니다. 지원되는 `tools/call`의 경우, 백엔드만이 일반적인 결과를 반환할지 `resultType: task`을 반환할지 결정합니다. 작업 결과는 `taskId`, `status`, 타임스탬프, `ttlMs` 및 선택적 `pollIntervalMs`를 결과에 직접 포함합니다. 해당 결과가 전송되기 전에 작업은 이미 내구성 있게 읽을 수 있는 상태여야 합니다.

게이트웨이는 인증된 주체(principal)와 백엔드 라우트를 불투명 작업 식별자(opaque task identifier)에 기록합니다. 이후 `tasks/get`, `tasks/update`, `tasks/cancel` 호출은 `params.taskId`을 `Mcp-Name`으로 사용하며, 이는 중개자에게 라우팅 키를 제공합니다. `tasks/get`는 현재 작업 상태와 함께 `resultType: complete`을 반환하며, 종결 상태에서는 최종 결과나 프로토콜 오류를 인라인합니다. `tasks/update`는 미완료 작업 입력에 대해 키가 지정된 `inputResponses`을 전송하고 빈 완료 확인 응답을 반환합니다. `tasks/cancel`는 빈 완료 확인 응답을 가진 협력적 의도(cooperative intent)이며, 작업 중지를 보장하지는 않습니다.

새로운 `tasks/list` 또는 `tasks/result` 메서드를 구현하지 마세요. 이들은 이전 실험적 모델에 속합니다. 입력이 필요한 작업은 `tasks/get`를 통해 내장된 완전한 요청을 노출합니다. 클라이언트는 원본 도구 호출을 재시도하는 것이 아니라 `tasks/update`를 통해 이에 응답합니다. 클라이언트는 제안된 간격으로 계속 폴링하며, 작업 생성은 서버 주도(server-directed)로 유지됩니다.

내구성 있는 작업 라우트 상태는 작업 핸들(task handle)을 키로 하는 애플리케이션 데이터이며, 프로토콜 세션이 아닙니다.

### 호환성 경계

게이트웨이가 이전 클라이언트나 백엔드를 제공해야 하는 경우:

- 에라(era)를 명시적으로 감지하세요.
- 초기화, 전송 세션, GET 스트림, 리소스 구독 및 이전 작업 어휘를 레거시 어댑터 내부에 유지하세요.
- 레거시 세션 ID가 현대적 라우팅이나 인증에 유출되지 않도록 하세요.
- 침묵의 다운그레이드(silent downgrade)보다는 제한된 발견(probe)과 명시적 폴백(fallback) 정책을 선호하세요.

```figure
t3-gateway-funnel
```

## 구현하기

`code/main.py`는 프로세스 내 프로토콜 게이트웨이와 두 개의 백엔드 서버를 구현합니다. 각 백엔드는 새로운 현재 프로토콜 요청을 받습니다. 게이트웨이는 발견(discovery), 사용자 필터링된 결정적 `tools/list`, 네임스페이스 라우팅, Registry `server.json` 및 외부 수용(admission) 상태, 디스크립터 고정(pin), RBAC, 주체(principal) 키 기반 속도 제한, 감사 결정(audit decisions) 및 모델링된 `subscriptions/listen` SSE 확인 응답을 제공합니다.

모델은 파싱된 요청 본문, 라우팅 헤더 및 인증된 베어러(bearer) 신원을 받습니다. 이는 완전한 HTTP 어댑터가 아니며 `Content-Type`이나 전체 `Accept` 계약을 파싱하지 않습니다. `Content-Type: application/json`와 `application/json` 및 `text/event-stream`를 모두 포함하는 `Accept` 값이 필요한 09강의 Streamable HTTP 어댑터에 연결하세요.

실행해 보세요:

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

데모는 외부 요청 ID와 새로운 백엔드 요청 ID를 출력하므로 상태 비저장(Stateless) 홉(Hop)이 명확하게 보입니다.

## 사용하기

프로세스 내 백엔드 객체를 실제 현재 프로토콜 클라이언트로 교체하세요. 동일한 접합점(Seam)을 유지합니다:

- 연결 전 수용 기록(Admission record).
- 기능 노출 전 백엔드 발견(Discovery).
- 권한 부여 전 자격을 갖춘 공개 이름(Qualified public name).
- 목록 조회 또는 호출 전 설명자(descriptor) 고정(Pin).
- 전달 전 요청별 신선한 메타데이터.
- 반환 전 결과 검증.

## 출시하기

이 강은 `outputs/skill-gateway-bootstrap.md`를 출시합니다. 이는 ingress, discovery, admission, namespaces, authorization, caching, streaming, subscriptions, MRTR (다중 왕복 요청)(Multi Round-Trip Request), Tasks, observability, legacy isolation을 포괄하는 현대적인 게이트웨이 설계를 생성합니다.

## 연습 문제

1. 외부 및 전달된 요청 메타데이터에 추적(trace) 컨텍스트를 추가하고 감사(audit) 이벤트에 상관 관계를 기록하세요.
2. Tasks 기능을 갖춘 백엔드를 추가하고 `Mcp-Name`에서 작업 ID를 통해 `tasks/get`를 라우팅하세요.
3. 백엔드 설명자(descriptor) 중 하나를 변경하고 discovery와 직접 호출이 모두 차단됨을 증명하세요.
4. 주체(principal)별 서버 기능을 추가하고 discovery가 왜 비공개 캐시(private cache)에 유지되어야 하는지 설명하세요.
5. 현대 `Gateway` 클래스에 레거시 상태를 추가하지 않고 레거시 어댑터 인터페이스를 작성하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| MCP 게이트웨이 | 클라이언트와 백엔드 MCP 서버 사이의 정책 및 라우팅 서버 |
| 수용 기록 | 하나의 백엔드를 게이트웨이로 허용하는 증거 및 정책 결정 |
| 자격을 갖춘 도구 이름 | `notes.search`와 같은 안정적인 공개 경로 |
| 설명자 고정 | Discovery 및 dispatch 중에 확인되는 승인된 다이제스트 |
| 비공개 캐시 범위 | 하나의 권한 부여 컨텍스트로 제한된 캐시된 결과 |
| 요청 범위 SSE | 하나의 POST 요청에 연결된 스트리밍 응답 |
| `subscriptions/listen` | 선택된 장기 지속 변경 알림을 위해 클라이언트가 연 SSE 스트림 |
| 작업 라우트 | 불투명(opaque)한 작업 ID를 백엔드에 매핑하는 애플리케이션 |
| 레거시 어댑터 | 구식 핸드셰이크 및 세션 동작에 대한 명시적 버전 게이트 경계 |

## 추가 읽기

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
