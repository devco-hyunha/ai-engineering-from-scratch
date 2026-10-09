# MCP Tasks 확장: 상태 비저장 코어에서의 내구성 있는 작업

> 상태 비저장 MCP는 모든 연산이 단일 요청 내에서 완료되어야 함을 의미하지 않습니다. 공식 Tasks 확장은 장기 실행 작업에 명시적인 내구성 핸들을 제공합니다. 서버는 `tools/call`에서 해당 핸들을 반환할 수 있으며, 모든 인스턴스가 `tasks/get`에 응답할 수 있고, 클라이언트 입력은 프로토콜 세션을 부활시키지 않고 `tasks/update`를 통해 도착합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계 · 09 (전송), 13단계 · 11 (상태 비저장 MRTR), 13단계 · 12 (elicitation)
**시간:** 약 90분

## 학습 목표

- 상태 비저장 프로토콜 전송과 내구성 있는 애플리케이션 작업 상태를 구분합니다.
- 요청별 기능 및 `server/discover`에서 `io.modelcontextprotocol/tasks` 확장을 협상합니다.
- 내구성 있는 생성이 완료된 후에만 `resultType: "task"`를 포함하는 서버 주도 `CreateTaskResult`를 반환합니다.
- `tasks/get`로 폴링하고, `tasks/update`로 작업 입력을 충족하며, `tasks/cancel`로 협력적 취소를 요청합니다.
- 구식 `tasks/status`, `tasks/result`, `tasks/list` 가정을 제거합니다.
- POST 응답 SSE 스트림에서 `subscriptions/listen`를 통해 선택적 작업 알림을 구독합니다.
- 작업 만료, 재시작 복구, 입력 키 중복 제거 및 실행 오류를 올바르게 모델링합니다.

## Tasks가 확장인 이유

Tasks는 2025-11-25에 실험적 코어 기능으로 처음 등장했습니다. 2026년 7월 재설계는 이를 공식 `io.modelcontextprotocol/tasks` 확장으로 이동시켜, 클라이언트와 서버가 코어 프로토콜을 모든 사용자에게 확장하지 않고도 추가적인 생명주기를 선택적으로 도입할 수 있게 합니다.

이 확장 사양은 Tasks의 현재 공식 위치임에도 불구하고 초안 상태입니다. SDK가 지원하는 확장 버전을 고정하고, 적합성 시나리오를 실행하며, 워커 및 저장소 도메인으로부터 와이어 어댑터를 분리하세요.

연산이 다음 속성 중 하나 이상을 가질 때 task를 사용하세요:

- 일반적인 요청 타임아웃보다 오래 지속될 수 있습니다.
- 워커 큐나 외부 작업 시스템이 이미 실행을 소유하고 있습니다.
- 클라이언트가 자체 재시작 후 복구해야 합니다.
- 연산이 실행 중에 사용자 또는 모델 입력을 위해 일시 중지됩니다.
- 취소 및 내구성 있는 결과 검색은 제품 요구 사항입니다.

저렴한 결정적 조회를 위해 작업을 생성하지 마세요. 핸들, 지속성, 폴링, 만료, 취소는 실제 복잡성입니다.

## 상태 비저장 코어, 상태 저장 애플리케이션

MCP 2026-07-28은 `initialize`, `notifications/initialized`, 프로토콜 세션 및 `Mcp-Session-Id`를 제거합니다. 이는 상태 저장 제품을 금지하지 않습니다.

작업 ID는 명시적인 애플리케이션 상태입니다:

- 서버는 반환하기 전에 이를 지속합니다.
- 클라이언트는 이를 저장하고 재시작 후 다시 폴링할 수 있습니다.
- ID는 동일한 내구성 스토어를 사용하는 모든 복제본으로 라우팅될 수 있습니다.
- 모든 작업 메서드에서 권한이 확인됩니다.
- 만료 및 삭제는 전송 수명이 아닌 작업 필드에 의해 정의됩니다.

이는 연결에 첨부된 숨겨진 상태와 운영적으로 다릅니다.

네 가지 수명을 분리하여 유지하세요:

| 상태 | 수명 | 위치 |
|---|---|---|
| 프로토콜 메타데이터 | 한 요청 | `params._meta`, 모든 호출에서 다시 검증 |
| 전송 작업 | 한 stdio 요청 또는 HTTP 응답 | 제한된 마감 시간을 가진 진행 중 조정자 |
| MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR)) 연속 | 한 재시도 시퀀스 | 무결성 보호된 `requestState`, 필요 시 재생 제어 포함 |
| 내구성 있는 작업 | 요청, 복제본, 재시작 및 재연결을 초월하여 | 권한이 부여된 `taskId`를 키로 하는 공유 애플리케이션 스토어 |

작업 레코드를 프로세스 메모리로 이동하는 것은 MCP를 상태 저장으로 만들지 않습니다. 이는 애플리케이션을 신뢰할 수 없게 만듭니다. 프로토콜은 상태 비저장 상태를 유지하지만, 다른 복제본으로 라우팅된 후속 `tasks/get`는 레코드를 복구할 수 없습니다. 핸들을 반환하기 전에 지속한 후, 모든 작업 메서드가 테넌트 및 주체 확인 하에 동일한 공유 레코드를 해석하도록 하세요.

## 기능 협상

클라이언트는 모든 적합한 요청에서 지원 여부를 광고합니다:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

서버는 `server/discover`에서 `supportedVersions`, 기능(capabilities), `ttlMs`, `cacheScope`를 정확히 반환하며, capabilities 아래에 동일한 확장(extension)을 포함합니다. 도구를 광고하므로 필수 `tools/list`도 구현합니다. 이 결과는 결정적인 `generate_report` 설명자(descriptor), 유효한 객체 `inputSchema`, `resultType: "complete"`, 서버 식별자 메타데이터, 공개 캐시 힌트를 반환합니다.

확장(extension)을 선언하지 않은 클라이언트에서 요청한 task 메서드는 `data.requiredCapabilities`이 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`으로 설정된 `-32021`, Missing Required Client Capability를 반환합니다. 지원되지 않는 프로토콜 문자열은 정확한 `supported` 및 `requested` 데이터를 포함하는 `-32022`을 반환하며, 버전이 없거나 문자열이 아닌 경우 `-32602`을 반환합니다.

JSON-RPC `id`가 없는 엔벨로프(envelope)는 알림(notification)입니다. 수신자는 이를 처리할 수 있지만, JSON-RPC 결과나 오류를 방출하지 않습니다. Streamable HTTP 어댑터는 승인된 알림에 대해 본문(body) 없이 `202 Accepted`을 반환합니다.

현재는 `tools/call`만 task-augmented 실행을 지원합니다. 향후 요청 유형이 스토리지 재작성을 요구하지 않도록 내부 추상화를 설계해 보세요.

## 서버 주도 작업 생성(Server-Directed Task Creation)

이전 클라이언트 플래그 `params._meta.task.required`는 제거되었습니다. 클라이언트가 확장(extension) 지원을 선언하면, 서버가 특정 `tools/call`가 task가 될지 여부를 결정합니다.

요청:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

응답:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

서버는 해당 id에 대한 `tasks/get`가 해결될 때까지 이 핸들(handle)을 반환해서는 안 됩니다. 궁극적 일관성(eventually consistent) 스토어에서는 응답하기 전에 읽기 가시성(read visibility)을 기다려야 합니다. 그렇지 않으면 클라이언트가 유효해 보이는 id를 받고 즉시 "not found"를 받을 수 있습니다.

task 응답은 클라이언트가 task 모드를 요청하지 않는다는 의미에서 비자발적(unsolicited)입니다. 그러나 협상되지 않은 것은 아닙니다: 현재 요청은 여전히 확장(extension)을 광고해야 합니다.

## Task 형상(Task Shape)

모든 task는 다음을 포함합니다:

- `taskId`: 서버가 생성한 안정적 식별자;
- `status`: `working`, `input_required`, `completed`, `cancelled`, 또는 `failed`;
- `createdAt` 및 `lastUpdatedAt`: ISO 8601 타임스탬프;
- `ttlMs`: 생성 시점부터의 만료 기간, 또는 광고된 제한이 없는 경우 `null`;
- 선택적 `pollIntervalMs`: 서버가 현재 제안하는 최소 폴링 주기(polling cadence);
- 선택적 `statusMessage`: 사용자-facing 또는 모델-facing 컨텍스트.

상태별 필드는 관련이 있을 때만 나타납니다:

- `input_required`는 `inputRequests`를 포함합니다.
- `completed`는 원본 요청의 `result` 형상을 포함합니다.
- `failed`는 JSON-RPC `error` 객체를 포함합니다.

클라이언트는 `pollIntervalMs`를 준수해야 합니다. 서버는 더 공격적인 폴링에 대해 속도 제한을 적용할 수 있으며, 작업 수명 동안 간격을 변경할 수 있습니다.

## `tasks/get`로 폴링하기

클라이언트는 현재 스냅샷을 요청합니다:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 자체가 완료되었으므로, 그 결과는 항상 `resultType: "complete"`를 가집니다. 중첩된 작업은 여전히 `status: "working"` 또는 `status: "input_required"`를 가질 수 있습니다.

이 구별은 일반적인 파서 버그를 방지합니다:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

`tasks/result` 호출은 없습니다. 작업이 완료되면, 다음 `tasks/get` 응답은 원본 `CallToolResult`를 `result` 아래에 인라인합니다:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

외부 `resultType`는 `tasks/get` RPC가 완료되었음을 나타냅니다. 중첩된 `result.resultType`는 원본 도구 호출이 완료되었음을 나타냅니다. 이 중첩된 판별자는 필수입니다. 중첩된 `CallToolResult`는 자체 `io.modelcontextprotocol/serverInfo`를 포함해야(SHOULD) 합니다. 이 강의에서는 타입이 지정되지 않은 페이로드를 저장하는 대신 이를 포함합니다.

`tasks/list`는 없습니다. 세션 없는 서버는 연결 범위 목록에 어떤 작업이 속하는지 안전하게 추론할 수 없습니다. 이력이 필요한 애플리케이션은 명시적인 필터와 소유권 규칙을 가진 승인된 도메인 도구를 노출해야 합니다.

## 작업 실행 중 입력

작업 입력과 핵심 MRTR은 유사해 보이지만, 서로 다른 연속성을 사용합니다.

### 작업 생성 전에 필요한 입력

원본 `tools/call`에서 핵심 `resultType: "input_required"`를 반환합니다. 클라이언트는 이를 충족하고 원본 호출을 재시도합니다. 이러한 동기 MRTR 라운드가 완료된 후에만 작업을 생성합니다.

### 작업 생성 후에 필요한 입력

작업을 `input_required`로 설정합니다. `tasks/get`는 미해결된 `inputRequests`를 노출하며, 클라이언트는 `tasks/update`를 통해 응답을 전송합니다. 클라이언트는 원본 `tools/call`를 재시도하지 않습니다.

스냅샷:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

업데이트:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

성공 응답은 빈 확인 응답과 `resultType: "complete"`입니다. 상태 변경은 궁극적으로 일관될 수 있으므로, 클라이언트는 폴링이나 리스닝을 계속합니다.

각 `inputRequests` 키는 전체 작업 수명 동안 고유해야 합니다. 반복된 `tasks/get` 스냅샷은 동일한 미처리 키를 표시할 수 있습니다. 클라이언트는 UI를 중복 제거하고 서버는 알 수 없거나, 대체되었거나, 이미 처리된 키에 대한 응답을 무시합니다. 부분 업데이트는 모든 필수 키가 응답될 때까지 작업을 `input_required` 상태로 남길 수 있습니다.

## 취소는 협력적입니다

`tasks/cancel`는 의도를 신호하고 빈 완료 확인 응답을 반환합니다. 이 확인 응답은 워커가 중단되었음을 보장하지 않습니다. 작업이 먼저 완료되거나, 취소를 무시하거나, 나중에 전환될 수 있습니다.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

세 가지 작업 메서드 모두에서 `Mcp-Name`는 `params.taskId`을 반영합니다. JSON-RPC 메서드 이름을 반복하지 않습니다. `code/main.py`는 `make_http_request`에서 이 규칙을 중앙화합니다.

레슨 워커는 취소를 즉시 준수하여 반복 호출을 멱등성(Idempotency) 있게 만듭니다. 프로덕션 클라이언트는 확인 응답으로부터 최종 작업 상태를 추론하는 대신 취소를 협력적인 것으로 취급해야 합니다.

`notifications/cancelled`를 사용하여 작업을 취소하지 마세요. 그 알림은 요청 취소에 속하며, 지속성 있는 작업(Tasks)에 속하지 않습니다.

이 구별은 라우팅 경계에서 중요합니다. 요청 취소는 하나의 진행 중인 JSON-RPC 작업이나 그 요청 범위 HTTP 응답을 대상으로 합니다. `tools/call`가 이미 `resultType: "task"`을 반환했다면, 그 요청은 완료된 것이며 그 전송을 닫는다고 해서 지속성 있는 작업을 지정하거나 중단할 수 없습니다. `tasks/cancel`는 새로 승인된 RPC입니다. `params.taskId`를 포함하고, `Mcp-Name`에서 그 id를 반영하며, 작업의 소유 백엔드를 해석하고, 협력적인 취소 의도를 기록하며, 워커가 중단되었다고 주장하지 않고 확인 응답을 반환합니다.

따라서 게이트웨이(gateway)는 요청 조정자와 작업 라우트를 서로 다른 테이블에 유지해야 합니다. 요청 테이블은 응답이 완료되면 사라질 수 있습니다. 작업 라우트는 종결 상태와 보존 만료까지 생존해야 합니다. [29강: MCP Reliability, Cancellation, and Flow Control](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md)는 두 경로 모두에 대해 경쟁, 시간 초과, 멱등성(Idempotency), 백프레셔(Backpressure), 재시도 규칙을 구축합니다.

## 선택적 알림

폴링은 기본입니다. 푸시 업데이트를 원하는 클라이언트는 작업 id와 함께 `subscriptions/listen`를 전송합니다. Streamable HTTP의 경우, 이는 응답이 요청 범위 SSE 스트림인 POST입니다. 독립적인 GET 이벤트 스트림은 없으며, 유지해야 할 프로토콜 세션도 없습니다.

서버는 `notifications/subscriptions/acknowledged`로 승인된 id를 확인하고, 이후 `notifications/tasks`을 통해 전체 스냅샷을 전송할 수 있습니다. 확인 응답 및 모든 작업 알림은 `_meta` 안에 `io.modelcontextprotocol/subscriptionId`를 포함하며, 이는 `subscriptions/listen` 요청 id와 동일합니다. 각 작업 알림은 그 시점에 `tasks/get`가 반환하는 것과 그 외에는 동일합니다.

클라이언트는 여전히 Tasks 확장 기능을 선언해야 합니다. 이벤트 재생이나 `Last-Event-ID`에 의존하기보다, 내구성 있는 작업 id를 사용하여 재연결하고 작업을 재개해야 합니다.

## 실패 의미론

두 개의 오류 계층을 올바르게 사용하세요.

### 프로토콜 오류

잘못된 메서드 매개변수나 알 수 없는 작업 id는 JSON-RPC 오류를 반환하며, 일반적으로 `-32602`입니다. 확장 기능 지원이 누락되면 `-32021`이 필요한 기능 객체와 함께 반환됩니다.

### 작업 실행 결과

- `isError: true`을 포함한 정상적인 도구 결과는 도구 호출이 정의된 결과를 생성했으므로 여전히 `completed` 작업입니다.
- 지연 실행 중 JSON-RPC 오류가 발생하면 작업이 `failed`이 되며, 해당 JSON-RPC 오류는 `error` 아래에 저장됩니다.
- 사용자 거부는 `cancelled`, 완료된 거부 결과, 또는 기타 도메인 특이적인 안전한 결과를 생성할 수 있습니다. 선택 사항을 문서화하세요.

## 내구성, 만료, 소유권

작업 id, 상태, 타임스탬프, ttl, 폴링 간격, 원본 작업 소유권, 결과 또는 오류, 미완료 입력 요청, 그리고 발급된 모든 입력 키를 최소한으로 영속화하세요.

저장소 키는 권위 있는 테넌트와 주체를 포함하거나 해석해야 합니다. 작업 id를 아는 것만으로는 접근 권한이 부여되지 않습니다. 모든 `tasks/get`, `tasks/update`, `tasks/cancel` 및 구독에서 소유권을 확인하세요.

`ttlMs`는 생성 시점부터 측정되며 변경될 수 있습니다. 클라이언트는 작업이 관측 가능한 업데이트를 더 이상 생성하지 않을 때 백스톱으로 이를 처리할 수 있습니다. 서버는 만료된 작업을 실패 처리한 후 나중에 삭제할 수 있습니다. 완료 후 그 밀리초 동안 완료된 결과를 보존하겠다는 약속으로 설명하지 마세요.

원자적 쓰기나 트랜잭션을 사용하세요. 이 강의에서는 임시 파일을 작성하고 원자적으로 이름을 변경합니다. 다중 복제 서비스는 공유 내구성 저장소와 워커 리스 또는 동등한 동시성 제어 기능을 사용해야 합니다.

```figure
tp-task-lifecycle
```

## 구현하기

`code/main.py`는 결정론적 작업 서비스를 구현합니다:

- `server/discover`은 `supportedVersions`, 캐시 힌트 및 Tasks 확장 기능을 반환합니다.
- `tools/list`은 유효한 입력 스키마를 가진 결정적이고 캐시 가능한 `generate_report` 디스크립터를 반환합니다.
- `tools/call`은 `resultType: "task"`을 반환하기 전에 작업을 생성하고 영속화합니다.
- 새로운 서비스 인스턴스가 동일한 작업을 다시 로드하여 재시작 복구를 시연합니다.
- `tasks/get`은 완전한 작업 스냅샷을 반환합니다.
- 워커는 `working`에서 `input_required`로 이동합니다.
- `tasks/update`은 양식 응답을 수락하고 빈 완료 확인 응답을 반환합니다.
- 워커는 자체 `resultType` 및 서버 식별자를 가진 중첩된 `CallToolResult`을 저장한 후 `completed`로 전환합니다.
- `tasks/cancel`은 이 구현에서 멱등(Idempotent)합니다.
- HTTP 빌더는 `tasks/get`, `tasks/update` 및 `tasks/cancel`에 대해 `Mcp-Name`을 `params.taskId`으로 설정합니다.
- 알림 헬퍼는 `notifications/subscriptions/acknowledged`과 `notifications/tasks`을 사용하며, 둘 다 listen 요청 id로 태그가 지정됩니다.
- ID가 없는 알림은 JSON-RPC 응답을 생성하지 않습니다.

워커는 백그라운드 스레드에서 대기하는 대신 명시적으로 진행합니다. 이는 모든 상태 전환을 결정적으로 만들고 프로토콜 예제를 큐 메커니즘과 분리합니다.

## 사용하기

저장소 루트에서:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

예상 결과 순서:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

`tasks/status`, `tasks/result` 및 `tasks/list`가 최신 서비스에서 method-not-found를 반환하는지 확인하세요.
`tools/list`이 결정적인지, 그리고 모든 현재 HTTP 작업 메서드가 `Mcp-Name`를 통해 작업 id를 반영하는지 확인하세요.

## 출시하기

`outputs/skill-task-store-designer.md`은 이제 확장 기능을 인식하는 설계를 생성합니다: 기능 협상, 반환 전 영속화 생성, 현재 메서드, 입력 업데이트 흐름, 소유권, 만료, 취소, 구독 및 제거된 실험적 메서드에서의 마이그레이션.

## 연습 문제

1. 두 번째 미답변 입력 키를 추가하세요. 부분적인 `tasks/update`을 보내고 두 키가 모두 응답될 때까지 작업이 `input_required` 상태를 유지함을 증명하세요.
2. 저장소에 테넌트 소유권을 추가하고, 잘못된 인증된 주체가 제시한 유효한 작업 id를 거부하세요.
3. 만료 기간이 있는 워커 리스를 추가하세요. 두 서비스 인스턴스가 동일한 작업을 동시에 완료할 수 없음을 시연하세요.
4. `subscriptions/listen`에 대한 POST 응답 SSE 어댑터를 구현하세요. GET, `Last-Event-ID`, 세션 헤더는 추가하지 마세요.
5. 만료 정리 로직을 추가하세요. 만료된 작업과 형식이 잘못된 작업 ID를 구분하되, 테넌트 간 존재 여부를 노출하지 마세요.

## 핵심 용어

| 용어 | 현재 확장에서의 의미 |
|------|----------------------------------|
| Tasks 확장 | 내구성 있는 비동기 작업을 위한 선택적 `io.modelcontextprotocol/tasks` 기능 |
| `CreateTaskResult` | 자격 있는 요청에 대한 서버 주도 `resultType: "task"` 응답 |
| `tasks/get` | 최종 결과나 대기 중인 입력을 포함하여 현재 작업 스냅샷 전체를 폴링 |
| `tasks/update` | 작업의 미해결 `inputRequests`에 대한 응답 제출 |
| `tasks/cancel` | 협력적 취소 의도 확인 |
| `input_required` | 클라이언트 입력이 미해결 상태임을 나타내는 작업 상태 |
| `pollIntervalMs` | 다음 폴링 전 서버가 제안하는 최소 지연 시간 |
| `ttlMs` | 작업 생성 시점부터 측정된 만료 기간 |
| Durable-before-return | 핸들 전송 전에 작업 ID가 반드시 해결되어야 하는 규칙 |
| `notifications/tasks` | 구독된 SSE 응답으로 전달되는 선택적 전체 작업 스냅샷 |

## 레거시 호환성

2025-11-25 실험적 인터페이스는 클라이언트 요청 작업 증강, `tasks/status`, `tasks/result`, 선택적 `tasks/list`를 사용했습니다. 이러한 이름은 고정된 레거시 어댑터 내부에서만 유지하세요. 현재 클라이언트는 확장 기능을 사용하며, 서버 주도 핸들을 수락하고, `tasks/get`를 폴링하며, `tasks/update`로 입력을 제공하고, 작업 스냅샷에서 최종 결과를 읽습니다.

## 추가 읽기

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
