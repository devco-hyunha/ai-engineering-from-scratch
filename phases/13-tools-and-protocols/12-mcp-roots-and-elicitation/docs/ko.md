# 명시적 범위와 상태 비저장 요청

> Roots는 MCP 2026-07-28에서 폐기되었으며, 보안 샌드박스로 사용된 적이 없습니다. 범위를 가시적인 도구 인자나 리소스 URI에 지정하고, 서버에서 이를 승인하며, 도구가 실제로 사용자 입력이 필요한 경우 MRTR을 사용하세요. 사용자는 결정을 보고, 모델은 핸들을 보며, 모든 서버 인스턴스가 재시도를 처리할 수 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계 · 07강 (MCP 서버), 13단계 · 11강 (상태 비저장 MRTR)
**시간:** 약 60분

## 학습 목표

- 폐기된 Roots를 명시적인 워크스페이스 매개변수, 리소스 URI, 또는 서버 구성으로 대체하세요.
- 범위 힌트를 승인, 경로 포함, 운영체제 샌드박싱과 분리하세요.
- 양식 모드 `elicitation/create`을 MRTR `input_required` 결과를 통해 전달하세요.
- 요청별 클라이언트 기능에 요청 elicitation 지원을 광고하고, 지원되지 않는 모드를 거부하세요.
- `accept`, `decline`, `cancel`를 서로 다른 결과로 검증하세요.
- 파괴적 확인을 인증된 주체, 원본 인자, 후보 집합, 만료 시간과 바인딩하세요.

## 유사해 보이는 두 가지 문제

메모 도구에서 이 요청을 받습니다: "오래된 TPS 보고서를 삭제하세요."

서버는 두 가지 다른 질문에 답해야 합니다.

1. 이 작업이 접근할 수 있는 워크스페이스는 어디인가요?
2. 사용자가 세 개의 일치하는 메모 중 어떤 것을 의도했나요?

첫 번째는 범위와 승인입니다. 두 번째는 대화식 모호성 해소입니다. 이 둘을 혼합하면 위험한 설계로 이어집니다. 예를 들어, 클라이언트가 제공한 폴더를 호출자가 그 안의 모든 것을 삭제할 수 있다는 증거로 취급하는 경우입니다.

## Roots는 마이그레이션 표면입니다

이전 MCP 개정판에서는 클라이언트가 Roots를 광고하고 목록이 변경될 때 서버에 알릴 수 있었습니다. Roots는 정보 제공 가이드였습니다. 서버 프로세스가 읽을 수 있는 것을 제한하지 않았고, 호출자를 승인하지 않았으며, 운영체제 샌드박스를 생성하지 않았습니다.

MCP 2026-07-28은 `roots/list`과 `notifications/roots/list_changed`을 새로운 설계에 대해 폐기합니다. 다음 명시적인 대체 수단 중 하나를 선호하세요:

- 호출마다 범위가 달라질 때 `workspaceUri` 또는 `directory` 도구 인자
- 연산이 이미 대상을 지정하고 있을 때의 리소스 URI
- 하나의 배포가 하나의 고정된 워크스페이스를 소유할 때의 서버 구성
- 코드가 기술적으로 탈출할 수 없어야 할 때의 프로세스 샌드박스(Sandbox)나 격리된 파일 시스템

기존 2026-07-28 통합이 폐기 기간 동안 `roots/list`을 계속 필요로 한다면, 서버는 이를 MRTR `inputRequests`에 포함시킵니다. 실시간 역방향 요청을 보내서는 안 됩니다. 이는 마이그레이션 어댑터이며, 새로운 핸들러는 명시적 범위를 받아야 합니다.

모델은 명시적 핸들(handle)을 보고 반복할 수 있습니다. 숨겨진 전송 세션 범위는 검사, 재생, 감사 및 라우팅이 더 어렵습니다.

### 3계층 규칙

명시적 URI는 스스로를 승인하지 않습니다. 세 계층 모두를 강제하세요:

1. **승인(Authorization):** 이 인증된 주체가 이 워크스페이스를 사용할 수 있습니까?
2. **격리(Containment):** 정규화된 대상 URI가 승인된 워크스페이스 경계 안에 남아 있습니까?
3. **샌드박스(Sandbox):** 운영체제가 손상된 서버가 어쨌든 탈출하는 것을 막을 수 있습니까?

실행 가능한 서버는 승인된 워크스페이스 URI의 허용 목록을 유지하고, 퍼센트 인코딩된 경로를 정규화하며, 실제 경로 구성 요소 경계를 확인하고, 삭제 직전에 격리 상태를 재확인합니다.

단순한 문자열 접두어 검사로는 부족합니다:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

두 적대적 경로 모두 오해를 불러일으키는 문자열로 시작합니다. 먼저 정규화한 후 경로 구성 요소를 비교하세요. 프로덕션 파일 시스템 서버는 심볼릭 링크 경쟁 상태 및 플랫폼별 경로 의미론에 대해서도 방어해야 합니다.

## 엘리시테이션(Elicitation)은 여전히 존재하지만, 전달 방식이 변경되었습니다

엘리시테이션(Elicitation)은 `tools/call`, `prompts/get` 또는 `resources/read` 동안 사용자 입력을 수집하기 위한 현재 클라이언트 기능입니다. 메서드 이름은 `elicitation/create`으로 유지됩니다. 변경된 것은 와이어 흐름의 방향입니다.

2026-07-28 서버는 역방향 JSON-RPC 요청을 보내지 않습니다. `InputRequiredResult`을 반환합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

호스트가 양식을 렌더링합니다. 사용자는 이를 수락, 명시적으로 거절하거나 닫을 수 있습니다. 이후 클라이언트는 새로운 id로 원래 `tools/call`을 재시도합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

두 호출 사이에 프로토콜 세션이 없습니다. 서버는 에코된 상태를 검증하고, 응답이 예상 스키마에 맞는지 확인하며, 선택된 노트가 서명된 후보 집합에 포함되었는지 점검하고, 작업 공간을 재승인하고, 포함 관계를 재확인한 후 삭제합니다.

## 기능 협상은 요청별로 수행됩니다

양식 모드 elicitation을 지원하는 클라이언트는 다음을 선언합니다:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

빈 elicitation 기능, `"elicitation": {}`,는 호환성을 위해 양식 전용 지원과 동일합니다. 명시적인 `"elicitation": {"form": {}}`도 양식 모드를 지원합니다. URL 전용 선언인 `"elicitation": {"url": {}}`는 이를 지원하지 않습니다. 서버는 이전 요청에서 광고되었더라도, 현재 요청의 기능에 없는 모드를 포함해서는 안 됩니다.

모든 요청은 `io.modelcontextprotocol/protocolVersion`를 포함합니다. 버전이 없거나 문자열이 아닌 경우 `-32602`를 반환합니다. 지원되지 않는 문자열은 정확한 `supported` 및 `requested` 데이터와 함께 `-32022`를 반환합니다. elicitation 지원이 없거나 URL 전용인 경우 `{"elicitation":{"form":{}}}`으로 설정된 `data.requiredCapabilities`와 함께 `-32021`를 반환합니다.

JSON-RPC `id`가 없는 봉투(envelope)는 알림(notification)입니다. JSON-RPC 성공 또는 오류 응답을 생성하지 않고 처리하세요. Streamable HTTP에서는 승인된 알림이 본문(body) 없이 `202 Accepted`를 받습니다.

진단 purposes를 위해 `clientInfo`를 포함해야 하지만, 이는 자기 보고(self-reported)이므로 사용자 인증을 식별하는 데 사용할 수 없습니다.

서버는 `server/discover`를 구현하고 `supportedVersions`, 기능, `ttlMs`, `resultType: "complete"`를 포함하는 `cacheScope`를 반환합니다. 이 현대적인 설계에서는 Roots를 광고하지 않습니다. 도구를 광고하므로 필수적인 `tools/list`도 구현합니다. 그 결과는 결정론적인 `notes_delete` descriptor, 유효한 객체 `inputSchema`, 서버 식별 메타데이터, 공개 캐시 힌트를 반환합니다.

## 양식 모드

양식 모드는 사용 가능한 대화(dialogs)를 위해 설계된 제한된 JSON Schema를 사용합니다. 루트는 객체이며, 그 속성은 평평한(flat) 원시 필드나 지원되는 enum 배열입니다. 깊게 중첩된 객체와 범용 문서 스키마는 확인 대화에 포함되지 않습니다.

양식 모드를 다음에 사용하세요:

- 여러 후보 중 하나를 선택할 때;
- 파괴적인 작업을 확인할 때;
- 민감하지 않은 선호도를 수집할 때;
- 모델이 아닌 사용자가 결정해야 하는 소수의 값을 모을 때.

비밀번호, API 키, 접근 토큰, 결제 자격 증명을 위해 양식 모드를 사용하지 마세요. 이러한 비밀은 MCP 클라이언트를 통과하여 로그나 모델 컨텍스트에 도달할 수 있습니다.

서버는 반환된 내용을 다시 검증합니다. 클라이언트 측 양식 검증은 UX를 개선하지만 신뢰를 생성하지는 않습니다.

## URL 모드

URL 모드는 외부 상호작용을 위해 안전한 웹 URL을 전송합니다:

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

제3자 인증과 같이 민감한 정보가 서버가 제어하는 웹 흐름으로 직접 전달되어야 할 때 사용하세요. 클라이언트는 전체 목적지를 표시하고 열기 전에 동의를 얻어야 합니다. URL을 사전 가져오기(prefetch)해서는 안 됩니다.

`accept` 응답은 사용자가 URL을 열기로 동의했음을 의미합니다. 외부 흐름이 완료되었음을 증명하지는 않습니다. 재시도 시 서버는 자체 상태를 확인하고 완료하거나 다른 `input_required` 결과를 반환합니다.

URL 유도(elicitation)는 MCP 클라이언트와 MCP 서버 간의 인증을 대체하는 것이 아닙니다. MCP 서버가 사용자를 대신해 수행해야 하는 외부 상호작용을 위한 것입니다. 서버는 브라우저 사용자를 MCP 작업을 시작한 동일한 인증된 주체(principal)와 연결해야 합니다.

## 응답 분기

동작을 별칭이 아닌 제품 결정 사항으로 취급하세요:

| 동작 | 의미 | 안전한 서버 동작 |
|--------|---------|----------------------|
| `accept` | 사용자가 상호작용을 제출함 | 내용을 검증하고 계속 진행 |
| `decline` | 사용자가 명시적으로 거부함 | 완전한 비오류 거부 결과를 반환 |
| `cancel` | 사용자가 닫았거나 완료하지 못함 | 안전하게 중단하고 나중에 재시도 허용 |

누락된 내용을 동의라고 해석하지 마세요. 거부를 반복적인 프롬프트 루프로 변환하지 마세요.

## 파괴적인 MRTR 상태 보호

후보 목록은 프롬프트나 서명되지 않은 Base64 값에만 존재할 수 없습니다. 클라이언트는 보내는 모든 것을 제어합니다.

이 강은 다음을 포함하는 상태 페이로드에 서명합니다:

- 인증된 주체(principal);
- 원래 메서드;
- `workspaceUri` 및 `title`의 다이제스트;
- 양식에 표시된 허용된 노트 ID;
- 작업 단계;
- 짧은 만료 시간.

변경(mutation) 전에 서버는 라이브 노트 기록도 확인합니다. 이를 통해 삭제 경쟁 상태와 양식이 표시된 후 워크스페이스 밖으로 이동한 대상을 포착합니다.

일회성 금융 작업이나 되돌릴 수 없는 작업의 경우, HMAC만으로는 만료 기간 내 유효한 상태의 재생(replay)을 방지하지 못합니다. 모든 핸들러 인스턴스가 공유하는 재생 저장소에 논스(nonce)를 정확히 한 번만 저장하고 소비해야 합니다. 이 강의에서는 제한된 TTL 기반 정리(pruning) 저장소를 주입하고, 메모리 내 삭제 작업을 수행하는 동안 논스의 원자적 클레임(atomic claim)을 유지합니다. 프로덕션 데이터베이스는 논스 클레임과 변경(mutation)을 하나의 트랜잭션이나 동등한 조건부 쓰기 경계로 결합해야 합니다.

논스를 클레임하기 전에 상호작용을 검증합니다. 잘못된 응답이나 `cancel`은 변경(mutation)을 수행하지 않으며, 만료 전까지 상태를 재시도 가능하게 유지합니다. 명시적인 `decline`은 종결 상태이므로, 강의에서는 아무것도 삭제하지 않고 논스를 소비합니다.

```figure
t3-roots-boundary
```

## **구현하기**

`code/main.py`은 현대적인 `notes_delete` 도구를 시연합니다:

- `tools/list`은 필수 워크스페이스 및 제목 스키마를 포함하는 결정적(deterministic)이고 캐시 가능한(descriptor)를 반환합니다.
- 범위(Scope)는 명시적인 `workspaceUri` 인자입니다.
- 서버 구성은 강의 주체(principal)에 대해 해당 워크스페이스를 승인합니다.
- URI 정규화(normalization)는 접두어 혼동 및 인코딩된 경로 순회(traversal)를 거부합니다.
- 모든 파괴적 삭제는 양식 모드(form-mode)의 유도(elicitation)가 필요합니다.
- 유도(elicitation)는 `resultType: "input_required"` 내부에서 전달됩니다.
- 서명된 `requestState`은 정확한 후보 목록과 원본 인자를 바인딩합니다.
- 주입된 재생 저장소는 서버 인스턴스 전반에 걸쳐 동일한 승인 또는 거부 상태를 거부합니다.
- 재시도는 새로운 요청 ID를 사용하며 `resultType: "complete"`을 반환합니다.

데이터 저장소는 메모리에 있으므로 프로토콜 동작을 쉽게 검사할 수 있습니다. 보안 규칙은 데이터베이스를 사용해도 동일합니다.

## **사용하기**

저장소 루트에서:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

예상 체크포인트:

- 디스커버리는 Roots 없이 도구를 광고합니다.
- 도구 디스커버리는 `resultType`, 서버 식별자 및 캐시 힌트를 포함하는 `notes_delete`을 반환합니다.
- 요청 ID `1`은 `inputRequests.delete_choice`에서 양식을 반환합니다.
- 요청 ID `2`은 서명된 상태를 에코(echo)하고 삭제를 완료합니다.
- 접두어 경로와 인코딩된 순회 경로 모두 포함(containment)에 실패합니다.
- 변경된 제목은 원본 확인 상태를 재사용할 수 없습니다.
- 거절하면 노트는 변경되지 않습니다.
- 노트와 재생(replay) 상태를 공유하는 두 서버 객체는 하나의 확인을 동시에 실행할 수 없습니다.
- 빈 양식 선언과 명시적 양식 선언은 작동하며, URL 전용 지원은 정확한 `-32021` 양식 요구 사항을 반환합니다.
- 지원되지 않는 버전 실패는 정확한 `-32022` 데이터 형태를 사용합니다.
- ID가 없는 알림은 JSON-RPC 응답을 생성하지 않습니다.

## 출시하기

`outputs/skill-elicitation-form-designer.md`는 명시적 범위, 권한 확인, MRTR 양식, 응답 분기, 상태 바인딩을 설계합니다. 폐기된 Roots를 샌드박스로 취급하거나 양식 모드를 통해 비밀을 수집하는 것을 거부합니다.

## 연습 문제

1. 메모리 내 재생 스토어를 SQLite로 교체하세요. 하나의 트랜잭션으로 nonce를 클레임하고 노트를 삭제한 후, 두 프로세스가 모두 커밋할 수 없음을 증명해 보세요.
2. `url` 기능 협상과 아웃오브밴드(out-of-band) 설정 흐름을 추가하세요. 서드파티 자격 증명은 `inputResponses`에 포함되지 않도록 유지하세요.
3. 메모리 내 노트 맵을 임시 SQLite 데이터베이스로 교체하세요. 변경 트랜잭션 내에서 권한과 포함(containment)을 다시 확인하세요.
4. 실제 파일 시스템 구현을 위해 심볼릭 링크 정책을 추가하세요. URI 어휘적 포함(containment)만으로는 심볼릭 링크 탈출을 막을 수 없는 이유를 설명하세요.
5. 현대적인 MRTR 핸들러 출력을 레거시 서버 주도 elicitation에 매핑하는 2025-11-25 어댑터를 설계하세요. 현재 핸들러와 격리된 상태로 유지하세요.

## 핵심 용어

| 용어 | 2026-07-28에서의 의미 |
|------|------------------------|
| Roots | 폐기된 정보성 워크스페이스 힌트이며, 권한 부여나 샌드박싱이 아님 |
| 명시적 범위 | 요청 인자에서 보이는 워크스페이스, 디렉터리, 또는 리소스 핸들 |
| 포함 (Containment) | 대상을 경계 내부에 유지하는 정규화된 경로 구성 요소 검사 |
| Elicitation | MCP 작업 중 사용자 입력을 얻기 위한 클라이언트 기능 |
| 양식 모드 | 제한된 평면 스키마를 사용하는 인밴드(in-band) 구조화된 사용자 입력 |
| URL 모드 | 민감하거나 외부적인 워크플로우를 위한 아웃오브밴드(out-of-band) 상호작용 |
| MRTR | 상태 비저장 입력 필요 결과 후 새로운 재시도 |
| `requestState` | 서버가 정확히 에코하고 무결성을 검증하는 불투명 상태 |
| Decline | 명시적 사용자 거부 |
| Cancel | 승인 없이 기각되거나 불완전한 상호작용 |

## 레거시 호환성

2025-11-25에 고정된 피어에 대해 `roots/list`, `notifications/roots/list_changed` 및 라이브 서버 주도 `elicitation/create`가 여전히 존재할 수 있습니다. 해당 어댑터를 레거시로 표시하세요. 레거시 Root 목록이 서버 인증을 우회하지 않도록 허용하지 마세요. 그리고 프로토콜 세션 가정을 최신 핸들러로 전달하지 마세요.

## 추가 읽기

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
