# MCP 서버 구축: 상태 비저장 Python 및 TypeScript

> 현대적인 MCP 서버는 핸드셰이크를 기억하지 않습니다. 모든 요청에서 메타데이터를 검증하고, 하나의 핸들러를 실행하며, 하나의 타입 지정된 결과를 반환합니다.

**유형:** Build
**언어:** Python, TypeScript
**선수 요건:** 13단계, 06강
**시간:** 약 85분

## 학습 목표

- MCP `2026-07-28`에 대해 필수 `server/discover`을 구현합니다.
- 모든 요청에서 프로토콜 버전 및 클라이언트 기능을 검증합니다.
- 도구, 리소스 및 프롬프트를 결정적인 목록 순서로 노출합니다.
- 올바른 결과에 `resultType`, 서버 식별자 및 캐시 힌트를 반환합니다.
- Python과 TypeScript에서 줄바꿈 구분 stdio를 통해 동일한 상태 비저장 계약을 제공합니다.

## 문제점

첫 번째 메시지 이후 클라이언트 기능을 저장하는 서버는 구축하기는 쉽지만 운영하기는 어렵습니다. 동일한 프로세스가 순차적인 클라이언트를 서비스할 수 있습니다. 원격 요청이 다른 워커에 도달할 수 있습니다. 오래된 기능 선언은 권한 경계를 넘어 동작을 유출할 수 있습니다.

MCP `2026-07-28`은 모든 요청을 자기 기술적으로(self-describing) 만듦으로써 이 문제의 프로토콜 부분을 해결합니다. 애플리케이션은 여전히 영구적인 메모, 작업 또는 명시적인 상태 핸들(state handles)을 유지할 수 있습니다. 유지할 수 없는 것은 후속 요청의 디코딩 방식을 변경하는 숨겨진 프로토콜 상태입니다.

이 강의에서는 노트 서버를 두 번 구축합니다. Python 및 TypeScript 버전은 프로토콜 코어에 표준 라이브러리만 사용합니다. 두 버전 모두 동일한 메서드를 노출하고 동일한 와이어(wire) 계약을 적용합니다.

## 개념

### 현대적인 디스패치 루프

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

세 가지 stdio 규칙은 여전히 중요합니다:

- stdout에는 JSON-RPC 메시지만 작성합니다. 진단 정보는 stderr로 전송합니다.
- 메시지를 줄바꿈으로 구분하고 각 응답을 플러시(flush)합니다.
- stdin이 EOF에 도달하면 즉시 종료합니다.

프로세스 수명은 전송 수명입니다. 현대적인 MCP 세션이 아닙니다.

### 요청 검증

모든 요청에는 다음이 필요합니다:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

첫 두 필드는 필수입니다. `clientInfo`는 권장됩니다. 존재하는 신원 형식을 검증하되, 이를 인증으로 취급하지 마세요.

버전이 지원되지 않으면 `requested`과 `supported`를 포함하여 코드 `-32022`를 반환하세요. 요청 메타데이터가 누락된 경우 유효하지 않은 매개변수이며, 코드 `-32602`입니다. 이전 호출에서 누락된 필드를 채우지 마세요.

### 필수 발견

최신 서버는 `server/discover`를 구현해야 합니다. 완전한 발견 결과에는 지원되는 최신 버전, 기능, 선택적 지침, 캐시 힌트 및 결과 `_meta`의 서버 신원이 포함됩니다:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

발견은 서버의 잠금을 해제하지 않습니다. `tools/list`은 이미 동일한 요청 메타데이터를 포함하고 있으므로, 클라이언트는 발견을 호출하지 않고도 `tools/list`를 호출할 수 있습니다.

### 도구

`tools/list`는 도구 설명자의 결정론적 목록을 반환합니다. 안정적인 순서는 응답 캐싱을 개선하고 모델 컨텍스트를 안정적으로 유지합니다. 결과에는 `ttlMs`과 `cacheScope`도 필요합니다.

`tools/call`는 콘텐츠 블록과 `isError`을 반환합니다. 프로토콜 엔벨로프 또는 메서드 매개변수가 유효하지 않은 경우 JSON-RPC 오류를 사용하세요. 유효한 도구 호출이 실행되었지만 도구 자체가 실패한 경우 `isError: true`를 사용하세요.

도구 주석은 강제력이 아닌 힌트로 남아 있습니다:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

호스트는 이를 확인 및 표시에 사용해야 합니다. 서버는 실제 권한 부여를 여전히 강제해야 합니다.

### 리소스

`resources/list`는 안정적인 URI 설명자를 반환합니다. `resources/read`은 타입이 지정된 콘텐츠를 반환합니다. 둘 다 `2026-07-28`에서 캐시 가능하므로, 둘 다 `ttlMs`과 `cacheScope`를 포함합니다.

사용자별 노트 데이터에는 `cacheScope: "private"`를 사용하세요. 공유 캐시는 권한 부여 컨텍스트 간에 개인 응답을 재사용하지 않아야 합니다.

최신 변경 전달은 `resources/subscribe`를 사용하지 않습니다. 클라이언트는 `subscriptions/listen`을 열고 `resourceSubscriptions` 또는 목록 변경 카테고리를 요청합니다. 10강이 이 흐름을 구축합니다.

### 프롬프트

`prompts/list`는 캐시 가능하고 결정론적입니다. `prompts/get`은 인수를 사용하여 이름이 지정된 프롬프트를 렌더링합니다. 렌더링된 프롬프트 결과는 완전하지만, 캐시 힌트가 필요한 캐시 가능한 목록 또는 읽기 결과 중 하나가 아닙니다.

### 모든 성공적인 결과는 타입이 지정되어 있습니다

예제에서는 모든 성공에 대해 하나의 래퍼를 사용합니다:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

목록, 읽기 및 발견 핸들러는 `ttlMs`와 `cacheScope`을 추가합니다. 이 래퍼를 중앙화하면 핸들러가 현대적인 결과 필드를 조용히 생략하는 것을 방지할 수 있습니다.

### 서버가 시작하는 요청 없음

현대적인 서버는 클라이언트 요청과 관련된 알림이나 클라이언트가 연 `subscriptions/listen` 스트림에 대한 알림을 보낼 수 있습니다. 서버는 자체 JSON-RPC 요청을 보내서는 안 됩니다.

핸들러가 샘플링, 유도(elicitation) 또는 루트 입력이 필요할 경우 `input_required` 결과를 반환합니다. 클라이언트는 내장된 입력 요청을 처리하고 새로운 요청 ID로 원본 메서드를 재시도합니다. 11강에서는 다중 왕복 요청(MRTR) 패턴을 다룹니다.

### 명시적인 레거시 호환성

듀얼 에라(dual-era) 서버는 명확하게 분리된 레거시 분기에서 `2025-11-25` 핸드셰이크를 구현할 수도 있습니다. 필수적인 현대적인 `_meta` 필드가 존재하면 현대적인 동작을 선택하고 `initialize`을 받으면 레거시 동작을 선택합니다.

`2026-07-28` 요청을 레거시 핸드셰이크 경로로 보내지 마세요. 레거시 초기화 결과에 현대적인 `resultType` 필드를 찍어 넣지 마세요. 이 강의의 코드는 불변 조건이 명확하게 보이도록 의도적으로 현대적인 전용으로 작성되었습니다.

```figure
t3-dispatch-loop
```

## 사용하기

Python 서버의 유한한 데모와 테스트를 실행하세요:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

TypeScript 러너를 사용하여 TypeScript 포트를 실행하세요:

```bash
npx tsx main.ts --demo
```

데모는 `server/discover`을 보내고, 각 프리미티브를 나열하며, 도구를 호출하고, 지원되지 않는 버전 오류를 표시합니다. 모든 현대적인 요청은 메타데이터를 반복합니다. 모든 성공에는 서버 식별자가 포함됩니다.

## 출시하기

이 강의는 `outputs/skill-mcp-server-scaffolder.md`을 출시합니다. 발견 계약, 요청별 검증, 결정적인 캐시 가능한 목록 및 선택적인 격리된 레거시 어댑터를 포함하는 현대적인 서버 계획을 생성합니다.

## 연습 문제

1. 한 요청에서 기능을 제거하고 서버가 이전 요청의 선언을 재사용하지 않는다는 것을 증명하세요.
2. `TOOLS`, `PROMPTS` 및 노트의 삽입 순서를 역순으로 변경하세요. 모든 목록 결과가 안정적으로 유지되는지 확인하세요.
3. 파괴적인 `notes_delete` 도구를 추가하고 실행기 내부에 권한 확인을 요구하세요. `destructiveHint`는 UX 힌트로만 유지하세요.
4. `resources/templates/list`을 `ttlMs`, `cacheScope` 및 결정적 순서와 함께 추가하세요.
5. `2025-11-25`용 별도의 레거시 어댑터를 구축하세요. 현대적 요청이 이를 절대 거치지 않음을 증명하는 테스트를 추가하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| 상태 비저장 서버 | 프로토콜 세션 메모리 없이 각 요청을 자체 메타데이터로 처리 |
| `server/discover` | 버전 및 기능을 알리는 필수 현대적 메서드 |
| 완전한 결과 | `resultType: "complete"`을 포함한 성공적인 현대적 결과 |
| 캐시 가능한 결과 | `ttlMs` 및 `cacheScope`을 포함한 검색, 목록 또는 리소스 읽기 결과 |
| 결정적 목록 | 동일한 논리적 레지스트리가 동일한 항목 순서를 생성 |
| 서버 식별자 | 결과 `_meta`에서 권장되는 `io.modelcontextprotocol/serverInfo` |
| 도구 오류 | `isError: true`을 포함하는 콘텐츠를 반환하는 유효한 도구 호출 |
| 프로토콜 오류 | `error`을 통해 반환되는 유효하지 않은 JSON-RPC 또는 MCP 요청 |

## 추가 읽기

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
