# MCP 모델 입력: 샘플링 마이그레이션 및 상태 비저장 MRTR

> MCP 2026-07-28은 새로운 설계에 대해 샘플링(Sampling)을 폐기하고 서버-클라이언트 요청 채널을 제거합니다. 기존 워크플로우가 여전히 클라이언트의 모델이 필요하다면, 서버는 `input_required` 결과를 반환하고 클라이언트는 모델 출력과 함께 원본 요청을 재시도합니다. 추론 루프는 프로토콜 계층에서 명시적이고, 경계가 있으며, 상태 비저장(stateless)이 됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 13단계 · 07강 (MCP 서버), 13단계 · 10강 (리소스 및 프롬프트)
**시간:** 약 75분

## 학습 목표

- MCP 2026-07-28에서 샘플링(Sampling)이 폐기된 이유를 설명하고, 새로운 서버에 대해 직접 모델 통합 기본값을 선택하세요.
- MRTR (다중 왕복 요청)(Multi Round-Trip Request (MRTR))를 통해 `sampling/createMessage`를 전달하는 호환 워크플로우를 구현하세요.
- 모든 요청 `_meta` 객체에 프로토콜 개정판과 클라이언트 기능을 포함하세요.
- `resultType: "input_required"`를 반환하고 새로운 JSON-RPC id로 원본 메서드를 재시도하세요.
- `requestState`의 무결성을 보호하고 주체(principal), 메서드, 인자, 만료 기간에 바인딩하세요.
- 기능 체크, 승인, 응답 검증, 라운드 제한으로 모델 보조 루프에 경계를 설정하세요.

## 프로토콜 이전의 결정

`summarize_repo`와 같은 도구는 두 가지 종류의 작업이 필요합니다:

1. 결정적 작업: 파일 목록, 허용된 파일 읽기, 경로 검증, 콘텐츠 조립.
2. 모델 작업: 대표 파일을 선택하고 요약 합성.

이제 두 가지 유효한 아키텍처가 있습니다.

### 새로운 서버: 모델 제공자와 직접 통합

이것이 현재 기본값입니다. 서버가 모델 선택, 자격 증명, 예산, 재시도, 관측 가능성(Observability)을 소유합니다. MCP 클라이언트에는 하나의 일반 `tools/call` 결과를 반환합니다.

서버가 이미 호스팅 서비스인 경우나 호스트의 모델 사용보다 예측 가능한 모델 동작이 더 중요한 경우 이 옵션을 선택하세요.

### 기존 샘플링(Sampling) 워크플로우: MRTR로 마이그레이션

샘플링은 폐기 기간 동안에도 존재합니다. 2026-07-28을 목표로 하는 서버는 클라이언트로 실시간 `sampling/createMessage` 요청을 보낼 수 없습니다. 대신 해당 요청을 `InputRequiredResult`에 포함합니다.

클라이언트의 모델 및 자격 증명을 사용하는 것이 실제 제품 요구 사항인 경우에만 이 호환성 경로를 선택하세요. 새로운 구현은 폐기된 샘플링을 채택하지 않아야 하므로 제거 계획을 기록하세요.

## 상태 비저장 계약(Stateless Contract)

2026년 7월 프로토콜에는 `initialize` 교환, `notifications/initialized`, `Mcp-Session-Id`가 없습니다. 모든 요청은 과거에 핸드셰이크에 있던 정보를 포함합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

서버는 모든 요청에서 리비전을 검증합니다. 버전이 없거나 문자열이 아닌 경우 유효하지 않은 매개변수이며, `-32602`입니다. 지원되지 않는 문자열은 정확한 데이터 `{"supported":["2026-07-28"],"requested":"<client version>"}`와 함께 `-32022`을 반환합니다. 샘플링 기능이 없는 경우 `data.requiredCapabilities`가 `{"sampling":{}}`로 설정된 `-32021`을 반환합니다.

JSON-RPC `id`가 없는 엔벨로프는 알림(notification)입니다. 수신자는 이를 처리할 수 있지만, 성공 응답이나 오류 응답을 보내지 않습니다. Streamable HTTP 어댑터는 승인된 알림에 대해 본문이 없는 `202 Accepted`을 반환합니다.

서버는 또한 정확한 `supportedVersions` 키, 기능, `ttlMs`, `cacheScope`를 사용하여 `server/discover`를 구현하므로 클라이언트는 도구 호출 전에 서버 계약을 학습하고 캐시할 수 있습니다. 발견(discovery)이 `tools`를 광고하므로 서버는 필수 `tools/list`도 구현합니다. 그 결정론적 `summarize_repo` 설명자는 유효한 객체 `inputSchema`, `resultType: "complete"`, 서버 식별자 메타데이터, 공개 캐시 힌트를 포함합니다.

모든 성공적인 최신 결과에는 판별자(discriminator)가 있습니다:

- `resultType: "complete"`는 작업이 완료되었음을 의미합니다.
- `resultType: "input_required"`는 클라이언트가 내장된 요청을 처리하고 재시도해야 함을 의미합니다.
- 확장 기능은 추가적인 결과 유형을 정의할 수 있습니다. Tasks 확장은 13강에서 `"task"`를 추가합니다.

## 단일 MRTR 라운드

서버는 요청을 처리하는 동안 클라이언트를 호출할 수 없습니다. 대신 이 결과를 반환합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

클라이언트는 샘플링을 지원하는지 확인하고, 승인 및 모델 정책을 적용하며, 모델 응답을 얻습니다. 그런 다음 다른 JSON-RPC id를 사용하여 새 요청을 보냅니다:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

재시도는 프로토콜 세션의 연속이 아닙니다. 원본 메서드와 인수를 반복하고, 현재 라운드의 `inputResponses`만 추가하며, `requestState`을 바이트 단위로 그대로 반영하는 새로운 요청입니다.

MRTR은 `tools/call`, `prompts/get`, `resources/read`에서만 허용됩니다. 서버는 관련 없는 메서드에서 `input_required`을 반환해서는 안 됩니다.

## 다중 라운드 상태

이 강의는 두 번의 모델 호출이 필요합니다:

1. `pick_files`은 JSON 배열을 반환합니다.
2. `summary`은 최종 텍스트를 반환합니다.

각 재시도는 해당 라운드의 응답만 포함합니다. 따라서 서버는 단계와 검증된 중간 데이터를 다음 `requestState`에 넣습니다.

해당 값을 공격자가 제어하는 것으로 취급하세요. 원본 단계 이름에 서명하는 것만으로는 충분하지 않습니다. 상태를 다음에 바인딩하세요:

- 인증된 주체, 자기 보고된 `clientInfo`가 아닙니다;
- 원본 메서드;
- 원본 인수의 다이제스트;
- 짧은 만료 시간;
- 현재 단계와 검증된 중간 값.

기밀성이 필요하지 않은 경우 HMAC을 사용하세요. 클라이언트가 상태를 읽어서는 안 되는 경우 인증된 암호화를 사용하세요. `-32602`으로 잘못된 서명, 만료된 값, 변경된 주체, 변경된 인수를 거부하세요.

클라이언트는 `requestState`을 파싱하거나 수정해서는 안 됩니다. 재시도 시 정확한 문자열을 그대로 반영하는 것이 유일한 역할입니다.

## 모델 선호도는 힌트입니다

`costPriority`, `speedPriority`, `intelligencePriority`는 독립적인 선호도입니다. 확률 분포가 아니며 합이 1이 될 필요도 없습니다. 클라이언트가 모델 정책을 소유하므로 클라이언트가 이를 무시할 수 있습니다.

레거시 샘플링 흐름을 유지한다면 `includeContext`을 `"none"`으로 유지하세요. 다른 컨텍스트 모드는 유출 위험을 높이며 자체적으로 폐기되었습니다. 요청에 최소한의 명시적 컨텍스트를 전달하세요.

## 안전 불변식

임베디드 샘플링 요청에 대해 클라이언트가 신뢰 경계입니다.

- 정책이 승인을 요구할 때, 서버가 모델에게 무엇을 요청하는지 사용자에게 보여 주세요.
- MRTR 라운드 수를 제한하세요. 악성 서버가 모델 소비 루프를 만들 수 있습니다.
- 파일 이름, URL, 도구 입력으로 사용하기 전에 모든 샘플링 응답을 검증하세요.
- 라운드당 바이트 및 토큰 수를 제한하세요.
- 현재 클라이언트 기능에 선언되지 않은 입력 요청을 거부하세요.
- 모델 출력을 권한 결정에 사용하지 마세요.
- 민감한 프롬프트 내용을 기록하지 않으면서, 발신 메서드와 입력 요청 키를 기록하세요.

`clientInfo`와 `serverInfo`는 표시 및 진단 메타데이터입니다. 인증된 식별자로 절대 사용하지 마세요.

```figure
t3-sampling-flip
```

## 구현하기

`code/main.py`는 서드파티 패키지 없이 완전한 2라운드 흐름을 구현합니다:

- `server/discover`는 `supportedVersions`를 반환하고, 도구 지원을 광고하며, 캐시 힌트를 반환합니다.
- `tools/list`는 객체 입력 스키마가 포함된 결정적이고 캐시 가능한 `summarize_repo` 디스크립터를 반환합니다.
- `tools/call`는 요청별 메타데이터를 검증합니다.
- 첫 번째 결과는 파일 선택을 위해 `sampling/createMessage`를 포함합니다.
- 첫 번째 재시도는 모델 결과를 검증하고 두 번째 요청을 포함합니다.
- HMAC으로 보호된 `requestState`는 독립적인 요청 간에 단계를 전달합니다.
- 최종 결과는 `resultType: "complete"`를 사용합니다.

가짜 호스트 모델은 예제를 결정적으로 만듭니다. 실제 호스트에 연결할 때만 `fake_host_model`를 교체하세요. 서버 측 상태 머신은 결정적이고 테스트 가능해야 합니다.

## 사용하기

저장소 루트에서:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

예상 체크포인트:

- 발견은 `ttlMs`와 `cacheScope`를 포함한 완전한 결과를 반환합니다.
- 도구 발견은 `resultType`, 서버 식별자 및 캐시 힌트를 포함한 동일한 정렬된 디스크립터를 반환합니다.
- 누락된 기능 및 지원되지 않는 버전은 정확한 `-32021` 및 `-32022` 오류 데이터를 사용합니다.
- ID가 없는 알림은 JSON-RPC 응답을 생성하지 않습니다.
- 요청 ID는 `[1, 2, 3]`이며, 각 MRTR 라운드가 독립적임을 증명합니다.
- 첫 두 결과는 `input_required`입니다.
- 최종 결과는 `complete`이며 선택된 파일과 요약이 포함되어 있습니다.
- 재시도 시 원본 인수를 변경하면 요청 상태 체크가 실패합니다.

## 출시하기

`outputs/skill-sampling-loop-designer.md`는 이제 마이그레이션 플래너입니다. 먼저 Sampling을 제거하고 직접 모델 통합으로 전환할지 결정합니다. 호환성이 필요한 경우, MRTR 라운드, 상태 바인딩, 기능 게이트, 예산, 검증 및 제거 계획을 생성합니다.

## 연습 문제

1. 파일 선택 응답을 유효하지 않은 JSON으로 변경하세요. 서버가 모델 출력을 신뢰하는 대신 `-32602`을 반환하는지 확인하세요.
2. 첫 번째 호출과 재시도 사이에 `audience`을 변경하세요. 봉인된 상태가 요청 간 재사용을 차단하는 이유를 설명하세요.
3. 호스트가 요약에 대해 비판하도록 요청하는 세 번째 라운드를 추가하세요. 이전 요약을 서명된 상태 안에 전달하고 전체 흐름을 세 라운드로 제한하세요.
4. 가짜 호스트 콜백을 서버 소유의 모델 어댑터로 교체하여 Sampling을 제거하세요. 승인, 청구 및 관측 가능성 책임이 서버로 이동하는 항목을 나열하세요.
5. 마감 시간보다 1초 지난 상태 값을 사용하여 만료 테스트를 추가하세요.

## 핵심 용어

| 용어 | 2026-07-28에서의 의미 |
|------|------------------------|
| Sampling | 클라이언트의 모델에 완성을 요청하는 폐기된 기능 |
| MRTR | 요청 중 클라이언트 입력이 필요할 때 사용하는 상태 비저장 재시도 패턴 |
| `InputRequiredResult` | `resultType: "input_required"`을 포함한 결과 |
| `inputRequests` | 서버가 할당한 임베디드 elicitation, sampling 또는 roots 요청의 맵 |
| `inputResponses` | `inputRequests`처럼 키가 지정된 현재 라운드의 클라이언트 결과 |
| `requestState` | 클라이언트가 정확히 에코하고 서버가 검증하는 불투명한 서버 상태 |
| `resultType` | 최신 MCP 결과에 필수적인 판별자 |
| 직접 모델 통합 | 모델 추론이 필요한 신규 서버에 권장되는 대체 수단 |
| 기능 게이트 | 클라이언트가 광고하지 않은 임베디드 요청을 전송하지 못하게 하는 규칙 |
| 루프 예산 | 작업에 허용되는 최대 라운드 수, 토큰, 바이트, 시간 및 비용 |

## 레거시 호환성

2025-11-25에 고정된 클라이언트는 라이브 연결에서 이전 서버 주도 `sampling/createMessage` 흐름을 여전히 사용할 수 있습니다. 이 동작은 버전별 어댑터에서만 유지하세요. 세션 기반 경로를 2026-07-28 서버의 아키텍처로 만들지 마세요.

공식 SDK는 구형 피어에 대해 최신 `input_required` 핸들러를 변환할 수 있습니다. 그 shim은 호환성 경계이며, 새로운 세션 의존 로직을 추가할 수 있는 권한이 아닙니다.

## 추가 읽기

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
