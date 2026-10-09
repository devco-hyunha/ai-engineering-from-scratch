# MCP 보안: 오염된 메타데이터, 라우팅 및 MRTR 상태

> 상태 비저장(Stateless)은 신뢰 비저장(Trustless)을 의미하지 않습니다. 모든 요청이 서버와 게이트웨이가 호출을 독립적으로 검증하는 데 필요한 증거를 노출한다는 의미입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 13단계 · 07강 (MCP 서버), 13단계 · 08강 (MCP 클라이언트)
**시간:** 약 60분

## 학습 목표

- 도구 설명, 주석, 클라이언트 정보 및 서버 정보를 신뢰할 수 없는 데이터로 취급해 보세요.
- 메타데이터 오염, 기술자(descriptor) 변경 및 서버 간 이름 충돌을 감지해 보세요.
- 2026-07-28 요청 메타데이터 및 Streamable HTTP 라우팅 헤더를 검증해 보세요.
- MRTR `requestState`을 변조로부터 보호하고 확인을 정확한 인자(arguments)에 바인딩해 보세요.
- 제거된 프로토콜 세션이 아닌 주체(principal)에 권한 부여 및 속도 제한을 적용해 보세요.

## 문제점

모델은 무엇을 호출할지 결정하기 위해 도구 설명을 읽습니다. 라우터는 요청을 어디로 보낼지 결정하기 위해 도구 이름을 읽습니다. 사용자는 무엇을 승인할지 결정하기 위해 레이블을 읽습니다. 하나의 악성 기술자(descriptor)가 이 세 가지를 모두 공격할 수 있습니다.

공식 MCP 보안 지침은 명확합니다. 신뢰할 수 있는 서버에서 제공되지 않는 한, 설명과 주석은 신뢰할 수 없는 데이터로 취급해야 합니다. 그 경우에도 배포 신뢰도는 변할 수 있습니다. 서버 업데이트, 손상된 패키지, 레지스트리 실수 또는 게이트웨이 병합은 모델이 보는 내용을 변경할 수 있습니다.

현재 프로토콜은 보안 경계도 변경합니다. 2026-07-28에는 핵심 핸드셰이크와 전송 세션이 없습니다. 승인, 속도 제한 또는 감사 이력을 `Mcp-Session-Id`에만 키(key)로 사용하는 보안 설계는 현재 설계가 아닙니다.

## 개념

### 점검해야 할 7가지 공격 표면

주의하라는 모호한 지시 대신 구체적인 목록을 사용해 보세요.

1. **메타데이터 오염.** 설명에 선언된 도구 동작과 관련 없는 지시가 포함되어 있습니다.
2. **기술자(descriptor) 러그 풀(rug pull).** 이전에 승인된 이름, 설명, 스키마 또는 주석이 변경됩니다.
3. **서버 간 섀도잉(shadowing).** 두 백엔드가 동일한 비한정(non-qualified) 도구 이름을 노출하며 라우팅이 하나를 조용히 선택합니다.
4. **헤더와 본문 혼동.** `Mcp-Method` 또는 `Mcp-Name`가 JSON-RPC 요청과 일치하지 않습니다.
5. **기능 권한 상승.** 피어가 확장 기능이나 클라이언트 기능을 주장하면 서버가 그 선언을 권한 부여로 오해합니다.
6. **MRTR 상태 변조.** 클라이언트가 `requestState`를 변경하거나, 다른 질문에 답하거나, 확인을 다른 인자와 함께 재사용합니다.
7. **공급망 식별 혼동.** 익숙한 표시 이름을 게시자나 서버의 식별 증명으로 취급합니다.

이 표면들은 겹칩니다. 해시 고정(pinning)은 디스크립터 변경에 도움이 되지만, 첫 번째 디스크립터가 안전했다는 것을 증명하지는 못합니다. 정적 스캐닝은 명백한 문구를 잡지만 미묘한 지시문은 잡지 못합니다. 네임스페이싱은 한 가지 충돌 유형을 방지하지만 악성 네임스페이스 서버는 방지하지 못합니다. 제어 장치를 겹쳐서 적용하세요.

### 현재 요청 엔벨로프는 증거이지 식별이 아닙니다

모든 2026-07-28 요청은 다음을 포함합니다:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

모든 요청에서 버전과 기능 형식을 검증하세요. 기능을 사용하여 호환되는 응답 형식을 선택하세요. `clientInfo`를 인증된 주체로 사용하지 마세요. 이는 자기 보고(self-reported)입니다.

결과 메타데이터의 `io.modelcontextprotocol/serverInfo`에도 동일한 경고가 적용됩니다. 이는 로그와 디버깅에 유용합니다. 이는 인증서, 레지스트리 증명, 또는 권한 부여 결정이 아닙니다.

### 정책 전에 라우팅을 검증하세요

`tools/call`의 경우, Streamable HTTP는 다음을 포함합니다:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

헤더 메서드는 본문 메서드와 같아야 합니다. 헤더 이름은 `params.name`와 같아야 합니다. 백엔드를 선택하기 전, RBAC를 적용하기 전, 속도 제한 토큰을 소비하기 전에 `-32020`와 불일치를 거부하세요.

이 순서는 일반적인 모호성을 해소합니다: 한 구성 요소는 본문을 권한 부여하고 다른 구성 요소는 헤더로 라우팅합니다.

와이어 검증은 하나의 정확한 순서를 따릅니다. JSON-RPC와 메타데이터 유형을 검증하고, 헤더 값을 본문과 비교한 후, 일치하는 버전이 지원되는지 확인하세요. 불일치하는 헤더는 `-32020`와 함께 HTTP 400을 반환합니다. 헤더와 본문이 지원되지 않는 버전에서 일치하면 `-32022`와 `data`가 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`인 HTTP 400을 반환합니다. 알 수 없는 메서드는 `-32601`와 함께 HTTP 404를 반환합니다.

모든 error 객체에는 계약이 구조화된 복구 정보를 필요로 할 때 선택적 `data`이 포함됩니다. 알림(notification)에는 `id`이 없으므로 JSON-RPC 성공 또는 error 응답을 받지 않습니다. 승인된 HTTP 알림은 빈 본문과 함께 202을 반환합니다.

### 전체 descriptor 고정하기

설명 해시만으로는 schema 및 annotation 변경을 놓칠 수 있습니다. 사용자가 승인한 descriptor 필드를 정규화(canonicalize)하고 해시하세요:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

이 단순한 예제 밖에서는 publisher 증거와 승인 시간과 함께 `notes.export` 같은 자격이 부여된 키(qualified key) 아래에 digest를 저장하세요.

모든 refresh 시:

- 알 수 없는 키: 검토할 때까지 격리(quarantine)하세요.
- 동일한 키, 다른 digest: 재승인될 때까지 rug pull로 격리(quarantine)하세요.
- 중복된 비자격(non-qualified) 이름: 결정적인(namespaced) 이름을 요구하세요.
- Scanner 적중: 전체 descriptor를 차단하고 검토하세요.

해시 일치성은 안정성을 증명할 뿐, 안전성을 증명하지는 않습니다. 완벽하게 고정(pinned)된 오염된 descriptor는 오염된 상태로 남습니다.

### 정적 스캐닝은 트리프와이어(tripwire)입니다

단순한 패턴은 role 태그, instruction override, 은폐(concealment), secret 접근, 숨겨진 네트워크 목적지를 플래그할 수 있습니다. 설치 시간과 CI에 사용해도 비용이 충분히 저렴합니다.

이것은 semantic proof가 아닙니다. 안전한 설명은 합법적인 경고에 플래그된 문구를 포함할 수 있습니다. 악성 설명은 모든 문구를 피할 수 있습니다. Scanner 출력은 검토 증거로 취급하고, 자동 무죄 점수로 취급하지 마세요.

### 병합 전에 네임스페이스 지정하기

두 서버가 모두 `search`을 노출한다고 가정해 보세요. discovery 순서가 승자를 결정하도록 절대 방치하지 마세요.

```text
notes.search
issues.search
```

자격이 부여된 이름(qualified name)은 public gateway 이름입니다. backend 매핑은 별도로 기록하세요. 안정적인 이름은 승인, 감사(audit), 해시 고정(hash pin), `Mcp-Name` 라우팅이 동일한 객체를 참조하도록 합니다.

### Capabilities는 호환성 선언입니다

요청별 `clientCapabilities`은 클라이언트가 처리할 수 있는 프로토콜 기능을 서버에 알려줍니다. 클라이언트에 도구, 데이터, 또는 작업에 대한 접근 권한을 부여하지는 않습니다.

Authorization은 여전히 인증된 principal과 resource policy에서 나옵니다. 순서는 다음과 같습니다:

1. 전송(transports) 자격 증명을 인증하세요.
2. 버전, 헤더, 요청 형식(request shape)을 검증하세요.
3. 기능 호환성을 확인합니다.
4. 주체, 도구, 리소스 및 인수를 승인합니다.
5. 실행하거나 사용자 입력을 요청합니다.

### 상태 비저장 MRTR 확인을 보호합니다

중대한 영향을 미치는 도구는 사용자 확인이 필요할 수 있습니다. 현재 MCP는 서버-클라이언트 콜백 대신 다중 왕복 요청(MRTR)을 사용합니다.

첫 번째 응답:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

클라이언트는 입력을 얻고 새로운 JSON-RPC id로 원시 메서드를 재시도합니다:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

각 `inputRequests` 값은 `method` 및 `params`를 포함하는 완전한 내장 요청입니다. 그 키는 `inputResponses`의 해당 항목과 일치해야 합니다. 양식 유도(elicitation)는 객체 루트 `requestedSchema`를 사용하며, 서버가 이를 요청하기 전에 클라이언트가 양식 유도 기능을 선언해야 합니다.

현재 기능에는 두 개의 유효한 양식 선언이 있습니다. `{"elicitation":{}}`은 양식 유도를 암묵적으로 지원하며, `{"elicitation":{"form":{}}}`은 이를 명시적으로 선언합니다. `{"elicitation":{"url":{}}}`와 같은 URL 전용 선언은 양식 요청을 지원하지 않습니다. 서버는 `-32021` 및 `data.requiredCapabilities`가 `{"elicitation":{"form":{}}}`와 동일한 HTTP 400을 반환합니다.

`requestState`를 적대적인 입력으로 취급합니다. 서명하거나 암호화하고, 검증하며, 메서드, 도구, 정확한 인수, 목적, 만료 시간, 주체 및 재현이 중요한 경우 일회성 논스(nonce)에 바인딩합니다. 강의 코드에서는 HMAC과 정확한 인수 매칭을 사용하여 경계를 명확히 합니다.

논스 장부는 하나의 게이트웨이 객체 내부에 존재해서는 안 됩니다. 실행 가능한 모델은 여러 게이트웨이 인스턴스가 공유할 수 있는 제한된 TTL 기반 재생 방지 저장소를 주입합니다. 그 원자적 클레임(claim)은 실행 경계입니다. 검증된 승인이나 명시적인 최종 거절만 상태를 소비합니다. 잘못된 응답이나 `cancel`은 아무것도 실행하지 않으며, 만료 전까지 재시도 가능한 상태로 남습니다. 프로덕션 플릿(fleet)은 공유된 내구성 있는 저장소에서 동일한 조건부 클레임을 필요로 합니다.

숨겨진 확인 컨텍스트를 프로토콜 세션에 저장하지 마세요. 모든 서버 인스턴스는 재시도를 검증할 수 있어야 합니다.

### 고위험 호출에 대한 '두 가지 규칙'

호출을 세 가지 축에 따라 분류합니다:

- 신뢰할 수 없는 입력을 소비합니다.
- 민감한 데이터에 접근할 수 있습니다.
- 중대한 외부 행동을 유발합니다.

단일 자동화 단계가 세 가지를 모두 결합해서는 안 됩니다. 이를 분리하거나, 권한을 줄이거나, MRTR를 통해 명시적인 사용자 입력을 요청하세요. 이는 설계 휴리스틱이며, 프로토콜 기능이 아닙니다.

### 실행 전에 권한을 줄이세요

상태 비저장(Statelessness) 그 자체는 안전성이 아닙니다. 이는 숨겨진 프로토콜 기록을 제거하지만, 자기 완결된 요청은 여전히 과도한 권한을 가진 핸들러에게 데이터 유출이나 되돌릴 수 없는 변경을 요청할 수 있습니다. 안전성은 각 경계에서 권한을 줄이는 데서 나옵니다:

1. **타입 지정된 동사(Typed verb).** `archive_note`와 같은 제한된 연산 하나만 노출하세요. `run`이나 `request`처럼 관련 없는 권한을 표현할 수 있는 범용 도구는 노출하지 마세요.
2. **검증된 인자(Validated arguments).** 가능한 경우 폐쇄된 스키마를 사용하세요. 알 수 없는 필드를 거부하고, 식별자를 한 번만 정규화하며, 크기를 제한하고, 정책 평가 전에 대상, 테넌트 및 리소스 소유권을 검증하세요.
3. **현재 인증(Current authorization).** 인증된 주체(principal)를 정확한 동사, 리소스, 환경 및 정규화된 인자에 바인딩하세요. 도구 주석과 클라이언트 기능은 이 권한을 부여하지 않습니다.
4. **행동 바인딩 승인(Action-bound approval).** 중요한 호출의 경우, 승인에 타입 지정된 동사와 정규화된 인자의 다이제스트, 주체, 만료 기간 및 일회성 정책을 바인딩하세요. 변경된 필드가 있으면 새로운 결정이 필요합니다.
5. **1급 거절(First-class refusal).** 거절, 만료된 승인, 사용자 거부 및 안전하지 않은 대상을 부작용을 실행하지 않는 일반적인 결과로 모델링하세요. 거절을 더 약한 폴백 도구로 변환하지 마세요.
6. **편집된 감사 증거(Redacted audit evidence).** 누가 요청했는지, 어떤 허용된 설명자(descriptor)와 정책 버전이 사용되었는지, 어떤 정규화된 대상이 승인되었는지, 결정이 허용 또는 거절된 이유, 그리고 실행이 시작되었는지 기록하세요. 비밀 대신 다이제스트나 편집된 값을 저장하세요.

각 단계는 다음 구성 요소가 할 수 있는 일을 좁힙니다. 최종 핸들러는 이미 검증된 도메인 명령을 받아야 하며, 원시 모델 텍스트와 광범위한 자격 증명을 받아서는 안 됩니다. MRTR 재시도, 작업 업데이트 또는 게이트웨이 전달 호출 시 전체 체인을 반복하세요. 이전 승인이 이후 요청을 신뢰된 세션 트래픽으로 바꾸지 않습니다.

### 현재 및 레거시 상호작용 경로

Roots, Sampling 및 Logging은 새로운 2026-07-28 구현에 대해 폐기되었습니다. 게이트웨이는 구버전 요청 채널 코드를 버전 게이트된 호환성 경로로만 유지할 수 있습니다.

세션별 샘플링 리미터를 중심으로 새로운 방어 체계를 구축하지 마세요. 인증된 주체(principal), 발급자(issuer), 리소스, 도구, 시간 창에 할당량을 적용하세요. 현재 대화형 작업의 경우, MRTR 입력 요청과 응답을 점검하세요.

### 상태 비저장 전송 검사

- 단일 POST 엔드포인트에서 최신 MCP 메시지를 수락하세요.
- 최신 GET 및 DELETE 요청에 대해 405를 반환하세요.
- `Mcp-Session-Id`을 생성하거나 이에 의존하지 마세요.
- 레거시 세션 및 재생 헤더를 권한 입력으로 취급하지 마세요.
- 해당 POST에 대해 JSON 또는 요청 범위 SSE를 반환하세요.
- `subscriptions/listen`는 장기 지속 변경 알림을 선택적으로 opted-in한 경우에만 사용하세요.

```figure
tp-tool-poisoning
```

## 구현하기

`code/main.py`은 프로세스 내 보안 게이트웨이 모델을 구현합니다. 전체 도구 설명자를 정규화하고 고정하며, 메타데이터 오염 및 shadowing을 보고하고, 최신 요청 엔벨로프 및 라우팅 값을 검증하며, 서명된 `requestState`과 주입된 공유 재생 저장소를 사용하여 2라운드 확인된 export를 수행합니다.

이 모델은 HTTP 어댑터가 JSON 본문과 라우팅 헤더를 파싱한 후 시작합니다. `Content-Type` 또는 `Accept`을 검증하지 않습니다. `Content-Type: application/json`와 `application/json` 및 `text/event-stream`를 모두 포함하는 `Accept` 값이 필요한 09강의 완전한 Streamable HTTP 어댑터에 동일한 dispatcher를 연결하세요.

실행하세요:

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

샘플은 의도적으로 설명자를 변형합니다. 스캐너와 digest 비교는 독립적인 findings를 생성합니다. 이후 export는 `input_required` 응답과 상태 비저장 재시도를 시연합니다.

## 사용하기

`SAFE_TOOLS`을 자체 승인된 서버의 정규화된 snapshot으로 교체하세요. 자격 증명과 비밀은 snapshot에 포함하지 마세요. digest를 업데이트하기 전에 모든 새 설명자나 변경된 설명자를 검토하세요.

게이트웨이에서는 discovery 중과 dispatch 전에 동일한 검사를 수행하세요. 캐시는 discovery 작업을 줄일 수 있지만, 설명자가 변경되면 캐시된 승인 만료되거나 무효화되어야 합니다.

## 출시하기

이 강은 `outputs/skill-mcp-threat-model.md`을 출시합니다. 메타데이터, 라우팅, 기능, 권한 부여, MRTR, 캐싱, 레지스트리 및 호환성 경계에 걸친 현재 프로토콜 위협 모델을 생성합니다.

## 연습 문제

1. 인증된 주체와 현재 권한 부여 결정을 봉인된 MRTR 상태에 바인딩한 후, 다른 주체로 재시도하는 경우 거부하세요.
2. 메모리 내 재생 저장소를 영속적인 조건부 삽입으로 교체하고, 두 프로세스가 하나의 논스를 모두 주장할 수 없음을 증명하세요.
3. 재생 주장 후 시뮬레이션된 내보내기 전에 실패를 주입하세요. 복구를 안전하게 만드는 트랜잭션 또는 멱등성(Idempotency) 규칙을 정의하고 테스트하세요.
4. 도구의 `inputSchema`를 설명 변경 없이 변경하세요. 전체 디스크립터 고정(pinning)이 이를 감지하는지 확인하세요.
5. 주체별로 `tools/list`가 다를 경우 공개 캐싱을 거부하는 정책을 추가하세요.
6. 게이트웨이 뒤에 구형 서버를 모델링하세요. 모든 핸드셰이크 및 세션 동작을 명시적인 `2025-11-25` 호환성 분기 뒤에 배치하세요.

## 핵심 용어

| 용어 | 의미 |
|------|---------|
| 메타데이터 오염 | 도구 디스크립터에 포함된 지침이나 기만적인 주장 |
| 러그 풀(Rug pull) | previously 승인된 디스크립터의 변경 |
| 도구 섀도잉 | 중복된 비수식 이름으로 인한 모호한 라우팅 |
| 헤더 불일치 | 라우팅 헤더와 JSON-RPC 본문 간 불일치, 오류 `-32020` |
| 해시 고정 | 완전히 승인된 디스크립터의 다이제스트 |
| MRTR | 서버가 요청한 입력에 대한 상태 비저장 응답 및 재시도 패턴 |
| `requestState` | 신뢰할 수 없는 입력으로 취급해야 하는 불투명한 왕복 값 |
| 기능 선언 | 프로토콜 호환성에 대한 진술, 권한 부여가 아님 |
| 암시적 양식 지원 | 빈 `elicitation` 기능 객체, 양식 지원과 동일 |
| 수식된 도구 이름 | `notes.search`와 같은 안정적인 게이트웨이 이름 |

## 추가 읽기

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
