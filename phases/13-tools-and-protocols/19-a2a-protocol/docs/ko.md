# A2A — 에이전트 간 프로토콜

> MCP는 에이전트-도구 간입니다. A2A (Agent2Agent)는 에이전트-에이전트 간입니다. 서로 다른 프레임워크로 구축된 불투명한 에이전트들이 협력할 수 있도록 하는 개방형 프로토콜입니다. Google이 2025년 4월에 공개했으며, 2025년 6월에 Linux Foundation에 기부되었고, AWS, Cisco, Microsoft, Salesforce, SAP, ServiceNow 등 150개 이상의 지원사를 포함하여 2026년 4월에 v1.0에 도달했습니다. IBM의 ACP를 흡수하고 AP2 결제 확장 기능을 추가했습니다. 이 강의에서는 A2A 1.0.1의 와이어 이름을 사용하여 에이전트 카드, 작업(Task) 생명주기 및 세 가지 프로토콜 바인딩을 살펴봅니다.

**유형:** Build
**언어:** Python (표준 라이브러리, 에이전트 카드 + 작업(Task) 하네스)
**선수 요건:** 13단계 · 06강 (MCP 기초), 13단계 · 08강 (MCP 클라이언트)
**시간:** 약 75분

## 학습 목표

- 에이전트-도구(MCP)와 에이전트-에이전트(A2A) 사용 사례를 구분해 보세요.
- `/.well-known/agent-card.json`에 스킬과 `supportedInterfaces` 메타데이터를 포함하여 에이전트 카드를 게시해 보세요.
- 작업(Task) 생명주기를 따라가 보세요: `TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_INPUT_REQUIRED`, 그리고 종단 상태 `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`, `TASK_STATE_REJECTED`.
- 각 Parts가 `text`, `raw`, `url` 또는 `data` 중 하나를 포함하는 Messages를 사용하고, Artifacts를 출력으로 사용해 보세요.

## 문제점

고객 서비스 에이전트가 보고서 작성을 전문 작성 에이전트에게 위임해야 합니다. A2A 이전의 선택지:

- 사용자 정의 REST API. 작동하지만 모든 페어링이 일회용입니다.
- 공유 코드베이스. 두 에이전트가 동일한 프레임워크를 실행해야 합니다.
- MCP. 적합하지 않습니다: MCP는 도구 호출용이며, 각 에이전트의 불투명한 내부 추론을 보존하면서 두 에이전트가 협력하는 용도에는 맞지 않습니다.

A2A가 이 공백을 채웁니다. 한 에이전트가 다른 에이전트에게 작업(Task)을 보내는 상호작용을 모델링하며, 생명주기, 메시지 및 아티팩트를 포함합니다. 호출된 에이전트의 내부 상태는 불투명하게 유지되며, 호출자는 작업 상태 전환과 최종 출력만 볼 수 있습니다.

A2A는 "프레임워크 간 에이전트들이 서로 대화할 수 있게 하는" 프로토콜입니다. MCP를 대체하지 않으며, 두 프로토콜은 상호 보완적입니다.

## 개념

### 에이전트 카드

모든 A2A 호환 에이전트는 `/.well-known/agent-card.json`에 카드를 게시합니다:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

발견은 URL 기반입니다: 카드를 가져오고, `protocolBinding`이 클라이언트가 사용하는 `supportedInterfaces` 항목을 선택하여 스킬을 나열합니다. 입력 및 출력 모드는 미디어 유형입니다.

### 서명된 에이전트 카드

카드는 `signatures` 배열을 포함할 수 있습니다. 각 항목은 카드의 RFC 8785 표준 JSON에 대해 `signatures` 필드를 제외하고 계산된 JWS(RFC 7515)입니다. 소비자는 카드를 동일한 방식으로 표준화하고 검증합니다. 사칭을 방지합니다.

### 작업 수명 주기

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

클라이언트는 `SendMessage`로 시작하며, 서버는 Task를 생성합니다. 호출된 에이전트는 상태를 전환하며; 클라이언트는 `GetTask`로 폴링하거나 `SendStreamingMessage` 및 `SubscribeToTask`를 통해 SSE로 스트리밍합니다. 스트림은 `statusUpdate` 및 `artifactUpdate` 이벤트를 포함하며, 작업이 종결 상태에 도달하면 닫힙니다. `final` 플래그는 없습니다.

### 메시지와 Parts

메시지는 `messageId`, `role` (`ROLE_USER` 또는 `ROLE_AGENT`), 그리고 하나 이상의 Parts를 포함합니다. 각 Part는 정확히 하나의 콘텐츠 필드를 포함하며, 그 필드 이름이 유형입니다. `kind` 필드는 없습니다.

- `text`: 일반 콘텐츠.
- `raw`: 파일 바이트, JSON에서 base64 인코딩, 일반적으로 `filename` 및 `mediaType` 포함.
- `url`: 파일 콘텐츠에 대한 링크.
- `data`: 구조화된 JSON 페이로드 (호출된 에이전트를 위한 구조화된 입력).

예시:

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Artifacts

출력은 Artifacts이며, 원시 문자열이 아닙니다. Artifact는 이름이 지정된 유형화된 출력입니다:

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Artifacts는 청크로 스트리밍될 수 있습니다. 각 `artifactUpdate` 이벤트는 아티팩트와 `append` 및 `lastChunk`를 포함합니다. 호출자는 이를 누적합니다.

### 세 가지 프로토콜 바인딩

1. **HTTP를 통한 JSON-RPC 2.0** (`JSONRPC`). 요청은 POST, 스트리밍은 SSE를 사용합니다. 메서드는 PascalCase입니다: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, `CreateTaskPushNotificationConfig`, `GetTaskPushNotificationConfig`, `ListTaskPushNotificationConfigs`, `DeleteTaskPushNotificationConfig`, `GetExtendedAgentCard`.
2. **gRPC** (`GRPC`). gRPC가 네이티브인 엔터프라이즈 환경을 위한 것입니다. 동일한 메서드 이름을 사용합니다.
3. **HTTP+JSON/REST** (`HTTP+JSON`). `POST /message:send` 및 `GET /tasks/{id}`와 같은 리소스 URL을 사용합니다.

세 가지 바인딩 모두 동일한 데이터 모델을 사용합니다. 각 `supportedInterfaces` 항목은 하나의 바인딩과 그 `protocolVersion`을 지정합니다. 클라이언트는 모든 요청에 `A2A-Version: 1.0` 헤더를 포함하여 전송해야 합니다. 서버가 이 헤더가 없는 요청을 버전 0.3으로 해석하기 때문입니다.

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### 투명성 보존

핵심 설계 원칙: 호출된 에이전트의 내부 상태는 투명하지 않습니다. 호출자는 작업 상태와 산출물만 볼 수 있습니다. 호출된 에이전트의 사고의 연쇄 (CoT)(Chain of Thought), 도구 호출, 하위 에이전트 위임 등은 모두 볼 수 없습니다. 이는 도구 호출이 투명한 MCP와 다릅니다.

이유: A2A는 경쟁사가 내부 구조를 공개하지 않고도 협력할 수 있게 합니다. A2A는 "이 고객 서비스 에이전트를 호출하라"는 식으로 호출할 수 있으며, 호출자는 해당 에이전트가 서비스를 어떻게 구현하는지 알 수 없습니다.

### 타임라인

- **2025-04-09.** Google이 A2A를 발표합니다.
- **2025-06-23.** Linux Foundation에 기부됩니다.
- **2025-08.** IBM의 ACP를 흡수합니다.
- **2025-09.** AP2 확장(Agent Payments)이 출시됩니다.
- **2026-04.** 150개 이상의 지원 조직과 함께 v1.0이 릴리스됩니다.

### MCP와의 관계

| 차원 | MCP | A2A |
|-----------|-----|-----|
| 사용 사례 | 에이전트-도구 | 에이전트-에이전트 |
| 투명성 | 투명한 도구 호출 | 불투명한 내부 추론 |
| 일반적인 호출자 | 에이전트 런타임 | 다른 에이전트 |
| 상태 | 도구 호출 결과 | 생명주기를 가진 작업 |
| 인증 | OAuth 2.1 (13단계 · 16) | 에이전트 카드 `securitySchemes` + `securityRequirements` |
| 전송 | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

특정 도구를 호출하려면 MCP를 사용하세요. 전체 작업을 다른 에이전트에게 위임하려면 A2A를 사용하세요. 많은 프로덕션 시스템은 둘 다 사용합니다. 에이전트는 도구 계층에는 MCP를, 협력 계층에는 A2A를 사용합니다.

```figure
a2a-task-lifecycle
```

## 사용하기

`code/main.py`은 최소한의 A2A 하네스(Agent Harness)(Agent Harness)를 구현합니다. 작성자 에이전트가 카드를 게시하고, 연구 에이전트가 PDF 부분과 텍스트 지시문을 포함한 `SendMessage` 요청을 보내며, 작업이 `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED`를 거쳐 텍스트 산출물을 반환합니다. 모두 표준 라이브러리만 사용하며, 메시지 형태에 집중하기 위해 메모리 내 전송을 사용합니다.

확인할 사항:

- 에이전트 카드 JSON 형식.
- 서버 측 작업 ID 할당 및 상태 전환.
- 콘텐츠 필드의 존재 여부에 따라 유형이 결정되는 Parts.
- `TASK_STATE_INPUT_REQUIRED` 분기 중 작업.
- 완료 시 아티팩트 반환.

## 출시하기

이 강은 `outputs/skill-a2a-agent-spec.md`를 생성합니다. 다른 에이전트가 호출할 수 있는 새로운 에이전트가 주어지면, 이 스킬은 에이전트 카드 JSON, 스킬 스키마 및 엔드포인트 청사진을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 호출된 에이전트가 명확한 설명을 요청하는 `TASK_STATE_INPUT_REQUIRED` 일시 중지를 포함하여 전체 작업 수명 주기를 추적하세요.

2. 서명된 에이전트 카드를 추가하세요. `signatures`에 JWS 항목을 하나 넣고 `alg`을 `HS256`로 설정하여 `signatures` 필드를 제외한 카드의 표준 JSON을 서명하세요. 검증기를 작성하고 변형된 카드에서 실패하는지 확인하세요.

3. `SendStreamingMessage`를 사용하여 작업 스트리밍을 구현하세요: 작성자 에이전트가 `task`, 세 개의 `artifactUpdate` 청크 및 `TASK_STATE_COMPLETED`가 포함된 `statusUpdate`를 방출한 후 스트림을 닫습니다. 호출자는 청크를 누적합니다.

4. MCP 서버를 래핑하는 A2A 에이전트를 설계하세요. 각 MCP 도구를 A2A 스킬에 매핑하세요. 트레이드오프를 고려하세요 — 어떤 투명성이 손실됩니까?

5. A2A v1.0 발표를 읽고, 2026년 4월 기준으로 어떤 프레임워크도 구현하지 않은 기능을 식별하세요. (힌트: 다중 홉 작업 위임과 관련됩니다.)

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| A2A | "에이전트 간 프로토콜" | 불투명한 에이전트 협업을 위한 개방형 프로토콜 |
| 에이전트 카드 | "`/.well-known/agent-card.json`" | 에이전트의 스킬 및 `supportedInterfaces`을 설명하는 공개된 메타데이터 |
| 스킬 | "호출 가능한 단위" | 에이전트가 지원하는 명명된 작업 (MCP 도구와 유사) |
| 작업 | "위임 단위" | 수명 주기와 최종 아티팩트를 가진 작업 항목 |
| 메시지 | "작업 입력" | Parts (`text`, `raw`, `url`, `data`)를 전달 |
| Part | "유형이 지정된 청크" | `text` / `raw` / `url` / `data` 중 정확히 하나, 선택적 `mediaType` 포함; `kind` 필드 없음 |
| 아티팩트 | "작업 출력" | 완료 시 반환되는 명명된 유형이 지정된 출력 |
| AP2 | "에이전트 결제 프로토콜" | A2A 위에 구축된 결제 확장; 카드 서명은 핵심 A2A (`signatures`) |
| Opacity | "블랙박스 협업" | 호출된 에이전트의 내부가 호출자에게 숨겨짐 |
| `TASK_STATE_INPUT_REQUIRED` | "작업 일시중지" | 에이전트가 추가 정보가 필요할 때의 중단 상태 |

## 추가 읽기

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 공식 A2A 사양
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 참조 구현 및 SDK
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1): 이 강의가 따르는 태그된 `docs/specification.md` 및 규범적 `specification/a2a.proto`
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025년 6월 거버넌스 이관
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 로드맵 및 파트너 모멘텀
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 릴리스 노트 및 하위 호환성 가이드
