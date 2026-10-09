# 라이브러리로서의 하네스 — 서브 에이전트와 세션 저장소

> import할 수 있는 하네스: 내장 도구, 컨텍스트 격리를 위한 서브 에이전트, 훅, W3C 추적 전파, 세션 지속성. Claude Agent SDK는 참조 예제입니다 — Claude Code 하네스의 라이브러리 형태이며, Claude Managed Agents는 장기 실행 비동기 작업을 위한 호스팅 대안입니다.

**유형:** Learn + Build
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 10강 (스킬 라이브러리)
**시간:** 약 75분

## 학습 목표

- Anthropic Client SDK (원시 API)와 Claude Agent SDK (하네스 형태)의 차이를 설명해 보세요.
- 서브 에이전트 — 병렬화 및 컨텍스트 격리 — 와 이를 활용해야 하는 시점을 설명해 보세요.
- Python SDK의 세션 저장소 표면 (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`)과 `--session-mirror`의 역할을 나열해 보세요.
- 내장 도구, 격리된 컨텍스트를 가진 서브 에이전트 생성, 생명주기 훅, 세션 저장소를 갖춘 stdlib 하네스를 구현해 보세요.

## 문제점

원시 LLM API는 한 번의 왕복 요청만 제공합니다. 프로덕션 에이전트에는 도구 실행, MCP 서버, 생명주기 훅, 서브 에이전트 생성, 세션 지속성, 추적 전파가 필요합니다. Claude Agent SDK는 이 형태를 라이브러리로 제공합니다 — Claude Code가 사용하는 동일한 하네스를 커스텀 에이전트용으로 노출한 것입니다.

## 개념

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** 원시 Messages API. 루프, 도구, 상태를 직접 관리합니다.
- **Agent SDK (`claude-agent-sdk`).** 내장 도구 실행, MCP 연결, 훅, 서브 에이전트 생성, 세션 저장소. 라이브러리로서의 Claude Code 루프.

### 내장 도구

SDK는 파일 읽기/쓰기, 셸, grep, glob, 웹 가져오기 등 10개 이상의 도구를 기본으로 제공합니다. 커스텀 도구는 표준 도구 스키마 인터페이스를 통해 등록합니다.

### 서브 에이전트

Anthropic이 문서화한 두 가지 목적:

1. **병렬화.** 독립적인 작업을 동시에 실행합니다. "이 20개 모듈 각각에 대한 테스트 파일을 찾아라"는 20개의 병렬 서브 에이전트 작업입니다.
2. **컨텍스트 격리.** 하위 에이전트는 자체 컨텍스트 윈도우를 사용하며, 결과만 오케스트레이터로 반환됩니다. 오케스트레이터의 예산은 보존됩니다.

Python SDK 최신 추가 기능: 하위 에이전트 트랜스크립트 읽기를 위한 `list_subagents()`, `get_subagent_messages()`.

### 세션 저장소

TypeScript와의 프로토콜 호환성:

- `append(session_id, message)` — 턴 추가.
- `load(session_id)` — 대화 복원.
- `list_sessions()` — 열거.
- `delete(session_id)` — 하위 에이전트 세션으로 캐스케이드.
- `list_subkeys(session_id)` — 하위 에이전트 키 목록.

`--session-mirror` (CLI 플래그)는 디버깅을 위해 트랜스크립트를 스트리밍하면서 외부 파일에 미러링합니다.

### 훅

등록할 수 있는 라이프사이클 훅:

- `PreToolUse`, `PostToolUse` — 도구 호출을 게이트하거나 감사.
- `SessionStart`, `SessionEnd` — 설정 및 종료.
- `UserPromptSubmit` — 모델이 보기 전에 사용자 입력에 작용.
- `PreCompact` — 컨텍스트 압축 전에 실행.
- `Stop` — 에이전트 종료 시 정리.
- `Notification` — 사이드 채널 알림.

훅은 프로 워크플로(14단계 커리큘럼 참조) 및 유사한 시스템이 횡단적 동작을 추가하는 방식입니다.

### W3C 추적 컨텍스트

호출자에게 활성인 OTel 스팬은 W3C 추적 컨텍스트 헤더를 통해 CLI 하위 프로세스로 전파됩니다. 전체 다중 프로세스 추적이 백엔드에서 하나의 추적처럼 표시됩니다.

### Claude 관리형 에이전트

호스팅된 대안(베타 헤더 `managed-agents-2026-04-01`). 장기 실행 비동기 작업, 내장 프롬프트 캐싱, 내장 압축. 관리형 인프라를 위해 제어권을 양도합니다.

### 이 패턴이 잘못된 경우

- **하위 에이전트 과다 생성.** 100개의 작은 작업을 위해 100개의 하위 에이전트를 생성합니다. 오버헤드가 지배적입니다. 대신 배치하세요.
- **훅 증가.** 모든 팀이 훅을 추가하면 시작 시간이 급증합니다. 훅을 분기별로 검토하세요.
- **세션 팽창.** 세션이 누적되어 크기가 커집니다. `list_sessions` + 만료 정책을 사용하세요.

```figure
ae-subagent-isolation
```

## 구현하기

`code/main.py`는 stdlib에서 SDK 형태를 구현합니다:

- 내장 `read_file`, `write_file`, `list_dir`가 포함된 `Tool`, `ToolRegistry`.
- `Subagent` — 비공개 컨텍스트, 격리된 실행, 결과 반환.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- 데모: 메인 에이전트(Agent)가 3개의 서브에이전트(Subagent)를 병렬로 생성(각각 격리됨)하고, 결과를 집계하며, 세션을 영속화합니다.

실행해 보세요:

```
python3 code/main.py
```

추적(Trace)은 서브에이전트 컨텍스트 격리(오케스트레이터 컨텍스트 크기가 제한됨), 훅(Hook) 실행, 세션 영속성을 보여줍니다.

## 사용하기

- **Claude Agent SDK**는 Claude Code 하네스(Harness) 형태를 원하는 Claude 우선 제품용입니다.
- **Claude Managed Agents**는 호스팅된 장기 실행 비동기 작업용입니다.
- **OpenAI Agents SDK** (16강)는 OpenAI 우선 대응 제품용입니다.
- **LangGraph + 커스텀 도구(Tool)**는 그래프 형태의 상태 머신(state machine)을 원할 경우 사용합니다.

## 출시하기

`outputs/skill-claude-agent-scaffold.md`는 서브에이전트(Subagent), 훅(Hook), 세션 스토어(Session store), MCP 서버 연결, W3C 추적 전파를 포함하는 Claude Agent SDK 앱을 스캐폴딩(scaffold)합니다.

## 연습 문제

1. 20개 작업을 5개 병렬 서브에이전트(Subagent) 그룹으로 배치하는 서브에이전트 생성기를 추가하세요. 작업당 하나씩 생성하는 경우와 비교하여 오케스트레이터 컨텍스트 크기를 측정하세요.
2. `PreToolUse` 훅(Hook)을 구현하여 `write_file` 호출을 속도 제한(rate-limit)하세요 (세션당 분당 5회). 동작을 추적(Trace)하세요.
3. `list_subkeys`를 연결하여 서브에이전트(Subagent) 트리를 렌더링하세요. 깊은 중첩(nesting)은 어떤 모습인가요?
4. 토이(toy)를 실제 `claude-agent-sdk` Python 패키지로 이식(port)하세요. 도구(tool) 등록에 어떤 변화가 있나요?
5. Claude Managed Agents 문서를 읽어보세요. 자가 호스팅(self-hosted)에서 관리형(managed)으로 언제 전환해야 할까요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | 하네스(Harness) 형태: 도구(Tool), MCP, 훅(Hook), 서브에이전트(Subagent), 세션 스토어(Session store) |
| 서브에이전트(Subagent) | "Child agent" | 분리된 컨텍스트, 자체 예산; 결과가 상위로 버블링 |
| 세션 스토어(Session store) | "Conversation DB" | 서브에이전트(Subagent) 캐스케이드(cascade)와 함께 턴(turn)을 영속화, 로드, 목록화, 삭제 |
| 훅(Hook) | "Lifecycle callback" | 도구(Tool), 세션, 프롬프트 제출, 컴팩트(compact), 중단(stop)의 전/후 |
| W3C 추적 컨텍스트(Trace context) | "Cross-process trace" | 부모 스팬(span)이 CLI 하위 프로세스(subprocess)로 전파됨 |
| 관리형 에이전트 | "호스트형 하네스" | Anthropic이 호스팅하는 장기 실행 비동기 작업 |
| `--session-mirror` | "트랜스크립트 미러" | 스트리밍 중 세션 턴을 외부 파일에 기록 |
| MCP 서버 | "도구 표면" | 에이전트에 연결된 외부 도구/리소스 소스 |

## 추가 읽기

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code의 라이브러리 형태
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 프로덕션 패턴
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 호스팅 대안
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 대응물
