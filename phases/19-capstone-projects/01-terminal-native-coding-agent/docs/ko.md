# 캡스톤 01 — 터미널 네이티브 코딩 에이전트

> 2026년 코딩 에이전트의 형태는 확정되었습니다. TUI 하네스, 상태 유지 계획, 샌드박스화된 도구 표면, 계획·실행·관찰·복구를 반복하는 루프. Claude Code, Cursor 3, OpenCode는 모두 50피트 거리에서 동일한 모습입니다. 이 캡스톤은 CLI 입력부터 풀 리퀘스트 출력까지 전체를 직접 구축하고, SWE-bench Pro에서 mini-swe-agent 및 Live-SWE-agent와 비교 측정하는 과제를 제시합니다. 어려운 부분은 모델 호출이 아니라 도구 루프, 샌드박스, 50턴 실행에서의 비용 상한선이라는 점을 배우게 됩니다.

**유형:** Capstone
**언어:** TypeScript / Bun (하네스), Python (평가 스크립트)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구 및 프로토콜), 14단계 (에이전트), 15단계 (자율 시스템), 17단계 (인프라)

**활용 단계:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**시간:** 35시간

## 문제점

코딩 에이전트는 2026년 AI 애플리케이션의 지배적인 카테고리가 되었습니다. Claude Code (Anthropic), Composer 2 및 Agent Tabs가 포함된 Cursor 3 (Cursor), Amp (Sourcegraph), OpenCode (112k 스타), Factory Droids, Google Jules는 모두 동일한 아키텍처의 변형을 출시했습니다: 터미널 하네스, 권한이 부여된 도구 표면, 샌드박스, 프론티어 모델을 중심으로 구축된 계획-실행-관찰 루프. 프론티어는 좁습니다 — Live-SWE-agent는 Opus 4.5로 SWE-bench Verified에서 79.2%를 달성했습니다 — 하지만 엔지니어링 기술은 넓습니다. 대부분의 실패 모드 모델의 실수가 아닙니다. 도구 루프 불안정성, 컨텍스트 오염, 토큰 비용 폭주, 파괴적인 파일 시스템 작업입니다.

이러한 에이전트를 외부에서 추론할 수 없습니다. 직접 구축해야 하며, ripgrep가 8MB의 매칭 결과를 반환할 때 47턴에서 루프가 충돌하는 것을 관찰하고, 절단(truncation) 계층을 재구축해야 합니다. 이것이 이 캡스톤의 요점입니다.

## 개념

하네스에는 네 가지 표면이 있습니다. **계획(Plan)**은 모델이 매 턴마다 재작성하는 TodoWrite 스타일의 상태 객체를 유지합니다. **실행(Act)**은 도구 호출(읽기, 편집, 실행, 검색, git)을 분배합니다. **관찰(Observe)**은 stdout / stderr / 종료 코드를 캡처하고, 잘라내며, 요약본을 피드백합니다. **복구(Recover)**는 컨텍스트 윈도우를 넘치거나 무한 루프에 빠지지 않도록 도구 오류를 처리합니다. 2026년 형태는 한 가지를 더 추가합니다: **훅(hooks)**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, `PreCompact` — 운영자가 정책, 텔레메트리, 가드레일(Guardrails)을 주입할 수 있는 설정 가능한 확장 지점입니다.

샌드박스(Sandbox)는 E2B 또는 Daytona입니다. 각 작업은 읽기-쓰기 권한이 있는 git 워크트리(Worktree)가 마운트된 새로운 devcontainer에서 실행됩니다. 하네스(Agent Harness)는 호스트 파일 시스템에 절대 접근하지 않습니다. 워크트리는 성공 또는 실패 시 해체됩니다. 비용 제어는 세 가지 레이어에서 강제됩니다: 턴별 토큰 상한, 세션별 달러 예산, 그리고 하드 턴 제한(통상 50). 관측 가능성(Observability) 레이어는 GenAI 시맨틱 컨벤션을 사용하는 OpenTelemetry 스팬으로, 자체 호스팅 Langfuse로 전송됩니다.

## 아키텍처

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 스택

- 하네스 런타임: Bun 1.2 + Ink 5 (터미널 내 React)
- 모델 접근: OpenRouter 통합 API로 Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (가장 어려운 작업용) 사용
- 도구 전송: Model Context Protocol StreamableHTTP (MCP 2026 개정판)
- 샌드박스: E2B 샌드박스(JS SDK) 또는 Daytona devcontainer
- 코드 검색: ripgrep 하위 프로세스, 17개 언어용 tree-sitter 파서(사전 컴파일됨)
- 격리: 작업별 `git worktree add`, 성공 / 실패 시 정리
- 평가(Evaluation (Eval)) 하네스: SWE-bench Pro (검증된 하위 집합) + Terminal-Bench 2.0 + 자체 30개 작업 홀드아웃
- 관측 가능성: `gen_ai.*` semconv를 사용하는 OpenTelemetry SDK → 자체 호스팅 Langfuse
- PR 게시: 세밀한 토큰을 사용하는 GitHub App, 범위는 대상 저장소로 제한

```figure
ce-agent-loop
```

## 구현하기

1. **TUI 및 명령 루프.** Ink를 사용하여 Bun 프로젝트를 스캐폴딩하세요. `agent run <repo> "<task>"`을 허용하세요. 분할 뷰를 출력하세요: 계획 패널(상단), 도구 호출 스트림(중간), 토큰 예산(Token Budget)(하단). 종료 전에 `SessionEnd` 훅을 발동하는 Ctrl-C 취소 기능을 추가하세요.

2. **계획 상태.** 타입이 지정된 TodoWrite 스키마(pending / in_progress / done 항목과 메모)를 정의하세요. 모델은 매 턴마다 전체 상태를 도구 호출로 재작성합니다 — 점진적으로 변경하도록 허용하지 마세요. 계획을 `.agent/state.json`에 저장하여 크래시가 발생해도 재개할 수 있도록 하세요.

3. **도구 인터페이스.** 6개의 도구를 정의하세요: `read_file`, `edit_file` (diff 미리보기 포함), `ripgrep`, `tree_sitter_symbols`, `run_shell` (타임아웃 포함), `git` (status / diff / commit / push). MCP StreamableHTTP를 통해 노출하여 하네스가 전송 방식에 의존하지 않도록 하세요. 모든 도구는 잘린 출력(호출당 4k 토큰 상한)을 반환합니다.

4. **샌드박스 래핑.** 각 작업은 E2B 샌드박스를 생성합니다. `git worktree add -b agent/$TASK_ID` 새 브랜치를 생성하세요. 모든 도구 호출은 샌드박스 내부에서 실행됩니다. 호스트 파일 시스템은 접근할 수 없습니다.

5. **훅.** 2026년형 훅 타입 8가지를 모두 구현하세요. 사용자가 작성한 훅을 최소 4개 연결하세요: (a) `PreToolUse` 파괴적 명령 가드, 워크트리 외부의 `rm -rf`를 차단, (b) `PostToolUse` 토큰 회계, (c) `SessionStart` 예산 초기화, (d) `Stop` 최종 추적 번들 작성.

6. **평가 루프.** SWE-bench Pro Python의 30개 이슈 하위 집합을 클론하세요. 각 이슈에 대해 하네스를 실행하세요. mini-swe-agent(최소 기반선)와 pass@1, 작업당 턴 수, 작업당 비용($)을 비교하세요. 결과를 `eval/results.jsonl`에 기록하세요.

7. **비용 제어.** 하드 컷오프: 50 턴, 200k 컨텍스트, 작업당 $5. `PreCompact` 훅은 150k 지점에서 이전 턴들을 prior-state 블록으로 요약하여, 계획을 잃지 않으면서 새로운 관찰을 위한 공간을 확보합니다.

8. **PR 게시.** 성공 시, 마지막 단계는 `git push` + GitHub API 호출로, 본문에 계획과 diff 요약이 포함된 PR을 엽니다.

## 사용하기

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 출시하기

제공된 스킬은 `outputs/skill-terminal-coding-agent.md`에 있습니다. 저장소 경로와 작업 설명이 주어지면, 샌드박스에서 전체 plan-act-observe 루프를 실행하고 PR URL과 추적 번들을 반환합니다. 이 캡스톤의 평가 기준은 다음과 같습니다:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 기반선 | 30개 일치하는 Python 작업에서 하네스 vs mini-swe-agent |
| 20 | 아키텍처 명확성 | plan/act/observe 분리, 훅 인터페이스, 도구 스키마 — Live-SWE-agent 레이아웃과 비교 검토 |
| 20 | 안전성 | 샌드박스 탈출 테스트, 권한 프롬프트, 파괴적 명령 가드, 레드팀 통과 |
| 20 | 관측 가능성 | 추적 완전성 (도구 호출의 100%가 스팬으로 기록됨), 턴별 토큰 회계 |
| 15 | 개발자 UX | 콜드 스타트 < 2초, 크래시 복구 시 계획 재개, Ctrl-C로 도구 실행 중 깔끔하게 취소 |
| **100** | | |

## 연습 문제

1. 백엔드 모델을 Claude Sonnet 4.7에서 vLLM으로 서빙되는 Qwen3-Coder-30B로 교체하세요. pass@1 및 작업당 비용($)을 비교하세요. 오픈 모델이 성능이 떨어지는 부분을 보고하세요.

2. PR 게시 전에 diff를 읽고 수정 루프를 요청할 수 있는 `reviewer` 하위 에이전트를 추가하세요. 오탐 리뷰가 단일 에이전트 기준선보다 SWE-bench pass rate를 낮추는지 측정하세요 (힌트: 보통 낮추는 경우가 많습니다).

3. 샌드박스를 스트레스 테스트하세요: 외부 URL에 `curl`을 시도하는 작업과 워크트리 외부에 쓰는 작업을 작성하세요. 두 경우 모두 PreToolUse hook에 의해 차단되는지 확인하세요. 시도를 로그에 기록하세요.

4. 더 작은 모델(Haiku 4.5)을 사용하여 `PreCompact` 요약 기능을 구현하세요. 3배 압축 시 계획 충실도가 얼마나 손실되는지 측정하세요.

5. MCP StreamableHTTP 트랜스포트 대신 stdio로 교체하세요. 콜드 스타트 및 호출당 지연 시간을 벤치마킹하세요. 로컬 전용 사용에 더 적합한 방식을 선택하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 하네스 | "에이전트 루프" | 모델을 둘러싼 코드로, 도구 분배, 계획 상태 유지, 예산 강제 등을 수행 |
| 훅 | "에이전트 이벤트 리스너" | 하네스가 8가지 생명주기 이벤트 중 하나에서 실행하는 사용자 작성 스크립트 |
| 워크트리 | "Git 샌드박스" | 별도 경로에 연결된 git 체크아웃; 메인 클론을 건드리지 않고 폐기 가능 |
| TodoWrite | "계획 상태" | 모델이 매 턴마다 재작성하는 대기/진행 중/완료 항목의 타입 지정된 목록 |
| StreamableHTTP | "MCP 트랜스포트" | 2026 MCP 개정판: 양방향 스트리밍을 지원하는 장기 지속 HTTP 연결; SSE를 대체 |
| 토큰 상한 | "컨텍스트 예산" | 입력+출력 토큰에 대한 턴별 또는 세션별 상한; 압축 또는 종료를 트리거 |
| pass@1 | "단일 시도 통과율" | 재시도나 테스트셋 확인 없이 첫 실행에서 해결된 SWE-bench 작업의 비율 |

## 추가 읽기

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic의 참고 하네스
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 및 Composer 2 제품 노트
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 하네스 비교를 위한 최소 기준선
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — Opus 4.5로 SWE-bench Verified 79.2% 달성
- [OpenCode](https://opencode.ai) — 오픈 하네스, 112k 스타
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 이 캡스톤이 목표로 하는 평가
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, 기능 메타데이터
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 도구 호출 및 토큰 사용량에 대한 스팬 스키마
