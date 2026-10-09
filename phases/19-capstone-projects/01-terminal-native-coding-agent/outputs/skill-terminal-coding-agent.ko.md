---
name: terminal-coding-agent
description: SWE-bench Pro를 대상으로 제한된 비용, 샌드박스화된 도구, 2026년 전체 hook 표면을 갖춘 터미널 네이티브 코딩 에이전트를 구축하고 평가합니다.
version: 1.0.0
phase: 19단계
lesson: 01강
tags: [capstone, coding-agent, claude-code, swe-bench, mcp, hooks, sandbox]
---

목표 저장소와 자연어 작업이 주어지면, 계획을 수립하고 샌드박스에서 실행하며 풀 리퀘스트를 여는 하네스를 구축하세요. 30개 작업의 SWE-bench Pro 하위 집합에서 mini-swe-agent 기준선과 동등하거나 더 나은 성능을 달성하면서 작업당 $5 예산을 초과하지 않도록 하세요.

구축 계획:

1. 계획 패널, 도구 호출 스트림, 실시간 토큰/달러 예산을 갖춘 Bun + Ink TUI 하네스를 구축하세요.
2. Model Context Protocol StreamableHTTP를 통해 여섯 개의 도구(read_file, edit_file, ripgrep, tree_sitter_symbols, run_shell, git)를 정의하세요. 모든 호출은 최대 4k 토큰을 반환합니다.
3. 모든 도구 호출은 새로운 `git worktree add` 브랜치에서 E2B 또는 Daytona 샌드박스 내에서 실행하세요. 호스트 파일 시스템을 절대 건드리지 마세요.
4. 2026년 hook 이벤트 8개(SessionStart, SessionEnd, PreToolUse, PostToolUse, UserPromptSubmit, Notification, Stop, PreCompact)를 모두 연결하세요. 파괴적 명령 가드, 토큰 회계, OTel 스팬 생성기, 추적 번들 작성기 등 사용자가 작성한 hook을 최소 4개 포함하세요.
5. 세 가지 예산(50 턴, 200k 토큰, $5)을 강제하세요. PreCompact는 150k 토큰에서 발동되어 이전 턴을 요약합니다.
6. GenAI 시맨틱 컨벤션을 사용하여 OpenTelemetry 스팬을 자체 호스팅 Langfuse로 전송하세요.
7. 성공 시, 브랜치를 푸시하고 본문에 계획과 추적 번들을 포함하여 PR을 여세요.
8. 30개 이슈의 SWE-bench Pro Python 하위 집합에서 mini-swe-agent와 비교 평가하고, 작업별 pass@1, 턴 수, 토큰 수, 비용을 기록하세요.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | mini-swe-agent 기준선 대비 30개 작업 하위 집합 일치 |
| 20 | 아키텍처 명확성 | 계획/실행/관찰 분리, hook 표면, 도구 스키마 가독성 |
| 20 | 안전성 | 샌드박스 탈출 레드 티밍 + 파괴적 명령 가드 감사 |
| 20 | 관측 가능성 | 도구 호출의 100%가 스팬으로 기록, 턴별 토큰 회계 |
| 15 | 개발자 UX | 2초 이내의 콜드 스타트, 크래시 복구, Ctrl-C 취소 시맨틱 |

하드 리젝트:

- 샌드박스 내부가 아닌 호스트 파일 시스템에서 git을 호출하는 하네스.
- 명시적인 허용 목록(allowlist) 훅 없이 워크트리 외부에 쓰거나 외부 URL에 curl을 수행할 수 있는 모든 에이전트.
- 동일한 30개 이슈에 대한 매칭된 기준선(baseline) 실행 없이 보고된 평가 수치.
- 재시도 간 `git reset --hard`에 의존하는 "통과율(pass rate)" 주장; SWE-bench Pro는 pass@1입니다.

거부 규칙:

- 어떤 구성에서도 main 브랜치에 직접 푸시하는 것을 거부합니다. PR 브랜치만 허용됩니다.
- 파괴적 명령 가드(destructive-command guard)를 비활성화하는 것을 거부합니다. 이는 루브릭의 필수 요구 사항입니다.
- 예산 상한(budget ceiling) 없이 실행하는 것을 거부합니다. 무제한 실행은 평가 비교를 오염시킵니다.

출력: 하네스를 포함하는 저장소, 매칭된 mini-swe-agent 기준선 실행을 포함하는 고정된 30개 작업의 SWE-bench Pro 평가 하네스, 최소 5회 전체 실행에 대한 OpenTelemetry 추적 아카이브, 그리고 하네스가 해결하지만 기준선이 해결하지 못하는 작업과 그 반대를 명시하는 문서로 구성됩니다. 관찰된 상위 세 가지 실패 모드와 각각을 수정한 훅 변경 사항에 대한 섹션으로 마무리하세요.
