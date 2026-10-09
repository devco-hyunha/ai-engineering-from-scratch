---
name: claude-agent-scaffold
description: Claude Agent SDK 앱을 하위 에이전트, 라이프사이클 훅, 세션 저장소, MCP 서버 연결, W3C 추적 전파와 함께 스캐폴딩합니다.
version: 1.0.0
phase: 14단계
lesson: 17강
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

제품 도메인과 MCP 서버 목록이 주어지면 Claude Agent SDK 앱을 스캐폴딩합니다.

생성물:

1. 지시문, 내장 도구 접근(read_file, write_file, shell, grep, glob, web fetch), 그리고 사용자 정의 함수 도구를 포함하는 메인 에이전트 정의입니다.
2. 병렬화 및 컨텍스트 격리를 위한 하위 에이전트 스포너입니다. 오케스트레이터가 컨텍스트 예산을 초과할 위험이 있을 때 사용하세요.
3. 등록된 라이프사이클 훅: 감사용 PreToolUse + PostToolUse, 설정용 SessionStart, 종료용 SessionEnd, 규칙 적용용 UserPromptSubmit (pro-workflow 패턴을 참고하세요).
4. 하위 에이전트 트리를 렌더링하도록 `list_subkeys`이 연결된 세션 저장소 (기본값은 SQLite)입니다.
5. 외부 도구/리소스 표면을 위한 MCP 서버 연결입니다.
6. 호출자의 OTel 스팬이 CLI를 통해 계속 이어지도록 W3C 추적 컨텍스트를 전파합니다.

엄격한 거부 조건:

- 단일 도구 작업을 위해 하위 에이전트를 스폰하는 것은 거부합니다. 하위 에이전트는 병렬화나 컨텍스트 격리를 위한 것이며, "read_file 호출 한 번"을 위한 것이 아닙니다.
- 동기식 고비용 작업이 포함된 훅은 거부합니다. 훅은 마이크로초에서 밀리초 단위로 실행되어야 합니다. 긴 작업은 하위 에이전트에서 처리해야 합니다.
- 연쇄 삭제(cascade-delete) 정책이 없는 세션 저장소는 거부합니다. 고립된 하위 에이전트 세션이 저장 공간을 팽창시킵니다.

거부 규칙:

- 제품이 장시간 비동기 작업(수 시간에서 수 일)을 필요로 한다면, 자체 호스팅 SDK를 거부하고 Claude Managed Agents로 라우팅하세요.
- 사용자가 `--session-mirror`을 공유 위치로 요청하면 거부하세요. 세션 트랜스크립트에는 PII가 포함되어 있으며, 사용자별 암호화 저장소에 미러링해야 합니다.
- 도구 사용 없이 UX를 위해 순수 LLM 스트리밍에 의존하는 에이전트라면 Agent SDK를 거부하고 Client SDK를 직접 권장하세요.

출력: `agent.py`, `tools.py`, `hooks.py`, `session.py`, `README.md`는 하위 에이전트 정책, 훅 레지스트리, 세션 백엔드, MCP 첨부 파일, OTel 연결을 설명합니다. "다음에 읽을 내용"으로 마무리하며, 음성 핸드오프는 22강, OTel 스팬 귀속은 23강, 제품이 프로덕션 런타임 형태를 필요로 하는 경우 18강을 가리킵니다.
