---
name: ecosystem-blueprint
description: 제품 요구 사항이 주어지면 전체 13단계 생태계 아키텍처를 산출합니다. 프리미티브, 보안 태세, 텔레메트리, 패키징을 지정합니다.
version: "1.0.0"
phase: "13"
lesson: "23"
tags: [mcp, capstone, ecosystem, architecture, a2a, otel]
---

제품 요구 사항(연구, 요약, 자동화, 에이전트 주도 워크플로우 등)이 주어지면 전체 아키텍처를 산출합니다.

산출물:

1. MCP 표면. `server/discover`, 요청별 프로토콜 메타데이터, 도구, 리소스, 프롬프트, 캐시 정책을 정의합니다. `ui://` Apps를 지정합니다.
2. 확장. 작업이 비동기인 경우 `io.modelcontextprotocol/tasks`를 선언하고 `tasks/get`, `tasks/update`, `tasks/cancel`를 설계합니다. 초기 핸들(handle)은 `resultType: task`로 유지하고, 폴링(polling) 결과는 `resultType: complete`로 처리하며, `tasks/result` 또는 `tasks/list`는 사용하지 않습니다.
3. 보안 태세. OAuth 2.1 범위(scope) 세트, 게이트웨이 RBAC 매트릭스, 고정된 해시 매니페스트, Rule of Two 감사.
4. A2A 협업. 하위 에이전트 호출을 식별합니다. 해당 에이전트 카드(Agent Cards)를 정의합니다.
5. 텔레메트리. OTel GenAI 스팬(span) 계층 구조. Exporter 및 백엔드 선택.
6. 패키징. AGENTS.md, SKILL.md, 배포 표면(Docker Compose, K8s).
7. 13단계 강의 매핑. 각 설계 선택이 어떤 강의로 거슬러 올라가는지.

하드 리젝트(Hard rejects):
- 신뢰할 수 없는 입력, 민감한 데이터, 중대한 조치를 단일 턴(turn)에 결합하는 모든 아키텍처(Rule of Two).
- MCP 및 A2A 홉(hop) 간 추적(trace) 전파가 없는 모든 아키텍처.
- LLM 계층에 최소한 하나의 대체(fallback) 제공자가 없는 모든 아키텍처.
- `initialize`, `Mcp-Session-Id`, `tasks/result`, `tasks/list`에 의존하는 모든 현재 MCP 설계.

거절 규칙:
- 제품 요구 사항이 직접 LLM 호출로 더 잘 충족되는 경우, 전체 생태계 스캐폴딩(scaffolding)을 거절합니다.
- 팀이 게이트웨이 운영 역량을 갖추지 못한 경우, 관리형(managed) 게이트웨이를 권장하고 신뢰 이전(trust transfer)을 문서화합니다.
- 아키텍처에 결제가 포함되는 경우, 별도로 검토된 결제 승인 프로토콜과 명시적인 서명(signoff)을 요구합니다.

출력: 기본 요소, 보안 태세, A2A 홉, 텔레메트리 계획, 패키징, 강의 맵을 담은 한 장의 청사진입니다. 배포에서 가장 어려운 운영 리스크를 한 문장으로 식별하며 마무리해 보세요.
