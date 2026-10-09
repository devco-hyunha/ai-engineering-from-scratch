---
name: handoff-generator
description: 워크벤치 산출물에서 세션 종료 핸드오프 패킷을 생성합니다. 인간이 읽기 쉬운 Markdown과 기계가 읽기 쉬운 JSON을 생성하며, 7개의 표준 필드를 키로 사용합니다.
version: 1.0.0
phase: 14단계
lesson: 40강
tags: [handoff, generator, session-end, packet, next-action]
---

워크벤치(상태, 판정, 리뷰, 피드백 로그, diff)가 주어지면, 에이전트 런타임에 연결된 세션 종료 핸드오프 생성기를 생성합니다.

생성 대상:

1. `tools/generate_handoff.py`이 `generate_handoff(snapshot) -> (markdown, payload)`을 노출합니다.
2. `outputs/handoff/<session_id>/handoff.md` 및 `handoff.json`.
3. `handoff.schema.json`는 7개의 필수 필드와 피드백 꼬리 형식을 포함합니다.
4. 생성기를 실행하고, 필드가 하나라도 누락되면 세션 종료를 거부하는 세션 종료 훅 스크립트.
5. `docs/handoff.md`는 7개 필드, 그 출처 및 트리밍 정책을 나열합니다.

하드 거부:

- `next_action`이 없는 핸드오프. 핸드오프로 위장한 상태 보고서는 다음 세션을 오염시킵니다.
- 요약을 수동으로 작성하는 생성기. 에이전트의 역할은 생성 가능한 상태로 워크벤치를 남기는 것입니다.
- JSON과 다른 Markdown 패킷. JSON이 소스이며, Markdown은 JSON의 렌더링입니다.
- 30개 항목보다 긴 피드백 꼬리. 전체 로그는 버전 관리에 있으며, 패킷은 작게 유지해야 합니다.

거부 규칙:

- 검증 보고서가 없으면 패킷 생성을 거부합니다. 판정이 없는 핸드오프는 희망 사항에 불과합니다.
- 리뷰 보고서가 없고 인간 리뷰어가 예상되었으면, 거부하고 먼저 리뷰 패스를 요구합니다.
- diff 요약이 비어 있지만 세션이 5분 이상 실행되었으면, 생성 전에 이상 현상을 표시합니다. 실제 무작동(no-op)이 아니라 막힌(wedged) 세션으로 의심합니다.

출력 구조:

```
<repo>/
├── outputs/handoff/<session_id>/
│   ├── handoff.md
│   └── handoff.json
├── tools/generate_handoff.py
├── handoff.schema.json
└── docs/handoff.md
```

"다음에 읽을 것"으로 끝납니다. 다음을 가리킵니다:

- 41강: 실제 스타일의 샘플 앱에 대한 엔드투엔드 연습.
- 42강: 생성기를 캡스톤 워크벤치 팩에 패키징.
- 29강 (프로덕션 런타임): 세션 종료를 큐, 이벤트 및 cron 트리거에 연결.
