---
name: verification-gate
description: 범위, 규칙, 피드백 산출물을 단일 verification_report.json으로 통합하는 결정적 검증 게이트와, 녹색 판정이 없으면 병합을 거부하는 CI 연결을 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 38강
tags: [verification, gate, deterministic, ci, override-log]
---

프로젝트의 수용 기준과 기존 워크벤치 산출물이 주어지면, 검증 게이트와 오버라이드 감사 로그를 생성해 보세요.

생성 대상:

1. `tools/verify_agent.py`는 `verify(task_id, artifacts) -> VerdictReport`을 노출합니다. 순수 함수이며, 결정적이고, LLM 호출이 없습니다.
2. `outputs/verification/<task_id>.json`를 단일 진실 공급원 판정으로 사용하세요.
3. `tools/override.py`는 `outputs/verification/overrides.jsonl`에 서명된 오버라이드 항목을 추가합니다 (사유, 사용자 ID, 타임스탬프, 발견 코드 포함 필수).
4. `passed: false`에서 실패하고 보고서를 인라인으로 표시하는 CI 워크플로우를 생성하세요.
5. 모든 체크, 심각도, 소스 산출물, 오버라이드 정책을 나열하는 `docs/verification.md`를 생성하세요.

하드 거부:

- LLM을 호출하는 체크는 허용되지 않습니다. 게이트는 결정적 플러밍이며, LLM 판단은 리뷰어의 몫입니다.
- 서명된 항목 없이 에이전트가 취할 수 있는 오버라이드 경로는 허용되지 않습니다. 오버라이드는 인간 전용입니다.
- 소비한 산출물 경로를 생략하는 검증 보고서는 허용되지 않습니다. 보고서는 감사 가능해야 합니다.
- 워크플로우가 조용히 심각도를 낮출 수 있는 블록 심각도 발견은 허용되지 않습니다. 심각도는 쓰기 시점에 고정되며, 읽기 시점에 변경되지 않습니다.

거부 규칙:

- 프로젝트에 수용 명령이 없으면, 게이트 출시를 거부하세요. 아무것도 증명하지 못하는 게이트는 연극일 뿐입니다.
- 규칙 보고서가 없으면, 규칙 체크를 건너뛰지 마세요. 실패로 닫히도록 처리하세요.
- 피드백 로그가 없으면, 수용 체크를 건너뛰지 마세요. 누락된 로그 자체가 블록입니다.
- 오버라이드 항목이 버전 관리되지 않으면, 오버라이드 경로 연결을 거부하세요. 기록되지 않은 오버라이드는 게이트를 무력화합니다.

출력 구조:

```
<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml
```

다음에 읽을 내용으로 끝맺으세요:

- 39강: 녹색 판정 이후 이어지는 리뷰어 에이전트.
- 40강: 판정을 패킷에 포함하는 핸드오프 생성기.
- 실제 스타일의 샘플 앱에 대해 게이트를 실행하는 41강입니다.
