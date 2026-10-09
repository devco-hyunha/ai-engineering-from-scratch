---
name: init-script
description: 프로젝트를 인터뷰하고, 5개의 프로브와 프로브 중 하나라도 실패하면 에이전트 실행을 거부하는 CI 워크플로우를 포함하는 결정적인 init_agent.py를 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 35강
tags: [init, probes, ci, workbench, fail-loud]
---

저장소, 에이전트 제품, 그리고 그 의존성 표면을 고려하여, 프로젝트 전용 init 스크립트와 CI 연결을 생성합니다.

생성물:

1. `tools/init_agent.py`는 다음 프로브를 포함합니다: 런타임 버전, 나열된 의존성, 테스트 명령의 해석 가능성, 필수 환경 변수, 상태 파일의 최신성.
2. `init_report.json` 스키마는 스크립트 옆에 문서화됩니다. 각 프로브는 `(name, status: pass|warn|fail, detail)`를 반환합니다.
3. `.github/workflows/agent-init.yml` (또는 동등한 것)는 스크립트를 실행하고, 심각도(fail-severity) 프로브가 실패하면 에이전트 작업을 차단합니다.
4. 에이전트 런타임이 각 세션 시작 전에 호출할 수 있는 `pre-task` 훅 스크립트.
5. `docs/init.md`에 모든 프로브, 그 심각도, 그리고 실패를 해결하는 방법을 나열한 문서.

엄격한 거부 조건:

- 타임아웃 없이 네트워크를 호출하는 프로브. Init는 빠르고 오프라인 안전해야 합니다.
- LLM 호출이 필요한 프로브. Init는 결정적인(plumbing) 작업입니다.
- 래퍼(wrapper)가 삼켜버리는 비-zero 종료 코드. Loud fail이 핵심입니다.
- 멱등성(idempotency) 없이 상태에 접근하는 프로브. 연속 두 번의 실행은 타임스탬프를 제외하고 동일한 보고서를 생성해야 합니다.

거부 규칙:

- 프로젝트에 테스트 명령이 없다면, 스크립트 출시를 거부합니다. 그 공백을 워크벤치 감사(audit)에 추가합니다.
- 환경 변수 목록에 스크립트가 출력할 비밀(secret)이 포함되어 있다면, 거부하고 강제 가려짐(redaction)을 적용합니다. Init 보고서는 절대 비밀을 포함하지 않아야 합니다.
- 드라이 런(dry run)에서 프로브가 3초 이상 걸린다면, 출시 전에 타이밍 발견 사항을 표면에 드러냅니다. 긴 프로브는 init를 의식(ceremony)으로 만듭니다.

출력 구조:

```
<repo>/
├── tools/
│   ├── init_agent.py
│   └── pre_task.sh
├── docs/
│   └── init.md
└── .github/
    └── workflows/
        └── agent-init.yml
```

"다음에 읽을 것"으로 끝내며, 다음을 가리킵니다:

- 36강: init 보고서의 `repo_paths`를 사용하는 작업별 범위 계약(scope contract)에 대해.
- 37강: 해석된 테스트 명령을 소비하는 런타임 피드백 루프에 대해.
- 38강: 프로브 통과에 의존하는 검증 게이트(verification gate)에 대해.
