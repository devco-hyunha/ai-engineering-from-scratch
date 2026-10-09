---
name: feedback-runner
description: 셸 명령을 결정적인 stdout/stderr/exit/duration 캡처로 감싸고, 명령마다 JSONL 레코드를 저장하며, 피드백이 없으면 에이전트 루프의 진행을 거부합니다.
version: 1.0.0
phase: 14단계
lesson: 37강
tags: [feedback, subprocess, runner, jsonl, loop-control]
---

에이전트 루프 내에서 셸 명령을 실행하는 프로젝트가 주어졌을 때, 피드백 러너와 그것이 작성하는 JSONL을 생성해 보세요.

생성 대상:

1. `tools/run_with_feedback.py`을 노출하는 `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord`입니다.
2. 워크벤치 아래 `feedback_record.jsonl` 위치, 한 줄에 하나의 레코드입니다.
3. 활성 작업에 대해 최근 N개의 레코드를 반환하는 `tools/feedback_loader.py`입니다.
4. 에이전트 루프가 성공을 주장하기 전에 호출하는 `loop_can_advance(record) -> bool` 헬퍼입니다.
5. 테스트는 성공 경로, 비영리 종료, 타임아웃, 바이너리 누락, 결정적인 head/tail 절단을 포함해야 합니다.

엄격한 거부 조건:

- 러너 내의 `shell=True`은 어디에도 허용되지 않습니다. Argv 전용입니다.
- 월 클록(wall clock)이나 랜덤 샘플링에 의존하는 절단은 허용되지 않습니다. 동일한 입력은 동일한 레코드를 생성해야 합니다.
- `duration_ms`이 없는 레코드는 허용되지 않습니다. 느린 프로브는 워크벤치가 멈춘(wedged) 첫 신호입니다.
- 무제한 리스트를 반환하는 로더는 허용되지 않습니다. 마지막 N개로 제한하거나 페이지네이션을 적용하세요.

거부 규칙:

- 프로젝트가 시크릿을 stdout으로 파이프한다면, 마스킹(redaction) 단계 없이 러너를 출시하는 것을 거부하세요. 캡처되었을 라인을 표시하세요.
- 프로젝트가 무한히 걸리는 명령을 포함한다면, 기본 타임아웃과 명시적인 오버라이드 목록 없이 출시하는 것을 거부하세요.
- 러너가 공유 상태를 가진 워커 내부에서 실행된다면, JSONL 추가 작업 주위의 파일 락(file lock)을 건너뛰는 것을 거부하세요. 여러 작성자가 파일을 손상(tear)시킬 것입니다.

출력 구조:

```
<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py
```

다음에 읽을 내용으로 끝내세요. 다음을 가리킵니다:

- 레코드를 소비하는 검증 게이트에 대한 38강입니다.
- 실행 점수를 매길 때 피드백을 읽는 리뷰어 에이전트에 대한 39강입니다.
- 피드백이 탄탄해지면 텔레메트리 측에 추가할 OTel GenAI 컨벤션에 대한 23강입니다.
