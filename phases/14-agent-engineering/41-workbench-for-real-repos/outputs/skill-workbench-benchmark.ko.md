---
name: workbench-benchmark
description: 프로젝트의 자체 샘플 앱에서 동일한 작업을 프롬프트 전용 및 워크벤치 가이드 파이프라인으로 실행하고, 5가지 결과에 대한 사전/사후 보고서를 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 41강
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

저장소, 에이전트 제품, 작은 샘플 앱이 주어졌을 때, 프롬프트 전용 파이프라인과 워크벤치 가이드 파이프라인을 비교하는 이식 가능한 평가 하네스를 생성합니다.

생성물:

1. `eval/sample_app/` — 프로젝트의 도메인에서 추출한 최소한의 실행 가능한 샘플 앱입니다.
2. 각각 작업 설명을 받아 `TaskOutcome`를 반환하는 `eval/run_prompt_only.py` 및 `eval/run_workbench.py`입니다.
3. 두 파이프라인을 실행하고 `before-after-report.md` 및 `comparison.json`를 작성하는 `eval/report.py`입니다.
4. 고정된 작업 세트에서 워크벤치 결과가 후퇴(regress)하면 실패하는 CI 워크플로우입니다.
5. 5가지 결과와 후퇴(regression)로 간주되는 조건을 설명하는 `docs/benchmark.md`입니다.

허용되지 않는 사항:

- 파이프라인이 하나만 있는 벤치마크는 허용되지 않습니다. 비교가 핵심입니다.
- 분모 없이 백분율로 표현된 결과는 허용되지 않습니다. 항상 `n / m`를 보고하세요.
- 에이전트 제품이 학습에 사용한 샘플 앱은 허용되지 않습니다. 도메인에 맞춘 픽스처를 사용하세요.
- 거짓 음(false negatives)을 숨기는 보고서는 허용되지 않습니다. 프롬프트 전용이 더 빨랐던 작업은 반드시 나열해야 합니다.

거절 규칙:

- 프로젝트에 수용 명령(acceptance command)이 없다면 벤치마크 출시를 거절하세요. 측정할 대상이 없습니다.
- 워크벤치 파이프라인이 중앙값 작업에서 프롬프트 전용 파이프라인보다 3배 이상 오래 걸린다면, 그 발견을 드러내세요. 모델이 아니라 워크벤치를 단순화해야 합니다.
- 하네스가 오프라인에서 실행할 수 없다면, CI에 연결하는 것을 거절하세요. 네트워크 불안정성이 비교를 오염시킬 것입니다.

출력 구조:

```
<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml
```

"다음에 읽을 내용"으로 끝내며, 다음을 가리키세요:

- 워크벤치 파이프라인이 사용하는 모든 인터페이스를 묶은 캡스톤 팩이 있는 42강.
- 이 벤치마크가 보완하는 거시 벤치마크(SWE-bench, GAIA, AgentBench)가 있는 19강.
- 벤치마크가 연결된 후의 지속적인 평가 루프에 대한 30강 (평가 주도 에이전트 개발).
