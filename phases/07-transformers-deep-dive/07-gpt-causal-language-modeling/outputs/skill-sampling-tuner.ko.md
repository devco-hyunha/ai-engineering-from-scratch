---
name: sampling-tuner
description: 주어진 생성 작업에 적합한 디코딩 전략(`greedy`, `temperature`, `top-k`, `top-p`, `min-p`, `speculative`)을 선택합니다.
version: 1.0.0
phase: 7
lesson: 7
tags: [gpt, sampling, decoding, inference]
---

주어진 생성 작업(코드, 창의적 글쓰기, 추론, 대화, 구조화된 출력)과 지연 시간(latency) 및 품질 목표를 고려하여 다음을 출력하세요:

1. **샘플링 방법(Sampling method)**: 다음 중 하나를 선택하세요: `greedy`, `temperature-only`, `top-k`, `top-p`, `min-p`, `beam-k`, `speculative`. 선택 이유를 한 문장으로 설명하세요.
2. **파라미터 값(Parameter values)**: `temperature`, `top-k`, `top-p`, `min-p`, `repetition penalty` 등 작업 유형에 맞춘 구체적인 수치를 제시하세요. (예: 코딩의 경우 `temperature` 0.2 + `top-p` 1.0, 채팅의 경우 `min-p` 0.1 + `temperature` 0.7)
3. **중단 조건(Stop conditions)**: `max_new_tokens`, 중단 토큰 목록(stop token list), 패턴 기반 중단(예: 닫는 태그 `</tool_call>`)을 포함하세요.
4. **결정론적 여부(Determinism toggle)**: 재현성을 위한 고정 시드(fixed seed) 설정 여부를 표시하세요. 해당 유스케이스(평가, 법률 등)에서 이것이 필요한지 명시해야 합니다.
5. **품질 검사(Quality check)**: 작업 목표에 따른 한 줄 테스트 항목을 제시하세요. (예: 컴파일/단위 테스트 통과 여부, 사실 관계 확인, 형식 유효성 등)

**주의 사항:**
- 구조화된 출력(structured output)이나 코드 완성(code completion) 작업에 대해 `temperature` > 1.0을 추천하지 마세요. 환각(hallucination) 위험이 급격히 높아집니다.
- 개방형 대화(open-ended dialogue)에 순수 `greedy` 방식을 추천하지 마세요. 모델이 무한 루프에 빠질 수 있습니다.
- 모델이 템플릿이나 도구(tools)를 생성해야 하는 경우, 지정된 중단 토큰 목록(stop-token list) 없이 샘플링 설정을 제공하지 마세요.
