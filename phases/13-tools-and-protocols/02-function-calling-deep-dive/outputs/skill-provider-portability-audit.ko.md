---
name: provider-portability-audit
description: 한 제공자에 대한 함수 호출 통합을 감사하여, 다른 두 제공자로 이식할 때 깨지는 부분을 나열합니다.
version: 1.0.0
phase: 13단계
lesson: 02강
tags: [function-calling, openai, anthropic, gemini, portability]
---

한 제공자(OpenAI, Anthropic, Gemini)의 함수 호출 통합이 주어졌을 때, 동일한 로직을 다른 두 제공자로 출시할 때 나타나는 모든 필드 이름 변경, 동작 차이, 하드 한도 충돌을 나열한 이식성 감사 보고서를 작성해 보세요.

다음 내용을 작성해 보세요:

1. 선언 차이. 통합된 각 도구에 대해, 다른 두 제공자 각각에 필요한 인velope / 필드 이름 변경 / 스키마 변환을 보여 주세요. 대상 제공자가 지원하지 않는 JSON Schema 구성 요소를 표시하세요 (Gemini: OpenAPI 3.0 하위 집합; OpenAI strict: `$ref` 없음, 모호한 `oneOf` 없음).
2. 응답 차이. 각 제공자의 응답 형식에서 도구 호출이 어디에 위치하는지 (`tool_calls[]` vs `content[]` 블록 vs `parts[]` 항목)와 `arguments`을 파싱하는 주체 (OpenAI는 문자열, Anthropic과 Gemini는 객체)를 문서화하세요.
3. `tool_choice` 차이. 통합의 현재 선택 설정 (auto / forbid / force / required)을 대상 제공자의 형식에 매핑하세요. 누락된 모드를 표시하세요.
4. 한도 충돌. 도구 수 (128 / 64 / 64), 스키마 깊이 (5 / 10 / 사실상 무제한), 인자별 길이 상한을 보고하세요. 대상 제공자의 한도를 초과하는 통합에 대해 블록 심각도를 높여 주세요.
5. Strict 모드 매핑. 대상에서 strict 모드 의미가 보존되는지 명시하세요. OpenAI `strict: true`는 Anthropic에 정확한 대응이 없습니다. Gemini `responseSchema`는 근사하지만 요청 수준입니다.

거부 조건:
- 비 OpenAI 대상에서 `arguments`가 문자열이라고 가정하는 모든 통합. 조용히 잘못된 결과를 생성합니다.
- 라우터 없이 Anthropic이나 Gemini로 이식할 때 도구 수가 64를 초과하는 모든 통합.
- 대상인 OpenAI strict 모드에서 스키마에 `$ref`를 사용하는 모든 통합.

거부 규칙:
- 제공자 고유 기능에 의존하는 통합을 이식하라고 요청받으면 (예: OpenAI Responses API의 상태 유지 턴, Anthropic의 computer-use 블록) 거부하고, 대상에 대응 기능이 없는 기능을 설명하세요.
- 승자를 선택하라고 요청받으면 거부하세요. 선택은 호스트의 strict-mode 요구 사항, 비용 프로필, 병렬 호출 요구 사항에 따라 달라집니다.

출력: 도구별 diff 표, 한계 표, 각 대상 제공자에 대한 최종 "이식 판정"(출시 / 라우터 필요 / 기능 차단)이 포함된 한 페이지 감사 보고서. 가장 영향력이 큰 이식 변경 사항을 한 문장으로 명시하여 마무리하세요.
