---
name: structured-output-designer
description: 자유 텍스트 추출 대상을 위해 엄격 모드(strict-mode) 호환 JSON Schema와 Pydantic 모델을 설계하고, 타입 지정된 거절 처리 및 재시도 핸들링을 포함합니다.
version: 1.0.0
phase: 13
lesson: 04
tags: [structured-output, json-schema, pydantic, strict-mode, extraction]
---

자유 텍스트 추출 대상(청구서, 이력서, 지원 티켓, 연구 요약)이 주어지면, JSON Schema 2020-12, Pydantic 모델, 거절 핸들러, 재시도 정책을 포함한 프로덕션급 추출 계약을 생성해 보세요.

생성 대상:

1. JSON Schema 2020-12. 모든 속성이 타입 지정되어 있습니다. `required`이 모든 속성을 나열합니다. 모든 객체에 `additionalProperties: false`이 있습니다. 닫힌 값 집합에는 Enum을 사용합니다. `$ref`는 없습니다. 모호한 `oneOf` / `anyOf`는 없습니다. OpenAI 엄격 모드(strict-mode) 요구 사항에 대해 검증되었습니다.
2. Pydantic v2 BaseModel. Python 타입으로 스키마를 미러링합니다. `model_json_schema()`은 (1)과 동등한 스키마를 생성해야 합니다.
3. 거절 핸들러. 타입 지정된 `Refusal(reason: str, category: str)` 결과를 생성합니다. 범주를 나열하세요: `safety`, `input_mismatch`, `insufficient_info`.
4. 재시도 정책. 세 가지 재시도 형태: (a) 검증 오류를 주입하고 한 번 재시도(엄격 모드 외부); (b) 거절을 최종 결과로 수용(엄격 모드); (c) 반복적인 거절 시 더 강력한 모델로 에스컬레이션.
5. 테스트 벡터. 해피 패스, 적대적 필드, 부분 입력, 거절 유발 케이스를 포함하는 10개의 입력. 각각에 예상 결과가 있습니다.

하드 리젝트(Hard rejects):
- 타입 지정되지 않은 필드가 있는 모든 스키마. 엄격 모드와 검증기 모두에서 실패합니다.
- `additionalProperties: false`이 누락된 모든 스키마. 환각(Hallucination)을 유출합니다.
- 판별 필드(discriminator field) 없이 `oneOf`을 사용하는 모든 스키마. 모호한 디코딩이 발생합니다.
- JSON Schema 왕복(round-trip) 검사가 없는 모든 Pydantic 모델.

거절 규칙:
- 대상 도메인에 문서화된 목적 없이 개인 식별 데이터가 포함되면, 거절하고 합법적 근거 논의를 위해 18단계(윤리)로 라우팅하세요.
- JSON Schema 2020-12로 표현할 수 없는 스키마(예: 재귀적 임의 그래프)를 사용자가 요청하면, 거절하고 가장 가까운 표현 가능한 완화(relaxation)를 제안하세요.
- 추출 대상이 "모든 것에서 구조화된 데이터를 추출"인 경우, 거절하고 특정 도메인을 요청하세요.

출력: 스키마 JSON, Pydantic 클래스, 거절 및 재시도 정책, 10개의 테스트 벡터를 포함한 한 장의 계약서입니다. 첫 번째로 타겟할 제공자와 그 이유에 대한 노트로 마무리해 보세요.
