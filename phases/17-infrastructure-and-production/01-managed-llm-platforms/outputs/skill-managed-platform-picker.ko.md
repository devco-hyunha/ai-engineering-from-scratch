---
name: managed-platform-picker
description: 워크로드, SLA, 컴플라이언스 요구 사항에 따라 관리형 LLM 플랫폼(Bedrock, Azure OpenAI, Vertex AI)과 이중화를 위한 두 번째 플랫폼을 선택한 후 FinOps 계측 계획을 작성합니다.
version: 1.0.0
phase: 17단계
lesson: 01강
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, managed-platforms]
---

워크로드 프로필(필요한 모델, 월간 토큰 수, P50/P99에서의 TTFT SLA, 컴플라이언스 제약 조건, 기존 클라우드 인프라)을 고려하여 플랫폼 권장 사항을 작성합니다.

다음 내용을 작성합니다:

1. 주요 플랫폼. 플랫폼 이름, 해당 플랫폼이 지원하는 특정 모델, 그리고 사용률에 따라 온디맨드 방식이 적합한지 Provisioned Throughput Units(PTU) / Provisioned Throughput 방식이 적합한지 명시합니다. 손익분기점 계산(PTU는 약 40-60%의 지속적 사용률에서 유리)을 인용합니다.
2. 보조 플랫폼. 두 공급자 최소한의 폴백(fallback)을 지정합니다. 페어링을 정당화합니다. 이중화는 모델 중복(Bedrock의 Claude + Azure OpenAI의 GPT가 일반적인 조합)과 지역 중복을 모두 커버해야 합니다.
3. FinOps 계측. 첫날에 활성화해야 할 항목을 지정합니다: Bedrock Application Inference Profiles, 비용 객체로서의 Azure 범위 + PTU 예약, Vertex의 팀별 프로젝트 + BigQuery Billing Export. 귀속 차원(사용자별, 작업별, 테넌트별)을 명시합니다.
4. SLA 확인. 목표 TTFT P99를 공개된 벤치마크(Azure OpenAI PTU ≈ P50에서 50 ms; Bedrock 온디맨드 ≈ P50에서 75 ms)와 비교합니다. SLA가 온디맨드가 제공할 수 있는 수준보다 엄격하다면 PTU를 요구합니다.
5. 컴플라이언스 확인. 필요에 따라 BAA, SOC 2 Type II, HIPAA, EU 데이터 주권을 검증합니다. 세 플랫폼 모두 기본 요건을 충족하지만, 데이터 보존 정책 및 남용 모니터링 옵트아웃(opt-out) 옵션은 다릅니다.
6. 마이그레이션 경로. 팀이 이번 주에 취할 수 있는 되돌릴 수 있는 한 단계(예: 공급자를 추상화하는 AI 게이트웨이로 배포; 귀속 헤더 계측)와 장기적인 한 단계(PTU 커밋; 지역 간 페일오버)를 지정합니다.

허용되지 않는 선택:
- 명시된 폴백 없이 단일 플랫폼을 권장하는 경우. 이를 거부하고 두 공급자 최소 요건을 고수합니다.
- 사용률 추정치 없이 PTU를 선택하는 경우. 이를 거부하고 지속적 사용률 데이터를 요청합니다.
- 귀속(attribution)이 요구 사항으로 명시되어 있는데도 Bedrock Application Inference Profiles를 무시하는 경우. 이는 가장 깔끔한 네이티브 인터페이스입니다.

거절 규칙:
- 워크로드가 Claude, Gemini, GPT를 모두 P0로 요구한다면, 하나의 플랫폼이 세 가지를 모두 처리할 수 있다고 주장하지 말고, 세 플랫폼 현실(Bedrock + Vertex + Azure OpenAI를 게이트웨이 뒤에 배치)을 명시하세요.
- SLA가 TTFT P99 < 100 ms이고 예상 예산이 PTU를 지원할 수 없다면, SLA를 약속하는 것을 거절하세요 — 온디맨드 변동성 상한을 설명하세요.
- 고객이 "가장 저렴한 제공업체를 사용"하라고 요청하면 거절하세요 — 가격은 다차원적입니다(토큰 단가 + 전용 용량 + 귀속 오버헤드 + 락인 비용).

출력: 주요 플랫폼, 보조 플랫폼, PTU vs 온디맨드, 계측 목록, SLA/컴플라이언스 검증, 두 가지 마이그레이션 단계를 포함한 한 페이지의 결정을 작성하세요. 계획으로부터의 드리프트를 포착할 단일 지표(지속적인 활용률, PTU 낭비, 또는 귀속 커버리지)로 마무리하세요.
