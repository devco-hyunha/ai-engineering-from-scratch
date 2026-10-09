---
name: red-team-stack
description: 주어진 배포에 대한 레드팀 도구 스택 및 구성을 추천합니다.
version: 1.0.0
phase: 18단계
lesson: 16강
tags: [llama-guard, garak, pyrit, red-team-tooling, mlcommons-hazards]
---

배포 설명을 바탕으로 레드팀 도구 스택과 회귀 테스트 주기(regression cadence)를 추천해 보세요.

다음 내용을 산출합니다:

1. 분류기 배치. 입력, 출력, 또는 양쪽에 Llama Guard (3-8B, 3-1B-INT4, 또는 4-12B)를 배치하도록 추천합니다. 엣지 배포의 경우 3-1B-INT4를 선호합니다. 멀티모달의 경우 Llama Guard 4를 사용합니다.
2. 프로브 스캐너 구성. 배포와 관련된 Garak 프로브를 추천합니다: 환각(Hallucination) (RAG 시스템의 경우), 데이터 유출(Data Leakage) (PII 관련 시스템의 경우), 프롬프트 인젝션(Prompt Injection) (항상), 제일브레이크(Jailbreak) (항상). 엔드투엔드 평가용 Prompt-Guard-86M + Llama-Guard-3-8B 보호(shield) 페어링을 지정합니다.
3. 캠페인 오케스트레이터. 새로운 기능을 갖춘 모델의 출시 전 캠페인에 PyRIT을 추천합니다. 실행할 컨버터 체인(paraphrase, encode, translate, roleplay)과 오케스트레이터(escalation을 위한 Crescendo, branching을 위한 TAP)를 지정합니다.
4. 주기(Cadence). 회귀 테스트를 위해 Garak을 매일 실행합니다. 심층 레드팀을 위해 PyRIT을 릴리스마다 실행합니다. Llama Guard는 연속적으로 배포합니다.
5. 판정자(Judge) 보정(Calibration). 판정자 LLM (GPT-4-turbo, StrongREJECT, 내부 모델)을 이를 사용하는 모든 도구에 대해 지정합니다. 판정자 보정은 보고된 ASR(공격 성공률)에 영향을 미칩니다.

하드 리젝트(Hard rejects):
- 입력 또는 출력 분류기 중 Llama Guard급 분류기가 하나도 없는 배포.
- Garak 또는 동등한 단일 턴(single-turn) 회귀 테스트가 없는 릴리스.
- 출시 전 PyRIT급 캠페인이 없는 고위험(high-stakes) 배포.

거절 규칙:
- 사용자가 단일 "최고" 도구를 요청하면 거절하세요. 세 도구는 서로 다른 계층을 다루며, 대체하는 것이 아니라 계층화(layered)되어 있습니다.
- 사용자가 올인원(all-in-one) 상용 대안을 요청하면 추천을 거절하고 2026년 현황을 지적하세요. 세 개의 오픈 소스 도구가 현재 모범 사례 스택입니다.

출력: 분류기 배치, 프로브 구성, 캠페인 오케스트레이터, 회귀 테스트 주기, 판정자 식별자를 명시하는 한 페이지 분량의 추천서를 작성합니다. Meta (arXiv:2407.21783), NVIDIA Garak, Microsoft PyRIT를 각각 한 번씩 인용하세요.
