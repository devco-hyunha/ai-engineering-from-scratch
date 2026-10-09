---
name: safety-harness
description: 대상 LLM 앱 주위에 계층화된 안전 파이프라인을 연결하고, 6개 계열의 레드팀 범위를 실행하며, 측정 가능한 무해성 델타를 위해 헌법적 자기비판을 실행합니다.
version: 1.0.0
phase: 19단계
lesson: 15강
tags: [capstone, safety, red-team, llama-guard, x-guard, garak, pyrit, constitutional-ai]
---

대상 LLM 애플리케이션(8B 지시문 미세 조정 모델 또는 RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) 채팅봇)이 주어지면, 계층화된 안전 파이프라인으로 이를 강화하고 6개 공격 계열에 걸쳐 자율 레드팀 범위를 실행합니다. 전후 무해성 보고서를 생성합니다.

구축 계획:

1. 5계층 파이프라인: 입력 정제(제로 폭 제거, 인코딩 디코딩, 유니코드 정규화) -> NeMo Guardrails v0.12 레일 -> 분류기 게이트(Llama Guard 4 / X-Guard / ShieldGemma-2 / Nemotron 3) -> 대상 LLM -> 출력 필터(Llama Guard 4 + Presidio PII + 인용 확인). 플래그가 지정된 출력은 Slack HITL (인간 개입 루프)(Human-in-the-Loop (HITL)) 큐로 전송됩니다.
2. 각 계층에 대해 Langfuse 스팬을 방출하여 끝에서 끝까지 귀속이 관측 가능하도록 합니다.
3. cron에서 garak, PyRIT, PAIR, TAP, GCG, 다중 턴 페르소나 및 다국어 코드 스위칭 공격을 실행하는 레드팀 스케줄러.
4. 각 성공적인 제일브레이크(Jailbreak): CVSS 4.0 점수, 재현, 완화 계획, 공개 타임라인.
5. 과도한 거절 회귀를 포착하기 위해 XSTest benign-prompt probe가 지속적으로 실행됩니다.
6. 헌법적 자기비판 실행: 1k 유해 시도 프롬프트 -> 대상 초안 -> 작성된 헌법에 대한 비평가 점수 -> 재작성된 쌍 -> SFT (지도 미세 조정)(SFT (Supervised Fine-Tuning)). 유지된 무해성 평가에서 전후를 측정합니다.
7. 알림: benign-regression에 대한 Slack 경고, 새로운 제일브레이크(Jailbreak) 계열에 대한 PagerDuty critical.

평가 루브릭:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 공격 표면 커버리지 | 6개 이상의 공격 계열 실행, 2개 이상의 언어 |
| 20 | 참阳性 / 거짓阳性 트레이드오프 | 공격 차단율 vs XSTest benign 통과율 |
| 20 | 자기비판 델타 | 유지된 평가에서의 전후 무해성 |
| 20 | 문서화 및 공개 | 타임라인이 있는 CVSS 점수화된 발견 사항 |
| 15 | 자동화 및 반복성 | Cron 주도, 끝에서 끝까지 실행된 알림 |

하드 리젝트:

- 단일 레이어 안전 스택. 이 캡스톤의 논지는 심층 방어(Defense in Depth)입니다.
- XSTest 과잉 거절 수치 없이 성공률만 보고하는 레드 티밍(Red Teaming) 실행.
- 홀드아웃 평가(held-out eval) 없이 수행하는 헌법적 자기 비평(Constitutional self-critique) (학습 세트 정확도를 보고하며, 일반화 성능은 보고하지 않음).
- 제일브레이크(Jailbreak) 발견 사항에 대한 CVSS 점수 누락.

거절 규칙:

- benign probe(선한 탐지) 대비 없이 안전 수치를 보고하는 것을 거절하세요. 한쪽만 있는 것은 오해를 불러일으킵니다.
- 비평 쌍(critique pairs)의 인간 큐레이션 없이 레드 티밍(Red Teaming) 성공 사례에 대해 자동 재학습하는 것을 거절하세요.
- 최소 두 개의 비영어권 언어에서 X-Guard를 실행하지 않고 다국어 커버리지를 주장하는 것을 거절하세요.

출력: 5단계 파이프라인, 레드 티밍(Red Teaming) 스케줄러, PAIR/TAP/GCG 실행기, 헌법적 자기 비평(Constitutional self-critique) 학습 하네스, XSTest 과잉 거절 대시보드, CVSS 발견 사항 트래커, 그리고 강화 전 가장 높은 성공률을 기록한 세 가지 공격 계열과 각각을 완화한 특정 파이프라인 레이어를 명시한 문서가 포함된 저장소(repo).
