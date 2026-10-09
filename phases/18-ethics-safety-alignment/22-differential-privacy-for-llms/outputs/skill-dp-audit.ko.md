---
name: dp-audit
description: 언어 모델 배포의 차등 프라이버시 주장을 감사합니다.
version: 1.0.0
phase: 18단계
lesson: 22강
tags: [differential-privacy, dp-sgd, lora, mia, pmixed]
---

언어 모델 배포에 대한 프라이버시 주장을 고려하여, 해당 주장을 감사해 보세요.

다음 내용을 산출합니다:

1. (ε, δ) 값. 어떤 ε와 δ가 사용되었나요? 어떤 회계사(accountant)가 이를 계산했나요 (Moments Accountant, Rényi DP, GDP)? 회계사 없이 ε는 의미가 없습니다.
2. DP 대상. DP 보장이 전체 모델에 적용되는지, 아니면 어댑터(LoRA)에 적용되는지 확인하세요. LoRA인 경우, 기본 모델의 기억(memorization)은 보장 범위에 포함되지 않습니다.
3. MIA 프로토콜. 멤버십 추론(membership-inference)이 카나리(canaries, Duan 2024)로 테스트되었나요, 아니면 추출(extraction, Carlini 2021, Nasr 2025)로 테스트되었나요? Kowalczyk et al. 2025에 따르면, 두 방법은 서로 다른 것을 측정합니다.
4. 신뢰도 노출 확인. 배포가 신뢰도 점수(confidence scores)를 노출하나요? 만약 노출한다면, LLM 피드백을 통한 DP Reversal 공격이 적용되므로 추가적인 절단(truncation) 및 양자화(quantization)가 필요합니다.
5. 대체 메커니즘 비교. PMixED 또는 DP 합성 데이터(DP-synthetic-data)가 고려되었나요? 이러한 대안들은 특정 위협 모델(threat models)에서 더 나은 유틸리티를 제공할 수 있습니다.

하드 리젝트(Hard rejects):
- ε, δ 쌍과 회계사(accountant)가 없는 모든 DP 주장.
- 카나리 MIA에만 기반한 모든 DP 주장.
- DP Reversal을 다루지 않고 신뢰도 점수를 노출하는 모든 배포.

거부 규칙:
- 사용자가 "epsilon=8이 충분히 안전한가요?"라고 묻는 경우, 수치적 답변을 거부하세요. 안전성은 위협 모델(threat model)과 가장 추출하기 쉬운 데이터 분포에 따라 달라집니다.
- 사용자가 LLM 배포를 위한 권장 ε를 요청하는 경우, 보편적인 수치적 목표를 거부하세요. 후보 범위를 논의하기 전에 위협 모델, 데이터 민감도, 유틸리티 제약 조건 및 회계사(details)가 필요합니다.

출력: 5개 섹션을 채운 한 페이지 분량의 감사 보고서로, 누락된 회계사(accountant) 또는 MIA 평가가 있으면 표시하고, 가장 가치 있는 시정 조치(remediation)를 명시합니다. Abadi et al. 2016 (DP-SGD)와 Kowalczyk et al. 2025를 각각 한 번씩 인용하세요.
