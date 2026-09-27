---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

기본 언어 모델(base LM), 목표 행동(정렬, 추론, 거절, 에이전트 등), 그리고 선호도(preference) 또는 검증기(verifier) 예산이 주어졌을 때, 다음 내용을 설계하여 출력하세요:

1. **단계(Stage)**: SFT, RM, DPO, GRPO 중 무엇을 사용할지 근거와 함께 제시하세요.
2. **선호도 또는 검증기 출처(Preference or verifier source)**: 인간, AI 피드백, 규칙 기반(rule-based), 유닛 테스트 통과 여부(unit-test-pass), 또는 보상 증류(reward distillation) 중 선택하세요.
3. **KL 전략(KL strategy)**: 고정 $\beta$, 적응형 $\beta$, 또는 DPO(암시적 KL)를 정의하세요.
4. **진단(Diagnostics)**: 평균 KL, 보상 안정성, 과최적화 방지책(holdout human eval)을 포함하세요.
5. **안전 게이트(Safety gate)**: 레드팀 세트, 거절률(refusal rate), 유용성 RM과 분리된 안전 RM을 포함하세요.

**설계 제약 사항:**
- KL 모니터링이 없는 RLHF-PPO 설계는 거부하세요.
- 타겟 정책(target policy)보다 크기가 작은 RM을 사용하는 설계를 거부하세요.
- 길이만을 기준으로 하는 보상(length-only rewards)을 사용하는 설계를 거부하세요.
- 블라인드 인간 평가 세트(blind human-eval set)를 별도로 확보하지 않은 파이프라인은 과최적화 보호(over-optimization protection)가 부족한 것으로 표시하세요.
