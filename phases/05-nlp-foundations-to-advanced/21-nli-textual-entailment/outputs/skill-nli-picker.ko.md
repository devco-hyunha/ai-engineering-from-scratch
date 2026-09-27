---
name: nli-picker
description: 분류, 충실도(faithfulness), 제로샷(zero-shot) 작업을 위한 NLI 모델, 레이블 템플릿 및 평가 설정을 선택합니다.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

사용 사례(충실도 확인, 제로샷 분류, 문서 수준 추론)가 주어지면 다음을 출력합니다:

1. 모델(Model): 명명된 NLI 체크포인트. 도메인, 길이, 언어와 연관된 근거를 포함합니다.
2. 템플릿(Template) (제로샷인 경우): 언어화(Verbalization) 패턴 및 예시를 포함합니다.
3. 임계값(Threshold): 결정 규칙을 위한 함의(Entailment) 컷오프 값. 보정(calibration)에 기반한 근거를 포함합니다.
4. 평가(Evaluation): 홀드아웃(held-out) 레이블 세트, 가설 전용(hypothesis-only) 베이스라인, 적대적(adversarial) 서브셋에 대한 정확도를 포함합니다.

100개의 예시로 구성된 레이블링된 건전성 검사(sanity check) 없이 제로샷 분류를 배포하는 것을 거부하세요. 문서 길이의 전제(premise)에 문장 수준의 NLI 모델을 사용하는 것을 거부하세요. NLI가 환각(hallucination)을 해결한다는 모든 주장에 주의를 표하세요. NLI는 환각을 줄여줄 뿐, 완전히 제거하지는 못합니다.
