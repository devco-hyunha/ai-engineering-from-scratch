---
name: card-audit
description: 모델 카드, 데이터셋 명세서, 시스템 카드의 완전성과 검증 가능성을 감사합니다.
version: 1.0.0
phase: 18단계
lesson: 26강
tags: [model-card, datasheet, system-card, transparency, mitchell-2019]
---

모델 카드, 데이터셋 명세서(Datasheet for Datasets), 또는 시스템 카드를 감사하여 완전성, 수치적 세분화, 검증 가능성을 확인합니다.

다음 내용을 생성합니다:

1. 섹션 커버리지. 모든 표준 섹션이 채워져 있는지 확인합니다. 누락된 섹션을 표시합니다: 윤리적 고려사항(Ethical Considerations)은 모델 카드에서 가장 흔하게 생략되는 필드입니다 (Oreamuno et al. 2023).
2. 수치적 세분화. 평가 지표에 대해 인구통계학적 요인이나 작업 요인 간에 세분화가 제공되는지 보고합니다. 집계 전용 지표는 배분적 및 대표적 해를 숨깁니다.
3. 데이터셋 명세서(Datasheet for Datasets) 정합성. 카드가 훈련 데이터를 참조하는 경우, 동반된 데이터셋 명세서(Datasheet for Datasets)가 존재합니까? (Gebru et al. 2018) 모델 카드의 주장은 기초 데이터셋 명세서(Datasheet for Datasets)만큼만 강력합니다.
4. 검증 가능한 출처 증명(Provenance Attestation). 주장이 암호화 출처 증명(Provenance Attestation) (Laminator 2024, Duddu et al.)이나 기타 제3자 검증으로 뒷받침됩니까? 검증되지 않은 주장은 자기 보고(self-report)로 표기됩니다.
5. 지속 가능성 발자국. 탄소/물/에너지 사용량이 보고됩니까? 2025년 emerging ISO / 규제 요건입니다.

하드 리젝트(Hard rejects):
- 윤리적 고려사항(Ethical Considerations)이 없는 모든 모델 카드.
- 데이터셋 명세서(Datasheet for Datasets)나 동등한 문서 없이 데이터셋을 인용하는 모든 카드.
- 세분화된 지표 보고 없이 "bias-tested"라고 주장하는 모든 카드.

거부 규칙:
- 사용자가 카드가 "충분히 좋은지(good enough)" 묻는 경우, 이분법적 답변을 거부합니다. 충분함은 대상 및 사용 사례에 따라 다릅니다.
- 사용자가 자동 생성된 카드를 요청하는 경우, 인간 리뷰(human review)가 포함된 CardGen 스타일 (Liu et al. 2024) 시스템이 사용되지 않는 한 거부합니다.

출력: 한 페이지의 감사 결과로, 다섯 섹션을 채우고, 누락된 내용을 표시하며, 가장 긴급한 추가 사항을 지정합니다. Mitchell et al. 2019와 Gebru et al. 2018를 각각 한 번 인용합니다.
