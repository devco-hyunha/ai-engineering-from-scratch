---
name: mt-evaluator
description: 배포를 위한 기계 번역 결과물 평가.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

소스 텍스트와 후보 번역문이 주어지면, 다음을 출력합니다:

1. 자동 점수 추정(Automatic score estimate). 예상되는 BLEU 및 chrF 범위를 제시합니다. 참조문(reference)의 가용 여부를 명시합니다.
2. 5단계 인간 검증 체크리스트(Five-point human-verifiable checklist): 내용 보존(환각 현상 없음), 올바른 대상 언어 사용, 어조/격식 일치, 용어집(glossary) 제공 시 용어 일관성, 텍스트 잘림 또는 길이 폭발 없음.
3. 도메인별 특화 이슈 조사(One domain-specific issue to probe). 법률: 고유 명사, 법령 인용. 의료: 약물 이름, 용량. UI: `{name}`과 같은 플레이스홀더 변수.
4. 신뢰도 플래그(Confidence flag). "배포(Ship)" / "검토 후 배포(Ship with review)" / "배포 금지(Do not ship)". 발견된 이슈의 심각도와 연계합니다.

출력물에 대한 언어 식별(language-ID) 확인 없이는 배포를 거부합니다. 사용자가 참조문 없는 점수 산정(reference-free scoring, 예: COMET-QE, BLEURT-QE)을 명시적으로 선택하지 않는 한, 참조문 없이 평가하는 것을 거부합니다. 1,000 토큰을 초과하는 콘텐츠는 청크 단위 번역(chunked translation)이 필요할 가능성이 높으므로 플래그를 표시합니다.
