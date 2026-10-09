---
name: cross-policy-diff
description: OpenAI Preparedness Framework v2, Anthropic RSP v3.0, DeepMind FSF v3를 참조하여 특정 기능에 대한 정책 간 비교를 생성합니다.
version: 1.0.0
phase: 15단계
lesson: 20강
tags: [preparedness-framework, fsf, rsp, cross-policy, scaling-policy]
---

특정 프론티어 기능(예: "장거리 자율성", "자율 복제 및 적응", "R&D 자동화")이 주어지면, 세 프레임워크가 해당 기능을 어떻게 분류하고 어떤 완화 조치가 발동되는지 보여주는 정책 간 비교를 생성해 보세요.

다음 내용을 생성합니다:

1. **OpenAI PF v2 분류.** Tracked 또는 Research. Tracked인 경우 Capabilities + Safeguards Report의 트리거를 명시합니다. Research인 경우 정책 용어가 "잠재적" 완화 조치임을 명시합니다.
2. **Anthropic RSP v3.0 분류.** 어떤 임계값(ASL-3, AI R&D-4, 하드코딩된 금지)에 해당합니까? 어떤 완화 조치(적극적 사례, 보안 + 배포)가 적용됩니까? 해당 약속이 Anthropic 단독(tier)에 속하는지, 산업 권고(tier)에 속하는지 확인합니다.
3. **DeepMind FSF v3 분류.** 어떤 도메인(Cyber, Bio, ML R&D, CBRN)에 해당합니까? 어떤 CCL 또는 Tracked Capability Level에 해당합니까? 기만적 정렬(deceptive alignment) 모니터링이 발동됩니까?
4. **수렴 요약.** 세 정책이 해당 기능의 심각성에 대해 동의합니까, 아니면 의미 있는 불일치가 있습니까? 어떤 분류가 가장 엄격하고, 어떤 분류가 가장 느슨합니까?
5. **측정 의존성.** 모든 분류는 기능 측정에 의존합니다. 기능이 어떻게 측정되는지, 그리고 어떤 평가 제공자(METR, Apollo, 내부, 제3자)가 해당 측정을 담당하는지 명시합니다.

허용되지 않는 내용:
- 문서 수준의 증거 없이 발표 용어의 유사성만을 근거로 한 정책 간 정렬 주장.
- 원본 문서의 특정 조항을 지목할 수 없는 분류.
- OpenAI의 "Research Category"를 "Tracked Category"와 동일하게 취급하는 것 — 두 범주는 운영적 결과가 다릅니다.

거부 규칙:
- 사용자가 각 분류에 대한 원본 문서의 발췌문을 제시할 수 없는 경우, 요청을 거부하고 먼저 인용을 요구합니다.
- 사용자가 정책의 존재 자체를 실제 완화 조치의 증거로 간주하는 경우, 이를 거부하고 특정 완화 조치가 작동했다는 증거를 요구합니다.
- 해당 기능이 프레임워크에 의해 "커버"된다고 주장되지만 문서에 해당 단어가 나타나지 않는 경우, 이를 거부하고 구체적인 조항 참조를 요구합니다.

출력 형식:

다음 내용을 포함한 diff 문서를 반환합니다:
- **기능 정의** (한 문장)
- **OpenAI PF v2 행** (분류, 트리거, 출처 조항)
- **Anthropic RSP v3.0 행** (분류, 트리거, 일방적 조치 대 권고)
- **DeepMind FSF v3 행** (영역, CCL / TCL, 기만적 정렬 관련성)
- **수렴 요약** (합의 사항 + 의미 있는 불일치)
- **측정 소유권** (평가 제공자, 평가 주기)
- **독자 권고** (가장 엄격한 기준, 가장 느슨한 기준, 근거)
