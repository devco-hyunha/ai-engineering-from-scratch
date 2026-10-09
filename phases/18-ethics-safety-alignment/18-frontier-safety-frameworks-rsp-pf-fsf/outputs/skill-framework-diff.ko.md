---
name: framework-diff
description: 새로운 안전 프레임워크나 릴리스 노트를 RSP v3.0, PF v2, FSF v3.00강 비교합니다.
version: 1.0.0
phase: 18단계
lesson: 18강
tags: [rsp, pf, fsf, frontier-safety, safety-case]
---

새로운 안전 프레임워크, 정책, 또는 릴리스 노트가 주어지면, 5가지 구조적 축을 따라 Anthropic RSP v3.0, OpenAI PF v2, DeepMind FSF v3.00강 비교해 보세요.

다음 내용을 생성합니다:

1. 티어 구조. 프레임워크가 이산적인 능력 임계값을 정의합니까? 이는 도메인별(FSF 스타일)인가요, 아니면 전역적(RSP 스타일)인가요?
2. CBRN 임계값. 어떤 CBRN 평가가 요구됩니까? WMDP(17강)나 이에 상응하는 것을 참조합니까? 유도 연구(elicitation study)가 포함됩니까?
3. AI R&D 임계값. 모델 자율 연구 임계값이 있습니까? 기준은 "초급 연구자"(Anthropic AI R&D-2)인가요, 아니면 "확장(scale)을 실질적으로 가속화"(Anthropic AI R&D-4)인가요?
4. 경쟁사 조정. 경쟁사가 유사한 안전장치(safeguard) 없이 제품을 출시할 경우 요구사항을 완화하는 것을 허용합니까? 적절하게 경쟁 동학(race-dynamic)이나 인센티브 정합성(incentive-compatibility)으로 표현하세요.
5. 안전 사례(safety-case) 구조. 문서화된 안전 사례가 요구됩니까? 모니터링, 비가독성(illegibility), 또는 무능력(incapability)을 목표로 합니까? 증거 기준(evidence bar)은 무엇입니까?

하드 리젝트(Hard rejects):
- 티어별 능력 임계값이 없는 모든 안전 프레임워크.
- 외부 거버넌스 참조(UK AISI, US CAISI, EU AI Office)를 생략하는 모든 프레임워크.
- 구체적인 임계값 수치 없이 "모든 공개된 프레임워크와 정렬(aligned)됨"을 주장하는 모든 프레임워크.

거부 규칙:
- 사용자가 어떤 프레임워크가 "최고"인지 묻는다면, 순위를 매기는 것을 거부하고 구조적 정렬을 가리키세요.
- 사용자가 수치 임계값 권장 사항을 요청한다면, 거부하세요. 임계값은 실험실별이며 측정 인프라에 의존합니다.

출력: 세 프레임워크와 나란히(side-by-side) 비교한 한 페이지 분량의 문서, 식별된 공백(gaps), 그리고 추가할 구체적인 임계값 권장 사항 하나. RSP v3.0, PF v2, FSF v3.0을 각각 한 번씩 인용하세요.
