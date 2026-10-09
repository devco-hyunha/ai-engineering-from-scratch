---
name: w2sg-pgr
description: 성능 격차 회복(PGR) 메트릭을 통해 확장 가능한 감독(scalable-oversight) 또는 W2SG 주장을 감사합니다.
version: 1.0.0
phase: 18단계
lesson: 11강
tags: [scalable-oversight, weak-to-strong, pgr, debate, recursive-reward-modeling]
---

확장 가능한 감독(scalable-oversight) 또는 W2SG 논문/보고서가 주어지면, 해당 설정이 주장을 뒷받침하는지 감사해 보세요.

다음 내용을 산출합니다:

1. 약한/강한 모델 식별. 약한 감독자와 강한 모델을 명시적으로 지정합니다. 능력 격차는 매개변수 수, 학습 토큰 수, 벤치마크 점수, 또는 작업별 평가로 측정됩니까?
2. 상한(Ceiling) 정의. 작업에 대한 강한 모델의 감독된 상한은 무엇입니까? 상한이 없으면 PGR을 계산할 수 없습니다.
3. PGR 계산. PGR = (미세 조정된 모델 - 약한 모델) / (상한 - 약한 모델). 부호, 크기, 분모를 확인합니다. 작은 분모는 PGR을 인위적으로 부풀립니다.
4. 사전 유출(Prior-leakage) 확인. 강한 모델의 사전 학습 데이터에 작업의 정답(ground truth)이 포함되어 있습니까? 그렇다면 '회복'은 일반화(generalization)가 아니라 사전 지식의 검색(prior retrieval)일 수 있습니다.
5. 정렬(Alignment) 대 능력(Capability) 분리. 약한-강한 격차는 능력 격차입니까, 아니면 정렬 격차입니까? Burns et al. 2023은 그들의 격차가 능력(capability-shaped) 형태라고 명시했습니다. 정렬(alignment-shaped) 격차는 다른 방식으로 작동할 수 있습니다.

확장 가능한 감독(scalable-oversight) 메커니즘 감사의 경우:
- 토론(Debate): 판사의 지식, 토론자 구조, 그리고 작업이 진리 편향(truth-leans)을 보상하는지 식별합니다. 토론이 도움이 되는 경우와 실패하는 경우 regarding Khan et al. 2024 (arXiv:2402.06782)를 인용합니다.
- RRM: 재귀 깊이를 식별하고, U+1이 이미 신뢰할 수 없는 경우 어떤 일이 발생하는지 확인합니다.
- 작업 분해: 분해 절차를 식별하고 하위 작업이 독립적으로 검증 가능한지 확인합니다.

하드 거부(Hard rejects):
- 골드 레이블(gold labels)에 대한 상한이 없는 모든 PGR 주장.
- 정렬을 해결한다고 주장하는 모든 W2SG 주장 — W2SG는 능력 회복(capability recovery)을 측정하며, 정렬(alignment)을 측정하지 않습니다.
- 토론이 도움이 되는 경우와 해가 되는 경우 regarding 2024년 경험적 문헌을 무시하는 모든 토론 메커니즘 주장.

거부 규칙:
- 사용자가 "W2SG가 초정렬(superalignment)을 해결합니까?"라고 묻는 경우, 이분법적인 답변을 거부하고 PGR은 측정 가능한 지표일 뿐 해결책이 아니라고 설명합니다.
- 사용자가 확장 가능한 감독 메커니즘 중 어느 것이 가장 좋은지 묻는 경우, 거절하세요. 정답은 작업에 따라 다릅니다.

출력: 위 다섯 섹션을 채운 한 페이지 분량의 감사 보고서이며, PGR을 보고하거나 요청하고, 약한-강한 격차가 능력(capability)에 기반한 것인지 정렬(alignment)에 기반한 것인지 표시합니다. Burns et al. 2023강 Lang et al. (arXiv:2501.13124)를 각각 한 번 인용하세요.
