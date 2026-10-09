# 정렬 연구 생태계 — MATS, Redwood, Apollo, METR

> 5개 조직이 2026년 비(非) 연구소 정렬 연구 계층을 정의합니다. MATS (ML Alignment & Theory Scholars): 2021년 말 이후 527명 이상의 연구자, 180편 이상의 논문, 10,000회 이상의 인용, h-index 47; 2024년 여름 코호트는 약 90명의 학자와 40명의 멘토를 포함하여 501(c)(3) 비영리 단체로 설립되었습니다. 2025년 이전 졸업생의 80%는 안전/보안 분야에서 활동하며, 200명 이상이 Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo에서 근무하고 있습니다. Redwood Research: Buck Shlegeris가 설립한 응용 정렬 연구소; AI Control (10강)을 도입했으며, UK AISI와 통제 안전 사례(control safety cases) 협력 중입니다. Apollo Research: 프론티어 연구소를 위한 배포 전 스키밍(scheming) 평가; In-Context Scheming (8강) 및 Towards Safety Cases for AI Scheming 저술. METR (Model Evaluation and Threat Research): 작업 기반 능력 평가, 자율 작업 시간 범위 연구; "Common Elements of Frontier AI Safety Policies"는 연구소 프레임워크를 비교합니다. Eleos AI Research: 모델 복지의 배포 전 평가 (19강); Claude Opus 4 복지 평가를 수행했습니다.

**유형:** Learn
**언어:** 없음
**선수 요건:** 18단계 · 01-27강 (이전 18단계 강의)
**시간:** 약 45분

## 학습 목표

- 비(非) 연구소 정렬 연구 생태계의 5개 조직과 그들의 핵심 산출물을 식별해 보세요.
- MATS의 규모(학자 수, 논문 수, h-index)와 인재 파이프라인으로서의 역할을 설명해 보세요.
- Redwood의 AI Control 의제와 UK AISI와의 파트너십을 설명해 보세요.
- METR의 작업 기반 평가 방법론을 설명해 보세요.

## 문제점

프론티어 연구소 (18강)는 안전 평가를 내부적으로 수행하고 선택된 결과를 공개합니다. 연구소 외부의 생태계는 평가가 검증되는 곳, 새로운 실패 모드가 처음 발견되는 곳, 그리고 인재가 양성되는 곳입니다. 생태계를 이해하면 어떤 연구 결과가 누구에 의해 신뢰되는지 해석하는 데 도움이 됩니다.

## 개념

### MATS (ML Alignment & Theory Scholars)

2021년 말에 시작되었습니다. 연구 멘토십 프로그램으로, 학자는 특정 정렬 문제에 대해 시니어 연구원과 10-12주간 함께합니다.

규모 (2026):
- 설립 이래 527명 이상의 연구자.
- 180편 이상의 논문이 발표되었습니다.
- 10,000회 이상의 인용.
- h-index 47.
- 2024년 여름: 90명의 학자 + 40명의 멘토; 501(c)(3) 비영리 단체로 설립되었습니다.

진로 결과: 2025년 이전 졸업생의 약 80%가 안전/보안 분야에서 활동하고 있습니다. Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo에 200명 이상이 재직 중입니다.

### Redwood Research

응용 정렬 연구소입니다. Buck Shlegeris가 설립했습니다. AI Control 의제를 도입했습니다 (10강). UK AISI와 통제 안전 사례(control safety cases)를 협력하고 있습니다. DeepMind와 Anthropic에 평가 설계에 대해 조언하고 있습니다.

정전 논문: Greenblatt, Shlegeris 등, "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright 등, arXiv:2412.14093, Anthropic와 공동).

스타일: 구체적인 위협 모델, 최악의 적대자, 스트레스 테스트가 가능한 구체적인 프로토콜.

### Apollo Research

프론티어 연구소를 위한 배포 전 사기(scheming) 평가입니다. In-Context Scheming을 집필했습니다 (8강, arXiv:2412.04984). 2025년 OpenAI의 사기 방지 훈련 협력의 파트너입니다. Towards Safety Cases for AI Scheming (2024)을 제작합니다.

스타일: 기만(deception)이 발생할 수 있는 에이전트 환경 평가; 세 가지 기둥 분해(misalignment, goal-directedness, situational awareness).

### METR (Model Evaluation and Threat Research)

작업 기반 능력 평가입니다. 자율 작업 완료 시간 범위(time-horizon) 연구입니다. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025)는 연구소 프레임워크를 비교합니다.

Apollo와 함께 AI Scheming 안전 사례 스케치(safety-case sketch)의 공동 저자입니다.

스타일: 장기 시간 범위(long-horizon) 작업 평가, 경험적 능력 측정, 프레임워크 종합.

### Eleos AI Research

모델 복지의 배포 전 평가입니다. 시스템 카드의 5.3절에 문서화된 Claude Opus 4 복지 평가를 수행했습니다. 19강의 복지 관련 주장에 대한 외부 방법론 검증을 제공합니다.

### 흐름

MATS는 연구원을 양성합니다. 졸업생은 Anthropic, DeepMind, OpenAI (실험실 안전 팀) 또는 Redwood, Apollo, METR, Eleos (외부 평가 기관)로 진출합니다. 외부 평가 기관은 실험실 및 UK AISI / CAISI와 협력합니다. 출판물은 생태계를 MATS의 다음 기수로 되돌려줍니다.

### 이 계층이 중요한 이유

단일 출처 평가는 신뢰할 수 없습니다. 자체 모델을 평가하는 실험실은 구조적인 이해 충돌이 있습니다. 외부 평가 기관은 실험실이 과소 보고할 수 있는 실패 모드들을 제기하고 검증할 수 있습니다. 2024년 Sleeper Agents 논문 (7강)은 Anthropic + Redwood; Alignment Faking은 Anthropic + Redwood; In-Context Scheming은 Apollo; Anti-Scheming은 Apollo + OpenAI였습니다. 다중 기관 구조가 품질 관리입니다.

### 18단계에서의 위치

7-11강은 Redwood와 Apollo의 작업을 참조합니다. 18강은 METR의 프레임워크 비교를 참조합니다. 19강은 Eleos를 참조합니다. 28강은 나머지 단계가 의존하는 생태계의 명시적인 조직 지도입니다.

```figure
sae-features
```

## 사용하기

코드가 없습니다. METR의 "Common Elements of Frontier AI Safety Policies"를 읽어, 외부 종합이 실험실 내부 정책 작업에 어떻게 가치를 더하는지 예시로 확인해 보세요.

## 출시하기

이 강의는 `outputs/skill-ecosystem-map.md`를 생성합니다. 정렬 주장이나 평가가 주어지면, 기관, 출판 장소 및 방법론적 스타일을 식별하고 알려진 대응 기관들과 교차 검증합니다.

## 연습 문제

1. 7-15강에서 한 논문을 선택하고 관련 기관을 식별하세요. 저자를 MATS 동문 및 현재 생태계 소속과 교차 검증하세요.

2. METR의 "Common Elements of Frontier AI Safety Policies"를 읽어보세요. 그들이 강조하는 세 가지 실험실 간 수렴점과 두 가지 가장 큰 차이점을 식별하세요.

3. MATS의 경력 결과는 약 80%가 안전/보안 분야입니다. 이 선택 압력이 적응적인지 (분야를 훈련시키는지) 편향적인지 (비정통적 입장을 필터링하는지) 논증하세요.

4. Redwood와 Apollo는 모두 통제/음모(scheming) 작업을 수행하지만 스타일이 다릅니다. 실패 모드를 하나 선택하고, 각 기관이 이를 어떻게 조사할지 설명하세요.

5. Eleos AI는 순수한 모델 복지 조직입니다. 다른 복지 관련 질문(인지적 자유, 로봇적 실체화 등)에 초점을 맞춘 가상의 두 번째 조직을 설계하고, 그 방법론을 명확히 서술해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| MATS | "멘토십 프로그램" | ML 정렬 및 이론 학자(MATS); 2021년 이후 527명 이상의 연구자 |
| Redwood Research | "제어 연구소" | 적용 정렬; AI Control 저자; UK AISI 파트너 |
| Apollo Research | "음모 평가" | 프론티어 랩을 위한 배포 전 음모 평가 |
| METR | "작업 범위 평가" | 작업 기반 능력 평가; 프레임워크 종합 |
| Eleos AI | "복지 연구소" | 모델 복지 배포 전 평가 |
| 인재 파이프라인 | "MATS -> 랩" | MATS 졸업생은 Anthropic, DM, OpenAI, Redwood, Apollo, METR로 유입 |
| 외부 평가 | "비랩 검증" | 모델 제작자가 수행하지 않는 평가; 신뢰성 향상 |

## 추가 읽기

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 멘토십 프로그램
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 논문
- [Apollo Research](https://www.apolloresearch.ai/) — 음모 평가
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 프레임워크 비교
- [Eleos AI Research](https://www.eleosai.org/research) — 모델 복지 방법론
