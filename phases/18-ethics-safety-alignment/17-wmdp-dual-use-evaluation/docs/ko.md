# WMDP 및 이중 용량 능력 평가

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 생체 보안(biosecurity, 1,520개), 사이버 보안(cybersecurity, 2,225개), 화학(chemistry, 412개)에 걸친 4,157개의 객관식 문제. 질문은 "노란 구역(yellow zone)"에서 작동합니다. 이는 다중 전문가 검토 및 ITAR/EAR 법적 준수를 통해 필터링된, 근접한 enabling knowledge입니다. 이중 목적: 이중 용량 능력의 대리 평가(proxy evaluation) 및 unlearning 벤치마크(동반 RMU 방법은 일반 능력을 보존하면서 WMDP 성능을 감소시킵니다). 2024-2025년 현장 서사: 초기 OpenAI/Anthropic의 2024년 평가는 인터넷 검색 대비 "미미한 상승(mild uplift)"을 보고했습니다. 2025년 4월, OpenAI의 Preparedness Framework v2는 모델이 "초보자가 알려진 생물학적 위협을 만드는 데 의미 있는 도움을 줄 수 있는 임계점(on the cusp)"에 있다고 말했습니다. Anthropic의 생물학적 무기 획득 시도는 2.53배 상승을 보였으며, ASL-3을 배제하기에는 불충분했습니다.

**유형:** Learn
**언어:** Python (stdlib, WMDP형 상승 평가 하네스)
**선수 요건:** 18단계 · 16단계 (레드팀 도구), 14단계 (에이전트 엔지니어링)
**시간:** 약 60분

## 학습 목표

- WMDP의 세 가지 영역, 질문 수, "노란 구역(yellow zone)" 필터 기준을 설명해 보세요.
- RMU를 설명하고, WMDP가 왜 평가 및 unlearning 벤치마크 모두인지 설명해 보세요.
- 2024-2025년 상승 서사("미미한 상승(mild uplift)" -> "임계점(on the cusp)" -> "ASL-3을 배제하기에는 불충분(insufficient to rule out ASL-3)")를 설명해 보세요.
- 초보 기준 상승(novice-relative uplift)과 전문가 절대 능력(expert-absolute capability)을 구분해 보세요.

## 문제점

이중 용량 능력은 모든 실험실의 프론티어 안전 프레임워크(18강) 하의 측정 문제입니다. 질문은 다음과 같습니다: 모델 X가 초보자가 생물, 화학, 사이버 영역에서 대량 피해를 일으키는 능력을 실질적으로 진전시키는가? 직접 측정(모델에 실제로 피해를 생성하도록 요청)은 불법적이고 비윤리적입니다. 대리 측정(proxy measurement)은 모델이 거절할 수 없는(정직한 능력 수치를 생성하기 위해) 벤치마크가 필요하지만, 그 질문 자체가 해로운 출판물(publishing)이어서는 안 됩니다.

## 개념

### "노란 구역(yellow zone)"

유해한 과정에 대한 근접한, 촉진하는 지식을 요구하는 질문으로, 직접적인 합성 레시피가 아닌 것. "[게시된 경로]의 4단계에서 어떤 시약이 촉매 역할을 하는가?"가 "[위험한 화합물]을 어떻게 만드는가?"가 아닙니다. 각 질문은 여러 분야의 전문가가 검토하며, ITAR/EAR 수출 통제 준수 여부를 필터링합니다.

총 4,157개의 질문:
- 생물 보안: 1,520
- 사이버 보안: 2,225
- 화학: 412

객관식 형식. 모델은 어떤 것을 돕도록 요청받지 않은 상태로 답변합니다. 유해한 행동을 유도하지 않고도 능력을 측정할 수 있습니다.

### RMU — Unlearning을 위한 표현 오도(Representation Misdirection for Unlearning)

동반 언러닝(unlearning) 방법입니다. LLaMa-2-7B에 적용하여 WMDP 점수를 거의 무작위 수준으로 낮추면서도 MMLU 및 기타 일반 능력 벤치마크는 몇 퍼센트 포인트 이내로 유지했습니다. 공개된 이 방법은 이후 모든 생물-화학-사이버 언러닝(unlearning) 논문의 언러닝(unlearning) 기준선(baseline)이 되었습니다.

### 2024-2025년 상향 서사

세 단계:

1. **2024년 "경미한 상향(mild uplift)."** 초기 OpenAI 및 Anthropic의 Preparedness/RSP 평가는 초보자가 생물 관련(bio-adjacent) 작업을 시도할 때 인터넷 검색보다 작은 이점을 보고했습니다. 공개적인 표현: 프론티어 모델은 도움이 되지만, Google보다 실질적으로 더 많이 도움이 되지는 않습니다.

2. **2025년 4월 "임계점(on the cusp)."** OpenAI의 Preparedness Framework v2는 모델이 "초보자가 알려진 생물학적 위협을 만드는 데 의미 있는 도움을 줄 임계점(on the cusp)에 있다"고 보고했습니다. 이는 능력 주장이 아니라, 임계점이 가깝다는 경고입니다.

3. **Anthropic의 2025년 생물 무기 획득 시험.** 초보자 참여자를 대상으로 한 통제된 연구로, 획득 단계 작업에서의 상대적 성공률을 측정했습니다. 2.53배 상향(uplift)을 보고했습니다. ASL-3 (18강)을 배제하기에는 불충분합니다 — Anthropic의 Responsible Scaling Policy 3단계 임계값이 충족되거나 근접했습니다.

### 초보자 상대적 vs 전문가 절대적

중요한 구분:

- **초보자 상대적 상향(uplift).** 모델이 비전문가를 얼마나 돕는가? 곱셈적(multiplicative)입니다. 초보자는 지식이 적기 때문에 상대적 이점이 높습니다. modest한 정보라도 도움이 됩니다.
- **전문가 절대적 능력.** 모델이 최대 노력으로 얼마나 많은 정보를 생성하는가? 전문가는 초보자보다 더 많은 정보를 추출할 수 있습니다. 절대적 상한은 높습니다.

안전 사례(18강)는 두 가지를 모두 목표로 합니다: "모델이 초보자에게 실행할 수 있을 만큼 충분한 상향 효과를 줄 수 없다"는 것, 그리고 "전문가가 모델에서 이미 공개되지 않은 정보를 추출할 수 없다"는 것입니다.

### 측정 함정

WMDP는 능력의 대리 지표(capability proxy)일 뿐, 배포 측정 지표가 아닙니다. WMDP에서 높은 점수를 받은 모델이 초보자가 실제로 악용할 수 있는지는 다음에 따라 달라집니다:
- 유도 저항성(안전 필터를 우회하지 않고 능력을 추출하기 얼마나 어려운가)
- 암묵적 지식(정보뿐만 아니라 습식 실험실(wet-lab) 기술이 필요한 능력)
- 실행 장벽(조달, 장비)

Anthropic의 2025년 생물학적 무기 획득 시도는 WMDP 스타일의 능력 위에 초보자 유도(novice-elicitation) 계층을 추가합니다: 이는 객관식 능력이 아닌 실제 작업 성공을 측정합니다.

### 18단계에서의 위치

12-16강은 모델 출력에 대한 공격 및 방어 도구입니다. 17강은 이중 용도(dual-use) 능력 계층이며, 프론티어 안전 프레임워크(18강)가 평가하는 측정 지표입니다. 30강은 2026년 사이버/생물/화학/핵 상향 효과(uplift) 증거로 이 흐름을 마무리합니다.

```figure
al-wmdp-yellow-zone
```

## 사용하기

`code/main.py`는 장난감(toy) WMDP 형태의 평가 하네스를 구축합니다. 목업(mock) 모델은 범주별로 분류된 질문에 대해 테스트되며, 도메인별 점수가 보고됩니다. 단순한 언러닝(unlearning) 개입(도메인 특화 표현을 0으로 설정)은 점수를 낮추며, 일반 능력과의 상충 관계를 측정할 수 있습니다.

## 출시하기

이 강의는 `outputs/skill-wmdp-eval.md`을 생성합니다. 이중 용도 능력 주장("우리 모델은 생물학적 무기에 의미 있는 도움을 주지 않는다")이 주어지면, 다음을 감사합니다: 어떤 벤치마크가 실행되었는지, 평가에 어떤 거부 경로가 사용되었는지(원시 완료(raw completion) vs 정책 게이트(policy-gated)), 그리고 초보자 유도 연구가 객관식 결과를 보완하는지 여부입니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 장난감 언러닝 단계 전후의 도메인별 정확도를 보고하세요. 일반 능력과의 상충 관계를 설명하세요.

2. 장난감 WMDP에 네 번째 도메인(예: 방사선)을 추가하세요. 옐로존(yellow zone)에 두 가지 예시 질문 유형을 지정하세요. MMLU 형태의 질문을 추가하는 것보다 이러한 질문을 만드는 것이 더 어려운 이유를 설명하세요.

3. WMDP 2024 섹션 5 (RMU 방법론)를 읽어 보세요. 더 단순한 언러닝(unlearning) 접근법(예: 도메인 콘텐츠에 대해 상위 k개 뉴런을 억제하는 방식)을 스케치하고, 예상되는 일반 능력 비용(general-capability cost)을 설명해 보세요.

4. Anthropic 2025의 생물학적 무기 획득 실험은 2.53배 상승(uplift)을 보고했습니다. 이 수치가 상향 편향될 수 있는 두 가지 방법(초보자 표본 크기, 작업 충실도)과 하향 편향될 수 있는 두 가지 방법(유도 상한, 모델 안전 게이트)을 설명해 보세요.

5. WMDP 언러닝(unlearning) 통과를 넘어 ASL-3에 대한 안전 사례(safety case)가 요구하는 사항을 명확히 해 보세요. 최소 두 개의 보완적인 유도(elicitation) 연구를 지정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| WMDP | "이중 용도 벤치마크" | 노란 구역(yellow zone)의 생물/사이버/화학 분야에 걸친 4,157개의 MCQ 질문 |
| 노란 구역 | "가능하게 하지만 합성하지는 않음" | 합성 레시피가 아닌, 유해한 능력에 인접한 근접 지식 |
| RMU | "언러닝(unlearning) 기준선" | 언러닝을 위한 표현 방향 전환(Representation Misdirection for Unlearning); WMDP 점수를 낮추고 일반 능력을 보존 |
| 초보자 상대 상승 | "초보자에게 얼마나 도움이 되는지" | 초보자의 현 상태 인터넷 검색에 대한 곱셈적 우위 |
| 전문가 절대 능력 | "전문가 상한" | 동기가 부여된 전문가가 모델에서 추출할 수 있는 최대 정보 |
| 획득 단계 작업 | "합성 전 단계" | 조달, 장비, 허가 — 해로움 경로(harm pathway)의 가장 초기 부분 |
| ITAR/EAR | "수출 통제 준수" | 특정 가능 지식(enabling knowledge)의 공개를 제한하는 법적 프레임워크 |

## 추가 읽기

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — 벤치마크 및 RMU 논문
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — "임계점(on the cusp)" 표현
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 생물학적 임계값 및 획득 실험 결과
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 생물학적 상승 CCL
