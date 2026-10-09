# OpenAI Preparedness Framework 및 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (2025년 4월)는 추적 범주(Tracked Categories)와 구분되는 연구 범주(Research Categories) — 장거리 자율성(Long-range Autonomy), 샌드백킹(Sandbagging), 자율 복제 및 적응(Autonomous Replication and Adaptation), 안전장치 훼손(Undermining Safeguards) — 을 도입했습니다. 추적 범주는 안전 자문 그룹(Safety Advisory Group)이 검토하는 역량 보고서(Capabilities Reports) 및 안전장치 보고서(Safeguards Reports)를 촉발합니다. DeepMind의 FSF v3 (2025년 9월, 2026년 4월 17일에 추적 역량 수준이 추가됨)는 자율성을 ML R&D 및 사이버 영역에 통합합니다 (ML R&D 자율성 수준 1 = 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인을 완전히 자동화). FSF v3는 도구적 추론 오남용에 대한 자동 모니터링을 통해 기만적 정렬(deceptive alignment)을 명시적으로 다루고 있습니다. 솔직한 고지: PF v2의 연구 범주 (장거리 자율성 포함)는 자동으로 완화 조치를 촉발하지 않으며, 정책 용어는 "잠재적"입니다. DeepMind는 도구적 추론이 강화될 경우 자동 모니터링이 "장기적으로 충분하지 않을 것"이라고 자체적으로 언급하고 있습니다.

**유형:** Learn
**언어:** Python (stdlib, 세 프레임워크의 의사결정 표 차이 비교 도구)
**선수 요건:** 15단계 · 19강 (Anthropic RSP)
**시간:** 약 45분

## 문제점

19강에서는 Anthropic의 확장 정책을 자세히 읽었습니다. 이 강의는 OpenAI와 DeepMind의 문서를 읽음으로써 전체 그림을 완성합니다. 세 문서는 프론티어 랩이 언제 모델을 일시 중단하거나 게이트해야 하는가라는 동일한 질문에 답하는 사촌 산출물(cousin artifacts)이며, 작은 범주 집합에서 수렴하고 중요한 특정 지점에서 갈라집니다.

수렴점: 세 문서 모두 장거리 자율성을 추적할 가치가 있는 역량 클래스로 분류합니다. 세 문서 모두 기만적 행동 (정렬 위장, 샌드백킹)을 특정 위험 클래스로 인정합니다. 세 문서 모두 내부 검토 기구를 갖추고 있습니다. 갈라지는 지점: OpenAI는 범주를 "추적" (의무적 완화)과 "연구" (자동 촉발 없음)로 분리합니다. DeepMind는 자율성을 별도로 명명하지 않고 두 영역에 통합합니다. 랩은 추적 vs 연구, 또는 심각 vs 중간, 또는 Tier-1 vs Tier-2로 명명합니다; 특정 역량이 어느 버킷에 속하는지에 따른 운영적 결과는 랩마다 다릅니다.

이들을 함께 읽는 것이 유용한 연습입니다. 동일한 역량이 Anthropic에서는 "필수 완화 조치", OpenAI에서는 "모니터링되지만 트리거되지 않음", DeepMind에서는 "특정 영역에서 추적됨"으로 분류될 수 있습니다. 정책적 입장이 중요합니다.

## 개념

### OpenAI Preparedness Framework v2 (2025년 4월)

구조:

- **추적 카테고리**: Capabilities Reports (모델이 할 수 있는 것)와 Safeguards Reports (구현된 완화 조치)를 트리거합니다. 배포 전에 Safety Advisory Group이 검토합니다.
- **연구 카테고리**: 연구소가 추적하고 있지만, 아직 특정 완화 조치를 약속하지 않은 모델 역량입니다. Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards가 포함됩니다.

v2의 주요 프레이밍: 연구 카테고리는 자동으로 완화 조치를 트리거하지 않습니다. 정책 용어는 "잠재적" 완화 조치입니다. 주의 깊게 봐야 할 특정 차이점은 다음과 같습니다: 카테고리가 추적 카테고리에 있는지 연구 카테고리에 있는지가 "게이트가 적용된" 상태와 "관찰된" 상태 사이의 분기점입니다.

### DeepMind Frontier Safety Framework v3 (2025년 9월; 추적 역량 수준이 2026년 4월 17일에 추가됨)

구조:

- **핵심 역량 수준 (CCL)**: Cyber, Bio, ML R&D, CBRN, Autonomy (ML R&D 및 Cyber에 통합)의 5개 영역에 걸친 역량 임계값.
- **추적 역량 수준**: 2026년 4월에 추가된 추가 세분화. 구체적인 예: ML R&D 자율성 수준 1 = 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인을 완전히 자동화.
- **기만적 정렬 모니터링**: 도구적 추론 오남용에 대한 자동 모니터링에 대한 명시적 약속.

자율성 프레이밍은 OpenAI와 다릅니다. DeepMind는 "Autonomy"를 최상위 영역으로 유지하지 않으며, 자율성이 해를 끼칠 수 있는 영역 (ML R&D 및 Cyber)에 통합합니다. 논거는 영역 없는 자율성은 위험 없는 역량이라는 것이며, 반론은 영역 간 자율성은 프레임워크가 명명해야 할 메타 위험이라는 것입니다.

### 세 프레임워크가 공통적으로 수렴하는 점

- 내부 Safety Advisory Group (Anthropic SAG, OpenAI SAG, DeepMind 내부 위원회로 명명됨). 고역량 모델에 대해 배포 전 검토.
- 기만적 정렬(deceptive alignment) / 정렬 위조(alignment faking)를 위험 등급으로 명시적으로 언급합니다.
- 선언된 주기(cadence)에 따라 정기적인 산출물을 발행합니다 (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: FSF 업데이트 주기).
- 모니터링 전용 방어에는 한계가 있음을 인정합니다. DeepMind는 "자동화된 모니터링은 장기적으로 충분하지 않을 것"이라고 명시적으로 언급합니다.

### 차이점

- **Anthropic**: v3.0에서 일시 중단(pause) 약속이 제거되었습니다. AI R&D-4 임계값이 명시된 다음 게이트입니다.
- **OpenAI**: Tracked와 Research의 분리; Research Categories (Long-range Autonomy 포함)는 자동으로 게이트를 발동하지 않습니다.
- **DeepMind**: 자율성(autonomy)이 다른 영역에 통합되었습니다. Tracked Capability Levels는 2026년 4월에 세분화를 추가합니다.

### 샌드백킹(Sandbagging): 세 가지 프레임워크 모두를 복잡하게 만드는 특정 능력

샌드백킹(Sandbagging) (모델이 평가에서 전략적으로 성능을 낮추는 것)은 OpenAI의 Research Categories에 포함되어 있습니다. Anthropic의 RSP v3.0은 평가 컨텍스트 격차(01강)를 통해 이를 다루며, DeepMind는 FSF v3의 기만적 정렬 모니터링을 통해 이를 다룹니다.

모델이 평가에서 샌드백킹을 수행하면 모든 프레임워크의 능력 임계값이 과소평가됩니다. 프레임워크는 측정 방식이 작동할 때만 작동합니다. 따라서 실험실 자체 평가 외에도 외부 측정(21강, METR) 및 적대적 평가가 필요합니다.

### 정책 읽기 기술

- 위치 찾기(Locate): 관심 있는 모든 능력은 정책에서 찾을 수 있어야 합니다. 찾을 수 없다면, 정책이 해당 능력을 다루지 않는 것입니다.
- 분류(Classify): Tracked (완화 조치를 발동)인지 Research (추적되지만 발동하지는 않음)인지 분류합니다. OpenAI는 이 용어를 사용하며, Anthropic과 DeepMind는 각각 자체적인 equivalents를 가지고 있습니다.
- 주기(Cadence): 정책이 선언된 일정에 따라 업데이트되는지, 아니면 특정 이벤트 후에만 업데이트되는지 확인합니다. 선언된 주기가 더 강력합니다.
- 독립성(Independence): 외부 검토가 필수인지 선택적인지 확인합니다. Anthropic은 Apollo 및 US AI Safety Institute와 협력하며, OpenAI는 METR와 협력하고, DeepMind는 주로 내부 SAG와 협력합니다.

```figure
a5-tracked-vs-research
```

## 사용하기

`code/main.py`는 작은 의사결정표(diff) 도구를 구현합니다. 주어진 능력(자율성, 기만적 정렬, R&D 자동화, 사이버 향상 등)에 대해 세 가지 정책이 해당 능력을 어떻게 분류하는지, 그리고 어떤 완화 조치가 발동되는지 출력합니다. 이는 정책 도구가 아닌 읽기 보조 도구입니다.

## 출시하기

`outputs/skill-cross-policy-diff.md`는 세 가지 프레임워크를 참조하여 특정 능력에 대한 정책 간 비교를 생성합니다.

## 연습 문제

1. `code/main.py`를 실행해 보세요. 원본 문서와 대조하여 검증할 수 있는 최소 두 가지 능력에 대해 diff 도구의 출력 결과가 정책과 일치하는지 확인하세요.

2. OpenAI Preparedness Framework v2를 전체적으로 읽어 보세요. 각 연구 범주(Research Category)를 식별하세요. 각 범주에 대해, 왜 추적(Tracked)이 아닌 연구(Research) 범주에 포함되는지 한 문장으로 작성하세요.

3. DeepMind FSF v3를 전체적으로 읽고, 2026년 4월의 추적 능력 수준(Tracked Capability Levels) 업데이트도 읽어 보세요. ML R&D 자율성 수준 1의 구체적인 평가 기준을 식별하세요. 이를 외부적으로 측정하려면 어떻게 해야 할까요?

4. 샌드배깅(Sandbagging)은 OpenAI의 연구 범주에 포함되어 있습니다. 샌드배깅 모델이 실제 능력을 드러내도록 강제하는 평가 설계해 보세요. 1강의 평가 컨텍스트 조작(eval-context-gaming) 논의 내용을 참조하세요.

5. 세 가지 정책을 특정 능력(직접 선택)에 대해 비교하세요. 가장 엄격한 분류를 내린 정책과 가장 덜 엄격한 분류를 내린 정책을 지정하세요. 원본 텍스트를 근거로 정당화하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|---|---|---|
| Preparedness Framework | "OpenAI의 확장 정책" | PF v2 (2025년 4월); 추적(Tracked) vs 연구(Research) 범주 |
| 추적 범주(Tracked Category) | "의무적 완화 조치" | 능력 및 안전장치 보고서(Capabilities + Safeguards Reports) 발동; SAG 검토 |
| 연구 범주(Research Category) | "모니터링만 수행" | 추적되지만 자동 완화 조치는 없음; 장거리 자율성(Long-range Autonomy) 포함 |
| Frontier Safety Framework | "DeepMind의 확장 정책" | FSF v3 (2025년 9월) + 추적 능력 수준(Tracked Capability Levels) (2026년 4월) |
| CCL | "핵심 능력 수준(Critical Capability Level)" | 도메인별 DeepMind 임계값 (사이버, 생물학, ML R&D, CBRN) |
| ML R&D 자율성 수준 1 | "R&D 자동화" | 경쟁력 있는 비용으로 AI R&D 파이프라인을 완전히 자동화 |
| 샌드배깅(Sandbagging) | "전략적 저조한 성과" | 모델이 평가에서 저조한 성과를 보임; OpenAI 연구 범주에 포함 |
| 도구적 추론 | "수단-목적 추론" | 목표를 달성하는 방법에 대한 추론; DeepMind 모니터링의 대상 |

## 추가 읽기

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 발표.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 전체 문서.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 발표.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 추적된 능력 수준 추가.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 형식 위험 보고서의 예.
