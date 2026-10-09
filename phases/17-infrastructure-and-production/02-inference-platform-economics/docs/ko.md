# 추론 플랫폼 경제학 — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> 2026년 추론 시장은 더 이상 GPU 시간 대여가 아닙니다. 시장은 커스텀 실리콘(Groq, Cerebras, SambaNova), GPU 플랫폼(Baseten, Together, Fireworks, Modal), API 우선 마켓플레이스(Replicate, DeepInfra)로 분화됩니다. Fireworks가 $1/hr per GPU on May 1, 2026, and $40억 달러의 기업 가치를 달성한 것은 하루 10조 개 이상의 토큰 처리량에 기반한 볼륨 주도 모델이 작동한다는 것을 보여줍니다. Baseten은 2026년 1월에 $300M Series E at $50억 달러의 투자를 유치했습니다. 경쟁적 포지셔닝 규칙은 간단합니다: Fireworks는 지연 시간을 최적화하고, Together는 카탈로그의 폭을 최적화하며, Baseten은 기업용 완성도를 최적화하고, Modal은 Python 네이티브 개발자 경험(DX)을 최적화하며, Replicate는 멀티모달 도달 범위를 최적화하고, Anyscale는 분산 Python을 최적화합니다. 이 강에서는 창업자에게 전달할 수 있는 매트릭스를 제공합니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 호출별 경제성 비교 도구)
**선수 요건:** 17단계 · 01강 (관리형 LLM 플랫폼), 17단계 · 04강 (서빙 엔진 내부 구조)
**시간:** 약 60분

## 학습 목표

- 세 가지 시장 세그먼트(커스텀 실리콘, GPU 플랫폼, API 우선)를 나열하고 각 벤더를 세그먼트에 매핑해 보세요.
- "토큰당" API 가격 모델이 하드웨어가 아닌 서빙 엔진의 비용 곡선으로 수렴하는 이유를 설명해 보세요.
- 최소 세 벤더에 걸쳐 요청당 유효 비용을 계산하고, 분당 과금(Baseten, Modal)이 토큰당 과금보다 유리한 시점을 설명해 보세요.
- 주어진 워크로드(서버리스 버스트형, 안정적 고처리량, 미세 조정 변형, 멀티모달)에 적합한 기본 플랫폼을 식별해 보세요.

## 문제점

관리형 하이퍼스케일러 플랫폼을 평가했습니다. 더 좁고 빠른 제공자가 필요하다고 결정했습니다 — 지연 시간에는 Fireworks, 폭에는 Together, 미세 조정된 커스텀 모델에는 Baseten. 이제 여섯 가지 실제 선택지가 있으며 가격 페이지는 정렬되지 않습니다. Fireworks는 $/M tokens; Baseten shows $/분, Modal은 $/second; Replicate shows $/예측을 표시합니다. 워크로드를 모델링하지 않고는 직접 비교할 수 없습니다.

더 나쁜 점은, 각 가격 페이지 뒤의 비즈니스 모델이 다르다는 것입니다. Fireworks는 공유 GPU에서 자체 커스텀 엔진(FireAttention)을 실행하며, 토큰당 요금은 그들의 활용 곡선을 반영합니다. Baseten은 Truss + 전용 GPU를 제공하며, 분당 요금은 독점성을 반영합니다. Modal은 진정한 Python 서버리스로, 초당 과금과 초당 단위 콜드 스타트를 제공합니다. 동일한 출력(LLM 응답)에 대해 세 가지 다른 비용 함수가 존재합니다.

이 강의는 이 여섯 가지를 모델링하며, 각각이 언제 우세한지 알려줍니다.

## 개념

### 세 가지 세그먼트

**커스텀 실리콘** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). 일반적으로 동일한 모델의 GPU 기반 클러스터보다 디코딩이 5-10배 더 빠릅니다. 토큰당 가격이 더 높습니다(Groq는 2025년 말 Llama-70B에서 약 $0.99/M)하지만, 지연 시간에 민감한 사용 사례에는 비교할 수 없는 성능을 제공합니다. Groq는 음성 에이전트와 실시간 번역의 생산 환경 선택지입니다.

**GPU 플랫폼** — Baseten, Together, Fireworks, Modal, Anyscale. NVIDIA(H100, H200, 2026년 B200) 또는 때때로 AMD에서 실행됩니다. "raw GPU rental"(RunPod, Lambda)과 "hyperscaler managed service"(Bedrock) 사이의 경제 계층입니다.

**API-first 마켓플레이스** — Replicate, DeepInfra, OpenRouter, Fal. 넓은 카탈로그, 예측당 또는 초당 과금, 첫 호출까지의 시간을 강조합니다.

### Fireworks — 지연 시간 최적화 GPU 플랫폼

- FireAttention 엔진(커스텀); 동등한 구성에서 vLLM보다 지연 시간이 4배 낮다고 마케팅됩니다.
- 비인터랙티브 워크로드를 위해 서버리스 요금의 약 50%인 배치 계층이 있습니다.
- 미세 조정된 모델은 기본 모델과 동일한 요금으로 제공되며, LoRA에 프리미엄을 부과하는 제공업체에 비해 진정한 차별점입니다.
- 2026년 중반: 2026년 5월 1일부터 온디맨드 GPU 렌탈을 시간당 $1로 인상했습니다. 대규모 볼륨 가격은 협상 가능합니다.
- 재무 신호: $4B 가치 평가, 하루 10T+ 토큰 처리.

### Together — 범위 최적화

- 상류 공개 후 며칠 내의 오픈소스 릴리스를 포함하여 200개 이상의 모델.
- 동등한 LLM 모델에서 Replicate보다 50-70% 저렴합니다. "AI Native Cloud" 포지셔닝은 볼륨과 카탈로그입니다.
- 추론 + 미세 조정 + 훈련을 하나의 API에서 제공합니다.

### Baseten — 엔터프라이즈 완성도 최적화

- Truss 프레임워크: 의존성, 시크릿, 서빙 구성을 하나의 매니페스트로 모델 패키징합니다.
- T4부터 B200까지의 GPU 범위. 합리적인 콜드 스타트 완화와 함께 분 단위 과금.
- SOC 2 Type II, HIPAA 준비 완료. 핀테크 및 헬스케어에서 흔히 선택합니다.
- $5B valuation, January 2026 Series E ($300M (CapitalG, IVP, NVIDIA로부터).

### Modal — Python 네이티브 최적화

- 순수 Python으로 인프라를 코드로 구현합니다. 함수에 `@modal.function(gpu="A100")`을 장식(decorate)하고 한 명령어로 배포하세요.
- 초 단위 과금. 프리워밍으로 콜드 스타트가 2-4초이며, 소형 모델은 <1초입니다.
- $87M Series B at $1.1B 밸류에이션 (2025). 독립적인 설문조사에서 가장 높은 개발자 경험 점수를 받았습니다.

### Replicate — 멀티모달 확장성

- 예측 단위 과금. 이미지, 비디오, 오디오 모델의 기본 플랫폼입니다.
- 통합 생태계 (Zapier, Vercel, CMS 플러그인).
- LLM 토큰 단가 경쟁력은 낮지만, 멀티모달 다양성에서는 우위입니다.

### Anyscale — Ray 네이티브

- Ray 기반으로 구축되었습니다. RayTurbo는 Anyscale의 독점 추론 엔진으로 vLLM과 경쟁합니다.
- 추론 단계가 더 큰 그래프의 한 노드인 분산 Python 워크로드에 가장 적합합니다.
- 관리형 Ray 클러스터; Ray AIR 및 Ray Serve와 긴밀한 통합.

### 토큰 단위 대 분 단위 — 각각이 유리한 경우

토큰 단위 과금은 워크로드가 지연에 민감하지 않고 버스트(bursty)할 때 합리적입니다 — 사용한 만큼만 지불합니다. 분 단위 과금은 사용률이 높고 예측 가능할 때 합리적입니다 — GPU가 포화되면 토큰 단위 과금보다 유리해집니다.

대략적인 규칙: 전용 GPU의 지속 사용률이 약 30% 이상인 워크로드에서는 분 단위 과금(Baseten, Modal)이 토큰 단위 과금(Fireworks, Together)보다 유리해집니다. 그 이하에서는 유휴 상태에 대한 비용을 피할 수 있으므로 토큰 단위 과금이 유리합니다.

### 커스텀 엔진이 진정한 해자(moat)입니다

vLLM과 SGLang 위에 있는 모든 플랫폼은 커스텀 엔진을 주장합니다. FireAttention, RayTurbo, Baseten의 추론 스택. 커스텀 엔진 주장은 마케팅의 영역입니다. 솔직한 표현은 vLLM + SGLang이 생산 오픈소스 추론의 약 80%를 차지하며, 플랫폼 계층의 차별점은 DX, 어트리뷰션, SLA라는 것입니다.

### 기억해야 할 수치

- Fireworks GPU 임대: 2026년 5월 1일부터 시간당 $1 인상.
- Fireworks 주장: 동등한 구성에서 vLLM보다 지연 시간이 4배 낮음.
- Together: LLM의 경우 Replicate보다 50-70% 저렴함.
- Baseten 기업 가치: $5B (Series E, Jan 2026, $300M 라운드).
- Modal 기업 가치: $1.1B (Series B, 2025).
- 지속적인 사용률이 약 30% 이상일 때, 분당 과금이 토큰당 과금보다 유리합니다.

```figure
cost-per-token
```

## 사용하기

`code/main.py`는 가격 모델 전반에 걸쳐 합성 워크로드를 사용하여 6개 벤더를 비교합니다. $/day and effective $/M 토큰을 보고합니다. 토큰당 과금과 분당 과금의 손익분기점을 찾기 위해 실행해 보세요.

## 출시하기

이 강의는 `outputs/skill-inference-platform-picker.md`를 생성합니다. 워크로드 프로필, SLA, 예산을 고려하여 주요 추론 플랫폼을 선택하고 차선책을 지정합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. H100 한 개에서 70B 모델을 사용할 때, Baseten(분당 과금)이 Fireworks(토큰당 과금)보다 유리해지는 지속 사용률은 몇 %인가요? 교차점을 직접 유도하고 경험칙과 비교해 보세요.
2. 귀사의 제품은 이미지 생성, 채팅, 음성 인식(STT)을 제공합니다. 각 모달리티에 적합한 플랫폼을 선택하고, 이를 통합하는 게이트웨이 패턴을 지정하세요.
3. Fireworks가 주요 모델의 가격을 시간당 $1 인상합니다. 트래픽의 40%가 배치 티어(50% 할인)로 이동할 경우, 혼합 비용 영향을 모델링하세요.
4. 규제 대상 고객이 SOC 2 Type II + HIPAA + 전용 GPU를 요구합니다. 실행 가능한 세 플랫폼은 무엇이며, FinOps 측면에서 가장 유리한 플랫폼은 어디인가요?
5. Llama 3.1 70B의 예측 1,000건당 비용을 Fireworks 서버리스, Together 온디맨드, Baseten 전용, Replicate API와 비교하세요. 하루 10건일 때 가장 저렴한 곳은 어디인가요? 10,000건일 때는요?

## 핵심 용어

| 용어 | 통용되는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Custom silicon | "비-GPU 칩" | Groq LPU, Cerebras WSE, SambaNova RDU — 디코딩에 최적화됨 |
| FireAttention | "Fireworks 엔진" | 커스텀 어텐션 커널; vLLM보다 지연 시간이 4배 낮다고 마케팅됨 |
| Truss | "Baseten의 형식" | 모델 패키징 매니페스트; 의존성 + 시크릿 + 서빙 구성 |
| Per-token | "API 과금" | 소비된 토큰 기준으로 과금; 유휴 상태에 대한 비용 없음 |
| 분당 | "전용 가격" | 벽시계 GPU 시간으로 과금; 높은 활용률에서 유리 |
| 예측당 | "복제 가격" | 모델 호출당 과금; 이미지/비디오에서 흔함 |
| RayTurbo | "Anyscale 엔진" | Ray 기반의 독점 추론; Ray 클러스터에서 vLLM과 경쟁 |
| 배치 티어 | "50% 할인" | 낮은 요금의 비인터랙티브 큐; Fireworks, OpenAI에서 흔함 |
| 기본 요금으로 미세 조정 | "Fireworks LoRA" | LoRA로 서빙된 요청을 기본 모델의 요금으로 과금 (차별화 요소) |

## 추가 읽기

- [Fireworks Pricing](https://fireworks.ai/pricing) — 토큰당 요금, 배치 티어, GPU 렌탈.
- [Baseten Pricing](https://www.baseten.co/pricing/) — 분당 요금, 커밋된 용량, 엔터프라이즈 티어.
- [Modal Pricing](https://modal.com/pricing) — 초당 GPU 요금 및 무료 티어.
- [Together AI Pricing](https://www.together.ai/pricing) — 모델 카탈로그 및 토큰당 요금.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 및 관리형 Ray 가격.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 비교 평가.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 벤더 현황.
