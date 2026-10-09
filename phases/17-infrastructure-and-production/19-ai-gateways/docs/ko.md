# AI 게이트웨이 — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> 게이트웨이는 애플리케이션과 모델 제공자 사이에 위치합니다. 핵심 기능은 제공자 라우팅, 폴백, 재시도, 속도 제한, 시크릿 참조, 관측 가능성, 가드레일입니다. 2026년 시장 점유율: **LiteLLM**은 MIT OSS로 100개 이상의 제공자를 지원하며 OpenAI 호환이지만, 약 2000 RPS(8 GB 메모리, 공개된 벤치마크에서 연쇄적 장애) 부근에서 성능이 저하됩니다. Python, 500 RPS 미만, 개발/프로토타이핑에 적합합니다. **Portkey**는 컨트롤 플레인 위치(가드레일, PII 마스킹, 제일브레이크 탐지, 감사 추적)에 있으며, 2026년 3월 Apache 2.0 오픈소스로 전환했고, 지연 오버헤드가 20-40 ms이며, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $100/모델/월 가격(Plus 티어에서 최대 5개)입니다. Kong를 이미 사용 중이라면 기업에 적합합니다. **Bifrost** (Maxim AI) — 구성 가능한 백오프로 자동 재시도, OpenAI 429 발생 시 Anthropic으로 폴백. **Cloudflare / Vercel AI Gateways** — 관리형, 무운영, 기본 재시도. 데이터 주권이 자체 호스팅 결정 요인입니다. Portkey와 Kong는 OSS + 선택적 관리형으로 중간 위치에 있습니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 게이트웨이 라우팅 시뮬레이터)
**선수 요건:** 17단계 · 01강 (관리형 LLM 플랫폼), 17단계 · 16강 (모델 라우팅)
**시간:** 약 60분

## 학습 목표

- 6가지 핵심 게이트웨이 기능(라우팅, 폴백, 재시도, 속도 제한, 시크릿, 관측 가능성, 가드레일)을 나열해 보세요.
- 4가지 2026년 게이트웨이(LiteLLM, Portkey, Kong AI, Bifrost)를 확장 한도와 사용 사례에 매핑해 보세요.
- Kong 벤치마크(Portkey 대비 228%, LiteLLM 대비 859%)를 인용하고, 500 RPS 이상에서 왜 중요한지 설명해 보세요.
- 데이터 주권과 운영 예산을 고려하여 자체 호스팅과 관리형 중 하나를 선택해 보세요.

## 문제점

제품이 OpenAI, Anthropic, 자체 호스팅 Llama를 호출합니다. 각 제공자는 SDK, 오류 모델, 속도 제한, 인증 방식이 다릅니다. 폴백(OpenAI가 429를 반환하면 Anthropic 시도), 단일 자격 증명 저장소, 통합 관측 가능성, 테넌트별 속도 제한이 필요합니다.

애플리케이션 레이어에서 이를 재발명하면 모든 서비스가 모든 제공자와 결합됩니다. 게이트웨이 레이어는 이를 하나의 프로세스와 하나의 API(통상 OpenAI 호환)로 통합하여 제공자 측으로 분산합니다.

## 개념

### 6가지 핵심 기능

1. **공급자 라우팅** — OpenAI, Anthropic, Gemini, 자체 호스팅 등을 하나의 API로 통합합니다.
2. **폴백** — 429, 5xx 오류 또는 품질 실패 시 다른 공급자로 재시도합니다.
3. **재시도** — 지수 백오프, 시도 횟수 제한.
4. **속도 제한** — 테넌트별, 키별, 모델별.
5. **비밀 참조** — 런타임에 볼트에서 자격 증명을 가져옵니다 (앱에 포함하지 마세요).
6. **관측 가능성** — OTel + GenAI 속성 (17단계 · 13강) + 비용 귀속.
7. **가드레일** — PII 마스킹, 제일브레이크 탐지, 허용 주제 필터.

### LiteLLM — MIT OSS, Python

- 100개 이상의 공급자, OpenAI 호환, 라우터 구성, 폴백, 기본 관측 가능성.
- Kong의 벤치마크에서 약 2000 RPS 부근에서 성능이 저하됩니다. 8 GB 메모리 사용량, 지속 부하 시 연쇄 장애가 발생합니다.
- 최적 사용 사례: Python 앱, 500 RPS 미만, 개발/스테이징 게이트웨이, 실험적 라우팅.
- 비용: OSS는 $0; 클라우드 무료 티어가 존재합니다.

### Portkey — 컨트롤 플레인 포지셔닝

- 2026년 3월 기준으로 Apache 2.0 OSS입니다. 가드레일, PII 마스킹, 제일브레이크 탐지, 감사 추적.
- 요청당 20-40 ms의 지연 오버헤드.
- 보존 기간 + SLA가 포함된 프로덕션 티어는 월 $49입니다.
- 최적 사용 사례: 가드레일과 관측 가능성을 통합하여 필요한 규제 산업.

### Kong AI Gateway — 스케일 전략

- Kong Gateway (성숙한 API 게이트웨이 제품, lua+OpenResty) 위에 구축되었습니다.
- 12-CPU 동등 환경에서의 Kong 자체 벤치마크: Portkey보다 228% 빠르고, LiteLLM보다 859% 빠릅니다.
- 가격: 모델당 월 $100, Plus 티어에서 최대 5개.
- 최적 사용 사례: 이미 Kong를 사용 중; 1000 RPS 이상; 라이선스 구매 의향이 있는 경우.

### Bifrost (Maxim AI)

- 구성 가능한 백오프를 사용하는 자동 재시도.
- OpenAI의 429 오류 시 Anthropic으로 폴백하는 것은 표준 레시피입니다.
- 신규 진입자; 상용 제품.

### Cloudflare AI Gateway / Vercel AI Gateway

- 관리형, 무운영. 기본 재시도 및 관측 가능성.
- 최적 사용 사례: Cloudflare/Vercel에서 가장자리 서빙을 하는 JavaScript 앱.
- Kong/Portkey에 비해 가드레일 및 속도 제한 기능이 제한적입니다.

### 셀프 호스트 vs 관리형

데이터 주권이 결정 요인입니다. 헬스케어와 금융은 기본적으로 셀프 호스트(LiteLLM, Portkey OSS, Kong)를 선택합니다. 소비자 제품은 기본적으로 관리형(Cloudflare AI Gateway)이나 중간 계층(Portkey 관리형)을 선택합니다. 하이브리드: 규제 대상 테넌트는 셀프 호스트, 나머지는 관리형으로 구성합니다.

### 레이턴시 예산

- LiteLLM: 일반적인 오버헤드는 5-15 ms입니다.
- Portkey: 오버헤드는 20-40 ms입니다.
- Kong: 오버헤드는 3-8 ms입니다.
- Cloudflare/Vercel: 오버헤드는 1-3 ms입니다 (엣지 이점).

게이트웨이 레이턴시는 TTFT에 직접적으로 추가됩니다. TTFT P99 < 100 ms SLA라면 Kong이나 Cloudflare를 선택하세요. P99 < 500 ms라면 어떤 것이든 가능합니다.

### 속도 제한의 의미는 중요합니다

단순한 토큰 버킷은 중간 규모까지 작동합니다. 멀티 테넌트는 슬라이딩 윈도우 + 버스트 허용 + 테넌트별 티어링이 필요합니다. LiteLLM은 토큰 버킷을, Kong은 슬라이딩 윈도우를, Portkey는 티어링을 제공합니다.

### 게이트웨이 + 관측 가능성 + 라우팅은 결합됩니다

17단계 · 13강(관측 가능성) + 16강(모델 라우팅) + 19강(게이트웨이)은 프로덕션에서 동일한 계층입니다. 세 가지를 모두 커버하는 도구를 선택하거나 신중하게 연결하세요: 대부분의 2026년 배포는 Helicone(관측 가능성)이나 Portkey(가드레일)를 Kong(확장성)과 결합하여 역할을 분리합니다.

### 기억해야 할 수치

- LiteLLM: 약 2000 RPS, 8 GB 메모리에서 한계에 도달합니다.
- Portkey: 오버헤드 20-40 ms; 2026년 3월부터 Apache 2.0 라이선스입니다.
- Kong: Portkey보다 228% 빠르고, LiteLLM보다 859% 빠릅니다.
- Kong 가격: 모델당 월 $100, Plus 티어는 최대 5개입니다.
- Cloudflare/Vercel: 엣지에서 오버헤드 1-3 ms입니다.

```figure
mx-gateway-fallback
```

## 사용하기

`code/main.py`는 429/5xx 주입 하에 3개 제공자 간 폴백을 포함한 게이트웨이 라우팅을 시뮬레이션합니다. 레이턴시, 재시도율, 폴백 적중률을 보고합니다.

## 출시하기

이 강의는 `outputs/skill-gateway-picker.md`를 생성합니다. 규모, 운영 태세, 컴플라이언스, 레이턴시 예산을 고려하여 게이트웨이를 선택합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. OpenAI→Anthropic→셀프 호스트로 폴백을 구성하세요. 제공자 오류율이 5%일 때 예상 적중률은 얼마인가요?
2. SLA가 TTFT P99 < 200 ms이고 기준선이 300 ms입니다. 어떤 게이트웨이가 예산 내에 유지되나요?
3. 헬스케어 고객은 자체 호스팅 + PII 마스킹 + 감사 로그가 필요합니다. Portkey OSS 또는 Kong을 선택하세요.
4. LiteLLM과 Kong을 비교해 보세요: 팀이 마이그레이션해야 하는 RPS 상한선은 어디인가요?
5. 다중 테넌트 SaaS를 위한 속도 제한 정책을 설계해 보세요: 무료 티어, 체험 티어, 유료 티어. 토큰 버킷 방식과 슬라이딩 윈도우 방식 중 어떤 것을 선택해야 할까요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 게이트웨이 | "API 중개자" | 앱과 제공자 사이에 위치하는 프로세스 |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "가드레일 게이트웨이" | 제어 평면 + 관측 가능성, Apache 2.0 |
| Kong AI Gateway | "규모 확장형" | Kong Gateway 기반, 벤치마크 리더 |
| Bifrost | "Maxim's gateway" | 재시도 + Anthropic 폴백 레시피 |
| Cloudflare AI Gateway | "엣지 관리형" | 엣지 배포형 관리 게이트웨이, 무운영 |
| PII 마스킹 | "data scrub" | 모델로 전송하기 전에 정규식 + NER 마스킹 적용 |
| 제일브레이크 감지 | "프롬프트 인젝션 가드" | 사용자 입력에 대한 분류기 |
| 감사 추적 | "규제된 로그" | 모든 LLM 호출의 변경 불가능한 기록 |
| 토큰 버킷 | "간단한 속도 제한" | 재충전 기반 속도 제한기 |
| 슬라이딩 윈도우 | "정밀한 속도 제한" | 시간 윈도우 기반 속도 제한기; 더 공정한 처리 |

## 추가 읽기

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
