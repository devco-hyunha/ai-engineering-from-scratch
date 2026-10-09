# 레드 티밍 도구 — Garak, Llama Guard, PyRIT

> 세 가지 프로덕션 도구가 2026년 레드 티밍 스택을 구성합니다. Llama Guard (Meta) — MLCommons의 14가지 위험 범주에 대해 미세 조정된 Llama-3.1-8B 분류기입니다. 2025년 Llama Guard 4는 Llama 4 Scout에서 가지치기(pruning)된 12B 네이티브 멀티모달 분류기입니다. Garak (NVIDIA) — 환각, 데이터 유출, 프롬프트 인젝션, 독성 및 제일브레이크에 대한 정적, 동적 및 적응형 프로브를 갖춘 오픈소스 LLM 취약점 스캐너입니다. PyRIT (Microsoft) — Crescendo, TAP 및 깊은exploitation을 위한 맞춤형 컨버터 체인을 사용하는 다중 턴 레드 티밍 캠페인입니다. Llama Guard 3는 Meta의 "Llama 3 Herd of Models" (arXiv:2407.21783)에 문서화되어 있으며, Llama Guard 3-1B-INT4는 arXiv:2411.17713에, Garak의 프로브 아키텍처는 github.com/NVIDIA/garak에 문서화되어 있습니다. 이 도구들은 레드 티밍 연구(12-15강)와 배포(17강+) 사이의 2026년 프로덕션 인터페이스입니다.

**유형:** Build
**언어:** Python (표준 라이브러리, 도구 아키텍처 시뮬레이터 및 Llama Guard 스타일 분류기 목업)
**선수 요건:** 18단계 · 12-15강 (제일브레이크 및 IPI)
**시간:** 약 75분

## 학습 목표

- 안전 스택에서 Llama Guard 3/4의 위치를 설명하세요: 입력 분류기, 출력 분류기, 또는 둘 다.
- 14가지 MLCommons 위험 범주를 나열하고, 비자명한 범주 하나(코드 인터프리터 남용)를 명시하세요.
- Garak의 프로브 아키텍처를 설명하세요: 프로브, 감지자, 하네스.
- PyRIT의 다중 턴 캠페인 구조와 Garak 프로브와 결합하는 방식을 설명하세요.

## 문제점

12-15강은 공격 표면을 제시합니다. 프로덕션 배포는 반복적이고 확장 가능한 평가가 필요합니다. 2026년에는 세 가지 도구가 지배적입니다: Llama Guard (방어 분류기), Garak (스캐너), PyRIT (캠페인 오케스트레이터). 각 도구는 레드 티밍 수명 주기의 다른 계층을 대상으로 합니다.

## 개념

### Llama Guard (Meta)

Llama Guard 3는 MLCommons AILuminate의 14가지 범주에 대한 입력/출력 분류를 위해 미세 조정된 Llama-3.1-8B 모델입니다:
- 폭력 범죄, 비폭력 범죄, 성 관련, CSAM, 명예훼손
- 전문적 조언, 개인정보, IP, 무차별적 무기, 증오
- 자살/자해, 성적인 콘텐츠, 선거, 코드 인터프리터 남용

8개 언어를 지원합니다. 사용법: LLM 앞에 배치(입력 모듈레이션), LLM 뒤에 배치(출력 모듈레이션), 또는 둘 다 배치합니다. 두 사용법은 서로 다른 학습 분포를 생성합니다 — Llama Guard 3는 두 용도를 모두 처리하는 단일 모델로 제공됩니다.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, 모바일 CPU에서 초당 약 30개 토큰)는 양자화된 엣지 변형입니다.

Llama Guard 4 (2025년 4월)는 12B 모델이며, 네이티브 멀티모달이고 Llama 4 Scout에서 가지치기(pruning)되었습니다. 텍스트 8B 및 비전 11B 선행 모델을 텍스트와 이미지를 함께 입력받는 단일 분류기로 대체합니다.

### Garak (NVIDIA)

오픈소스 취약점 스캐너. 아키텍처:
- **프로브.** 환각(Hallucination), 데이터 유출(Data Leakage), 프롬프트 인젝션(Prompt Injection), 독성, 제일브레이크(Jailbreak)를 위한 공격 생성기. 정적(고정 프롬프트), 동적(생성된 프롬프트), 적응형(타겟 출력에 반응).
- **디텍터.** 예상 실패 모드에 대해 출력 점수를 매깁니다 — 독성, 유출, 제일브레이크.
- **하네스.** 프로브-디텍터 쌍을 관리하고, 캠페인을 실행하며, 보고서를 생성합니다.

TrustyAI는 Garak을 Llama-Stack shields(Prompt-Guard-86M 입력 분류기, Llama-Guard-3-8B 출력 분류기)와 통합하여 엔드투엔드 shielded-target 평가를 수행합니다. 티어 기반 점수(TBSA)는 이진 pass/fail을 대체합니다 — 모델은 같은 프로브에서 심각도 티어 3에서는 통과하고 심각도 티어 5에서는 실패할 수 있습니다.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. 다중 턴 레드 티밍(Red Teaming) 캠페인. 다음을 중심으로 구축됩니다:
- **컨버터.** 시드 프롬프트를 변환합니다 — 패러프레이즈(paraphrase), 인코딩, 번역, 역할극(roleplay).
- **오케스트레이터.** 캠페인을 실행합니다: Crescendo(점진적 고조), TAP(분기), RedTeaming(사용자 정의 루프).
- **점수 매기기.** LLM-as-judge 또는 classifier-as-judge.

PyRIT은 Garak의 더 무거운 사촌입니다. Garak은 수천 개의 단일 턴 프로브를 실행하는 반면, PyRIT은 특정 실패 모드를 깨뜨리도록 설계된 깊은 다중 턴 캠페인을 실행합니다.

### 스택

모델의 양쪽에 Llama Guard를 배치합니다. 회귀 테스트를 위해 Garak을 야간에 실행합니다. 릴리스 전 캠페인을 위해 PyRIT을 실행합니다. 이는 대부분의 프로덕션 배포에 대한 2026년 기본 구성입니다.

### 평가 함정

- **판정자 식별.** 세 도구 모두 LLM 판정자를 사용할 수 있습니다. 판정자 보정(calibration)은 보고된 ASR에 영향을 미칩니다(12강). 도구와 함께 판정자를 명시해 보세요.
- **프로브 노후화.** 모델이 패치됨에 따라 Garak 프로브는 노후화됩니다. 적응형 프로브(PAIR 형태)는 정적 프로브보다 노후화 속도가 느립니다.
- **Llama Guard의 정상 콘텐츠에 대한 FPR.** 초기 Llama Guard 버전은 정치적 및 LGBTQ+ 콘텐츠를 과잉 플래그했습니다. Llama Guard 3/4 보정은 개선되었지만, 배포별 보정은 이루어지지 않았습니다.

### 18단계에서의 위치

12-15강은 공격 계열입니다. 16강은 프로덕션 도구입니다. 17강(WMDP)은 이중 용량 능력에 대한 평가입니다. 18강은 이러한 도구를 정책 구조로 감싸는 프론티어 안전 프레임워크입니다.

```figure
al-guard-stack
```

## 사용하기

`code/main.py`는 장난감 Llama Guard 스타일 분류기(14개 범주에 대한 키워드 + 시맨틱 기능), 장난감 Garak 하네스(프로브-감지 루프), PyRIT 스타일 다중 턴 변환기 체인을 구축합니다. 세 도구를 모의 대상에 대해 실행하여 서로 다른 커버리지 서명(coverage signatures)을 관찰해 보세요.

## 출시하기

이 강의는 `outputs/skill-red-team-stack.md`을 생성합니다. 배포 설명이 주어지면, 세 도구 중 어떤 것이 적합한지, 각 도구에서 무엇을 구성해야 하는지, 어떤 회귀 테스트 주기(regression cadence)를 실행해야 하는지 명시합니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 단일 턴 공격과 다중 턴 공격에 대한 Llama-Guard 스타일 분류기의 감지율을 비교하세요.

2. 새로운 Garak 프로브를 구현하세요: base64로 인코딩된 유해한 요청. Llama-Guard 스타일 분류기에 의한 감지를 측정하세요.

3. PyRIT 스타일 변환기 체인을 "프랑스어로 번역한 후 패러프레이즈(paraphrase)" 변환기로 확장하세요. 공격 성공률을 다시 측정하세요.

4. Llama Guard 3의 위험 범주 목록을 읽어 보세요. 훈련 데이터가 합법적인 개발자 콘텐츠에서 높은 오탐율(false-positive rates)을 현실적으로 생성할 두 범주를 식별하세요.

5. Garak과 PyRIT의 설계 원칙을 비교하세요. 각각이 적합한 도구인 배포에 대해 논증하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Llama Guard | "분류기" | 14개 위험 범주를 가진 미세 조정된 Llama-3.1-8B/4-12B 안전 분류기 |
| Garak | "the scanner" | NVIDIA 오픈소스 취약점 스캐너; probes, detectors, harnesses |
| PyRIT | "the campaign tool" | Microsoft 다중 턴 레드팀 오케스트레이터; converters, orchestrators, scoring |
| Prompt-Guard | "the small classifier" | Meta의 86M 프롬프트 인젝션 분류기, Llama Guard와 함께 사용 |
| TBSA | "tier-based scoring" | 이진 결과를 대체하는 Garak의 티어 기반 통과/실패 |
| Converter chain | "paraphrase + encode + ..." | 다단계 공격을 구축하기 위한 PyRIT 구성 프리미티브 |
| MLCommons hazard categories | "the 14 taxonomies" | Llama Guard가 목표로 하는 산업 표준 분류 체계 |

## 추가 읽기

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 분류기
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 양자화된 모바일 분류기
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 스캐너 저장소 및 문서
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 캠페인 툴킷
