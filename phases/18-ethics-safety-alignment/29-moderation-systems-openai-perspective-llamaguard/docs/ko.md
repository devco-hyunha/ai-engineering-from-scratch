# 모더레이션 시스템 — OpenAI, Perspective, Llama Guard

> 프로덕션 모더레이션 시스템은 16-16강에서 정의된 안전 정책을 실제 운영에 적용합니다. OpenAI Moderation API: `omni-moderation-latest` (2024)는 GPT-4o 기반으로 구축되어 한 번의 호출로 텍스트와 이미지를 분류합니다. 다국어 테스트 세트에서 이전 버전보다 42% 더 나은 성능을 보이며, 응답 스키마는 13개 카테고리 부울 값을 반환합니다 — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic. 대부분의 개발자에게 무료입니다. 계층적 패턴: 입력 모더레이션(생성 전), 출력 모더레이션(생성 후), 사용자 정의 모더레이션(도메인 규칙). 비동기 병렬 호출로 지연 시간을 숨기며, 플래그가 설정되면 플레이스홀더 응답을 반환합니다. Llama Guard 3/4 (16강): 14개 MLCommons 위험 요소, Code Interpreter Abuse, 8개 언어(v3), 다중 이미지(v4). Perspective API (Google Jigsaw): LLM 기반 모더레이터 물결 이전의 독성 점수 산정 도구로, 주로 단일 차원 독성 점수를 제공하며 severe-toxicity/insult/profanity 변형이 있습니다. 콘텐츠 모더레이션 연구의 기준선으로 사용됩니다. 폐기 일정: Azure Content Moderator는 2024년 2월에 폐기 예정이 발표되었으며, 2027년 2월에 완전히 폐기되어 Azure AI Content Safety로 대체됩니다.

**유형:** Build
**언어:** Python (stdlib, 3계층 모더레이션 하네스)
**선수 요건:** 18단계 · 16강 (Llama Guard / Garak / PyRIT)
**시간:** 약 60분

## 학습 목표

- OpenAI Moderation API의 카테고리 분류 체계가 Llama Guard 3의 MLCommons 세트와 어떻게 다른지 설명해 보세요.
- 3계층 모더레이션 패턴(입력, 출력, 사용자 정의)을 설명하고 각 패턴의 실패 모드 하나를 지정해 보세요.
- Perspective API가 LLM 이전 시대의 기준선으로서의 위치를 설명하고, 연구에서 계속 사용되는 이유를 설명해 보세요.
- Azure 폐기 일정을 명시해 보세요.

## 문제점

16-16강은 공격 및 방어 도구를 설명합니다. 29강은 사용자가 제품과 상호작용하는 표면에서 방어를 실제 운영에 적용하는 배포된 모더레이션 시스템을 다룹니다. 3계층 패턴은 2026년 기본 구성입니다.

## 개념

### OpenAI Moderation API

`omni-moderation-latest` (2024). GPT-4o 기반으로 구축되었습니다. 한 번의 호출로 텍스트와 이미지를 분류합니다. 대부분의 개발자에게 무료입니다.

카테고리 (응답 스키마의 13개 부울 값):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

멀티모달 지원은 `violence`, `self-harm`, `sexual`에 적용되지만 `sexual/minors`에는 적용되지 않으며, 나머지는 텍스트 전용입니다.

`code/main.py`의 코드 하네스에서는 교육적 단순성을 위해 `/threatening`, `/intent`, `/instructions`, `/graphic` 하위 카테고리를 상위 부모 카테고리로 통합합니다. 프로덕션 코드는 전체 13개 카테고리 스키마를 사용해야 합니다.

이전 세대 모더레이션 엔드포인트보다 다국어 테스트 세트에서 42% 더 나은 성능을 보입니다. 카테고리별 점수; 애플리케이션이 임계값을 설정합니다.

### Llama Guard 3/4

16강에서 다루었습니다. 14개의 MLCommons 위험 카테고리 (OpenAI의 13개 응답 스키마 부울 값과 조직 방식이 다름). 8개 언어 지원 (v3). Llama Guard 4 (2025년 4월)는 네이티브 멀티모달이며, 12B입니다.

OpenAI와 Llama Guard의 분류 체계는 겹치지만 다릅니다. OpenAI는 "illicit"을 광범위한 카테고리로 포함하고 있으며, Llama Guard는 "violent crimes"와 "non-violent crimes"를 별도로 구분합니다. 배포는 정책 분류 체계 적합성에 따라 선택합니다.

### Perspective API (Google Jigsaw)

LLM-as-moderator 물결 이전 (2020년 이전)의 독성 점수 시스템. 카테고리: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. 단일 차원 주요 점수 (TOXICITY)와 하위 차원 변형.

API가 안정적이고 문서화되어 있으며 수년간의 보정 데이터가 있기 때문에 콘텐츠 모더레이션 연구 기준으로 널리 사용됩니다. 현대적인 LLM 관련 사용 사례의 경우, Llama Guard나 OpenAI Moderation이 일반적으로 더 적합합니다.

### 3계층 패턴

1. **입력 모더레이션.** 생성 전에 사용자 프롬프트를 분류합니다. 플래그가 지정되면 거부합니다. 지연 시간: 분류기 호출 한 번.
2. **출력 모더레이션.** 전달 전에 모델의 출력을 분류합니다. 플래그가 지정되면 거절 응답으로 대체합니다. 지연 시간: 생성 후 분류기 호출 한 번.
3. **맞춤형 Moderation.** 도메인 특화 규칙(정규식, 허용 목록, 비즈니스 정책). 입력 또는 출력 단계에서 실행됩니다.

세 계층은 설계상 순차적입니다: 입력 Moderation은 생성 전에 완료되어야 하며, 출력 Moderation은 생성 후 실행됩니다. 병렬 처리는 계층 내에서 적용됩니다 — 여러 분류기(예: OpenAI Moderation + Llama Guard + Perspective)를 동일한 텍스트에 동시에 실행하면 분류기별 지연 시간을 숨길 수 있습니다. 선택적 최적화로서, 입력 Moderation이 완료되는 동안 토큰-1 스트리밍을 지연시키고 플레이스홀더 응답("잠시만요, 확인 중...")을 표시할 수 있습니다. 플래그 동작은 설정 가능합니다: 거부, 정화, 인간 검토로 에스컬레이션.

### 실패 모드

- **입력 전용.** 출력 환각(Hallucination)을 잡지 못합니다(12-14강의 인코딩 공격은 입력 분류기를 우회합니다).
- **출력 전용.** 모든 입력이 모델에 도달하도록 허용합니다; 비용을 증가시킵니다; 내부 추론을 공격자에게 노출합니다.
- **맞춤형 전용.** 카테고리 전반에 걸쳐 견고하지 않습니다; 정규식은 취약합니다.

계층형이 기본입니다. 이중 안전장치입니다.

### Azure 폐기

Azure Content Moderator: 2024년 2월에 폐기, 2027년 2월에 은퇴. LLM 기반이며 Azure OpenAI와 통합되는 Azure AI Content Safety로 대체됩니다. 이 마이그레이션은 Azure 배포를 위한 2024-2027 필드 단위 프로젝트입니다.

### 18단계에서의 위치

16강은 레드 티밍(Red Teaming) 맥락에서의 Moderation 도구를 다룹니다. 29강은 배포된 Moderation을 다룹니다. 30강은 현재의 이중 용량 능력 증거로 마무리됩니다.

```figure
an-moderation-layers
```

## 사용하기

`code/main.py`는 세 계층 Moderation 하네스를 구축합니다: 입력 Moderation(키워드 + 카테고리 점수), 출력 Moderation(출력에 동일한 분류기 적용), 맞춤형 Moderation(도메인 규칙). 입력을 실행하여 어떤 계층이 무엇을 잡는지 관찰할 수 있습니다.

## 출시하기

이 강은 `outputs/skill-moderation-stack.md`를 생성합니다. 배포가 주어지면 Moderation 스택 구성을 권장합니다: 입력에 어떤 분류기, 출력에 어떤 분류기, 어떤 맞춤형 규칙, 그리고 엣지 케이스에 어떤 판정자를 사용할지.

## 연습 문제

1. `code/main.py`를 실행하세요. 양성, 경계, 유해 입력을 세 계층 모두를 통해 실행하세요. 각 입력에 대해 어떤 계층이 발동하는지 보고하세요.

2. 하네스를 확장하여 특정 범주에 Perspective-API 스타일의 독성 점수를 적용해 보세요. 임계값 동작을 범주 점수와 비교해 보세요.

3. OpenAI Moderation API 문서와 Llama Guard 3 범주 목록을 읽어 보세요. 각 OpenAI 범주를 가장 가까운 Llama Guard 범주에 매핑해 보세요. 명확하게 매핑되지 않는 세 범주를 식별해 보세요.

4. 코드 어시스턴트 배포(예: GitHub Copilot)를 위한 모더레이션 스택을 설계해 보세요. 가장 관련성이 높고 낮은 범주를 식별하고 사용자 정의 규칙을 제안해 보세요.

5. Azure Content Moderator는 2027년 2월에 종료됩니다. Azure AI Content Safety로의 마이그레이션을 계획해 보세요. 마이그레이션에서 가장 위험한 요소를 식별해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o 기반의 13개 범주(텍스트) 분류기로 부분적인 멀티모달 지원 |
| Perspective API | "Google Jigsaw toxicity" | LLM 이전 시대의 독성 점수 기준선 |
| Llama Guard | "MLCommons 14-category" | Meta의 위험 분류기(v3: 8B 텍스트, 8개 언어; v4: 12B 멀티모달) |
| 입력 모더레이션 | "pre-generation filter" | 모델 호출 전 사용자 프롬프트에 대한 분류기 |
| 출력 모더레이션 | "post-generation filter" | 전달 전 모델 출력에 대한 분류기 |
| 사용자 정의 모더레이션 | "domain rules" | 배포별 규칙(정규식, 허용 목록, 정책) |
| 계층적 모더레이션 | "all three layers" | 표준 프로덕션 배포 패턴 |

## 추가 읽기

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 엔드포인트
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 저장소
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 독성 점수
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 대체 서비스
