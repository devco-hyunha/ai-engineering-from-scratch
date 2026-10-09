# 프롬프트 캐싱과 컨텍스트 캐싱

> 시스템 프롬프트가 4,000 토큰이고, RAG 컨텍스트가 20,000 토큰이라고 가정해 보세요. 모든 요청에 두 항목을 함께 보내며, 매번 두 항목에 대한 비용을 지불합니다. 프롬프트 캐싱을 사용하면 제공자가 해당 접두어를 서버 측에 따뜻하게(warm) 유지하고, 재사용 시 정상 요금의 10%만 청구합니다. 올바르게 사용하면 추론 비용을 50–90% 절감하고 첫 토큰까지의 시간(TTFT) 지연을 40–85% 줄일 수 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 11단계 · 01강 (프롬프트 엔지니어링), 11단계 · 05강 (컨텍스트 엔지니어링), 11단계 · 11강 (캐싱과 비용)
**시간:** 약 60분

## 문제점

코딩 에이전트(Coding Agent)는 대화의 모든 턴(turn)에서 동일한 15,000 토큰의 시스템 프롬프트를 Claude에 전송합니다. 20번의 턴은 입력 비용만으로도 $3/M input tokens is $0.90이 발생하며, 이는 사용자의 실제 메시지 비용은 포함하지 않은 값입니다. 하루 10,000건의 대화로 계산하면, 변하지 않는 텍스트에 대한 비용이 하루 $9,000에 달합니다.

프롬프트를 줄이면 품질이 저하되므로 축소할 수 없습니다. 모델이 모든 턴에서 이를 필요로 하므로 전송을 피할 수도 없습니다. 유일한 해결책은 제공자가 이미 확인한 접두어(prefix)에 대해 전체 요금을 지불하는 것을 멈추는 것입니다.

그 해결책이 프롬프트 캐싱입니다. Anthropic은 2024년 8월에 이를 출시했으며(2025년 1시간 연장 TTL 변형 포함), OpenAI는 그해 후반에 이를 자동화했고, Google은 Gemini 1.5와 함께 명시적인 컨텍스트 캐싱을 출시했습니다. 세 제공자 모두 현재 프론티어 모델의 일급(first-class) 기능으로 이를 제공합니다.

## 개념

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**메커니즘.** 요청의 접두어가 최근 요청의 접두어와 일치하면, 제공자는 토큰을 다시 인코딩하는 대신 이전 실행의 KV 캐시를 제공합니다. 첫 번째 요청 시에는 작은 쓰기 프리미엄(write premium)을 지불하고, 이후에는 큰 읽기 할인(read discount)을 적용받습니다.

**2026년 세 제공자의 특징.**

| 제공자 | API 스타일 | 적중 할인 | 쓰기 프리미엄 | 기본 TTL | 최소 캐싱 가능 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 콘텐츠 블록에 명시적인 `cache_control` 마커 사용 | 입력 비용 90% 할인 | 25% 추가 요금 | 5분 (1시간까지 연장 가능) | 1,024 토큰 (Sonnet/Opus), 2,048 토큰 (Haiku) |
| OpenAI | 자동 접두어 감지 | 입력 비용 50% 할인 | 없음 | 최대 1시간 (최선 노력) | 1,024 토큰 |
| Google (Gemini) | 명시적 `CachedContent` API | 저장소 과금; 정상 요금의 약 25%로 읽기 | 토큰·시간당 저장소 요금 | 사용자 설정 (기본값 1시간) | 4,096 토큰 (Flash), 32,768 (Pro) |

**불변 조건.** 세 가지 캐시 모두 접두어만 캐시합니다. 요청 간에 토큰이 하나라도 다르면, 첫 번째로 다른 토큰 이후의 모든 내용은 미스(miss)가 됩니다. *안정적인* 부분은 상단에, *변동하는* 부분은 하단에 배치하세요.

### 캐시에 친화적인 레이아웃

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

순서를 위반하면 — 사용자 메시지를 시스템 프롬프트 위에 배치하거나, 소수 예시(few-shots) 사이에 동적 검색 결과를 섞으면 — 캐시는 절대 적중하지 않습니다.

### 손익분기점 계산

Anthropic의 25% 쓰기 프리미엄은 캐시된 블록이 최소 두 번 읽혀야 순이익이 발생한다는 의미입니다. 1회 쓰기 + 1회 읽기는 요청당 평균 0.675배 비용 (32% 절감); 1회 쓰기 + 10회 읽기는 평균 0.205배 (80% 절감)입니다. 경험칙: TTL 내에서 최소 3회 이상 재사용할 것으로 예상되는 모든 것을 캐시하세요.

```figure
prompt-cache-hit
```

## 구현하기

### 1단계: Anthropic 프롬프트 캐싱과 명시적 마커

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 마커는 Anthropic에 블록을 5분간 저장하도록 지시합니다. 해당 시간 내 재사용은 적중(hit)하고, 만료 후 재사용은 만료되어 다시 쓰기가 발생합니다.

**응답 사용량 필드:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 1.25배로 과금
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 0.1배로 과금
```

CI에서 두 필드를 모두 확인하세요 — 요청 간에 `cache_read_input_tokens`이 계속 0이면 캐시 키가漂移(drift)하고 있습니다.

### 2단계: 1시간 확장 TTL

장기 실행 배치 작업의 경우, 5분 기본 TTL은 작업 간에 만료됩니다. `ttl`를 설정하세요:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1시간 TTL은 쓰기 프리미엄의 2배 비용 (기본 대비 50% 증가, 25%가 아닌)이 들지만, 접두어를 5회 이상 재사용하는 배치에서는 빠르게 회수됩니다.

### 3단계: OpenAI 자동 캐싱

OpenAI는 설정할 것이 없습니다. 최근 요청과 일치하는 1,024 토큰 이상의 모든 접두어는 자동으로 50% 할인을 받습니다.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 길고 안정적
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 할인된 부분
```

동일한 캐시 친화적 레이아웃 규칙이 적용됩니다. Anthropic의 캐시를 죽이지 않는 두 가지가 OpenAI의 캐시를 죽입니다: `user` 필드 변경(캐시 키 구성 요소로 사용됨) 및 도구 순서 재배열입니다.

### 4단계: Gemini 명시적 컨텍스트 캐싱

Gemini는 캐시를 생성하고 이름을 지정하는 일급 객체로 취급합니다:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini는 캐시가 존재하는 동안 토큰·시간당 저장 비용을 청구하며, 읽기 비용은 정상 입력 속도의 약 25%입니다. 여러 세션에 걸쳐 며칠 동안 동일한 거대한 프롬프트를 재사용할 때 이 형태가 적합합니다.

### 5단계: 프로덕션에서 적중률 측정

쓰기/읽기/미스 횟수를 추적하고 1K 요청당 혼합 비용을 계산하는 시뮬레이션된 3개 제공업체 회계사는 `code/main.py`를 참조하세요. 목표 적중률에 따라 배포를 게이트하세요. 대부분의 프로덕션 Anthropic 설정은 워밍업 후 읽기 비율이 80% 이상이어야 합니다.

## 2026년에도 여전히 출시되는 함정

- **상단에 동적 타임스탬프.** 시스템 프롬프트 상단에 `"Current time: 2026-04-22 15:30:02"`가 있습니다. 모든 요청이 미스됩니다. 타임스탬프를 캐시 분할점 아래로 이동하세요.
- **도구 순서 재배열.** 도구를 안정적인 순서로 직렬화하세요. 배포 간 dict 재배열은 모든 적중을 깨뜨립니다.
- **자유 텍스트 유사 중복.** "You are helpful." vs "You are a helpful assistant." — 한 바이트 차이 = 완전 미스입니다.
- **너무 작은 블록.** Anthropic은 1,024 토큰 하한을 강제합니다(Haiku는 2,048). 더 작은 블록은 조용히 캐시되지 않습니다.
- **맹목적인 비용 대시보드.** "입력 토큰"을 캐시됨과 캐시되지 않음으로 분리하세요. 그렇지 않으면 트래픽 감소가 캐시 성공처럼 보입니다.

## 사용하기

2026년 캐싱 스택:

| 상황 | 선택 |
|-----------|------|
| 안정적인 10k+ 시스템 프롬프트가 있는 에이전트, 많은 턴 | Anthropic `cache_control`, 5분 TTL |
| 접두어를 30분 이상 재사용하는 배치 작업 | Anthropic `ttl: "1h"` |
| GPT-5 서버리스 엔드포인트, 커스텀 인프라 없음 | OpenAI 자동(접두어를 안정적이고 길게 유지하기만 하면 됩니다) |
| 거대한 코드/문서 코퍼스의 며칠간 재사용 | Gemini 명시적 `CachedContent` |
| 제공업체 간 폴백 | 캐시 가능한 접두어 레이아웃을 모든 제공업체에서 동일하게 유지하여 어떤 적중이든 작동하도록 하세요 |

사용자 메시지 계층에 시맨틱 캐싱(11단계 · 11강)을 결합해 보세요: 프롬프트 캐싱은 *토큰이 동일한* 재사용을 처리하고, 시맨틱 캐싱은 *의미가 동일한* 재사용을 처리합니다.

## 출시하기

`outputs/skill-prompt-caching-planner.md`을 저장하세요:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 연습 문제

1. **쉬움.** 5,000토큰 시스템 프롬프트가 포함된 10턴 대화를 Claude에 대해 실행하세요. `cache_control` 없이 실행한 후 을 사용하여 실행하세요. 각 경우의 입력 토큰 청구 금액을 보고하세요.
2. **중간.** 프롬프트 템플릿과 요청 로그가 주어지면 각 제공자(Anthropic 5m, Anthropic 1h, OpenAI 자동, Gemini 명시적)에 대한 예상 적중률과 비용 절감액을 계산하는 테스트 하네스를 작성하세요.
3. **어려움.** 레이아웃 최적화기를 구축하세요: 프롬프트와 `stable=True/False`으로 표시된 필드 목록이 주어지면, 정보를 잃지 않으면서 최대 캐시 친화적 위치에 단일 캐시 분할 지점을 배치하도록 프롬프트를 재작성하세요. 실제 Anthropic 엔드포인트에서 검증하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 프롬프트 캐싱(Prompt caching) | "긴 프롬프트를 저렴하게 만듭니다" | 일치하는 접두사에 대해 제공자 측 KV 캐시를 재사용하는 것; 반복되는 입력 토큰에 대해 50-90% 할인. |
| `cache_control` | "Anthropic 마커" | "여기까지의 모든 내용은 캐시 가능"이라고 선언하는 콘텐츠 블록 속성; `{"type": "ephemeral"}`. |
| 캐시 쓰기(Cache write) | "프리미엄 지불" | 캐시를 채우는 첫 번째 요청; Anthropic에서는 입력 요금의 약 1.25배로 청구되며, OpenAI에서는 무료. |
| 캐시 읽기(Cache read) | "할인" | 접두사와 일치하는 후속 요청; Anthropic은 10%, OpenAI는 50%, Gemini는 약 25%로 청구. |
| TTL | "수명" | 캐시가 따뜻하게 유지되는 시간(초); Anthropic은 기본 5분(1시간으로 연장 가능), OpenAI는 최대 1시간까지 최선 노력(best-effort), Gemini는 사용자 설정. |
| 연장 TTL(Extended TTL) | "1시간 Anthropic 캐시" | `{"type": "ephemeral", "ttl": "1h"}`; 쓰기 프리미엄이 2배이지만 배치 재사용에는 가치가 있음. |
| 접두어 일치(Prefix match) | "캐시가 실패한 이유" | 시작부터 분할 지점까지의 모든 토큰이 바이트 단위로 동일할 때만 캐시가 적중합니다. |
| 컨텍스트 캐싱(Context caching, Gemini) | "명시적인 캐싱" | Google의 이름 지정된 저장소 청구 캐시 객체; 대규모 코퍼pus의 며칠 단위 재사용에 최적. |

## 추가 읽기

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1h TTL, 손익분기점 표.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 자동 접두어 매칭입니다.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 및 저장소 가격입니다.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 지연 시간 수치가 포함된 최초 출시 게시물입니다.
- 11단계 · 05강 (컨텍스트 엔지니어링) — 캐시가 적용될 수 있도록 프롬프트를 분할하는 위치입니다.
- 11단계 · 11강 (캐싱 및 비용) — 사용자 메시지에 시맨틱 캐시를 프롬프트 캐싱과 함께 짝지어 보세요.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 프롬프트 캐싱이 사용자에게 노출하는 KV 캐시 메모리 모델입니다. 캐시된 접두어를 다시 읽는 비용이 재계산하는 비용보다 약 10배 저렴한 이유를 설명합니다.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 프리필은 프롬프트 캐싱이 단축하는 단계입니다. 이 논문은 캐시 적중 시 TTFT가 극적으로 감소하는 반면 TPOT은 영향을 받지 않는 이유를 설명합니다.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 프롬프트 캐싱은 추론 비용 곡선을 바꾸는 레버로서 추론적 디코딩, Flash Attention, MQA/GQA와 함께 위치합니다. 나머지 세 가지를 위해 이 문서를 읽어 보세요.
