# 스트리밍 음성-음성 변환 — Moshi, Hibiki, 그리고 풀 듀플렉스 대화

> 2024-2026년은 음성 AI를 재정의했습니다. Moshi는 200 ms 지연으로 동시에 듣고 말하는 단일 모델을 출시합니다. Hibiki는 청크 단위로 음성-음성 번역을 수행합니다. 두 모델 모두 ASR → LLM → TTS 파이프라인을 버리고, Mimi 코덱 토큰 기반의 통합 풀 듀플렉스 아키텍처를 채택했습니다. 이것이 새로운 참조 설계입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 6단계 · 13강 (신경망 오디오 코덱), 6단계 · 11강 (실시간 오디오), 7단계 · 05강 (풀 트랜스포머)
**시간:** 약 75분

## 문제점

11강 + 12강으로 구축된 모든 음성 에이전트에는 약 300-500 ms의 근본적인 지연 하한선이 있습니다: VAD가 발화하고, STT가 처리하고, LLM이 추론하고, TTS가 생성합니다. 각 단계에는 자체 최소 지연이 있습니다. 튜닝과 병렬화를 할 수 있지만, 파이프라인 형태가 한계를 설정합니다.

Moshi (Kyutai, 2024-2026)는 다른 질문을 던집니다: 파이프라인이 없다면 어떨까요? 하나의 모델이 오디오를 입력받아 오디오를 직접, 연속적으로 출력하고, 텍스트를 필수 단계가 아닌 중간 "내면의 독백"으로 사용한다면 어떨까요?

답은 **풀 듀플렉스 음성-음성 변환**입니다. 이론적 지연은 160 ms (80 ms Mimi 프레임 + 80 ms 음향 지연)입니다. 단일 L4 GPU에서의 실용적 지연은 200 ms입니다. 이는 최고 수준의 파이프라인 기반 음성 에이전트가 달성하는 지연의 절반입니다.

## 개념

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 아키텍처

**입력.** 두 개의 Mimi 코덱 스트림, 모두 12.5 Hz × 8 코드북:

- 스트림 1: 사용자 오디오 (Mimi 인코딩, 지속적으로 도착)
- 스트림 2: Moshi의 자체 오디오 (Moshi가 생성)

**트랜스포머.** 70억 매개변수 Temporal Transformer가 두 스트림과 텍스트 "내면의 독백" 스트림을 처리합니다. 각 80 ms 단계에서 다음을 수행합니다:

1. 최신 사용자 Mimi 토큰 (8 코드북)을 소비합니다.
2. 가장 최근의 Moshi Mimi 토큰 (8 코드북, 생성된 대로)을 소비합니다.
3. 다음 Moshi 텍스트 토큰 (내면의 독백)을 생성합니다.
4. 다음 Moshi Mimi 토큰 (8 코드북, 작은 Depth Transformer를 통해)을 생성합니다.

세 개의 스트림 — 사용자 오디오, Moshi 오디오, Moshi 텍스트 — 모두 병렬로 실행됩니다. Moshi는 말하는 동안 사용자의 음성을 들을 수 있으며, 사용자가 방해할 때 자기 발화를 중단할 수 있고, 주요 발화를 깨뜨리지 않으면서 백채널("mhm")을 사용할 수 있습니다.

**깊이 트랜스포머.** 프레임 내에서 8개의 코드북은 병렬로 예측되지 않으며, 코드북 간 의존성이 존재합니다. 작은 2층 "깊이 트랜스포머"가 80ms 내에서 순차적으로 이를 예측합니다. 이는 AR 코덱 LM의 표준 분해 방식이며(VALL-E, VibeVoice에서도 사용됨), 이 구조가 적용됩니다.

### 내적 독백 텍스트가 도움이 되는 이유

명시적인 텍스트가 없으면 모델은 음성 스트림 내에서 언어를 암묵적으로 모델링해야 합니다. Moshi의 통찰은 오디오와 함께 텍스트 토큰을 방출하도록 강제하는 것입니다. 텍스트 스트림은 본질적으로 Moshi가 말하는 내용의 전사(transcript)입니다. 이는 의미적 일관성을 개선하고, 언어 모델 헤드를 교체하기 쉽게 만들며, 전사(transcript)를 무료로 제공합니다.

### Hibiki: 스트리밍 음성-음성 번역

동일한 아키텍처가 번역 쌍으로 학습되었습니다. 소스 오디오가 입력되고, 타겟 언어의 오디오가 연속적으로 출력됩니다. Hibiki-Zero (2026년 2월)는 단어 수준 정렬 학습 데이터의 필요성을 없애며, 문장 수준 데이터와 GRPO 강화 학습을 사용하여 지연 시간을 최적화합니다.

초기에는 4개 언어 쌍이 지원되며, 약 1000시간의 데이터로 새로운 언어에 적응할 수 있습니다.

### 더 넓은 Kyutai 스택 (2026)

- **Moshi** — 풀 듀플렉스 대화 (프랑스어 우선, 영어 잘 지원)
- **Hibiki / Hibiki-Zero** — 동시 음성 번역
- **Kyutai STT** — 스트리밍 ASR (500ms 또는 2.5초 선조회)
- **Kyutai Pocket TTS** — 1억 매개변수 TTS가 CPU에서 실행 (2026년 1월)
- **Unmute** — 공개 서버에서 이들을 결합한 전체 파이프라인

L40S GPU에서의 처리량: 3배 실시간으로 64개 동시 세션.

### Sesame CSM — 사촌 격인 모델

Sesame CSM (2025)은 유사한 아이디어를 사용합니다. Llama-3 백본과 Mimi 코덱 헤드를 사용합니다. 그러나 CSM은 단방향(컨텍스트와 텍스트를 받아 음성 생성)이며 풀 듀플렉스가 아닙니다. 시중에서 가장 좋은 "목소리 존재감" TTS이며, Moshi의 풀 듀플렉스 능력과는 완전히 같지 않습니다.

### 2026년 성능 수치

| 모델 | 지연 시간 | 사용 사례 | 라이선스 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 영어 / 프랑스어 풀듀플렉스 대화 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 프레임 레이트 | 프랑스어 ↔ 영어 스트리밍 번역 | CC-BY 4.0 |
| Hibiki-Zero | 동일 | 5개 언어 쌍, 정렬된 데이터 없음 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 컨텍스트 조건부 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 폐쇄형, OpenAI API | 상용 |
| Gemini 2.5 Live | ~350 ms | 폐쇄형, Google API | 상용 |

```figure
sp-fullduplex
```

## 구현하기

### 1단계: 인터페이스

Moshi는 Mimi로 인코딩된 오디오의 80 ms 청크를 받아 Mimi로 인코딩된 오디오의 80 ms 청크를 반환하는 WebSocket 서버를 노출합니다. 양방향 모두, 지속적으로 작동합니다.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 2단계: 풀듀플렉스 루프

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

양방향 모두 동시에 실행됩니다. Python asyncio 또는 Rust futures가 표준 전송 수단입니다.

### 3단계: 학습 목표 (개념적)

모든 80 ms 프레임 `t`에 대해:

- 입력: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 예측: `moshi_text[t]`, 그 다음 `moshi_mimi[t, codebook_0..7]`

텍스트는 오디오보다 먼저 예측됩니다 (내면의 독백); 오디오는 깊이 트랜스포머 내에서 코드북 순차적으로 예측됩니다.

### 4단계: Moshi가 승리하는 지점과 승리하지 못하는 지점

Moshi가 승리하는 지점:

- 저비용 하드웨어에서 250 ms 미만의 엔드투엔드 지연.
- 자연스러운 백채널 및 인터럽트.
- 파이프라인 글루 코드 없음.

Moshi가 승리하지 못하는 지점:

- 도구 호출 (이를 위해 학습되지 않았으므로, 별도의 LLM 경로가 필요합니다).
- 긴 추론 (Moshi는 8B급 대화 모델이며, Claude/GPT-4가 아닙니다).
- 니치 주제에 대한 사실적 정확도.
- 대부분의 생산적 엔터프라이즈 사용 사례 (2026년에도 여전히 파이프라인을 사용합니다).

## 사용하기

| 상황 | 선택 |
|-----------|------|
| 최저 지연 음성 동반자 | Moshi |
| 실시간 번역 통화 | Hibiki |
| 음성 데모 / 연구 | Moshi, CSM |
| 도구 사용 기업 에이전트 | 파이프라인 (12강), Moshi 아님 |
| 컨텍스트 내 맞춤형 음성 TTS | Sesame CSM |
| 음성 간 대화, 모든 언어 | GPT-4o Realtime 또는 Gemini 2.5 Live (상용) |

## 문제점

- **도구 호출이 제한적입니다.** Moshi는 대화 모델이며 에이전트 프레임워크가 아닙니다. 도구 사용을 위해 파이프라인과 결합해 보세요.
- **특정 음성 조건부 처리.** Moshi는 단일 학습된 페르소나를 사용하며, 음성 클로닝은 별도의 학습 실행입니다.
- **언어 커버리지.** 프랑스어 + 영어는 우수하지만, 다른 언어는 제한적입니다. Hibiki-Zero가 도움이 되지만, 여전히 학습 데이터가 필요합니다.
- **리소스 비용.** 전체 Moshi 세션은 GPU 슬롯을 점유하므로, 저렴한 공유 테넌트 배포 패턴이 아닙니다.

## 출시하기

`outputs/skill-duplex-pipeline.md`로 저장하세요. 음성 에이전트 워크로드에 대해 파이프라인과 풀 듀플렉스 아키텍처 중 하나를 선택하고, 그 이유를 설명해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행해 보세요. 이는 두 스트림 + 내적 독백 아키텍처를 상징적으로 시뮬레이션합니다.
2. **중간.** HuggingFace에서 Moshi를 가져와 서버를 실행하고, 한 번의 대화를 테스트해 보세요. 사용자 발화 종료부터 Moshi 응답 시작까지의 실제 시간(latency)을 측정해 보세요.
3. **어려움.** 12강 파이프라인 에이전트를 가져와 20개의 일치하는 테스트 발화에서 Moshi와 P50 지연 시간을 비교해 보세요. 파이프라인 아키텍처가 어떤 상황에서 우세한지 정리해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 풀 듀플렉스 | 동시에 듣고 말하기 | 동일한 모델에서 두 오디오 스트림이 동시에 활성화됩니다. |
| 내적 독백 | 모델의 텍스트 스트림 | Moshi는 오디오 출력과 함께 텍스트 토큰을 방출합니다. |
| 깊이 트랜스포머 | 코드북 간 예측기 | 하나의 80 ms 프레임 내에서 8개 코드북을 예측하는 작은 트랜스포머입니다. |
| Mimi | Kyutai의 코덱 | 12.5 Hz × 8 코드북; 의미론적 + 음향적; Moshi의 기반이 됩니다. |
| 스트리밍 S2S | 오디오 → 오디오 실시간 | 파이프라인 단계 없이 청크 단위로 번역/대화를 수행합니다. |
| 백채널링 | "음" 반응 | Moshi는 자신의 차례를 깨뜨리지 않고 작은 인정 신호를 방출할 수 있습니다. |

## 추가 읽기

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 논문.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 정렬된 데이터 없이 스트리밍 번역을 수행합니다.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 사양입니다.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 설치 및 서버입니다.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 폐쇄형 상용 피어입니다.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 내부적으로 사용되는 STT/TTS 프레임워크입니다.
