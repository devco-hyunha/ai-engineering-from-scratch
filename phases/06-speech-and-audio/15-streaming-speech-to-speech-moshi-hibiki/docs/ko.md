# 스트리밍 음성-대-음성(Streaming Speech-to-Speech) — Moshi, Hibiki, 그리고 전이중 대화(Full-Duplex Dialogue)

> 2024-2026년은 음성 AI의 정의를 다시 쓴 시기였습니다. Moshi는 200ms의 지연 시간으로 듣기와 말하기를 동시에 수행하는 단일 모델을 선보였습니다. Hibiki는 음성-대-음성 번역을 청크(chunk) 단위로 수행합니다. 두 모델 모두 ASR → LLM → TTS 파이프라인을 탈피하여, Mimi 코덱 토큰을 활용한 통합 전이중(full-duplex) 아키텍처를 채택했습니다. 이것이 새로운 레퍼런스 디자인입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제점 (The Problem)

레슨 11과 12를 통해 구축한 모든 음성 에이전트는 약 300-500ms 정도의 근본적인 지연 시간 하한선(latency floor)을 가집니다. VAD가 작동하고, STT가 처리하며, LLM이 추론하고, TTS가 생성하는 과정을 거치기 때문입니다. 각 단계는 고유한 최소 지연 시간을 가집니다. 튜닝과 병렬화를 시도할 수는 있지만, 파이프라인 구조 자체가 성능의 한계를 결정합니다.

Moshi (Kyutai, 2024-2026)는 다른 질문을 던집니다: 만약 파이프라인이 없다면 어떨까요? 하나의 모델이 오디오를 입력받아 직접 오디오를 출력하되, 텍스트를 필수 단계가 아닌 중간적인 "내적 독백(inner monologue)"으로 사용한다면 어떨까요?

그 해답은 **전이중 음성-대-음성(full-duplex speech-to-speech)** 방식입니다. 이론적 지연 시간은 160ms(80ms Mimi 프레임 + 80ms 음향 지연)입니다. 단일 L4 GPU에서의 실제 지연 시간은 200ms입니다. 이는 최고 수준의 파이프라인 방식 음성 에이전트가 달성하는 성능의 절반 수준입니다.

## 개념 (The Concept)

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 아키텍처 (The Moshi architecture)

**입력값 (Inputs).** 12.5 Hz × 8 코드북(codebooks) 구성의 두 가지 Mimi 코덱 스트림:

- 스트림 1: 사용자 오디오 (Mimi로 인코딩되며, 지속적으로 유입됨)
- 스트림 2: Moshi 자체 오디오 (Moshi에 의해 생성됨)

**트랜스포머 (The transformer).** 7B 파라미터 규모의 Temporal Transformer가 두 스트림과 텍스트 "내적 독백(inner monologue)" 스트림을 함께 처리합니다. 매 80ms 단계마다 트랜스포머는 다음을 수행합니다:

1. 최신 사용자 Mimi 토큰(8개 코드북)을 소비합니다.
2. 가장 최근의 Moshi Mimi 토큰(생성된 8개 코드북)을 소비합니다.
3. 다음 Moshi 텍스트 토큰(내적 독백)을 생성합니다.
4. 다음 Moshi Mimi 토큰(소형 Depth Transformer를 통한 8개 코드북)을 생성합니다.

사용자 오디오, Moshi 오디오, Moshi 텍스트의 세 스트림은 모두 병렬로 실행됩니다. Moshi는 말을 하는 동안에도 사용자의 소리를 들을 수 있으며, 사용자가 말을 끊으면 스스로의 발화를 중단할 수 있습니다. 또한, 주요 발화를 끊지 않고도 맞장구("음, 네")를 칠 수 있습니다.

**뎁스 트랜스포머 (The depth transformer).** 하나의 프레임 내에서 8개의 코드북은 병렬로 예측되지 않으며, 코드북 간의 의존성(inter-codebook dependencies)을 가집니다. 소형 2계층 "depth transformer"가 80ms 내에 이들을 순차적으로 예측합니다. 이는 AR 코덱 언어 모델(VALL-E, VibeVoice에서도 사용됨)의 표준적인 인수분해(factorization) 방식입니다.

### 내부 독백(Inner-monologue) 텍스트가 도움이 되는 이유

명시적인 텍스트가 없다면, 모델은 음향 스트림(acoustic stream) 내에서 언어를 암묵적으로 모델링해야 합니다. Moshi의 통찰은 오디오와 함께 텍스트 토큰을 생성하도록 강제하는 것이었습니다. 텍스트 스트림은 본질적으로 Moshi가 말하는 내용의 전사(transcript) 역할을 합니다. 이는 의미적 일관성(semantic coherence)을 향상시키고, 언어 모델 헤드(language model head)를 교체하기 쉽게 만들며, 전사 데이터를 별도의 비용 없이 얻을 수 있게 해줍니다.

### Hibiki: 스트리밍 음성-대-음성 번역 (streaming speech-to-speech translation)

동일한 아키텍처를 사용하며, 번역 쌍(translation pairs)으로 학습되었습니다. 소스 오디오를 입력받아 대상 언어의 오디오를 지속적으로 출력합니다. Hibiki-Zero (2026년 2월)는 단어 수준의 정렬된 학습 데이터(word-level aligned training data)의 필요성을 제거했습니다. 대신 문장 수준의 데이터와 지연 시간 최적화를 위한 GRPO 강화 학습을 사용합니다.

초기에는 4개의 언어 쌍을 지원하며, 약 1,000시간의 데이터로 새로운 언어에 적응할 수 있습니다.

### Kyutai 전체 스택 (2026년) (The broader Kyutai stack)

- **Moshi** — 전이중(full-duplex) 대화 (프랑스어 우선 지원, 영어도 원활히 지원)
- **Hibiki / Hibiki-Zero** — 동시 음성 번역(simultaneous speech translation)
- **Kyutai STT** — 스트리밍 ASR (500ms 또는 2.5s 룩어헤드(look-ahead) 지원)
- **Kyutai Pocket TTS** — CPU에서 실행 가능한 1억 개(100M) 파라미터 규모의 TTS (2026년 1월)
- **Unmute** — 공개 서버에서 위 기술들을 결합한 전체 파이프라인

L40S GPU에서의 처리량(Throughput): 3배 실시간 속도(3× real-time)로 64개의 동시 세션 처리 가능.

### Sesame CSM — 사촌 모델 (the cousin)

Sesame CSM (2025)은 유사한 아이디어를 사용합니다. 즉, Llama-3 백본(backbone)에 Mimi 코덱 헤드(codec head)를 결합한 형태입니다. 하지만 CSM은 전이중(full-duplex) 방식이 아닌 단방향(single-directional) 방식입니다(문맥과 텍스트를 입력받아 음성을 생성). 이는 시장에서 가장 뛰어난 "음성 존재감(voice presence)"을 가진 TTS이지만, Moshi의 전이중(full-duplex) 능력과는 완전히 동일하지는 않습니다.

### 2026년 성능 수치 (2026 performance numbers)

| 모델 (Model) | 지연 시간 (Latency) | 사용 사례 (Use case) | 라이선스 (License) |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 전이중(full-duplex) 영어 / 프랑스어 대화 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 프레임레이트 | 프랑스어 ↔ 영어 스트리밍 번역 | CC-BY 4.0 |
| Hibiki-Zero | 동일 | 5개 언어 쌍, 정렬된 데이터 없음 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 문맥 조건부 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 폐쇄형, OpenAI API | 상업용 (commercial) |
| Gemini 2.5 Live | ~350 ms | 폐쇄형, Google API | 상업용 (commercial) |

```figure
sp-fullduplex
```

## 직접 구현해 보기 (Build It)

### 1단계: 인터페이스 (the interface)

Moshi는 Mimi로 인코딩된 80ms 단위의 오디오 청크를 입력받고, 다시 80ms 단위의 Mimi 인코딩된 오디오 청크를 반환하는 WebSocket 서버를 노출합니다. 양방향으로, 끊임없이 이루어집니다.

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

### 2단계: 전이중 루프 (the full-duplex loop)

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

양방향이 동시에 실행됩니다. Python의 `asyncio` 또는 Rust의 `futures`가 표준 전송 방식입니다.

### 3단계: 학습 목표 (개념적 이해)

매 80ms 프레임 `t`에 대하여:

- 입력(Input): `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 예측(Predict): `moshi_text[t]`, 그 다음 `moshi_mimi[t, codebook_0..7]`

텍스트는 오디오보다 먼저 예측되며(내적 독백, inner monologue), 오디오는 depth transformer 내에서 코드북 순서대로(codebook-sequential) 예측됩니다.

### 4단계: Moshi가 우세한 지점과 그렇지 않은 지점

Moshi가 우세한 지점:

- 저가형 하드웨어에서 250ms 미만의 엔드 투 엔드(end-to-end) 지연 시간 달성.
- 자연스러운 백채널(back-channels) 및 중단(interruptions) 처리.
- 파이프라인 글루 코드(pipeline glue code)가 필요 없음.

Moshi가 우세하지 않은 지점:

- 도구 호출(Tool calling) (해당 용도로 학습되지 않았으며, 별도의 LLM 경로가 필요합니다).
- 긴 추론(Long reasoning) (Moshi는 약 8B 규모의 대화 모델이며, Claude나 GPT-4가 아닙니다).
- 니치(niche)한 주제에 대한 사실적 정확도.
- 대부분의 프로덕션 엔터프라이즈 유스케이스 (2026년에도 여전히 파이프라인 방식이 사용됩니다).

## 활용 방법 (Use It)

| 상황 (Situation) | 선택 (Pick) |
|-----------|------|
| 초저지연 음성 동반자 (Lowest-latency voice companion) | Moshi |
| 실시간 통역 전화 (Live translation call) | Hibiki |
| 음성 데모 / 연구 (Voice demo / research) | Moshi, CSM |
| 도구를 사용하는 엔터프라이즈 에이전트 (Enterprise agent with tools) | Pipeline (Lesson 12), Moshi 아님 |
| 문맥 내 커스텀 음성 TTS (Custom-voice TTS in context) | Sesame CSM |
| 모든 언어 지원 음성-대-음성 (Speech-to-speech, any languages) | GPT-4o Realtime 또는 Gemini 2.5 Live (상용) |

## 주의 사항 (Pitfalls)

- **제한적인 도구 호출 (Limited tool calling).** Moshi는 대화 모델이며, 에이전트 프레임워크가 아닙니다. 도구 사용을 위해서는 파이프라인과 결합하여 사용하세요.
- **특정 음성 조건화 (Specific-voice conditioning).** Moshi는 단일하게 학습된 페르소나를 사용합니다. 음성 복제(Cloning)는 별도의 학습 과정이 필요합니다.
- **언어 지원 범위 (Language coverage).** 프랑스어와 영어 조합은 매우 뛰어나지만, 그 외 언어는 제한적입니다. Hibiki-Zero가 도움이 되지만, 여전히 학습 데이터가 필요합니다.
- **리소스 비용 (Resource cost).** 전체 Moshi 세션은 GPU 슬롯을 점유합니다. 저렴한 공유 테넌트(shared-tenant) 배포 패턴에는 적합하지 않습니다.

## Ship It (실행해 보세요)

`outputs/skill-duplex-pipeline.md`로 저장하세요. 음성 에이전트(voice-agent) 워크로드에 대해 파이프라인(pipeline) 방식과 전이중(full-duplex) 아키텍처 중 하나를 선택하고 그 이유를 설명해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 이 코드는 two-stream + inner-monologue 아키텍처를 상징적으로 시뮬레이션합니다.
2. **중간 (Medium).** HuggingFace에서 Moshi를 가져와 서버를 실행하고, 대화를 한 번 테스트해 보세요. 사용자의 음성이 끝난 시점부터 Moshi의 응답이 시작되는 시점까지의 실제 시간 지연(wall-clock latency)을 측정해 보세요.
3. **어려움 (Hard).** Lesson 12에서 만든 파이프라인 에이전트를 사용하여, 20개의 일치하는 테스트 발화(test utterances)에 대해 Moshi와 P50 지연 시간(latency)을 비교해 보세요. 파이프라인 방식이 아키텍처 측면에서 결과적으로 승리하는 경우를 정리하여 작성해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| Full-duplex (전이중 방식) | 동시에 듣고 말하기 | 동일한 모델에서 두 개의 오디오 스트림이 동시에 활성화됨. |
| Inner monologue (내적 독백) | 모델의 텍스트 스트림 | Moshi는 오디오 출력과 함께 텍스트 토큰을 방출함. |
| Depth transformer (뎁스 트랜스포머) | 코드북 간 예측기 (Inter-codebook predictor) | 하나의 80ms 프레임 내에서 8개의 코드북을 예측하는 작은 트랜스포머. |
| Mimi (미미) | Kyutai의 코덱 | 12.5 Hz × 8개 코드북; 의미론적(semantic)+음향적(acoustic) 정보 포함; Moshi의 기반 기술. |
| Streaming S2S (스트리밍 S2S) | 오디오 → 오디오 실시간 처리 | 파이프라인 단계 없이 청크(chunk) 단위로 진행되는 번역/대화. |
| Back-channeling (백채널링) | "음", "아"와 같은 맞장구 | Moshi는 자신의 발화 순서를 끊지 않고도 작은 확인 응답을 보낼 수 있음. |

## 추가 읽을거리 (Further Reading)

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 논문.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 정렬된 데이터 없이 수행하는 스트리밍 번역.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 사양.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 설치 및 서버.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 폐쇄형 상용 경쟁 모델.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 내부에서 작동하는 STT/TTS 프레임워크.
