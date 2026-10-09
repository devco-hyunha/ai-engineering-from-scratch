# 음성 비서 파이프라인 구축하기 — 6단계 캡스톤

> 01-11강의 모든 내용을 연결합니다. 듣고, 추론하고, 응답하는 음성 비서를 구축해 보세요. 2026년 현재 이는 연구 문제가 아니라 이미 해결된 엔지니어링 문제입니다. 하지만 통합 세부 사항이 출시 여부를 결정합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 04, 05, 06, 07, 11강; 11단계 · 09강 (함수 호출(Function Calling)); 14단계 · 01강 (에이전트 루프(Agent Loop))
**시간:** 약 120분

## 문제점

엔드투엔드(end-to-end) 비서를 구축합니다:

1. 마이크 입력(16 kHz 모노)을 캡처합니다.
2. 사용자 발화의 시작과 끝을 감지합니다.
3. 스트리밍 전사(transcription)를 수행합니다.
4. 도구(타이머, 날씨, 캘린더)를 호출할 수 있는 LLM (대규모 언어 모델)(LLM (Large Language Model))에 전사(transcript)를 전달합니다.
5. LLM 텍스트를 TTS로 스트리밍합니다.
6. 오디오를 사용자에게 재생합니다.
7. 사용자가 응답 중 방해(interrupt)하면 중지합니다.

레이턴시(latency) 목표: 노트북 CPU에서 사용자가 발화를 마친 후 800 ms 이내에 첫 TTS 오디오 바이트가 나오도록 합니다. 품질 목표: 단어 누락 없음, 침묵 중 환각(Hallucination) 자막 없음, 음성 클로닝 유출 없음, 프롬프트 인젝션(Prompt Injection) 성공 없음.

## 개념

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 7가지 구성 요소

1. **오디오 캡처.** 마이크 → 16 kHz 모노 → 20 ms 청크(chunk). 일반적으로 Python의 `sounddevice`를 사용하거나, 프로덕션에서는 AudioUnit/ALSA/WASAPI를 사용합니다.
2. **VAD (11강).** Silero VAD @ 임계값 0.5, 최소 발화 250 ms, 침묵 유지(hang-over) 500 ms. "시작"과 "끝"을 신호합니다.
3. **스트리밍 STT (04-05강).** Whisper-streaming, Parakeet-TDT, 또는 Deepgram Nova-3 (API). 부분 및 최종 전사(transcript)를 생성합니다.
4. **도구 호출을 지원하는 LLM.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. 도구용 JSON 스키마. 토큰을 스트리밍합니다.
5. **스트리밍 TTS (07강).** Kokoro-82M (가장 빠른 오픈 소스) 또는 Cartesia Sonic (상용). LLM 토큰 20개 이후에 TTS를 시작합니다.
6. **재생.** 스피커 출력; 저대역폭 네트워크를 위해 opus 인코딩을 사용합니다.
7. **방해(interruption) 핸들러.** TTS 재생 중 VAD가 발동하면 재생을 중지하고, LLM을 취소하며, STT를 재시작합니다.

### 발생할 세 가지 실패 모드

1. **첫 단어 잘림.** VAD가 한 박자 늦게 시작합니다. 사용자의 "hey"가 누락됩니다. 시작 임계값을 0.5가 아닌 0.3으로 설정하세요.
2. **응답 중 인터럽트 혼란.** 사용자가 인터럽트한 후에도 LLM이 계속 생성하며, 어시스턴트가 사용자의 말을 덮습니다. VAD → LLM 취소로 연결하세요.
3. **침묵 환각.** Whisper는 웜업 중 침묵 프레임에서 "Thanks for watching"을 출력합니다. 항상 VAD 게이트를 적용하세요.

### 2026년 프로덕션 참조 스택

| 스택 | 지연 시간 | 라이선스 | 비고 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 상용 API | 2026년 산업 표준 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 대부분 오픈 | DIY 친화적 |
| Moshi (풀 듀플렉스) | 200-300 ms | CC-BY 4.0 | 단일 모델; 아키텍처가 다름, 15강 |
| Vapi / Retell (관리형) | 300-500 ms | 상용 | 가장 빠르게 출시 가능; 커스터마이징 제한 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 오프라인 | 오픈 | 프라이버시 / 엣지 |

```figure
v4-voice-latency
```

## 구현하기

### 1단계: 청킹을 포함한 마이크 캡처 (의사 코드)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 2단계: VAD 게이트가 적용된 턴 캡처

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 3단계: 스트리밍 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 4단계: LLM 루프 내의 함수 호출(Function Calling)

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 5단계: 인터럽트 처리

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 사용하기

`code/main.py`를 참고하여 모든 7개 구성 요소를 스텁 모델로 연결한 실행 가능한 시뮬레이션을 확인하세요. 하드웨어 없이도 파이프라인 형태를 볼 수 있습니다. 실제 구현에서는 스텁을 다음으로 교체하세요:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 또는 `openai-whisper`
- `openai` (`gpt-4o`) 또는 `anthropic`
- `kokoro` 또는 `cartesia`
- I/O용 `sounddevice`

## 문제점

- **PII를 영구적으로 로깅.** 전체 턴 오디오는 대부분의 관할권에서 PII입니다. 30일 보관, 저장 시 암호화하세요.
- **바지인(barge-in) 없음.** 사용자는 인터럽트할 것입니다. 어시스턴트는 말하기를 멈춰야 합니다.
- **블로킹 TTS.** 동기식 TTS는 이벤트 루프를 블로킹합니다. 비동기나 별도 스레드를 사용하세요.
- **함수 호출(Function Calling) 오류 처리 없음.** 도구는 실패합니다. LLM은 오류를 받아야 하며 한 번 재시도한 후, 우아한 저하(Graceful Degradation)로 처리해야 합니다.
- **과도한 환각 필터.** 과도하게 필터링하면 어시스턴트가 "그건 도와줄 수 없어요."를 반복합니다. 필터링이 부족하면 임의의 응답을 생성합니다. 홀드아웃 세트에서 보정해 보세요.
- **웨이크워드 옵션 없음.** 상시 청취는 개인정보 보호 위험입니다. 웨이크워드 게이트(Porcupine 또는 openWakeWord)를 추가해 보세요.

## 출시하기

`outputs/skill-voice-assistant-architect.md`로 저장하세요. 예산 + 규모 + 언어 + 컴플라이언스 제약 조건을 고려하여 전체 스택 사양을 작성하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 스텁 모듈로 한 전체 턴을 엔드투엔드로 시뮬레이션하고 단계별 지연 시간을 출력합니다.
2. **중간.** STT 스텁을 사전 녹음된 `.wav`에 대한 실제 Whisper 모델로 교체하세요. WER과 엔드투엔드 지연 시간을 측정하세요.
3. **어려움.** 도구 호출을 추가하세요: `get_weather` (임의의 API)와 `set_timer`을 구현하세요. LLM을 도구로 라우팅하고, 사용자가 "5분 타이머 설정"이라고 말하면 올바른 함수가 실행되고 음성 응답이 이를 확인하는지 검증하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 턴 | 사용자 + 어시스턴트 왕복 | VAD로 경계 지어진 사용자 발화 + LLM-TTS 응답 하나. |
| 바지인 | 인터럽션 | 어시스턴트가 말하는 동안 사용자가 발화하면 어시스턴트가 멈춤. |
| 웨이크워드 | "Hey assistant" | 짧은 키워드 감지기; Porcupine, Snowboy, openWakeWord. |
| 엔드포인팅 | 턴 종료 | 사용자가 발화를 마쳤다고 판단하는 VAD + 최소 침묵 결정. |
| 프리롤 | 발화 전 버퍼 | VAD가 트리거되기 전 200-400 ms의 오디오를 유지하여 첫 단어 클리핑을 방지. |
| 도구 호출 | 함수 호출 | LLM이 JSON을 방출; 런타임이 디스패치; 결과가 루프 내로 피드백. |

## 추가 읽기

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 프로덕션급 레퍼런스.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 친화적 프레임워크.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 관리형 음성 네이티브 경로.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 풀듀플렉스 레퍼런스 (15강).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 웨이크워드 게이트.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 함수 호출.
