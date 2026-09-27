# 음성 비서 파이프라인 구축하기 — Phase 6 캡스톤 (The Phase 6 Capstone)

> 01~11강의 모든 내용을 하나로 통합합니다. 듣고, 추론하고, 대답하는 음성 비서를 구축해 보세요. 2026년에는 이것이 연구 과제가 아닌 해결된 엔지니어링 문제가 되겠지만, 통합(integration)의 디테일이 제품의 출시 여부를 결정할 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## 문제 (The Problem)

엔드 투 엔드(end-to-end) 어시스턴트를 구축하세요:

1. 마이크 입력 캡처 (16 kHz mono).
2. 사용자 음성의 시작/종료 감지.
3. 스트리밍 전사(Transcription).
4. 전사된 텍스트를 도구 호출(timer, weather, calendar)이 가능한 LLM에 전달.
5. LLM 텍스트를 TTS로 스트리밍.
6. 사용자에게 오디오 재생.
7. 사용자가 응답 도중 말을 하면 중단.

지연 시간(Latency) 목표: 노트북 CPU 환경에서 사용자의 발화가 끝난 후 800ms 이내에 첫 번째 TTS 오디오 바이트가 출력되어야 합니다. 품질(Quality) 목표: 단어 누락 없음, 무음 구간에서의 환각 자막(hallucinated subtitles) 없음, 음성 복제 유출(voice cloning leakage) 없음, 프롬프트 주입(prompt injection) 성공 사례 없음.

## 개념 (The Concept)

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 7가지 구성 요소 (The seven components)

1. **오디오 캡처 (Audio capture).** 마이크 → 16 kHz 모노 → 20 ms 청크(chunks). 일반적으로 Python의 `sounddevice`를 사용하거나, 프로덕션 환경에서는 네이티브 AudioUnit/ALSA/WASAPI를 사용합니다.
2. **VAD (Lesson 11).** 임계값(threshold) 0.5, 최소 음성 길이 250 ms, 무음 유지 시간(silence hang-over) 500 ms를 적용한 Silero VAD. "시작(start)"과 "종료(end)" 신호를 생성합니다.
3. **스트리밍 STT (Lesson 4-5).** Whisper-streaming, Parakeet-TDT 또는 Deepgram Nova-3 (API). 부분(partial) 및 최종(final) 전사(transcript)를 제공합니다.
4. **도구 호출 기능이 있는 LLM (LLM with tool calling).** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. 도구 사용을 위한 JSON 스키마를 활용하며, 토큰을 스트리밍합니다.
5. **스트리밍 TTS (Lesson 7).** Kokoro-82M (가장 빠른 오픈 소스) 또는 Cartesia Sonic (상용). LLM 토큰이 20개 생성된 후 TTS를 시작합니다.
6. **재생 (Playback).** 스피커 출력; 저대역폭 네트워크를 위해 opus-encode를 사용합니다.
7. **중단 처리기 (Interruption handler).** TTS 재생 중 VAD 신호가 발생하면, 재생을 중단하고 LLM을 취소한 뒤 STT를 재시작합니다.

### 여러분이 마주하게 될 세 가지 실패 모드 (The three failure modes you will hit)

1. **첫 단어 잘림 (First-word clip).** VAD가 한 박자 늦게 시작됩니다. 사용자의 "헤이(hey)"가 누락됩니다. 시작 임계값(threshold)을 0.5가 아닌 0.3으로 설정해 보세요.
2. **응답 중단 혼선 (Mid-response interrupt confusion).** 사용자가 말을 끊었는데도 LLM이 계속해서 생성하여, 어시스턴트가 사용자의 말을 가로채며 말하게 됩니다. VAD를 `cancel-LLM`에 연결(wire)하세요.
3. **침묵 환각 (Silence hallucination).** Whisper가 침묵 상태인 워밍업 프레임에서 "시청해 주셔서 감사합니다(Thanks for watching)"와 같은 문구를 출력합니다. 항상 VAD 게이트(VAD-gate)를 사용하세요.

### 2026년 프로덕션 참조 스택 (2026 production reference stacks)

| 스택 (Stack) | 지연 시간 (Latency) | 라이선스 (License) | 비고 (Notes) |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 상용 API (commercial API) | 2026년 업계 표준 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 대부분 오픈 소스 (mostly open) | DIY 친화적 |
| Moshi (전이중 방식, full-duplex) | 200-300 ms | CC-BY 4.0 | 단일 모델; 다른 아키텍처, 15과 참조 |
| Vapi / Retell (관리형, managed) | 300-500 ms | 상용 (commercial) | 가장 빠른 출시 가능; 커스텀 제한적 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 오프라인 (offline) | 오픈 소스 (open) | 개인정보 보호 / 엣지(edge) 컴퓨팅 |

```figure
v4-voice-latency
```

## 직접 구현해 보기 (Build It)

### 1단계: 청킹(chunking)을 이용한 마이크 캡처 (의사 코드)

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

### 2단계: VAD 게이트 기반 발화 캡처 (VAD-gated turn capture)

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    # buf: 버퍼, pre: 프리롤(pre-roll) 저장용 데크, triggered: 발화 감지 여부
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                # 발화가 시작되면 이전의 프리롤 데이터를 버퍼에 포함
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            # 설정된 침묵 시간(silence_ms)을 초과하면 발화 종료로 간주
            if silent >= silence_ms:
                return b"".join(buf)
```

### 3단계: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 4단계: LLM 루프 내부의 도구 호출(tool calling)

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        # 도구 호출이 발생하면 해당 함수를 실행합니다.
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        # 텍스트가 생성되면 TTS로 스트리밍합니다.
        await tts.stream(chunk.text)
```

### 5단계: 중단 처리 (interruption handling)

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

## 사용 방법 (Use It)

하드웨어 없이도 파이프라인의 형태를 확인할 수 있도록, 7개의 모든 구성 요소를 스텁(stub) 모델로 연결한 실행 가능한 시뮬레이션이 `code/main.py`에 포함되어 있습니다. 실제 구현을 위해서는 스텁을 다음 라이브러리들로 교체해 보세요:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 또는 `openai-whisper`
- `openai` (`gpt-4o`) 또는 `anthropic`
- `kokoro` 또는 `cartesia`
- I/O를 위한 `sounddevice`

## 주의 사항 (Pitfalls)

- **개인정보(PII)의 무기한 로깅.** 전체 오디오 데이터는 대부분의 관할권에서 개인정보(PII)에 해당합니다. 30일 보관 정책을 준수하고, 저장 시 암호화(encrypted at rest)를 적용하세요.
- **끼어들기(Barge-in) 미지원.** 사용자는 대화 도중 말을 끊을 수 있습니다. 어시스턴트는 사용자가 말을 시작하면 즉시 말을 멈춰야 합니다.
- **블로킹(Blocking) 방식의 TTS.** 동기식(Synchronous) TTS는 이벤트 루프를 차단합니다. 비동기(async) 방식이나 별도의 스레드를 사용하세요.
- **도구 호출(Tool-call) 에러 처리 부재.** 도구는 실패할 수 있습니다. LLM은 에러 메시지를 전달받아 한 번 더 재시도해야 하며, 실패 시 우아하게 성능을 저하시키며(gracefully degrade) 대응해야 합니다.
- **지나치게 엄격한 환각(Hallucination) 필터.** 필터링이 너무 강하면 어시스턴트가 "도와드릴 수 없습니다"라는 말만 반복하게 됩니다. 반대로 너무 약하면 아무 말이나 내뱉게 됩니다. 별도의 검증 데이터셋(held-out set)을 통해 보정(Calibrate)하세요.
- **웨이크 워드(Wake-word) 옵션 부재.** 항상 듣고 있는 상태(Always-listening)는 프라이버시 책임 문제가 발생할 수 있습니다. 웨이크 워드 게이트(Porcupine 또는 openWakeWord)를 추가하세요.

## Ship It (실행하기)

`outputs/skill-voice-assistant-architect.md`로 저장하세요. 예산 + 규모 + 언어 + 컴플라이언스 제약 조건을 고려하여 풀스택 사양(full stack spec)을 생성하세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 스텁(stub) 모듈을 사용하여 전체 과정을 엔드투엔드(end-to-end)로 시뮬레이션하고 각 단계별 지연 시간(latency)을 출력합니다.
2. **중간 (Medium).** STT 스텁을 미리 녹음된 `.wav` 파일에 대한 실제 Whisper 모델로 교체해 보세요. WER(Word Error Rate)과 엔드투엔드 지연 시간을 측정합니다.
3. **어려움 (Hard).** 도구 호출(tool calling) 기능을 추가해 보세요: `get_weather`(임의의 API)와 `set_timer`를 구현합니다. LLM이 도구를 거쳐 경로를 지정하도록 설정하고, 사용자가 "5분 타이머 설정해 줘"라고 말했을 때 올바른 함수가 실행되며 음성 답변이 이를 확인해 주는지 검증합니다.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| Turn (턴) | 사용자 + 어시스턴트의 왕복 과정 | VAD로 경계가 지정된 한 번의 사용자 음성 + 한 번의 LLM-TTS 응답. |
| Barge-in (바지인) | 중단 (Interruption) | 어시스턴트가 말하는 동안 사용자가 말을 하여 어시스턴트가 멈추는 것. |
| Wake word (웨이크 워드) | "헤이 어시스턴트" | 짧은 키워드 탐지기; Porcupine, Snowboy, openWakeWord 등. |
| End-pointing (엔드포인팅) | 턴 종료 (Turn ending) | 사용자가 말을 마쳤는지 판단하는 VAD + 최소 침묵(min-silence) 결정. |
| Pre-roll (프리롤) | 발화 전 버퍼 (Pre-speech buffer) | 첫 단어가 잘리는 것을 방지하기 위해 VAD가 작동하기 전 200-400ms의 오디오를 유지하는 것. |
| Tool call (툴 콜) | 함수 호출 (Function invocation) | LLM이 JSON을 생성하고, 런타임이 이를 배정하며, 결과가 루프 내로 다시 피드백되는 과정. |

## 추가 학습 자료 (Further Reading)

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 프로덕션급 레퍼런스.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 친화적인 프레임워크.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 관리형 보이스 네이티브(voice-native) 경로.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 전이중(full-duplex) 레퍼런스 (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 웨이크 워드(wake-word) 게이팅.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 함수 호출(function calling).
