# 실시간 오디오 처리 (Real-Time Audio Processing)

> 배치 파이프라인(Batch pipelines)은 파일을 처리합니다. 실시간 파이프라인(Real-time pipelines)은 다음 20밀리초(ms)가 도착하기 전에 현재의 20밀리초를 처리해야 합니다. 모든 대화형 AI, 방송 스튜디오, 전화 봇의 성패는 바로 이 지연 시간 예산(latency budget)에 달려 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 문제 (The Problem)

살아있는 듯한 느낌을 주는 음성 비서를 원한다고 가정해 봅시다. 인간의 대화 차례 주고받기(turn-taking) 지연 시간은 약 230ms(침묵에서 응답까지)입니다. 500ms를 넘어가면 로봇처럼 느껴지고, 1500ms를 넘어가면 시스템이 고장 난 것처럼 느껴집니다. 2026년 기준, 전체 **듣기 → 이해하기 → 응답하기 → 말하기** 루프를 위한 예산(Budget)은 다음과 같습니다:

| 단계 | 예산 (Budget) |
|-------|--------|
| 마이크 → 버퍼 (Mic → buffer) | 20 ms |
| VAD | 10 ms |
| ASR (스트리밍) | 150 ms |
| LLM (첫 번째 토큰) | 100 ms |
| TTS (첫 번째 청크) | 100 ms |
| 렌더링 → 스피커 (Render → speaker) | 20 ms |
| **합계** | **~400 ms** |

Moshi (Kyutai, 2024)는 200ms의 전이중(full-duplex) 통신을 기록했습니다. GPT-4o-realtime (2024)은 약 320ms를 기록합니다. 2022년의 계층형 파이프라인(Cascaded pipelines)은 2500ms로 출시되었습니다. 이러한 10배의 성능 향상은 세 가지 기술을 통해 이루어졌습니다: (1) 모든 과정의 스트리밍화(streaming everywhere), (2) 부분 결과물을 활용한 비동기 파이프라이닝(asynchronous pipelining with partial results), (3) 중단 가능한 생성(interruptible generation).

## 개념 (The Concept)

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**프레임 / 청크 / 윈도우 (Frame / chunk / window).** 실시간 오디오는 고정된 크기의 블록 단위로 흐릅니다. 일반적으로 20ms(16kHz 기준 320개 샘플)를 선택합니다. 이후의 모든 하위 프로세스는 이 리듬에 맞춰 동작해야 합니다.

**링 버퍼 (Ring buffer).** 고정 크기의 순환 버퍼입니다. 프로듀서(Producer) 스레드는 새로운 프레임을 쓰고, 컨슈머(Consumer) 스레드는 이를 읽습니다. 이는 핫 패스(hot path)에서의 메모리 할당을 방지합니다. 크기는 `최대 지연 시간 × 샘플 레이트`로 설정합니다. 예를 들어, 16kHz에서 2초 분량의 링 버퍼는 32,000개 샘플이 됩니다.

**VAD (Voice Activity Detection, 음성 활동 감지).** 아무도 말하지 않을 때 하위 프로세스가 작동하지 않도록 게이트 역할을 합니다. Silero VAD 4.0(2024)은 CPU에서 30ms 프레임당 1ms 미만으로 실행됩니다. `webrtcvad`는 기존의 대안입니다.

**스트리밍 ASR (Streaming ASR).** 오디오가 도착함에 따라 부분적인 전사(transcript)를 생성하는 모델입니다. 스트리밍 모드의 Parakeet-CTC-0.6B(NeMo, 2024)는 320ms의 지연 시간에서 2~5%의 WER(단어 오류율)을 기록합니다. Whisper-Streaming(Macháček et al., 2023)은 Whisper를 청크 단위로 나누어 약 2초의 지연 시간으로 준실시간(near-streaming) 성능을 구현합니다.

**끼어들기 (Interruption).** 어시스턴트가 말하는 동안 사용자가 말을 하면, (a) 끼어들기(barge-in)를 감지하고, (b) TTS를 중단하며, (c) 남은 LLM 출력을 폐기해야 합니다. 이 모든 과정은 100ms 이내에 이루어져야 하며, 그렇지 않으면 사용자는 어시스턴트가 귀가 먹었다고 느낄 수 있습니다.

**WebRTC Opus 전송 (WebRTC Opus transport).** 20ms 프레임, 48kHz, 8~128kbps의 적응형 비트레이트를 사용합니다. 브라우저와 모바일의 표준입니다. LiveKit, Daily.co, Pion은 2026년 음성 앱 구축을 위한 핵심 스택입니다.

**지터 버퍼 (Jitter buffer).** 네트워크 패킷은 순서가 바뀌거나 늦게 도착할 수 있습니다. 지터 버퍼는 이를 재정렬하고 부드럽게 만듭니다. 버퍼가 너무 작으면 오디오 끊김이 발생하고, 너무 크면 지연 시간이 늘어납니다. 일반적으로 60~80ms가 적당합니다.

### 일반적인 주의 사항 (Common gotchas)

- **스레드 경합 (Thread contention).** Python의 GIL과 무거운 모델은 오디오 스레드를 고갈(starve)시킬 수 있습니다. C-콜백 오디오 라이브러리(`sounddevice`, `PortAudio`)를 사용하고, Python이 핫 패스(hot path)에 포함되지 않도록 유지하세요.
- **샘플 레이트 변환 지연 (Sample-rate conversion latency).** 파이프라인 내부에서 리샘플링을 수행하면 5~20ms의 지연이 추가됩니다. 사전에 리샘플링을 완료하거나, 제로 레이턴시 리샘플러(`PolyPhase`, `soxr_hq`)를 사용하세요.
- **TTS 프리밍 (TTS priming).** Kokoro와 같이 빠른 TTS라도 첫 요청 시 100~200ms의 워밍업 시간이 필요합니다. 모델을 캐싱하고, 첫 번째 실제 턴이 시작되기 전에 더미 실행(dummy run)으로 워밍업을 해보세요.
- **에코 캔슬레이션 (Echo cancellation).** AEC(Acoustic Echo Cancellation)가 없으면 TTS 출력이 마이크로 다시 유입되어 봇 자신의 목소리에 대해 ASR이 트리거됩니다. WebRTC AEC3가 오픈 소스 표준으로 사용됩니다.

```figure
nyquist-aliasing
```

## 직접 구현해 보기 (Build It)

### 1단계: 링 버퍼 (ring buffer)

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

용량(Capacity)은 최대 버퍼링 지연 시간(buffering latency)을 결정합니다. 16 kHz에서 32,000개 샘플은 2초에 해당합니다.

### 2단계: VAD 게이트 (VAD gate)

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

프로덕션 환경에서는 Silero VAD로 교체하세요:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 3단계: 스트리밍 ASR (Streaming ASR)

```python
# NeMo를 통한 Parakeet-CTC-0.6B 스트리밍
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 4단계: 중단 핸들러 (Interruption Handler)

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in (끼어들기)
        # 그 다음 스트리밍 ASR로 전달

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

비동기 I/O(async I/O)와 취소 가능한 TTS 스트리밍에 의존합니다. 오디오 트랙에 대해 `WebRTC peerconnection.stop()`을 호출하는 것이 표준적인 방법입니다.

## 사용하기 (Use It)

2026년 스택:

| 계층 (Layer) | 선택 (Pick) |
|-------|------|
| 전송 (Transport) | LiveKit (WebRTC) 또는 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 스트리밍 ASR | Parakeet-CTC-0.6B 또는 Whisper-Streaming |
| LLM 첫 번째 토큰 (LLM first-token) | Groq, Cerebras, vLLM-streaming |
| 스트리밍 TTS | Kokoro 또는 ElevenLabs Turbo v2.5 |
| 에코 제거 (Echo cancel) | WebRTC AEC3 |
| 엔드 투 엔드 네이티브 (End-to-end native) | OpenAI Realtime API 또는 Moshi |

## 주의 사항 (Pitfalls)

- **안전하게 하려고 500ms 버퍼링을 하는 것.** 버퍼는 곧 지연 시간(latency)의 하한선입니다. 버퍼를 줄이세요.
- **스레드 고정(Pinning threads)을 하지 않는 것.** UI 스레드보다 우선순위가 낮은 스레드에서 오디오 콜백을 실행하면 부하가 걸릴 때 글리치(glitches)가 발생합니다.
- **TTS 청크(chunks)가 너무 작은 것.** 200ms 미만의 청크는 보코더(vocoder) 아티팩트를 유발할 수 있습니다. 320ms 청크가 가장 적절합니다.
- **지터 버퍼(Jitter buffer)가 없는 것.** 실제 네트워크는 지터(jitter)가 발생합니다. 평활화(smoothing) 작업이 없으면 팝 노이즈(pops)가 발생합니다.
- **단발성 에러 처리(Single-shot error handling).** 오디오 파이프라인은 충돌로부터 안전해야 합니다. 예외 하나가 세션 전체를 종료시킬 수 있습니다.

## Ship It (실행해 보세요)

`outputs/skill-realtime-designer.md`로 저장하세요. 각 단계별로 구체적인 지연 시간 예산(latency budgets)을 설정하여 실시간 오디오 파이프라인을 설계해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 링 버퍼(ring buffer)와 에너지 기반 VAD를 시뮬레이션하며, 가상의 10초 스트림에 대한 각 단계별 지연 시간(latency)을 출력합니다.
2. **중간 (Medium).** `sounddevice`를 사용하여 마이크 입력을 20ms 프레임 단위로 처리하고, 각 프레임마다 VAD 상태를 출력하는 패스스루(passthrough) 루프를 구축해 보세요.
3. **어려움 (Hard).** `aiortc`를 사용하여 전체 이중(full duplex) 에코 테스트를 구축해 보세요: 브라우저 → WebRTC → Python → WebRTC → 브라우저. 1 kHz 펄스를 사용하여 글래스 투 글래스(glass-to-glass) 지연 시간을 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 (Term) | 통용되는 표현 (What people say) | 실제 의미 (What it actually means) |
|------|-----------------|-----------------------|
| Ring buffer | 원형 큐 (The circular queue) | 오디오 프레임을 위한 고정 크기, lock-free(또는 SPSC-locked) FIFO 구조. |
| VAD | 무음 게이트 (Silence gate) | 음성 대 비음성을 구분하는 모델 또는 휴리스틱. |
| Streaming ASR | 실시간 STT (Real-time STT) | 오디오가 도착함에 따라 부분 텍스트를 출력하며, 제한된 lookahead를 가짐. |
| Jitter buffer | 네트워크 평활화 장치 (Network smoother) | 순서가 잘못된 패킷을 재정렬하는 큐; 일반적으로 60–80 ms. |
| AEC | 에코 제거 (Echo cancellation) | 스피커에서 마이크로 이어지는 피드백 경로를 차감함. |
| Barge-in | 사용자 인터럽트 (User interrupt) | TTS 도중 사용자의 음성을 감지하여 재생을 취소해야 하는 상황. |
| Full duplex | 양방향 동시 통신 (Simultaneous both ways) | 사용자와 봇이 동시에 말할 수 있음; Moshi는 full duplex 방식임. |

## 추가 읽을거리 (Further Reading)

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 청크(chunk) 단위의 준실시간(near-streaming) Whisper 구현.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200ms 지연 시간을 가진 전이중(full-duplex) 모델.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 프로덕션 환경의 오디오 에이전트 오케스트레이션.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 1ms 미만의 VAD, Apache 2.0 라이선스.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 오픈 소스 기반의 에코 캔슬레이션(echo cancellation).
