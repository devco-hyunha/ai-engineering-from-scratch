# 실시간 오디오 처리

> 배치 파이프라인은 파일을 처리합니다. 실시간 파이프라인은 다음 20ms가 도착하기 전에 다음 20ms를 처리합니다. 모든 대화형 AI, 방송 스튜디오, 전화 봇은 이 지연 시간 예산에 따라 성패가 결정됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램), 6단계 · 04강 (ASR), 6단계 · 07강 (TTS)
**시간:** 약 75분

## 문제점

살아있는 듯한 느낌을 주는 음성 어시스턴트를 원합니다. 인간의 대화 전환 지연 시간은 약 230ms (침묵에서 응답까지)입니다. 500ms 이상은 로봇처럼 느껴지고, 1500ms 이상은 고장 난 것처럼 느껴집니다. 2026년 기준 **듣기 → 이해하기 → 응답하기 → 말하기** 전체 루프의 예산은 다음과 같습니다:

| 단계 | 예산 |
|-------|--------|
| 마이크 → 버퍼 | 20ms |
| VAD | 10ms |
| ASR (스트리밍) | 150ms |
| LLM (첫 토큰) | 100ms |
| TTS (첫 청크) | 100ms |
| 렌더링 → 스피커 | 20ms |
| **합계** | **약 400ms** |

Moshi (Kyutai, 2024)는 200ms의 완전 이중 통신(full-duplex)을 기록했습니다. GPT-4o-realtime (2024)은 약 320ms입니다. 2022년의 캐스케이드 파이프라인은 2500ms로 출시되었습니다. 10배의 개선은 세 가지 기술에서 나왔습니다: (1) 모든 곳에서 스트리밍, (2) 부분 결과를 활용한 비동기 파이프라인, (3) 중단 가능한 생성.

## 개념

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**프레임 / 청크 / 윈도우.** 실시간 오디오는 고정 크기의 블록으로 흐릅니다. 일반적인 선택은 20ms (16kHz에서 320 샘플)입니다. 모든 다운스트림은 이 주기를 따라가야 합니다.

**링 버퍼.** 고정 크기의 원형 버퍼입니다. 생산자 스레드가 새 프레임을 쓰고, 소비자 스레드가 읽습니다. 핫 패스에서의 할당을 방지합니다. 크기는 최대 지연 시간 × 샘플 레이트와 같습니다. 2초 16kHz 링은 32,000 샘플입니다.

**VAD (음성 활동 감지).** 아무도 말하지 않을 때 다운스트림 작업을 차단합니다. Silero VAD 4.0 (2024)은 CPU에서 30ms 프레임당 1ms 미만으로 실행됩니다. `webrtcvad`는 더 오래된 대안입니다.

**스트리밍 ASR.** 오디오가 도착하는 동안 부분적인 전사(transcript)를 생성하는 모델입니다. NeMo의 Parakeet-CTC-0.6B는 스트리밍 모드에서 320 ms 지연으로 2–5% WER를 달성합니다. Whisper-Streaming(Macháček et al., 2023)은 Whisper를 청킹하여 약 2 s 지연으로 준-스트리밍을 수행합니다.

**인터럽션.** 어시스턴트가 말하는 동안 사용자가 말하면, (a) 바지인(barge-in)을 감지하고, (b) TTS를 중지하고, (c) 남은 LLM 출력을 폐기해야 합니다. 이 모든 과정을 100 ms 내에 완료하지 못하면 사용자는 어시스턴트가 듣지 못한다고 인식합니다.

**WebRTC Opus 전송.** 20 ms 프레임, 48 kHz, 적응형 비트레이트 8–128 kbps. 브라우저 및 모바일의 표준입니다. LiveKit, Daily.co, Pion은 2026년 음성 앱 구축을 위한 스택입니다.

**지터 버퍼.** 네트워크 패킷이 순서대로 도착하지 않거나 늦게 도착합니다. 지터 버퍼는 순서를 재배열하고 매끄럽게 처리합니다. 너무 작으면 청각적 공백이 생기고, 너무 크면 지연이 발생합니다. 일반적으로 60–80 ms입니다.

### 공통적인 함정

- **스레드 경합.** Python의 GIL과 무거운 모델은 오디오 스레드를 고갈시킬 수 있습니다. C 콜백 오디오 라이브러리(sounddevice, PortAudio)를 사용하고 Python을 핫 패스(hot path)에서 제외하세요.
- **샘플 레이트 변환 지연.** 파이프라인 내부의 리샘플링은 5–20 ms를 추가합니다. 사전에 리샘플링하거나 제로 지연 리샘플러(PolyPhase, `soxr_hq`)를 사용하세요.
- **TTS 워밍업.** Kokoro처럼 빠른 TTS도 첫 요청 시 100–200 ms의 워밍업이 필요합니다. 모델을 캐싱하고 첫 실제 턴(turn) 전에 더미 실행으로 워밍업하세요.
- **에코 제거.** AEC가 없으면 TTS 출력은 마이크에 재진입하여 봇의 목소리에 대해 ASR을 트리거합니다. WebRTC AEC3는 오픈소스 기본값입니다.

```figure
nyquist-aliasing
```

## 구현하기

### 1단계: 링 버퍼

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

용량은 최대 버퍼링 지연을 결정합니다. 16 kHz에서 32,000 샘플은 2 s입니다.

### 2단계: VAD 게이트

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

프로덕션에서는 Silero VAD로 교체하세요:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 3단계: 스트리밍 ASR

```python
# NeMo를 통한 Parakeet-CTC-0.6B 스트리밍
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 4단계: 인터럽션 핸들러

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 바지인
        # 그 후 스트리밍 ASR에 공급

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

비동기 I/O 및 취소 가능한 TTS 스트리밍에 달려 있습니다. 오디오 트랙에 대한 WebRTC peerconnection.stop()이 표준적인 방법입니다.

## 사용하기

2026년 스택은 다음과 같습니다:

| 계층 | 선택 |
|-------|------|
| 전송 | LiveKit (WebRTC) 또는 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 스트리밍 ASR | Parakeet-CTC-0.6B 또는 Whisper-Streaming |
| LLM 첫 토큰 | Groq, Cerebras, vLLM-streaming |
| 스트리밍 TTS | Kokoro 또는 ElevenLabs Turbo v2.5 |
| 에코 제거 | WebRTC AEC3 |
| 엔드투엔드 네이티브 | OpenAI Realtime API 또는 Moshi |

## 문제점

- **500 ms 버퍼링으로 안전을 보장하세요.** 버퍼가 곧 지연 시간의 하한선입니다. 버퍼를 줄여 보세요.
- **스레드를 고정하지 마세요.** 오디오 콜백이 UI보다 낮은 우선순위의 스레드에서 실행되면 부하 시 글리치가 발생합니다.
- **TTS 청크가 너무 작습니다.** 200 ms 미만 청크는 보코더 아티팩트를 들리게 만듭니다. 320 ms 청크가 최적입니다.
- **지터 버퍼가 없습니다.** 실제 네트워크는 지터가 있으며, 스무딩이 없으면 팝 노이즈가 발생합니다.
- **단일 예외 처리.** 오디오 파이프라인은 크래시 방지되어야 합니다. 하나의 예외가 세션을 종료시킵니다.

## 출시하기

`outputs/skill-realtime-designer.md`로 저장하세요. 각 단계별 지연 시간 예산을 구체적으로 설정한 실시간 오디오 파이프라인을 설계해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 링 버퍼 + 에너지 VAD를 시뮬레이션하며, 가짜 10초 스트림의 단계별 지연 시간을 출력합니다.
2. **중간.** `sounddevice`를 사용하여 마이크를 20 ms 프레임으로 처리하고 각 프레임의 VAD 상태를 출력하는 패스스루 루프를 구축해 보세요.
3. **어려움.** `aiortc`로 전체 이중 통신 에코 테스트를 구축하세요: 브라우저 → WebRTC → Python → WebRTC → 브라우저. 1 kHz 펄스로 유리-유리(glass-to-glass) 지연 시간을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Ring buffer | 순환 큐 | 오디오 프레임을 위한 고정 크기, 락 프리(또는 SPSC 락) FIFO. |
| VAD | 침묵 게이트 | 음성 대 비음성을 표시하는 모델 또는 휴리스틱. |
| Streaming ASR | 실시간 STT | 오디오가 도착하면 부분 텍스트를 방출; 유한한 선조회. |
| Jitter buffer | 네트워크 스무더 | 순서가 뒤섞인 패킷을 재순서하는 큐; 일반적으로 60–80 ms. |
| AEC | 에코 제거 | 스피커에서 마이크까지의 피드백 경로를 차감합니다. |
| 바지인 | 사용자 인터럽트 | 시스템이 TTS 재생 중 사용자의 음성을 감지하며, 재생을 취소해야 합니다. |
| 풀 듀플렉스 | 양방향 동시 통신 | 사용자와 봇이 동시에 대화할 수 있으며, Moshi는 풀 듀플렉스입니다. |

## 추가 읽기

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 청크 기반 준 스트리밍 Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 풀 듀플렉스 200 ms 지연.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 프로덕션 오디오 에이전트 오케스트레이션.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 1 ms 미만 VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 오픈 소스 에코 제거.
