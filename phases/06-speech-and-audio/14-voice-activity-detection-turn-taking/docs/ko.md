# 음성 활동 감지 및 턴 테이킹 — Silero, Cobra, 그리고 Flush 트릭

> 모든 음성 에이전트(Agent)는 두 가지 결정에 따라 성패가 갈립니다: 사용자가 지금 말하고 있는지, 그리고 말을 끝냈는지입니다. VAD는 첫 번째 질문에 답합니다. 턴 감지(VAD + 침묵 잔존 + 의미적 종단점 모델)는 두 번째 질문에 답합니다. 둘 중 하나라도 잘못하면 어시스턴트가 사용자의 말을 끊거나, 끝없이 말하게 됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 11강 (실시간 오디오), 6단계 · 12강 (음성 어시스턴트)
**시간:** 약 45분

## 문제점

음성 에이전트(Agent)는 모든 20 ms 청크에서 세 가지 서로 다른 결정을 내립니다:

1. **이 프레임이 음성인가요?** — VAD. 프레임별 이진 분류.
2. **사용자가 새로운 발화를 시작했나요?** — 시작 감지(onset detection).
3. **사용자가 말을 끝냈나요?** — 종단점 감지(turn-end).

단순한 접근법(에너지 임계값)은 소음(교통 소음, 키보드 소리, 군중의 소음)이 있는 환경에서는 실패합니다. 2026년의 해답은 Silero VAD(오픈 소스, 딥러닝 기반) + 턴 감지 모델(의미적 종단점 처리) + VAD로 보정된 침묵 잔존 시간입니다.

## 개념

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 3단계 VAD 캐스케이드

**1단계: 에너지 게이트.** 가장 저렴합니다. RMS를 -40 dBFS 임계값으로 설정합니다. 명백한 침묵은 필터링하지만, 임계값 이상의 모든 소음에 반응합니다.

**2단계: Silero VAD** (2020-2026, MIT 라이선스). 100만 매개변수(Parameter). 6000개 이상의 언어로 학습되었습니다. 단일 CPU 스레드에서 30 ms 청크당 약 1 ms가 소요됩니다. FPR 5%에서 TPR 87.7%. 오픈 소스 기본값입니다.

**3단계: 의미적 턴 감지 모델.** LiveKit의 턴 감지 모델(2024-2026) 또는 자체 소형 분류기. '문장 중의 휴식'과 '말을 끝냄'을 구분합니다. 침묵뿐만 아니라 언어적 컨텍스트(억양 + 최근 단어)를 사용합니다.

### 핵심 매개변수(Parameter) 및 기본값

- **임계값.** Silero는 확률을 출력합니다. 음성으로 분류하는 기준은 &gt; 0.5 (기본값) 또는 &gt; 0.3 (민감)입니다. 임계값이 낮으면 첫 단어 잘림이 줄어들지만, 오탐(false positive)이 증가합니다.
- **최소 음성 지속 시간.** 250 ms 미만인 음성은 거부합니다. 보통 기침이나 의자 소음입니다.
- **침묵 잔류 (종료 판정).** VAD가 0으로 돌아간 후, 턴 종료로 선언하기 전에 500-800 ms를 대기하세요. 너무 짧으면 → 사용자를 방해합니다. 너무 길면 → 둔감하게 느껴집니다.
- **프리롤 버퍼.** VAD가 발동하기 전의 오디오 300-500 ms를 유지하세요. "hey"가 잘리는 것을 방지합니다.

### 플러시 트릭 (Kyutai 2025)

스트리밍 STT 모델은 선행 지연(look-ahead delay)을 가집니다 (Kyutai STT-1B는 500 ms, STT-2.6B는 2.5 s). 일반적으로 발화 종료 후 전사(transcript)를 위해 그 시간만큼 기다려야 합니다. 플러시 트릭: VAD가 발화 종료를 발동하면, **STT에 플러시 신호를 보내** 즉시 출력을 강제합니다. STT는 실시간의 약 4배 속도로 처리하므로, 500 ms 버퍼는 약 125 ms 안에 완료됩니다.

엔드투엔드: 125 ms VAD + 플러시 STT = 대화형 지연(latency).

### 2026 VAD 비교

| VAD | TPR @ 5% FPR | 지연 | 라이선스 |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | 상용 |
| pyannote segmentation | 95% | ~10 ms | MIT 유사 |

Silero가 올바른 기본값입니다. Cobra는 컴플라이언스 / 정확도 업그레이드입니다. 에너지 전용 VAD는 2026년 프로덕션 환경에서 쓰일 곳이 없습니다.

```figure
sp-vad-cascade
```

## 구현하기

### 1단계: 에너지 게이트

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 2단계: Python에서 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 3단계: 턴 종료 상태 머신

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 4단계: 플러시 트릭 골격

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI)는 이 방식이 작동하려면 플러시를 지원해야 합니다. Whisper 스트리밍은 이를 지원하지 않습니다 — 블록 기반이며 항상 청크를 기다립니다.

## 사용하기

| 상황 | VAD 선택 |
|-----------|-----------|
| 개방형, 빠름, 범용 | Silero VAD |
| 상용 콜센터 | Cobra VAD |
| 온디바이스 (폰) | Silero VAD ONNX |
| 연구 / 화자 분리(diarization) | pyannote segmentation |
| 의존성 없는 폴백 | WebRTC VAD (레거시) |
| 턴 종료 품질 필요 | Silero + LiveKit turn-detector 계층화 |

경험칙: 다른 선택지가 정말로 없는 경우가 아니라면, 에너지 전용 VAD는 절대 출시하지 마세요.

## 문제점

- **고정 임계값.** 조용한 환경에서는 작동하지만, 시끄러운 환경에서는 실패합니다. 기기 자체에서 보정하거나 Silero로 전환해 보세요.
- **너무 짧은 침묵 유지 시간.** 에이전트가 문장 중간에 끼어들게 됩니다. 대화형 발화에서는 500-800 ms가 최적입니다.
- **너무 긴 침묵 유지 시간.** 반응이 느려집니다. 대상 사용자와 A/B 테스트를 진행해 보세요.
- **프리롤 버퍼 없음.** 사용자 오디오의 첫 200-300 ms가 손실됩니다. 항상 롤링 프리롤을 유지하세요.
- **시맨틱 엔드포인트링 무시.** "음, 잠깐 생각해보자..."는 긴 침묵을 포함합니다. 사용자는 생각 중간에 끊기는 것을 싫어합니다. LiveKit의 turn-detector나 유사한 도구를 사용하세요.

## 출시하기

`outputs/skill-vad-tuner.md`로 저장하세요. 워크로드에 맞는 VAD 모델, 임계값, 침묵 유지 시간, 프리롤 및 턴 감지 전략을 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 발화 + 침묵 + 발화 + 기침 시퀀스를 시뮬레이션하고 세 가지 VAD 티어를 테스트합니다.
2. **중간.** `silero-vad`를 설치하고, 5분 녹음을 처리하며, 임계값을 조정하여 첫 단어 클립과 오탐을 최소화하세요. 정밀도/재현율을 보고하세요.
3. **어려움.** 미니 턴 감지기를 구축하세요: Silero VAD + 마지막 10개 단어의 임베딩에 대한 3층 MLP (sentence-transformers 사용). 수동으로 라벨링된 턴 종료 데이터셋으로 학습하세요. Silero 단독 대비 F1을 10% 개선하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| VAD | 음성 감지 | 프레임별 바이너리: 이것이 발화인가? |
| 턴 감지 | 엔드포인트링 | VAD + 침묵 유지 시간 + 시맨틱 엔드포인트. |
| 침묵 유지 시간 | 발화 후 대기 | 턴 종료 선언 전 대기 시간; 500-800 ms. |
| 프리롤 | 발화 전 버퍼 | VAD가 발동하기 전 300-500 ms 오디오를 유지하세요. |
| 플러시 트릭 | Kyutai 해킹 | VAD → STT 플러시 → 500 ms 지연 대신 125 ms. |
| 시맨틱 엔드포인트 | "그들이 멈추려 했는가?" | 침묵뿐만 아니라 단어를 보는 ML 분류기. |
| TPR @ FPR 5% | ROC 포인트 | 표준 VAD 벤치마크; Silero는 87.7%, WebRTC는 50%. |

## 추가 읽기

- [Silero VAD](https://github.com/snakers4/silero-vad) — 참조 오픈 VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/) — 상용 정확도 리더.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 200 ms 미만 엔지니어링 트릭.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 프로덕션 환경에서의 시맨틱 엔드포인트 감지입니다.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 레거시 기준선입니다.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 화자 분리 등급의 세그멘테이션입니다.
