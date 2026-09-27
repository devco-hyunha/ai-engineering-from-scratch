# 음성 활동 감지 및 발화 전환 (Voice Activity Detection & Turn-Taking) — Silero, Cobra, 그리고 Flush Trick

> 모든 음성 에이전트의 성패는 두 가지 결정에 달려 있습니다: 사용자가 지금 말하고 있는가, 그리고 말을 마쳤는가? VAD는 첫 번째 질문에 답합니다. 발화 전환 감지(Turn-detection: VAD + silence-hangover + semantic endpoint model)는 두 번째 질문에 답합니다. 이 중 하나라도 틀리면, 어시스턴트는 사용자의 말을 끊어버리거나 멈추지 않고 계속 떠들게 됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## 문제점 (The Problem)

음성 에이전트가 매 20ms 청크(chunk)마다 내리는 세 가지 별개의 결정:

1. **이 프레임이 음성인가?** — VAD(Voice Activity Detection). 프레임별 이진 분류(Binary).
2. **사용자가 새로운 발화를 시작했는가?** — 온셋 탐지(onset detection).
3. **사용자가 말을 마쳤는가?** — 엔드포인팅(end-pointing, 발화 종료).

단순한 방식(에너지 임계값 사용)은 교통 소음, 키보드 소리, 군중의 웅성거림과 같은 모든 종류의 노이즈에서 실패합니다. 2026년의 해답은 다음과 같습니다: Silero VAD(오픈 소스, 딥러닝 기반) + 발화 전환 탐지 모델(semantic endpointing) + VAD로 보정된 침묵 유예 시간(silence hangover).

## 개념 (The Concept)

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 3계층 VAD 캐스케이드 (The three-tier VAD cascade)

**1계층: 에너지 게이트 (energy gate).** 가장 비용이 저렴합니다. 임계값 RMS를 -40 dBFS로 설정합니다. 명백한 침묵은 필터링하지만, 임계값을 넘는 모든 노이즈에도 반응합니다.

**2계층: Silero VAD** (2020-2026, MIT). 100만 개의 파라미터를 가집니다. 6,000개 이상의 언어로 학습되었습니다. 단일 CPU 스레드에서 30ms 청크당 약 1ms 내에 실행됩니다. 5% FPR(오탐률)에서 87.7%의 TPR(정탐률)을 기록합니다. 오픈 소스의 표준 모델입니다.

**3계층: 의미론적 발화 전환 탐지기 (semantic turn detector).** LiveKit의 발화 전환 모델(2024-2026) 또는 자체 제작한 소형 분류기를 사용합니다. "문장 중간의 일시 정지"와 "말하기 종료"를 구분합니다. 단순히 침묵을 감지하는 것이 아니라 언어적 문맥(억양 + 최근 단어)을 사용합니다.

### 주요 파라미터 및 기본값 (Key parameters and their defaults)

- **임계값 (Threshold).** Silero는 확률값을 출력합니다. 0.5(기본값) 초과 또는 0.3(민감함) 초과 시 음성으로 분류하세요. 임계값이 낮을수록 첫 단어가 잘리는 현상은 줄어들지만, 오탐(false positives)은 늘어납니다.
- **최소 음성 지속 시간 (Minimum speech duration).** 250ms보다 짧은 음성은 거부하세요. 보통 기침 소리나 의자 소음인 경우가 많습니다.
- **침묵 유예 시간 (Silence hangover / end-pointing).** VAD가 0으로 돌아간 후, 발화 종료(end-of-turn)를 선언하기 전까지 500-800ms 동안 대기합니다. 이 시간이 너무 짧으면 사용자의 말을 끊게 되고, 너무 길면 반응이 느리게 느껴집니다.
- **프리롤 버퍼 (Pre-roll buffer).** VAD가 작동하기 전의 오디오를 300-500ms 동안 유지하세요. "hey"와 같은 첫 단어가 잘리는 것을 방지합니다.

### 플러시 트릭 (The flush trick, Kyutai 2025)

스트리밍 STT 모델은 룩어헤드 지연(look-ahead delay)을 가집니다 (Kyutai STT-1B의 경우 500ms, STT-2.6B의 경우 2.5s). 일반적으로는 발화 종료(end-of-speech) 후 해당 시간만큼 기다려야 전사(transcript) 결과를 얻을 수 있습니다. 플러시 트릭(Flush trick)은 VAD가 발화 종료를 감지했을 때, **STT에 즉시 출력을 강제하는 플러시 신호(flush signal)를 전송**하는 방식입니다. STT는 실시간 대비 약 4배 속도로 처리하므로, 500ms의 버퍼는 약 125ms 만에 처리가 완료됩니다.

엔드 투 엔드(End-to-end): 125ms VAD + 플러시 STT = 대화 지연 시간(conversational latency).

### 2026 VAD 비교 (VAD comparison)

| VAD | TPR @ 5% FPR | 지연 시간 (Latency) | 라이선스 (License) |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | 상용 (commercial) |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero가 적절한 기본값(default)입니다. Cobra는 컴플라이언스 및 정확도 향상을 위한 업그레이드 옵션입니다. 에너지 기반 VAD(Energy-only VAD)는 2026년 프로덕션 환경에 적합하지 않습니다.

```figure
sp-vad-cascade
```

## 직접 구현해 보기 (Build It)

### 1단계: 에너지 게이트 (the energy gate)

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 2단계: Python에서의 Silero VAD

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

### 3단계: 발화 종료 상태 머신 (turn-end state machine)

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

### 4단계: flush 트릭 스켈레톤 (the flush trick skeleton)

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

이 방식이 작동하려면 STT(Kyutai, Deepgram, AssemblyAI)가 `flush`를 지원해야 합니다. Whisper 스트리밍은 이를 지원하지 않습니다. Whisper는 블록 기반(block-based) 방식이며 항상 청크(chunk)가 채워지기를 기다립니다.

## 사용 방법 (Use It)

| 상황 (Situation) | VAD 선택 (VAD choice) |
|-----------|-----------|
| 개방형, 빠른 속도, 범용적 사용 | Silero VAD |
| 상업용 콜센터 | Cobra VAD |
| 온디바이스 (휴대전화) | Silero VAD ONNX |
| 연구 / 화자 분리 (diarization) | pyannote segmentation |
| 의존성 없는 폴백 (fallback) | WebRTC VAD (legacy) |
| 발화 종료 품질이 필요한 경우 | Silero + LiveKit turn-detector 계층 구조 |

경험 법칙(Rule of thumb): 정말 다른 대안이 없는 경우가 아니라면, 에너지 기반(energy-only) VAD는 절대 출시하지 마세요.

## 주의 사항 (Pitfalls)

- **고정 임계값 (Fixed threshold).** 조용한 환경에서는 잘 작동하지만, 소음이 있는 환경에서는 실패합니다. 기기에서 직접 보정(calibrate)하거나 Silero로 전환해 보세요.
- **너무 짧은 침묵 유예 시간 (Too-short silence hangover).** 에이전트가 문장 중간에 말을 끊습니다. 대화형 음성에는 500-800ms가 가장 적절합니다.
- **너무 긴 유예 시간 (Too-long hangover).** 반응이 느리게 느껴집니다. 타겟 사용자를 대상으로 A/B 테스트를 진행해 보세요.
- **프리롤 버퍼 부재 (No pre-roll buffer).** 사용자 오디오의 첫 200-300ms가 손실됩니다. 항상 롤링 프리롤(rolling pre-roll)을 유지하세요.
- **의미론적 종결 처리 무시 (Ignoring semantic endpointing).** "음, 잠시만요..."와 같은 문장에는 긴 휴지(pause)가 포함됩니다. 사용자는 생각하는 도중에 말이 끊기는 것을 싫어합니다. LiveKit의 `turn-detector` 또는 유사한 기술을 사용해 보세요.

## Ship It (실행하기)

`outputs/skill-vad-tuner.md`로 저장하세요. 특정 워크로드에 적합한 VAD 모델, 임계값(threshold), 침묵 유예 시간(hangover), 프리롤(pre-roll) 및 발화 전환 감지 전략(turn-detection strategy)을 선택해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 이 코드는 음성 + 침묵 + 음성 + 기침 시퀀스를 시뮬레이션하여 세 가지 VAD 계층을 테스트합니다.
2. **중간 (Medium).** `silero-vad`를 설치하고 5분 분량의 녹음 파일을 처리해 보세요. 첫 단어 잘림(first-word clips)과 오탐지(false triggers)를 모두 최소화하도록 임계값(threshold)을 조정하세요. 정밀도(precision)와 재현율(recall)을 보고하세요.
3. **어려움 (Hard).** 미니 발화 전환 탐지기(turn-detector)를 구축해 보세요: Silero VAD와 마지막 10개 단어의 임베딩(sentence-transformers 사용)에 대한 3계층 MLP를 결합합니다. 수동으로 라벨링된 발화 종료(turn-end) 데이터셋으로 학습시키세요. Silero만 사용했을 때보다 F1 점수를 10% 높게 달성해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| VAD | 음성 탐지기 (Voice detector) | 프레임별 이진 분류: 이것이 음성인가? |
| Turn detection | 발화 전환 감지 / 엔드포인팅 (End-pointing) | VAD + 침묵 유예 시간(silence-hangover) + 의미론적 엔드포인트(semantic endpoint). |
| Silence hangover | 음성 후 대기 (Wait-after-speech) | 발화 종료로 간주하기 전 대기 시간; 500-800 ms. |
| Pre-roll | 음성 전 버퍼 (Pre-speech buffer) | VAD가 작동하기 전 300-500 ms의 오디오를 유지함. |
| Flush trick | Kyutai 해킹 (Kyutai hack) | VAD → flush-STT → 500 ms 지연 대신 125 ms 지연 적용. |
| Semantic endpoint | "정말로 멈추려던 것인가?" | 침묵뿐만 아니라 단어를 분석하는 ML 분류기. |
| TPR @ FPR 5% | ROC 지점 (ROC point) | 표준 VAD 벤치마크; Silero는 87.7%, WebRTC는 50%. |

## 추가 학습 자료 (Further Reading)

- [Silero VAD](https://github.com/snakers4/silero-vad) — 참조용 오픈 소스 VAD입니다.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 상용 제품 중 정확도 면에서 선두를 달리고 있습니다.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 200ms 미만의 지연 시간을 달성하기 위한 엔지니어링 트릭입니다.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 프로덕션 환경에서의 의미론적 엔드포인팅(semantic endpointing)을 다룹니다.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 레거시 베이스라인입니다.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 화자 분리(diarization) 수준의 세그멘테이션을 제공합니다.
