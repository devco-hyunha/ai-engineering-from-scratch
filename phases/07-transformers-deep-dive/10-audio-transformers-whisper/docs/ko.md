# 오디오 트랜스포머(Audio Transformers) — Whisper 아키텍처

> 오디오는 시간에 따른 주파수의 이미지입니다. Whisper는 멜 스펙트로그램(mel spectrograms)을 입력받아 다시 텍스트로 변환하는 ViT(Vision Transformer)와 유사한 구조를 가집니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## 문제점 (The Problem)

Whisper(OpenAI, Radford et al. 2022)가 등장하기 전, 최첨단 자동 음성 인식(ASR) 기술은 wav2vec 2.0 및 HuBERT가 주도했습니다. 이는 자기 지도 학습(self-supervised) 기반의 특징 추출기(feature extractor)에 미세 조정된 헤드(fine-tuned head)를 결합한 방식이었습니다. 품질은 높았으나 데이터 파이프라인 구축 비용이 비쌌고, 특정 도메인에 취약했습니다. 또한 다국어 음성 인식을 위해서는 언어군별로 별도의 모델이 필요했습니다.

Whisper는 세 가지 전략에 승부수를 던졌습니다:

1. **모든 데이터를 통한 학습 (Train on everything).** 인터넷에서 수집한 97개 언어, 680,000시간 분량의 약하게 레이블링된(weakly-labeled) 오디오 데이터를 사용했습니다. 깨끗한 학술용 코퍼스나 음소(phoneme) 레이블은 사용하지 않았습니다.
2. **단일 모델의 멀티태스크 수행 (Multi-task single model).** 하나의 디코더가 태스크 토큰(task tokens)을 통해 전사(transcription), 번역(translation), 음성 활동 감지(voice activity detection), 언어 식별(language ID), 타임스탬프 생성(timestamping)을 공동으로 학습합니다.
3. **표준 인코더-디코더 트랜스포머 (Standard encoder-decoder transformer).** 인코더는 로그-멜 스펙트로그램(log-mel spectrograms)을 입력으로 받습니다. 디코더는 자기회귀(autoregressively) 방식으로 텍스트 토큰을 생성합니다. 보코더(vocoder), CTC, HMM은 사용하지 않습니다.

그 결과: Whisper large-v3는 억양, 소음, 그리고 깨끗한 레이블링 데이터가 전혀 없는 언어에 대해서도 강력한 성능을 보여줍니다. 이는 2026년 기준, 모든 오픈 소스 음성 비서와 대부분의 상용 서비스에서 기본 음성 프런트엔드(speech front-end)로 자리 잡았습니다.

## 개념 (The Concept)

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 1단계: 리샘플링 + 윈도우 (resample + window)

오디오를 16 kHz로 리샘플링합니다. 30초 길이에 맞춰 클립(clip)하거나 패딩(pad)합니다. 로그-멜 스펙트로그램(log-mel spectrogram)을 계산합니다: 80개의 mel bin, 10 ms 스트라이드(stride) → 약 3,000 프레임 × 80개 특징(features). 이것이 Whisper가 보는 "입력 이미지(input image)"입니다.

### 2단계: 컨볼루션 스템 (convolutional stem)

커널 크기 3과 스트라이드 2를 가진 두 개의 `Conv1D` 레이어가 3,000개의 프레임을 1,500개로 줄입니다. 이는 많은 파라미터를 추가하지 않고도 시퀀스 길이를 절반으로 줄여줍니다.

### 3단계: 인코더 (encoder)

1,500개의 타임스텝(timesteps)에 대해 24개 레이어(large 모델 기준)로 구성된 트랜스포머 인코더(transformer encoder)입니다. 사인 함수 기반 위치 인코딩(Sinusoidal positional encoding), 셀프 어텐션(self-attention), GELU FFN을 사용합니다. 1,500 × 1,280 크기의 은닉 상태(hidden states)를 생성합니다.

### 4단계: 디코더 (decoder)

24개의 레이어로 구성된 트랜스포머 디코더(transformer decoder)입니다. 이 디코더는 GPT-2의 어휘 집합(vocabulary)을 포함하면서 몇 가지 오디오 전용 특수 토큰(special tokens)이 추가된 BPE 어휘 집합으로부터 토큰을 자기회귀(autoregressively) 방식으로 생성합니다.

### 5단계: 태스크 토큰 (task tokens)

디코더 프롬프트는 모델에게 무엇을 해야 할지 알려주는 제어 토큰(control tokens)으로 시작합니다:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

또는

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

모델은 이러한 관례(convention)에 따라 학습되었습니다. 여러분은 접두사(prefix)를 통해 태스크를 제어합니다. 이는 음성 분야에 적용된, 2026년 버전의 인스트럭션 튜닝(instruction-tuning)과 같습니다.

### 6단계: 출력 (output)

로그 확률(log-prob) 임계값이 적용된 빔 서치(Beam search, 너비 5)를 사용합니다. `<|notimestamps|>` 토큰이 없는 경우, 오디오의 0.02초마다 타임스탬프가 예측됩니다.

### Whisper 모델 크기 (Whisper sizes)

| 모델 (Model) | 파라미터 (Params) | 레이어 (Layers) | d_model | 헤드 (Heads) | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024)는 디코더를 32개 레이어에서 4개 레이어로 줄였습니다. WER(Word Error Rate) 저하가 1점 미만인 상태에서 8배 더 빠른 디코딩 속도를 제공합니다. 이러한 디코딩 속도의 혁신 덕분에 Whisper-turbo는 2026년 실시간 음성 에이전트의 기본 모델로 자리 잡았습니다.

### Whisper가 수행하지 않는 작업 (What Whisper does not do)

- 화자 분리(Diarization, 누가 말하고 있는지) 기능이 없습니다. 이를 위해서는 `pyannote`와 함께 사용해 보세요.
- 기본적으로 실시간 스트리밍(Real-time streaming)을 지원하지 않습니다 — 30초 단위의 윈도우가 고정되어 있습니다. 최신 래퍼(`faster-whisper`, `WhisperX`)들은 VAD + 오버랩(overlap) 방식을 통해 스트리밍 기능을 추가로 구현합니다.
- 외부 청킹(Chunking) 없이는 30초 이상의 장문 컨텍스트(Long-form context)를 처리하지 못합니다. 하지만 실제로는 인간의 음성이 전사(Transcription)를 위해 긴 범위의 컨텍스트를 필요로 하는 경우가 드물기 때문에 실무에서는 잘 작동합니다.

### 2026년 전망 (2026 landscape)

| 작업 (Task) | 모델 (Model) | 비고 (Notes) |
|------|-------|-------|
| 영어 ASR | Whisper-turbo, Moonshine | Moonshine은 엣지 기기에서 4배 더 빠름 |
| 다국어 ASR | Whisper-large-v3 | 97개 언어 지원 |
| 스트리밍 ASR | faster-whisper + VAD | 150ms 지연 시간 목표 달성 가능 |
| TTS | Piper, XTTS-v2, Kokoro | 인코더-디코더 패턴이지만, Whisper 형태를 따름 |
| 오디오 + 언어 | AudioLM, SeamlessM4T | 하나의 트랜스포머 내에 텍스트 토큰 + 오디오 토큰 포함 |

```figure
n5-mel-decode
```

## 구현하기 (Build It)

`code/main.py`를 확인해 보세요. 우리는 Whisper를 직접 학습시키지 않습니다. 대신 log-mel spectrogram 파이프라인과 task-token 프롬프트 포맷터(prompt formatter)를 구축합니다. 이 부분들이 실제 프로덕션 환경에서 여러분이 직접 다루게 될 핵심 요소들입니다.

### 1단계: 오디오 합성 (synthesize audio)

440 Hz의 주파수를 가진 1초 길이의 사인파(sine wave)를 16 kHz 샘플링 레이트로 생성하세요. 총 16,000개의 샘플이 생성됩니다.

### 2단계: log-mel spectrogram (간소화 버전)

전체 mel spectrogram을 구하려면 FFT(고속 푸리에 변환)가 필요합니다. 여기서는 `librosa` 라이브러리 없이도 파이프라인을 이해할 수 있도록, 프레이밍(framing)과 프레임별 에너지(per-frame energy)를 사용하는 간소화된 버전을 구현해 보겠습니다.

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

프레임 크기(Frame)는 25ms, 홉(hop)은 10ms로 설정했습니다. 이는 Whisper의 윈도잉(windowing) 방식과 일치합니다. 교육적 목적을 위해 프레임별 에너지를 mel bin 대신 사용합니다.

### 3단계: 30초로 패딩(pad)하기

Whisper는 항상 30초 단위의 청크(chunk)를 처리합니다. 스펙트로그램(spectrogram)이 3,000 프레임이 되도록 패딩(또는 클리핑)해 보세요.

### 4단계: 프롬프트 토큰(prompt tokens) 구축하기

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

이것이 전체 작업 제어 인터페이스(task-control surface)입니다. 4개의 토큰으로 구성된 접두사(prefix)입니다.

## 사용 방법 (Use It)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

더 빠르고 OpenAI와 호환되는 방식:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026년에 Whisper를 선택해야 하는 경우:**

- 단일 모델로 다국어 ASR(자동 음성 인식)을 구현할 때.
- 노이즈가 많거나 다양한 오디오 환경에서도 견고한 전사가 필요할 때.
- 연구 또는 프로토타입 ASR 개발 시 — 가장 빠르게 시작할 수 있는 지점입니다.

**다른 대안을 선택해야 하는 경우:**

- 엣지(edge) 환경에서 초저지연 스트리밍이 필요할 때 — 동일한 품질 기준에서 Moonshine이 Whisper보다 우수합니다.
- 200ms 미만의 응답 속도가 필요한 실시간 대화형 AI — 전용 스트리밍 ASR을 사용하세요.
- 화자 분리(Speaker diarization)가 필요할 때 — Whisper는 이 기능을 지원하지 않으므로 `pyannote`를 결합하여 사용해야 합니다.

## Ship It (실전 적용)

`outputs/skill-asr-configurator.md`를 참조하세요. 이 스킬은 새로운 음성 애플리케이션을 위해 ASR 모델, 디코딩 파라미터(decoding parameters) 및 전처리 파이프라인(preprocessing pipeline)을 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 16 kHz 샘플링 레이트에서 10 ms 홉(hop) 크기를 가진 1초 길이 신호의 프레임 수가 약 100 프레임인지 확인해 보세요. 30초의 경우 약 3,000 프레임이 나와야 합니다.
2. **중간 (Medium).** `numpy.fft`를 사용하여 전체 로그-멜 스펙트로그램(log-mel spectrogram)을 직접 구현해 보세요. 생성된 80개의 멜 빈(mel bins)이 수치적 오차 범위 내에서 `librosa.feature.melspectrogram(n_mels=80)`의 결과와 일치하는지 검증해 보세요.
3. **어려움 (Hard).** 스트리밍 추론(streaming inference)을 구현해 보세요. 오디오를 2초의 중첩(overlap)을 가진 10초 단위 윈도우로 분할하고, 각 청크(chunk)에 대해 Whisper를 실행한 뒤 전사(transcript)를 병합하세요. 5분 길이의 팟캐스트 샘플을 사용하여 단일 패스(single-pass) 방식 대비 단어 오류율(word-error rate)을 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Mel spectrogram (멜 스펙트로그램) | "오디오 이미지" | 2D 표현 방식: 한 축은 주파수 빈(frequency bins), 다른 축은 시간 프레임(time frames)이며, 각 셀은 로그 스케일의 에너지를 나타냅니다. |
| Log-mel (로그-멜) | "Whisper가 보는 것" | 멜 스펙트로그램에 로그를 적용한 것; 인간의 소리 크기 인지 방식을 근사화합니다. |
| Frame (프레임) | "한 번의 시간 슬라이스" | 25ms 길이의 샘플 윈도우; 10ms 스트라이드(stride)로 중첩됩니다. |
| Task token (태스크 토큰) | "음성을 위한 프롬프트 접두사" | 디코더 프롬프트에 포함되는 `<|transcribe|>` / `<|translate|>`와 같은 특수 토큰입니다. |
| Voice activity detection (VAD, 음성 활동 감지) | "음성 구간 찾기" | ASR 수행 전 무음 구간을 제거하는 게이트 역할을 하며, 비용을 대폭 절감합니다. |
| CTC (Connectionist Temporal Classification) | "연결주의 시간적 분류" | 정렬(alignment) 없이 학습하기 위한 고전적인 ASR 손실 함수; Whisper는 이를 사용하지 않습니다. |
| Whisper-turbo (Whisper-터보) | "작은 디코더, 전체 인코더" | large-v3 인코더 + 4개 레이어 디코더 조합; 8배 빠른 디코딩을 제공합니다. |
| Faster-whisper (Faster-whisper) | "프로덕션용 래퍼" | CTranslate2를 이용한 재구현체; int8 양자화 적용; OpenAI의 레퍼런스 모델보다 4배 빠릅니다. |

## 추가 학습 자료 (Further Reading)

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 논문입니다.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 참조 코드와 모델 가중치입니다. `whisper/model.py`를 읽어보시면 약 400줄의 코드 안에 `Conv1D` stem + encoder + decoder가 어떻게 구성되어 있는지 처음부터 끝까지 확인하실 수 있습니다.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 5~6단계에서 설명한 빔 서치(beam-search)와 태스크 토큰(task-token) 로직이 구현되어 있습니다. 약 500줄로 구성되어 있어 충분히 읽어볼 만합니다.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — Whisper의 전신 격인 연구로, 일부 환경에서는 여전히 SOTA(State-of-the-Art) 기능을 제공합니다.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 프로덕션용 래퍼(wrapper)로, 참조 구현보다 4배 더 빠릅니다.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024년 발표된 엣지(edge) 친화적 ASR 모델로, Whisper와 유사한 구조를 가지면서 크기는 더 작습니다.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — mel spectrogram 전처리기와 토큰-타임스탬프(token-timestamp) 처리를 포함한 표준적인 미세 조정(fine-tuning) 레시피를 제공합니다.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 본 강의의 아키텍처 다이어그램을 그대로 반영한 전체 구현체(encoder, decoder, cross-attention, generation)입니다.
