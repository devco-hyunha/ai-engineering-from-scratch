# 오디오 트랜스포머 — Whisper 아키텍처

> 오디오는 시간에 따른 주파수의 이미지입니다. Whisper는 mel 스펙트로그램을 입력으로 받아 텍스트를 출력하는 ViT입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 7단계 · 05강 (전체 트랜스포머), 7단계 · 08강 (인코더-디코더), 7단계 · 09강 (ViT)
**시간:** 약 45분

## 문제점

Whisper (OpenAI, Radford et al. 2022) 이전에는 최첨단 자동 음성 인식 (ASR)(Automatic Speech Recognition (ASR))은 wav2vec 2.00강 HuBERT를 의미했습니다. 이는 자기 지도 학습 특징 추출기와 미세 조정된 헤드(fine-tuned head)를 결합한 방식입니다. 높은 품질의 데이터 파이프라인이 필요하며, 도메인에 민감했습니다. 다국어 음성 인식은 언어 가족별로 별도의 모델이 필요했습니다.

Whisper는 세 가지 전략을 선택했습니다:

1. **모든 데이터로 학습.** 인터넷에서 수집한 97개 언어의 680,000시간의 약한 라벨이 붙은 오디오를 사용했습니다. 깨끗한 학술 코퍼스가 없었고, 음소 라벨도 없었습니다.
2. **단일 모델로 다중 작업 수행.** 하나의 디코더가 전사, 번역, 음성 활동 감지, 언어 식별, 타임스탬핑을 태스크 토큰(task tokens)을 통해 공동으로 학습합니다.
3. **표준 인코더-디코더 트랜스포머.** 인코더는 log-mel 스펙트로그램을 소비합니다. 디코더는 자기회귀(Autoregressive) 방식으로 텍스트 토큰을 생성합니다. 보코더(vocoder), CTC, HMM이 없습니다.

결과적으로, Whisper large-v3는 악센트, 잡음, 깨끗한 라벨 데이터가 전혀 없는 언어에서도 견고합니다. 2026년 현재 모든 오픈소스 음성 어시스턴트와 대부분의 상용 음성 어시스턴트의 기본 음성 프론트엔드입니다.

## 개념

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 1단계 — 리샘플링 및 윈도우

오디오는 16 kHz로 처리됩니다. 30초로 클리핑(clipping)하거나 패딩(padding)합니다. log-mel 스펙트로그램을 계산합니다: 80 mel bins, 10 ms stride → 약 3,000 프레임 × 80 특징(features). 이것이 Whisper가 보는 "입력 이미지"입니다.

### 2단계 — 컨볼루션 스템(convolutional stem)

커널(kernel) 3, stride 2를 가진 두 개의 Conv1D 레이어가 3,000 프레임을 1,500으로 줄입니다. 많은 매개변수(Parameter)를 추가하지 않고 시퀀스 길이를 절반으로 줄입니다.

### 3단계 — 인코더

1,500 타임스텝(timestep)에 대해 24층(large 모델 기준) 트랜스포머 인코더를 적용합니다. 사인(positional encoding), 셀프 어텐션(Self-Attention), GELU FFN을 사용합니다. 1,500 × 1,280 은닉 상태(hidden states)를 생성합니다.

### 4단계 — 디코더

24층 트랜스포머 디코더입니다. GPT-2의 어휘를 포함하는 BPE 어휘에서 몇 가지 오디오 전용 특수 토큰을 추가하여 자기회귀 방식으로 토큰을 생성합니다.

### 5단계 — 작업 토큰

디코더 프롬프트는 모델이 무엇을 해야 하는지 알려주는 제어 토큰으로 시작합니다:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

또는

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

모델은 이 규칙에 따라 학습되었습니다. 접두어로 작업을 제어합니다. 2026년版的 지시문 미세 조정(instruction-tuning)에 해당하지만, 음성 영역에 적용된 것입니다.

### 6단계 — 출력

로그 확률 임계값을 사용하는 빔 검색(너비 5)을 수행합니다. `<|notimestamps|>` 토큰이 없을 때 오디오 0.02초마다 타임스탬프가 예측됩니다.

### Whisper 크기

| 모델 | 파라미터 | 레이어 | d_model | 헤드 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4층 디코더) |

Large-v3-turbo (2024)는 디코더를 32층에서 4층으로 줄였습니다. WER 점수 손실이 1 미만인 상태로 디코딩 속도를 8배 높였습니다. 이 디코딩 속도 향상 덕분에 Whisper-turbo는 2026년 실시간 음성 에이전트의 기본 모델이 되었습니다.

### Whisper가 하지 않는 것

- 화자 분리(diarization, 누가 말하는지 식별)는 지원하지 않습니다. 이 기능은 pyannote와 함께 사용하세요.
- 네이티브 실시간 스트리밍은 지원되지 않습니다. 30초 윈도우가 고정되어 있습니다. 최신 래퍼(`faster-whisper`, `WhisperX`)는 VAD + 겹침을 통해 스트리밍을 추가합니다.
- 외부 청킹 없이는 30초 이상의 긴 문맥을 지원하지 않습니다. 인간의 발화는 전사 시 긴 범위의 문맥이 거의 필요하지 않으므로 실제로는 잘 작동합니다.

### 2026년 현황

| 작업 | 모델 | 비고 |
|------|-------|-------|
| 영어 ASR | Whisper-turbo, Moonshine | Moonshine는 엣지에서 4배 빠름 |
| 다국어 ASR | Whisper-large-v3 | 97개 언어 |
| 스트리밍 ASR | faster-whisper + VAD | 150 ms 지연 목표 달성 가능 |
| TTS | Piper, XTTS-v2, Kokoro | 인코더-디코더 패턴, Whisper 형태 |
| 오디오 + 언어 | AudioLM, SeamlessM4T | 하나의 트랜스포머에서 텍스트 토큰 + 오디오 토큰 |

```figure
n5-mel-decode
```

## 구현하기

`code/main.py`를 참고하세요. Whisper를 훈련하지는 않습니다. 로그-멜 스펙트로그램 파이프라인과 작업 토큰 프롬프트 포매터를 구축합니다. 프로덕션에서 실제로 다루는 부분입니다.

### 1단계: 오디오 합성

16 kHz로 샘플링된 440 Hz의 1초 사인파를 생성하세요. 16,000개의 샘플입니다.

### 2단계: 로그-멜 스펙트로그램 (단순화)

전체 멜 스펙트로그램은 FFT가 필요합니다. `librosa` 없이 파이프라인을 보여주기 위해 단순화된 프레임 분할 + 프레임별 에너지 버전을 수행합니다:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

프레임 = 25 ms, 호프 = 10 ms. Whisper의 윈도우 처리와 일치합니다. 교육적 목적을 위해 프레임별 에너지가 멜 빈을 대체합니다.

### 3단계: 30초로 패딩

Whisper는 항상 30초 청크를 처리합니다. 스펙트로그램을 3,000 프레임으로 패딩(또는 클리핑)하세요.

### 4단계: 프롬프트 토큰 구축

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

이것이 전체 작업 제어 표면입니다. 4개 토큰의 접두어입니다.

## 사용하기

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

더 빠르고 OpenAI 호환:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026년에 Whisper를 선택해야 하는 경우:**

- 하나의 모델로 다국어 ASR를 수행합니다.
- 잡음이 많고 다양한 오디오의 견고한 전사(transcription)를 수행합니다.
- 연구 / 프로토타입 ASR — 가장 빠른 시작점입니다.

**다른 것을 선택해야 하는 경우:**

- 엣지에서 초저지연 스트리밍 — 동일한 품질에서 Moonshine가 Whisper보다 우수합니다.
- <200 ms가 필요한 실시간 대화형 AI — 전용 스트리밍 ASR를 사용하세요.
- 화자 분리(speaker diarization) — Whisper는 이 기능을 수행하지 않으며, pyannote를 추가해야 합니다.

## 출시하기

`outputs/skill-asr-configurator.md`를 참고하세요. 이 스킬은 새로운 음성 애플리케이션을 위해 ASR 모델, 디코딩 매개변수 및 전처리 파이프라인을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 16 kHz에서 10 ms 호프를 사용하는 1초 신호의 프레임 수가 약 100 프레임임을 확인하세요. 30초의 경우: 약 3,000 프레임입니다.
2. **중간.** `numpy.fft`를 사용하여 전체 로그-멜 스펙트로그램을 구축하세요. 80개의 멜 빈이 `librosa.feature.melspectrogram(n_mels=80)`와 수치 오차 범위 내에서 일치하는지 검증하세요.
3. **난이도: 상.** 스트리밍 추론을 구현해 보세요. 오디오를 10초 윈도우로 청킹하고 2초 겹침을 적용한 후, 각 청크에 Whisper를 실행하여 전사문을 병합합니다. 5분 분량의 팟캐스트 샘플에서 단일 패스(single-pass) 대비 단어 오류율(word-error rate)을 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 멜 스펙트로그램 | "오디오 이미지" | 2D 표현: 한 축은 주파수 빈(bin), 다른 축은 시간 프레임이며, 각 셀은 로그 스케일 에너지입니다. |
| 로그-멜 | "Whisper가 보는 것" | 로그를 거친 멜 스펙트로그램; 인간의 청각 지각을 근사합니다. |
| 프레임 | "하나의 시간 조각" | 25ms 샘플 윈도우; 10ms 스트라이드로 겹침. |
| 태스크 토큰 | "음성용 프롬프트 접두어" | 디코더 프롬프트의 `<\|transcribe\|>` / `<\|translate\|>` 같은 특수 토큰. |
| 음성 활동 감지 (VAD) | "음성을 찾아라" | ASR 전에 침묵을 제거하는 게이트; 비용을 대폭 절감합니다. |
| CTC | "Connectionist Temporal Classification" | 정렬 없는 학습을 위한 고전적인 ASR 손실 함수; Whisper는 이를 사용하지 않습니다. |
| Whisper-turbo | "작은 디코더, 완전한 인코더" | large-v3 인코더 + 4층 디코더; 디코딩 속도가 8배 빠릅니다. |
| Faster-whisper | "프로덕션 래퍼" | CTranslate2 재구현; int8 양자화; OpenAI의 참조 구현보다 4배 빠릅니다. |

## 추가 읽기

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 논문.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 참조 코드 + 모델 가중치. `whisper/model.py`를 읽어 Conv1D 스템 + 인코더 + 디코더를 약 400줄에 걸쳐 상에서 하로 살펴보세요.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 5~6단계에서 설명한 빔 검색 + 태스크 토큰 로직이 여기 있습니다; 500줄로 완전히 읽을 수 있습니다.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 선구자; 일부 환경에서는 여전히 SOTA 기능입니다.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 프로덕션 래퍼, 참조 구현보다 4배 빠름.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024년 엣지 친화적 ASR, Whisper 형태이지만 더 작습니다.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 멜 스펙트로그램 전처리기와 토큰 타임스탬프 처리를 포함한 표준 미세 조정 레시피.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 강의의 아키텍처 다이어그램을 반영하는 전체 구현 (인코더, 디코더, 교차 어텐션, 생성).
