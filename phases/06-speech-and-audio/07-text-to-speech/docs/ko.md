# Text-to-Speech (TTS) — Tacotron에서 F5 및 Kokoro까지

> ASR은 음성을 텍스트로 변환하고, TTS는 텍스트를 음성으로 변환합니다. 2026년 스택은 세 부분으로 구성됩니다: 텍스트 → 토큰, 토큰 → 멜(mel), 멜 → 파형. 각 부분에는 노트북에서 실행 가능한 기본 모델이 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램 및 멜), 5단계 · 09강 (Seq2Seq), 7단계 · 05강 (Full Transformer)
**시간:** 약 75분

## 문제점

문자열이 있습니다: "Please remind me to water the plants at 6 pm." 자연스러운 3초 오디오 클립이 필요합니다. 올바른 운율(휴지, 강세)을 포함하고, "plants"의 모음을 정확히 발음하며, 라이브 음성 어시스턴트에서 CPU로 300ms 미만에 실행되어야 합니다. 또한 음성을 교체하고, 코드 전환 입력("remind me at 6 pm, daijoubu?")을 처리하며, 이름 발음으로 당황하지 않아야 합니다.

현대적인 TTS 파이프라인은 다음과 같습니다:

1. **텍스트 프론트엔드.** 텍스트를 정규화(날짜, 숫자, 이메일)하고, 음소나 하위 단어(subword) 토큰으로 변환하며, 운율(prosody) 특성을 예측합니다.
2. **음향 모델.** 텍스트 → 멜 스펙트로그램. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **보코더(Vocoder).** 멜 → 파형. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), 2024년 이후의 뉴럴 코덱 보코더.

2026년에는 엔드투엔드(end-to-end) 확산(diffusion) 및 흐름 매칭(flow-matching) 모델로 인해 음향 모델과 보코더의 경계가 모호해지고 있습니다. 하지만 디버깅을 위해서는 세 부분으로 나누는 개념이 여전히 유효합니다.

## 개념

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: 문자 임베딩(char-embedding) → BiLSTM 인코더 → 위치 민감 주의(location-sensitive attention) → 자기회귀(autoregressive) LSTM 디코더가 멜 프레임을 생성합니다. 느린 AR 방식이며, 긴 텍스트에서는 불안정합니다. 여전히 기준선(baseline)으로 인용됩니다.

**FastSpeech 2 (2020).** 비자기회귀(non-autoregressive) 방식. 지속 시간 예측기(duration predictor)가 각 음소에 몇 개의 멜 프레임이 할당되는지 출력합니다. 1회 패스로, Tacotron보다 10배 빠릅니다. 자연스러움(단조 정렬)이 일부 손실되지만, 널리 사용되고 있습니다.

**VITS (2021).** 인코더 + 흐름 기반 지속 시간 + HiFi-GAN 보코더를 변분 추론(variational inference)을 사용하여 엔드투엔드로 함께 학습합니다. 높은 품질의 단일 모델입니다. 2022–2024년 동안 지배적인 오픈소스 TTS였습니다. 변형: YourTTS (다중 화자 제로샷), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** 플로우 매칭(flow matching) 기반의 확산 트랜스포머(Diffusion Transformer)입니다. 자연스러운 운율(prosody)을 생성하며, 5초의 참조 오디오로 제로샷(Zero-Shot) 음성 클로닝이 가능합니다. 2026년 오픈소스 TTS 리더보드에서 최상위에 위치합니다. 매개변수(Parameter)는 335M입니다.

**Kokoro (2024).** 소형 모델(82M)로 CPU에서 실행 가능하며, 실시간 사용에 최적화된 최고 수준의 영어 TTS입니다. 어휘(Vocabulary)가 제한된 영어 전용이며, 라이선스는 apache-2.0입니다.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** 상용 모델의 최신 기술입니다. ElevenLabs v2.5의 감정 태그("[whispered]", "[laughing]")와 캐릭터 음성 기능은 2026년 오디오북 제작을 주도하고 있습니다.

### 보코더(Vocoder)의 진화

| 시대 | 보코더(Vocoder) | 지연(Latency) | 품질 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 오프라인 전용 | 출시 당시 SOTA |
| 2018 | WaveRNN | ~실시간 | 좋음 |
| 2020 | HiFi-GAN | 실시간의 100배 | 인간에 근접 |
| 2022 | BigVGAN | 실시간의 50배 | 화자/언어에 걸쳐 일반화 |
| 2024 | SNAC, DAC (신경 코덱) | AR 모델과 통합 | 이산 토큰(Token), 비트 효율적 |

2026년에는 대부분의 "TTS" 모델이 텍스트에서 파형(waveform)까지 엔드투엔드(end-to-end)로 처리하며, 멜 스펙트로그램(mel spectrogram)은 내부 표현으로 사용됩니다.

### 평가(Evaluation)

- **MOS (Mean Opinion Score).** 1–5 척도, 크라우드소싱(crowd-sourced) 방식입니다. 여전히 골드 스탠다드(gold standard)이지만, 매우 느립니다.
- **CMOS (Comparative MOS).** A 대 B 선호도 비교입니다. 각 주석(annotation)에 대해 더 좁은 신뢰 구간(confidence intervals)을 제공합니다.
- **UTMOS, DNSMOS.** 참조(reference)가 없는 신경 MOS 예측기입니다. 리더보드에 사용됩니다.
- **CER (Character Error Rate) via ASR.** TTS 출력물을 Whisper로 처리하여 입력 텍스트와 CER를 계산합니다. 명료성(intelligibility)의 대리 지표(proxy)입니다.
- **SECS (Speaker Embedding Cosine Similarity).** 음성 클로닝 품질을 측정합니다.

LibriTTS test-clean에서의 2026년 수치입니다:

| 모델 | UTMOS | CER (Whisper 경유) | 크기 |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## 구현하기

### 1단계: 입력을 음소화(phonemize)합니다

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

음운소는 보편적인 연결 고리입니다. VITS급 품질 미만인 시스템에 원시 텍스트를 입력하지 마세요.

### 2단계: Kokoro 실행 (2026년 CPU 기본값)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 미국 영어
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 텐서, sr=24000
```

오프라인에서 단일 파일로 실행되며, 매개변수는 82M입니다.

### 3단계: 음성 클로닝으로 F5-TTS 실행

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5초 참조 클립과 그 전사문을 전달하세요. F5는 운율과 음색을 클로닝합니다.

### 4단계: HiFi-GAN 보코더를 처음부터 구축하기

튜토리얼 스크립트에 담기에는 너무 크지만, 형태는 다음과 같습니다:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4개의 업샘플링 블록, 총 256배로 멜 비율에서 오디오 비율로 변환
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 파형
```

학습: 적대적 (짧은 윈도우에 대한 판별자) + 멜 스펙트로그램 재구성 손실 + 특징 매칭 손실. 상품화됨 — `hifi-gan` 저장소나 nvidia-NeMo의 사전 학습된 체크포인트를 사용하세요.

### 5단계: 전체 파이프라인 (의사 코드)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 실시간 영어 음성 어시스턴트 | Kokoro (CPU) 또는 XTTS v2 (GPU) |
| 5초 참조로 음성 클로닝 | F5-TTS |
| 상업적 캐릭터 음성 | ElevenLabs v2.5 |
| 오디오북 내레이션 | ElevenLabs v2.5 또는 XTTS v2 + 미세 조정 |
| 저자원 언어 | 5–20시간의 대상 언어 데이터로 VITS 학습 |
| 표현적 / 감정 태그 | ElevenLabs v2.5 또는 StyleTTS 2 미세 조정 |

2026년 기준 오픈소스 리더: **품질은 F5-TTS, 효율성은 Kokoro**. 역사가가 아닌 한 Tacotron을 사용하지 마세요.

## 문제점

- **텍스트 정규화기가 없습니다.** "Dr. Smith"는 "Doctor"로 읽히나요, "Drive"로 읽히나요? "2026"은 "twenty twenty six"로 읽히나요, "two zero two six"로 읽히나요? 음운 변환기 BEFORE 정규화하세요.
- **OOV 고유 명사.** "Ghumare" → "ghyu-mair"? 알 수 없는 토큰에 대해 그래프eme-to-phoneme 모델의 폴백을 출시하세요.
- **클리핑.** 보코더 출력은 거의 클리핑되지 않지만, 추론 시 멜 스케일링 불일치로 ±1.0를 초과할 수 있습니다. 항상 `np.clip(wav, -1, 1)`를 사용하세요.
- **샘플링 속도 불일치.** Kokoro는 24 kHz를 출력합니다. 다운스트림 파이프라인이 16 kHz를 기대한다면 → 리샘플링을 하거나, 그렇지 않으면 에이리어싱이 발생합니다.

## 출시하기

`outputs/skill-tts-designer.md`로 저장하세요. 주어진 음성, 지연 시간, 언어 목표를 위해 TTS 파이프라인을 설계해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 장난감 어휘 목록에서 음소 사전(phoneme dictionary)을 구축하고, 음소별 지속 시간을 추정하며, 가짜 "mel" 스케줄을 출력합니다.
2. **중간.** Kokoro를 설치하고, 같은 문장을 음성 `af_bella`과 `am_adam`로 합성하세요. 오디오 지속 시간과 주관적 품질을 비교해 보세요.
3. **어려움.** 자신의 5초 참조 클립을 녹음하세요. F5-TTS를 사용하여 이를 클론하세요. 참조 클립과 클론된 출력 간의 SECS를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Phoneme (음소) | 소리 단위 | 추상적인 소리 클래스; 영어에는 39개(Arpabet)가 있습니다. |
| Duration predictor (지속 시간 예측기) | 각 음소가 얼마나 오래 지속되는지 | 비자기회귀(non-AR) 모델의 출력; 음소당 정수 프레임 수. |
| Vocoder (보코더) | Mel → 파형 | Mel 스펙트럼을 원시 샘플로 매핑하는 신경망. |
| HiFi-GAN | 표준 보코더 | GAN 기반; 2020–2024년 동안 지배적이었습니다. |
| MOS | 주관적 품질 | 인간 평가자로부터 얻은 1–5 평균 의견 점수. |
| SECS | 음성 클론 지표 | 목표 화자와 출력 화자의 임베딩 간 코사인 유사도. |
| F5-TTS | 2024 오픈소스 SOTA | 흐름 매칭(flow-matching) 확산 모델; 제로샷 클로닝. |
| Kokoro | CPU 영어 리더 | 82M 매개변수 모델, Apache 2.0 라이선스. |

## 추가 읽기

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 기준선(baseline).
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 엔드투엔드(end-to-end) 흐름 기반 모델.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 현재 오픈소스 SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026년에도 여전히 배포되는 보코더.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024년 CPU 친화적 영어 TTS.
