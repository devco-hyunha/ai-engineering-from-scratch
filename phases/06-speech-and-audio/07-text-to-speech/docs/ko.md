# 텍스트 음성 변환 (Text-to-Speech, TTS) — Tacotron에서 F5 및 Kokoro까지

> ASR은 음성을 텍스트로 역변환하며, TTS는 텍스트를 음성으로 역변환합니다. 2026년의 기술 스택은 세 부분으로 구성됩니다: 텍스트 → 토큰(tokens), 토큰 → 멜(mel), 멜 → 파형(waveform). 각 단계는 노트북에서 구동 가능한 기본 모델을 가집니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제 (The Problem)

다음과 같은 문자열이 있습니다: "Please remind me to water the plants at 6 pm." 여러분은 실시간 음성 비서를 위해 자연스럽게 들리고, 올바른 운율(pauses, stress)을 가지며, "plants"의 모음을 정확하게 발음하고, CPU에서 300ms 이내에 실행되는 3초 길이의 오디오 클립이 필요합니다. 또한 목소리를 교체(voice swapping)해야 하고, 코드 스위칭(code-switched) 입력("remind me at 6 pm, daijoubu?")을 처리해야 하며, 고유 명사를 잘못 발음하는 실수를 범해서도 안 됩니다.

현대적인 TTS 파이프라인은 다음과 같은 구조를 가집니다:

1. **텍스트 프론트엔드 (Text frontend).** 텍스트 정규화(날짜, 숫자, 이메일), 음소(phonemes) 또는 서브워드 토큰(subword tokens)으로 변환, 운율 특징(prosody features) 예측.
2. **음향 모델 (Acoustic model).** 텍스트 → 멜 스펙트로그램(mel spectrogram). Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **보코더 (Vocoder).** 멜 → 파형(waveform). WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), 2024년 이후의 신경망 코덱 보코더(neural codec vocoders).

2026년에는 엔드투엔드 확산 모델(end-to-end diffusion) 및 플로우 매칭(flow-matching) 모델의 등장으로 음향 모델과 보코더의 경계가 모호해집니다. 하지만 디버깅을 위한 사고 모델(mental model)로서 이 세 부분으로 나누어 생각하는 방식은 여전히 유효합니다.

## 개념 (The Concept)

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq 방식: `char-embedding` → `BiLSTM encoder` → `location-sensitive attention` → 자기회귀(autoregressive) `LSTM decoder`가 mel 프레임을 생성합니다. 속도가 느리고(AR), 긴 문장에서 불안정한 모습을 보입니다. 여전히 베이스라인으로 자주 인용됩니다.

**FastSpeech 2 (2020).** 비자기회귀(Non-autoregressive) 방식입니다. `Duration predictor`가 각 음소(phoneme)에 할당될 mel 프레임의 수를 출력합니다. 단일 패스(1-pass)로 동작하며 Tacotron보다 10배 빠릅니다. 자연스러움이 다소 떨어질 수 있지만(단조로운 정렬 문제), 광범위하게 사용됩니다.

**VITS (2021).** 변분 추론(variational inference)을 사용하여 `encoder` + `flow-based duration` + `HiFi-GAN vocoder`를 엔드투엔드(end-to-end)로 공동 학습합니다. 고품질의 단일 모델입니다. 2022~2024년 사이 오픈소스 TTS 시장을 주도했습니다. 변형 모델로는 YourTTS(다중 화자 제로샷), XTTS v2(2024, Coqui) 등이 있습니다.

**F5-TTS (2024).** Flow matching 기반의 Diffusion transformer입니다. 자연스러운 운율(prosody)을 제공하며, 5초의 참조 오디오만으로 제로샷 음성 복제(zero-shot voice cloning)가 가능합니다. 2026년 오픈소스 TTS 리더보드 최상위에 위치하며, 파라미터 수는 335M입니다.

**Kokoro (2024).** 소형(82M) 모델로 CPU에서도 실행 가능하며, 실시간 사용을 위한 동급 최강의 영어 TTS입니다. 폐쇄형 어휘(closed-vocabulary) 기반의 영어 전용 모델이며, Apache-2.0 라이선스를 따릅니다.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** 상용 기술의 최첨단(state of the art) 모델들입니다. ElevenLabs v2.5의 감정 태그(예: "[whispered]", "[laughing]")와 캐릭터 음성은 2026년 오디오북 제작 시장을 주도하고 있습니다.

### 보코더의 진화 (Vocoder evolution)

| 시대 (Era) | 보코더 (Vocoder) | 지연 시간 (Latency) | 품질 (Quality) |
|-----|---------|---------|---------|
| 2016 | WaveNet | 오프라인 전용 | 출시 당시 SOTA |
| 2018 | WaveRNN | 실시간 수준 (~realtime) | 양호 |
| 2020 | HiFi-GAN | 실시간의 100배 | 인간에 근접 |
| 2022 | BigVGAN | 실시간의 50배 | 화자/언어 간 일반화 가능 |
| 2024 | SNAC, DAC (neural codecs) | AR 모델과 통합됨 | 이산 토큰(discrete tokens), 비트 효율적 |

2026년경에는 대부분의 "TTS" 모델이 텍스트에서 파형(waveform)까지 엔드투엔드(end-to-end)로 작동하며, 멜 스펙트로그램(mel spectrogram)은 내부 표현(internal representation)으로 사용됩니다.

### 평가 (Evaluation)

- **MOS (Mean Opinion Score).** 1–5 척도, 크라우드소싱 방식. 여전히 황금 표준(gold standard)이지만, 진행 속도가 매우 느립니다.
- **CMOS (Comparative MOS).** A 대 B 선호도 비교. 주석(annotation)당 신뢰 구간이 더 좁습니다.
- **UTMOS, DNSMOS.** 참조 데이터가 필요 없는(Reference-free) 신경망 기반 MOS 예측기입니다. 리더보드에서 주로 사용됩니다.
- **ASR을 통한 CER (Character Error Rate).** TTS 출력을 Whisper로 실행하여 입력 텍스트와 비교해 CER을 계산합니다. 명료도(intelligibility)의 대리 지표로 사용됩니다.
- **SECS (Speaker Embedding Cosine Similarity).** 음성 복제(voice-cloning) 품질을 측정합니다.

LibriTTS test-clean 기준 2026년 수치:

| 모델 (Model) | UTMOS | CER (via Whisper) | 크기 (Size) |
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

## 직접 구현해 보기 (Build It)

### 1단계: 입력값 음소화(phonemize)

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

음소(Phonemes)는 보편적인 가교 역할을 합니다. VITS 수준 미만의 품질을 가진 모델에는 가공되지 않은 텍스트(raw text)를 입력하지 않도록 주의하세요.

### 2단계: Kokoro 실행하기 (2026 CPU 기본 설정)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 미국식 영어
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 텐서, sr=24000
```

오프라인으로 실행되며, 단일 파일 구성, 82M 파라미터를 가집니다.

### 3단계: 음성 복제(Voice Cloning)를 활용한 F5-TTS 실행

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5초 분량의 참조 클립(reference clip)과 해당 전사 데이터(transcript)를 전달하세요. F5는 운율(prosody)과 음색(timbre)을 복제합니다.

### 4단계: HiFi-GAN 보코더(vocoder) 밑바닥부터 구현하기

튜토리얼 스크립트에 담기에는 너무 방대하지만, 전체적인 구조는 다음과 같습니다:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4개의 업샘플링 블록, mel-rate에서 audio-rate로 가기 위해 총 256배 업샘플링
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

학습: 적대적 학습(adversarial training, 짧은 윈도우에 대한 판별기 사용) + mel-spectrogram 재구성 손실(reconstruction loss) + 특징 매칭 손실(feature-matching loss). 범용화된 기술이므로 `hifi-gan` 저장소나 nvidia-NeMo의 사전 학습된 체크포인트를 사용하세요.

### 5단계: 전체 파이프라인 (pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 활용하기 (Use It)

2026년 스택:

| 상황 | 선택 (Pick) |
|-----------|------|
| 실시간 영어 음성 비서 | Kokoro (CPU) 또는 XTTS v2 (GPU) |
| 5초 참조 음성을 통한 보이스 클로닝 | F5-TTS |
| 상업용 캐릭터 목소리 | ElevenLabs v2.5 |
| 오디오북 낭독 | ElevenLabs v2.5 또는 XTTS v2 + 미세 조정(fine-tune) |
| 저자원 언어 (Low-resource language) | 5~20시간의 대상 언어 데이터로 VITS 학습 |
| 표현력 / 감정 태그 | ElevenLabs v2.5 또는 StyleTTS 2 미세 조정(fine-tune) |

2026년 기준 오픈 소스 리더: **품질은 F5-TTS, 효율성은 Kokoro**. 역사학자가 아니라면 Tacotron을 찾지 마세요.

## 주의 사항 (Pitfalls)

- **텍스트 정규화기(Text normalizer) 부재.** "Dr. Smith"를 "Doctor"로 읽어야 할까요, 아니면 "Drive"로 읽어야 할까요? "2026"은 "twenty twenty six"인가요, 아니면 "two zero two six"인가요? 음소 변환기(phonemizer)를 거치기 **전**에 정규화를 수행하세요.
- **OOV(Out-of-Vocabulary) 고유 명사.** "Ghumare"가 "ghyu-mair"로 읽혀야 한다면? 미등록 토큰을 위해 폴백(fallback)용 자소-음소 변환(grapheme-to-phoneme) 모델을 함께 배포하세요.
- **클리핑(Clipping).** 보코더(Vocoder) 출력에서 클리핑이 발생하는 경우는 드물지만, 추론 시 멜 스케일링(mel scaling) 불일치로 인해 $\pm1.0$ 범위를 초과할 수 있습니다. 항상 `np.clip(wav, -1, 1)`을 사용하세요.
- **샘플링 레이트 불일치(Sample-rate mismatch).** Kokoro는 24 kHz로 출력되지만, 다운스트림 파이프라인이 16 kHz를 기대한다면 리샘플링을 하거나 에일리어싱(aliasing)이 발생할 수 있습니다.

## Ship It (실행해 보기)

`outputs/skill-tts-designer.md`로 저장하세요. 주어진 목소리, 지연 시간(latency), 언어 목표에 맞는 TTS 파이프라인을 설계해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 토이 어휘집(toy vocab)으로부터 음소 사전(phoneme dictionary)을 구축하고, 음소당 지속 시간을 추정하며, 가상의 "mel" 스케줄을 출력합니다.
2. **중간 (Medium).** Kokoro를 설치하고, 동일한 문장을 `af_bella`와 `am_adam` 목소리로 합성해 보세요. 오디오 지속 시간과 주관적 품질을 비교해 보세요.
3. **어려움 (Hard).** 본인의 목소리를 5초 동안 녹음하여 레퍼런스 클립을 만드세요. F5-TTS를 사용하여 이를 클로닝(cloning)해 보세요. 레퍼런스와 클로닝된 출력물 사이의 SECS를 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| Phoneme (음소) | 소리 단위 | 추상적인 소리 클래스; 영어의 경우 39개(ARPABet). |
| Duration predictor (지속 시간 예측기) | 각 음소가 지속되는 시간 | Non-AR 모델의 출력값; 음소당 정수 프레임 단위. |
| Vocoder (보코더) | Mel → 파형 | Mel-spec을 원시 샘플(raw samples)로 매핑하는 신경망. |
| HiFi-GAN | 표준 보코더 | GAN 기반; 2020~2024년 사이 주류 모델. |
| MOS | 주관적 품질 | 인간 평가자가 매긴 1~5점 사이의 평균 의견 점수. |
| SECS | 음성 복제 지표 | 타겟 화자와 출력 화자의 임베딩 간 코사인 유사도. |
| F5-TTS | 2024년 오픈소스 SOTA | Flow-matching 확산 모델; 제로샷(zero-shot) 복제 지원. |
| Kokoro | CPU 영어 모델 선두주자 | 82M 파라미터 모델, Apache 2.0 라이선스. |

## 추가 읽을거리 (Further Reading)

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 베이스라인 모델입니다.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 엔드투엔드(end-to-end) 흐름 기반(flow-based) 모델입니다.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 현재 오픈 소스 SOTA(최첨단) 모델입니다.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026년에도 여전히 사용되는 보코더(vocoder)입니다.
- [HuggingFace의 Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — 2024년 기준 CPU 친화적인 영어 TTS 모델입니다.
