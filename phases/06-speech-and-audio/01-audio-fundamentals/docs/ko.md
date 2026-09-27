# 오디오 기초 — 파형, 샘플링, 푸리에 변환 (Audio Fundamentals — Waveforms, Sampling, Fourier Transform)

> 파형(Waveforms)은 가공되지 않은 신호입니다. 스펙트로그램(Spectrograms)은 그 표현 방식입니다. 멜 특징량(Mel features)은 머신러닝에 최적화된 형태입니다. 모든 현대적인 ASR(자동 음성 인식) 및 TTS(음성 합성) 파이프라인은 이 단계를 거치며, 그 첫 번째 단계는 샘플링과 푸리에 변환을 이해하는 것입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## 문제점 (The Problem)

마이크는 시간에 따른 압력 신호(pressure-vs-time signal)를 생성합니다. 반면 여러분의 신경망은 텐서(tensors)를 소비합니다. 이 둘 사이에는 일련의 관례(conventions)가 존재하며, 이를 위반할 경우 소리 없는 버그가 발생합니다. 모델은 정상적으로 학습되는 것 같지만 WER(Word Error Rate)이 두 배로 뛰거나, TTS(Text-to-Speech)에서 쉬익 하는 잡음(hiss)이 발생하거나, 음성 복제(voice cloning) 시스템이 화자가 아닌 마이크의 특성을 학습해 버리는 식입니다.

음성 시스템에서 발생하는 모든 버그는 다음 세 가지 질문 중 하나로 귀결됩니다:

1. 데이터가 어떤 샘플링 레이트(sample rate)로 녹음되었으며, 모델이 기대하는 값은 무엇인가?
2. 신호에 에일리어싱(aliasing)이 발생했는가?
3. 원시 샘플(raw samples)을 처리하고 있는가, 아니면 주파수 표현(frequency representation)을 처리하고 있는가?

이 질문들에 올바른 답을 내놓는다면 6단계(Phase 6)의 나머지 과정은 수월하게 진행될 것입니다. 만약 이를 틀린다면, Whisper-Large-v4조차 쓰레기 같은 결과물(garbage)을 만들어낼 것입니다.

## 개념 (The Concept)

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**파형 (Waveform).** `[-1.0, 1.0]` 범위 내의 부동 소수점(float)으로 이루어진 1차원 배열입니다. 샘플 번호로 인덱싱됩니다. 초(seconds) 단위로 변환하려면 샘플 레이트로 나눕니다: `t = n / sr`. 16 kHz에서 10초 길이의 클립은 160,000개의 부동 소수점으로 구성된 배열입니다.

**샘플링 레이트 (Sampling rate, `sr`).** 초당 샘플의 개수입니다. 2026년 기준 일반적인 레이트는 다음과 같습니다:

| 레이트 | 용도 |
|------|-----|
| 8 kHz | 전화 통화, 레거시 VOIP. 4 kHz의 나이퀴스트(Nyquist) 한계로 인해 자음이 뭉개집니다. ASR 용도로는 피하세요. |
| 16 kHz | ASR 표준. Whisper, Parakeet, SeamlessM4T v2 모두 16 kHz를 사용합니다. |
| 22.05 kHz | 구형 모델의 TTS 보코더 학습용. |
| 24 kHz | 현대적 TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD 오디오, 음악. |
| 48 kHz | 영화, 프로 오디오, 고충실도(high-fidelity) TTS (VALL-E 2, NaturalSpeech 3). |

**나이퀴스트-섀넌 (Nyquist-Shannon).** `sr`의 샘플 레이트는 `sr/2`까지의 주파수를 모호함 없이 표현할 수 있습니다. `sr/2` 경계선을 *나이퀴스트 주파수(Nyquist frequency)*라고 합니다. 나이퀴스트를 초과하는 에너지는 *에일리어싱(aliased)*되어 낮은 주파수로 접혀 들어가며 신호를 손상시킵니다. 다운샘플링을 하기 전에는 항상 저역 통과 필터(low-pass filter)를 적용하세요.

**비트 심도 (Bit depth).** 16비트 PCM(signed int16, 범위 ±32,767)은 범용적인 교환 형식입니다. 음악에는 24비트가, 내부 DSP에는 32비트 부동 소수점이 사용됩니다. `soundfile`과 같은 라이브러리는 int16을 읽어오지만, 사용자에게는 `[-1, 1]` 범위의 float32 배열로 노출합니다.

**푸리에 변환 (Fourier Transform).** 모든 유한한 신호는 서로 다른 주파수를 가진 사인파(sinusoids)의 합입니다. 이산 푸리에 변환(Discrete Fourier Transform, DFT)은 `N`개의 샘플에 대해, 주파수 빈(frequency bin)당 하나씩 총 `N`개의 복소수 계수를 계산합니다. `bin k`는 `k · sr / N` Hz 주파수에 매핑됩니다. 크기(Magnitude)는 해당 주파수의 진폭이며, 각도(angle)는 위상(phase)입니다.

**FFT.** 고속 푸리에 변환(Fast Fourier Transform): `N`이 2의 거듭제곱일 때 DFT를 수행하는 `O(N log N)` 알고리즘입니다. 모든 오디오 라이브러리는 내부적으로 FFT를 사용합니다. 16 kHz에서 1024개 샘플의 FFT를 수행하면, 0–8 kHz 범위를 15.6 Hz 해상도로 커버하는 512개의 사용 가능한 주파수 빈을 얻을 수 있습니다.

**프레이밍 + 윈도우 (Framing + window).** 클립 전체를 한 번에 FFT 하지 않습니다. 클립을 겹치는 *프레임(frames)*(통상 25ms 길이, 10ms 홉(hop))으로 자른 뒤, 가장자리의 불연속성을 제거하기 위해 각 프레임에 윈도우 함수(Hann, Hamming)를 곱하고, 그 다음 각 프레임을 FFT 합니다. 이것이 단시간 푸리에 변환(Short-Time Fourier Transform, STFT)입니다. 02단계에서 이 내용을 이어서 다룹니다.

```figure
mel-scale
```

## 직접 구현해 보기 (Build It)

### 1단계: 클립을 읽고 파형(waveform)을 그리기

`code/main.py`는 데모의 의존성을 없애기 위해 표준 라이브러리인 `wave` 모듈만을 사용합니다. 실제 프로덕션 환경에서는 `soundfile` 또는 `torchaudio.load`를 사용하게 됩니다 (두 방식 모두 `(waveform, sr)` 튜플을 반환합니다):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 2단계: 기본 원리로부터 사인파(sine wave) 합성하기

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 샘플링 레이트에서 1초 동안 재생되는 440 Hz 사인파(concert A)는 16,000개의 부동 소수점(float) 데이터로 구성됩니다. `wave.open(..., "wb")`를 사용하여 16비트 PCM 인코딩 방식으로 작성해 보세요.

### 3단계: DFT를 수동으로 계산하기 (compute the DFT by hand)

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — `N=256` 정도의 크기에서 정확성을 확인하는 용도로는 괜찮지만, 실제 오디오 데이터에는 쓸모가 없습니다. 실제 코드에서는 `numpy.fft.rfft` 또는 `torch.fft.rfft`를 호출합니다.

### 4단계: 지배적 주파수(dominant frequency) 찾기

크기 피크 인덱스(Magnitude peak index) `k_star`는 주파수 `k_star * sr / N`에 매핑됩니다. 이를 440 Hz 사인파에 실행하면 `440 * N / sr` 빈(bin)에서 피크가 반환되어야 합니다.

### 5단계: 에일리어싱(Aliasing) 시연

10 kHz 샘플링 레이트에서 7 kHz 사인파를 샘플링해 보세요 (나이퀴스트(Nyquist) 주파수 = 5 kHz). 7 kHz 톤은 나이퀴스트 주파수보다 높으므로 `10 − 7 = 3 kHz`로 폴딩(fold)됩니다. FFT 피크는 3 kHz에서 나타납니다. 이것이 전형적인 에일리어싱 시연이며, 모든 DAC/ADC에 브릭월 저역 통과 필터(brick-wall low-pass filter)가 포함되어 출하되는 이유입니다.

## 사용하기 (Use It)

2026년에 실제로 배포하게 될 기술 스택입니다:

| 작업 | 라이브러리 | 이유 |
|------|---------|-----|
| WAV/FLAC/OGG 읽기/쓰기 | `soundfile` (libsndfile 래퍼) | 가장 빠르고 안정적이며, `float32`를 반환합니다. |
| 리샘플링 (Resample) | `torchaudio.transforms.Resample` 또는 `librosa.resample` | 올바른 안티앨리어싱(anti-aliasing) 기능이 내장되어 있습니다. |
| STFT / Mel | `torchaudio` 또는 `librosa` | GPU 친화적이며 PyTorch 생태계와 잘 맞습니다. |
| 실시간 스트리밍 | `sounddevice` 또는 `pyaudio` | 크로스 플랫폼 PortAudio 바인딩을 제공합니다. |
| 파일 검사 | `ffprobe` 또는 `soxi` | CLI 기반으로 빠르며 샘플 레이트/채널/코덱을 보고합니다. |

결정 규칙: **다른 무엇보다 먼저 샘플 레이트(sample rate)를 맞추세요**. Whisper는 16 kHz 모노 `float32`를 기대합니다. 44.1 kHz 스테레오 파일을 입력하면 모델 버그처럼 보이는 쓰레기 값(garbage)을 얻게 될 것입니다.

## Ship It (실전 적용)

`outputs/skill-audio-loader.md`로 저장하세요. 이 스킬은 오디오 입력이 다운스트림 모델의 기대치와 일치하는지 확인하고, 일치하지 않을 경우 올바르게 리샘플링(resampling)되도록 도와줍니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 220 Hz + 440 Hz + 880 Hz를 16 kHz 샘플링 레이트로 1초 동안 합성하여 믹스를 만드세요. DFT를 실행하고, 예상되는 빈(bin)에서 세 개의 피크가 나타나는지 확인하세요.
2. **중간 (Medium).** 자신의 목소리를 48 kHz로 3초 동안 WAV 파일로 녹음하세요. `torchaudio.transforms.Resample`(안티앨리어싱 포함)을 사용하여 16 kHz로 다운샘플링한 결과와, 단순 데시메이션(naive decimation, 매 세 번째 샘플 선택)을 사용하여 16 kHz로 다운샘플링한 결과를 각각 만드세요. 두 결과 모두 FFT를 수행하세요. 앨리어싱(aliasing)은 어디에서 나타나나요?
3. **어려움 (Hard).** `math` 모듈과 3단계에서 구현한 DFT만을 사용하여 STFT를 처음부터 직접 구현해 보세요. 프레임 크기(frame size) 400, 홉 크기(hop size) 160, Hann window를 사용합니다. `matplotlib.pyplot.imshow`를 사용하여 크기(magnitude)를 시각화하세요. 이것이 바로 Lesson 02의 스펙트로그램(spectrogram)입니다.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 샘플링 레이트 (Sample rate) | 초당 샘플 수 | ADC가 신호를 측정하는 Hz 단위의 주파수. |
| 나이퀴스트 (Nyquist) | 표현 가능한 최대 주파수 | `sr/2`; 이보다 높은 에너지는 에일리어싱(aliasing)되어 낮게 나타남. |
| 비트 심도 (Bit depth) | 각 샘플의 해상도 | `int16` = 65,536 단계; `float32` = `[-1, 1]` 범위 내 24비트 정밀도. |
| DFT | 시퀀스를 위한 푸리에 변환 | `N`개 샘플 → `N`개의 복소 주파수 계수. |
| FFT | 빠른 DFT | `N`이 2의 거듭제곱이어야 하는 `O(N log N)` 알고리즘. |
| 빈 (Bin) | 주파수 열 | `k · sr / N` Hz; 해상도 = `sr / N`. |
| STFT | 스펙트로그램의 내부 원리 | 시간에 따라 프레임화(Framed) 및 윈도잉(windowed)된 FFT. |
| 에일리어싱 (Aliasing) | 이상한 주파수 유령 현상 | 나이퀴스트 이상의 에너지가 낮은 빈(bin)으로 거울처럼 반사되는 현상. |

## 추가 읽을거리 (Further Reading)

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 샘플링 정리(sampling theorem)의 근간이 되는 논문입니다.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 무료로 제공되는 표준적인 DSP 교과서입니다.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 코드를 활용한 실무적인 가이드입니다.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 실제 오디오가 왜 깨끗한 사인파(sinusoid)가 아닌지에 대한 참고 문헌입니다.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10분 만에 주파수 빈(frequency bin)에 대한 직관을 명확히 해줍니다.
