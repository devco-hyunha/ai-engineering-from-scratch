# 오디오 기초 — 파형, 샘플링, 푸리에 변환

> 파형은 원시 신호입니다. 스펙트로그램은 표현입니다. Mel 특징은 ML 친화적인 형태입니다. 모든 최신 ASR 및 TTS 파이프라인은 이 사다리를 따라 올라가며, 첫 번째 칸은 샘플링과 푸리에 변환을 이해하는 것입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 1단계 · 06강 (벡터 및 행렬), 1단계 · 14강 (확률 분포)
**시간:** 약 45분

## 문제점

마이크는 압력 대 시간 신호를 생성합니다. 신경망은 텐서를 소비합니다. 그 사이에는 관습의 스택이 있으며, 이를 위반하면 조용한 버그가 발생합니다: 모델은 잘 학습되지만 WER가 두 배가 되거나, TTS가 히스 소리를 내보내거나, 음성 클로닝 시스템이 화자 대신 마이크를 기억합니다.

음성 시스템의 모든 버그는 세 가지 질문 중 하나로 거슬러 올라갑니다:

1. 데이터가 어떤 샘플 레이트로 기록되었으며 모델이 무엇을 기대합니까?
2. 신호가 에일리어싱되었습니까?
3. 원시 샘플에서 작업하고 있습니까, 아니면 주파수 표현에서 작업하고 있습니까?

이것들을 정확히 처리하면 6단계의 나머지 부분은 다루기 쉬워집니다. 이것들을 잘못하면 Whisper-Large-v4조차도 쓰레기를 생성합니다.

## 개념

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**파형.** `[-1.0, 1.0]`의 1차원 부동 소수점 배열입니다. 샘플 번호로 인덱싱됩니다. 초 단위로 변환하려면 샘플 레이트로 나누세요: `t = n / sr`. 16 kHz의 10초 클립은 160,000개의 부동 소수점 배열입니다.

**샘플 레이트(sr).** 초당 샘플 수입니다. 2026년 일반적인 레이트는 다음과 같습니다:

| 레이트 | 용도 |
|------|-----|
| 8 kHz | 전화 통신, 레거시 VOIP. 4 kHz의 나이퀴스트는 자음을 죽입니다. ASR에는 피하세요. |
| 16 kHz | ASR 표준. Whisper, Parakeet, SeamlessM4T v2 모두 16 kHz를 소비합니다. |
| 22.05 kHz | 구형 모델의 TTS 보코더 학습. |
| 24 kHz | 최신 TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD 오디오, 음악. |
| 48 kHz | 영화, 프로 오디오, 고충실도 TTS (VALL-E 2, NaturalSpeech 3). |

**나이퀴스트-섀넌.** `sr`의 샘플링 속도는 `sr/2`까지의 주파수를 모호함 없이 표현할 수 있습니다. `sr/2` 경계는 *나이퀴스트 주파수*입니다. 나이퀴스트 주파수 이상의 에너지는 *에이리어싱*되어 — 낮은 주파수로 접혀 내려가며 — 신호를 오염시킵니다. 다운샘플링 전에는 항상 저역통과 필터를 적용하세요.

**비트 깊이.** 16비트 PCM (부호 있는 int16, 범위 ±32,767)은 범용 교환 형식입니다. 음악에는 24비트, 내부 DSP에는 32비트 부동소수점을 사용합니다. `soundfile`와 같은 라이브러리는 int16을 읽지만 `[-1, 1]`에서는 float32 배열을 노출합니다.

**푸리에 변환.** 모든 유한한 신호는 서로 다른 주파수의 사인파의 합입니다. 이산 푸리에 변환(DFT)은 `N` 샘플에 대해 `N`개의 복소 계수를 계산합니다 — 주파수 빈(bin)마다 하나씩. `bin k`는 `k · sr / N` Hz 주파수에 대응합니다. 크기는 해당 주파수의 진폭이며, 각도는 위상입니다.

**FFT.** 고속 푸리에 변환: `N`이 2의 거듭제곱일 때 DFT를 계산하는 `O(N log N)` 알고리즘입니다. 모든 오디오 라이브러리는 내부적으로 FFT를 사용합니다. 16 kHz에서 1024 샘플 FFT를 수행하면 0–8 kHz 범위를 15.6 Hz 해상도로 다루는 512개의 사용 가능한 주파수 빈(bin)이 생성됩니다.

**프레이밍 + 윈도우.** 전체 클립을 FFT하지 않습니다. 클립을 겹치는 *프레임*으로 잘라 (일반적으로 25 ms 길이, 10 ms 호프) 각 프레임에 윈도우 함수(Hann, Hamming)를 곱하여 가장자리 불연속성을 제거한 후, 각 프레임을 FFT합니다. 이것이 단기 푸리에 변환(STFT)입니다. 02강에서 이 내용을 이어갑니다.

```figure
mel-scale
```

## 구현하기

### 1단계: 클립을 읽고 파형 플롯하기

`code/main.py`는 데모의 의존성을 없애기 위해 표준 라이브러리 `wave` 모듈만 사용합니다. 프로덕션에서는 `soundfile` 또는 `torchaudio.load`를 사용하게 될 것입니다 (둘 다 `(waveform, sr)` 튜플을 반환합니다):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 2단계: 첫 원리로부터 사인파 합성하기

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz에서 1초 동안의 440 Hz 사인파 (콘서트 A)는 16,000개의 부동소수점 값입니다. 16비트 PCM 인코딩을 사용하여 `wave.open(..., "wb")`로 작성하세요.

### 3단계: DFT를 직접 계산하기

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

`O(N²)` — `N=256`에 대해 정확성을 확인하는 데는 적합하지만, 실제 오디오에는 쓸모가 없습니다. 실제 코드는 `numpy.fft.rfft` 또는 `torch.fft.rfft`를 호출합니다.

### 4단계: 지배 주파수 찾기

크기 피크 인덱스 `k_star`는 `k_star * sr / N` 주파수에 대응합니다. 440 Hz 사인파에 대해 이 코드를 실행하면 빈(bin) `440 * N / sr`에서 피크가 반환되어야 합니다.

### 5단계: 에이리어싱 시연

10 kHz로 7 kHz 사인파를 샘플링하세요 (나이퀴스트 = 5 kHz). 7 kHz 톤은 나이퀴스트 주파수보다 높으며 `10 − 7 = 3 kHz`으로 접힙니다. FFT 피크는 3 kHz에 나타납니다. 이는 전형적인 에이리어싱 시연이며, 모든 DAC/ADC가 벽돌벽(low-pass) 필터를 포함하는 이유입니다.

## 사용하기

2026년에 실제로 출시할 스택입니다:

| 작업 | 라이브러리 | 이유 |
|------|---------|-----|
| WAV/FLAC/OGG 읽기/쓰기 | `soundfile` (libsndfile 래퍼) | 가장 빠르고 안정적이며, float32를 반환합니다. |
| 리샘플링 | `torchaudio.transforms.Resample` 또는 `librosa.resample` | 올바른 안티 에이리어싱이 내장되어 있습니다. |
| STFT / Mel | `torchaudio` 또는 `librosa` | GPU 친화적; PyTorch 생태계. |
| 실시간 스트리밍 | `sounddevice` 또는 `pyaudio` | 크로스 플랫폼 PortAudio 바인딩. |
| 파일 검사 | `ffprobe` 또는 `soxi` | CLI, 빠르며, sr/channels/codec를 보고합니다. |

결정 규칙: **다른 모든 것을 맞추기 전에 샘플링 레이트를 먼저 맞추세요**. Whisper는 16 kHz 모노 float32를 기대합니다. 44.1 kHz 스테레오를 넘기면 모델 버그처럼 보이는 엉뚱한 결과를 얻게 됩니다.

## 출시하기

`outputs/skill-audio-loader.md`으로 저장하세요. 이 스킬은 오디오 입력이 다운스트림 모델의 기대치와 일치하는지 확인하고, 일치하지 않을 경우 올바르게 리샘플링하도록 돕습니다.

## 연습 문제

1. **쉬움.** 16 kHz에서 220 Hz + 440 Hz + 880 Hz의 1초 믹스를 합성하세요. DFT를 실행하세요. 예상된 빈(bin)에서 세 개의 피크가 확인되는지 확인하세요.
2. **중간.** 48 kHz로 자신의 목소리를 3초 WAV로 녹음하세요. `torchaudio.transforms.Resample`를 사용하여 안티 에이리어싱을 적용해 16 kHz로 다운샘플링한 후, 단순 디메이션(세 번째 샘플마다)으로 16 kHz로 변환하세요. 두 경우 모두 FFT를 수행하세요. 에이리어싱이 어디에 나타나나요?
3. **어려움.** `math`와 3단계의 DFT만 사용하여 STFT를 처음부터 구축하세요. 프레임 크기 400, 호프 160, Hann 창을 사용하세요. `matplotlib.pyplot.imshow`로 크기를 플롯하세요. 이는 02강의 스펙트로그램입니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 샘플링 레이트 | 초당 샘플 수 | ADC가 신호를 측정하는 주파수(Hz). |
| 나이퀴스트 | 표현할 수 있는 최대 주파수 | `sr/2`; 그 이상의 에너지는 아래로 접힙니다. |
| 비트 깊이 | 각 샘플의 해상도 | `int16` = 65,536 단계; `[-1, 1]`에서 `float32` = 24비트 정밀도. |
| DFT | 시퀀스에 대한 푸리에 변환 | `N` 샘플 → `N` 복소 주파수 계수. |
| FFT | 빠른 DFT | `N` = 2의 거듭제곱이 필요한 `O(N log N)` 알고리즘. |
| 빈 | 주파수 열 | `k · sr / N` Hz; 해상도 = `sr / N`. |
| STFT | 스펙트로그램의 내부 원리 | 시간에 따라 프레임을 나누고 창(window)을 적용한 FFT. |
| 에이리어싱 | 이상한 주파수 유령 | 나이퀴스트 이상의 에너지가 낮은 빈으로 반사되어 나타남. |

## 추가 읽기

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 샘플링 정리의 원 논문.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 무료, 표준 DSP 교과서.
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html) — 코드를 포함한 실전 가이드.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff) — 실제 오디오가 순수한 사인파가 아닌 이유에 대한 참고 자료.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 주파수 빈 직관을 10분 만에 정리한 자료.
