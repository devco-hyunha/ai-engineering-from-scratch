# 스펙트로그램, 멜 스케일 및 오디오 특징

> 신경망은 원시 파형(waveform)을 잘 소비하지 못합니다. 스펙트로그램을 소비합니다. 멜 스펙트로그램은 더 잘 소비합니다. 2026년의 모든 ASR, TTS, 오디오 분류기는 이 단일 전처리 선택에 따라 성패가 결정됩니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 01강 (오디오 기초)
**시간:** 약 45분

## 문제점

10초 길이의 16 kHz 클립을 생각해 보세요. 이는 `[-1, 1]`에 있는 160,000개의 부동소수점이며, "개 짖는 소리"나 "고양이 단어"라는 레이블과 거의 완벽하게 무상관입니다. 원시 파형에는 정보가 담겨 있지만, 모델이 쉽게 추출할 수 없는 형태입니다. 100 ms 간격으로 발화된 동일한 음소 두 개는 완전히 다른 원시 샘플을 가집니다.

스펙트로그램은 이 문제를 해결합니다. 인간 지각이 무시하는 시간적 세부 사항(마이크로초 단위 지터)을 축약하고, 지각이 주의하는 구조(~10–25 ms 시간 윈도우에 걸쳐 어떤 주파수가 에너지적인지)를 보존합니다.

멜 스펙트로그램은 한 발 더 나아갑니다. 인간은 음높이를 로그적으로 지각합니다: 100 Hz 대 200 Hz는 1000 Hz 대 2000 Hz와 "같은 거리 간격"으로 들립니다. 멜 스케일은 주파수 축을 왜곡하여 이에 맞춰줍니다. 멜 스케일 스펙트로그램은 2010년부터 2026년까지 음성 ML에서 가장 중요한 단일 특징입니다.

## 개념

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** 파형을 겹치는 프레임으로 슬라이스합니다 (일반적으로: 25 ms 윈도우, 10 ms hop = 16 kHz에서 400 샘플 / 160 샘플). 각 프레임에 윈도우 함수를 곱합니다 (Hann이 기본값; Hamming은 약간 다른 트레이드오프). 각 프레임에 FFT를 수행합니다. 크기 스펙트럼을 `(n_frames, n_freq_bins)` 형태의 행렬로 쌓습니다. 이것이 스펙트로그램입니다.

**로그 크기.** 원시 크기는 5-6 자릿수의 범위를 가집니다. 동적 범위를 압축하기 위해 `log(|X| + 1e-6)` 또는 `20 * log10(|X|)`를 취합니다. 모든 프로덕션 파이프라인은 원시 크기가 아닌 로그 크기를 사용합니다.

**멜 스케일.** Hz 단위의 주파수 `f`는 `m = 2595 * log10(1 + f / 700)`에 의해 mel `m`로 매핑됩니다. 이 매핑은 1 kHz 미만에서는 대략 선형이고 그 이상에서는 대략 로그적입니다. 0–8 kHz를 커버하는 80 mel bins가 표준 ASR 입력입니다.

**멜 필터 뱅크.** 멜 스케일에서 균등하게 간격이 배치된 삼각형 필터 집합입니다. 각 필터는 인접한 FFT 빈(bin)의 가중 합입니다. STFT 크기에 필터 뱅크 행렬을 곱하면 하나의 행렬 곱 연산(matmul)으로 멜 스펙트로그램을 얻을 수 있습니다.

**로그-멜 스펙트로그램.** `log(mel_spec + 1e-10)`. Whisper의 입력입니다. Parakeet의 입력입니다. SeamlessM4T의 입력입니다. 2026년 오디오 프론트엔드의 표준입니다.

**MFCC.** 로그-멜 스펙트로그램에 DCT (type II)를 적용하고 첫 13개 계수를 유지합니다. 특징을 비상관화(decorrelate)하고 더 압축합니다. CNN/트랜스포머가 원시 로그-멜을 처리하는 방식이 따라잡기 전인 약 2015년까지 지배적인 특징이었습니다. 화자 인식(x-vector, ECAPA)에서는 여전히 사용되고 있습니다.

**해상도 트레이드오프.** 더 큰 FFT는 더 좋은 주파수 해상도를 제공하지만 시간 해상도는 더 나빠집니다. 오디오 ML의 기본값은 25 ms / 10 ms이며, 음악은 50 ms / 12.5 ms, 트랜지언트(transient) 감지(드럼 히트, 파열음)는 5 ms / 2 ms를 사용합니다.

```figure
spectrogram-window
```

## 구현하기

### 1단계: 파형 프레임화

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10초 길이의 16 kHz 클립을 `frame_len=400, hop=160`로 처리하면 998개의 프레임이 생성됩니다.

### 2단계: Hann 창

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 전에 요소별로 곱합니다. 비-0 끝점에서 잘림(truncation)으로 인해 발생하는 스펙트럼 누출(spectral leakage)을 제거합니다.

### 3단계: STFT 크기

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

프로덕션에서는 `torch.stft` 또는 `librosa.stft` (FFT 기반, 벡터화)를 사용합니다. 여기의 루프는 교육적 목적이며 `code/main.py`에서 짧은 클립에 대해 실행됩니다.

### 4단계: 멜 필터 뱅크

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

0–8 kHz를 커버하는 80개의 멜을 `n_fft=400`로 사용하면 `(80, 201)` 행렬이 됩니다. `(n_frames, 201)` STFT 크기에 전치(transpose)를 곱하면 `(n_frames, 80)` 멜 스펙트로그램을 얻습니다.

### 5단계: 로그-멜

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

일반적인 대안: `librosa.power_to_db` (참조 정규화 dB), `10 * log10(power + eps)`. Whisper는 더 복잡한 클립 + 정규화 루틴을 사용합니다 (Whisper의 `log_mel_spectrogram` 참조).

### 6단계: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

각 로그-멜 프레임에 DCT를 적용하고 첫 13개 계수를 유지합니다. 이것이 MFCC 행렬입니다. 첫 번째 계수는 전체 에너지를 인코딩하므로 보통 버립니다.

## 사용하기

2026년 스택:

| 작업 | 특징 |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 로그-멜, 10 ms hop, 25 ms 창 |
| TTS 음향 모델 (VITS, F5-TTS, Kokoro) | 미세한 시간 제어용 80 mel, 5–12 ms hop |
| 오디오 분류 (AST, PANNs, BEATs) | 128 log-mel, 10 ms hop |
| 화자 임베딩 (ECAPA-TDNN, WavLM) | 80 log-mel 또는 원시 파형 SSL |
| 음악 (MusicGen, Stable Audio 2) | EnCodec 이산 토큰 (mel 아님) |
| 키워드 스팟팅 | 소형 디바이스용 40 MFCC |

경험칙: **음악 작업이 아니라면 80 log-mel로 시작하세요.** 모든 편차에 대한 입증 책임이 따릅니다.

## 2026년에도 여전히 출시되는 함정들

- **Mel 개수 불일치.** 학습은 80 mel로, 추론은 128 mel로 진행합니다. 조용한 실패입니다. 양쪽 끝에서 피처 형태를 로깅하세요.
- **상류 샘플 레이트 불일치.** 22.05 kHz에서 계산된 mel은 16 kHz와 다릅니다. 피처화 *전에* SR을 고정하세요.
- **dB vs log.** Whisper는 dB-mel이 아닌 log-mel을 기대합니다. 일부 HF 파이프라인은 자동 감지하지만, 사용자 정의 코드는 그렇지 않습니다.
- **정규화 드리프트.** 학습 중에는 발화별 정규화, 추론 중에는 전역 정규화를 사용합니다. WER를 두 배로 만드는 프로덕션 버그입니다.
- **패딩으로 인한 누수.** 클립 끝에 0 패딩을 하면 후미 프레임에서 평평한 스펙트럼이 생성됩니다. 대칭적으로 패딩하거나 복제하세요.

## 출시하기

`outputs/skill-feature-extractor.md`로 저장하세요. 이 스킬은 주어진 모델 목표에 대해 피처 유형, mel 개수, 프레임/hop, 정규화를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 치프(chirp, 주파수 200 → 4000 Hz 스윕)를 합성하고 프레임별 argmax mel bin을 출력합니다. (선택적으로) 플롯을 그려 스윕과 일치하는지 확인하세요.
2. **중간.** `{40, 80, 128}`에서 `n_mels`를, `{200, 400, 800}`에서 `frame_len`를 다시 실행하세요. 시간 축에 걸쳐 날카로운 피크의 대역폭을 측정하세요. 어떤 조합이 치프를 가장 잘 해결합니까?
3. **어려움.** `power_to_db`를 구현하고 AudioMNIST에서 소형 CNN 분류기의 ASR 정확도를 (a) 원시 log-mel, (b) `ref=max`를 사용한 dB-mel, (c) MFCC-13 + delta + delta-delta로 비교하세요. top-1 정확도를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 프레임 | 슬라이스 | 하나의 FFT에 입력되는 25 ms 파형 청크. |
| 호프 | 스트라이드 | 연속 프레임 사이의 샘플 수; 10 ms는 ASR 기본값입니다. |
| 윈도우 | Hann/Hamming 함수 | 프레임 가장자리를 0으로 테이퍼링하는 점별 곱셈 연산자입니다. |
| STFT | 스펙트로그램 생성기 | 프레임화 + 윈도우 적용 FFT; 시간 × 주파수 행렬을 생성합니다. |
| 멜 | 왜곡된 주파수 | 로그 지각 스케일; `m = 2595·log10(1 + f/700)`. |
| 필터뱅크 | 행렬 | STFT를 멜 빈(mel bins)에 투영하는 삼각형 필터입니다. |
| 로그-멜 | Whisper 입력 | `log(mel_spec + eps)`; 2026년에 표준화되었습니다. |
| MFCC | 전통적 특징 | 로그-멜의 DCT; 13개 계수, 비상관화(decorrelated)됨. |

## 추가 읽기

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 논문.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 원본 멜 스케일.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 참조 구현을 읽어 보세요.
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html) — `mfcc`, `melspectrogram` 및 호프/윈도우에 대한 참조입니다.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 모델을 위한 생산 규모 파이프라인입니다.
