# 스펙트로그램, 멜 스케일 및 오디오 특징 (Spectrograms, Mel Scale & Audio Features)

> 신경망은 가공되지 않은 원시 파형(raw waveforms)을 잘 처리하지 못합니다. 대신 스펙트로그램(spectrograms)을 입력으로 사용하며, 특히 멜 스펙트로그램(mel spectrograms)을 사용할 때 훨씬 더 효과적입니다. 2026년의 모든 ASR(자동 음성 인식), TTS(텍스트 음성 변환) 및 오디오 분류기의 성패는 바로 이 단 하나의 전처리 선택에 의해 결정됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## 문제점 (The Problem)

10초 길이의 16 kHz 클립을 예로 들어보겠습니다. 이는 160,000개의 부동 소수점(float) 데이터이며, 모두 `[-1, 1]` 범위 내에 있습니다. 이 데이터는 "개 짖는 소리"나 "cat이라는 단어"라는 레이블과 거의 완벽하게 상관관계가 없습니다. 원시 파형(raw waveform)에는 정보가 들어있지만, 모델이 쉽게 추출할 수 없는 형태입니다. 100ms 간격으로 발음된 동일한 음소라도 원시 샘플 값은 완전히 다를 수 있습니다.

스펙트로그램(Spectrogram)은 이 문제를 해결합니다. 스펙트로그램은 인간의 지각이 무시하는 시간적 세부 사항(마이크로초 단위의 지터)은 압축하고, 지각이 집중하는 구조(약 10~25ms의 시간 창 동안 어떤 주파수가 에너지를 갖는지)는 보존합니다.

멜 스펙트로그램(Mel spectrograms)은 여기서 한 단계 더 나아갑니다. 인간은 음높이(pitch)를 로그 스케일로 인지합니다. 즉, 100 Hz와 200 Hz의 차이는 1000 Hz와 2000 Hz의 차이와 "동일한 거리"로 들립니다. 멜 스케일(mel scale)은 이 인지 방식에 맞춰 주파수 축을 왜곡합니다. 멜 스케일 스펙트로그램은 2010년부터 2026년까지 음성 머신러닝(speech ML) 분야에서 가장 중요한 단일 특징(feature)입니다.

## 개념 (The Concept)

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** 파형을 중첩되는 프레임 단위로 자릅니다 (일반적 설정: 25ms 윈도우, 10ms 홉 = 16kHz 기준 400개 / 160개 샘플). 각 프레임에 윈도우 함수를 곱합니다 (Hann이 기본값이며, Hamming은 약간 다른 트레이드오프를 가집니다). 각 프레임에 FFT를 수행합니다. 크기 스펙트럼(magnitude spectra)을 `(n_frames, n_freq_bins)` 형태의 행렬로 쌓습니다. 이것이 바로 스펙트로그램(spectrogram)입니다.

**로그 크기 (Log-magnitude).** 가공되지 않은 크기 값은 5~6자릿수(orders of magnitude)에 걸쳐 분포합니다. 동적 범위(dynamic range)를 압축하기 위해 `log(|X| + 1e-6)` 또는 `20 * log10(|X|)`를 취합니다. 모든 프로덕션 파이프라인은 가공되지 않은 크기가 아닌 로그 크기를 사용합니다.

**멜 스케일 (Mel scale).** Hz 단위의 주파수 `f`는 `m = 2595 * log10(1 + f / 700)` 식에 의해 멜(mel) 단위 `m`으로 매핑됩니다. 이 매핑은 1kHz 미만에서는 대략 선형적이고, 그 이상에서는 대략 로그적입니다. 0~8kHz 범위를 커버하는 80개의 멜 빈(mel bins)이 표준 ASR 입력입니다.

**멜 필터뱅크 (Mel filterbank).** 멜 스케일 상에서 동일한 간격으로 배치된 삼각형 필터 세트입니다. 각 필터는 인접한 FFT 빈들의 가중치 합입니다. STFT 크기에 필터뱅크 행렬을 곱하면 단 한 번의 행렬 곱셈(matmul)으로 멜 스펙트로그램을 얻을 수 있습니다.

**로그-멜 스펙트로그램 (Log-mel spectrogram).** `log(mel_spec + 1e-10)`입니다. Whisper, Parakeet, SeamlessM4T의 입력값입니다. 2026년의 범용 오디오 프론트엔드입니다.

**MFCCs.** 로그-멜 스펙트로그램을 구한 뒤, DCT(type II)를 적용하고 처음 13개의 계수만 남깁니다. 이는 특징(features) 간의 상관관계를 제거(decorrelate)하고 더욱 압축합니다. 가공되지 않은 로그-멜 스펙트로그램을 사용하는 CNN/Transformer 모델들이 발전하기 전인 2015년경까지 지배적인 특징이었으며, 현재도 화자 인식(x-vectors, ECAPA) 분야에서 사용됩니다.

**해상도 트레이드오프 (Resolution trade).** FFT 크기가 커지면 주파수 해상도는 좋아지지만 시간 해상도는 나빠집니다. 25ms / 10ms는 오디오-ML의 기본값이며, 음악의 경우 50ms / 12.5ms, 과도 응답 탐지(transient detection, 드럼 타격음이나 파열음 등)의 경우 5ms / 2ms를 사용합니다.

```figure
spectrogram-window
```

## 직접 구현해 보기 (Build It)

### 1단계: 파형 프레임화(frame the waveform)

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160` 설정을 적용한 10초 길이의 16 kHz 클립은 998개의 프레임을 생성합니다.

### 2단계: Hann window (한 윈도우)

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT를 수행하기 전에 요소별(element-wise)로 곱해줍니다. 0이 아닌 종단점에서 절단(truncating)될 때 발생하는 스펙트럼 누설(spectral leakage)을 제거합니다.

### 3단계: STFT 크기(magnitude)

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

실제 서비스 환경에서는 `torch.stft` 또는 `librosa.stft`(FFT 기반, 벡터화 처리)를 사용합니다. 여기의 루프는 교육용이며, `code/main.py`에서 짧은 클립을 대상으로 실행됩니다.

### 4단계: mel filterbank (멜 필터뱅크)

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

0–8 kHz 범위를 커버하는 80 mels를 `n_fft=400`으로 설정하면 `(80, 201)` 행렬이 생성됩니다. `(n_frames, 201)` 크기의 STFT 크기(magnitude) 행렬에 이 행렬의 전치(transpose)를 곱하면 `(n_frames, 80)` 크기의 멜 스펙트로그램(mel spectrogram)을 얻을 수 있습니다.

### 5단계: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

일반적인 대안: `librosa.power_to_db` (참조값으로 정규화된 dB), `10 * log10(power + eps)`. Whisper는 더 복잡한 clip + normalize 루틴을 사용합니다 (Whisper의 `log_mel_spectrogram`을 참조하세요).

### 6단계: MFCCs (Mel-frequency cepstral coefficients)

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

각 로그-멜 프레임(log-mel frame)에 DCT를 적용하고, 처음 13개의 계수(coefficients)를 유지하세요. 이것이 여러분의 MFCC 행렬이 됩니다. 첫 번째 계수는 보통 전체 에너지(overall energy)를 인코딩하므로 제거하기도 합니다.

## 활용하기 (Use It)

2026년 스택:

| 작업 (Task) | 특징 (Features) |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS 음향 모델 (VITS, F5-TTS, Kokoro) | 80 mels, 정밀한 시간 제어를 위한 5–12 ms hop |
| 오디오 분류 (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| 화자 임베딩 (ECAPA-TDNN, WavLM) | 80 log-mels 또는 raw-waveform SSL |
| 음악 (MusicGen, Stable Audio 2) | EnCodec 이산 토큰 (mels 아님) |
| 키워드 검출 (Keyword spotting) | 초소형 기기를 위한 40 MFCCs |

경험 법칙: **음악 작업을 하는 것이 아니라면, 80 log-mels로 시작하세요.** 다른 설정을 사용하려면 그에 합당한 근거를 제시해야 합니다.

## 2026년에도 여전히 발생하는 실수들 (Pitfalls that still ship in 2026)

- **Mel 개수 불일치 (Mel count mismatch).** 80개의 mel을 사용하여 학습하고, 추론 시에는 128개의 mel을 사용하는 경우입니다. 이는 소리 없이 실패(Silent failure)를 일으킵니다. 양 끝단에서 피처 형상(feature shape)을 로그로 남기세요.
- **상류 단계의 샘플링 레이트 불일치 (Sample-rate mismatch upstream).** 22.05 kHz에서 계산된 mel은 16 kHz와 다르게 보입니다. 피처 추출(featurization)을 하기 *전*에 샘플링 레이트(SR)를 수정하세요.
- **dB vs log.** Whisper는 dB-mel이 아닌 log-mel을 기대합니다. 일부 Hugging Face(HF) 파이프라인은 이를 자동 감지하지만, 직접 작성한 코드는 그렇지 않습니다.
- **정규화 드리프트 (Normalization drift).** 학습 시에는 발화별 정규화(Per-utterance normalization)를 수행하고, 추론 시에는 전역 정규화(Global normalization)를 사용하는 경우입니다. 이는 WER(Word Error Rate)을 두 배로 만드는 운영 환경의 버그를 초래합니다.
- **패딩으로 인한 누수 (Leakage from padding).** 클립의 끝에 제로 패딩(Zero-padding)을 하면 마지막 프레임들에서 평탄한 스펙트럼(flat spectrum)이 생성됩니다. 대칭적으로 패딩하거나 복제(replicate) 방식을 사용하세요.

## Ship It (실행하기)

`outputs/skill-feature-extractor.md`로 저장하세요. 이 스킬은 주어진 모델 타겟에 대해 피처 유형(feature type), 멜 개수(mel count), 프레임/홉(frame/hop) 및 정규화(normalization)를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 이 코드는 처프 신호(chirp, 주파수가 200 → 4000 Hz로 스윕됨)를 합성하고 프레임당 `argmax` mel bin을 출력합니다. 그래프를 그려(선택 사항) 스윕 결과와 일치하는지 확인해 보세요.
2. **중간 (Medium).** `n_mels`를 `{40, 80, 128}`로, `frame_len`을 `{200, 400, 800}`으로 설정하여 다시 실행해 보세요. 시간축에 따른 날카로운 피크 대역폭(sharp-peak bandwidth)을 측정해 보세요. 어떤 조합이 처프 신호를 가장 잘 분해(resolve)하나요?
3. **어려움 (Hard).** `power_to_db`를 직접 구현하고, AudioMNIST 데이터셋에서 아주 작은 CNN 분류기를 사용하여 다음 세 가지 방식의 ASR 정확도를 비교해 보세요: (a) raw log-mel, (b) `ref=max`를 적용한 dB-mel, (c) MFCC-13 + delta + delta-delta. Top-1 정확도를 보고해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Frame (프레임) | 슬라이스 (A slice) | 하나의 FFT에 입력되는 25ms 길이의 파형 조각. |
| Hop (홉) | 스트라이드 (Stride) | 연속된 프레임 사이의 샘플 간격; ASR의 기본값은 10ms입니다. |
| Window (윈도우) | Hann/Hamming 함수 | 프레임의 가장자리를 0으로 서서히 줄여주는 점별 곱셈기(Point-wise multiplier). |
| STFT | 스펙트로그램 생성기 | 프레임화 및 윈도우 처리가 적용된 FFT; 시간 × 주파수 행렬을 생성합니다. |
| Mel (멜) | 왜곡된 주파수 (Warped frequency) | 로그 인지 스케일; `m = 2595·log10(1 + f/700)`. |
| Filterbank (필터뱅크) | 행렬 (The matrix) | STFT를 mel 빈(bin)으로 투영하는 삼각형 필터들. |
| Log-mel (로그 멜) | Whisper의 입력값 | `log(mel_spec + eps)`; 2026년에 표준화되었습니다. |
| MFCC | 전통적인 특징량 (Old-school feature) | log-mel의 DCT; 13개의 계수를 가지며 상관관계가 제거(decorrelated)되어 있습니다. |

## 추가 읽을거리 (Further Reading)

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 관련 논문입니다.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 최초의 멜 스케일(mel scale) 논문입니다.
- [OpenAI — Whisper 소스 코드, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 참조 구현(reference implementation)을 읽어 보세요.
- [librosa 특징 추출 문서(feature extraction docs)](https://librosa.org/doc/main/feature.html) — `mfcc`, `melspectrogram`, 그리고 hop/window에 대한 참조 자료입니다.
- [NVIDIA NeMo — 오디오 전처리(audio preprocessing)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 모델을 위한 프로덕션 규모의 파이프라인입니다.
