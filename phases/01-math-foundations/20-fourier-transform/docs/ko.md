# 푸리에 변환

> 모든 신호는 사인파의 합입니다. 푸리에 변환은 어떤 사인파가 포함되는지 알려줍니다.

**유형:** Build
**언어:** Python
**선수 요건:** 1단계, 01-04강, 19강 (복소수)
**시간:** 약 90분

## 학습 목표

- DFT를 처음부터 구현하고 O(N log N) Cooley-Tukey FFT와 비교 검증해 보세요
- 주파수 계수를 해석해 보세요: 신호에서 진폭, 위상, 전력 스펙트럼을 추출합니다
- 컨볼루션 정리를 적용해 FFT 곱셈으로 컨볼루션을 수행해 보세요
- 푸리에 주파수 분해를 트랜스포머 위치 인코딩 및 CNN 컨볼루션 레이어와 연결해 보세요

## 문제점

오디오 녹음은 시간에 따른 압력 측정값의 시계열입니다. 주식 가격은 일별 값의 시계열입니다. 이미지는 공간에 따른 픽셀 강도의 격자입니다. 이 모든 것은 시간 영역(또는 공간 영역)의 데이터입니다. 어떤 인덱스에 따라 값이 변하는 것을 볼 수 있습니다.

하지만 시간 영역에서는 많은 패턴이 보이지 않습니다. 이 오디오 신호는 순수한 톤일까요, 화음일까요? 이 주식 가격에는 주간 주기성이 있을까요? 이 이미지에는 반복되는 텍스처가 있을까요? 이 질문들은 주파수 콘텐츠에 관한 것이며, 시간 영역은 이를 숨깁니다.

푸리에 변환은 데이터를 시간 영역에서 주파수 영역으로 변환합니다. 신호를 받아 서로 다른 주파수의 사인파로 분해합니다. 각 사인파는 진폭(강도)과 위상(시작 위치)을 가집니다. 푸리에 변환은 이 두 가지를 모두 알려줍니다.

ML에서 주파수 영역적 사고가 모든 곳에 나타나므로 이는 중요합니다. 합성곱 신경망(CNN)은 컨볼루션을 수행하며, 이는 주파수 영역에서의 곱셈입니다. 트랜스포머 위치 인코딩은 위치를 표현하기 위해 주파수 분해를 사용합니다. 오디오 모델(음성 인식, 음악 생성)은 스펙트로그램 -- 소리의 주파수 표현 -- 위에서 동작합니다. 시계열 모델은 주기적 패턴을 찾습니다. 푸리에 변환을 이해하면 이 모든 것을 다루는 어휘를 갖게 됩니다.

## 개념

### DFT 정의

N개의 샘플 x[0], x[1], ..., x[N-1]가 주어지면, 이산 푸리에 변환(DFT)은 N개의 주파수 계수 X[0], X[1], ..., X[N-1]을 생성합니다:

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1
```

각 X[k]는 복소수입니다. 그 크기 |X[k]|는 주파수 k의 진폭을 알려줍니다. 그 위상 angle(X[k])는 해당 주파수의 위상 오프셋을 알려줍니다.

핵심 통찰: `e^(-2*pi*i*k*n/N)`는 주파수 k에서 회전하는 페이저입니다. DFT는 신호와 N개의 균일하게 간격이 떨어진 주파수 각각 간의 상관관계를 계산합니다. 신호가 주파수 k에서 에너지를 포함하고 있다면, 상관관계는 큽니다. 그렇지 않다면, 거의 0에 가깝습니다.

### 각 계수의 의미

**X[0]: DC 성분.** 이는 모든 샘플의 합이며, 평균에 비례합니다. 신호의 상수(0 주파수) 오프셋을 나타냅니다.

```
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples
```

**X[k] for 1 <= k <= N/2: 양의 주파수.** X[k]는 N개의 샘플당 k 사이클의 주파수를 나타냅니다. k가 높을수록 주파수가 높습니다(더 빠른 진동).

**X[N/2]: 나이퀴스트 주파수.** N개의 샘플로 표현할 수 있는 가장 높은 주파수입니다. 이 주파수 이상에서는 에일리어싱이 발생하며, 높은 주파수가 낮은 주파수처럼 위장됩니다.

**X[k] for N/2 < k < N: 음의 주파수.** 실수 값 신호의 경우, X[N-k] = conj(X[k])입니다. 음의 주파수는 양의 주파수의 거울상입니다. 이 때문에 유용한 정보는 첫 N/2 + 1개의 계수에 있습니다.

### 역 DFT

역 DFT는 주파수 계수로부터 원본 신호를 재구성합니다:

```
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1
```

순방향 DFT와의 유일한 차이점은 지수의 부호가 양수(음수가 아님)이며, 1/N 정규화 계수가 있다는 점입니다.

역 DFT는 완벽한 재구성입니다. 정보는 손실되지 않습니다. 시간 영역에서 주파수 영역으로 이동하고 다시 돌아오면서 어떤 오류도 발생하지 않습니다. DFT는 기저(change of basis)의 변화이며, 동일한 정보를 다른 좌표계로 재표현합니다.

### FFT: 빠르게 만들기

위에서 정의한 DFT는 O(N^2)입니다: N개의 출력 계수 각각에 대해 N개의 입력 샘플을 합산합니다. N = 100만인 경우, 이는 10^12번의 연산입니다.

고속 푸리에 변환(FFT)은 O(N log N)으로 동일한 결과를 계산합니다. N = 100만인 경우, 1조 번의 연산 대신 약 2천만 번의 연산이 필요합니다. 이것이 주파수 분석을 실용적으로 만드는 요인입니다.

Cooley-Tukey 알고리즘(가장 일반적인 FFT)은 분할 정복 방식으로 작동합니다:

1. 신호를 짝수 인덱스 샘플과 홀수 인덱스 샘플로 분리합니다.
2. 각 절반의 DFT를 재귀적으로 계산합니다.
3. 두 개의 절반 크기 DFT를 "트윙들 팩터(twiddle factors)" e^(-2*pi*i*k/N)를 사용하여 결합합니다.

```
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples
```

대칭성으로 인해 각 재귀 단계는 O(N) 연산을 수행하며, 총 log2(N) 단계가 있습니다. 총 연산량: O(N log N).

```mermaid
graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 샘플"] -->|"짝수/홀수 분리"| E["짝수: x["0,2,4,6"]"]
        X -->|"짝수/홀수 분리"| O["홀수: x["1,3,5,7"]"]
        E -->|"4점 FFT"| EK["E[0..3]"]
        O -->|"4점 FFT"| OK["O[0..3]"]
        EK -->|"트윙들 팩터로 결합"| XK["X[0..7]"]
        OK -->|"트윙들 팩터로 결합"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64번의 곱셈"]
        C2["FFT: O(N log N) = 24번의 곱셈"]
    end
```

FFT는 신호 길이가 2의 거듭제곱이어야 합니다. 실제로는 신호를 다음 2의 거듭제곱으로 제로 패딩(zero-padding)합니다.

### 스펙트럼 분석

**파워 스펙트럼(power spectrum)**은 |X[k]|^2 -- 각 주파수 계수의 크기의 제곱입니다. 각 주파수에 얼마나 많은 에너지가 있는지 보여줍니다.

**위상 스펙트럼(phase spectrum)**은 angle(X[k]) -- 각 주파수의 위상 오프셋입니다. 대부분의 분석 작업에서는 파워 스펙트럼에 집중하고 위상은 무시합니다.

```
Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)
```

### 주파수 해상도

DFT의 주파수 해상도는 샘플 수 N과 샘플링 속도 fs에 따라 달라집니다.

```
Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)
```

가까운 두 주파수를 구분하려면 더 많은 샘플이 필요합니다. 높은 주파수를 포착하려면 더 높은 샘플링 속도가 필요합니다.

### 컨볼루션 정리

이것은 신호 처리에서 가장 중요한 결과 중 하나이며, CNN과 직접적으로 관련이 있습니다.

**시간 영역에서의 컨볼루션은 주파수 영역에서의 점별 곱(pointwise multiplication)과 같습니다.**

```
x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication
```

이것이 중요한 이유:

- 길이 N과 M인 두 신호의 직접 컨볼루션은 O(N*M) 연산이 필요합니다.
- FFT 기반 컨볼루션은 O(N log N)이 필요합니다: 두 신호를 변환하고, 곱한 후, 다시 변환합니다.
- 큰 커널의 경우, FFT 합성곱이 훨씬 더 빠릅니다.
- 이는 큰 receptive field를 가진 합성곱 층에서 정확히 일어나는 현상입니다.

참고: DFT는 순환 합성곱을 계산합니다(신호가 wraps around). 선형 합성곱(wraparound 없음)을 계산하려면, 두 신호를 길이 N + M - 1로 zero-pad한 후 계산해 보세요.

```mermaid
graph LR
    subgraph "Time Domain"
        TA["신호 x["n"]"] -->|"합성곱 (느림: O(NM))"| TC["출력 y["n"]"]
        TB["필터 h["n"]"] -->|"합성곱"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"곱셈 (빠름: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"곱셈"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC
```

### 윈도잉

DFT는 신호가 주기적이라고 가정합니다 -- N개의 샘플을 무한히 반복되는 신호의 한 주기로 취급합니다. 신호가 시작값과 종료값이 같지 않으면, 경계에서 불연속성이 발생하며 이는 스퓨리어스 고주파 성분으로 나타납니다. 이를 스펙트럼 누출(spectral leakage)이라고 합니다.

윈도잉은 DFT를 계산하기 전에 신호의 양 끝을 0으로 tapering하여 누출을 줄입니다.

공통적인 윈도우:

| 윈도우 | 모양 | 주엽(main lobe) 폭 | 부엽(side lobe) 레벨 | 사용 사례 |
|--------|-------|----------------|-----------------|----------|
| Rectangular | Flat (윈도우 없음) | 가장 좁음 | 가장 높음 (-13 dB) | 신호가 N 샘플에서 정확히 주기적일 때 |
| Hann | Raised cosine | 중간 | 낮음 (-31 dB) | 일반적인 스펙트럼 분석 |
| Hamming | Modified cosine | 중간 | 더 낮음 (-42 dB) | 오디오 처리, 음성 분석 |
| Blackman | Triple cosine | 넓음 | 매우 낮음 (-58 dB) | 부엽 억제가 중요한 경우 |

```
Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

DFT 전에 윈도우를 신호와 요소별로 곱하여 적용하세요: `X = DFT(x * w)`.

### DFT 특성

| 특성 | 시간 영역 | 주파수 영역 |
|----------|-------------|-----------------|
| 선형성 | a*x + b*y | a*X + b*Y |
| 시간 이동 | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| 주파수 이동 | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| 합성곱 | x * h | X * H (pointwise) |
| 곱셈 | x * h (점별 곱) | X * H (순환 합성곱, 1/N으로 스케일링) |
| 파르세발 정리 | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| 켤레 대칭 (실수 입력) | x[n] 실수 | X[k] = conj(X[N-k]) |

파르세발 정리는 두 영역에서 총 에너지가 동일하다고 말합니다. 변환을 통해 에너지가 보존됩니다.

### 포지셔널 인코딩과의 연결

원래 트랜스포머는 사인(positional encodings) 포지셔널 인코딩을 사용합니다:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

각 차원 쌍 (2i, 2i+1)은 서로 다른 주파수로 진동합니다. 주파수는 높은 것(차원 0,1)에서 낮은 것(마지막 차원)까지 기하급수적으로 간격이 배치됩니다. 이는 푸리에 계수가 신호를 고유하게 식별하는 것과 유사하게, 모든 주파수 대역에 걸쳐 각 위치가 고유한 패턴을 갖도록 합니다.

이것이 제공하는 주요 특성:

- **고유성:** 두 위치가 동일한 인코딩을 가질 수 없습니다.
- **유한한 값:** sin과 cos는 항상 [-1, 1] 범위에 있습니다.
- **상대적 위치:** 위치 p+k의 인코딩은 위치 p의 인코딩에 대한 선형 함수로 표현할 수 있습니다. 모델은 상대적 위치에 주의(attend)하는 것을 학습할 수 있습니다.

### CNN과의 연결

합성곱 레이어는 학습된 필터(커널)를 신호나 이미지 위에 슬라이딩하여 입력에 적용합니다. 수학적으로 이는 합성곱 연산입니다.

합성곱 정리에 의해, 이는 다음 연산과 동일합니다:
1. 입력의 FFT
2. 커널의 FFT
3. 주파수 영역에서 곱셈
4. 결과의 IFFT

표준 CNN 구현은 작은 3x3 커널의 경우 더 빠른 직접 합성곱을 사용합니다. 하지만 큰 커널이나 전역 합성곱의 경우, FFT 기반 접근 방식이 훨씬 더 빠릅니다. 일부 아키텍처(예: FNet)는 어텐션을 완전히 FFT로 대체하여 O(N^2) 대신 O(N log N) 복잡도로 경쟁력 있는 정확도를 달성합니다.

### 스펙트로그램과 단기 푸리에 변환

단일 FFT는 전체 신호의 주파수 내용을 제공하지만, 그 주파수가 언제 발생하는지에 대해서는 아무것도 알려주지 않습니다. 치프(시간에 따라 주파수가 증가하는 신호)와 화음(모든 주파수가 동시에 존재하는 신호)은 동일한 진폭 스펙트럼을 가질 수 있습니다.

단시간 푸리에 변환(STFT)은 신호의 겹치는 윈도우에 FFT를 계산하여 이 문제를 해결합니다. 그 결과 스펙트로그램이 생성됩니다. 스펙트로그램은 한 축에 시간, 다른 축에 주파수를 가진 2D 표현입니다. 각 지점의 강도는 해당 시간에서의 해당 주파수 에너지를 나타냅니다.

```
STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram
```

스펙트로그램은 오디오 ML 모델의 표준 입력 표현입니다. 음성 인식 모델(Whisper, DeepSpeech)은 멜 스펙트로그램을 사용합니다. 멜 스펙트로그램은 주파수를 멜 스케일에 매핑한 스펙트로그램으로, 인간의 피치 지각과 더 잘 일치합니다.

### 에일리어싱(Aliasing)

신호에 Nyquist 주파수(fs/2) 이상의 주파수가 포함되어 있으면, fs 속도로 샘플링할 때 에일리어싱된 사본이 생성됩니다. 100 Hz로 샘플링한 90 Hz 신호는 10 Hz 신호와 구별할 수 없습니다. 샘플만으로는 이들을 구분할 방법이 없습니다.

```
Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.
```

이 때문에 아날로그-디지털 변환기는 샘플링 전에 Nyquist 주파수 이상의 주파수를 제거하는 안티 에일리어싱 필터를 포함합니다. ML에서는 적절한 저역 통과 필터링 없이 특징 맵을 다운샘플링할 때 에일리어싱이 발생합니다. 일부 아키텍처는 안티 에일리어싱 풀링 레이어로 이를 해결합니다.

### 제로 패딩은 해상도를 높이지 않습니다

흔한 오해: FFT 전에 신호에 제로 패딩을 하면 주파수 해상도가 향상된다고 생각하지만, 이는 사실이 아닙니다. 제로 패딩은 기존 주파수 빈(bin) 사이를 보간하여 스펙트럼을 더 매끄럽게 보이게 할 뿐입니다. 원본 샘플에 존재하지 않았던 주파수 세부 정보를 드러낼 수는 없습니다.

진정한 주파수 해상도는 관측 시간 T = N / fs에만 의존합니다. delta_f만큼 분리된 두 주파수를 구분하려면 최소 T = 1 / delta_f 초의 데이터가 필요합니다. 제로 패딩의 양은 이 근본적인 한계를 변경하지 못합니다.

```figure
fourier-synthesis
```

## 구현하기

### 1단계: DFT를 처음부터 구현하기

O(N^2) DFT는 정의에서 직접 따릅니다.

```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```

### 2단계: 역 DFT

구조는 동일하며, 지수는 양수이고 N으로 나눕니다.

```python
def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

### 3단계: FFT (Cooley-Tukey)

재귀적 FFT는 길이가 2의 거듭제곱이어야 합니다. 짝수 및 홀수 부분으로 나누고, 재귀적으로 계산한 후 트윙들 팩터(twiddle factors)로 결합합니다.

```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result
```

### 4단계: 스펙트럼 분석 헬퍼

```python
def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]
```

## 사용하기

실제 작업에서는 고도로 최적화된 C 라이브러리를 사용하는 numpy의 FFT를 사용하세요.

```python
import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]
```

윈도잉 및 더 고급 스펙트럼 분석을 위해:

```python
from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)
```

컨볼루션을 위해:

```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')
```

스펙트로그램을 위해:

```python
from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2
```

스펙트로그램 행렬은 (n_frequencies, n_time_frames) 형태를 가집니다. 각 열은 하나의 시간 윈도우에서의 파워 스펙트럼입니다. 오디오 ML 모델이 입력으로 사용하는 형태입니다.

## 출시하기

`code/fourier.py`을 실행하여 `outputs/prompt-spectral-analyzer.md`을 생성하세요.

## 연습 문제

1. **순수 톤 식별.** 알 수 없는 주파수(1~50 Hz 사이)의 단일 사인파 신호를 생성하고, 1초 동안 128 Hz로 샘플링하세요. DFT를 사용하여 주파수를 식별하세요. 답이 일치하는지 확인하세요. 이제 표준 편차 0.5의 가우시안 잡음을 추가하고 반복하세요. 잡음이 스펙트럼에 어떤 영향을 미치나요?

2. **FFT vs DFT 검증.** 길이가 64인 랜덤 신호를 생성하세요. DFT (O(N^2))와 FFT를 모두 계산하세요. 모든 계수가 1e-10 이내로 일치하는지 확인하세요. 길이가 256, 512, 1024, 2048인 신호에 대해 두 함수의 시간을 측정하세요. DFT 시간과 FFT 시간의 비율을 플롯하세요.

3. **예제를 통한 컨볼루션 정리 증명.** 신호 x = [1, 2, 3, 4, 0, 0, 0, 0]과 필터 h = [1, 1, 1, 0, 0, 0, 0, 0]을 생성하세요. 이 둘의 순환 컨볼루션을 직접(중첩 루프) 계산하세요. 그런 다음 FFT를 통해 계산하세요(변환, 곱셈, 역변환). 결과가 일치하는지 확인하세요. 이제 적절히 제로 패딩하여 선형 컨볼루션을 수행하세요.

4. **윈도잉 효과.** 10 Hz와 12 Hz의 두 사인파의 합인 신호를 생성하세요(매우 가까운 주파수). 1초 동안 128 Hz로 샘플링하세요. 윈도우 없음, Hann 윈도우, Hamming 윈도우로 파워 스펙트럼을 계산하세요. 어떤 윈도우가 두 피크를 가장 쉽게 구별할 수 있게 하나요? 왜 그런가요?

5. **위치 인코딩 분석.** d_model = 128, max_pos = 512에 대한 사인파 위치 인코딩을 생성하세요. 각 위치 쌍 (p1, p2)에 대해 인코딩의 내적을 계산하세요. 내적이 |p1 - p2|에만 의존하고 절대 위치에는 의존하지 않음을 보이세요. 거리가 증가함에 따라 내적은 어떻게 변하나요?

## 핵심 용어

| 용어 | 의미 |
|------|---------------|
| DFT (이산 푸리에 변환)(Discrete Fourier Transform) | N개의 시간 영역 샘플을 N개의 주파수 영역 계수로 변환합니다. 각 계수는 해당 주파수의 복소 사인파와의 상관관계입니다 |
| FFT (고속 푸리에 변환) | DFT를 계산하는 O(N log N) 알고리즘. Cooley-Tukey 알고리즘은 짝수/홀수 인덱스를 재귀적으로 분할합니다 |
| 역 DFT | 주파수 계수로부터 시간 영역 신호를 재구성합니다. DFT와 동일한 공식이며 지수 부호를 반전하고 1/N 스케일링을 적용합니다 |
| 주파수 빈 | DFT 출력의 각 인덱스 k는 주파수 k*fs/N Hz를 나타냅니다. "빈"은 이산 주파수 슬롯입니다 |
| DC 성분 | X[0], 즉 0 주파수 계수입니다. 신호 평균에 비례합니다 |
| 나이퀴스트 주파수 | fs/2, 샘플링 속도 fs에서 표현 가능한 최대 주파수입니다. 이 이상의 주파수는 에일리어싱됩니다 |
| 전력 스펙트럼 | \|X[k]\|^2, 각 주파수 계수의 크기의 제곱입니다. 주파수 간 에너지 분포를 보여줍니다 |
| 위상 스펙트럼 | angle(X[k]), 각 주파수 성분의 위상 오프셋입니다. 분석에서 종종 무시됩니다 |
| 스펙트럼 누출 | 비주기 신호를 주기 신호로 취급하여 발생하는 가짜 주파수 콘텐츠입니다. 윈도우링으로 감소시킵니다 |
| 윈도우 함수 | 스펙트럼 누출을 줄이기 위해 DFT 전에 적용하는 테이퍼링 함수 (Hann, Hamming, Blackman)입니다 |
| 트윌 팩터 | FFT 버터플라이 계산에서 하위 DFT를 결합하는 데 사용되는 복소 지수 e^(-2*pi*i*k/N)입니다 |
| 컨볼루션 정리 | 시간 영역의 컨볼루션은 주파수 영역의 점별 곱과 같습니다. 신호 처리 및 CNN의 기본 원리입니다 |
| 순환 컨볼루션 | 신호가_wrap_되는 컨볼루션입니다. DFT가 자연스럽게 계산하는 연산입니다 |
| 선형 컨볼루션 |_wrap_이 없는 표준 컨볼루션입니다. DFT 전에 제로 패딩을 적용하여 달성합니다 |
| 파르셀의 정리 | 푸리에 변환을 통해 총 에너지가 보존됩니다. sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| 에일리어싱 | 샘플링 속도가 불충분하여 나이퀴스트 이상의 주파수가 낮은 주파수로 나타나는 현상입니다 |

## 추가 읽기

- [Cooley & Tukey: An Algorithm for the Machine Calculation of Complex Fourier Series (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/) - 컴퓨팅을 바꾼 원본 FFT 논문
- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) - 푸리에 변환에 대한 최고의 시각적 소개
- [Lee-Thorp et al.: FNet: Mixing Tokens with Fourier Transforms (2021)](https://arxiv.org/abs/2105.03824) - 트랜스포머에서 셀프 어텐션을 FFT로 대체합니다
- [Smith: The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/) - FFT, 창 함수(windowing), 스펙트럼 분석을 심층적으로 다루는 무료 온라인 교과서
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 푸리에 주파수 분해에서 파생된 사인(positional) 위치 인코딩
- [Radford et al.: Whisper (2022)](https://arxiv.org/abs/2212.04356) - 멜 스펙트로그램을 입력 표현으로 사용하는 음성 인식
