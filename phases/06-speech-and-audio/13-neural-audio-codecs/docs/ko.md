# 신경망 오디오 코덱 — EnCodec, SNAC, Mimi, DAC 및 의미-음향 분리 (Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split)

> 2026년 오디오 생성은 거의 모두 토큰으로 이루어집니다. EnCodec, SNAC, Mimi, DAC는 연속적인 파형(waveform)을 트랜스포머가 예측할 수 있는 이산 시퀀스로 변환합니다. 첫 번째 코드북은 의미(semantic), 나머지는 음향(acoustic)으로 나누는 의미 대 음향 토큰 분리(semantic-vs-acoustic token split)는 오디오 분야에서 트랜스포머 등장 이후 가장 중요한 구조적 변화입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 문제 상황 (The Problem)

언어 모델은 이산 토큰을 기반으로 작동합니다. 반면 오디오는 연속적입니다. MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus와 같은 음성/음악용 LLM 스타일 모델을 구축하려면 먼저 **신경망 오디오 코덱(neural audio codec)**이 필요합니다. 이는 오디오를 작은 어휘의 토큰으로 이산화하는 학습된 인코더와, 파형을 재구성하는 일치하는 디코더로 구성됩니다.

크게 두 가지 계열이 등장했습니다:

1. **재구성 우선 코덱 (Reconstruction-first codecs)** — EnCodec, DAC. 지각적 오디오 품질을 최적화합니다. 토큰은 "음향적(acoustic)"이며, 화자 정체성, 음색, 배경 잡음을 포함한 모든 요소를 포착합니다.
2. **의미 우선 코덱 (Semantic-first codecs)** — Mimi (Kyutai), SpeechTokenizer. 첫 번째 코드북이 언어적/음성적(linguistic / phonetic) 콘텐츠를 인코딩하도록 강제합니다(주로 WavLM에서 증류). 후속 코드북은 음향적 세부 사항을 인코딩합니다.

2024~2026년의 핵심 통찰: **텍스트로부터 생성을 시도할 때 순수 재구성 코덱은 흐릿한 음성(blurry speech)을 만듭니다.** 코덱 토큰을 다루는 LLM이 동일한 코드북 안에서 언어 구조와 음향 구조를 모두 학습해야 하므로 확장성이 떨어집니다. 이를 분리하여 0번 코드북은 의미, 1~N번 코드북은 음향으로 나누는 것이 바로 Moshi와 Sesame CSM을 작동하게 만드는 원동력입니다.

## 핵심 개념 (The Concept)

![4가지 코덱 지형도: EnCodec, DAC, SNAC (멀티스케일), Mimi (의미+음향)](../assets/codec-comparison.svg)

### 핵심 기법: 잔차 벡터 양자화 (Residual Vector Quantization, RVQ)

좋은 품질을 위해 수백만 개의 코드가 필요한 하나의 거대한 코드북 대신, 모든 현대적인 오디오 코덱은 작은 코드북들을 직렬로 연결한 **RVQ**를 사용합니다. 첫 번째 코드북이 인코더 출력을 양자화하고, 두 번째 코드북은 잔차(residual)를 양자화하는 식입니다. 각 코드북은 1024개의 코드로 구성됩니다. 8개 코드북의 경우 실효 어휘 크기는 1024^8 = 10^24에 달합니다.

추론 시 디코더는 프레임당 선택된 모든 코드를 합산하여 파형을 재구성합니다.

### 2026년에 중요한 4가지 코덱 (The four codecs that matter in 2026)

**EnCodec (Meta, 2022).** 기준 모델(baseline)입니다. 파형에 대한 인코더-디코더 구조와 RVQ 병목(bottleneck)을 사용합니다. 24 kHz, 최대 32개 코드북을 지원하며, 기본값은 1.5 kbps에서 4개 코드북입니다. `1D conv + transformer + 1D conv` 아키텍처를 사용하며, MusicGen에서 채택되었습니다.

**DAC (Descript, 2023).** L2 정규화 코드북, 주기적 활성화 함수, 개선된 손실 함수를 적용한 RVQ입니다. 오픈소스 코덱 중 가장 높은 재구성 충실도를 보이며, 12개 코드북 사용 시 원본 음성과 구별하기 어려울 정도입니다. 44.1 kHz 풀밴드를 지원합니다.

**SNAC (Hubert Siuzdak, 2024).** 멀티스케일 RVQ입니다. 거친(coarse) 코드북이 미세한(fine) 코드북보다 낮은 프레임 레이트로 작동합니다. 약 12 Hz의 거친 "스케치"에 50 Hz의 디테일을 더해 오디오를 계층적으로 효과적으로 모델링합니다. 계층 구조가 LM 기반 생성에 잘 부합하여 Orpheus-3B에서 사용됩니다.

**Mimi (Kyutai, 2024).** 2026년의 게임 체인저입니다. 12.5 Hz라는 극도로 낮은 프레임 레이트와 4.4 kbps에서 8개 코드북을 사용합니다. 코드북 0은 **WavLM에서 증류(distilled from WavLM)**되어 WavLM의 음성 콘텐츠 특징을 예측하도록 학습됩니다. 코드북 1~7은 음향 잔차입니다. 이 분리 구조가 Moshi(15번째 레슨)와 Sesame CSM의 기반이 됩니다.

### 언어 모델링에서 프레임 레이트의 중요성

프레임 레이트가 낮을수록 = 시퀀스 길이가 짧아짐 = LM 속도가 빨라짐.

| 코덱 | 프레임 레이트 | 1초 = N 프레임 | 적합한 용도 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 음악, 일반 오디오 |
| DAC-44.1k | 86 Hz | 86 | 고충실도 음악 |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM 효율성 최적화 |
| Mimi | 12.5 Hz | 12.5 | 스트리밍 음성 |

12.5 Hz에서는 10초 분량의 발화가 125개 코덱 프레임에 불과하므로 트랜스포머가 손쉽게 예측할 수 있습니다.

### 의미 토큰 vs 음향 토큰 (Semantic vs acoustic tokens)

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **의미 토큰 (Mimi의 코드북 0).** 발화된 내용(음소, 단어, 텍스트 콘텐츠)을 인코딩합니다. 보조 예측 손실을 통해 WavLM으로부터 증류됩니다.
- **음향 토큰 (코드북 1~7).** 음색, 화자 정체성, 운율, 배경 잡음, 미세한 디테일을 인코딩합니다.

AR LM은 (텍스트를 조건으로) 의미 토큰을 먼저 예측한 뒤, (의미 토큰 + 화자 참조를 조건으로) 음향 토큰을 예측합니다. 이러한 분해(factorization) 덕분에 최신 TTS는 제로샷 음성 복제가 가능합니다. 의미 모델이 내용을 처리하고, 음향 모델이 음색을 처리하기 때문입니다.

### 2026년 재구성 품질 (초당 비트 수, 비트레이트가 낮을수록 우수)

| 코덱 | 비트레이트 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus 같은 기존 전통 코덱은 비트당 지각 품질 면에서 여전히 우세합니다. 반면 신경망 코덱은 **이산 토큰**(Opus는 생성 불가)과 **생성 모델 품질**(LM이 해당 토큰으로 달성할 수 있는 성능) 면에서 우위를 점합니다.

```figure
rvq-codec-cascade
```

## 직접 구현해 보기 (Build It)

### 1단계: EnCodec으로 인코딩하기

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps에서 `n_codebooks=8`입니다. 각 코드는 0~1023(10비트) 범위입니다.

### 2단계: 디코딩 및 재구성 측정하기

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 3단계: 의미-음향 분리 (Mimi 방식)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

의미 코드북 0은 WavLM과 정렬되어 있습니다. 오디오로 직접 생성하는 것보다 훨씬 작은 어휘 사전으로 텍스트-의미 트랜스포머를 학습시킬 수 있습니다. 그런 다음 별도의 음향-파형 디코더가 화자 참조를 조건으로 받아 작동합니다.

### 4단계: 코덱 토큰 대상 AR LM이 작동하는 이유

Mimi의 12.5 Hz × 8개 코드북 기준 10초 음성 클립의 토큰 수:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1,000개 토큰은 트랜스포머에게 매우 가벼운 컨텍스트입니다. 256M 파라미터 트랜스포머는 최신 GPU에서 10초 분량의 음성을 수 밀리초 만에 생성할 수 있습니다.

## 적용해 보기 (Use It)

문제 유형 → 코덱 매핑:

| 작업 | 코덱 |
|------|-------|
| 일반 음악 생성 | EnCodec-24k |
| 최고 충실도 재구성 | DAC-44.1k |
| 음성 대상 AR LM (TTS) | SNAC 또는 Mimi |
| 스트리밍 전이중 음성 | Mimi (12.5 Hz) |
| 텍스트 조건부 효과음 라이브러리 | EnCodec + T5 조건 |
| 세밀한 오디오 편집 | DAC + 인페인팅(inpainting) |

실무 지침(Rule of thumb): **생성 모델을 구축하는 경우 Mimi 또는 SNAC로 시작하세요. 압축 파이프라인을 구축하는 경우 Opus를 사용하세요.**

## 주의할 점 (Pitfalls)

- **너무 많은 코드북.** 코드북을 추가하면 충실도가 선형적으로 증가하지만 LM 시퀀스 길이도 선형적으로 늘어납니다. 8~12개 수준에서 멈추세요.
- **프레임 레이트 불일치.** 12.5 Hz Mimi로 LM을 학습시킨 후 50 Hz EnCodec으로 미세조정하면 경고 없이 실패합니다.
- **모든 코드북이 동등하다고 가정.** Mimi에서 코드북 0은 콘텐츠를 담고 있어 이를 잃으면 명료도가 파괴됩니다. 반면 코드북 7을 잃는 것은 거의 체감되지 않습니다.
- **재구성 품질만을 유일한 지표로 사용.** 코덱의 재구성 품질이 뛰어나더라도 의미 구조가 부실하면 LM 기반 생성에는 쓸모가 없습니다.

## 완성 및 배포 (Ship It)

`outputs/skill-codec-picker.md`로 저장하세요. 주어진 생성 또는 압축 작업에 맞는 코덱을 선택합니다.

## 실습 과제 (Exercises)

1. **초급 (Easy).** `code/main.py`를 실행해 보세요. 간단한 스칼라 + 잔차 양자화기를 구현하고 코드북을 추가함에 따라 재구성 오차가 어떻게 변하는지 측정합니다.
2. **중급 (Medium).** `encodec`를 설치하고 별도로 보관된 음성 클립에서 1, 4, 8, 32개 코드북을 비교해 보세요. 비트레이트 대비 PESQ 또는 MSE를 그래프로 그려 보세요.
3. **고급 (Hard).** Mimi를 로드하세요. 클립을 인코딩합니다. 코드북 0을 무작위 정수로 교체한 후 디코딩해 보세요. 그런 다음 코드북 7을 비슷하게 교체해 보세요. 두 가지 변조 결과를 비교해 보세요 — 코드북 0 변조는 명료도를 완전히 파괴해야 하지만, 코드북 7 변조는 거의 아무런 차이를 일으키지 않아야 합니다.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| RVQ | 잔차 양자화 | 작은 코드북의 캐스케이드 구조. 각 코드북은 이전 단계의 잔차를 양자화합니다. |
| 프레임 레이트 (Frame rate) | 코덱 속도 | 초당 토큰 프레임 수. 낮을수록 LM 속도가 빨라집니다. |
| 의미 코드북 (Semantic codebook) | 코드북 0 (Mimi) | SSL 특징에서 증류된 코드북. 콘텐츠를 인코딩합니다. |
| 음향 코드북 (Acoustic codebooks) | 그 외 나머지 | 음색, 운율, 잡음, 세부 디테일. |
| PESQ / ViSQOL | 지각 품질 | MOS(평균 의견 점수)와 상관관계가 있는 객관적 평가 지표. |
| EnCodec | Meta 코덱 | RVQ 베이스라인. MusicGen에서 사용됩니다. |
| Mimi | Kyutai 코덱 | 12.5 Hz 프레임 레이트. 의미-음향 분리 구조. Moshi의 기반입니다. |

## 추가 읽을거리 (Further Reading)

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 베이스라인 모델.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 오픈 소스 중 최고 충실도 코덱.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 멀티스케일 RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 의미-음향 분리 및 WavLM 증류.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 2단계 의미/음향 패러다임.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 최초의 스트리밍 가능한 RVQ 코덱.
