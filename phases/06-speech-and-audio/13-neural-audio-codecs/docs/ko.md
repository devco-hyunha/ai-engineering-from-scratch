# 신경 오디오 코덱 — EnCodec, SNAC, Mimi, DAC 및 의미-음성 분리

> 2026년 오디오 생성은 거의 전부 토큰 기반입니다. EnCodec, SNAC, Mimi, DAC는 연속적인 파형을 트랜스포머가 예측할 수 있는 이산 시퀀스로 변환합니다. 의미 대 음성 토큰 분리 — 첫 번째 코드북은 의미, 나머지는 음성으로 처리 — 는 오디오 분야에서 트랜스포머 이후 가장 중요한 아키텍처 변화입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램), 10단계 · 11강 (양자화), 5단계 · 19강 (서브워드 토큰화)
**시간:** 약 60분

## 문제점

언어 모델은 이산 토큰을 다룹니다. 오디오는 연속적입니다. 음성/음악을 위한 LLM 스타일 모델(MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus)을 만들려면 먼저 **신경 오디오 코덱**이 필요합니다. 이는 오디오를 작은 토큰 어휘로 이산화하는 학습된 인코더와 파형을 재구성하는 대응 디코더로 구성됩니다.

두 가지 계열이 등장했습니다:

1. **재구성 우선 코덱** — EnCodec, DAC. 지각 오디오 품질을 최적화합니다. 토큰은 "음성"이며 화자 식별, 음색, 배경 잡음 등 모든 것을 포착합니다.
2. **의미 우선 코덱** — Mimi (Kyutai), SpeechTokenizer. 첫 번째 코드북이 언어/음성학적 내용을 인코딩하도록 강제합니다 (종종 WavLM에서 증류하여). 이후 코드북은 음성 세부 정보를 담당합니다.

2024-2026년의 통찰: **순수 재구성 코덱은 텍스트로부터 생성을 시도할 때 흐릿한 음성을 만들어냅니다.** 코덱 토큰 기반 LLM은 같은 코드북에서 언어 구조와 음성 구조를 모두 학습해야 하며, 이는 확장되지 않습니다. 이들을 분리하는 것 — 의미 코드북 0, 음성 코드북 1-N — 이 Moshi와 Sesame CSM이 작동하는 이유입니다.

## 개념

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 핵심 트릭: 잔차 벡터 양자화 (RVQ)

하나의 큰 코드북(좋은 품질을 위해 수백만 개의 코드가 필요함) 대신, 모든 최신 오디오 코덱은 **RVQ**를 사용합니다: 작은 코드북의 캐스케이드. 첫 번째 코드북은 인코더 출력을 양자화하고, 두 번째는 잔차를 양자화하며, 그 다음도 마찬가지입니다. 각 코드북은 1024개의 코드를 가집니다. 8개 코드북 = 유효 어휘는 1024^8 = 10^24입니다.

추론 시 디코더는 선택된 모든 코드를 프레임별로 합산하여 재구성합니다.

### 2026년에 중요한 네 가지 코덱

**EnCodec (Meta, 2022).** 기준선. 파형에 대한 인코더-디코더, RVQ 병목 현상. 24 kHz, 32개 코드북 가능, 기본값은 4개 코드북 @ 1.5 kbps. `1D conv + transformer + 1D conv` 아키텍처를 사용합니다. MusicGen에서 사용됩니다.

**DAC (Descript, 2023).** L2 정규화된 코드북, 주기적 활성화 함수, 개선된 손실 함수를 사용하는 RVQ. 오픈 코덱 중 가장 높은 재구성 충실도 — 12개 코드북으로 원본 음성 구분이 불가능한 경우도 있습니다. 44.1 kHz 풀 밴드.

**SNAC (Hubert Siuzdak, 2024).** 다중 스케일 RVQ — 거친 코드북은 세밀한 코드북보다 낮은 프레임 레이트로 작동합니다. 오디오를 계층적으로 모델링합니다: ~12 Hz의 거친 "스케치"와 50 Hz의 세부 사항. LM 기반 생성에 계층적 구조가 잘 매핑되므로 Orpheus-3B에서 사용됩니다.

**Mimi (Kyutai, 2024).** 2026년 게임 체인저. 12.5 Hz 프레임 레이트 (매우 낮음), 8개 코드북 @ 4.4 kbps. 코드북 0은 **WavLM에서 증류(distilled)**된 것입니다 — WavLM의 음성 내용 특성을 예측하도록 학습되었습니다. 코드북 1-7은 음향 잔차입니다. 이 분리는 Moshi (15강)와 Sesame CSM을 구동합니다.

### 언어 모델링에서 프레임 레이트가 중요합니다

낮은 프레임 레이트 = 짧은 시퀀스 = 빠른 LM.

| 코덱 | 프레임 레이트 | 1초 = N 프레임 | 용도 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 음악, 일반 오디오 |
| DAC-44.1k | 86 Hz | 86 | 고충실도 음악 |
| SNAC-24k (거친) | ~12 Hz | 12 | AR-LM 효율적 |
| Mimi | 12.5 Hz | 12.5 | 스트리밍 음성 |

12.5 Hz에서는 10초 발화가 125개 코덱 프레임뿐입니다 — 트랜스포머가 쉽게 예측할 수 있습니다.

### 시맨틱 토큰 vs 음향 토큰

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **시맨틱 토큰 (Mimi의 코드북 0).** 발화된 내용 — 음소, 단어, 내용 —을 인코딩합니다. 보조 예측 손실을 통해 WavLM에서 증류(distilled)되었습니다.
- **음향 토큰 (코드북 1-7).** 음색, 화자 식별, 운율, 배경 잡음, 세부 사항을 인코딩합니다.

s34

### 2026년 재구성 품질 (초당 비트 수, 낮은 비트레이트가 더 좋음)

| 코덱 | 비트레이트 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus와 같은 전통적인 코덱은 지각 품질 측면에서 비트당 성능이 여전히 우세합니다. 뉴럴 코덱은 **이산 토큰**(Opus가 생성하지 않는 토큰)과 **생성 모델 품질**(LM이 해당 토큰으로 수행할 수 있는 작업)에서 우세합니다.

```figure
rvq-codec-cascade
```

## 구현하기

### 1단계: EnCodec으로 인코딩

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

`n_codebooks=8`는 6 kbps입니다. 각 코드는 0-1023 (10비트)입니다.

### 2단계: 디코딩 및 재구성 측정

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 3단계: 의미-음향 분리 (Mimi 스타일)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

의미 코드북 0은 WavLM에 정렬되어 있습니다. 텍스트를 의미 토큰으로 변환하는 트랜스포머를 훈련할 수 있으며, 이는 오디오로 직접 변환하는 것보다 훨씬 작은 어휘를 사용합니다. 이후 별도의 음향-파형 디코더가 화자 참조에 조건을 부여합니다.

### 4단계: 코덱 토큰에 대한 AR LM이 작동하는 이유

Mimi의 12.5 Hz × 8 코드북으로 10초 음성 클립을 처리할 경우:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000개 토큰은 트랜스포머에 있어 사소한 컨텍스트입니다. 2억 5600만 매개변수를 가진 트랜스포머는 최신 GPU에서 10초의 음성을 밀리초 단위로 생성할 수 있습니다.

## 사용하기

문제 → 코덱 매핑:

| 작업 | 코덱 |
|------|-------|
| 일반 음악 생성 | EnCodec-24k |
| 최고 충실도 재구성 | DAC-44.1k |
| 음성 AR LM (TTS) | SNAC 또는 Mimi |
| 스트리밍 풀듀플렉스 음성 | Mimi (12.5 Hz) |
| 텍스트가 포함된 효과음 라이브러리 | EnCodec + T5 조건 |
| 세밀한 오디오 편집 | DAC + 인페인팅 |

경험칙: **생성 모델을 구축한다면 Mimi나 SNAC으로 시작하세요. 압축 파이프라인을 구축한다면 Opus를 사용하세요.**

## 함정

- **코드북이 너무 많습니다.** 코드북을 추가하면 충실도는 선형적으로 증가하지만 LM 시퀀스 길이도 선형적으로 증가합니다. 8-12에서 멈추세요.
- **프레임 레이트 불일치.** 12.5 Hz Mimi로 LM을 학습한 후 50 Hz EnCodec으로 미세 조정하면 조용히 실패합니다.
- **모든 코드북이 동일하다고 가정합니다.** Mimi에서는 코드북 0이 콘텐츠를 담고 있으며, 이를 잃으면 명료성이 파괴됩니다. 코드북 7을 잃는 것은 거의 눈에 띄지 않습니다.
- **재구성 품질을 유일한 지표로 사용합니다.** 코덱은 재구성 품질이 훌륭할 수 있지만, 시맨틱 구조가 나쁘면 LM 기반 생성에는 쓸모가 없습니다.

## 출시하기

`outputs/skill-codec-picker.md`로 저장하세요. 주어진 생성 또는 압축 작업에 적합한 코덱을 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 이는 장난감 스칼라 + 잔차 양자화를 구현하며, 코드북을 추가할 때 재구성 오차를 측정합니다.
2. **중간.** `encodec`을 설치하고 홀드아웃 음성 클립에서 1, 4, 8, 32개의 코드북을 비교하세요. 비트레이트 대비 PESQ 또는 MSE를 플롯하세요.
3. **어려움.** Mimi를 로드하세요. 클립을 인코딩하세요. 코드북 0을 랜덤 정수로 교체하고 디코딩하세요. 그런 다음 코드북 7도 유사하게 교체하세요. 두 가지 손상(corruption)을 비교하세요. 코드북 0 손상은 명료성을 파괴해야 하며, 코드북 7 손상은 거의 아무것도 변경하지 않아야 합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| RVQ | 잔차 양자화 | 작은 코드북의 캐스케이드; 각각이 이전 잔차를 양자화합니다. |
| 프레임 레이트 | 코덱 속도 | 초당 토큰 프레임 수. 낮을수록 LM이 빠릅니다. |
| 시맨틱 코드북 | 코드북 0 (Mimi) | SSL 기능에서 증류된 코드북; 콘텐츠를 인코딩합니다. |
| 음향 코드북 | 나머지 전부 | 음색, 운율, 잡음, 세밀한 디테일. |
| PESQ / ViSQOL | 지각 품질 | MOS와 상관관계가 있는 객관적 지표. |
| EnCodec | Meta 코덱 | RVQ 기준선; MusicGen에서 사용됩니다. |
| Mimi | Kyutai 코덱 | 12.5 Hz 프레임 레이트; 의미-음성 분리; Moshi의 기반 기술. |

## 추가 읽기

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 기준선.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 최고 충실도의 오픈 소스.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 다중 스케일 RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 의미-음성 분리, WavLM 증류.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 2단계 의미/음성 패러다임.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 최초의 스트리밍 가능한 RVQ 코덱.
