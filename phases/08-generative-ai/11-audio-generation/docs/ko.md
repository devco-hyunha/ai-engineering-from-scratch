# 오디오 생성 (Audio Generation)

> 오디오는 16-48 kHz의 1차원(1-D) 신호입니다. 5초 길이의 클립은 8만~24만 개의 샘플로 구성됩니다. 어떤 트랜스포머(Transformer)도 이 시퀀스를 직접 처리하지 않습니다. 2026년 모든 상용 오디오 모델의 해결책은 동일합니다. 신경망 코덱(Neural Codec; Encodec, SoundStream, DAC)이 오디오를 50-75 Hz의 이산 토큰(Discrete Tokens)으로 압축하면, 트랜스포머나 확산 모델(Diffusion Model)이 해당 토큰을 생성하는 방식입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 문제 (The Problem)

세 가지 오디오 생성 작업:

1. **텍스트 음성 변환 (Text-to-speech).** 텍스트가 주어지면 음성을 생성합니다. 깨끗한 음성은 협대역(narrow-band)이며 강한 음성학적 구조를 가집니다. 이는 토큰 기반 트랜스포머(transformer-over-tokens)로 잘 해결됩니다. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **음악 생성 (Music generation).** 프롬프트(텍스트, 멜로디, 코드 진행, 장르)가 주어지면 음악을 생성합니다. 분포가 훨씬 더 넓습니다. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **오디오 효과 / 사운드 디자인 (Audio effects / sound design).** 프롬프트가 주어지면 환경음(ambient sound)이나 폴리(Foley)를 생성합니다. AudioGen, AudioLDM 2, Stable Audio Open.

세 작업 모두 동일한 기반 위에서 작동합니다: 신경망 오디오 코덱(neural audio codec) + 토큰 기반 자기회귀(token-AR) 또는 확산 생성기(diffusion generator).

## 개념 (The Concept)

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 신경 오디오 코덱 (Neural audio codecs)

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). 합성곱 인코더(convolutional encoder)는 파형(waveform)을 타임스텝당 하나의 벡터로 압축하며, 잔차 벡터 양자화(residual vector quantization, RVQ)는 각 벡터를 $K$개의 코드북 인덱스 계층으로 변환합니다. 디코더는 이 과정을 역으로 수행합니다. 75 Hz에서 8개의 RVQ 코드북을 사용하여 2 kbps로 24 kHz 오디오를 처리하면 초당 600개의 토큰이 생성됩니다.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → 75 Hz에서의 인덱스
                     ├─ RVQ layer 2 → 75 Hz에서의 인덱스
                     ├─ ...
                     └─ RVQ layer 8
```

### 상위의 두 가지 생성 패러다임 (Two generative paradigms on top)

**토큰 자기회귀 (Token-autoregressive).** RVQ 토큰을 하나의 시퀀스로 펼친 후, 디코더 전용(decoder-only) 트랜스포머를 실행합니다. MusicGen은 "지연 병렬(delayed parallel)" 방식을 사용하여 스트림별 오프셋과 함께 K개의 코드북 스트림을 병렬로 방출합니다. VALL-E는 텍스트 프롬프트와 3초 분량의 음성 샘플로부터 음성 토큰을 생성합니다.

**잠재 확산 (Latent diffusion).** 코덱 토큰을 연속적인 잠재 변수(continuous latents)로 패킹하거나 범주형 확산(categorical diffusion)으로 모델링합니다. Stable Audio 2.5는 연속적인 오디오 잠재 변수에 대해 플로우 매칭(flow matching)을 사용합니다. AudioLDM 2는 text-to-mel-to-audio 확산 방식을 사용합니다.

2024-2026년 트렌드: 음악 분야에서는 플로우 매칭(flow matching)이 승기를 잡고 있으며(더 빠른 추론, 더 깨끗한 샘플), 음성 분야에서는 토큰-AR(token-AR)이 자연스러운 인과성(causality)과 원활한 스트리밍 능력 덕분에 여전히 지배적입니다.

## 프로덕션 환경 (Production landscape)

| 시스템 (System) | 작업 (Task) | 백본 (Backbone) | 지연 시간 (Latency) |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | 첫 토큰까지 ~300ms |
| OpenAI GPT-4o audio | 전이중 음성 (Full-duplex speech) | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | 비스트리밍 (Non-streaming) |
| Stable Audio 2.5 | 음악 / SFX | DiT + audio latents 기반 flow matching | 1분 클립 기준 ~10s |
| Suno v4 | 전체 곡 (Full songs) | 미공개; token-AR로 추정 | 곡당 ~30s |
| Udio v1.5 | 전체 곡 (Full songs) | 미공개 | 곡당 ~30s |
| MusicGen 3.3B | 음악 | Encodec 32kHz 기반 Token-AR | 실시간 (Real-time) |
| AudioCraft 2 | 음악 + SFX | Flow matching | 5초 클립 기준 ~5s |
| Riffusion v2 | 음악 | Spectrogram diffusion | ~10s |

```figure
score-matching
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 핵심 아이디어를 시뮬레이션합니다: 두 가지 서로 다른 "스타일"(스타일 A는 저음과 고음 토큰이 교차하고, 스타일 B는 단조로운 램프 형태인 "오디오 토큰" 시퀀스)로부터 생성된 합성 데이터셋을 사용하여 아주 작은 차세대 토큰 트랜스포머(next-token transformer)를 학습시킵니다. 스타일과 샘플을 조건(condition)으로 사용해 보세요.

### 1단계: 합성 오디오 토큰 (synthetic audio tokens)

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": 교차 방식
        return [i % vocab_size for i in range(length)]
    # "music-like": 램프(ramp) 방식
    return [(i * 3) % vocab_size for i in range(length)]
```

### 2단계: 아주 작은 토큰 예측기(tiny token predictor) 학습하기

스타일에 따라 조건화(conditioned)되는 바이그램(bigram) 스타일의 예측기입니다. 핵심은 패턴입니다: 코덱 토큰(codec tokens) → 교차 엔트로피(cross-entropy) 학습 → 자기회귀 샘플링(autoregressive sampling).

### 3단계: 조건부 샘플링 (sample conditionally)

스타일 토큰(style token)과 시작 토큰(starting token)이 주어지면, 예측된 분포로부터 다음 토큰을 샘플링합니다. 이 과정을 20~40개의 토큰에 대해 반복해 보세요.

## 주의 사항 (Pitfalls)

- **코덱 품질이 출력 품질을 제한합니다 (Codec quality caps output quality).** 코덱이 소리를 충실하게 표현할 수 없다면, 생성기(generator)의 품질이 아무리 좋아도 소용이 없습니다. 현재 DAC가 오픈 소스 중 가장 뛰어난 성능을 보입니다.
- **RVQ 오류 누적 (RVQ error accumulation).** 각 RVQ 계층은 이전 계층의 잔차(residual)를 모델링합니다. 1계층에서의 오류는 계속 전파됩니다. 상위 계층에서 온도를 0으로 설정하여 샘플링하는 것이 도움이 됩니다.
- **음악적 구조 (Musical structure).** 75 Hz 기준 30초 분량의 토큰은 2만 개가 넘습니다. 이는 트랜스포머(transformer)에게 매우 어려운 작업입니다. MusicGen은 슬라이딩 윈도우(sliding window)와 프롬프트 연속(prompt continuation) 방식을 사용하며, Stable Audio는 더 짧은 클립과 크로스페이딩(crossfading)을 사용합니다.
- **경계면의 아티팩트 (Artifacts at boundaries).** 생성된 클립 사이를 크로스페이딩할 때는 세심한 중첩-더하기(overlap-add) 과정이 필요합니다.
- **깨끗한 데이터에 대한 갈증 (Clean-data appetite).** 음악 생성기에는 수만 시간의 라이선스가 확보된 음악 데이터가 필요합니다. Suno / Udio의 RIAA 소송(2024)이 이 문제를 수면 위로 끌어올렸습니다.
- **음성 복제 윤리 (Voice cloning ethics).** VALL-E / XTTS / ElevenLabs와 같은 모델은 3초 분량의 샘플과 텍스트 프롬프트만으로도 목소리를 복제할 수 있습니다. 모든 프로덕션 모델에는 오남용 탐지 기능과 거부 목록(opt-out lists)이 반드시 포함되어야 합니다.

## 활용하기 (Use It)

| 작업 (Task) | 2026년 스택 (2026 stack) |
|------|------------|
| 상용 TTS (Commercial TTS) | ElevenLabs, OpenAI TTS, 또는 Azure Neural |
| 음성 복제 (Voice cloning, 동의 확인됨) | XTTS v2 (오픈 소스) 또는 ElevenLabs Pro |
| 배경 음악, 빠른 생성 (Background music, fast) | Stable Audio 2.5 API, Suno, 또는 Udio |
| 가사가 포함된 음악 (Music with lyrics) | Suno v4 또는 Udio v1.5 |
| 효과음 / 폴리 (Sound effects / Foley) | AudioCraft 2, ElevenLabs SFX, 또는 Stable Audio Open |
| 실시간 음성 에이전트 (Real-time voice agent) | GPT-4o realtime 또는 Gemini Live |
| 오픈 웨이트 음악 연구 (Open-weights music research) | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 더빙 / 번역 (Dubbing / translation) | HeyGen, ElevenLabs Dubbing |

## Ship It (실행하기)

`outputs/skill-audio-brief.md`를 저장하세요. 이 스킬은 오디오 브리프(작업, 재생 시간, 스타일, 목소리, 라이선스)를 입력받아 다음 항목을 출력합니다: 모델 + 호스팅, 프롬프트 형식(장르 태그, 스타일 기술어, 구조적 마커), 코덱 + 생성기 + 보코더 체인, 시드 프로토콜, 그리고 평가 계획(MOS / CLAP 점수 / TTS를 위한 CER / 사용자 A/B 테스트).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하고 스타일을 명시적으로 설정해 보세요. 생성된 시퀀스가 해당 스타일의 패턴과 일치하는지 확인합니다.
2. **중간 (Medium).** 지연 병렬 디코딩(delayed parallel decoding)을 추가해 보세요: 반드시 1단계의 오프셋(offset)을 유지해야 하는 2개의 토큰 스트림을 시뮬레이션합니다. 이후 공동 예측기(joint predictor)를 학습시킵니다.
3. **어려움 (Hard).** HuggingFace `transformers`를 사용하여 MusicGen-small을 로컬에서 실행해 보세요. 세 가지 서로 다른 프롬프트를 사용하여 10초 길이의 클립을 생성하고, 스타일 준수 여부에 대해 A/B 테스트를 진행해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Codec (코덱) | "신경망 압축(Neural compression)" | 오디오를 위한 인코더/디코더; 일반적인 출력은 50-75 Hz 토큰입니다. |
| RVQ | "잔차 벡터 양자화(Residual VQ)" | K개의 양자화기(quantizer)가 직렬로 연결된 구조; 각 단계는 이전 단계의 잔차(residual)를 모델링합니다. |
| Token (토큰) | "하나의 코덱 심볼" | 코드북(codebook) 내의 이산 인덱스; 일반적으로 1024 또는 2048개를 사용합니다. |
| Delayed parallel (지연 병렬 방식) | "오프셋 코드북(Offset codebooks)" | 시퀀스 길이를 줄이기 위해 시차를 둔 오프셋을 사용하여 K개의 토큰 스트림을 방출합니다. |
| Flow matching (플로우 매칭) | "2024년 오디오 분야의 승리" | 확산(diffusion) 모델의 대안으로, 더 직선적인 경로를 가집니다; 샘플링 속도가 더 빠릅니다. |
| Voice prompt (음성 프롬프트) | "3초 샘플" | 복제할 목소리를 유도하는 화자 임베딩(speaker embedding) 또는 토큰 접두사(prefix)입니다. |
| Mel spectrogram (멜 스펙트로그램) | "시각적 데이터" | 로그 크기 기반의 인지적 스펙트로그램; 많은 TTS 시스템에서 사용됩니다. |
| Vocoder (보코더) | "Mel을 파형으로" | 멜 스펙트로그램을 다시 오디오로 변환하는 신경망 구성 요소입니다. |

## 프로덕션 노트: 오디오는 스트리밍 문제입니다 (Production note: audio is a streaming problem)

오디오는 사용자가 한꺼번에 받는 것이 아니라, *생성되는 즉시* 전달되기를 기대하는 유일한 출력 모달리티(modality)입니다. 프로덕션 관점에서 이는 TPOT(Time Per Output Token)가 중요하다는 것을 의미합니다. 왜냐하면 사용자의 청취 속도가 목표 처리량(throughput)이 되기 때문입니다. 사용자의 읽기 속도가 아닙니다. 초당 약 75개의 토큰으로 토큰화되는 16kHz 오디오(Encodec 기준)의 경우, 재생을 매끄럽게 유지하려면 서버는 사용자당 초당 75개 이상의 토큰을 생성해야 합니다.

두 가지 아키텍처 측면의 결과는 다음과 같습니다:

- **플로우 매칭(Flow-matching) 오디오 모델은 간단하게 스트리밍할 수 없습니다.** Stable Audio 2.5 및 AudioCraft 2는 한 번의 패스(pass)로 고정된 클립 길이를 렌더링합니다. 스트리밍을 하려면 클립을 청크(chunk)로 나누고 경계 부분을 중첩시켜야 합니다. 슬라이딩 윈도우 확산(sliding-window diffusion) 방식을 생각하면 되는데, 이는 코덱 AR 모델에 비해 100~300ms의 지연 시간(latency) 오버헤드를 추가합니다.

만약 제품이 "라이브 음성 채팅"이나 "실시간 음악 이어가기"라면, 코덱 AR(codec AR) 경로를 선택하세요. 만약 "제출 시 30초 클립 렌더링"이 목적이라면, 품질과 총 지연 시간 측면에서 플로우 매칭(flow-matching)이 유리합니다.

## 추가 읽을거리 (Further Reading)

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 코덱 표준입니다.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 널리 사용된 최초의 신경망 오디오 코덱입니다.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC입니다.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E입니다.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen입니다.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2입니다.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 플로우 매칭(flow matching)을 활용한 2025년형 텍스트-음악 생성 모델입니다.
