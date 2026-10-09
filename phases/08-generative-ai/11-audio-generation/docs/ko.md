# 오디오 생성

> 오디오는 16-48 kHz의 1차원 신호입니다. 5초 클립은 80-240k 샘플입니다. 트랜스포머는 해당 시퀀스를 직접 처리하지 않습니다. 2026년 모든 프로덕션 오디오 모델의 해결책은 동일합니다: 뉴럴 코덱(Encodec, SoundStream, DAC)이 오디오를 50-75 Hz의 이산 토큰으로 압축하고, 트랜스포머 또는 확산 모델이 토큰을 생성합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (오디오 특징), 6단계 · 04강 (ASR), 8단계 · 06강 (DDPM)
**시간:** 약 45분

## 문제점

세 가지 오디오 생성 작업:

1. **텍스트 음성 변환.** 텍스트가 주어지면 음성을 생성합니다. 깨끗한 음성은 좁은 대역폭을 가지며 강한 음운 구조를 지니므로, 토큰 기반 트랜스포머로 잘 해결됩니다. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **음악 생성.** 프롬프트(텍스트, 멜로디, 코드 진행, 장르)가 주어지면 음악을 생성합니다. 분포가 훨씬 넓습니다. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **오디오 효과 / 사운드 디자인.** 프롬프트가 주어지면 앰비언트 사운드나 Foley를 생성합니다. AudioGen, AudioLDM 2, Stable Audio Open.

세 가지 모두 동일한 기반 위에서 실행됩니다: 뉴럴 오디오 코덱 + 토큰 AR 또는 확산 생성기.

## 개념

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 뉴럴 오디오 코덱

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). 컨볼루션 인코더가 파형을 시간 단계별 벡터로 압축하며, 잔차 벡터 양자화(RVQ)가 각 벡터를 K 코드북 인덱스의 캐스케이드로 변환합니다. 디코더가 이를 역으로 수행합니다. 24 kHz 오디오를 2 kbps로 사용하며, 8개의 RVQ 코드북을 75 Hz에서 사용 = 초당 600 토큰.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 그 위에 두 가지 생성 패러다임

**토큰 자기회귀.** RVQ 토큰을 시퀀스로 평탄화하고 디코더 전용 트랜스포머를 실행합니다. MusicGen은 지연 병렬(delayed parallel) 방식을 사용하여 K 코드북 스트림을 스트림별 오프셋과 함께 병렬로 방출합니다. VALL-E는 텍스트 프롬프트 + 3초 음성 샘플에서 음성 토큰을 생성합니다.

**잠재 확산(Latent diffusion).** 코덱 토큰을 연속 잠재 변수(latent)로 포장하거나 범주형 확산으로 모델링합니다. Stable Audio 2.5는 연속 오디오 잠재 변수에 flow matching을 사용하며, AudioLDM 2는 텍스트-멜-오디오 확산을 사용합니다.

2024-2026년 추세: flow matching은 음악 분야에서 승리하고 있습니다(더 빠른 추론, 더 깨끗한 샘플). 반면 토큰 기반 자기회귀(Token-AR)는 자연적으로 인과적이며 스트리밍이 잘 되기 때문에 음성 분야에서 여전히 지배적입니다.

## 생산 환경

| 시스템 | 작업 | 백본 | 지연 시간 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | 토큰 기반 자기회귀 + 뉴럴 보코더 | 첫 토큰까지 약 300ms |
| OpenAI GPT-4o audio | 풀 듀플렉스 음성 | 엔드투엔드 멀티모달 자기회귀 | 약 200ms |
| NaturalSpeech 3 | TTS | 잠재 변수 flow matching | 비스트리밍 |
| Stable Audio 2.5 | 음악 / SFX | DiT + 오디오 잠재 변수에 대한 flow matching | 1분 클립에 약 10초 |
| Suno v4 | 전체 노래 | 비공개; 토큰 기반 자기회귀로 추정 | 노래당 약 30초 |
| Udio v1.5 | 전체 노래 | 비공개 | 노래당 약 30초 |
| MusicGen 3.3B | 음악 | Encodec 32kHz에 대한 토큰 기반 자기회귀 | 실시간 |
| AudioCraft 2 | 음악 + SFX | Flow matching | 5초 클립에 약 5초 |
| Riffusion v2 | 음악 | 스펙트로그램 확산 | 약 10초 |

```figure
score-matching
```

## 구현하기

`code/main.py`는 핵심 아이디어를 시뮬레이션합니다: 두 가지 서로 다른 "스타일"(스타일 A는 낮은 토큰과 높은 토큰이 번갈아 나타나고, 스타일 B는 단조로운 상승)로 생성된 합성 "오디오 토큰" 시퀀스에 작은 next-token 트랜스포머를 학습시킵니다. 스타일에 조건을 부여하고 샘플링합니다.

### 1단계: 합성 오디오 토큰

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "음악 같은": 상승
    return [(i * 3) % vocab_size for i in range(length)]
```

### 2단계: 작은 토큰 예측기 학습

스타일에 조건을 부여한 bigram 스타일 예측기입니다. 요점은 패턴입니다: 코덱 토큰 → 교차 엔트로피 학습 → 자기회귀 샘플링.

### 3단계: 조건부 샘플링

스타일 토큰과 시작 토큰이 주어지면, 예측된 분포에서 다음 토큰을 샘플링합니다. 20-40개 토큰까지 계속 진행합니다.

## 문제점

- **코덱 품질이 출력 품질의 상한을 결정합니다.** 코덱이 소리를 충실히 표현할 수 없다면, 생성기의 품질이 아무리 좋아도 도움이 되지 않습니다. DAC는 현재 오픈 소스 중 최선입니다.
- **RVQ 오차 누적.** 각 RVQ 레이어는 이전 레이어의 잔차를 모델링합니다. 1번째 레이어의 오차가 전파됩니다. 상위 레이어에서 온도 0으로 샘플링하면 도움이 됩니다.
- **음악적 구조.** 30초의 토큰은 75 Hz에서 2만 개 이상의 토큰입니다. 트랜스포머에는 어렵습니다. MusicGen은 슬라이딩 윈도우 + 프롬프트 연속성을 사용하며, Stable Audio는 짧은 클립 + 크로스페이딩을 사용합니다.
- **경계에서의 아티팩트.** 생성된 클립 간의 크로스페이딩에는 주의 깊은 오버랩 애드(overlap-add)가 필요합니다.
- **클린 데이터 요구량.** 음악 생성기는 수만 시간의 라이선스된 음악이 필요합니다. Suno / Udio의 RIAA 소송(2024)은 이 문제를 표면화했습니다.
- **보이스 클로닝 윤리.** 3초 샘플과 텍스트 프롬프트만으로도 VALL-E / XTTS / ElevenLabs가 보이스를 클로닝할 수 있습니다. 모든 프로덕션 모델은 남용 감지 + 옵트아웃 목록이 필요합니다.

## 사용하기

| 작업 | 2026 스택 |
|------|------------|
| 상용 TTS | ElevenLabs, OpenAI TTS, 또는 Azure Neural |
| 보이스 클로닝 (동의 확인됨) | XTTS v2 (오픈) 또는 ElevenLabs Pro |
| 배경 음악, 빠른 생성 | Stable Audio 2.5 API, Suno, 또는 Udio |
| 가사가 있는 음악 | Suno v4 또는 Udio v1.5 |
| 사운드 효과 / Foley | AudioCraft 2, ElevenLabs SFX, 또는 Stable Audio Open |
| 실시간 음성 에이전트 | GPT-4o realtime 또는 Gemini Live |
| 오픈 웨이트 음악 연구 | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 더빙 / 번역 | HeyGen, ElevenLabs Dubbing |

## 출시하기

`outputs/skill-audio-brief.md`을 저장하세요. 스킬은 오디오 브리프(작업, 길이, 스타일, 보이스, 라이선스)를 받아 모델 + 호스팅, 프롬프트 형식(장르 태그, 스타일 설명자, 구조적 마커), 코덱 + 생성기 + 보코더 체인, 시드 프로토콜, 평가 계획(MOS / CLAP 점수 / TTS의 CER / 사용자 A/B)을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하고 스타일을 명시적으로 설정하세요. 생성된 시퀀스가 스타일의 패턴과 일치하는지 확인하세요.
2. **중간.** 지연된 병렬 디코딩을 추가하세요: 1단계 오프셋을 유지해야 하는 2개의 토큰 스트림을 시뮬레이션하고, 결합 예측기를 학습하세요.
3. **난이도: 상.** HuggingFace transformers를 사용하여 MusicGen-small을 로컬에서 실행합니다. 세 가지 다른 프롬프트로 10초 클립을 생성하고, 스타일 준수 여부를 A/B 테스트해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Codec | "신경망 압축" | 오디오용 인코더 / 디코더; 일반적인 출력은 50-75 Hz 토큰입니다. |
| RVQ | "잔차 VQ" | K개의 양자화기 캐스케이드; 각각이 이전 양자화기의 잔차를 모델링합니다. |
| 토큰 | "하나의 codec 심볼" | 코드북의 이산 인덱스; 일반적으로 1024 또는 2048입니다. |
| 지연 병렬 | "오프셋 코드북" | 시퀀스 길이를 줄이기 위해 K개의 토큰 스트림을 오프셋을 두고 순차적으로 방출합니다. |
| Flow matching | "2024년 오디오 분야의 승리" | 확산 모델보다 더 직선적인 경로를 사용하는 대안; 샘플링이 더 빠릅니다. |
| Voice prompt | "3초 샘플" | 클론된 목소리를 유도하는 화자 임베딩 또는 토큰 접두어입니다. |
| Mel spectrogram | "시각적 표현" | 로그 크기 지각 스펙트로그램; 많은 TTS 시스템에서 사용됩니다. |
| Vocoder | "Mel에서 파형으로" | Mel 스펙트로그램을 오디오로 변환하는 신경망 구성 요소입니다. |

## 프로덕션 노트: 오디오는 스트리밍 문제입니다

오디오는 사용자가 *생성되는 대로* 도착하기를 기대하는 유일한 출력 모달리티이며, 한 번에 모두 도착하는 것을 기대하지 않습니다. 프로덕션 관점에서 이는 TPOT (출력 토큰당 시간)가 중요하다는 의미입니다. 왜냐하면 사용자의 청취 속도가 목표 처리량이기 때문입니다. 읽기 속도가 아닙니다. ~75 토큰/초로 토큰화된 16kHz 오디오(Encodec)의 경우, 재생이 매끄럽게 유지되려면 서버는 사용자당 ≥75 토큰/초를 생성해야 합니다.

두 가지 아키텍처적 결과:

- **Flow-matching 오디오 모델은 간단하게 스트리밍할 수 없습니다.** Stable Audio 2.5와 AudioCraft 2는 고정된 클립 길이를 한 번의 패스로 렌더링합니다. 스트리밍하려면 클립을 청킹하고 경계를 겹쳐야 합니다 — 슬라이딩 윈도우 확산을 생각해 보세요. 이는 codec AR 모델에 비해 100-300ms의 지연 오버헤드를 추가합니다.

제품이 "라이브 음성 채팅"이나 "실시간 음악 연속 생성"이라면 codec AR 경로를 선택하세요. "제출 시 30초 클립 렌더링"이라면 flow-matching이 품질과 총 지연 시간에서 승리합니다.

## 추가 읽기

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec 표준입니다.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 널리 사용된 최초의 신경망 오디오 codec입니다.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC입니다.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E입니다.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen입니다.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2입니다.
- [Stability AI (2025). Stable Audio 2.5](https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale) — 2025년 플로우 매칭을 활용한 텍스트-음악 생성입니다.
