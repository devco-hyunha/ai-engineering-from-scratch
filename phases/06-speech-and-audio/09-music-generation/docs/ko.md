# 음악 생성 — MusicGen, Stable Audio, Suno, 그리고 라이선싱 지진

> 2026년 음악 생성: Suno v5와 Udio v4가 상용 시장을 지배하고 있으며, MusicGen, Stable Audio Open, ACE-Step이 오픈소스를 주도하고 있습니다. 기술적 문제는 대부분 해결되었습니다. 법적 문제(Warner Music의 5억 달러 합의, UMG 합의)는 2025-2026년에 이 분야를 재편했습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램), 4단계 · 10강 (확산 모델)
**시간:** 약 75분

## 문제점

텍스트 → 가사, 보컬, 구조를 포함한 30초에서 4분 길이의 음악 클립. 세 가지 하위 문제:

1. **악기 생성.** "lo-fi hip-hop drums with warm keys"와 같은 텍스트 → 오디오. MusicGen, Stable Audio, AudioLDM.
2. **노래 생성 (보컬 + 가사 포함).** "Country song about rainy Texas nights" → 전체 노래. Suno, Udio, YuE, ACE-Step.
3. **조건부 / 제어 가능.** 기존 클립 확장, 브리지 재생성, 장르 교체, 스템 분리, 인페인팅. Udio의 인페인팅 + 스템 분리는 2026년에 벤치마킹해야 할 기능입니다.

## 개념

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 신경 코덱 토큰에 대한 토큰 LM

Meta의 **MusicGen** (2023, MIT) 및 많은 파생 모델: 텍스트/멜로디 임베딩에 조건을 적용하고, EnCodec 토큰(32 kHz, 4 코드북)을 자기회귀적으로 예측하며, EnCodec으로 디코딩합니다. 3억 ~ 33억 파라미터. 강력한 기준선; 30초 이후에는 어려움을 겪습니다.

**ACE-Step** (오픈소스, 4B XL이 2026년 4월에 공개됨)은 전체 노래의 가사 조건부 생성을 위해 이를 확장합니다. 오픈 커뮤니티에서 Suno에 가장 가까운 모델입니다.

### 멜 스펙트럼 또는 잠재 변수에 대한 확산

**Stable Audio (2023)** 및 **Stable Audio Open (2024)**: 압축된 오디오에 대한 잠재 확산. 루프, 사운드 디자인, 앰비언트 텍스처에 뛰어납니다. 구조화된 전체 노래에는 적합하지 않습니다.

**AudioLDM / AudioLDM2**: T2I 스타일의 잠재 확산을 통한 텍스트-오디오 변환, 음악, 효과음, 음성으로 일반화됨.

### 하이브리드 (상용) — Suno, Udio, Lyria

가중치가 비공개입니다. AR 코덱 LM + 확산 기반 보코더와 전문화된 보이스 / 드럼 / 멜로디 헤드를 사용하는 것으로 추정됩니다. Suno v5 (2026)는 ELO 1293 품질 리더입니다. Udio v4는 인페인팅 + 스템 분리(베이스, 드럼, 보컬을 개별 다운로드)를 추가했습니다.

### 평가

- **FAD (Fréchet Audio Distance).** VGGish 또는 PANNs 특징을 사용하여 생성 오디오와 실제 오디오 분포 간의 임베딩 수준 거리입니다. 낮을수록 좋습니다. MusicGen small: MusicCaps에서 4.5 FAD; SOTA는 약 3.0입니다.
- **음악성 (주관적).** 인간 선호도입니다. Suno v5가 ELO 1293으로 선두를 달리고 있습니다.
- **텍스트-오디오 정렬.** 프롬프트와 출력 간의 CLAP 점수입니다.
- **음악성 아티팩트.** 박자 이탈 전환, 보컬 구문 드리프트, 30초 이후 구조 손실.

## 2026 모델 맵

| 모델 | 매개변수 | 길이 | 보컬 | 라이선스 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30초 | 없음 | MIT |
| Stable Audio Open | 1.2B | 47초 | 없음 | Stability 비영리 |
| ACE-Step XL (2026년 4월) | 4B | 2분 이상 | 있음 | Apache-2.0 |
| YuE | 7B | 2분 이상 | 있음, 다국어 | Apache-2.0 |
| Suno v5 (비공개) | ? | 4분 | 있음, ELO 1293 | 상업적 |
| Udio v4 (비공개) | ? | 4분 | 있음 + 스템 | 상업적 |
| Google Lyria 3 (비공개) | ? | 실시간 | 있음 | 상업적 |
| MiniMax Music 2.5 | ? | 4분 | 있음 | 상업적 API |

## 법적 환경 (2025-2026)

- **Warner Music vs Suno 합의.** 5억 달러. WMG는 현재 Suno의 AI 유사성, 음악 권리 및 사용자 생성 트랙에 대한 감독권을 가지고 있습니다. Udio에 대한 유사한 UMG 합의도 있습니다.
- **EU AI Act** + **California SB 942**: AI 생성 음악은 공개되어야 합니다.
- **Riffusion / MusicGen**은 MIT 라이선스 하에 컴플라이언스 부담이 없지만, 상업적 보컬도 없습니다.

출시하기 안전한 패턴:

1. 악기 전용 생성 (MusicGen, Stable Audio Open, MIT/CC0 출력).
2. 상업적 API (Suno, Udio, ElevenLabs Music)를 생성별 라이선스와 함께 사용하세요.
3. 소유하거나 라이선스된 카탈로그로 학습하세요 (대부분의 기업은 이 방법을 선택합니다).
4. 생성물에 워터마크 + 메타데이터를 태그하세요.

```figure
sp-codec-tokens
```

## 구현하기

### 1단계: MusicGen으로 생성

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

세 가지 크기: `small` (300M, 빠름), `medium` (1.5B), `large` (3.3B). "아이디어가 잘 전달되는지" 확인하는 데는 작은 모델만으로도 충분합니다.

### 2단계: 멜로디 조건부 생성

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody는 크로마그램을 받아 음색을 바꾸면서도 멜로디는 유지합니다. "이 멜로디를 현악 4중주로 바꿔 주세요" 같은 요청에 유용합니다.

### 3단계: FAD 평가

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish 임베딩 거리를 계산합니다. 장르 수준의 회귀 테스트에 유용하며, 인간 청취자를 대체할 수는 없습니다.

### 4단계: LLM-음악 워크플로에 추가

7-8강의 아이디어와 결합하세요:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 사용하기

| 목표 | 스택 |
|------|-------|
| 악기 사운드 디자인 | Stable Audio Open |
| 게임 / 적응형 음악 | Google Lyria RealTime (비공개) |
| 보컬이 포함된 전체 곡 (상업용) | 명시적 라이선스가 있는 Suno v5 또는 Udio v4 |
| 보컬이 포함된 전체 곡 (오픈) | ACE-Step XL 또는 YuE |
| 짧은 광고 로고송 | 허밍한 참조에 조건부 생성된 MusicGen |
| 음악 비디오 배경 | MusicGen + Stable Video Diffusion |

## 2026년에도 여전히 출시되는 함정

- **저작권 세탁 프롬프트.** "Taylor Swift 스타일의 노래" — 상용 Suno/Udio는 이제 이를 필터링하지만, 오픈 모델은 필터링하지 않습니다. 자체 필터 목록을 추가하세요.
- **30초 이후 반복 / 드리프트.** AR 모델은 루프합니다. 여러 생성물을 크로스페이드하거나, 구조적 일관성을 위해 ACE-Step을 사용하세요.
- **템포 드리프트.** 모델이 BPM에서 벗어납니다. 프롬프트에 BPM 태그를 사용하고, librosa의 `beat_track`으로 사후 필터링하세요.
- **보명성.** Suno는 훌륭하지만, 오픈 모델은 단어가 뭉개지는 경우가 많습니다. 가사가 중요하면 상용 API를 사용하거나 미세 조정하세요.
- **모노 출력.** 오픈 모델은 모노 또는 가짜 스테레오를 생성합니다. 적절한 스테레오 복원(ezst, Cartesia의 스테레오 확산)으로 업그레이드하세요.

## 출시하기

`outputs/skill-music-designer.md`으로 저장하세요. 음악 생성 배포를 위해 모델, 라이선스 전략, 길이 / 구조 계획 및 공개 메타데이터를 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. "생성형" 코드 진행 + 드럼 패턴을 ASCII 기호로 생성합니다 — 음악 생성 만화입니다. 원한다면 MIDI 렌더러로 재생해 보세요.
2. **중간.** `audiocraft`을 설치하고, MusicGen-small로 4가지 장르 프롬프트를 사용하여 10초 클립을 생성한 후, 참조 장르 세트에 대해 FAD를 측정하세요.
3. **어려움.** ACE-Step (또는 MusicGen-melody)을 사용하여 동일한 곡의 세 가지 변형을 서로 다른 음색 프롬프트로 생성하세요. 프롬프트에 대한 CLAP 유사도를 계산하여 정렬을 검증하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| FAD | Audio FID | 실제 오디오와 생성된 오디오의 임베딩 분포 간 Fréchet 거리. |
| Chromagram | 멜로디를 음높이로 표현 | 프레임별 12차원 벡터; 멜로디 조건 입력으로 사용. |
| Stems | 악기 트랙 | 분리된 베이스 / 드럼 / 보컬 / 멜로디를 WAV로 제공. |
| Inpainting | 섹션 재생성 | 시간 윈도우를 마스킹하면 모델이 해당 부분만 재생성합니다. |
| CLAP | Text-audio CLIP | 대조 오디오-텍스트 임베딩; 텍스트-오디오 정렬을 평가합니다. |
| EnCodec | 음악 코덱 | MusicGen이 사용하는 Meta의 신경 코덱; 32 kHz, 4개 코드북. |

## 추가 읽기

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 오픈 자기회귀 벤치마크.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 사운드 디자인 기본 도구.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 오픈 4B 전체 곡 생성기, 2026년 4월.
- [Suno v5 platform docs](https://suno.com) — 상용 품질 리더.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 음악 + 사운드 효과를 위한 잠재 확산 모델.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/) — 2025년 11월 선례.
