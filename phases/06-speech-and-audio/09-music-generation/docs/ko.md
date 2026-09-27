# 음악 생성(Music Generation) — MusicGen, Stable Audio, Suno, 그리고 라이선싱 대지진

> 2026년 음악 생성: Suno v5와 Udio v4가 상업 시장을 지배하고 있으며, MusicGen, Stable Audio Open, ACE-Step이 오픈 소스 분야를 선도하고 있습니다. 기술적 문제는 대부분 해결되었습니다. 법적 문제(Warner Music 5억 달러 합의, UMG 합의)가 2025-2026년 사이 이 분야의 지형을 재편했습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## 문제 (The Problem)

텍스트를 입력하면 가사, 보컬, 구조를 포함한 30초에서 4분 길이의 음악 클립을 생성합니다. 세 가지 하위 문제가 존재합니다:

1. **연주곡 생성 (Instrumental generation).** "따뜻한 건반이 어우러진 로파이 힙합 드럼(lo-fi hip-hop drums with warm keys)"과 같은 텍스트를 오디오로 변환합니다. MusicGen, Stable Audio, AudioLDM이 사용됩니다.
2. **곡 생성 (보컬 + 가사 포함) (Song generation (with vocals + lyrics)).** "비 내리는 텍사스의 밤에 관한 컨트리 송(Country song about rainy Texas nights)"과 같은 프롬프트로부터 전체 곡을 생성합니다. Suno, Udio, YuE, ACE-Step이 사용됩니다.
3. **조건부 / 제어 가능 생성 (Conditional / controllable).** 기존 클립 확장, 브릿지(bridge) 재생성, 장르 교체, 스템 분리(stem separation) 또는 인페인팅(inpaint)을 수행합니다. Udio의 인페인팅 및 스템 분리 기능은 2026년 기준 핵심 기능입니다.

## 개념 (The Concept)

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 신경망 코덱 토큰 기반의 토큰 언어 모델 (Token LM over neural-codec tokens)

Meta의 **MusicGen** (2023, MIT) 및 그 파생 모델들: 텍스트/멜로디 임베딩을 조건(condition)으로 사용하여 EnCodec 토큰(32 kHz, 4개 코드북)을 자기회귀(autoregressively) 방식으로 예측하고, EnCodec으로 디코딩합니다. 파라미터 수는 300M에서 3.3B 사이입니다. 강력한 베이스라인 모델이지만, 30초 이상의 긴 길이를 생성하는 데 어려움을 겪습니다.

**ACE-Step** (오픈 소스, 2026년 4월 4B XL 모델 출시)은 이를 확장하여 전체 노래의 가사 조건부 생성(lyric-conditioned generation)을 지원합니다. 오픈 소스 커뮤니티에서 Suno에 가장 근접한 모델입니다.

### 멜(mels) 또는 잠재 공간(latents)에서의 확산(Diffusion)

**Stable Audio (2023)** 및 **Stable Audio Open (2024)**: 압축된 오디오 상에서의 잠재 확산(latent diffusion) 모델입니다. 루프(loops), 사운드 디자인, 앰비언트 질감(ambient textures) 생성에 탁월합니다. 다만, 구조화된 전체 곡을 생성하는 데에는 적합하지 않습니다.

**AudioLDM / AudioLDM2**: T2I(Text-to-Image) 스타일의 잠재 확산을 통한 텍스트-오디오(text-to-audio) 모델로, 음악, 효과음, 음성 등으로 일반화되었습니다.

### 하이브리드(Hybrid, 프로덕션) — Suno, Udio, Lyria

폐쇄형 가중치(Closed weights) 모델입니다. AR 코덱 언어 모델(AR codec LM)과 특화된 보이스/드럼/멜로디 헤드를 갖춘 확산 기반 보코더(diffusion-based vocoder)가 결합된 형태일 가능성이 높습니다. Suno v5 (2026)는 ELO 1293 점수로 품질을 선도하고 있습니다. Udio v4는 인페인팅(inpainting)과 스템 분리(stem separation, 베이스/드럼/보컬 개별 다운로드) 기능을 추가했습니다.

### 평가 (Evaluation)

- **FAD (Fréchet Audio Distance).** VGGish 또는 PANNs 특징(features)을 사용하여 생성된 오디오 분포와 실제 오디오 분포 사이의 임베딩 수준 거리를 측정합니다. 값이 낮을수록 좋습니다. MusicGen small은 MusicCaps 데이터셋에서 4.5 FAD를 기록했으며, SOTA(최신 기술)는 약 3.0입니다.
- **음악성 (Musicality, 주관적 평가).** 인간의 선호도를 측정합니다. Suno v5가 ELO 1293점으로 선두를 달리고 있습니다.
- **텍스트-오디오 정렬 (Text-audio alignment).** 프롬프트와 출력물 사이의 CLAP 점수를 측정합니다.
- **음악적 아티팩트 (Musicality artifacts).** 박자가 맞지 않는 전환(off-beat transitions), 보컬 구절의 표류(vocal-phrase drift), 30초 이후의 구조 상실 등이 포함됩니다.

## 2026 모델 맵 (2026 model map)

| 모델 (Model) | 파라미터 (Params) | 길이 (Length) | 보컬 (Vocals) | 라이선스 (License) |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | 아니요 | MIT |
| Stable Audio Open | 1.2B | 47 s | 아니요 | Stability non-commercial |
| ACE-Step XL (2026년 4월) | 4B | &gt; 2 min | 예 | Apache-2.0 |
| YuE | 7B | &gt; 2 min | 예, 다국어 지원 | Apache-2.0 |
| Suno v5 (폐쇄형) | ? | 4 min | 예, ELO 1293 | 상업용 |
| Udio v4 (폐쇄형) | ? | 4 min | 예 + stems | 상업용 |
| Google Lyria 3 (폐쇄형) | ? | 실시간 (real-time) | 예 | 상업용 |
| MiniMax Music 2.5 | ? | 4 min | 예 | 상업용 API |

## 법적 환경 (The legal landscape) (2025-2026)

- **Warner Music vs Suno 합의.** 5억 달러 규모. WMG는 이제 Suno 내의 AI 유사성(AI-likeness), 음악 권리 및 사용자 생성 트랙에 대한 감독권을 가집니다. Udio에 대해서도 유사한 UMG 합의가 이루어졌습니다.
- **EU AI Act** + **California SB 942**: AI가 생성한 음악은 반드시 공개(disclosure)되어야 합니다.
- **Riffusion / MusicGen** (MIT 라이선스)은 규제 준수 부담은 없으나, 상업적 보컬을 포함할 수 없습니다.

안전한 배포 패턴(Safe-to-ship patterns):

1. 연주곡(Instrumental)만 생성 (MusicGen, Stable Audio Open, MIT/CC0 출력물).
2. 생성당 라이선스가 부여되는 상업용 API 사용 (Suno, Udio, ElevenLabs Music).
3. 소유권이 있거나 라이선스를 확보한 카탈로그로 학습 (대부분의 기업이 최종적으로 선택하는 방식).
4. 생성물에 워터마크 및 메타데이터 태그 부착.

```figure
sp-codec-tokens
```

## 직접 구현해 보기 (Build It)

### 1단계: MusicGen으로 생성하기

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

세 가지 크기가 있습니다: `small` (300M, 빠름), `medium` (1.5B), `large` (3.3B). 아이디어가 적절한지 확인하는 용도로는 `small` 모델로도 충분합니다.

### 2단계: 멜로디 컨디셔닝 (Melody Conditioning)

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody는 크로마그램(chromagram)을 입력받아 음색(timbre)을 바꾸면서도 멜로디를 유지합니다. 이는 "이 멜로디를 현악 4중주 버전으로 만들어 줘"와 같은 요청을 수행할 때 유용합니다.

### 3단계: FAD 평가 (FAD evaluation)

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish 임베딩 거리를 계산합니다. 장르 수준의 회귀 테스트(regression tests)에 유용하지만, 인간 청취자를 완전히 대체할 수는 없습니다.

### 4단계: LLM-음악 워크플로우(workflow)에 추가하기

레슨 7-8의 아이디어와 결합해 보세요:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 활용 방법 (Use It)

| 목표 (Goal) | 스택 (Stack) |
|------|-------|
| 악기 사운드 디자인 (Instrumental sound design) | Stable Audio Open |
| 게임 / 적응형 음악 (Game / adaptive music) | Google Lyria RealTime (폐쇄형) |
| 보컬이 포함된 완곡 (상업용) (Full songs with vocals (commercial)) | 명시적 라이선스가 있는 Suno v5 또는 Udio v4 |
| 보컬이 포함된 완곡 (오픈 소스) (Full songs with vocals (open)) | ACE-Step XL 또는 YuE |
| 짧은 광고 징글 (Short ad jingle) | 허밍 참조를 기반으로 한 MusicGen melody-conditioned |
| 뮤직비디오 배경 (Music-video background) | MusicGen + Stable Video Diffusion |

## 2026년에도 여전히 발생하는 함정들 (Pitfalls that still ship in 2026)

- **저작권 세탁 프롬프트 (Copyright-laundering prompts).** "Taylor Swift 스타일의 노래"와 같은 방식입니다. 상용 Suno/Udio는 현재 이를 필터링하지만, 오픈 모델은 필터링하지 않습니다. 직접 필터 목록을 추가해 보세요.
- **30초 이후의 반복 및 드리프트 (Repetition / drift past 30 s).** AR 모델은 루프(loop) 현상이 발생할 수 있습니다. 여러 생성물을 크로스페이드(crossfade)하거나, 구조적 일관성을 위해 ACE-Step을 사용해 보세요.
- **템포 드리프트 (Tempo drift).** 모델이 BPM에서 벗어나는 경우가 있습니다. 프롬프트에 BPM 태그를 사용하고, `librosa`의 `beat_track`으로 사후 필터링을 수행해 보세요.
- **보컬 명료도 (Vocal intelligibility).** Suno는 매우 뛰어나지만, 오픈 모델은 단어 발음이 뭉개지는 경우가 많습니다. 가사가 중요하다면 상용 API를 사용하거나 미세 조정(fine-tuning)을 진행해 보세요.
- **모노 출력 (Mono output).** 오픈 모델은 모노 또는 가짜 스테레오(fake-stereo)를 생성합니다. 적절한 스테레오 재구성 기술(ezst, Cartesia의 stereo diffusion 등)을 사용하여 업그레이드해 보세요.

## Ship It (실전 적용)

`outputs/skill-music-designer.md`로 저장하세요. 음악 생성(music-gen) 배포를 위해 모델, 라이선스 전략, 길이/구조 계획, 그리고 공개 메타데이터를 선택해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. ASCII 기호로 "생성적(generative)" 코드 진행과 드럼 패턴을 출력합니다 — 일종의 음악 생성 카툰입니다. 원한다면 MIDI 렌더러를 통해 재생해 보세요.
2. **중간 (Medium).** `audiocraft`를 설치하고, MusicGen-small을 사용하여 4가지 장르 프롬프트에 대해 10초 길이의 클립을 생성한 뒤, 참조 장르 세트와 비교하여 FAD를 측정해 보세요.
3. **어려움 (Hard).** ACE-Step(또는 MusicGen-melody)을 사용하여 서로 다른 음색(timbre) 프롬프트를 가진 동일한 곡의 세 가지 변형을 생성해 보세요. 프롬프트와의 정렬(alignment)을 확인하기 위해 CLAP 유사도를 계산해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| FAD | Audio FID | 실제 데이터와 생성된 데이터의 임베딩 분포 간의 Fréchet 거리. |
| Chromagram | 음높이로서의 멜로디 | 프레임당 12차원 벡터; 멜로디 조건부 생성(melody conditioning)의 입력값. |
| Stems | 악기 트랙 | 분리된 베이스 / 드럼 / 보컬 / 멜로디를 WAV 형식으로 추출한 것. |
| Inpainting | 특정 구간 재생성 | 시간 창(time window)을 마스킹하여 모델이 해당 구간만 재생성하도록 하는 것. |
| CLAP | Text-audio CLIP | 대조적 오디오-텍스트 임베딩; 텍스트와 오디오 간의 정렬(alignment)을 평가함. |
| EnCodec | 음악 코덱 | MusicGen에서 사용하는 Meta의 신경망 코덱; 32 kHz, 4개의 코드북(codebooks) 사용. |

## 추가 읽을거리 (Further Reading)

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 오픈 소스 자기회귀(autoregressive) 벤치마크입니다.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 사운드 디자인의 표준 모델입니다.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 2026년 4월 출시된 오픈 소스 4B 파라미터 전체 곡 생성기입니다.
- [Suno v5 platform docs](https://suno.com) — 상업적 품질을 선도하는 플랫폼입니다.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 음악 및 효과음을 위한 잠재 확산(latent diffusion) 모델입니다.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025년 11월의 선례를 다룬 기사입니다.
