# 음성 안티 스푸핑 및 오디오 워터마킹 — ASVspoof 5, AudioSeal, WaveVerify

> 음성 클로닝은 방어 기술보다 더 빠르게 출시되었습니다. 2026년 프로덕션 음성 시스템은 두 가지가 필요합니다: 실제 음성인지 가짜 음성인지 분류하는 탐지 모델(AASIST, RawNet2)과 압축 및 편집에도 살아남는 워터마크(AudioSeal)입니다. 둘 다 출시하거나, 음성 클로닝을 출시하지 마세요.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 06강 (화자 인식), 6단계 · 08강 (음성 클로닝)
**시간:** 약 75분

## 문제점

세 가지 관련 방어 기술:

1. **안티 스푸핑 / 딥페이크 탐지.** 오디오 클립이 합성된 것인지 실제인지 판별합니다. ASVspoof 벤치마크(ASVspoof 2019 → 2021 → 5)가 표준입니다.
2. **오디오 워터마킹.** 생성된 오디오에 지각할 수 없는 신호를 삽입하여 나중에 탐지기가 추출할 수 있게 합니다. AudioSeal (Meta)와 WavMark가 오픈 옵션입니다.
3. **인증된 출처.** 오디오 파일과 메타데이터의 암호화 서명. C2PA / Content Authenticity Initiative.

탐지는 협력하지 않는 적대자를 처리합니다. 워터마킹은 컴플라이언스를 처리합니다 — AI 생성 오디오는 AI 생성된 것으로 식별 가능해야 합니다. 2026년에는 둘 다 필요합니다.

## 개념

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 벤치마크

이전 판과 가장 큰 차이점:

- **크라우드소싱 데이터** (스튜디오 클린 데이터가 아님) — 현실적인 조건.
- **약 2000명의 화자** (이전 약 100명 대비).
- **32가지 공격 알고리즘.** TTS + 음성 변환 + 적대적 섭동.
- **두 트랙.** 카운터메저(CM) 단독 탐지; 생체 인식 시스템을 위한 스푸핑 강건 ASV(SASV).

ASVspoof 5에서의 최신 기술: 약 7.23% EER. 이전 ASVspoof 2019 LA에서는 0.42% EER. 실제 배포에서는 야생 클립에 대해 5-10% EER를 예상하세요.

### AASIST 및 RawNet2 — 탐지 모델 계열

**AASIST** (2021, 2026까지 업데이트). 스펙트럼 특성에 대한 그래프 어텐션. ASVspoof 5 카운터메저 작업에서 현재 SOTA입니다.

**RawNet2.** 원시 파형에 대한 컨볼루션 프런트엔드 + TDNN 백본. 더 단순한 기본선; 미세 조정과 경쟁력 있는 성능을 유지합니다.

**NeXt-TDNN + SSL 기능.** 2025 변형: ECAPA 스타일 + WavLM 기능 + 초점 손실. ASVspoof 2019 LA에서 0.42% EER를 달성합니다.

### AudioSeal — 2024년 워터마크 기본값

Meta의 **AudioSeal** (2024년 1월, v0.2는 2024년 12월). 주요 설계:

- **지역화됨.** 16 kHz 샘플 해상도(1/16000초)에서 프레임별로 워터마크를 감지합니다.
- **생성기와 감지기를 공동 학습합니다.** 생성기는 들리지 않는 신호를 삽입하는 것을 학습하고, 감지기는 증강을 통해 이를 찾는 것을 학습합니다.
- **강건합니다.** MP3 / AAC 압축, EQ, ±10% 속도 변경, +10 dB SNR 잡음 혼합을 견딥니다.
- **빠릅니다.** 감지기는 실시간의 485배로 실행되며, WavMark보다 1000배 빠릅니다.
- **용량.** 16비트 페이로드(모델 ID, 생성 타임스탬프, 사용자 ID를 인코딩할 수 있음)를 각 발화에 삽입할 수 있습니다.

### WavMark

AudioSeal 이전의 오픈 기본선. 가역 신경 네트워크, 초당 32비트. 문제점:

- 동기화 브루트 포스가 느립니다.
- 가우시안 잡음이나 MP3 압축으로 제거될 수 있습니다.
- 실시간 친화적이지 않습니다.

### WaveVerify (2025년 7월)

AudioSeal의 약점, 특히 시간적 조작(역전, 속도)을 해결합니다. FiLM 기반 생성기 + 혼합 전문가(MoE) 감지기를 사용합니다. 표준 공격에 대해 AudioSeal과 경쟁력 있는 성능을 보이며, 시간적 편집을 처리합니다.

### 적대자가 악용하는 격차

AudioMarkBench에 따르면: "피치 시프트 하에서 모든 워터마크는 비트 복원 정확도가 0.6 미만으로, 거의 완전한 제거를 나타냅니다." **피치 시프트는 보편적인 공격입니다.** 2026년 워터마크 중 공격적인 피치 수정에 완전히 강건한 것은 없습니다. 이 때문에 워터마킹과 함께 감지(AASIST)가 필요합니다.

### C2PA / 콘텐츠 진위성 이니셔티브

ML 기술이 아닌 매니페스트 형식입니다. 오디오 파일은 생성 도구, 저자, 날짜에 대한 암호화 서명된 메타데이터를 포함합니다. Audobox / Seamless가 이를 사용합니다. 출처 추적에 유용하지만, 악의적인 행위자가 재인코딩하고 메타데이터를 제거하면 아무런 효과가 없습니다.

```figure
v4-audio-watermark
```

## 구현하기

### 1단계: 간단한 스펙트럼 기능 감지기(토이)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

합성 음성에는 고주파 에너지가 비정상적으로 평평한 경우가 많습니다. 생산 환경에서는 AASIST를 사용하며, 이 방법은 사용하지 않습니다. 하지만 직관은 유효합니다.

### 2단계: AudioSeal 임베딩 + 감지

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 범위의 float 값 — 워터마크 존재 확률
# decoded_payload: 16비트; 임베딩된 페이로드와 일치 여부 확인
```

### 3단계: 평가 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 4단계: 생산 환경 통합

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

모든 생성물에는 (1) 워터마크, (2) 서명된 매니페스트, (3) 보존 정책 준수 감사 로그가 포함됩니다.

## 사용하기

| 사용 사례 | 방어 |
|----------|---------|
| TTS / 음성 클로닝 전송 | 모든 출력에 AudioSeal 임베딩 (타협 불가) |
| 생체 음성 잠금 해제 | AASIST + ECAPA 앙상블; 라이브니스 챌린지 |
| 콜센터 사기 탐지 | 들어오는 통화의 20% 샘플에 AASIST 적용 |
| 팟캐스트 진위 확인 | 업로드 시 C2PA 서명, AI 생성 시 AudioSeal |
| 연구 / 탐지 모델 훈련 | ASVspoof 5 train/dev/eval 세트 |

## 주의할 점

- **감지기가 실행되지 않는 워터마크.** 무의미합니다. CI에 감지기를 포함하세요.
- **보정 없는 감지.** ASVspoof LA로 훈련된 AASIST는 과적합되며, 실제 환경에서의 정확도가 떨어집니다. 자신의 도메인에서 보정하세요.
- **피치 시프트 간격.** 공격적인 피치 시프트는 대부분의 워터마크를 제거합니다. 감지 폴백을 준비하세요.
- **메타데이터 제거 및 재호스팅.** C2PA는 재인코딩으로 쉽게 우회됩니다. 항상 암호화 + 지각적(워터마크) 방어를 함께 추가하세요.
- **감지로 취급되는 라이브니스.** 사용자에게 랜덤 문구를 말하게 하세요. 리플레이 공격은 방지하지만, 실시간 클로닝은 방지하지 못합니다.

## 출시하기

`outputs/skill-spoof-defender.md`로 저장하세요. 음성 생성 배포를 위해 감지 모델, 워터마크, 출처 매니페스트 및 운영 플레이북을 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 합성 오디오에 토이 감지기 + 토이 워터마크 임베딩/감지를 수행하세요.
2. **중간.** `audioseal`를 설치하고, TTS 출력에 16비트 페이로드를 임베딩한 후 재디코딩하세요. 오디오에 잡음을 추가하여 비트 복구 정확도를 측정하세요.
3. **난이도: 높음.** ASVspoof 2019 LA 데이터셋으로 RawNet2 또는 AASIST를 미세 조정(Fine-tuning)하세요. EER을 측정하고, F5-TTS로 생성된 클립의 홀드아웃 세트에서 테스트하여 OOD 탐지 성능이 얼마나 저하되는지 확인해 보세요.

## 핵심 용어

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| ASVspoof | 벤치마크 | 격년 대회; 2024년은 ASVspoof 5입니다. |
| CM (대응책) | 탐지자 | 분류기: 실제 음성 vs 합성/변환된 음성. |
| SASV | 화자 검증 + CM | 생체 인증과 스푸핑 탐지를 통합한 시스템. |
| AudioSeal | Meta 워터마크 | 지역화된 16비트 페이로드, WavMark보다 485배 빠름. |
| 비트 복원 정확도 | 워터마크 생존율 | 공격 후 페이로드 비트가 복원된 비율. |
| C2PA | 출처 매니페스트 | 생성/저작자에 대한 암호화 메타데이터. |
| AASIST | 탐지자 계열 | 그래프 어텐션 기반 스푸핑 방지 SOTA. |

## 추가 읽기

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 현재 벤치마크.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 워터마크 기본 설정.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 시간적 공격을 위한 MoE 탐지자.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 탐지 백본.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 강건성 평가.
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html) — 출처 매니페스트 형식.
