# 음성 안티 스푸핑 및 오디오 워터마킹 (Voice Anti-Spoofing & Audio Watermarking) — ASVspoof 5, AudioSeal, WaveVerify

> 음성 복제(Voice cloning) 기술이 방어 기술보다 더 빠르게 발전했습니다. 2026년의 상용 음성 시스템에는 두 가지가 필요합니다: 실제 음성과 가짜 음성을 분류하는 탐지기(`AASIST`, `RawNet2`), 그리고 압축과 편집에도 살아남는 워터마크(`AudioSeal`)입니다. 이 두 가지를 모두 구현하거나, 아니면 음성 복제 기술을 출시하지 마십시오.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## 문제점 (The Problem)

세 가지 관련 방어 기제:

1. **안티 스푸핑 / 딥페이크 탐지 (Anti-spoofing / deepfake detection).** 오디오 클립이 주어졌을 때, 이것이 합성된 것인지 실제인지 판별합니다. ASVspoof 벤치마크(ASVspoof 2019 → 2021 → 5)가 표준 지표로 사용됩니다.
2. **오디오 워터마킹 (Audio watermarking).** 생성된 오디오에 탐지기가 나중에 추출할 수 있는 인지 불가능한 신호를 삽입합니다. AudioSeal (Meta)과 WavMark가 공개된 옵션입니다.
3. **인증된 출처 (Authenticated provenance).** 오디오 파일과 메타데이터에 암호화 서명을 수행합니다. C2PA / Content Authenticity Initiative가 이에 해당합니다.

탐지(Detection)는 협조하지 않는 공격자를 다룹니다. 워터마킹(Watermarking)은 규제 준수(Compliance)를 다룹니다. 즉, AI로 생성된 오디오는 그 사실을 식별할 수 있어야 합니다. 2026년에는 이 두 가지가 모두 필요할 것입니다.

## 개념 (The Concept)

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 벤치마크 (benchmark)

이전 에디션과 비교했을 때 가장 큰 변화는 다음과 같습니다:

- **크라우드소싱 데이터** (studio clean 데이터가 아님) — 현실적인 환경 반영.
- **약 2,000명의 화자** (이전의 약 100명 대비 대폭 증가).
- **32개의 공격 알고리즘.** TTS + 음성 변환(voice conversion) + 적대적 섭동(adversarial perturbation).
- **두 개의 트랙.** 대응책(Countermeasure, CM) 단독 탐지; 생체 인식 시스템을 위한 스푸핑 강건 ASV(Spoofing-robust ASV, SASV).

ASVspoof 5에서의 최신 기술 수준(SOTA): 약 7.23% EER. 이전 버전인 ASVspoof 2019 LA에서의 EER: 0.42%. 실제 환경 배포 시: 야생(in-the-wild) 클립에서 5-10%의 EER를 예상해야 합니다.

### AASIST 및 RawNet2 — 탐지 모델 제품군 (detection model families)

**AASIST** (2021, 2026년까지 업데이트됨). 스펙트럼 특징(spectral features)에 대한 그래프 어텐션(Graph-attention) 적용. ASVspoof 5 대응(countermeasure) 태스크의 현재 SOTA(State-of-the-Art).

**RawNet2.** 원시 파형(raw waveform)에 대한 컨볼루션 프론트엔드(Convolutional front-end) + TDNN 백본(backbone). 더 단순한 베이스라인이지만, 미세 조정(fine-tuning)을 통해 여전히 경쟁력을 유지합니다.

**NeXt-TDNN + SSL 특징(features).** 2025년 변형 모델: ECAPA 스타일 + WavLM 특징 + focal loss 적용. ASVspoof 2019 LA에서 0.42% EER 달성.

### AudioSeal — 2024년 워터마크의 표준 (the 2024 watermark default)

Meta의 **AudioSeal** (2024년 1월 출시, 2024년 12월 v0.2). 주요 설계 특징:

- **국소화(Localized).** 16 kHz 샘플 해상도(1/16000초)로 프레임당 워터마크를 감지합니다.
- **생성기 + 탐지기 공동 학습(Generator + detector jointly trained).** 생성기는 들리지 않는 신호를 삽입하는 법을 배우고, 탐지기는 데이터 증강(augmentation)을 통해 해당 신호를 찾는 법을 배웁니다.
- **강건성(Robust).** MP3 / AAC 압축, EQ, 속도 변화 ±10%, 노이즈 혼합 +10 dB SNR 환경에서도 유지됩니다.
- **속도(Fast).** 탐지기는 실시간 대비 485배 속도로 작동하며, WavMark보다 1000배 빠릅니다.
- **용량(Capacity).** 각 발화(utterance)에 삽입 가능한 16비트 페이로드(모델 ID, 생성 타임스탬프, 사용자 ID 등을 인코딩할 수 있음)를 가집니다.

### WavMark

AudioSeal 이전의 오픈 베이스라인(open baseline)입니다. 가역 신경망(Invertible neural network)을 사용하며, 초당 32비트(32 bits/sec)를 지원합니다. 문제점은 다음과 같습니다:

- 동기화 브루트 포스(Synchronization brute-force) 방식이 느립니다.
- 가우시안 노이즈(Gaussian noise)나 MP3 압축에 의해 제거될 수 있습니다.
- 실시간(real-time) 환경에 적합하지 않습니다.

### WaveVerify (2025년 7월)

AudioSeal의 약점, 특히 시간적 조작(temporal manipulations, 예: 역재생, 속도 조절)을 해결합니다. FiLM 기반 생성기(generator)와 전문가 혼합(Mixture-of-Experts) 탐지기를 사용합니다. 표준 공격에 대해서는 AudioSeal과 대등한 성능을 보이며, 시간적 편집(temporal edits)을 처리할 수 있습니다.

### 공격자가 악용하는 격차 (The gap adversaries exploit)

AudioMarkBench에 따르면: "피치 시프트(pitch shift) 상황에서 모든 워터마크의 비트 복구 정확도(Bit Recovery Accuracy)가 0.6 미만으로 나타났으며, 이는 워터마크가 거의 완전히 제거되었음을 의미합니다." **피치 시프트는 보편적인 공격 방식입니다.** 2026년 기준 그 어떤 워터마크도 공격적인 피치 변조에 대해 완전한 강건성(robustness)을 보이지 못합니다. 이것이 바로 워터마킹과 함께 탐지(AASIST) 기술을 병행해야 하는 이유입니다.

### C2PA / 콘텐츠 진위성 이니셔티브 (Content Authenticity Initiative)

ML 기술이 아닌 매니페스트(manifest) 형식입니다. 오디오 파일은 생성 도구, 저자, 날짜에 대한 암호화 서명된 메타데이터를 포함합니다. Audobox / Seamless에서 이를 사용합니다. 출처(provenance) 확인에는 유용하지만, 악의적인 사용자가 재인코딩을 통해 메타데이터를 제거할 경우 아무런 역할을 하지 못합니다.

```figure
v4-audio-watermark
```

## 직접 구현해 보기 (Build It)

### 1단계: 간단한 스펙트럼 특징 검출기 (toy)

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

합성된 음성(Synthetic speech)은 종종 비정상적으로 평탄한 고주파 에너지 특성을 보입니다. 실제 서비스용 검출기는 이 방식이 아닌 AASIST를 사용합니다. 하지만 그 직관은 유효합니다.

### 2단계: AudioSeal 삽입(embed) + 탐지(detect)

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
# result: [0, 1] 사이의 float 값 — 워터마크 존재 확률
# decoded_payload: 16비트; 삽입된 payload와 대조
```

### 3단계: 평가(Evaluation) — EER

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

### 4단계: 프로덕션 통합 (Production Integration)

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

모든 생성물은 다음을 포함하여 배포됩니다: (1) 워터마크(watermark), (2) 서명된 매니페스트(signed manifest), (3) 데이터 보존 정책을 준수하는 감사 로그(retention-policy-compliant audit log).

## 활용 방법 (Use It)

| 활용 사례 (Use case) | 방어 전략 (Defense) |
|----------|---------|
| TTS / 음성 복제 배포 (Shipping TTS / voice cloning) | 모든 출력물에 AudioSeal 임베딩 적용 (필수 사항) |
| 생체 인식 음성 잠금 해제 (Biometric voice unlock) | AASIST + ECAPA 앙상블; 생동성 검증(liveness challenge) |
| 콜센터 사기 탐지 (Call-center fraud detection) | 수신되는 통화 샘플의 20%에 AASIST 적용 |
| 팟캐스트 진위 확인 (Podcast authenticity) | 업로드 시 C2PA 서명 적용, AI 생성물인 경우 AudioSeal 적용 |
| 탐지기 연구 / 학습 (Research / training detectors) | ASVspoof 5 train/dev/eval 데이터셋 활용 |

## 함정 (Pitfalls)

- **탐지기 없이 워터마크만 적용하는 경우.** 무의미합니다. CI(지속적 통합) 과정에 탐지기를 포함하여 배포하세요.
- **보정(Calibration) 없는 탐지.** ASVspoof LA 데이터로 학습된 AASIST는 과적합(overfitting) 문제가 발생하며, 실제 환경에서의 정확도가 떨어집니다. 사용자의 도메인에 맞춰 보정하세요.
- **피치 시프트(Pitch-shift) 격차.** 공격적인 피치 시프트는 대부분의 워터마크를 제거합니다. 탐지를 위한 폴백(fallback) 메커니즘을 마련하세요.
- **메타데이터 삭제 및 재호스팅.** C2PA는 재인코딩을 통해 매우 쉽게 우회할 수 있습니다. 항상 암호학적 방어와 지각적 방어(워터마크)를 함께 적용하세요.
- **탐지 수단으로서의 생동성(Liveness) 확인.** 사용자에게 무작위 문구를 말하도록 요청하는 방식입니다. 이는 재생 공격(replay attacks)은 방지할 수 있지만, 실시간 클로닝(real-time cloning)은 막지 못합니다.

## Ship It (실행하기)

`outputs/skill-spoof-defender.md`로 저장하세요. 음성 생성(voice-gen) 배포를 위해 탐지 모델(detection model), 워터마크(watermark), 출처 매니페스트(provenance manifest), 그리고 운영 플레이북(operational playbook)을 선택해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 합성 오디오(synthetic audio)를 대상으로 간단한 탐지기(toy detector)와 워터마크 삽입/탐지(toy watermark embed/detect)를 수행합니다.
2. **중간 (Medium).** `audioseal`을 설치하고, TTS 출력물에 16비트 페이로드(payload)를 삽입한 뒤 다시 디코딩해 보세요. 오디오에 노이즈를 추가하여 손상시킨 후 비트 복구 정확도(Bit Recovery Accuracy)를 측정해 보세요.
3. **어려움 (Hard).** ASVspoof 2019 LA 데이터셋을 사용하여 `RawNet2` 또는 `AASIST`를 미세 조정(Fine-tune)해 보세요. EER을 측정합니다. 학습에 사용되지 않은(held-out) F5-TTS 생성 클립 세트로 테스트하여, 분포 외 탐지(OOD detection) 성능이 어떻게 저하되는지 확인해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| ASVspoof | 벤치마크 | 격년제로 진행되는 챌린지; 2024년 = ASVspoof 5. |
| CM (countermeasure) | 탐지기 (Detector) | 분류기(Classifier): 실제 음성 vs 합성/변환된 음성 구분. |
| SASV | 화자 검증 + CM | 통합 생체 인식(Biometric) + 스푸핑 탐지. |
| AudioSeal | Meta 워터마크 | 국소화 가능(Localized), 16비트 페이로드, WavMark보다 485배 빠름. |
| Bit Recovery Accuracy | 워터마크 생존율 | 공격 후 복구된 페이로드 비트의 비율. |
| C2PA | 출처 증명 매니페스트 (Provenance manifest) | 생성 및 저작권에 관한 암호화된 메타데이터. |
| AASIST | 탐지기 제품군 (Detector family) | 그래프 어텐션(Graph-attention) 기반의 최첨단(SOTA) 안티 스푸핑 모델. |

## 추가 읽을거리 (Further Reading)

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 현재의 벤치마크입니다.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 기본 워터마크 방식입니다.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 시간적 공격(temporal attacks)을 위한 MoE 탐지기입니다.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA(최첨단) 탐지 백본입니다.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 강건성(robustness) 평가를 다룹니다.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 출처 증명(provenance) 매니페스트 형식입니다.
