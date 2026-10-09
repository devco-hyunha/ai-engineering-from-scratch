# 음성 클로닝 및 음성 변환

> 음성 클로닝은 다른 사람의 목소리로 텍스트를 읽어줍니다. 음성 변환은 발화 내용을 보존하면서 내 목소리를 다른 사람의 목소리로 재작성합니다. 두 기술 모두 동일한 분해 구조에 기반합니다: 화자 식별 정보와 내용을 분리하는 것입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 06강 (화자 인식), 6단계 · 07강 (TTS)
**시간:** 약 75분

## 문제점

2026년 현재, 소비자용 GPU만으로도 5초짜리 오디오 클립을 이용해 누구나의 목소리를 고품질로 클로닝할 수 있습니다. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox는 모두 제로샷(Zero-Shot) 또는 소수 예시(Few-Shot) 클로닝을 제공합니다. 이 기술은 축복(접근성 TTS, 더빙, 보조 음성)이자 무기(사기 전화, 정치적 딥페이크, IP 도용)이기도 합니다.

두 가지 밀접하게 관련된 작업:

- **음성 클로닝 (TTS 측):** 텍스트 + 5초 참조 음성 → 해당 목소리로 생성된 오디오.
- **음성 변환 (Speech 측):** 소스 오디오 (사람 A가 X를 말함) + 사람 B의 참조 음성 → 사람 B가 X를 말하는 오디오.

두 작업 모두 파형(Waveform)을 (내용, 화자, 운율)로 분해하고, 한 소스의 내용과 다른 소스의 화자를 재결합합니다.

2026년 현재 준수해야 하는 핵심 제약 조건: **워터마킹 및 동의 게이트는 EU (AI Act, 2026년 8월 시행)와 캘리포니아 (AB 2905, 2025년 발효)에서 법적으로 요구됩니다.** 파이프라인은 들리지 않는 워터마킹을 포함해야 하며, 비동의 클로닝을 거부해야 합니다.

## 개념

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**제로샷 클로닝.** 수천 명의 화자로 학습된 모델에 5초 클립을 전달합니다. 화자 인코더는 클립을 화자 임베딩(Embedding)으로 매핑하며, TTS 디코더는 해당 임베딩과 텍스트를 조건으로 사용합니다.

사용 사례: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**소수 예시 미세 조정(Few-Shot Fine-Tuning).** 대상 목소리를 5~30분 녹음합니다. LoRA (저랭크 적응)(LoRA (Low-Rank Adaptation))로 기본 모델을 1시간 동안 미세 조정합니다. 품질이 "괜찮은 수준"에서 "구분할 수 없는 수준"으로 급상승합니다. Coqui와 ElevenLabs 모두 이 패턴을 지원하며, 커뮤니티는 F5-TTS와 함께 이 패턴을 사용합니다.

**음성 변환 (VC).** 두 가지 계열이 있습니다:

- **인식-합성.** ASR 유사 모델을 실행하여 콘텐츠 표현(예: 부드러운 음소 사후 확률, PPG)을 추출한 후, 대상 화자 임베딩으로 재합성합니다. 언어와 억양에 대해 강건합니다. KNN-VC (2023), Diff-HierVC (2023)에서 사용됩니다.
- **분리.** 병목점(latent space)에서 콘텐츠, 화자, 운율을 분리하는 오토인코더를 학습합니다. 추론 시 화자 임베딩을 교체합니다. 품질은 낮지만 더 빠릅니다. AutoVC (2019), VITS-VC 변형에서 사용됩니다.

**신경 코덱 기반 클로닝 (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — 오디오를 SoundStream / EnCodec의 이산 토큰으로 취급하고, 코덱 토큰에 대해 대규모 자기회귀(Autoregressive) 또는 흐름 매칭(flow-matching) 모델을 학습합니다. 짧은 프롬프트에서 ElevenLabs와 비교 가능한 품질을 제공합니다.

### 윤리 부분은 부속품이 아닙니다

**워터마킹.** PerTh (Perth)와 SilentCipher (2024)는 약 16-32비트 ID를 오디오에 지각되지 않게 임베드합니다. 재인코딩, 스트리밍, 일반적인 편집을 견디며, 생산 준비가 된 오픈 소스입니다.

**동의 게이트.** 모든 클로닝된 출력은 검증 가능한 동의 기록과 짝을 이루어야 합니다. "나 Rohit는 2026-04-22에 X 목적을 위해 이 목소리를 승인합니다." 변조 방지 로그에 저장하세요.

**탐지.** AASIST, RawNet2, Wav2Vec2-AASIST는 탐지기로 제공됩니다. ASVspoof 2025 챌린지는 ElevenLabs, VALL-E 2, Bark 출력에 대한 최신 탐지기의 EER이 0.8–2.3%임을 발표했습니다.

### 수치 (2026)

| 모델 | 제로샷? | SECS (대상 유사도) | WER (지능) | 매개변수 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 예 | 0.72 | 2.1% | 335M |
| XTTS v2 | 예 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 예 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 예 | 0.77 | 2.4% | 370M |
| VoiceBox | 예 | 0.78 | 2.1% | 330M |

SECS > 0.70은 대부분의 청취자에게 대상과 구별이 불가능한 수준입니다.

```figure
sp-voice-factorize
```

## 구현하기

### 1단계: 인식-합성으로 분해 (main.py의 코드 전용 데모)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

개념적으로 간단합니다; 구현의 대부분은 `tts_model`과 화자 인코더에 있습니다.

### 2단계: F5-TTS로 제로샷 클로닝

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

참고 전사(transcript)는 오디오와 정확히 일치해야 합니다; 불일치는 정렬을 깨뜨립니다.

### 3단계: KNN-VC로 음성 변환

```python
import torch
from knnvc import KNNVC  # 2023년 모델, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC는 WavLM을 실행하여 소스 및 타겟 풀의 프레임별 임베딩을 추출한 후, 소스 프레임을 풀 내의 최근접 이웃으로 교체합니다. 비파라메트릭 방식이며, 1분 분량의 타겟 음성으로 작동합니다.

### 4단계: 워터마크 삽입

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 페이로드 바이트를 반환합니다
```

약 32비트의 페이로드이며, MP3 재인코딩 및 가벼운 잡음 후에도 감지 가능합니다.

### 5단계: 동의 게이트

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 5초 제로샷 클로닝, 오픈소스 | F5-TTS 또는 OpenVoice v2 |
| 상업적 프로덕션 클로닝 | ElevenLabs Instant Voice Clone v2.5 |
| 음성 변환 (재작성) | KNN-VC 또는 Diff-HierVC |
| 다화자 미세 조정 | StyleTTS 2 + 화자 어댑터 |
| 언어 간 클로닝 | XTTS v2 또는 VALL-E X |
| 딥페이크 탐지 | Wav2Vec2-AASIST |

## 함정

- **참고 전사 불일치.** F5-TTS 및 유사 모델은 참고 텍스트가 참고 오디오와 정확히 일치해야 하며, 문구도 포함됩니다.
- **잔향 있는 참고 음성.** 에코는 클로닝을 망칩니다. 드라이하고 근접 마이크 녹음을 하세요.
- **감정 불일치.** "밝은" 참고 음성으로 학습하면 모든 것이 밝은 클론이 됩니다. 참고 음성의 감정을 타겟 용도에 맞게 매칭하세요.
- **언어 누수.** 영어 화자를 클로닝한 후 모델에 프랑스어를 요청하면 억양이 그대로 전달되는 경우가 많습니다. 언어 간 모델(XTTS, VALL-E X)을 사용하세요.
- **워터마크 없음.** 2026년 8월부터 EU에서는 법적 유통이 불가능합니다.

## 출시하기

`outputs/skill-voice-cloner.md`로 저장하세요. 동의 게이트 + 워터마크 + 품질 목표를 포함한 클로닝 또는 변환 파이프라인을 설계해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 스왑 전후의 두 "화자" 간 코사인 유사도를 계산하여 화자 임베딩 스왑을 시연합니다.
2. **중간.** OpenVoice v2를 사용하여 자신의 음성을 클로닝하세요. 참고 음성과 클론 간 SECS를 측정하고, Whisper를 통해 CER를 측정하세요.
3. **어려움.** SilentCipher 워터마크를 20개 클론에 적용하고, 128 kbps MP3 인코딩 및 디코딩을 거쳐 페이로드를 감지하세요. 비트 정확도를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Zero-shot 클로닝 | 5초면 충분 | 사전 학습된 모델 + 화자 임베딩; 학습 없음. |
| PPG | 음소 사후 확률 그래프 | 언어에 독립적인 콘텐츠 표현으로 사용되는 프레임별 ASR 사후 확률. |
| KNN-VC | 최근접 이웃 변환 | 각 소스 프레임을 가장 가까운 타겟 풀 프레임으로 교체. |
| 뉴럴 코덱 TTS | VALL-E 스타일 | EnCodec/SoundStream 토큰에 대한 AR 모델. |
| 워터마크 | 들리지 않는 서명 | 오디오에 내장된 비트, 재인코딩에도 생존. |
| SECS | 클로닝 충실도 | 타겟 및 클론 화자 임베딩 간의 코사인 유사도. |
| AASIST | 딥페이크 탐지 | 스푸핑 방지 모델; 합성된 음성 탐지. |

## 추가 읽기

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 오픈소스 SOTA zero-shot 클로닝.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 및 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 뉴럴 코덱 TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 분리를 기반으로 한 음성 변환.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 검색 기반 VC.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 프로덕션 준비가 완료된 32-bit 오디오 워터마크.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 탐지기와 합성기 간의 군비 경쟁, 2026년 업데이트.
