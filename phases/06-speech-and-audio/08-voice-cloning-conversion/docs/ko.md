# 음성 복제 및 음성 변환 (Voice Cloning & Voice Conversion)

> 음성 복제(Voice cloning)는 다른 사람의 목소리로 텍스트를 읽어주는 기술입니다. 음성 변환(Voice conversion)은 말하는 내용은 유지하면서 목소리만 다른 사람의 목소리로 바꾸는 기술입니다. 두 기술 모두 화자의 정체성(speaker identity)과 내용(content)을 분리한다는 동일한 분해 원리에 기반합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 문제 (The Problem)

2026년에는 소비자용 GPU만으로도 5초 분량의 오디오 클립만 있으면 누구의 목소리든 고품질로 복제할 수 있습니다. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox는 모두 제로샷(zero-shot) 또는 퓨샷(few-shot) 클로닝 기능을 제공합니다. 이 기술은 축복(접근성 높은 TTS, 더빙, 보조 음성)인 동시에 무기(스캠 전화, 정치적 딥페이크, 지식재산권(IP) 도용)이기도 합니다.

서로 밀접하게 연관된 두 가지 작업은 다음과 같습니다:

- **음성 복제 (Voice cloning, TTS 측면):** 텍스트 + 5초 참조 음성 → 해당 목소리로 생성된 오디오.
- **음성 변환 (Voice conversion, 음성 측면):** 소스 오디오 (A가 X라고 말함) + B의 참조 음성 → B가 X라고 말하는 오디오.

두 작업 모두 파형(waveform)을 (콘텐츠, 화자, 운율)로 분해한 뒤, 한 소스의 콘텐츠와 다른 소스의 화자 정보를 재결합합니다.

2026년 현재 당신이 준수해야 할 핵심 제약 사항은 다음과 같습니다: **워터마킹(watermarking)과 동의 게이트(consent gates)가 EU(AI Act, 2026년 8월 시행) 및 캘리포니아(AB 2905, 2025년 발효)에서 법적으로 의무화되었습니다.** 당신의 파이프라인은 들리지 않는 워터마크를 생성해야 하며, 동의되지 않은 복제 요청은 거부해야 합니다.

## 개념 (The Concept)

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**제로샷 클로닝 (Zero-shot cloning).** 수천 명의 화자로 학습된 모델에 5초 분량의 클립을 전달합니다. 화자 인코더(speaker encoder)가 해당 클립을 화자 임베딩(speaker embedding)으로 매핑하며, TTS 디코더는 해당 임베딩과 텍스트를 조건(condition)으로 하여 음성을 생성합니다.

사용 사례: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**퓨샷 파인튜닝 (Few-shot fine-tuning).** 대상 화자의 목소리를 5~30분 정도 녹음합니다. 기본 모델을 한 시간 정도 LoRA로 파인튜닝합니다. 품질이 "괜찮은 수준"에서 "구분이 불가능한 수준"으로 비약적으로 향상됩니다. Coqui와 ElevenLabs 모두 이 방식을 지원하며, 커뮤니티에서는 F5-TTS와 함께 이 방식을 사용합니다.

**음성 변환 (Voice conversion, VC).** 두 가지 체계가 있습니다:

- **인식-합성 (Recognition-synthesis).** ASR과 유사한 모델을 실행하여 콘텐츠 표현(예: soft phoneme posteriors, PPGs)을 추출한 다음, 대상 화자 임베딩을 사용하여 재합성합니다. 언어와 억양에 강건(robust)합니다. KNN-VC (2023), Diff-HierVC (2023)에서 사용됩니다.
- **엉킴 해제 (Disentanglement).** 잠재 공간(latent space)의 병목(bottleneck) 지점에서 콘텐츠, 화자, 운율(prosody)을 분리하는 오토인코더를 학습합니다. 추론 시 화자 임베딩을 교체합니다. 품질은 낮지만 속도가 빠릅니다. AutoVC (2019), VITS-VC 변형 모델들에서 사용됩니다.

**뉴럴 코덱 기반 클로닝 (Neural codec-based cloning, 2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — 오디오를 SoundStream / EnCodec의 이산 토큰(discrete tokens)으로 취급하고, 코덱 토큰에 대해 대규모 자기회귀(autoregressive) 또는 플로우 매칭(flow-matching) 모델을 학습합니다. 짧은 프롬프트에서 ElevenLabs와 대등한 품질을 보여줍니다.

### 윤리적 측면: 부가적인 요소가 아닌 핵심 요소 (The ethics bit, not a bolt-on)

**워터마킹(Watermarking).** PerTh (Perth) 및 SilentCipher (2024)는 오디오에 약 16~32비트의 ID를 인지할 수 없게 삽입합니다. 이는 재인코딩, 스트리밍 및 일반적인 편집 후에도 유지됩니다. 상용화 가능한 오픈 소스입니다.

**동의 게이트(Consent gates).** 모든 클로닝된 출력물에는 검증 가능한 동의 기록을 반드시 결합해야 합니다. 예: "나, Rohit는 2026-04-22에 X 목적을 위해 이 목소리를 사용하는 것을 승인합니다." 이를 위변조 방지 로그(tamper-evident log)에 저장하십시오.

**탐지(Detection).** AASIST, RawNet2, Wav2Vec2-AASIST가 탐지기(detectors)로 제공됩니다. ASVspoof 2025 챌린지에서 발표된 결과에 따르면, ElevenLabs, VALL-E 2, Bark의 출력물을 대상으로 한 최신 탐지기들의 EER(Equal Error Rate)은 0.8~2.3% 수준입니다.

### 수치 (Numbers, 2026)

| 모델 (Model) | 제로샷 (Zero-shot)? | SECS (대상 유사도) | WER (지능형) | 파라미터 (Params) |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70은 일반적으로 대부분의 청취자에게 대상과 구별할 수 없는 수준입니다.

```figure
sp-voice-factorize
```

## 직접 구현해 보기 (Build It)

### 1단계: 인식-합성(recognition-synthesis)을 통한 분해 (`main.py`의 코드 전용 데모)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

개념적으로는 간단합니다. 구현의 핵심 비중은 `tts_model`과 화자 인코더(speaker encoder)에 있습니다.

### 2단계: F5-TTS를 이용한 제로샷 클로닝(zero-shot clone)

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

참조 텍스트(Reference transcript)는 오디오와 정확히 일치해야 합니다. 일치하지 않으면 정렬(alignment)이 깨집니다.

### 3단계: KNN-VC를 이용한 음성 변환 (Voice Conversion)

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC는 WavLM을 실행하여 소스(source)와 타겟 풀(target pool)에 대한 프레임별 임베딩을 추출한 다음, 각 소스 프레임을 풀 내의 가장 가까운 이웃(nearest neighbor)으로 교체합니다. 비매개변수(Non-parametric) 방식이며, 1분 정도의 타겟 음성만으로도 작동합니다.

### 4단계: 워터마크 삽입(embed a watermark)

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 페이로드 바이트를 반환합니다
```

약 32비트의 페이로드를 포함하며, MP3 재인코딩 및 가벼운 노이즈 추가 후에도 탐지 가능합니다.

### 5단계: 동의 게이트 (consent gate)

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 활용하기 (Use It)

2026년 기술 스택:

| 상황 | 선택 (Pick) |
|-----------|------|
| 5초 제로샷 클로닝, 오픈 소스 | F5-TTS 또는 OpenVoice v2 |
| 상업적 제작용 클로닝 | ElevenLabs Instant Voice Clone v2.5 |
| 음성 변환 (Voice conversion/rewriting) | KNN-VC 또는 Diff-HierVC |
| 다중 화자 미세 조정 (Many-speaker fine-tune) | StyleTTS 2 + speaker adapter |
| 교차 언어 클로닝 (Cross-lingual cloning) | XTTS v2 또는 VALL-E X |
| 딥페이크 탐지 (Deepfake detection) | Wav2Vec2-AASIST |

## 주의 사항 (Pitfalls)

- **참조 텍스트 불일치 (Misaligned reference transcript).** F5-TTS 및 유사 모델들은 참조 텍스트가 문장 부호를 포함하여 참조 오디오와 정확히 일치해야 합니다.
- **잔향이 있는 참조 오디오 (Reverberant reference).** 에코(echo)는 음성 복제 품질을 저하시킵니다. 잔향이 없는(dry) 상태에서 마이크를 가까이 대고 녹음하세요.
- **감정 불일치 (Emotional mismatch).** "쾌활한(cheerful)" 참조 오디오로 학습하면 모든 음성이 쾌활하게 복제됩니다. 참조 오디오의 감정을 대상 용도와 일치시키세요.
- **언어 누출 (Language leakage).** 영어 화자를 복제한 후 모델에게 프랑스어를 말하도록 요청하면 영어 억양이 그대로 남는 경우가 많습니다. 교차 언어 모델(XTTS, VALL-E X)을 사용하세요.
- **워터마크 부재 (No watermark).** 2026년 8월부터 EU에서는 법적으로 출시가 불가능합니다.

## Ship It (실행해 보세요)

`outputs/skill-voice-cloner.md`로 저장하세요. 동의 게이트(consent gate) + 워터마크(watermark) + 품질 목표(quality target)를 포함한 복제 또는 변환 파이프라인을 설계해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 스왑 전후 두 "화자(speaker)" 사이의 코사인 유사도를 계산하여 화자 임베딩 스왑(speaker-embedding swap) 과정을 시연합니다.
2. **중간 (Medium).** OpenVoice v2를 사용하여 자신의 목소리를 복제해 보세요. 참조 음성과 복제된 음성 사이의 SECS를 측정합니다. Whisper를 통해 CER을 측정해 보세요.
3. **어려움 (Hard).** 20개의 복제 음성에 SilentCipher 워터마크를 적용하고, 이를 128 kbps MP3 인코딩 및 디코딩 과정을 거치게 한 뒤 페이로드(payload)를 탐지해 보세요. 비트 정확도(bit-accuracy)를 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5초면 충분하다 | 사전 학습된 모델 + 화자 임베딩(speaker embedding); 별도의 학습 없음. |
| PPG | Phonetic posteriorgram | 언어 중립적 콘텐츠 표현으로 사용되는 프레임별 ASR 사후 확률(posteriors). |
| KNN-VC | Nearest-neighbor conversion | 각 소스 프레임을 타겟 풀(target-pool) 내의 가장 가까운 프레임으로 교체. |
| Neural codec TTS | VALL-E 스타일 | EnCodec/SoundStream 토큰을 활용한 자기회귀(AR) 모델. |
| Watermark | 들리지 않는 서명 | 오디오에 삽입되어 재인코딩 후에도 유지되는 비트 정보. |
| SECS | 복제 충실도(Cloning fidelity) | 타겟 화자와 복제된 화자의 임베딩 간 코사인 유사도. |
| AASIST | 딥페이크 탐지기 | 안티 스푸핑(Anti-spoof) 모델; 합성된 음성을 탐지. |

## 추가 학습 자료 (Further Reading)

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 오픈 소스 SOTA 제로샷 클로닝(zero-shot cloning).
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 및 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 신경망 코덱(neural-codec) TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 엉킴 해제(disentanglement) 기반 음성 변환.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 검색(retrieval) 기반 음성 변환.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 상용 수준의 32비트 오디오 워터마킹.
- [ASVspoof 2025 결과](https://www.asvspoof.org/) — 탐지기(detector)와 합성기(synthesizer) 간의 군비 경쟁, 2026년 업데이트 예정.
