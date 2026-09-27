# 오디오 평가 — WER, MOS, UTMOS, MMAU, FAD 및 오픈 리더보드 (Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards)

> 측정할 수 없는 것은 출시(ship)할 수 없습니다. 이 레슨에서는 2026년 기준 모든 오디오 작업에 대한 지표를 다룹니다: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), 오디오-언어 (MMAU, LongAudioBench), 음악 (FAD, CLAP), 그리고 화자 (EER). 또한 성능을 비교할 수 있는 리더보드도 소개합니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## 문제점 (The Problem)

모든 오디오 작업에는 서로 다른 축을 측정하는 여러 지표가 존재합니다. 잘못된 지표를 사용하는 것은 대시보드에서는 훌륭해 보이지만 실제 운영 환경(production)에서는 끔찍한 성능을 보이는 모델을 출시하게 만드는 원인이 됩니다. 2026년 기준 표준 목록은 다음과 같습니다:

| 작업 (Task) | 주요 지표 (Primary) | 보조 지표 (Secondary) |
|------|---------|-----------|
| ASR (음성 인식) | WER | CER · RTFx · 첫 토큰 지연 시간(first-token latency) |
| TTS (음성 합성) | MOS / UTMOS | SECS · ASR 왕복 WER(WER-on-ASR-round-trip) · CER · TTFA |
| Voice cloning (음성 복제) | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification (화자 확인) | EER | minDCF · 동작 지점에서의 FAR / FRR |
| Diarization (화자 분할) | DER | JER · 화자 혼동(speaker confusion) |
| Audio classification (오디오 분류) | top-1 · mAP | macro F1 · 클래스별 재현율(per-class recall) |
| Music generation (음악 생성) | FAD | CLAP · 청취 패널 MOS |
| Audio language model (오디오 언어 모델) | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S (스트리밍 S2S) | latency P50/P95 | WER · MOS |

## 개념 (The Concept)

![오디오 평가 매트릭스 — 지표 vs 작업 vs 2026년 리더보드](../assets/eval-landscape.svg)

### ASR 지표 (ASR metrics)

**WER (Word Error Rate, 단어 오류율).** `(S + D + I) / N`. 점수를 매기기 전에 소문자 변환, 문장 부호 제거, 숫자 정규화 과정을 거쳐야 합니다. `jiwer` 또는 OpenAI의 `whisper_normalizer`를 사용하세요. < 5%는 사람이 듣고 받아쓰는 수준(human-parity)과 유사합니다.

**CER (Character Error Rate, 음절 오류율).** 동일한 공식을 사용하되, 음절(character) 단위로 계산합니다. 단어 분절이 모호한 성조 언어(Mandarin, Cantonese)에서 주로 사용됩니다.

**RTFx (inverse real-time factor, 역 실시간 계수).** 실제 경과 시간 1초당 처리된 오디오 초 단위 시간입니다. 값이 높을수록 좋습니다. Parakeet-TDT는 3380×를 기록했으며, Whisper-large-v3는 약 30×입니다.

**First-token latency (첫 토큰 지연 시간).** 오디오 입력부터 첫 번째 전사(transcript) 토큰이 나올 때까지의 실제 경과 시간입니다. 스트리밍 서비스에서 매우 중요합니다. Deepgram Nova-3의 경우 약 150ms입니다.

### TTS 지표 (TTS metrics)

**MOS (Mean Opinion Score).** 1~5점 사이의 인간 평가 점수입니다. 황금 표준(Gold standard)이지만 속도가 느립니다. 샘플당 20명 이상의 청취자와 모델당 100개 이상의 샘플을 수집해야 합니다.

**UTMOS (2022-2026).** 학습된 MOS 예측기입니다. 표준 벤치마크에서 인간의 MOS와 약 0.9의 상관관계를 보입니다. F5-TTS의 경우 UTMOS 점수는 3.95이며, 실제 정답(ground truth)은 4.08입니다.

**SECS (Speaker Encoder Cosine Similarity).** 음성 복제(Voice cloning)를 위한 지표입니다. 참조 음성과 복제된 출력 음성 사이의 ECAPA 임베딩 코사인 유사도를 측정합니다. 0.75보다 크면 식별 가능한 수준의 복제라고 판단합니다.

**WER-on-ASR-round-trip.** TTS 출력물에 Whisper를 실행하여 입력 텍스트와 비교한 WER(Word Error Rate)을 계산합니다. 명료도(Intelligibility)의 저하를 포착할 수 있습니다. 2026년 SOTA 기준: CER(Character Error Rate) 2% 미만입니다.

**TTFA (time-to-first-audio).** 첫 오디오가 출력될 때까지의 실제 시간 지연(Latency)입니다. Kokoro-82M은 약 100ms, F5-TTS는 약 1s입니다.

### 음성 복제 특화 (Voice-cloning-specific)

**SECS + MOS + CER**을 하나의 세트로 활용합니다. SECS 점수는 높지만 MOS 점수가 낮은 경우, 음색(timbre)은 정확하지만 부자연스러운 상태를 의미합니다. 반대로 MOS는 높지만 SECS가 낮은 경우, 목소리는 자연스럽지만 화자(speaker)가 일치하지 않는 상태를 의미합니다.

### 화자 검증 (Speaker verification)

**EER (Equal Error Rate, 동일 오류율).** 오인식률(False Accept Rate)과 오거부율(False Reject Rate)이 같아지는 임계값입니다. VoxCeleb1-O 데이터셋에 대한 ECAPA의 EER은 0.87%입니다.

**minDCF (min Detection Cost, 최소 탐지 비용).** 선택된 동작 지점(주로 FAR=0.01)에서의 가중치 적용 비용입니다. EER보다 실제 서비스 환경(production)에 더 적합한 지표입니다.

### 화자 분할 (Diarization)

**DER (Diarization Error Rate, 화자 분할 오류율).** `(FA + Miss + Confusion) / total_speaker_time`. 누락된 음성(Missed speech) + 오탐지 음성(false-alarm speech) + 화자 혼동(speaker-confusion)을 각각의 비율로 계산합니다. AMI 회의 데이터셋 기준: DER ~10-20% 정도가 현실적입니다. pyannote 3.1 + Precision-2 상용 모델 사용 시: 녹음 상태가 양호한 오디오에서 DER <10%를 달성합니다.

**JER (Jaccard Error Rate, 자카드 오류율).** DER의 대안으로, 짧은 세그먼트에 편향되는 현상에 강건(robust)합니다.

### 오디오 분류 (Audio classification)

멀티 레이블(Multi-label): 모든 클래스에 대한 **mAP (mean Average Precision)**. AudioSet 기준: BEATs-iter3 모델이 0.548 mAP 달성.

단일 클래스 배타적 분류(Multi-class exclusive): **top-1, top-5 정확도(accuracy)**. Speech Commands v2 기준: Audio-MAE 모델이 99.0% top-1 달성.

불균형 데이터(Imbalanced): **macro F1** + **클래스별 재현율(per-class recall)**. 클래스별로 보고하십시오 — 전체 집계 정확도는 어떤 클래스가 실패했는지 숨길 수 있습니다.

### 음악 생성 (Music generation)

**FAD (Fréchet Audio Distance).** 실제 오디오와 생성된 오디오 간의 `VGGish-embedding` 분포 사이의 거리입니다. MusicCaps 데이터셋에 대한 MusicGen-small의 점수는 4.5이며, MusicLM은 4.0입니다. 점수가 낮을수록 좋습니다.

**CLAP Score.** `CLAP` 임베딩을 사용하여 텍스트와 오디오 간의 정렬(alignment) 정도를 측정하는 점수입니다. 0.3보다 크면 적절한 정렬 상태로 간주합니다.

**청취 패널 MOS (Listening panel MOS).** 소비자급 음악 품질을 판단하는 최종적인 기준입니다. TTS Arena에서 측정된 Suno v5의 ELO 점수는 1293입니다 (쌍을 이룬 인간 선호도 기반).

### 오디오-언어 벤치마크 (Audio-language benchmarks)

**MMAU (Massive Multi-Audio Understanding).** 1만 개의 오디오-QA 쌍으로 구성됩니다.

**MMAU-Pro.** 1,800개의 고난도 항목으로 구성되며, 음성(speech) / 소리(sound) / 음악(music) / 멀티 오디오(multi-audio)의 네 가지 카테고리가 있습니다. 4지 선다형에서 무작위 확률은 25%입니다. Gemini 2.5 Pro의 전체 성능은 약 60%이며, 모든 모델을 통틀어 멀티 오디오 카테고리는 약 22%를 기록합니다.

**LongAudioBench.** 의미론적 질의(semantic queries)가 포함된 수 분 길이의 클립을 다룹니다. Audio Flamingo Next가 Gemini 2.5 Pro를 능가합니다.

**AudioCaps / Clotho.** 캡셔닝(Captioning) 벤치마크입니다. SPICE, CIDEr, FENSE 지표를 사용합니다.

### 스트리밍 음성-대-음성 (Streaming speech-to-speech)

**지연 시간(Latency) P50 / P95 / P99.** 사용자의 발화 종료 시점부터 첫 번째 가청 응답이 나올 때까지의 실제 시간(Wall-clock). Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**출력물에 대한 WER / MOS.**

**끼어들기 반응성(Barge-in responsiveness).** 사용자의 중단(interrupt) 시점부터 어시스턴트가 음소거(mute)될 때까지의 시간. 목표치 < 150 ms.

### 2026년 리더보드 (The 2026 leaderboards)

| 리더보드 (Leaderboard) | 트랙 (Tracks) | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 영어 + 다국어 + 장문 (English + multilingual + long-form) | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 영어 TTS (English TTS) | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, 쌍체 투표 기반 ELO (TTS + STT, ELO from paired votes) | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 추론 (LALM reasoning) | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 화자 인식 (Speaker recognition) | `voxsrc.github.io` |
| MMAU music subset | 음악 LALM (Music LALM) | (MMAU 내 포함) |
| HEAR benchmark | 자기지도 학습 오디오 (Self-supervised audio) | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 직접 구현해 보기 (Build It)

### 1단계: 정규화를 적용한 WER (WER with normalization)

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 2단계: TTS 왕복 WER (TTS round-trip WER)

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 3단계: 음성 복제를 위한 SECS (SECS for voice cloning)

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 4단계: 음악 생성을 위한 FAD (FAD for music generation)

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 5단계: 화자 검증을 위한 EER (Lesson 6와 동일한 코드)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 활용하기 (Use It)

모든 배포 시에는 모델이 업데이트될 때마다 실행되는 고정된 평가 하네스(eval harness)를 쌍으로 구성하세요. 세 가지 핵심 규칙은 다음과 같습니다:

1. **점수를 매기기 전에 정규화(Normalize)하세요.** 소문자 변환, 문장 부호 제거, 숫자 확장 등을 수행합니다. 사용된 정규화 규칙을 보고하세요.
2. **평균이 아닌 분포(Distributions)를 보고하세요.** 지연 시간(latency)의 경우 P50/P95/P99를 사용하세요. 분류(classification)의 경우 클래스별 재현율(per-class recall)을 사용하세요. MMAU의 경우 카테고리별로 보고하세요.
3. **하나의 표준 공개 벤치마크(canonical public benchmark)를 실행하세요.** 실제 운영 데이터가 다르더라도, Open ASR / TTS Arena / MMAU에 대해 보고하면 검토자들이 동일한 기준으로 비교(apples-to-apples)할 수 있습니다.

## 주의 사항 (Pitfalls)

- **UTMOS 외삽(extrapolation) 문제.** VCTK 스타일의 깨끗한 음성으로 학습되었기 때문에, 노이즈가 있거나 복제된(cloned) 음성, 또는 감정이 실린 오디오에 대해서는 점수가 낮게 측정될 수 있습니다.
- **MOS 패널 편향(bias).** Amazon Mechanical Turk 작업자 20명은 실제 타겟 사용자 20명과 다를 수 있습니다. 리스크가 큰 프로젝트라면 해당 도메인의 전문 패널을 활용하세요.
- **FAD는 참조 세트(reference set)에 의존함.** 모델 간 비교 시에는 반드시 동일한 참조 분포(reference distribution)를 사용하여 비교해야 합니다.
- **전체 WER(Aggregate WER).** 전체 WER이 5%라 하더라도, 특정 억양(accented speech)이 있는 음성에서의 WER은 30%에 달할 수 있습니다. 인구통계학적 세그먼트(demographic slice)별로 나누어 보고하세요.
- **공개 벤치마크의 포화(saturation).** 대부분의 최첨단(frontier) 모델들은 표준 벤치마크에서 이미 한계치(ceiling)에 도달해 있습니다. 실제 서비스 트래픽을 반영하는 자체 홀드아웃 세트(in-house held-out set)를 구축하세요.

## Ship It (실행하기)

`outputs/skill-audio-evaluator.md`로 저장하세요. 모든 오디오 모델 출시를 위해 지표(metrics), 벤치마크(benchmarks), 그리고 보고 형식(reporting format)을 선정해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. Toy input(예제 입력)에 대해 WER / CER / EER / SECS / FAD-ish / MMAU-ish를 계산해 보세요.
2. **중간 (Medium).** TTS 라운드트립(round-trip) WER 테스트 환경을 구축해 보세요. Kokoro 또는 F5-TTS의 출력물을 Whisper로 처리해 보세요. 50개의 프롬프트에 대해 WER을 계산하고, WER이 10%를 초과하는 프롬프트를 표시해 보세요.
3. **어려움 (Hard).** 레슨 10에서 선택한 LALM을 MMAU-Pro 음성(speech) + 멀티 오디오(multi-audio) 서브셋(각 50개 항목)으로 평가해 보세요. 카테고리별 정확도를 보고하고, 발표된 수치와 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| WER | ASR 점수 | 정규화 후 단어 수준에서의 `(S+D+I)/N`. |
| CER | Character WER | 성조 언어 또는 문자 단위 시스템을 위한 지표. |
| MOS | 인간의 평가 | 1-5점 척도; 20명 이상의 청취자 × 100개 샘플. |
| UTMOS | ML MOS 예측기 | 학습된 모델; 인간의 MOS와 약 0.9의 상관관계. |
| SECS | 음성 복제 유사도 | 참조 음성과 복제 음성 간의 ECAPA 코사인 유사도. |
| EER | 화자 검증 점수 | FAR = FRR가 되는 임계값. |
| DER | 화자 분할(Diarization) 점수 | (FA + Miss + Confusion) / 전체. |
| FAD | 음악 생성 품질 | VGGish 임베딩 상의 Fréchet 거리. |
| RTFx | 처리량(Throughput) | 실제 경과 시간(wall-clock second) 대비 오디오 초 단위 처리량. |

## 추가 읽을거리 (Further Reading)

- [jiwer](https://github.com/jitsi/jiwer) — 정규화 유틸리티를 포함한 WER/CER 라이브러리.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 학습된 MOS 예측기.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — music-gen 표준 지표.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026년 실시간 순위.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 인간 투표 기반 TTS 리더보드.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 추론 리더보드.
- [HEAR benchmark](https://hearbenchmark.com/) — 오디오 SSL 벤치마크.
