# 오디오 평가 — WER, MOS, UTMOS, MMAU, FAD 및 공개 리더보드

> 측정할 수 없는 것은 출시할 수 없습니다. 이 강의에서는 모든 오디오 작업에 대한 2026년 표준 지표인 ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, ASR 왕복 WER), 오디오-언어 (MMAU, LongAudioBench), 음악 (FAD, CLAP), 화자 (EER)를 소개합니다. 또한 비교를 위한 리더보드도 다룹니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 6단계 · 04, 06, 07, 09, 10; 2단계 · 09 (모델 평가)
**시간:** 약 60분

## 문제점

모든 오디오 작업에는 서로 다른 축을 측정하는 여러 지표가 있습니다. 잘못된 지표를 사용하면 대시보드에서는 훌륭해 보이지만 프로덕션에서는 형편없는 모델을 출시하게 됩니다. 2026년 표준 목록은 다음과 같습니다:

| 작업 | 주요 지표 | 보조 지표 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 첫 토큰 지연 |
| TTS | MOS / UTMOS | SECS · ASR 왕복 WER · CER · TTFA |
| 음성 클로닝 | SECS (ECAPA 코사인) | MOS · CER |
| 화자 검증 | EER | minDCF · 운영 지점에서의 FAR / FRR |
| 화자 분리 | DER | JER · 화자 혼동 |
| 오디오 분류 | top-1 · mAP | 매크로 F1 · 클래스별 재현율 |
| 음악 생성 | FAD | CLAP · 청취 패널 MOS |
| 오디오 언어 모델 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 스트리밍 S2S | 지연 P50/P95 | WER · MOS |

## 개념

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 지표

**WER (단어 오류율).** `(S + D + I) / N`. 점수를 매기기 전에 소문자로 변환하고, 구두점을 제거하며, 숫자를 정규화합니다. `jiwer` 또는 OpenAI의 `whisper_normalizer`를 사용하세요. &lt; 5%는 인간 수준의 읽기 음성입니다.

**CER (문자 오류율).** 동일한 공식으로 문자 단위입니다. 단어 분할이 모호한 성조 언어 (중국어, 광동어)에 사용됩니다.

**RTFx (역 실시간 계수).** 벽시계 시간 초당 처리되는 오디오 초 수입니다. 높을수록 좋습니다. Parakeet-TDT는 3380×를 달성합니다. Whisper-large-v3는 약 30×입니다.

**첫 토큰 지연.** 오디오 입력부터 첫 번째 전사 토큰까지의 실제 시간(Wall-clock). 스트리밍에 중요합니다. Deepgram Nova-3: ~150 ms.

### TTS 지표

**MOS (Mean Opinion Score).** 1-5점의 인간 평가. 표준 지표이지만 느립니다. 샘플당 청취자 20명 이상, 모델당 샘플 100개 이상을 수집하세요.

**UTMOS (2022-2026).** 학습된 MOS 예측기. 표준 벤치마크에서 인간 MOS와 약 0.9의 상관관계를 보입니다. F5-TTS: UTMOS 3.95; 실제 값: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** 음성 클로닝용. 참조 음성 및 클로닝된 출력 간의 ECAPA 임베딩 코사인 유사도. &gt; 0.75 = 인식 가능한 클론.

**ASR 왕복 WER.** TTS 출력에 Whisper를 실행하고 입력 텍스트 대비 WER을 계산하세요. 명료성 저하를 포착합니다. 2026 SOTA: &lt; 2% CER.

**TTFA (time-to-first-audio).** 실제 시간(Wall-clock) 지연. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### 음성 클로닝 전용

**SECS + MOS + CER**를 삼중 지표로 사용하세요. SECS 점수가 높지만 MOS 점수가 낮은 클로닝은 음색은 맞지만 부자연스럽다는 의미이며, 그 반대는 자연스러운 목소리지만 화자가 잘못되었다는 의미입니다.

### 화자 검증

**EER (Equal Error Rate).** False Accept Rate가 False Reject Rate와 동일한 임계값. VoxCeleb1-O에서의 ECAPA: 0.87%.

**minDCF (min Detection Cost).** 선택된 운영 지점(종종 FAR=0.01)에서의 가중 비용. EER보다 생산 환경에 더 관련이 있습니다.

### 화자 분리(Diarization)

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. 놓친 발화 + 오탐 발화 + 화자 혼동 각각의 비율. AMI 회의: DER ~10-20%가 현실적입니다. pyannote 3.1 + Precision-2 상용 버전: 잘 녹음된 오디오에서 &lt;10% DER.

**JER (Jaccard Error Rate).** DER의 대안으로, 짧은 세그먼트 편향에 강건합니다.

### 오디오 분류

멀티 레이블: 모든 클래스에 대한 **mAP (mean Average Precision)**. AudioSet: BEATs-iter3의 경우 0.548 mAP.

멀티 클래스 배타적: **top-1, top-5 정확도**. Speech Commands v2: top-1 99.0% (Audio-MAE).

불균형 데이터: **macro F1** + **클래스별 재현율**. 클래스별로 보고하세요. 집계된 정확도는 어떤 클래스가 실패하는지 숨깁니다.

### 음악 생성

**FAD (Fréchet Audio Distance).** 실제 오디오와 생성 오디오의 VGGish 임베딩 분포 간 거리입니다. MusicCaps에서 MusicGen-small은 4.5, MusicLM은 4.0입니다. 값이 낮을수록 좋습니다.

**CLAP Score.** CLAP 임베딩을 사용한 텍스트-오디오 정렬 점수입니다. &gt; 0.3이면 합리적인 정렬로 간주합니다.

**청취 패널 MOS.** 소비자급 음악의 최종 평가 지표입니다. Suno v5는 TTS Arena에서 짝을 이룬 인간 선호도 기준 ELO 1293을 기록했습니다.

### 오디오-언어 벤치마크

**MMAU (Massive Multi-Audio Understanding).** 1만 개의 오디오-QA 쌍을 포함합니다.

**MMAU-Pro.** 1800개의 어려운 항목, 네 가지 범주: 음성 / 소리 / 음악 / 멀티 오디오. 4지선다의 랜덤 확률은 25%입니다. Gemini 2.5 Pro의 전체 점수는 약 60%이며, 모든 모델에서 멀티 오디오 점수는 약 22%입니다.

**LongAudioBench.** 수 분 길이의 클립과 시맨틱 쿼리를 포함합니다. Audio Flamingo Next가 Gemini 2.5 Pro를 능가합니다.

**AudioCaps / Clotho.** 캡셔닝 벤치마크입니다. SPICE, CIDEr, FENSE 지표를 사용합니다.

### 스트리밍 음성-음성

**Latency P50 / P95 / P99.** 사용자 발화 종료부터 첫 가청 응답까지의 벽시계 시간입니다. Moshi는 200 ms, GPT-4o Realtime은 300 ms입니다.

출력에 대한 **WER / MOS**

**Barge-in 응답성.** 사용자 인터럽트부터 어시스턴트 음소거까지의 시간입니다. 목표는 &lt; 150 ms입니다.

### 2026년 리더보드

| 리더보드 | 트랙 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 영어 + 다국어 + 롱폼 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 영어 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, 짝을 이룬 투표 기반 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 추론 | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | 화자 인식 | `voxsrc.github.io` |
| MMAU 음악 하위 집합 | 음악 LALM | (MMAU 내부) |
| HEAR 벤치마크 | 자기 지도 오디오 | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 구현하기

### 1단계: 정규화를 포함한 WER

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

### 2단계: TTS 왕복 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 3단계: 음성 클로닝을 위한 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 4단계: 음악 생성을 위한 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 5단계: 화자 검증을 위한 EER (06강과 동일한 코드)

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

## 사용하기

모든 배포에 모델 업데이트마다 실행되는 고정된 평가 하네스를 짝지으세요. 세 가지 기본 원칙은 다음과 같습니다:

1. **채점 전에 정규화하세요.** 소문자화, 구두점 제거, 숫자 확장. 정규화 규칙을 보고하세요.
2. **평균이 아닌 분포를 보고하세요.** 지연 시간에 대해 P50/P95/P99를 보고하세요. 분류에 대해 클래스별 재현율을 보고하세요. MMAU에 대해 카테고리별 결과를 보고하세요.
3. **하나의 표준 공개 벤치마크를 실행하세요.** 프로덕션 데이터가 다르더라도 Open ASR / TTS Arena / MMAU에 대해 보고하면 리뷰어가 동등한 조건에서 비교할 수 있습니다.

## 함정

- **UTMOS 외삽.** VCTK 스타일의 깨끗한 음성으로 학습되어 잡음이 있는 / 클론된 / 감정적인 오디오에 대해 점수가 낮습니다.
- **MOS 패널 편향.** Amazon Mechanical Turk의 작업자 20명 ≠ 대상 사용자 20명. 위험도가 높다면 도메인 패널에 비용을 지불하세요.
- **FAD는 참조 집합에 의존합니다.** 모델 간에 동일한 참조 분포와 비교하세요.
- **집계된 WER.** 전체 WER이 5%인 경우에도 억양이 있는 발화에 대해 WER이 30%일 수 있습니다. 인구통계학적 슬라이스별로 보고하세요.
- **공개 벤치마크 포화.** 대부분의 최첨단 모델은 표준 벤치마크에서 상한에 근접합니다. 트래픽을 반영하는 내부 홀드아웃 세트 구축하세요.

## 출시하기

`outputs/skill-audio-evaluator.md`로 저장하세요. 모든 오디오 모델 릴리스에 대해 지표, 벤치마크, 보고 형식을 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 장난감 입력에 대해 WER / CER / EER / SECS / FAD 유사 / MMAU 유사를 계산하세요.
2. **중간.** TTS 왕복 WER 하네스를 구축하세요. Kokoro 또는 F5-TTS 출력에 Whisper를 실행하세요. 50개 프롬프트에 대해 WER을 계산하세요. WER이 10%를 초과하는 프롬프트를 플래그하세요.
3. **어려움.** 10강 LALM 선택에 대해 MMAU-Pro 음성 + 다중 오디오 하위 집합(각각 50개 항목)에 점수를 매기세요. 카테고리별 정확도를 보고하고 공개된 수치와 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| WER | ASR 점수 | 정규화 후 단어 수준의 `(S+D+I)/N`. |
| CER | 문자 WER | 성조 언어 또는 문자 단위 시스템에 사용. |
| MOS | 인간 의견 | 1-5점 평가; 청취자 20명 이상 × 샘플 100개. |
| UTMOS | ML MOS 예측기 | 학습된 모델; 인간 MOS와 약 0.9의 상관관계. |
| SECS | 음성 클론 유사도 | 참조와 클론 간 ECAPA 코사인 유사도. |
| EER | 화자 검증 점수 | FAR = FRR인 임계값. |
| DER | 화자 분리 점수 | (FA + 누락 + 혼동) / 총합. |
| FAD | 음악 생성 품질 | VGGish 임베딩에 대한 Fréchet 거리. |
| RTFx | 처리량 | 벽시계 초당 오디오 초 수. |

## 추가 읽기

- [jiwer](https://github.com/jitsi/jiwer) — 정규화 유틸리티가 포함된 WER/CER 라이브러리.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 학습된 MOS 예측기.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 음악 생성 표준.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026년 실시간 순위.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 인간 투표 기반 TTS 리더보드.
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/) — LALM 추론 리더보드.
- [HEAR benchmark](https://hearbenchmark.com/) — 오디오 SSL 벤치마크.
