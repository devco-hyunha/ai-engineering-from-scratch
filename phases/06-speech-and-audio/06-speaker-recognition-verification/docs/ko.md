# 화자 인식 및 검증 (Speaker Recognition & Verification)

> ASR(자동 음성 인식)이 "무엇을 말했는가?"를 묻는다면, 화자 인식은 "누가 말했는가?"를 묻습니다. 수학적 구조는 임베딩(embeddings)과 코사인 유사도(cosine similarity)를 사용하는 것으로 동일해 보이지만, 모든 프로덕션 결정은 단 하나의 EER(Equal Error Rate) 수치에 달려 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## 문제 (The Problem)

사용자가 암호를 말합니다. 여러분은 다음을 알고 싶어 합니다: 이 사람이 주장하는 본인이 맞는지 (*검증(verification)*, 1:1), 아니면 등록된 데이터베이스의 인물 중 한 명인지 (*식별(identification)*, 1:N)? 혹은 둘 다 아닌 미지의 화자인지 (*개방형 집합(open-set)*)?

2018년 이전: GMM-UBM + i-vectors. EER은 준수하지만 채널 변화(전화기 vs 노트북)와 감정에 취약합니다. 2018~2022년: x-vectors (angular margin으로 학습된 TDNN 백본). 2022년 이후: ECAPA-TDNN 및 WavLM-large 임베딩. 2026년경에는 세 가지 모델과 한 가지 지표가 이 분야를 지배하게 됩니다.

그 지표는 바로 **EER** — 동일 오류율(Equal Error Rate)입니다. 오인식률(False Accept Rate)과 오거부율(False Reject Rate)이 같아지도록 결정 임계값(decision threshold)을 설정하세요. 그 교차점이 바로 EER입니다. 모든 논문, 모든 리더보드, 모든 조달 공고에서 사용됩니다.

## 개념 (The Concept)

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**파이프라인 (The pipeline).** 등록(Enrollment): 대상 화자의 음성을 5~30초간 녹음합니다. 고정 차원의 임베딩(ECAPA-TDNN은 192차원, WavLM-large는 256차원)을 계산합니다. 검증(Verification): 테스트 발화의 임베딩을 얻습니다. 코사인 유사도(cosine similarity)를 계산하고 임계값(threshold)과 비교합니다.

**ECAPA-TDNN (2020년 발표, 2026년 현재도 주류).** 채널 어텐션, 전파 및 집계(Channel Attention, Propagation and Aggregation)를 강조한 시간 지연 신경망(Time-Delay Neural Network)입니다. Squeeze-excitation이 포함된 1D conv 블록, 멀티 헤드 어텐션 풀링(multi-head attention pooling)을 거쳐 192차원의 선형 레이어(`linear layer`)로 연결됩니다. VoxCeleb 1+2(화자 2,700명, 발화 110만 개) 데이터셋에서 Additive Angular Margin 손실(AAM-softmax)을 사용하여 학습되었습니다.

**WavLM-SV (2022+).** 사전 학습된 WavLM-large SSL 백본을 AAM 손실을 사용하여 미세 조정(Fine-tune)합니다. 품질은 더 높지만, 모델 크기가 300MB 이상으로 ECAPA-TDNN(약 15MB)보다 느리고 무겁습니다.

**x-vector (베이스라인).** TDNN와 통계적 풀링(statistics pooling)을 결합한 방식입니다. 고전적인 방식이며, CPU 또는 엣지(edge) 환경에서 여전히 유용합니다.

**AAM-softmax.** 각도 공간(angular space)에 마진 `m`을 추가한 표준 소프트맥스 방식입니다. 정답 클래스에 대해 `cos(θ + m)`을 적용합니다. 이는 클래스 간의 각도 분리(angular separation)를 강제합니다. 전형적인 설정값은 `m=0.2`, 스케일 `s=30`입니다.

### 점수 산출 (Scoring)

- **코사인 유사도 (Cosine)**: 등록(enrollment) 임베딩과 테스트 임베딩 사이의 코사인 유사도를 측정합니다. 임계값(Threshold) 기반의 의사결정을 수행합니다.
- **PLDA (Probabilistic LDA)**: 임베딩을 잠재 공간(latent space)으로 투영하여, 동일 화자 여부에 따른 폐쇄형 가능도비(closed-form likelihood ratio)를 계산합니다. 코사인 유사도 위에 추가하여 EER(Equal Error Rate)을 10~20% 감소시킬 수 있습니다. 2020년 이전의 표준 방식이었으나, 현재는 폐쇄형 설정(closed-set setups)에서만 사용됩니다.
- **점수 정규화 (Score normalization)**: `S-norm` 또는 `AS-norm`을 사용합니다. 각 점수를 가짜 화자(imposter) 집단의 평균 및 표준편차를 기준으로 정규화합니다. 교차 도메인 평가(cross-domain eval)를 위해 필수적입니다.

### 알아두어야 할 수치 (2026)

| 모델 (Model) | VoxCeleb1-O EER | 파라미터 (Params) | 처리량 (Throughput, A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 화자 분할 (Diarization)

다중 화자 클립에서 "누가 언제 말했는가"를 파악하는 기술입니다. 파이프라인은 다음과 같습니다: VAD(음성 활동 감지) → 세그먼트 분할 → 각 세그먼트 임베딩 → 클러스터링(응집형 또는 스펙트럴) → 경계 평활화(smooth boundaries). 최신 스택으로는 `pyannote.audio` 3.1이 있으며, 이는 화자 세그먼트 분할 + 임베딩 + 클러스터링을 단 한 번의 호출로 통합하여 제공합니다. 2026년 AMI 데이터셋 기준 SOTA(최고 수준) DER은 약 15%입니다 (2022년 23%에서 감소).

```figure
sp-eer-crossover
```

## 직접 구현해 보기 (Build It)

### 1단계: MFCC 통계량을 이용한 토이 임베딩(toy embedding)

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26차원
```

최첨단(SOTA) 기술과는 거리가 멀며, 오직 교육용으로만 사용됩니다. `code/main.py`에서는 이를 합성 화자 데이터(synthetic speaker data)에 대한 개념 증명(proof-of-concept) 용도로 사용합니다.

### 2단계: 코사인 유사도(cosine similarity) + 임계값(threshold)

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 3단계: 유사도 쌍으로부터의 EER (EER from similarity pairs)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

`(eer, threshold_at_eer)`를 반환합니다. 두 값을 모두 보고하세요.

### 4단계: SpeechBrain을 이용한 프로덕션(Production) 적용

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 등록(enroll): 3~5개의 깨끗한 샘플의 임베딩을 평균화합니다.
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 검증(verify)
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA의 일반적인 임계값입니다. 데이터에 맞춰 조정해 보세요.
```

### 5단계: pyannote를 이용한 화자 분할(Diarization)

```python
from pyannote.audio import Pipeline

# 사전 학습된 파이프라인 로드
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# 화자 분할 수행
diarization = pipe("meeting.wav", num_speakers=None)

# 결과 출력
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 활용하기 (Use It)

2026년 권장 스택:

| 상황 (Situation) | 선택 (Pick) |
|-----------|------|
| 폐쇄형(Closed-set) 1:1 검증, 엣지 환경 | ECAPA-TDNN + cosine threshold |
| 개방형(Open-set) 검증, 클라우드 환경 | WavLM-SV + AS-norm |
| 화자 분할 (회의, 팟캐스트) | `pyannote/speaker-diarization-3.1` |
| 안티 스푸핑 (재생 / 딥페이크 탐지) | AASIST 또는 RawNet2 |
| 초소형 임베디드 (KWS + 등록) | Titanet-Small (NeMo) |

## 주의 사항 (Pitfalls)

- **채널 불일치 (Channel mismatch).** VoxCeleb(웹 비디오)로 학습된 모델은 전화 통화 오디오와 다릅니다. 항상 대상 채널(target channel)에서 평가하세요.
- **짧은 발화 (Short utterances).** 테스트 오디오가 3초 미만일 경우 EER이 급격히 저하됩니다.
- **노이즈가 포함된 등록 (Enrollment with noise).** 노이즈가 섞인 단 하나의 등록 데이터가 앵커(anchor)를 오염시킬 수 있습니다. 3개 이상의 깨끗한 샘플을 사용하여 평균을 내세요.
- **조건에 관계없는 고정 임계값 (Fixed threshold across conditions).** 항상 대상 도메인의 별도 검증 세트(held-out dev set)에서 임계값을 조정하세요.
- **정규화되지 않은 임베딩에 대한 코사인 유사도 (Cosine on non-normalized embeddings).** 먼저 `L2-normalize`를 수행하세요. 그렇지 않으면 벡터의 크기(magnitude)가 결과를 지배하게 됩니다.

## Ship It (실전 적용)

`outputs/skill-speaker-verifier.md`로 저장하세요. 모델, 등록 프로토콜(enrollment protocol), 임계값 조정 계획(threshold-tuning plan), 그리고 부정 사용 방지책(fraud safeguards)을 선정하세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 합성된 "화자(speakers)"(서로 다른 톤 프로필)를 생성하고 등록한 뒤, 100쌍의 테스트 리스트에 대해 EER을 계산합니다.
2. **중간 (Medium).** 30개의 VoxCeleb1 발화(화자 5명 × 각 6개)에 SpeechBrain ECAPA를 사용해 보세요. 코사인 유사도(cosine)와 PLDA를 사용하여 EER을 계산합니다.
3. **어려움 (Hard).** `pyannote.audio`를 사용하여 전체 `enroll` → `diarize` → `verify` 파이프라인을 구축해 보세요. AMI dev 세트에서 DER을 평가합니다.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| EER | 헤드라인 지표 | False Accept(오인식)와 False Reject(오거부)가 같아지는 임계값(Threshold). |
| Verification (검증) | 1:1 | "이 사람이 Alice인가?" |
| Identification (식별) | 1:N | "누가 말하고 있는가?" |
| Open-set (오픈셋) | 미등록 가능성 | 테스트 세트에 등록되지 않은 화자가 포함될 수 있음. |
| Enrollment (등록) | 등록 과정 | 화자의 참조 임베딩(reference embedding)을 계산하는 과정. |
| AAM-softmax | 손실 함수(Loss) | 가산 각도 마진(additive angular margin)을 적용한 Softmax; 클러스터 간 분리를 강제함. |
| PLDA | 전통적인 스코어링 | Probabilistic LDA; 임베딩 기반의 우도비(likelihood-ratio) 스코어링 방식. |
| DER | 다이어리제이션 지표 | Diarization Error Rate — Miss(미검출) + False Alarm(오검출) + Confusion(혼동)의 합. |

## 추가 읽을거리 (Further Reading)

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 고전적인 딥 임베딩(deep-embedding) 논문입니다.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020~2026년 사이 주도적인 아키텍처입니다.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 화자 인식(SV) 및 화자 분할(diarization)을 위한 SSL 백본입니다.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 프로덕션급 화자 분할 및 임베딩 스택입니다.
- [VoxCeleb 리더보드 (2026년 업데이트)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 모델별 최신 EER 순위를 확인할 수 있습니다.
