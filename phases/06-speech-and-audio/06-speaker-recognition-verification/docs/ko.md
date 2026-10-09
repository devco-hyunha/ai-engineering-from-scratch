# 화자 인식 및 검증

> ASR은 "무엇을 말했는가?"를 묻습니다. 화자 인식은 "누가 말했는가?"를 묻습니다. 수학적 구조는 임베딩과 코사인 유사도 계산으로 동일해 보이지만, 모든 프로덕션 결정은 단일 EER 수치에 달려 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램 및 멜), 5단계 · 22강 (임베딩 모델)
**시간:** 약 45분

## 문제점

사용자가 패스프레이즈를 발화합니다. 확인해야 할 것은 다음과 같습니다: 이 사람이 주장하는 본인인가 (*검증*, 1:1), 등록 뱅크의 첫 번째 사람인가 (*식별*, 1:N)? 아니면 둘 다 아닌, 알 수 없는 화자인가 (*오픈셋*)?

2018년 이전: GMM-UBM + i-vector. 합리적인 EER를 제공했지만, 채널 변화(전화 대 노트북)와 감정 변화에 취약했습니다. 2018–2022년: x-vector (각도 마진으로 학습된 TDNN 백본). 2022년 이후: ECAPA-TDNN 및 WavLM-large 임베딩. 2026년 현재, 이 분야는 세 가지 모델과 하나의 지표가 지배하고 있습니다.

지표는 **EER** — Equal Error Rate입니다. False Accept Rate와 False Reject Rate가 같아지도록 결정 임계값을 설정합니다. 그 교차점이 EER입니다. 모든 논문, 리더보드, 구매 요청에서 사용됩니다.

## 개념

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**파이프라인.** 등록: 대상 화자의 5–30초를 녹음하고 고정 차원 임베딩(ECAPA-TDNN은 192-d, WavLM-large는 256-d)을 계산합니다. 검증: 테스트 발화의 임베딩을 가져와 코사인 유사도를 계산하고 임계값과 비교합니다.

**ECAPA-TDNN (2020, 2026년에도 지배적).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Squeeze-excitation이 포함된 1D 컨볼루션 블록, 멀티 헤드 어텐션 풀링을 거친 후 192-d 선형 레이어로 연결됩니다. VoxCeleb 1+2 (2,700 화자, 110만 발화)에서 Additive Angular Margin loss (AAM-softmax)로 학습되었습니다.

**WavLM-SV (2022+).** 사전 학습된 WavLM-large SSL 백본을 AAM loss로 미세 조정합니다. 품질은 더 높지만 속도는 느립니다 — 300MB 이상 대 15MB.

**x-vector (기본).** TDNN + 통계 풀링. 고전적인 방식이며, CPU / 엣지 환경에서 여전히 유용합니다.

**AAM-softmax.** 각도 공간에 마진 `m`을 추가한 표준 softmax: 올바른 클래스에 대해 `cos(θ + m)`. 클래스 간 각도 분리 강제를 유도합니다. 일반적인 `m=0.2`, 스케일 `s=30`.

### 점수 산정

- 등록 임베딩과 테스트 임베딩 간의 **코사인 유사도**. 임계값 기반 결정.
- **PLDA (확률적 LDA).** 임베딩을 잠재 공간으로 투영하여 같은 화자와 다른 화자 간의 닫힌 형식 우도비를 계산합니다. 코사인 유사도 위에 추가하여 EER를 10–20% 감소시킵니다. 2020년 이전 표준; 현재는 폐쇄 집합 설정에서만 사용.
- **점수 정규화.** `S-norm` 또는 `AS-norm`: 각 점수를 사칭자 평균과 표준 편차의 코호트 기준으로 정규화합니다. 교차 도메인 평가에 필수적입니다.

### 알아야 할 수치 (2026)

| 모델 | VoxCeleb1-O EER | 파라미터 | 처리량 (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (클래식) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 분할 + 임베딩 | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 화자 분리

다중 화자 클립에서 "누가 언제 말했는가"를 식별합니다. 파이프라인: VAD → 분할 → 각 분할 임베딩 → 클러스터링 (응집적 또는 스펙트럴) → 경계 스무딩. 최신 스택: `pyannote.audio` 3.1, 화자 분할 + 임베딩 + 클러스터링을 하나의 호출로 통합합니다. 2026년 AMI 기준 SOTA DER는 약 15% (2022년 23%에서 감소).

```figure
sp-eer-crossover
```

## 구현하기

### 1단계: MFCC 통계 기반 토이 임베딩

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

SOTA와는 거리가 멀며, 교육용으로만 사용됩니다. `code/main.py`는 합성 화자 데이터에 대한 개념 증명(proof-of-concept)으로 이를 사용합니다.

### 2단계: 코사인 유사도 + 임계값

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 3단계: 유사성 쌍에서 EER 계산

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, 임계값)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

(eer, eer_at_threshold)를 반환합니다. 둘 다 보고하세요.

### 4단계: SpeechBrain을 사용한 프로덕션

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: 3-5개의 깨끗한 샘플 임베딩 평균
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 일반 임계값; 데이터에 맞춰 튜닝하세요
```

### 5단계: pyannote로 화자 분리

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 폐쇄 집합 1:1 검증, 엣지 | ECAPA-TDNN + 코사인 임계값 |
| 개방 집합 검증, 클라우드 | WavLM-SV + AS-norm |
| 화자 분리 (회의, 팟캐스트) | `pyannote/speaker-diarization-3.1` |
| 스푸핑 방지 (리플레이 / 딥페이크 탐지) | AASIST 또는 RawNet2 |
| 소형 임베디드 (KWS + 등록) | Titanet-Small (NeMo) |

## 함정

- **채널 불일치.** VoxCeleb (웹 비디오)로 학습된 모델 ≠ 전화 통화 오디오. 항상 대상 채널에서 평가하세요.
- **짧은 발화.** 테스트 오디오가 3초 미만이면 EER이 급격히 악화됩니다.
- **잡음이 있는 등록.** 잡음이 있는 등록 하나만으로도 앵커가 오염됩니다. 깨끗한 샘플을 ≥3개 사용하고 평균을 내세요.
- **조건에 고정된 임계값.** 항상 대상 도메인의 홀드아웃 개발 세트에서 임계값을 튜닝하세요.
- **비정규화 임베딩에 대한 코사인.** 먼저 L2 정규화를 수행하세요. 그렇지 않으면 크기가 지배적이게 됩니다.

## 출시하기

`outputs/skill-speaker-verifier.md`로 저장하세요. 모델, 등록 프로토콜, 임계값 튜닝 계획 및 사기 방지 안전장치를 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 합성 "화자"(다른 톤 프로필)를 생성하고, 등록하며, 100개 쌍의 트라이얼 목록에서 EER을 계산합니다.
2. **중간.** SpeechBrain ECAPA를 사용하여 VoxCeleb1 발화 30개(화자 5명 × 각 6개)를 처리하세요. 코사인 vs PLDA로 EER을 계산하세요.
3. **어려움.** `pyannote.audio`를 사용하여 등록 → 화자 분리 → 검증 전체 파이프라인을 구축하세요. AMI dev 세트에서 DER을 평가하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| EER | 헤드라인 지표 | False Accept = False Reject가 되는 임계값. |
| 검증 | 1:1 | "이 사람이 Alice인가요?" |
| 식별 | 1:N | "누가 말하고 있나요?" |
| 개방 집합 | 미등록 가능성 | 테스트 세트에 미등록 화자가 포함될 수 있습니다. |
| 등록 | 등록하기 | 화자의 참조 임베딩을 계산하는 것. |
| AAM-softmax | 손실 함수 | 가산 각도 마진(additive angular margin)이 있는 Softmax; 클러스터 분리를 강제합니다. |
| PLDA | 고전적 스코어링 | 확률적 LDA; 임베딩 위에 likelihood-ratio 스코어링을 적용합니다. |
| DER | 화자 분리 지표 | Diarization Error Rate — 누락 + 오탐 + 혼동. |

## 추가 읽기

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 고전적인 deep-embedding 논문입니다.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026년 주류 아키텍처입니다.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV 및 화자 분리를 위한 SSL 백본입니다.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 프로덕션 화자 분리 + 임베딩 스택입니다.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 모델 간 현재 EER 순위입니다.
