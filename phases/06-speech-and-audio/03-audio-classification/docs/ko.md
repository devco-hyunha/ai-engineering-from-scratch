# 오디오 분류 — MFCC에 대한 k-NN부터 AST 및 BEATs까지

> "개 짖는 소리와 사이렌 소리"부터 "이 언어가 무엇인지"까지 모든 것이 오디오 분류입니다. 특징은 멜(mel) 스펙트로그램입니다. 아키텍처는 10년마다 변화합니다. 평가는 AUC, F1, 클래스별 재현율을 유지합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램 및 멜), 3단계 · 06강 (CNN), 5단계 · 08강 (텍스트를 위한 CNN 및 RNN)
**시간:** 약 75분

## 문제점

10초 클립을 받습니다. "이것이 무엇인가?"를 알고 싶을 것입니다. 도시 소리(사이렌, 드릴, 개), 음성 명령(예/아니오/중지), 언어 식별(en/es/ar), 화자 감정(분노/중립), 또는 환경 소리(실내/실외, 소음)입니다. 이 모든 것은 *오디오 분류*이며, 2026년에는 기본 아키텍처가 성숙했습니다: log-mel → CNN 또는 Transformer → softmax.

핵심 난이도는 네트워크가 아닙니다. 데이터입니다. 오디오 데이터셋은 극심한 클래스 불균형, 강한 도메인 이동(클린 vs 노이즈), 그리고 라벨 노이즈("도시 소음" vs "식당 소음"을 누가 결정했나요?)가 있습니다. 문제의 80%는 큐레이션, 증강, 평가이며, CNN을 Transformer로 교체하는 것이 아닙니다.

## 개념

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC에 대한 k-NN (1990년대 기준).** 클립별 MFCC를 평탄화(flatten)하고, 라벨이 붙은 뱅크와의 코사인 유사도(Cosine Similarity)를 계산하여, 상위 K의 다수결 투표를 반환합니다. 클린하고 작은 데이터셋(Speech Commands, ESC-50)에서 놀라울 정도로 강력합니다. GPU 없이 실행됩니다.

**log-mel에 대한 2D CNN (2015-2019).** `(T, n_mels)` log-mel을 이미지로 취급합니다. ResNet-18 또는 VGG 스타일을 적용합니다. 시간 축에 대해 전역 평균 풀링(global mean pool)을 수행합니다. 클래스에 대해 softmax를 적용합니다. 2026년 Kaggle 대회 대부분에서 여전히 기준선입니다.

**오디오 스펙트로그램 트랜스포머, AST (2021-2024).** log-mel을 패치화(예: 16×16 패치)하고, 위치 임베딩을 추가하여 ViT에 입력합니다. 지도 학습에서 AudioSet의 최신 기술(mAP 0.485)입니다.

**BEATs 및 WavLM-base (2024-2026).** 수백만 시간의 데이터에 대해 자기 지도 사전 학습(Self-supervised pretraining)을 수행합니다. 지도 학습에 필요한 데이터의 1-10%로 태스크에 미세 조정(Fine-tuning)합니다. 2026년에는 비음성 오디오의 기본 시작점입니다. BEATs-iter3는 AudioSet에서 AST보다 mAP가 1-2 높으며, 연산량은 1/4입니다.

**Whisper 인코더를 고정 백본으로 사용 (2024).** Whisper의 인코더를 가져와 디코더를 제거하고 선형 분류기를 부착합니다. 오디오 증강 없이 언어 ID 및 단순 이벤트 분류에서 SOTA에 가까운 성능을 달성합니다. "무료 점심" 기반 모델입니다.

### 클래스 불균형이 진정한 도전 과제입니다

ESC-50: 50개 클래스, 각 40개 클립 — 균형 잡혀 있고 쉽습니다. UrbanSound8K: 10개 클래스, 10:1 불균형. AudioSet: 632개 클래스, 100,000:1 롱 테일. 효과가 있는 기법:

- 학습 중 균형 잡힌 샘플링 (평가 중에는 적용하지 않음).
- Mixup: 두 클립과 그 레이블을 선형 보간하여 증강으로 사용.
- SpecAugment: 랜덤한 시간 및 주파수 대역을 마스킹합니다. 단순하지만 중요합니다.

### 평가

- 다중 클래스 단일 레이블 (Speech Commands): top-1 정확도, top-5 정확도.
- 다중 클래스 다중 레이블 (AudioSet, UrbanSound 스타일): 평균 평균 정밀도 (mAP).
- 심한 불균형: 클래스별 재현율 + 매크로 F1.

알아야 할 2026년 수치:

| 벤치마크 | 기반 모델 | 2026 SOTA | 출처 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 논문 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 리더보드 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 결과 |

```figure
mfcc-pipeline
```

## 구현하기

### 1단계: 특징 추출

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 2단계: 고정 길이 요약

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

단순하지만 강력합니다: 시간에 대한 평균 + 분산은 13개 계수 MFCC에 대해 26차원 고정 임베딩을 생성합니다. 즉시 실행됩니다. 2017년까지만 해도 ESC-50에서 최신 NN 기반 모델을 능가했습니다.

### 3단계: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 4단계: 로그 멜 스펙트로그램에서 CNN으로 업그레이드

PyTorch에서:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300만 매개변수. 단일 RTX 4090로 ESC-50에서 약 10분 만에 학습됩니다. 80%+ 정확도.

### 5단계: 사전 학습된 오디오 트랜스포머 미세 조정 (AST 예시)

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

예시에서는 Hub에서 AST를 미세 조정합니다. 2026년 기본 모델인 BEATs는 Hugging Face Hub에 없습니다. [BEATs release in microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats)에서 체크포인트를 다운로드하고 해당 저장소의 `BEATs` 및 `BEATsConfig` 클래스로 로드하세요. 미세 조정 루프는 동일한 형태를 유지합니다.

## 사용하기

2026년 스택:

| 상황 | 시작점 |
|-----------|-----------|
| 작은 데이터셋 (<1000 클립) | MFCC 평균에 대한 k-NN (기본선) + 오디오 증강 |
| 중간 데이터셋 (1K–100K) | BEATs 또는 AST 미세 조정 |
| 큰 데이터셋 (>100K) | 처음부터 학습하거나 Whisper 인코더 미세 조정 |
| 실시간, 엣지 | 40-MFCC CNN, int8로 양자화 (KWS 스타일) |
| 다중 레이블 (AudioSet) | BCE 손실 + mixup + SpecAugment를 사용한 BEATs-iter3 |
| 언어 식별 | MMS-LID, SpeechBrain VoxLingua107 기본선 |

결정 규칙: **새로운 모델이 아닌 고정된 백본(backbone)으로 시작하세요**. BEATs 헤드 미세 조정은 몇 주가 아닌 몇 시간 안에 SOTA의 95%를 달성합니다.

## 출시하기

`outputs/skill-classifier-designer.md`로 저장하세요. 주어진 오디오 분류 작업에 대해 아키텍처, 증강 기법, 클래스 균형 전략 및 평가 지표를 선택해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 이는 4클래스 합성 데이터셋(다른 피치의 순수 톤)에 k-NN MFCC 기본선을 학습합니다. 혼동 행렬을 보고하세요.
2. **중간.** `summarize`을 [평균, 분산, 왜도, 첨도]로 교체하세요. 4모멘트 풀링이 동일한 합성 데이터셋에서 평균+분산보다 더 잘 작동하나요?
3. **어려움.** `torchaudio`을 사용하여 ESC-50 fold 1에 2D CNN을 학습하세요. 5-폴드 교차 검증 정확도를 보고하세요. SpecAugment(시간 마스크 = 20, 주파수 마스크 = 10)를 추가하고 델타를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| AudioSet | 오디오의 ImageNet | Google의 200만 클립, 632클래스 약한 라벨링 YouTube 데이터셋. |
| ESC-50 | 작은 분류 벤치마크 | 환경 소리의 50클래스 × 40클립. |
| AST | 오디오 스펙트로그램 트랜스포머 | 로그-멜 패치에 적용된 ViT; 2021 SOTA. |
| BEATs | 자기 지도 오디오 | Microsoft 모델, 2026년 기준 AudioSet에서 iter3가 선두. |
| Mixup | 쌍 증강 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | 마스크 기반 증강 | 스펙트로그램의 랜덤 시간 및 주파수 밴드를 0으로 설정. |
| mAP | 주요 다중 레이블 지표 | 클래스 및 임계값에 걸친 평균 평균 정밀도. |

## 추가 읽기

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021~2024년 기록된 아키텍처입니다.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024년 이후의 기본값입니다.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 지배적인 오디오 증강 기법입니다.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 현재까지 유지되는 50개 클래스 벤치마크입니다.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632개 클래스의 YouTube 분류 체계로, 여전히 표준입니다.
