# 오디오 분류 (Audio Classification) — MFCC 기반 k-NN에서 AST 및 BEATs까지

> "개 짖는 소리 vs 사이렌 소리"부터 "이것은 어떤 언어인가"까지 모든 것이 오디오 분류(audio classification)에 해당합니다. 특징량(features)은 멜(mels)을 사용합니다. 아키텍처는 10년 단위로 진화해 왔습니다. 평가는 AUC, F1, 그리고 클래스별 재현율(per-class recall)을 유지합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## 문제 (The Problem)

10초짜리 클립을 받았습니다. 여러분은 다음을 알고 싶어 합니다: "이것은 무엇인가?" 도시 소음(사이렌, 드릴, 개 짖는 소리), 음성 명령(예/아니오/정지), 언어 식별(영어/스페인어/아랍어), 화자의 감정(화남/중립), 또는 환경 소음(실내/실외, 웅성거림) 등이 해당됩니다. 이 모든 것은 *오디오 분류(audio classification)*이며, 2026년 기준 베이스라인 아키텍처는 성숙한 단계에 도달해 있습니다: `log-mel` → `CNN` 또는 `Transformer` → `softmax`.

핵심적인 어려움은 네트워크가 아닙니다. 바로 데이터입니다. 오디오 데이터셋은 극심한 클래스 불균형(class imbalance), 강력한 도메인 변화(domain shift, 깨끗한 소리 vs 소음이 섞인 소리), 그리고 레이블 노이즈(label noise, 누가 "도시의 웅성거림"과 "식당 소음"을 구분했는가?) 문제를 안고 있습니다. 문제의 80%는 `CNN`을 `Transformer`로 교체하는 것이 아니라, 데이터 큐레이션(curation), 증강(augmentation), 그리고 평가(evaluation)에 있습니다.

## 개념 (The Concept)

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 기반 k-NN (1990년대 베이스라인).** 클립별로 MFCC를 평탄화(Flatten)하고, 레이블이 지정된 뱅크와의 코사인 유사도를 계산하여 상위 K개의 다수결(majority vote)을 반환합니다. 깨끗하고 작은 데이터셋(Speech Commands, ESC-50)에서 놀라울 정도로 강력한 성능을 보입니다. GPU 없이도 실행 가능합니다.

**log-mels 기반 2D CNN (2015-2019).** `(T, n_mels)` 형태의 log-mel을 이미지로 취급합니다. ResNet-18 또는 VGG 스타일을 적용합니다. 시간 축에 대해 글로벌 평균 풀링(Global mean pool)을 수행한 후, 클래스에 대해 Softmax를 적용합니다. 2026년 대부분의 Kaggle 경진대회에서도 여전히 베이스라인으로 사용됩니다.

**Audio Spectrogram Transformer, AST (2021-2024).** log-mel을 패치 단위로 나누고(예: 16×16 패치), 위치 임베딩(position embeddings)을 추가하여 ViT에 입력합니다. 지도 학습(supervised learning) 기반 AudioSet에서 SOTA(mAP 0.485)를 기록했습니다.

**BEATs 및 WavLM-base (2024-2026).** 수백만 시간의 데이터를 통한 자기지도 학습(Self-supervised pretraining)을 수행합니다. 기존에 필요했던 지도 학습 데이터의 1~10%만 사용하여 여러분의 태스크에 맞춰 미세 조정(Fine-tune)할 수 있습니다. 2026년 현재, 비음성 오디오(non-speech audio) 분야의 기본 시작점입니다. BEATs-iter3는 AST보다 1/4의 연산량만 사용하면서도 AudioSet에서 1-2 mAP 더 높은 성능을 보입니다.

**Frozen backbone으로서의 Whisper-encoder (2024).** Whisper의 인코더를 가져와 디코더를 제거하고 선형 분류기(linear classifier)를 부착합니다. 별도의 오디오 증강(audio augmentation) 없이도 언어 식별(language ID) 및 간단한 이벤트 분류에서 SOTA에 근접한 성능을 보여줍니다. "공짜 점심(free lunch)"과 같은 베이스라인입니다.

### 클래스 불균형(Class imbalance)이 진정한 과제입니다

ESC-50: 50개 클래스, 각 클래스당 40개의 클립 — 균형 잡혀 있으며, 쉽습니다. UrbanSound8K: 10개 클래스, 10:1의 불균형을 보입니다. AudioSet: 632개 클래스이며 100,000:1의 롱테일(long tail) 분포를 가집니다. 효과적인 기술들은 다음과 같습니다:

- 학습 중 균형 잡힌 샘플링(Balanced sampling) (평가 시에는 적용하지 않음).
- Mixup: 데이터 증강(augmentation)으로서 두 개의 클립(및 해당 레이블)을 선형적으로 보간(linearly interpolate)합니다.
- SpecAugment: 무작위 시간 및 주파수 대역을 마스킹(mask)합니다. 단순하지만 매우 중요합니다.

### 평가 (Evaluation)

- 다중 클래스 배타적 분류 (Multiclass exclusive, 예: Speech Commands): top-1 정확도(accuracy), top-5 정확도(accuracy).
- 다중 클래스 다중 레이블 분류 (Multiclass multi-label, 예: AudioSet, UrbanSound 스타일): 평균 정밀도(mean average precision, mAP).
- 심각한 불균형 데이터 (Heavily imbalanced): 클래스별 재현율(per-class recall) + 매크로 F1(macro F1).

반드시 숙지해야 할 2026년 수치:

| 벤치마크 (Benchmark) | 베이스라인 (Baseline) | 2026년 SOTA | 출처 (Source) |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 논문 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 리더보드 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 결과 |

```figure
mfcc-pipeline
```

## 직접 구현해 보기 (Build It)

### 1단계: 특징 추출 (featurize)

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 2단계: 고정 길이 요약 (fixed-length summary)

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

단순하지만 강력합니다: 시간에 따른 평균(mean)과 분산(variance)을 계산하면 13개의 계수를 가진 MFCC에 대해 26차원의 고정 임베딩을 생성합니다. 즉각적으로 실행됩니다. 심지어 2017년 최근까지도 ESC-50 데이터셋에서 최첨단(state-of-the-art) 신경망 베이스라인 모델들을 능가했습니다.

### 3단계: k-NN (k-Nearest Neighbors)

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

### 4단계: log-mels에 CNN 적용하기 (upgrade to CNN on log-mels)

PyTorch 예시:

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

파라미터 수는 3M(3백만 개)입니다. 단일 RTX 4090을 사용하여 ESC-50 데이터셋에서 약 10분 만에 학습이 완료되며, 80% 이상의 정확도를 보입니다.

### 5단계: 2026년의 기본값 — BEATs 미세 조정(fine-tune)

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

BEATs의 경우, `beats` 라이브러리를 통해 `microsoft/BEATs-base`를 사용하세요. `transformers` API와 구조는 동일합니다.

## 활용 방법 (Use It)

2026년 기술 스택:

| 상황 | 시작점 (Start with) |
|-----------|-----------|
| 소규모 데이터셋 (<1000 clips) | MFCC 평균값 기반 k-NN (베이스라인) + 오디오 증강(audio augmentation) |
| 중간 규모 데이터셋 (1K–100K) | BEATs 또는 AST 미세 조정(fine-tune) |
| 대규모 데이터셋 (>100K) | 처음부터 학습(Train from scratch) 또는 Whisper-encoder 미세 조정 |
| 실시간, 엣지(edge) 환경 | int8로 양자화된 40-MFCC CNN (KWS 스타일) |
| 다중 레이블 (AudioSet) | BCE loss + mixup + SpecAugment를 적용한 BEATs-iter3 |
| 언어 식별 (Language ID) | MMS-LID, SpeechBrain VoxLingua107 베이스라인 |

의사 결정 규칙: **새 모델을 만들기보다 고정된 백본(frozen backbone)으로 시작하세요**. BEATs 헤드를 미세 조정하는 것만으로도 몇 주가 아닌 단 몇 시간 만에 SOTA 성능의 95%에 도달할 수 있습니다.

## Ship It (실행하기)

`outputs/skill-classifier-designer.md`로 저장하세요. 주어진 오디오 분류 작업(audio classification task)을 위해 아키텍처, 증강(augmentation), 클래스 균형 전략(class-balance strategy), 그리고 평가 지표(eval metric)를 선택해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 이 코드는 4개 클래스의 합성 데이터셋(서로 다른 피치의 순음)을 사용하여 k-NN MFCC 베이스라인을 학습합니다. 혼동 행렬(confusion matrix)을 보고하세요.
2. **중간 (Medium).** `summarize`를 [mean, var, skew, kurtosis]로 교체해 보세요. 동일한 합성 데이터셋에서 4차 모멘트 풀링(4-moment pooling)이 평균+분산(mean+var) 방식보다 성능이 더 좋은가요?
3. **어려움 (Hard).** `torchaudio`를 사용하여 ESC-50 fold 1 데이터셋으로 2D CNN을 학습시켜 보세요. 5-겹 교차 검증(5-fold cross-validation) 정확도를 보고하세요. 여기에 SpecAugment(time mask = 20, freq mask = 10)를 추가하고 그 차이(delta)를 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| AudioSet | 오디오계의 ImageNet | Google의 200만 개 클립, 632개 클래스로 구성된 약지도 학습(weakly-labeled) YouTube 데이터셋입니다. |
| ESC-50 | 소규모 분류 벤치마크 | 50개 클래스 × 40개 클립으로 구성된 환경음 데이터셋입니다. |
| AST | Audio Spectrogram Transformer | log-mel 패치에 ViT를 적용한 모델로, 2021년 SOTA(최고 성능)를 기록했습니다. |
| BEATs | 자기지도 학습 오디오 (Self-supervised audio) | Microsoft 모델이며, iter3 버전은 2026년 기준 AudioSet 성능 1위를 기록하고 있습니다. |
| Mixup | 쌍 증강 (Pair augmentation) | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2` 방식을 사용합니다. |
| SpecAugment | 마스크 기반 증강 (Mask-based augmentation) | 스펙트로그램의 임의의 시간 및 주파수 대역을 0으로 만듭니다. |
| mAP | 주요 다중 레이블 지표 | 클래스 및 임계값(threshold) 전체에 대한 평균 정밀도(Mean average precision)입니다. |

## 추가 읽을거리 (Further Reading)

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021~2024년 기간 동안 표준이 되었던 아키텍처입니다.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024년 이후의 기본 모델입니다.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 가장 지배적인 오디오 증강(audio augmentation) 기법입니다.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 여전히 널리 사용되는 50개 클래스 벤치마크입니다.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632개 클래스로 구성된 YouTube 분류 체계이며, 여전히 업계 표준(gold standard)으로 통합니다.
