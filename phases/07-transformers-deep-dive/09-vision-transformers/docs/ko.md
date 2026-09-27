# Vision Transformers (ViT)

> 이미지는 패치(patch)의 그리드입니다. 문장은 토큰(token)의 그리드입니다. 동일한 트랜스포머가 이 둘을 모두 처리합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## 문제점 (The Problem)

2020년 이전의 컴퓨터 비전(Computer Vision)은 곧 컨볼루션(Convolution)을 의미했습니다. ImageNet, COCO, 그리고 탐지(Detection) 벤치마크의 모든 SOTA(State-of-the-Art) 모델들은 CNN 백본을 사용했습니다. 트랜스포머(Transformer)는 언어를 위한 것이었습니다.

Dosovitskiy 외 연구진(2020)의 "An Image is Worth 16x16 Words" 논문은 컨볼루션을 완전히 제거할 수 있음을 보여주었습니다. 이미지를 고정된 크기의 패치(Patch)로 나누고, 각 패치를 선형 투영(Linearly project)하여 임베딩으로 만든 뒤, 이 시퀀스를 일반적인 트랜스포머 인코더(Transformer encoder)에 입력하는 방식입니다. 충분한 규모(ImageNet-21k 사전 학습 또는 그 이상)가 확보되면, ViT는 ResNet 기반 모델과 대등하거나 이를 능가합니다.

ViT는 2026년의 더 광범위한 패턴인 '하나의 아키텍처, 다양한 모달리티(One architecture, many modalities)'의 시작이었습니다. Whisper는 오디오를 토큰화합니다. ViT는 이미지를 토큰화합니다. 로보틱스를 위한 액션 토큰(Action tokens), 비디오를 위한 픽셀 토큰(Pixel tokens)이 존재합니다. 트랜스포머는 상관하지 않습니다. 시퀀스를 입력하면 학습할 뿐입니다.

2026년에 이르러, ViT와 그 후손들(DeiT, Swin, DINOv2, ViT-22B, SAM 3)이 비전 분야의 대부분을 점유하고 있습니다. CNN은 여전히 엣지 디바이스(Edge devices)와 지연 시간(Latency)에 민감한 작업에서 승리합니다. 그 외의 모든 것에는 스택 어딘가에 ViT가 포함되어 있습니다.

## 개념 (The Concept)

![Image → patches → tokens → transformer](../assets/vit.svg)

### 1단계: 패치화 (patchify)

`H × W × C` 크기의 이미지를 `N × (P·P·C)` 형태의 평탄화된 패치(flat patches) 시퀀스로 분할합니다. 일반적인 설정은 다음과 같습니다: `224 × 224` 이미지, `16 × 16` 패치 → 각각 768개의 값을 가진 196개의 패치.

```
image (224, 224, 3) → 16x16x3 패치로 구성된 14 × 14 그리드 → 길이가 768인 196개의 벡터
```

패치 크기(Patch size)는 조절 가능한 레버와 같습니다. 패치가 작을수록 토큰 수가 많아지고 해상도가 높아지지만, 어텐션 비용(attention cost)이 이차 함수적으로 증가합니다. 패치가 클수록 해상도는 낮아지지만, 비용은 저렴해집니다.

### 2단계: 선형 임베딩 (Linear Embedding)

학습 가능한 단일 행렬이 각 평탄화된 패치(flat patch)를 `d_model` 차원으로 투영합니다. 이는 커널 크기 `P`와 스트라이드(stride) `P`를 가진 컨볼루션(convolution)과 동일합니다. PyTorch에서는 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`를 사용하여 단 두 줄로 구현할 수 있습니다.

### 3단계: `[CLS]` 토큰 추가 및 위치 임베딩(Positional Embeddings) 더하기

- 학습 가능한 `[CLS]` 토큰을 앞에 추가합니다. 이 토큰의 최종 은닉 상태(final hidden state)가 분류(classification)에 사용되는 이미지 표현(image representation)이 됩니다.
- 학습 가능한 위치 임베딩(ViT-original) 또는 사인 함수 기반의 2D(sinusoidal 2D) 임베딩(이후 변형 모델들)을 더합니다.
- 2024년 이후에는 위치 정보를 위해 RoPE(Rotary Positional Embedding)를 2D로 확장하여 사용하며, 때로는 명시적인 임베딩 없이 사용하기도 합니다.

### 4단계: 표준 트랜스포머 인코더 (Standard Transformer Encoder)

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 구조를 가진 $L$개의 블록을 쌓습니다. 이는 BERT와 동일합니다. 비전 전용 레이어(vision-specific layers)는 포함되지 않습니다. 이것이 이 논문이 전달하고자 하는 교육적 핵심(pedagogical punchline)입니다.

### 5단계: 헤드 (head)

분류(classification)를 위한 경우: `[CLS]` 은닉 상태(hidden state)를 가져와서 → 선형(linear) → 소프트맥스(softmax)를 적용합니다. DINOv2 또는 SAM의 경우, `[CLS]`를 버리고 패치 임베딩(patch embeddings)을 직접 사용합니다.

### 중요한 변형 모델들 (Variants that mattered)

| 모델 (Model) | 연도 (Year) | 변경 사항 (Change) |
|-------|------|--------|
| ViT | 2020 | 오리지널 모델. 고정된 패치 크기, 전체 글로벌 어텐션(full global attention) 적용. |
| DeiT | 2021 | 증류(Distillation) 기법 사용; ImageNet-1k만으로 학습 가능. |
| Swin | 2021 | 계층적 구조(Hierarchical) 및 이동 윈도우(shifted windows) 적용. 고정된 sub-quadratic 비용. |
| DINOv2 | 2023 | 자기 지도 학습(Self-supervised, 라벨 없음). 최상의 범용 비전 특징(vision features) 추출. |
| ViT-22B | 2023 | 220억(22B) 개의 파라미터; 스케일링 법칙(scaling laws) 적용. |
| SigLIP | 2023 | ViT + 언어 쌍(language pair), 시그모이드 대조 손실(sigmoid contrastive loss) 적용. |
| SAM 3 | 2025 | 무엇이든 분할(Segment anything); ViT-Large + 프롬프트 기반 마스크 디코더(promptable mask decoder). |

### 왜 시간이 걸렸는가 (Why it took a while)

ViT는 CNN의 귀납적 편향(inductive biases, 예: 이동 불변성(translation invariance), 국소성(locality))을 전혀 가지고 있지 않기 때문에, CNN과 대등해지기 위해서는 *매우 많은* 양의 데이터가 필요합니다. 1억 개 이상의 라벨링된 이미지나 강력한 자기지도 학습(self-supervised pretraining)이 없다면, 동일한 연산량(compute) 조건에서 여전히 CNN이 우세합니다. DeiT는 2021년에 증류(distillation) 기법을 통해 이 문제를 해결했으며, DINOv2는 2023년에 자기지도 학습(self-supervision)을 통해 이 문제를 영구적으로 해결했습니다.

```figure
n5-patch-stream
```

## 구현하기 (Build It)

`code/main.py`를 확인해 보세요. 순수 표준 라이브러리(pure-stdlib)를 사용한 patchify + linear embedding + 무결성 검사(sanity checks)로 구성되어 있습니다. 별도의 학습 과정은 포함되어 있지 않습니다. 현실적인 규모의 ViT를 구현하려면 PyTorch와 수 시간의 GPU 연산 시간이 필요하기 때문입니다.

### 1단계: 가짜 이미지 (fake image)

`(R, G, B)` 튜플의 행(row) 리스트로 구성된 24 × 24 RGB 이미지입니다. 6×6 패치(patch)를 사용하여 총 16개의 패치를 생성하며, 각 패치는 108차원의 임베딩 벡터(embedding vector)를 가집니다.

### 2단계: patchify (패치화)

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

래스터 순서(Raster order): 그리드 전체에 걸쳐 행 우선(row-major) 방식으로 진행됩니다. 모든 ViT는 이 순서를 사용합니다.

### 3단계: 선형 임베딩 (linear embed)

각 평탄화된 패치(flat patch)에 무작위 `(patch_flat_size, d_model)` 행렬을 곱합니다. `[CLS]` 토큰을 앞에 추가한 후, 출력 형상(shape)이 `(N_patches + 1, d_model)`인지 확인해 보세요.

### 4단계: 현실적인 ViT의 파라미터 수 계산하기 (count parameters for a realistic ViT)

ViT-Base(12개 레이어, 12개 헤드, $d=768$, 패치 크기=16)의 파라미터 수를 출력해 보세요. ResNet-50(약 25M)과 비교해 보세요. ViT-Base는 약 86M에 달합니다. ViT-Large는 약 307M, ViT-Huge는 약 632M입니다.

## 사용 방법 (Use It)

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 이미지 표현(image representation)
```

**DINOv2 임베딩(embeddings)은 2026년 이미지 특징(image features)의 기본 표준입니다.** 백본(backbone)을 동결(freeze)하고 아주 작은 헤드(head)를 학습시켜 보세요. 분류(classification), 검색(retrieval), 탐지(detection), 캡셔닝(captioning) 모두에 효과적입니다. Meta의 DINOv2 체크포인트는 텍스트가 포함되지 않은 모든 비전 작업에서 CLIP보다 뛰어난 성능을 보여줍니다.

**패치 크기(Patch-size) 선택.** 작은 모델은 16×16(`ViT-B/16`)을 사용합니다. 밀집 예측(dense prediction, 예: 세그멘테이션)에는 8×8 또는 14×14(`SAM`, `DINOv2`)를 사용합니다. 매우 큰 모델은 14×14를 사용합니다.

## Ship It (실전 적용)

`outputs/skill-vit-configurator.md`를 참조하세요. 이 스킬은 데이터셋 크기, 해상도 및 연산 예산(compute budget)이 주어졌을 때, 새로운 비전 태스크를 위한 ViT 변형(variant)과 패치 크기(patch size)를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 패치(patch)의 개수가 `(H/P) * (W/P)`와 일치하는지, 그리고 평탄화된 패치 차원(flat patch dimension)이 `P*P*C`와 일치하는지 확인해 보세요.
2. **중간 (Medium).** 2D 사인형 위치 임베딩(2D sinusoidal positional embeddings)을 구현해 보세요. 각 패치의 `row`와 `col`에 대해 두 개의 독립적인 사인형 코드를 생성하고 이를 결합(concatenate)합니다. 이를 작은 규모의 PyTorch ViT에 입력하고, CIFAR-10 데이터셋에서 학습 가능한 위치 임베딩(learnable positional embeddings)과 정확도를 비교해 보세요.
3. **어려움 (Hard).** 3-레이어 ViT(PyTorch)를 구축하고, 4×4 패치를 사용하여 1,000개의 MNIST 이미지로 학습시켜 보세요. 테스트 정확도를 측정합니다. 이제 동일한 1,000개의 이미지에 대해 DINOv2 사전 학습(pretraining)을 추가해 보세요(단순화 버전: 마스킹된 패치로부터 패치 임베딩을 예측하도록 인코더를 학습시킵니다). 정확도가 향상되나요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 정의 | 실제 의미 |
|------|-----------------|-----------------------|
| Patch (패치) | "비전 트랜스포머의 토큰" | 이미지의 `P × P × C` 영역에 대한 픽셀 값들의 평탄화된(flat) 벡터. |
| Patchify (패치화) | "자르고 펼치기" | 이미지를 겹치지 않는 패치로 슬라이싱한 후, 각 패치를 벡터로 평탄화하는 과정. |
| `[CLS]` token | "이미지 요약본" | 맨 앞에 추가되는 학습 가능한 토큰; 이 토큰의 최종 임베딩이 이미지의 표현(representation)이 됩니다. |
| Inductive bias (귀납적 편향) | "모델이 가정하는 것" | ViT는 CNN보다 사전 지식(priors)이 적으며, 그 격차를 메우기 위해 더 많은 데이터가 필요합니다. |
| DINOv2 | "자기지도 학습 ViT" | 이미지 증강(augmentation)과 모멘텀 교사(momentum teacher)를 사용하여 레이블 없이 학습됨. 2026년 기준 최고의 범용 이미지 특징 추출기. |
| SigLIP | "CLIP의 후계자" | 시그모이드 대조 손실(sigmoid contrastive loss)로 학습된 ViT + 텍스트 인코더; 동일한 연산량 대비 CLIP보다 성능이 우수함. |
| Swin | "윈도우 기반 ViT" | 로컬 어텐션(local attention)과 이동 윈도우(shifted windows)를 사용하는 계층적 ViT; 하위 이차 복잡도(sub-quadratic)를 가짐. |
| Register tokens | "2023년의 트릭" | 어텐션 싱크(attention sinks)를 흡수하기 위해 추가된 몇 개의 학습 가능한 토큰; DINOv2의 특징 추출 성능을 향상시킴. |

## 추가 읽을거리 (Further Reading)

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 논문입니다.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT 논문입니다.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin 논문입니다.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2 논문입니다.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2를 위한 register-token 수정 사항을 다룬 논문입니다.
