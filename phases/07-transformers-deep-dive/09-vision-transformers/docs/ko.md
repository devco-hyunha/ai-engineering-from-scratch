# 비전 트랜스포머 (ViT)(Vision Transformers (ViT))

> 이미지는 패치 격자입니다. 문장은 토큰 격자입니다. 동일한 트랜스포머가 둘 다 처리합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 05강 (전체 트랜스포머), 4단계 · 03강 (CNN), 4단계 · 14강 (비전 트랜스포머 소개)
**시간:** 약 45분

## 문제점

2020년 이전까지 컴퓨터 비전은 합성곱을 의미했습니다. ImageNet, COCO 및 탐지 벤치마크의 모든 SOTA는 CNN 백본을 사용했습니다. 트랜스포머는 언어를 위한 것이었습니다.

Dosovitskiy 등(2020) — "An Image is Worth 16x16 Words" — 합성곱을 완전히 제거할 수 있음을 보여 주었습니다. 이미지를 고정 크기의 패치로 잘라 각 패치를 선형적으로 임베딩으로 투사하고, 시퀀스를 바닐라 트랜스포머 인코더에 입력합니다. 충분한 규모(ImageNet-21k 사전 학습 이상)에서는 ViT가 ResNet 기반 모델과 동등하거나 더 나은 성능을 냅니다.

ViT는 2026년 더 넓은 패턴의 시작이었습니다. 하나의 아키텍처, 여러 모달리티. Whisper는 오디오를 토큰화합니다. ViT는 이미지를 토큰화합니다. 로보틱스를 위한 액션 토큰, 비디오를 위한 픽셀 토큰. 트랜스포머는 상관하지 않습니다. 시퀀스를 입력하면 학습합니다.

2026년 현재, ViT와 그 후손들(DeiT, Swin, DINOv2, ViT-22B, SAM 3)은 비전의 대부분을 장악했습니다. CNN은 여전히 엣지 디바이스와 지연에 민감한 작업에서 승리합니다. 스택의 어딘가에 ViT가 있는 모든 것들입니다.

## 개념

![Image → patches → tokens → transformer](../assets/vit.svg)

### 1단계 — 패치화

`H × W × C` 이미지를 `N × (P·P·C)` 평평한 패치 시퀀스로 분할합니다. 일반적인 설정: `224 × 224` 이미지, `16 × 16` 패치 → 각각 768 값인 196개 패치.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

패치 크기가 레버입니다. 더 작은 패치 = 더 많은 토큰, 더 좋은 해상도, 2차적 어텐션 비용. 더 큰 패치 = 더 거친, 더 저렴한.

### 2단계 — 선형 임베딩

단일 학습된 행렬이 각 평평한 패치를 `d_model`로 투사합니다. 커널 크기가 `P`이고 스트라이드가 `P`인 합성곱과 동일합니다. PyTorch에서는 문자 그대로 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`입니다. 2줄 구현입니다.

### 3단계 — `[CLS]` 토큰을 앞에 붙이고 위치 임베딩 추가

- 학습 가능한 `[CLS]` 토큰을 앞에 붙입니다. 그 최종 은닉 상태는 분류에 사용되는 이미지 표현입니다.
- 학습 가능한 위치 임베딩(ViT 원본) 또는 2D 사인 임베딩(후기 변형)을 추가합니다.
- 2024년 이후 RoPE는 위치를 위해 2D로 확장되었으며, 때로는 명시적인 임베딩 없이 사용됩니다.

### 4단계 — 표준 트랜스포머 인코더

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 블록을 L개 쌓습니다. BERT와 동일합니다. 비전 전용 레이어는 없습니다. 이것이 논문에서 교육적으로 가장 중요한 포인트입니다.

### 5단계 — 헤드

분류의 경우: `[CLS]` 은닉 상태 → 선형 → 소프트맥스를 수행합니다. DINOv2나 SAM의 경우 `[CLS]`를 버리고 패치 임베딩을 직접 사용합니다.

### 중요했던 변형들

| 모델 | 연도 | 변경 사항 |
|-------|------|--------|
| ViT | 2020 | 원본. 고정된 패치 크기, 완전한 전역 어텐션. |
| DeiT | 2021 | 증류; ImageNet-1k에서만 학습 가능. |
| Swin | 2021 | 이동 윈도우를 사용하는 계층적 구조. 고정된 준-2차 비용. |
| DINOv2 | 2023 | 자기 지도 학습(레이블 없음). 가장 일반적인 비전 특징. |
| ViT-22B | 2023 | 22B 매개변수; 스케일링 법칙 적용. |
| SigLIP | 2023 | ViT + 언어 쌍, 시그모이드 대조 손실. |
| SAM 3 | 2025 | Segment Anything; ViT-Large + 프롬프트 가능한 마스크 디코더. |

### 왜 시간이 오래 걸렸는가

ViT는 CNN의 귀납적 편향(병합 불변성, 지역성)이 전혀 없기 때문에 CNN과 성능을 맞추려면 *매우 많은* 데이터가 필요합니다. 1억 개 이상의 레이블이 지정된 이미지나 강력한 자기 지도 사전 학습이 없으면, 동일한 연산량에서 CNN이 여전히 승리합니다. DeiT는 2021년 증류 기법으로 이를 해결했고, DINOv2는 2023년 자기 지도 학습으로 영구적으로 해결했습니다.

```figure
n5-patch-stream
```

## 구현하기

`code/main.py`를 참조하세요. 순수 표준 라이브러리를 사용한 패치화 + 선형 임베딩 + 무결성 검사. 학습은 없습니다. 현실적인 규모의 ViT는 PyTorch와 몇 시간의 GPU 시간이 필요합니다.

### 1단계: 가짜 이미지

`(R, G, B)` 튜플의 행 목록으로 구성된 24 × 24 RGB 이미지. 6×6 패치를 사용하므로 → 16개 패치, 각각 108차원 임베딩 벡터가 됩니다.

### 2단계: 패치화

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

래스터 순서: 그리드 전체에서 행 우선 순서. 모든 ViT는 이 순서를 사용합니다.

### 3단계: 선형 임베딩

각 평평한 패치를 랜덤 `(patch_flat_size, d_model)` 행렬과 곱합니다. `[CLS]`를 앞에 붙인 후 출력 모양이 `(N_patches + 1, d_model)`인지 확인합니다.

### 4단계: 현실적인 ViT의 매개변수 개수를 세어 보세요

ViT-Base의 매개변수 개수를 출력해 보세요: 12층, 12헤드, d=768, patch=16. ResNet-50 (~25M)과 비교해 보세요. ViT-Base는 ~86M입니다. ViT-Large는 ~307M입니다. ViT-Huge는 ~632M입니다.

## 사용하기

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196개 패치
cls_emb = out[:, 0]                       # 이미지 표현
```

**DINOv2 임베딩은 이미지 특징을 위한 2026년 기본값입니다.** 백본을 고정하고 작은 헤드를 학습하세요. 분류, 검색, 탐지, 캡셔닝에 모두 작동합니다. Meta의 DINOv2 체크포인트는 모든 비텍스트 비전 작업에서 CLIP보다 성능이 뛰어납니다.

**패치 크기 선택.** 작은 모델은 16×16을 사용합니다 (ViT-B/16). 밀집 예측 (분할)은 8×8 또는 14×14를 사용합니다 (SAM, DINOv2). 매우 큰 모델은 14×14를 사용합니다.

## 출시하기

`outputs/skill-vit-configurator.md`을 참고하세요. 이 스킬은 데이터셋 크기, 해상도, 컴퓨팅 예산을 고려하여 새로운 비전 작업에 적합한 ViT 변형과 패치 크기를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 패치 개수가 `(H/P) * (W/P)`과 같고, 평면 패치 차원이 `P*P*C`과 같은지 확인하세요.
2. **중간.** 2D 사인 positional 임베딩을 구현하세요 — 각 패치의 `row`과 `col`에 대해 독립적인 두 사인 코드를 생성하고 연결합니다. 이를 작은 PyTorch ViT에 입력하여 CIFAR-10에서 학습 가능한 positional 임베딩과 정확도를 비교하세요.
3. **어려움.** 3층 ViT (PyTorch)를 구축하고, 4×4 패치를 사용하여 1,000개의 MNIST 이미지로 학습하세요. 테스트 정확도를 측정하세요. 이제 같은 1,000개 이미지에 DINOv2 사전 학습을 추가하세요 (단순화: 마스킹된 패치에서 패치 임베딩을 예측하도록 인코더를 학습). 정확도가 향상되나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 패치 | "비전 트랜스포머 토큰" | 이미지의 `P × P × C` 영역에 대한 픽셀 값의 평면 벡터. |
| Patchify | "분할 + 평탄화" | 이미지를 겹치지 않는 패치로 자르고, 각 패치를 벡터로 평탄화합니다. |
| `[CLS]` 토큰 | "이미지 요약" | 앞에 추가된 학습 가능한 토큰; 최종 임베딩이 이미지 표현이 됩니다. |
| 귀납적 편향 | "모델이 가정하는 것" | ViT는 CNN보다 사전 가정이 적습니다; 격차를 메우기 위해 더 많은 데이터가 필요합니다. |
| DINOv2 | "자기 지도 ViT" | 레이블 없이 이미지 증강 + 모멘텀 티처로 학습. 2026년 기준 최상의 범용 이미지 특징. |
| SigLIP | "CLIP의 후속작" | 시그모이드 대비 손실로 학습된 ViT + 텍스트 인코더; 동일한 연산량에서 CLIP보다 우수. |
| Swin | "윈도우 기반 ViT" | 지역 어텐션 + 이동 윈도우를 사용하는 계층적 ViT; 준이차 계산 복잡도. |
| Register tokens | "2023년 트릭" | 어텐션 싱크를 흡수하는 몇 개의 추가 학습 가능한 토큰; DINOv2 특징을 개선. |

## 추가 읽기

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 논문.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2용 register-token 수정.
