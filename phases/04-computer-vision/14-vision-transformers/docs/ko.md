# 비전 트랜스포머 (ViT)(Vision Transformer (ViT))

> 이미지를 패치로 잘라 각각의 패치를 단어로 취급하고, 표준 트랜스포머를 실행합니다. 뒤를 돌아보지 마세요.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 02강 (셀프 어텐션), 4단계 04강 (이미지 분류)
**시간:** 약 45분

## 학습 목표

- 패치 임베딩, 학습된 위치 임베딩, 클래스 토큰 및 트랜스포머 인코더 블록을 처음부터 구현하여 최소한의 ViT를 구축해 보세요
- DeiT와 MAE가 이를 증명하기 전까지 ViT가 대규모 사전 학습 데이터를 필요하다고 여겨졌던 이유를 설명해 보세요
- ViT, Swin, ConvNeXt의 구조적 사전 지식(없음, 지역 윈도우 어텐션, 컨볼루션 백본)을 비교해 보세요
- `timm`와 표준 선형 프로브 / 미세 조정 레시피를 사용하여 작은 데이터셋에 사전 학습된 ViT를 미세 조정해 보세요

## 문제점

10년 동안 컨볼루션은 컴퓨터 비전의 대명사였습니다. CNN은 지역성, 이동 등변성 등 강력한 귀납적 편향을 가지고 있었으며, 아무도 이를 대체할 수 있다고 생각하지 않았습니다. 그러나 Dosovitskiy 등(2020)은 평면 트랜스포머를 평탄화된 이미지 패치에 적용하고 컨볼루션 메커니즘을 전혀 사용하지 않는 것만으로 대규모에서 최고의 CNN과 동등하거나 더 나은 성능을 낼 수 있음을 보여 주었습니다.

함정은 "대규모"였습니다. ImageNet-1k에서의 ViT는 ResNet에 패했습니다. ImageNet-21k 또는 JFT-300M에서 사전 학습한 후 ImageNet-1k에서 미세 조정된 ViT는 ResNet을 이겼습니다. 결론은 트랜스포머가 유용한 사전 지식을 결여하고 있었지만, 충분한 데이터로부터 이를 학습할 수 있다는 것이었습니다. 이후 연구(DeiT, MAE, DINO)는 올바른 학습 레시피(강한 증강, 자기 지도 사전 학습, 증류)를 사용하면 ViT가 작은 데이터에서도 잘 학습된다는 것을 보여 주었습니다.

2026년 현재, 순수 CNN은 엣지 디바이스에서는 여전히 경쟁력 있습니다(ConvNeXt가 가장 강함). 그러나 트랜스포머는 그 외 모든 영역을 지배합니다: 분할(Mask2Former, SegFormer), 탐지(DETR, RT-DETR), 멀티모달(CLIP, SigLIP), 비디오(VideoMAE, VJEPA). ViT 블록 구조는 반드시 알아야 하는 구조입니다.

## 개념

### 파이프라인

```mermaid
flowchart LR
    IMG["이미지<br/>(3, 224, 224)"] --> PATCH["패치 임베딩<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["(196, 768) 토큰으로<br/>평탄화"]
    FLAT --> CAT["[CLS] 토큰을<br/>앞에 추가"]
    CAT --> POS["학습된<br/>위치 임베딩 추가"]
    POS --> ENC["N개의 트랜스포머<br/>인코더 블록"]
    ENC --> CLS["[CLS]<br/>토큰 출력 가져오기"]
    CLS --> HEAD["MLP 분류기"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

7단계. 패치 -> 토큰 -> 어텐션 -> 분류기. 모든 변형(DeiT, Swin, ConvNeXt, MAE 사전 학습)은 7단계 중 하나 또는 두 가지를 변경하고 나머지는 그대로 둡니다.

### 패치 임베딩

첫 번째 컨볼루션이 핵심입니다. 커널 크기 16, 스트라이드 16이므로 224x224 이미지는 14x14 그리드의 16x16 패치가 되며, 각 패치는 768차원 임베딩으로 투사됩니다. 이 단일 컨볼루션이 패치화와 선형 투사를 모두 수행합니다.

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196개 패치 = 196개 토큰. 각 토큰의 특징 차원은 768(ViT-B), 1024(ViT-L), 또는 1280(ViT-H)입니다.

### 클래스 토큰

시퀀스 앞에 추가되는 단일 학습된 벡터:

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

N개의 트랜스포머 블록 이후, `[CLS]` 출력은 전역 이미지 표현입니다. 분류 헤드는 이 하나의 벡터만 읽습니다.

### 위치 임베딩

트랜스포머는 공간 위치에 대한 내장 개념이 없습니다. 모든 토큰에 학습된 벡터를 추가합니다:

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

임베딩은 모델의 매개변수이며, 기울기 기반 학습이 이를 2D 이미지 구조에 적응시킵니다. 사인(cos) 기반 2D 대안도 존재하지만 실무에서는 거의 사용되지 않습니다.

### 트랜스포머 인코더 블록

표준. 다중 헤드 셀프 어텐션, MLP, 잔여 연결, 사전 LayerNorm.

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16은 이러한 블록을 12개 쌓으며, 각 블록은 12개의 어텐션 헤드를 가지고 총 86M 매개변수를 가집니다.

### 왜 사전 LayerNorm(pre-LN)인가

초기 트랜스포머는 사후 LayerNorm(post-LN)(`x = LN(x + sublayer(x))`)을 사용했으며, 워밍업 없이 6-8 레이어 이상 학습하는 데 어려움을 겪었습니다. 사전 LayerNorm(`x = x + sublayer(LN(x))`)은 워밍업 없이 더 깊은 네트워크를 안정적으로 학습합니다. 모든 ViT와 모든 최신 LLM은 사전 LayerNorm을 사용합니다.

### 패치 크기 트레이드오프

- 16x16 패치 -> 196개 토큰, 표준.
- 32x32 패치 -> 49개 토큰, 더 빠르지만 해상도가 낮음.
- 8x8 패치 -> 784개 토큰, 더 세밀하지만 O(n^2) 어텐션 비용이 급격히 증가합니다.

더 큰 패치 = 더 적은 토큰 = 더 빠르지만 공간적 세부 사항이 적음. SwinV2는 계층적 윈도우에서 4x4 패치를 사용합니다.

### ImageNet-1k에서 ViT를 학습하기 위한 DeiT의 레시피

원래 ViT는 CNN을 능가하기 위해 JFT-300M이 필요했습니다. DeiT (Touvron et al., 2020)는 ImageNet-1k만으로 ViT-B를 훈련하여 top-1 정확도 81.8%를 달성했으며, 이는 네 가지 변경 사항에 기인합니다:

1. 강한 증강: RandAugment, Mixup, CutMix, Random Erasing.
2. 확률적 깊이 (Stochastic Depth) (훈련 중 블록 전체를 무작위로 드롭).
3. 반복 증강 (배치당 동일한 이미지를 3번 샘플링).
4. CNN 교사로부터의 지식 증류(Knowledge Distillation) (선택 사항, 정확도를 추가로 높임).

모든 현대적인 ViT 훈련 레시피는 DeiT에서 파생되었습니다.

### Swin vs ConvNeXt

- **Swin** (Liu et al., 2021) — 윈도우 기반 어텐션. 각 블록은 로컬 윈도우 내에서 어텐션하며, 교대로 배치된 블록은 윈도우를 이동하여 윈도우 간 정보를 혼합합니다. 어텐션 연산자를 유지하면서 CNN과 유사한 지역성 사전(locality prior)을 복원합니다.
- **ConvNeXt** (Liu et al., 2022) — Swin의 아키텍처 선택(깊이별 합성곱, LayerNorm, GELU, 역 병목 구조)을 따르도록 재설계된 CNN. 격차는 "어텐션 대 합성곱"이 아니라 "현대적인 훈련 레시피 + 아키텍처"에 있음을 보여 주었습니다.

2026년 현재, ConvNeXt-V2와 Swin-V2 모두 프로덕션급입니다. 올바른 선택은 추론 스택(ConvNeXt는 엣지에서 더 잘 컴파일됨)과 사전 훈련 코퍼스에 따라 달라집니다.

### MAE 사전 훈련

Masked Autoencoder (He et al., 2022): 패치의 75%를 무작위로 마스킹하고, 인코더가 가시적인 25%만 처리하도록 훈련하며, 인코더의 출력으로부터 마스킹된 패치를 복원하기 위해 작은 디코더를 훈련합니다. 사전 훈련 후, 디코더를 버리고 인코더를 미세 조정(Fine-tuning)합니다.

MAE는 ViT를 ImageNet-1k만으로 훈련 가능하게 만들며, SOTA를 달성하고, 현재 기본 자기 지도 학습(self-supervised) 레시피가 되었습니다.

```figure
batchnorm-inference
```

## 구현하기

### 1단계: 패치 임베딩(Patch Embedding)

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

하나의 합성곱, 하나의 평탄화(flatten), 하나의 전치(transpose). 이것이 이미지에서 토큰으로 변환하는 전체 단계입니다.

### 2단계: 트랜스포머 블록

Pre-LN, 멀티 헤드 셀프 어텐션(Self-Attention), GELU가 포함된 MLP, 잔여 연결(residual connections).

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention`는 헤드 분할, 스케일된 점곱(scaled dot-product), 출력 투사를 처리합니다. `batch_first=True`이므로 형태는 `(N, seq, dim)`입니다.

### 3단계: ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

약 2.8M 매개변수 — CPU에서 다루기 쉬운 작은 ViT입니다. 실제 ViT-B는 86M이며, `dim=768, depth=12, num_heads=12`을 사용하여 동일한 클래스 정의를 가집니다.

### 4단계: Sanity check — 단일 이미지 추론

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

오류 없이 실행되어야 합니다. 확률의 합은 1입니다.

## 사용하기

`timm`는 ImageNet 사전 학습 가중치를 가진 모든 ViT 변형을 제공합니다. 한 줄로:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm`는 2026년 비전 트랜스포머(Vision Transformer)의 기본 프로덕션 도구입니다. ViT, DeiT, Swin, Swin-V2, ConvNeXt, ConvNeXt-V2, MaxViT, MViT, EfficientFormer 및 수십 개의 기타 모델을 동일한 API로 지원합니다.

멀티모달 작업(이미지 + 텍스트)의 경우, `transformers`는 CLIP, SigLIP, BLIP-2, LLaVA를 제공합니다. 이 모든 모델의 이미지 인코더는 ViT 변형입니다.

## 출시하기

이 강의에서 생성되는 결과물:

- `outputs/prompt-vit-vs-cnn-picker.md` — 데이터셋 크기, 컴퓨팅 자원, 추론 스택에 따라 ViT, ConvNeXt, Swin 중 하나를 선택하는 프롬프트입니다.
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — ViT의 패치 임베딩(Patch Embedding)과 위치 임베딩(Positional Embedding)의 형태가 모델의 예상 시퀀스 길이와 일치하는지 검증하여 가장 흔한 이식 버그를 잡아내는 스킬입니다.

## 연습 문제

1. **(쉬움)** 위 작은 ViT의 순방향 전파(forward pass)에서 모든 중간 텐서(Tensor)의 형태를 출력해 보세요. 확인 사항: 입력 `(N, 3, 64, 64)` -> 패치 `(N, 16, 192)` -> CLS 포함 `(N, 17, 192)` -> 분류기 입력 `(N, 192)` -> 출력 `(N, num_classes)`.
2. **(중간)** 4강의 합성 CIFAR 데이터셋으로 사전 학습된 `timm` ViT-S/16을 미세 조정(Fine-tuning)해 보세요. 같은 데이터로 ResNet-18을 미세 조정했을 때와 비교하고, 학습 시간과 최종 정확도를 보고하세요.
3. **(어려움)** 작은 ViT에 대해 MAE 사전 학습을 구현해 보세요: 패치의 75%를 마스킹하고, 인코더와 작은 디코더를 훈련하여 마스킹된 패치를 복원합니다. 사전 학습 전후의 합성 데이터에 대한 선형 프로브(linear-probe) 정확도를 평가하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 패치 임베딩(Patch Embedding) | "첫 번째 conv" | 커널 크기 = 스트라이드 = 패치 크기인 합성곱(convolution); 이미지를 토큰 임베딩의 그리드로 변환합니다 |
| 클래스 토큰(Class Token) | "[CLS]" | 토큰 시퀀스 앞에 추가되는 학습된 벡터; 최종 출력은 전역 이미지 표현(global image representation)입니다 |
| 위치 임베딩(Positional Embedding) | "학습된 pos" | 모든 토큰에 더해지는 학습된 벡터로, 트랜스포머가 각 패치의 위치를 알 수 있게 합니다 |
| Pre-LN | "서브레이어 앞의 LayerNorm" | 안정적 트랜스포머 변형: `LN(x + sublayer(x))` 대신 `x + sublayer(LN(x))` |
| Multi-head attention | "병렬 어텐션" | 표준 트랜스포머 어텐션을 num_heads개의 독립적인 하위 공간으로 분할한 후, 이후에 연결(concatenate) |
| ViT-B/16 | "Base, patch 16" | 표준 크기: dim=768, depth=12, heads=12, patch_size=16, image=224; 약 86M 파라미터 |
| DeiT | "데이터 효율적인 ViT" | ImageNet-1k만으로 강한 증강을 적용하여 학습한 ViT; 대규모 사전 학습 데이터셋이 반드시 필요하지 않음을 증명 |
| MAE | "마스크 오토인코더" | 자기 지도 사전 학습: 패치의 75%를 마스킹하고 복원; 지배적인 ViT 사전 학습 레시피 |

## 추가 읽기

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 논문
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — ImageNet-1k만으로 ViT를 학습하는 방법
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 사전 학습
- [timm documentation](https://huggingface.co/docs/timm) — 프로덕션에서 사용할 모든 비전 트랜스포머의 참고 자료
