# 비전 트랜스포머 (Vision Transformers, ViT)

> 이미지를 패치로 자르고, 각 패치를 단어처럼 취급하고, 표준 트랜스포머를 돌리세요. 뒤돌아보지 마세요.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## 학습 목표 (Learning Objectives)

- 패치 임베딩, 학습된 위치 임베딩, 클래스 토큰, 트랜스포머 인코더 블록을 처음부터 구현해 최소 ViT를 만듭니다
- DeiT와 MAE가 달리 증명하기 전까지 ViT가 대규모 사전학습 데이터가 필요하다고 여겨진 이유를 설명합니다
- ViT, Swin, ConvNeXt를 아키텍처 사전(없음, 로컬 윈도우 어텐션, conv 백본)으로 비교합니다
- `timm`과 표준 linear-probe / fine-tune 레시피로 작은 데이터셋에서 사전학습 ViT를 파인튜닝합니다

## 문제 상황 (The Problem)

십 년 동안 컨볼루션은 컴퓨터 비전과 동의어였습니다. CNN에는 강한 귀납 편향 — 국소성, 이동 등변성 — 이 있어 아무도 대체할 수 없다고 생각했습니다. 그러다 Dosovitskiy et al.(2020)이 컨볼루션 기계 없이, 평탄화된 이미지 패치에 평범한 트랜스포머를 적용해도 규모에서 최고의 CNN에 맞먹거나 이길 수 있음을 보였습니다.

함정은 "규모에서"였습니다. ImageNet-1k의 ViT는 ResNet에 졌습니다. ImageNet-21k 또는 JFT-300M에서 사전학습한 뒤 ImageNet-1k에 파인튜닝한 ViT는 이겼습니다. 결론은 트랜스포머에 유용한 사전이 없지만 충분한 데이터에서 배울 수 있다는 것이었습니다. 이후 연구(DeiT, MAE, DINO)는 올바른 학습 레시피 — 강한 증강, 자기지도 사전학습, 증류 — 로 작은 데이터에서도 ViT가 잘 학습함을 보였습니다.

2026년에도 순수 CNN은 엣지 기기에서 여전히 경쟁력 있지만(ConvNeXt가 가장 강함), 다른 모든 것은 트랜스포머가 지배합니다: 세그멘테이션(Mask2Former, SegFormer), 검출(DETR, RT-DETR), 멀티모달(CLIP, SigLIP), 비디오(VideoMAE, VJEPA). ViT 블록 구조가 알아야 할 것입니다.

## 핵심 개념 (The Concept)

### 파이프라인 (The pipeline)

```mermaid
flowchart LR
    IMG["이미지<br/>(3, 224, 224)"] --> PATCH["패치 임베딩<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["평탄화하여<br/>(196, 768) 토큰"]
    FLAT --> CAT["[CLS] 토큰<br/>앞에 붙이기"]
    CAT --> POS["학습된<br/>위치 임베딩 더하기"]
    POS --> ENC["N개 트랜스포머<br/>인코더 블록"]
    ENC --> CLS["[CLS]<br/>토큰 출력 취하기"]
    CLS --> HEAD["MLP 분류기"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

일곱 단계. 패치 -> 토큰 -> 어텐션 -> 분류기. 모든 변형(DeiT, Swin, ConvNeXt, MAE 사전학습)은 일곱 중 하나·둘을 바꾸고 나머지는 그대로 둡니다.

### 패치 임베딩 (Patch embedding)

첫 conv가 비밀입니다. 커널 크기 16, stride 16이므로 224x224 이미지가 16x16 패치의 14x14 격자가 되고, 각각이 768차원 임베딩으로 사영됩니다. 그 단일 conv가 패치화와 선형 사영을 모두 합니다.

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 패치 = 196 토큰. 각 토큰의 특징 차원은 768(ViT-B), 1024(ViT-L), 또는 1280(ViT-H)입니다.

### 클래스 토큰 (Class token)

시퀀스 앞에 붙이는 단일 학습 벡터:

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

N개 트랜스포머 블록 이후 `[CLS]` 출력이 전역 이미지 표현입니다. 분류 헤드는 이 벡터 하나만 읽습니다.

### 위치 임베딩 (Positional embedding)

트랜스포머에는 공간 위치에 대한 내장 개념이 없습니다. 모든 토큰에 학습된 벡터를 더합니다:

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

임베딩은 모델의 파라미터이며; 기울기 기반 학습이 2D 이미지 구조에 맞게 적응합니다. 사인파 2D 대안이 있지만 실무에서는 거의 쓰이지 않습니다.

### 트랜스포머 인코더 블록 (Transformer encoder block)

표준입니다. Multi-head self-attention, MLP, 잔차 연결, pre-LayerNorm.

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16은 이 블록을 12개 쌓고, 각각 어텐션 헤드 12개로 총 86M 파라미터입니다.

### 왜 pre-LN인가 (Why pre-LN)

초기 트랜스포머는 post-LN(`x = LN(x + sublayer(x))`)을 썼고, 워밍업 없이 6–8층 이상 학습하기 어려웠습니다. Pre-LN(`x = x + sublayer(LN(x))`)은 워밍업 없이 더 깊은 네트워크를 안정적으로 학습합니다. 모든 ViT와 모든 현대 LLM이 pre-LN을 씁니다.

### 패치 크기 트레이드오프 (Patch size trade-off)

- 16x16 패치 -> 196 토큰, 표준.
- 32x32 패치 -> 49 토큰, 더 빠르지만 해상도 낮음.
- 8x8 패치 -> 784 토큰, 더 세밀하지만 O(n^2) 어텐션 비용이 나빠짐.

더 큰 패치 = 더 적은 토큰 = 더 빠르지만 공간 디테일 감소. SwinV2는 계층적 윈도우에서 4x4 패치를 씁니다.

### ImageNet-1k에서 ViT를 학습하는 DeiT 레시피 (DeiT's recipe for training ViT on ImageNet-1k)

원본 ViT는 CNN을 이기려면 JFT-300M이 필요했습니다. DeiT(Touvron et al., 2020)는 네 가지 변경으로 ImageNet-1k만으로 ViT-B를 top-1 81.8%까지 학습했습니다:

1. 강한 증강: RandAugment, Mixup, CutMix, Random Erasing.
2. Stochastic depth(학습 중 블록 전체를 무작위로 드롭).
3. Repeated augmentation(배치당 같은 이미지를 3번 샘플링).
4. CNN 교사로부터의 증류(선택적, 정확도를 더 올림).

모든 현대 ViT 학습 레시피는 DeiT에서 내려옵니다.

### Swin vs ConvNeXt

- **Swin** (Liu et al., 2021) — 윈도우 기반 어텐션. 각 블록이 로컬 윈도우 안에서 attend하고; 교차 블록이 윈도우를 시프트해 윈도우 간 정보를 섞습니다. 어텐션 연산자는 유지하면서 CNN 같은 국소성 사전을 되돌립니다.
- **ConvNeXt** (Liu et al., 2022) — Swin의 아키텍처 선택(depthwise conv, LayerNorm, GELU, inverted bottleneck)에 맞춘 재설계 CNN. 격차가 "어텐션 vs 컨볼루션"이 아니라 "현대 학습 레시피 + 아키텍처"임을 보였습니다.

2026년에 ConvNeXt-V2와 Swin-V2는 둘 다 프로덕션급이며; 올바른 선택은 추론 스택(ConvNeXt가 엣지 컴파일에 더 나음)과 사전학습 코퍼스에 달립니다.

### MAE 사전학습 (MAE pretraining)

Masked Autoencoder(He et al., 2022): 패치의 75%를 무작위로 마스킹하고, 인코더가 보이는 25%만 처리하도록 학습하며, 작은 디코더가 인코더 출력에서 마스킹된 패치를 재구성하도록 학습합니다. 사전학습 후 디코더를 버리고 인코더를 파인튜닝합니다.

MAE는 ImageNet-1k만으로 ViT를 학습 가능하게 만들고, SOTA를 치며, 현재 기본 자기지도 레시피입니다.

```figure
batchnorm-inference
```

## 직접 만들기 (Build It)

### 1단계: 패치 임베딩 (Step 1: Patch embedding)

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

conv 하나, flatten 하나, transpose 하나. 이것이 전체 이미지-토큰 단계입니다.

### 2단계: 트랜스포머 블록 (Step 2: Transformer block)

Pre-LN, multi-head self-attention, GELU MLP, 잔차 연결.

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

`nn.MultiheadAttention`이 헤드 분할, 스케일된 내적, 출력 사영을 처리합니다. `batch_first=True`이므로 shape는 `(N, seq, dim)`입니다.

### 3단계: ViT (Step 3: The ViT)

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

약 2.8M 파라미터 — CPU에서 다룰 수 있는 작은 ViT입니다. 실제 ViT-B는 86M; 같은 클래스 정의에 `dim=768, depth=12, num_heads=12`.

### 4단계: 정상성 검사 — 단일 이미지 추론 (Step 4: Sanity check — single image inference)

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

오류 없이 돌아가야 합니다. 확률 합은 1입니다.

## 활용하기 (Use It)

`timm`은 ImageNet 사전학습 가중치가 있는 모든 ViT 변형을 제공합니다. 한 줄:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm`은 2026년 비전 트랜스포머의 프로덕션 기본값입니다. 같은 API로 ViT, DeiT, Swin, Swin-V2, ConvNeXt, ConvNeXt-V2, MaxViT, MViT, EfficientFormer 등 수십 개를 지원합니다.

멀티모달 작업(이미지 + 텍스트)에는 `transformers`가 CLIP, SigLIP, BLIP-2, LLaVA를 제공합니다. 그 모두의 이미지 인코더는 ViT 변형입니다.

## 결과물 배포 (Ship It)

이 레슨이 만드는 것:

- `outputs/prompt-vit-vs-cnn-picker.md` — 데이터셋 크기, 연산, 추론 스택에 따라 ViT, ConvNeXt, 또는 Swin을 고르는 프롬프트.
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — ViT의 패치 임베딩과 위치 임베딩 shape가 모델의 예상 시퀀스 길이와 맞는지 검증해 가장 흔한 포팅 버그를 잡는 스킬.

## 연습 문제 (Exercises)

1. **(Easy)** 위 작은 ViT의 순방향 패스에서 모든 중간 텐서의 shape를 출력하세요. 확인: 입력 `(N, 3, 64, 64)` -> 패치 `(N, 16, 192)` -> CLS 포함 `(N, 17, 192)` -> 분류기 입력 `(N, 192)` -> 출력 `(N, num_classes)`.
2. **(Medium)** Lesson 4의 합성 CIFAR 데이터셋에서 사전학습 `timm` ViT-S/16을 파인튜닝하세요. 같은 데이터에서 ResNet-18 파인튜닝과 비교하세요. 학습 시간과 최종 정확도를 보고하세요.
3. **(Hard)** 작은 ViT용 MAE 사전학습을 구현하세요: 패치의 75%를 마스킹하고, 인코더 + 작은 디코더가 마스킹된 패치를 재구성하도록 학습하세요. 사전학습 전후 합성 데이터에서 linear-probe 정확도를 평가하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| Patch embedding | "첫 conv" | kernel size = stride = patch size인 conv; 이미지를 토큰 임베딩 격자로 바꿈 |
| Class token | "[CLS]" | 토큰 시퀀스 앞에 붙인 학습 벡터; 최종 출력이 전역 이미지 표현 |
| Positional embedding | "학습된 pos" | 모든 토큰에 더해져 트랜스포머가 각 패치가 어디서 왔는지 알게 하는 학습 벡터 |
| Pre-LN | "서브레이어 앞 LayerNorm" | 안정적인 트랜스포머 변형: `LN(x + sublayer(x))` 대신 `x + sublayer(LN(x))` |
| Multi-head attention | "병렬 어텐션" | num_heads개의 독립 부분공간으로 나뉜 표준 트랜스포머 어텐션, 이후 연결 |
| ViT-B/16 | "Base, patch 16" | 표준 크기: dim=768, depth=12, heads=12, patch_size=16, image=224; ~86M params |
| DeiT | "데이터 효율적 ViT" | 강한 증강으로 ImageNet-1k만에서 학습한 ViT; 대규모 사전학습 데이터셋이 엄밀히 필수는 아님을 증명 |
| MAE | "마스크드 오토인코더" | 자기지도 사전학습: 패치 75% 마스킹, 재구성; 지배적 ViT 사전학습 레시피 |

## 더 읽을거리 (Further Reading)

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 논문
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — ImageNet-1k만으로 ViT를 학습하는 방법
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 사전학습
- [timm documentation](https://huggingface.co/docs/timm) — 프로덕션에서 쓸 모든 비전 트랜스포머 참고
