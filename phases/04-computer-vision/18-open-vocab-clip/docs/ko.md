# 오픈 보케이블 비전 — CLIP

> 이미지 인코더와 텍스트 인코더를 함께 학습하여 일치하는 (이미지, 캡션) 쌍이 공유 공간의 같은 지점에 위치하도록 합니다. 이것이 핵심 트릭입니다.

**유형:** Build + Use
**언어:** Python
**선수 요건:** 4단계 14강 (ViT), 4단계 17강 (자기 지도 학습)
**시간:** 약 45분

## 학습 목표

- CLIP의 두 타워 아키텍처와 대조 학습 목적 함수를 설명할 수 있습니다
- 태스크별 학습 없이 사전 학습된 CLIP (또는 SigLIP)을 사용하여 제로샷 분류를 수행할 수 있습니다
- 클래스 프롬프트를 인코딩하고, 코사인 유사도를 계산하며, argmax를 취하는 방식으로 제로샷 분류를 처음부터 구현할 수 있습니다
- CLIP, SigLIP, OpenCLIP 및 LLaVA/LLaMA-vision 모델을 구분하고, 2026년 기준 각각의 용도를 파악할 수 있습니다

## 문제점

전통적인 분류기는 폐쇄 보케이블입니다. 1000개 클래스의 ImageNet 모델은 1000개 레이블만 예측할 수 있습니다. 새로운 카테고리를 추가하려면 레이블이 지정된 데이터와 재학습된 헤드가 필요합니다.

CLIP (Radford et al., OpenAI 2021)은 웹에서 수집한 4억 개의 (이미지, 캡션) 쌍으로 학습하면 추론 시 자연어로만 설명된 임의의 카테고리 집합으로 분류할 수 있는 모델을 생성한다는 것을 보여 주었습니다. 문장을 작성하여 새로운 클래스를 지정합니다.

이 능력, 즉 제로샷 전이(zero-shot transfer)는 모든 현대 비전 시스템이 CLIP 계열 체크포인트로 시작하는 이유입니다. 검출(Grounding DINO, OWL-ViT), 분할(CLIPSeg, SAM), 검색, 콘텐츠 Moderation, VLM 및 텍스트-이미지 생성은 모두 CLIP 스타일의 결합 임베딩을 기반으로 구축됩니다.

## 개념

### 두 타워

```mermaid
flowchart LR
    IMG["Image"] --> IENC["이미지 인코더<br/>(ViT-L/14)"] --> IEMB["이미지 임베딩<br/>(1024,)"]
    TXT["Caption"] --> TENC["텍스트 인코더<br/>(트랜스포머)"] --> TEMB["텍스트 임베딩<br/>(1024,)"]
    IEMB --> SIM["코사인 유사도"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```

두 인코더 모두 동일한 임베딩 차원(CLIP-B/32는 512, CLIP-L/14는 1024)으로 선형 투영을 수행합니다. L2 정규화를 수행하고 코사인 유사도를 계산합니다.

### 목적 함수

N개의 (이미지, 캡션) 쌍으로 이루어진 배치가 주어지면, NxN 유사성 행렬을 구축합니다. 두 인코더를 학습하여 대각선 요소(일치하는 쌍)는 높은 유사성을, 비대각선 요소(일치하지 않는 쌍)는 낮은 유사성을 갖도록 합니다.

```
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```

이미지에서 텍스트로, 그리고 텍스트에서 이미지로의 검색이 모두 작동해야 하므로 대칭적입니다. `tau` (온도)는 일반적으로 스칼라 매개변수로 학습되며, 초기값은 0.07입니다.

### SigLIP: 더 나은 손실 함수

SigLIP (Zhai et al., 2023)은 softmax를 쌍별 시그모이드로 대체했습니다:

```
loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise
```

쌍별 손실은 CLIP이 요구하는 배치 수준 정규화를 제거합니다. SigLIP은 작은 배치 크기에서 더 잘 학습되며, 동일한 데이터량에서 CLIP과 동등하거나 더 나은 성능을 보입니다.

### 제로샷 분류

학습된 CLIP이 주어지면:

1. 각 클래스에 대해 프롬프트를 작성합니다: "a photo of a {class}".
2. 모든 클래스 프롬프트를 텍스트 인코더로 인코딩합니다 -> `T`, 형태는 (C, d)입니다.
3. 테스트 이미지를 인코딩합니다 -> `I`, 형태는 (1, d)입니다.
4. 유사성 = `I @ T.T`, 형태는 (1, C)입니다.
5. Argmax -> 예측된 클래스.

프롬프트 엔지니어링이 중요합니다. OpenAI는 ImageNet을 위해 80개의 프롬프트 템플릿("a photo of a {}", "a blurry photo of a {}", "a sketch of a {}", ...)을 공개했습니다. 각 클래스에 대해 모든 템플릿의 임베딩을 평균 내면 top-1 정확도가 1-3% 추가됩니다.

### 2026년 CLIP 스타일 모델이 사용되는 곳

- **제로샷 분류** — 직접 사용.
- **이미지 검색** — 모든 이미지를 한 번 인코딩하고, 추론 시 쿼리를 임베딩합니다.
- **텍스트 조건 감지** — Grounding DINO, OWL-ViT는 감지기에 CLIP 텍스트 타워를 래핑합니다.
- **텍스트 조건 분할** — CLIPSeg; SAM은 CLIP를 통해 텍스트 프롬프트 입력을 사용합니다.
- **VLMs** — LLaVA, Qwen-VL, InternVL은 CLIP 계열 비전 인코더를 LLM에 연결합니다.
- **텍스트-이미지 생성** — Stable Diffusion, DALL-E 3는 CLIP 텍스트 임베딩에 조건을 적용합니다.

공유 임베딩 공간(Shared Embedding Space)을 확보하면, 모든 비전+언어 작업은 거리 계산이 됩니다.

```figure
clip-contrastive
```

## 구현하기

### 1단계: 작은 두 타워 모델

실제 CLIP은 ViT + 트랜스포머입니다. 이 강의에서는 학습 신호가 CPU에서 보이도록, 미리 추출된 특징 위에 작은 MLP를 타워로 사용합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```

두 개의 투영, 공유 차원 출력, 학습된 온도. 실제 CLIP API와 동일한 형태입니다.

### 2단계: 대조 손실

```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```

대칭적입니다. logit_scale이 높을수록 softmax가 날카로워져 더 확신에 차지만, 불안정성의 위험이 있습니다.

### 3단계: 제로샷 분류기

```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```

각 단계에 한 줄씩입니다. 이는 프로덕션 CLIP 체크포인트와 함께 사용되는 정확한 제로샷 절차입니다.

### 4단계: 정상성 확인

```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```

무작위로 초기화된 모델의 경우 손실은 `log(N) = log(8) = 2.08`에 가까워야 합니다. 이는 구조가 아직 학습되지 않았을 때의 대칭 교차 엔트로피 목표값입니다.

## 사용하기

OpenCLIP은 2026년 커뮤니티의 기본 선택입니다:

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```

SigLIP은 더 최신이며, 작은 규모에서 더 잘 학습되고, 새로운 작업에 선호됩니다: `google/siglip-base-patch16-224`. Hugging Face는 두 가지를 모두 제공합니다.

## 출시하기

이 강의는 다음을 생성합니다:

- `outputs/prompt-zero-shot-class-picker.md` — 클래스 목록과 도메인이 주어졌을 때 제로샷 CLIP용 클래스 템플릿을 설계하는 프롬프트입니다.
- `outputs/skill-image-text-retriever.md` — 임의의 CLIP 체크포인트로 이미지 임베딩 인덱스를 구축하고, 텍스트 기반 쿼리와 이미지 기반 쿼리를 지원하는 스킬입니다.

## 연습 문제

1. **(쉬움)** 사전 학습된 OpenCLIP ViT-B/32를 사용하여 CIFAR-10에서 80개 템플릿 프롬프트 세트로 제로샷 분류를 수행합니다. top-1 정확도를 보고하세요. 85-90% 정도여야 합니다.
2. **(중간)** 동일한 CIFAR-10 작업에서 단일 템플릿("a photo of a {}")과 80개 템플릿 평균 임베딩을 비교하세요. 격차를 정량화하고 템플릿이 도움이 되는 이유를 설명하세요.
3. **(어려움)** 제로샷 이미지 검색 인덱스를 구축하세요: CLIP으로 1,000개의 이미지를 임베딩하고, FAISS 인덱스를 구축한 후 자연어 설명으로 쿼리하세요. 직접 작성한 20개의 홀드아웃 쿼리에 대한 retrieval recall@5를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 투워(two-tower) | "듀얼 인코더" | 공유 차원 투영 헤드로 끝나는 분리된 이미지 및 텍스트 인코더 |
| 제로샷(zero-shot) | "작업 특화 학습 없음" | 추론 시 텍스트로만 설명된 클래스로 분류; 레이블을 전혀 건드리지 않음 |
| 온도 / logit_scale | "tau" | softmax 전에 유사성 행렬을 스케일링하는 학습된 스칼라 |
| 프롬프트 템플릿 | "A photo of a {}" | 클래스 이름을 둘러싼 자연어 래퍼; 여러 템플릿을 평균 내면 제로샷 정확도가 향상됩니다 |
| CLIP | "이미지+텍스트 모델" | 2021년 OpenAI 모델; 2026년 현재 이 분야의 표준 용어입니다 |
| SigLIP | "Sigmoid CLIP" | 소프트맥스를 쌍별 시그모이드로 교체; 작은 배치에서 더 잘 학습됩니다 |
| OpenCLIP | "오픈 재현" | LAION에서 커뮤니티가 학습한 CLIP 변형; 오픈소스 파이프라인의 기본 선택입니다 |
| VLM | "비전-언어 모델(Vision-Language Model)" | CLIP 계열 인코더와 LLM을 결합하여 이미지에 대한 질문에 답하도록 학습된 모델입니다 |

## 추가 읽기

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language-Image Pre-Training (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 커뮤니티 코드베이스
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — 논문, CLIP 계열 및 MAE 계열 모델에 대한 기능 벤치마크 포함
