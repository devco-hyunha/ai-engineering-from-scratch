# 트랜스포머를 처음부터 구축하기 — 캡스톤 프로젝트

> 13강. 하나의 모델.捷径 없음.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 01~13강. 건너뛰지 마세요.
**시간:** 약 120분

## 문제점

모든 논문을 읽었습니다. 어텐션, 멀티헤드 분할, 위치 인코딩, 인코더 및 디코더 블록, BERT와 GPT 손실 함수, MoE, KV 캐시를 구현했습니다. 이제 실제 작업에서 이 모든 것이 함께 작동하도록 해 보세요.

캡스톤 프로젝트: 문자 단위 언어 모델링 작업에 작은 디코더 전용 트랜스포머를 엔드투엔드(end-to-end)로 학습합니다. 셰익스피어를 읽고, 새로운 셰익스피어를 생성합니다. 노트북에서 10분 미만에 학습할 수 있을 만큼 작습니다. 더 큰 데이터셋과 더 긴 학습을 적용하면 실제 LLM을 얻을 수 있을 만큼 정확합니다.

이것은 과정의 "nanoGPT"입니다. 독창적인 것은 아닙니다 — Karpathy의 2023년 nanoGPT 튜토리얼은 모든 학생이 적어도 한 번은 작성하는 참조 구현입니다. 우리는 그 형태를 취하고, 우리가 다루었던 내용으로 재구성합니다.

## 개념

![Transformer-from-scratch block diagram](../assets/capstone.svg)

주석이 달린 아키텍처:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── 04강 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 05강
│  MultiHeadAttention (causal)      │  ◀── 03강 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 05강
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── 07강
```

### 우리가 출시하는 것

- `GPTConfig` — 모든 하이퍼파라미터를 한 곳에서 설정합니다.
- `MultiHeadAttention` — 인과적(causal), 배치 처리, 선택적 Flash 스타일 경로(PyTorch의 `scaled_dot_product_attention`)를 포함합니다.
- `SwiGLUFFN` — 현대적인 FFN.
- `Block` — 사전 정규화(pre-norm), 잔차 연결(residual-wrapped) 어텐션 + FFN.
- `GPT` — 임베딩, 스택된 블록, LM 헤드, generate().
- AdamW, 코사인 학습률(LR), 기울기 클리핑을 포함한 학습 루프.
- 셰익스피어 텍스트에 대한 문자 단위 토크나이저.

### 우리가 출시하지 않는 것

- RoPE — 04강에서 개념적으로 구현했습니다. 여기서는 단순함을 위해 학습된 위치 임베딩을 사용합니다. 연습 문제에서는 RoPE로 교체하는 것을 요구합니다.
- 생성 중 KV 캐시 — 각 생성 단계에서 전체 접두어에 대해 어텐션을 다시 계산합니다. 느리지만 더 단순합니다. 연습 문제에서는 KV 캐시를 추가하는 것을 요구합니다.
- Flash Attention — PyTorch 2.0+는 입력이 일치하면 자동으로 디스패치(dispatch)합니다. 우리는 `F.scaled_dot_product_attention`를 사용합니다.
- MoE — 블록당 단일 FFN. 11강에서 MoE를 보았습니다.

### 목표 지표

Mac M2 노트북에서 `tinyshakespeare.txt`에 대해 4층, 4헤드, d_model=128 GPT를 2,000 스텝 동안 학습하면:

- 학습 손실이 약 4.2 (무작위)에서 약 1.5로 수렴하는 데 약 6분이 걸립니다.
- 샘플링된 출력은 셰익스피어 형태를 띱니다: 고어, 줄 바꿈, "ROMEO:"와 같은 고유명사가 나타납니다.
- 검증 손실(텍스트의 마지막 10%를 보존한 데이터)은 학습 손실과 밀접하게 일치합니다. 이 크기/예산에서는 과적합이 없습니다.

```figure
n5-block-stack
```

## 구현하기

이 강의는 PyTorch를 사용합니다. `torch`을 설치하세요 (CPU 빌드로 충분합니다). `code/main.py`을 참조하세요. 스크립트가 다음을 처리합니다:

- `tinyshakespeare.txt`이 없으면 다운로드하거나 (또는 로컬 사본을 읽음).
- 바이트 단위 문자 토크나이저.
- 90/10 비율의 학습/검증 분할.
- 지원되는 하드웨어에서 bf16 오토캐스트를 사용하는 학습 루프.
- 학습 완료 후 샘플링.

### 1단계: 데이터

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65개의 고유 문자. 매우 작은 어휘. 4바이트 vocab_size에 맞습니다. BPE 없음, 토크나이저 복잡성 없음.

### 2단계: 모델

`code/main.py`을 참조하세요. 블록은 05강의 교과서적인 내용입니다 — 프리 노름, RMSNorm, SwiGLU, 인과 MHA. 4/4/128의 매개변수 수: 약 800K.

### 3단계: 학습 루프

길이 256 토큰 윈도우의 무작위 배치를 가져옵니다. 순전파. 한 칸 이동 교차 엔트로피. 역전파. AdamW 스텝. 로깅. 반복.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 4단계: 샘플링

프롬프트가 주어지면, 반복적으로 순전파하고, top-p 로짓에서 샘플링하고, 추가하고, 계속합니다. 500 토큰 후 중지합니다.

### 5단계: 출력 읽기

2,000 스텝 후:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

셰익스피어는 아닙니다. 하지만 셰익스피어 형태입니다. 약 800K 매개변수와 노트북에서 6분이라는 조건에 대한 명확한 승리입니다.

## 사용하기

이 캡스톤은 참조 아키텍처입니다. 이를 실제 무언가로 출시하기 위한 세 가지 확장:

1. **토크나이저 교체.** BPE (예: `tiktoken.get_encoding("cl100k_base")`)를 사용하세요. 어휘 크기가 65에서 약 50,000으로 급증합니다. 모델 용량이 이를 보상하기 위해 확장되어야 합니다.
2. **더 큰 코퍼스로 학습.** `OpenWebText` 또는 `fineweb-edu` (HuggingFace)를 사용하세요. 단일 A100에서 10B 토큰은 125M 매개변수 GPT에 약 24시간이 걸립니다.
3. **RoPE + KV 캐시 + Flash Attention 추가.** 아래 연습 문제가 각각을 안내합니다.

이 코드는 유창한 영어를 생성하는 125M 매개변수(Parameter)의 GPT로 마무리됩니다. 프론티어 모델은 아니지만, Karpathy, EleutherAI, Allen Institute가 2026년에 연구 체크포인트(Checkpoint)를 훈련하는 데 사용하는 것과 동일한 코드 경로이며, 단지 규모가 더 클 뿐입니다.

## 출시하기

`outputs/skill-transformer-review.md`을 참조하세요. 이 스킬은 트랜스포머(Transformer)를 처음부터 구현한 코드가 이전 13강 전체에 걸쳐 정확성을 유지하는지 검토합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 훈련된 모델의 최종 단계 검증 손실(Loss)이 2.0 미만인지 확인하세요. `max_steps`을 2,000에서 5,000으로 변경하면 검증 손실이 계속 개선되는지 확인해 보세요.
2. **중간.** 학습된 위치 임베딩(Embedding)을 RoPE로 교체하세요. `MultiHeadAttention` 내부에서 Q와 K에 회전을 적용하세요. 훈련 후 검증 손실이 적어도 그만큼 낮아지는지 확인하세요.
3. **중간.** 샘플링 루프에 KV 캐시(KV Cache)를 구현하세요. 캐시 유무에 따라 500 토큰(Token)을 생성하세요. 노트북에서 벽시계 시간이 5–20배 개선되어야 합니다.
4. **어려움.** 모델에 다음 다음 토큰을 예측하는 두 번째 헤드(Head)를 추가하세요 (DeepSeek-V3의 MTP — 다중 토큰 예측(Multi-Token Prediction)). jointly 훈련하세요. 도움이 되는지 확인해 보세요.
5. **어려움.** 각 블록의 단일 FFN을 4-expert MoE (혼합 전문가)(MoE (Mixture of Experts))로 교체하세요. Router + top-2 라우팅을 적용하세요. 활성 매개변수(Parameter)가 일치할 때 검증 손실이 어떻게 변하는지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy의 튜토리얼 저장소" | 최소한의 디코더 전용 트랜스포머(Transformer) 훈련 코드, 약 300 LOC; 표준 참조 자료. |
| tinyshakespeare | "표준 장난감 코퍼스" | 약 1.1 MB의 텍스트; 2015년 이후 모든 문자 LM 튜토리얼이 이를 사용합니다. |
| Tied embeddings | "입력/출력 행렬 공유" | LM 헤드 가중치(Weight)는 토큰 임베딩(Embedding) 행렬의 전치(transpose)입니다; 매개변수(Parameter)를 절약하고 품질을 개선합니다. |
| bf16 autocast | "훈련 정밀도 트릭" | 순방향/역방향은 bf16으로 실행하고 옵티마이저(Optimizer) 상태는 fp32으로 유지합니다; 2021년 이후 표준입니다. |
| Gradient clipping | "스파이크 방지" | 전역 기울기(Gradient) 노름(norm)을 1.0으로 제한합니다; 훈련 폭발을 방지합니다. |
| Cosine LR schedule | "2020년 이후의 기본값" | 학습률(Learning Rate)이 선형으로 상승(워밍업(Warmup))한 후 코사인 형태로 피크의 10%까지 감소합니다. |
| MFU | "모델 FLOP 활용률" | 달성된 FLOPs / 이론적 피크; 2026년 기준 밀집(dense) 모델은 40%, MoE (혼합 전문가)(MoE (Mixture of Experts))는 30%가 강합니다. |
| 검증 손실 | "홀드아웃 손실" | 모델이 한 번도 본 적 없는 데이터에 대한 교차 엔트로피(Cross-Entropy); 과적합(Overfitting) 감지기. |

## 추가 읽기

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 주석이 달린 고전적인 구현입니다.
