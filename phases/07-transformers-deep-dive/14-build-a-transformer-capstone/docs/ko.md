# 트랜스포머 처음부터 구축하기 — 캡스톤 (Build a Transformer from Scratch — The Capstone)

> 13개의 레슨. 하나의 모델. 지름길은 없습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## 문제 (The Problem)

여러분은 모든 논문을 읽었습니다. 어텐션(attention), 멀티 헤드 분할(multi-head splits), 위치 인코딩(positional encodings), 인코더 및 디코더 블록, BERT와 GPT 손실 함수(losses), MoE, KV 캐시(KV cache)를 모두 구현해 보았습니다. 이제 이 요소들을 실제 작업에서 함께 작동하게 만들어 보세요.

캡스톤 프로젝트(Capstone): 문자 단위 언어 모델링(character-level language modeling) 작업에서 소규모 디코더 전용 트랜스포머(decoder-only transformer)를 엔드 투 엔드(end-to-end)로 학습시킵니다. 이 모델은 셰익스피어(Shakespeare)를 읽고, 새로운 셰익스피어 문장을 생성합니다. 노트북에서 10분 이내에 학습할 수 있을 만큼 작지만, 더 큰 데이터셋과 더 긴 학습 시간을 적용하면 실제 언어 모델(LM)이 될 수 있을 만큼 충분히 정확합니다.

이것이 본 과정의 "nanoGPT"입니다. 독창적인 것은 아닙니다. Karpathy의 2023년 nanoGPT 튜토리얼은 모든 학생이 적어도 한 번은 작성하게 되는 참조 구현(reference implementation)입니다. 우리는 그 형태를 가져와 우리가 학습한 내용에 맞춰 재구성할 것입니다.

## 개념 (The Concept)

![Transformer-from-scratch block diagram](../assets/capstone.svg)

주석이 달린 아키텍처:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE 옵션)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (token embedding과 결합됨)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 우리가 구현하는 것 (What we ship)

- `GPTConfig` — 모든 하이퍼파라미터를 설정하는 단일 지점.
- `MultiHeadAttention` — 인과적(causal)이며 배치 처리가 가능하고, 선택적으로 Flash 스타일 경로(PyTorch의 `scaled_dot_product_attention`)를 지원합니다.
- `SwiGLUFFN` — 현대적인 FFN(Feed-Forward Network).
- `Block` — Pre-norm 방식이며, 잔차 연결(residual)로 감싸진 Attention + FFN 구조입니다.
- `GPT` — 임베딩(embeddings), 쌓인 블록들(stacked blocks), LM head, `generate()` 메서드를 포함합니다.
- AdamW, 코사인 학습률(cosine LR), 그래디언트 클리핑(gradient clipping)이 적용된 학습 루프.
- 셰익스피어(Shakespeare) 텍스트를 사용한 문자 단위(char-level) 토크나이저.

### 구현하지 않는 기능 (What we don't ship)

- RoPE — Lesson 04에서 개념적으로 구현됩니다. 여기서는 단순화를 위해 학습 가능한 위치 임베딩(learned positional embeddings)을 사용합니다. 연습 문제에서 RoPE로 교체해 보세요.
- 생성 중 KV 캐시(KV cache during generation) — 각 생성 단계에서 전체 접두사(prefix)에 대해 어텐션을 다시 계산합니다. 속도는 느리지만 구조는 더 단순합니다. 연습 문제에서 KV 캐시를 추가해 보세요.
- Flash Attention — 입력값이 일치하면 PyTorch 2.0+에서 자동으로 디스패치됩니다. 여기서는 `F.scaled_dot_product_attention`을 사용합니다.
- MoE — 블록당 단일 FFN을 사용합니다. MoE는 Lesson 11에서 다룹니다.

### 목표 지표 (Target metrics)

Mac M2 노트북 환경에서 `tinyshakespeare.txt`를 사용하여 2,000 스텝 동안 학습시킨 4개 레이어, 4개 헤드, `d_model=128` 설정의 GPT 결과:

- 학습 손실(Training loss)이 약 6분 만에 ~4.2(무작위 상태)에서 ~1.5로 수렴합니다.
- 샘플링된 출력은 셰익스피어의 문체와 유사한 형태를 보입니다: 고어(archaic words), 줄 바꿈, "ROMEO:"와 같은 고유 명사 등이 나타납니다.
- 검증 손실(Val loss, 텍스트의 마지막 10%를 별도로 분리)이 학습 손실을 밀접하게 따라갑니다. 이 정도 규모와 예산에서는 과적합(overfitting)이 발생하지 않습니다.

```figure
n5-block-stack
```

## 구현하기 (Build It)

이 레슨에서는 PyTorch를 사용합니다. `torch`를 설치하세요 (CPU 빌드도 괜찮습니다). `code/main.py`를 참조하세요. 이 스크립트는 다음 작업을 수행합니다:

- `tinyshakespeare.txt` 파일이 없는 경우 다운로드(또는 로컬 복사본 읽기).
- 바이트 수준 문자 토크나이저(Byte-level char tokenizer).
- 90/10 비율의 훈련/검증(Train/val) 데이터 분할.
- 지원되는 하드웨어에서 bf16 autocast를 사용하는 훈련 루프.
- 훈련 완료 후 샘플링.

### 1단계: 데이터 (data)

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65개의 고유 문자가 있습니다. 매우 작은 어휘 집합(vocabulary)입니다. 4바이트 `vocab_size`에 적합합니다. BPE나 토크나이저 관련 복잡한 문제도 없습니다.

### 2단계: 모델 (model)

`code/main.py`를 참조하세요. 이 블록은 Lesson 05의 핵심 내용을 담고 있습니다 — pre-norm, `RMSNorm`, `SwiGLU`, causal MHA가 포함됩니다. 4/4/128 설정 시 파라미터 수는 약 800K입니다.

### 3단계: 학습 루프 (training loop)

길이가 256인 토큰 윈도우(token windows)로 구성된 무작위 배치를 가져옵니다. 순전파(Forward)를 수행합니다. 1칸씩 밀린(Shift-by-one) 교차 엔트로피(cross-entropy)를 계산합니다. 역전파(Backward)를 수행합니다. AdamW 스텝을 진행합니다. 로그를 기록합니다. 이 과정을 반복합니다.

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

### 4단계: 샘플링 (sample)

프롬프트가 주어지면, 반복적으로 순전파(forward)를 수행하고, top-p 로짓(logits)에서 샘플링하여 결과물을 추가하며 계속 진행합니다. 500개 토큰이 생성되면 중단합니다.

### 5단계: 출력 결과 읽기 (read the output)

2,000단계 이후:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

셰익스피어는 아닙니다. 하지만 셰익스피어의 형태를 띠고 있습니다. 약 80만(800K) 개의 파라미터와 노트북에서의 6분이라는 시간으로 거둔 명백한 승리입니다.

## 활용하기 (Use It)

이 캡스톤 프로젝트는 레퍼런스 아키텍처(reference architecture)입니다. 이를 실제 서비스 수준으로 끌어올리기 위한 세 가지 확장 방법은 다음과 같습니다:

1. **토크나이저 교체 (Swap the tokenizer).** BPE(예: `tiktoken.get_encoding("cl100k_base")`)를 사용해 보세요. 어휘 사전 크기(Vocab size)가 65에서 약 50,000으로 급증합니다. 이를 보완하기 위해 모델 용량(Model capacity)을 확장해야 합니다.
2. **더 큰 코퍼스로 학습 (Train on a bigger corpus).** `OpenWebText` 또는 `fineweb-edu`(HuggingFace)를 사용해 보세요. 단일 A100에서 125M 파라미터 GPT를 10B 토큰으로 학습시키는 데 약 24시간이 소요됩니다.
3. **RoPE + KV 캐시 + Flash Attention 추가 (Add RoPE + KV cache + Flash Attention).** 아래의 연습 문제들이 각 단계를 안내해 드립니다.

이 과정을 마치면 유창한 영어를 생성하는 125M 파라미터 GPT가 완성됩니다. 최첨단(frontier) 모델은 아니지만, 규모만 키운다면 Karpathy, EleutherAI, 그리고 Allen Institute가 2026년에 연구용 체크포인트를 학습할 때 사용하는 것과 동일한 코드 경로를 사용하게 됩니다.

## Ship It (실행하기)

`outputs/skill-transformer-review.md`를 참조하세요. 이 스킬은 이전 13개 레슨 전체에 걸쳐 트랜스포머를 처음부터 구현한(transformer-from-scratch) 코드가 정확한지 검토합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 학습된 모델의 최종 단계 검증 손실(validation loss)이 2.0 미만인지 확인합니다. `max_steps`를 2,000에서 5,000으로 변경했을 때, 검증 손실이 계속 개선되나요?
2. **중간 (Medium).** 학습된 위치 임베딩(positional embeddings)을 RoPE로 교체해 보세요. `MultiHeadAttention` 내부의 `Q`와 `K`에 회전(rotation)을 적용합니다. 학습을 진행하고 검증 손실이 최소한 이전과 비슷하거나 더 낮게 나오는지 확인합니다.
3. **중간 (Medium).** 샘플링 루프에 KV 캐시(KV cache)를 구현해 보세요. 캐시를 사용했을 때와 사용하지 않았을 때 각각 500개의 토큰을 생성합니다. 노트북 환경에서 실제 실행 시간(Wall-clock time)이 5~20배 정도 개선되어야 합니다.
4. **어려움 (Hard).** 다음 다음 토큰(next-plus-one token)을 예측하는 두 번째 헤드를 모델에 추가해 보세요 (DeepSeek-V3의 MTP — Multi-Token Prediction 방식). 공동 학습(joint training)을 진행합니다. 성능 향상에 도움이 되나요?
5. **어려움 (Hard).** 블록당 하나의 FFN을 4개의 전문가(expert)를 가진 MoE(Mixture-of-Experts)로 교체해 보세요. Router와 top-2 라우팅을 적용합니다. 활성 파라미터(active parameters) 수를 동일하게 맞췄을 때 검증 손실이 어떻게 변하는지 확인해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy의 튜토리얼 저장소" | 약 300줄의 코드(LOC)로 구성된 최소한의 디코더 전용(decoder-only) 트랜스포머 학습 코드; 표준 참조 모델. |
| tinyshakespeare | "표준 토이 코퍼스" | 약 1.1 MB의 텍스트; 2015년 이후 모든 character-LM 튜토리얼에서 사용됨. |
| Tied embeddings | "입력/출력 행렬 공유" | LM head 가중치 = 토큰 임베딩 행렬의 전치(transpose); 파라미터를 절약하고 품질을 향상시킴. |
| bf16 autocast | "학습 정밀도 트릭" | 순전파(forward)/역전파(back)는 `bf16`으로 실행하고, 옵티마이저 상태는 `fp32`로 유지; 2021년 이후 표준 방식. |
| Gradient clipping | "스파이크 방지" | 전역 그래디언트 노름(global grad norm)을 1.0으로 제한; 학습 폭주(blowups)를 방지함. |
| Cosine LR schedule | "2020년 이후의 기본값" | 학습률(LR)이 선형적으로 상승(warmup)한 후, 코사인 함수 형태로 피크치의 10%까지 감소함. |
| MFU | "Model FLOP Utilization" | 달성한 FLOPs / 이론적 최대치; 2026년 기준으로 Dense 모델 40%, MoE 모델 30%는 강력한 성능임. |
| Val loss | "홀드아웃 손실" | 모델이 본 적 없는 데이터에 대한 교차 엔트로피(cross-entropy); 과적합(overfit) 탐지기. |

## 추가 학습 자료 (Further Reading)

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 고전적인 주석 기반 구현체입니다.
