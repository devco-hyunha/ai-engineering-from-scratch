# 시퀀스-투-시퀀스 모델 (Sequence-to-Sequence Models)

> 번역기처럼 행동하는 RNN 두 개. 이들이 부딪히는 병목이 어텐션(attention)이 존재하는 이유입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## 문제 (The Problem)

분류는 가변 길이 시퀀스를 단일 라벨에 매핑합니다. 번역은 가변 길이 시퀀스를 또 다른 가변 길이 시퀀스에 매핑합니다. 입력과 출력은 서로 다른 어휘(vocabulary)에, 때로는 서로 다른 언어에 살며, 길이가 같다고 보장되지 않습니다.

seq2seq 아키텍처(Sutskever, Vinyals, Le, 2014)는 의도적으로 단순한 처방으로 이 문제를 풀었습니다. RNN 두 개. 하나는 원문(source) 문장을 읽고 고정 크기 컨텍스트 벡터(context vector)를 만듭니다. 다른 하나는 그 벡터를 읽고 목표(target) 문장을 토큰 단위로 생성합니다. 레슨 08에서 쓴 것과 같은 코드, 다만 붙이는 방식이 다릅니다.

이걸 공부할 이유는 두 가지입니다. 첫째, 컨텍스트 벡터 병목은 NLP에서 교육적으로 가장 쓸모 있는 실패입니다. 어텐션과 트랜스포머가 잘하는 모든 것의 동기가 됩니다. 둘째, 학습 레시피(teacher forcing, scheduled sampling, 추론 시 beam search)는 LLM을 포함한 모든 현대 생성 시스템에 여전히 적용됩니다.

## 개념 (The Concept)

**인코더 (Encoder).** 원문 문장을 읽는 RNN. 최종 은닉 상태가 **컨텍스트 벡터**입니다 — 전체 입력의 고정 크기 요약. 원문만 빼고는 아무것도 잃지 않는다고 가정합니다.

**디코더 (Decoder).** 컨텍스트 벡터로 초기화된 또 다른 RNN. 매 단계에서 이전에 생성한 토큰을 입력으로 받아 목표 어휘에 대한 분포를 만듭니다. 샘플링하거나 argmax로 다음 토큰을 고릅니다. 다시 넣습니다. `<EOS>` 토큰이 나오거나 최대 길이에 닿을 때까지 반복합니다.

**학습:** 각 디코더 단계에서 교차 엔트로피(cross-entropy) 손실을 시퀀스에 걸쳐 합산합니다. 두 네트워크에 걸친 표준 BPTT(backprop through time)입니다.

**Teacher forcing.** 학습 중 단계 `t`에서 디코더 입력은 디코더 자신의 이전 예측이 아니라, 위치 `t-1`의 *정답(ground-truth)* 토큰입니다. 이렇게 하면 학습이 안정됩니다. 없으면 초반 실수가 연쇄되어 모델이 배우지 못합니다. 추론 시에는 모델 자신의 예측을 써야 하므로, 항상 학습/추론 분포 격차가 있습니다. 그 격차를 **노출 편향(exposure bias)**이라고 합니다.

**병목.** 인코더가 원문에 대해 배운 모든 것은 그 하나의 컨텍스트 벡터에 압축되어야 합니다. 긴 문장은 디테일을 잃습니다. 희귀 단어는 흐려집니다. 어순 재배치(chat noir vs. black cat)는 계산이 아니라 암기로 처리해야 합니다.

어텐션(레슨 10)은 디코더가 마지막 은닉 상태만이 아니라 *모든* 인코더 은닉 상태를 보게 해 이 문제를 고칩니다. 그게 전체 요지입니다.

```figure
lstm-gates
```

## 직접 만들기 (Build It)

### 1단계: 인코더

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`의 형태는 `[batch, seq_len, hidden_dim]` — 입력 위치마다 은닉 상태 하나. `hidden`의 형태는 `[1, batch, hidden_dim]` — 마지막 단계. 레슨 08은 "분류를 위해 outputs 위를 풀링하라"고 했습니다. 여기서는 마지막 은닉 상태를 컨텍스트 벡터로 쓰고, 단계별 outputs는 무시합니다.

### 2단계: 디코더

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

디코더는 한 단계씩 호출됩니다. 입력: 단일 토큰 배치와 현재 은닉 상태. 출력: 다음 토큰에 대한 어휘 로짓과 갱신된 은닉 상태.

### 3단계: teacher forcing이 있는 학습 루프

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

이름 붙일 만한 노브가 두 개입니다. `ignore_index=0`은 패딩 토큰에 대한 손실을 건너뜁니다. `teacher_forcing_ratio`는 각 단계에서 정답 토큰을 쓸지 모델 예측을 쓸지 확률입니다. 1.0(완전 teacher forcing)에서 시작해 학습이 진행되며 ~0.5까지 줄여 노출 편향 격차를 좁힙니다.

### 4단계: 추론 루프 (탐욕적)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

탐욕적 디코딩(greedy decoding)은 매 단계에서 가장 확률 높은 토큰을 고릅니다. 길을 잃을 수 있습니다. 한 토큰에 한 번 커밋하면 되돌릴 수 없습니다. **빔 서치(beam search)**는 상위 `k`개의 부분 시퀀스를 살려 두고, 마지막에 점수가 가장 높은 완성 시퀀스를 고릅니다. 빔 너비 3–5가 표준입니다.

### 5단계: 병목을 보여 주기

토이 복사 과제에서 모델을 학습합니다. 원문 `[a, b, c, d, e]`, 목표 `[a, b, c, d, e]`. 시퀀스 길이를 늘립니다. 정확도를 관찰합니다.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

단일 GRU 은닉 상태는 40토큰 입력을 무손실로 기억할 수 없습니다. 정보는 모든 인코더 단계에 있지만, 디코더는 마지막 상태만 봅니다. 어텐션이 이를 직접 고칩니다.

## 사용하기 (Use It)

PyTorch에는 `nn.Transformer`와 `nn.LSTM` 기반 seq2seq 템플릿이 있습니다. Hugging Face의 `transformers` 라이브러리는 수십억 토큰으로 학습된 완전한 인코더-디코더 모델(BART, T5, mBART, NLLB)을 제공합니다.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

현대 인코더-디코더는 RNN을 버리고 트랜스포머를 씁니다. 상위 수준 형태(인코더, 디코더, 토큰 단위 생성)는 2014 seq2seq 논문과 동일합니다. 각 블록 안의 메커니즘만 다릅니다.

### 여전히 RNN 기반 seq2seq를 쓸 때

새 프로젝트에서는 거의 없습니다. 예외는 구체적입니다.

- 입력을 토큰 하나씩, 제한된 메모리로 소비하는 스트리밍 번역.
- 트랜스포머 메모리 비용이 감당되지 않는 온디바이스 텍스트 생성.
- 교육. 인코더-디코더 병목을 이해하는 것이 트랜스포머가 이긴 이유를 이해하는 가장 빠른 길입니다.

### 노출 편향과 완화책

- **Scheduled sampling.** 학습 중 teacher forcing 비율을 줄여, 모델이 자신의 실수에서 회복하는 법을 배우게 합니다.
- **Minimum risk training.** 토큰 수준 교차 엔트로피 대신 문장 수준 BLEU 점수로 학습합니다. 실제로 원하는 것에 더 가깝습니다.
- **강화학습 파인튜닝.** 시퀀스 생성기에 메트릭으로 보상을 줍니다. 현대 LLM RLHF에서 쓰입니다.

세 가지 모두 트랜스포머 기반 생성에도 여전히 적용됩니다.

## 결과물 내보내기 (Ship It)

`outputs/prompt-seq2seq-design.md`로 저장하세요.

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 연습 문제 (Exercises)

1. **쉬움.** 토이 복사 과제를 구현하세요. 목표가 원문과 같은 입출력 쌍으로 GRU seq2seq를 학습합니다. 길이 5, 10, 20에서 정확도를 측정합니다. 병목을 재현하세요.
2. **중간.** 빔 너비 3으로 빔 서치 디코딩을 추가하세요. 작은 병렬 말뭉치에서 탐욕적 디코딩 대비 BLEU를 측정합니다. 빔 서치가 이기는 곳(보통 마지막 토큰)과 차이가 없는 곳을 문서화하세요.
3. **어려움.** 1만 쌍 패러프레이즈 데이터셋에서 `facebook/bart-base`를 파인튜닝하세요. 파인튜닝된 모델의 beam-4 출력을 베이스 모델과 홀드아웃 입력에서 비교합니다. BLEU를 보고하고 정성 예시 10개를 고르세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-------------------|-------------------|
| Encoder | 입력 RNN | 원문을 읽습니다. 단계별 은닉 상태와 최종 컨텍스트 벡터를 만듭니다. |
| Decoder | 출력 RNN | 컨텍스트 벡터로 초기화됩니다. 목표 토큰을 하나씩 생성합니다. |
| Context vector | 요약 | 최종 인코더 은닉 상태. 고정 크기. 어텐션이 푸는 병목입니다. |
| Teacher forcing | 정답 토큰 사용 | 학습 시 이전 정답 토큰을 넣습니다. 학습을 안정시킵니다. |
| Exposure bias | 학습/테스트 격차 | 정답 토큰으로 학습한 모델이 자신의 실수에서 회복하는 연습을 하지 못한 상태입니다. |
| Beam search | 더 나은 디코딩 | 탐욕적으로 커밋하는 대신 매 단계에서 상위 k개 부분 시퀀스를 살려 둡니다. |

## 더 읽어보기 (Further Reading)

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 원조 seq2seq 논문. 네 페이지.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRU와 인코더-디코더 프레이밍을 도입했습니다.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 어텐션 논문. 이 레슨 직후에 읽으세요.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 직접 만들 수 있는 seq2seq + 어텐션 코드.
