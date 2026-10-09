# 시퀀스-투-시퀀스 모델

> 번역기를 흉내 내는 두 개의 RNN. 이들이 직면한 병목 현상은 어텐션이 존재하는 이유입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 08강 (텍스트를 위한 CNN + RNN), 3단계 · 11강 (PyTorch 입문)
**시간:** 약 75분

## 문제점

분류는 가변 길이 시퀀스를 단일 레이블로 매핑합니다. 번역은 가변 길이 시퀀스를 다른 가변 길이 시퀀스로 매핑합니다. 입력과 출력은 서로 다른 어휘(Vocabulary)에 속하며, 서로 다른 언어일 수도 있고, 길이 일치 보장이 없습니다.

seq2seq 아키텍처(Sutskever, Vinyals, Le, 2014)는 의도적으로 단순한 레시피로 이 문제를 해결했습니다. 두 개의 RNN을 사용합니다. 하나는 원문 문장을 읽고 고정 크기의 컨텍스트 벡터를 생성합니다. 다른 하나는 그 벡터를 읽고 타겟 문장을 토큰(Token) 단위로 생성합니다. 08강에서 작성한 동일한 코드를 다른 방식으로 연결한 것입니다.

이 내용은 두 가지 이유로 학습할 가치가 있습니다. 첫째, 컨텍스트 벡터 병목 현상은 NLP에서 가장 교육적으로 유용한 실패 사례입니다. 이는 어텐션(Attention)과 트랜스포머(Transformer)가 잘하는 모든 것을 동기 부여합니다. 둘째, 학습 레시피(교사 강제, 스케줄링 샘플링, 추론 시 빔 서치)는 LLM을 포함한 모든 현대적 생성 시스템에 여전히 적용됩니다.

## 개념

**인코더(Encoder).** 원문 문장을 읽는 RNN입니다. 최종 은닉 상태는 **컨텍스트 벡터**입니다. 이는 전체 입력의 고정 크기 요약입니다. 원문만 제외하고 아무것도 잃지 않는다고 가정합니다.

**디코더(Decoder).** 컨텍스트 벡터에서 초기화된 다른 RNN입니다. 각 단계에서 이전에 생성된 토큰을 입력으로 받아 타겟 어휘(Vocabulary)에 대한 분포를 생성합니다. 샘플링(Sampling) 또는 argmax로 다음 토큰을 선택합니다. 이를 다시 입력으로 전달합니다. `<EOS>` 토큰이 생성되거나 최대 길이에 도달할 때까지 반복합니다.

**학습:** 각 디코더 단계에서 교차 엔트로피(Cross-Entropy) 손실(Loss Function)을 시퀀스 전체에 대해 합산합니다. 두 네트워크 모두에 대해 표준 시간 기반 역전파(Backpropagation)를 수행합니다.

**교사 강제(Teacher forcing).** 학습 중, 디코더의 입력은 `t` 단계에서 `t-1` 위치의 *정답(ground-truth)* 토큰이며, 디코더가 이전에 예측한 토큰이 아닙니다. 이는 학습을 안정화합니다. 이 기법을 사용하지 않으면 초기 오류가 연쇄적으로 발생하여 모델이 학습되지 않습니다. 추론 시에는 모델의 자체 예측을 사용해야 하므로, 학습과 추론 간의 분포 차이가 항상 존재합니다. 이 차이를 **노출 편향(exposure bias)**이라고 부릅니다.

**병목 현상.** 인코더가 소스 문장에서 학습한 모든 정보는 하나의 컨텍스트 벡터에 압축되어야 합니다. 긴 문장은 세부 정보가 손실되고, 희소한 단어는 흐려집니다. 어순 변경(chat noir vs. black cat)은 계산이 아니라 암기해야 합니다.

어텐션(10강)은 디코더가 마지막 상태뿐만 아니라 *모든* 인코더 은닉 상태를 참조할 수 있게 함으로써 이 문제를 해결합니다. 이것이 핵심 제안입니다.

```figure
lstm-gates
```

## 구현하기

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

`outputs`는 `[batch, seq_len, hidden_dim]` 모양을 가지며, 입력 위치마다 하나의 은닉 상태를 포함합니다. `hidden`는 `[1, batch, hidden_dim]` 모양을 가지며, 최종 단계입니다. 08강에서는 "분류를 위해 출력들을 풀링(pooling)한다"고 설명했습니다. 여기서는 마지막 은닉 상태를 컨텍스트 벡터로 유지하고, 단계별 출력은 무시합니다.

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

디코더는 한 단계씩 호출됩니다. 입력: 단일 토큰 배치와 현재 은닉 상태. 출력: 다음 토큰의 어휘 로짓(logits)과 업데이트된 은닉 상태.

### 3단계: 교사 강제(Teacher forcing)를 포함한 학습 루프

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

언급할 가치가 있는 두 가지 매개변수가 있습니다. `ignore_index=0`는 패딩 토큰에 대한 손실을 생략합니다. `teacher_forcing_ratio`는 각 단계에서 모델의 예측 대신 실제 토큰을 사용할 확률입니다. 1.0(완전한 교사 강제)에서 시작하여 학습 동안 약 0.5로 점진적으로 낮추어 노출 편향 격차를 줄입니다.

### 4단계: 추론 루프 (탐욕적(greedy))

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

탐욕적 디코딩은 매 단계에서 확률이 가장 높은 토큰을 선택합니다. 이는 경로에서 벗어날 수 있습니다: 한 번 토큰을 선택하면 되돌릴 수 없습니다. **빔 검색(Beam search)**은 상위 `k`개의 부분 시퀀스를 유지하며, 최종적으로 점수가 가장 높은 완전한 시퀀스를 선택합니다. 빔 폭은 3-5가 표준입니다.

### 5단계: 병목 현상 시연

장난감 복사 작업(toy copy task)으로 모델을 학습하세요: 소스 `[a, b, c, d, e]`, 타겟 `[a, b, c, d, e]`. 시퀀스 길이를 늘려가며 정확도를 관찰해 보세요.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

단일 GRU 은닉 상태는 40개 토큰 입력을 손실 없이 기억할 수 없습니다. 정보는 모든 인코더 단계에 존재하지만, 디코더는 마지막 상태만 봅니다. 어텐션은 이 문제를 직접적으로 해결합니다.

## 사용하기

PyTorch는 `nn.Transformer` 및 `nn.LSTM` 기반 seq2seq 템플릿을 제공합니다. Hugging Face의 `transformers` 라이브러리는 수십억 개 토큰으로 학습된 완전한 인코더-디코더 모델(BART, T5, mBART, NLLB)을 제공합니다.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

현대적인 인코더-디코더는 RNN을 버리고 트랜스포머를 채택했습니다. 고수준 구조(인코더, 디코더, 토큰 단위 생성)는 2014년 seq2seq 논문과 동일합니다. 각 블록 내부의 메커니즘은 다릅니다.

### RNN 기반 seq2seq를 여전히 고려해야 하는 경우

새로운 프로젝트에서는 거의 고려하지 않습니다. 특정 예외는 다음과 같습니다:

- 메모리 사용량이 제한된 상태에서 입력을 토큰 단위로 소비하는 스트리밍 번역.
- 트랜스포머의 메모리 비용이 감당하기 어려운 기기 내(on-device) 텍스트 생성.
- 교육적 목적. 인코더-디코더 병목 현상을 이해하는 것은 트랜스포머가 승리한 이유를 이해하는 가장 빠른 길입니다.

### 노출 편향(exposure bias) 및 그 완화 방법

- **스케줄링 샘플링(Scheduled Sampling).** 학습 중 교사 강제(teacher forcing) 비율을 점진적으로 낮추어 모델이 자신의 실수로부터 회복하는 법을 학습하도록 합니다.
- **최소 위험 학습(Minimum Risk Training).** 토큰 단위 교차 엔트로피 대신 문장 단위 BLEU 점수로 학습합니다. 실제로 원하는 것에 더 가깝습니다.
- **강화 학습 미세 조정.** 시퀀스 생성기에 지표(metric)를 보상(reward)으로 제공합니다. 현대 LLM의 RLHF에서 사용됩니다.

이 세 가지 방법은 트랜스포머 기반 생성에도 여전히 적용됩니다.

## 출시하기

`outputs/prompt-seq2seq-design.md`로 저장하세요:

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

## 연습 문제

1. **쉬움.** 장난감 복사(copy) 작업을 구현하세요. 타깃이 소스와 동일한 입력-출력 쌍으로 GRU seq2seq를 학습하세요. 길이 5, 10, 20에서 정확도를 측정하세요. 병목 현상을 재현해 보세요.
2. **중간.** 빔 폭(beam width)이 3인 빔 검색 디코딩을 추가하세요. 작은 병렬 코퍼스로 greedy 디코딩과 비교하여 BLEU를 측정하세요. 빔 검색이 이득을 보는 위치(보통 마지막 토큰)와 차이가 없는 위치를 문서화하세요.
3. **어려움.** `facebook/bart-base`를 1만 쌍의 패러프레이즈(paraphrase) 데이터셋으로 미세 조정하세요. 미세 조정된 모델의 beam-4 출력과 기본 모델의 출력을 홀드아웃(held-out) 입력에 대해 비교하세요. BLEU를 보고하고 10개의 정성적 예시를 선택하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Encoder | 입력 RNN | 소스를 읽습니다. 단계별 은닉 상태와 최종 컨텍스트 벡터를 생성합니다. |
| Decoder | 출력 RNN | 컨텍스트 벡터에서 초기화됩니다. 타겟 토큰을 하나씩 생성합니다. |
| Context vector | 요약 | 최종 인코더 은닉 상태입니다. 고정 크기입니다. 어텐션이 해결하는 병목 현상입니다. |
| Teacher forcing | 실제 토큰 사용 | 학습 시 실제(previous) 토큰을 입력합니다. 학습을 안정화합니다. |
| Exposure bias | 학습/테스트 간격 | 실제 토큰으로 학습된 모델은 자신의 실수로부터 회복하는 연습을 하지 못했습니다. |
| Beam search | 더 나은 디코딩 | 각 단계에서 greedily 확정하는 대신 상위 k개의 부분 시퀀스를 유지합니다. |

## 추가 읽기

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 원본 seq2seq 논문입니다. 4페이지입니다.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRU와 인코더-디코더 프레임워크를 도입했습니다.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 어텐션 논문입니다. 이 강의를 바로 읽은 후 읽어 보세요.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 구축 가능한 seq2seq + 어텐션 코드입니다.
