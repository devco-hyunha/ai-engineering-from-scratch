# 텍스트를 위한 CNN과 RNN (CNNs and RNNs for Text)

> 합성곱(convolution)은 n-gram을 학습합니다. 순환(recurrence)은 기억합니다. 둘 다 어텐션(attention)에 밀렸습니다. 그래도 제약이 있는 하드웨어에서는 여전히 중요합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 문제 (The Problem)

TF-IDF와 Word2Vec은 어순을 무시하는 평평한 벡터를 만들었습니다. 그 위에 올린 분류기는 `dog bites man`과 `man bites dog`을 구분하지 못했습니다. 어순이 신호를 담는 경우도 있습니다.

트랜스포머가 오기 전에 그 공백을 메운 아키텍처 계열이 두 가지입니다.

**텍스트용 합성곱 네트워크 (TextCNN).** 단어 임베딩 시퀀스 위에 1D 합성곱을 적용합니다. 폭 3인 필터는 학습 가능한 트라이그램(trigram) 검출기입니다. 단어 세 개를 걸쳐 점수를 냅니다. 폭을 (2, 3, 4, 5)처럼 여러 개 쌓아 다중 스케일 패턴을 잡습니다. Max-pool로 고정 크기 표현을 만듭니다. 평평하고, 병렬이며, 빠릅니다.

**순환 네트워크 (RNN, LSTM, GRU).** 토큰을 하나씩 처리하며, 정보를 앞으로 나르는 은닉 상태(hidden state)를 유지합니다. 순차적이고, 기억을 가지며, 입력 길이에 유연합니다. 2014년부터 2017년까지 시퀀스 모델링을 지배했고, 그다음 어텐션이 등장했습니다.

이 레슨은 둘을 직접 만든 뒤, 어텐션을 촉발한 실패를 이름 붙입니다.

## 개념 (The Concept)

**TextCNN** (Kim, 2014). 토큰을 임베딩합니다. 폭 `k`인 1D 합성곱이 연속된 `k`-gram 임베딩 위를 필터로 훑으며 특성 맵(feature map)을 만듭니다. 그 맵에 대한 전역 max-pooling이 가장 강한 활성화를 고릅니다. 여러 필터 폭의 max-pool 출력을 이어 붙입니다. 분류기 헤드에 넣습니다.

왜 동작하는가. 필터는 학습 가능한 n-gram입니다. Max-pooling은 위치에 불변이므로, 리뷰 앞이든 중간이든 "not good"이 같은 특성을 켭니다. 필터 폭 세 종류에 각각 필터 100개면 학습된 n-gram 검출기가 300개입니다. 학습은 병렬이며 순차 의존이 없습니다.

**RNN.** 각 시간 단계 `t`에서 은닉 상태는 `h_t = f(W * x_t + U * h_{t-1} + b)`입니다. `W`, `U`, `b`를 시간에 걸쳐 공유합니다. 시각 `T`의 은닉 상태는 전체 접두사(prefix)의 요약입니다. 분류에서는 `h_1 ... h_T`에 걸쳐 풀링합니다(max, mean, 또는 last).

일반 RNN은 기울기 소실(vanishing gradient)에 시달립니다. **LSTM**은 무엇을 잊을지, 무엇을 저장할지, 무엇을 출력할지 정하는 게이트를 더해 긴 시퀀스에서도 기울기를 안정시킵니다. **GRU**는 LSTM을 게이트 두 개로 단순화합니다. 성능은 비슷하고 파라미터가 더 적습니다.

**양방향 RNN (Bidirectional RNNs)**은 한 RNN을 앞으로, 다른 하나를 뒤로 돌리고 은닉 상태를 이어 붙입니다. 모든 토큰 표현이 왼쪽과 오른쪽 문맥을 모두 봅니다. 태깅 과제에 필수입니다.

```figure
rnn-unroll
```

## 직접 만들기 (Build It)

### 1단계: PyTorch로 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`는 `[batch, seq_len, embed_dim]`을 `[batch, embed_dim, seq_len]`으로 바꿉니다. `nn.Conv1d`가 가운데 축을 채널로 다루기 때문입니다. 풀링된 출력은 입력 길이와 무관하게 고정 크기입니다.

### 2단계: LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

시퀀스에 대해 max-pool을 쓰고, last-state pool은 쓰지 않습니다. 분류에서는 max-pooling이 보통 마지막 은닉 상태를 쓰는 것보다 낫습니다. 긴 시퀀스의 끝 정보가 마지막 상태를 지배하는 경향이 있기 때문입니다.

### 3단계: 기울기 소실 데모 (직관)

게이팅이 없는 일반 RNN은 장거리 의존을 학습하지 못합니다. 장난감 과제를 생각해 보세요. 시퀀스 어딘가에 토큰 `A`가 나타났는지 예측합니다. `A`가 위치 1에 있고 시퀀스가 100토큰이면, 손실에서 나온 기울기는 순환 가중치의 곱셈 99번을 거슬러 올라가야 합니다. 가중치가 1보다 작으면 기울기가 소실됩니다. 1보다 크면 폭발합니다.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM은 **셀 상태(cell state)**로 이를 고칩니다. 셀 상태는 네트워크를 가로지르며 주로 덧셈 상호작용만 합니다(forget gate가 곱으로 스케일하지만, 기울기는 여전히 "고속도로"를 따라 흐릅니다). GRU도 파라미터가 더 적게 비슷한 일을 합니다. 둘 다 100단계 이상 시퀀스에서도 안정적인 학습을 줍니다.

### 4단계: 왜 그래도 충분하지 않았는가

LSTM이 있어도 세 가지 문제가 남았습니다.

1. **순차 병목(Sequential bottleneck).** 길이 1000 시퀀스에서 RNN을 학습하려면 순방향/역방향 단계가 1000번 직렬로 필요합니다. 시간에 걸쳐 병렬화할 수 없습니다.
2. **인코더-디코더의 고정 크기 문맥 벡터.** 디코더는 인코더의 최종 은닉 상태만 봅니다. 전체 입력이 압축된 것입니다. 긴 입력은 세부 정보가 사라집니다. 레슨 09가 이를 직접 다룹니다.
3. **먼 의존 정확도의 천장.** LSTM은 일반 RNN보다 낫지만, 200단계 이상에 걸쳐 특정 정보를 전달하는 데는 여전히 고전합니다.

어텐션이 세 가지를 모두 풀었습니다. 트랜스포머는 순환을 통째로 버렸습니다. 레슨 10이 전환점입니다.

## 활용하기 (Use It)

PyTorch의 `nn.LSTM`, `nn.GRU`, `nn.Conv1d`는 프로덕션에 쓸 수 있습니다. 학습 코드는 표준입니다.

Hugging Face는 입력층에 꽂을 수 있는 사전학습 임베딩을 제공합니다.

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

제약에 맞을 때 쓰는 체크리스트입니다.

- **엣지 / 온디바이스 추론.** GloVe 임베딩을 쓰는 TextCNN은 트랜스포머보다 10–100배 작습니다. 배포 대상이 휴대폰이면 이 스택입니다.
- **스트리밍 / 온라인 분류.** RNN은 토큰을 하나씩 처리합니다. 트랜스포머는 전체 시퀀스가 필요합니다. 실시간으로 들어오는 텍스트에서는 LSTM이 여전히 이깁니다.
- **베이스라인용 작은 모델.** 새 과제에서 빠른 반복. CPU에서 TextCNN을 5분 안에 학습합니다.
- **데이터가 제한된 시퀀스 라벨링.** BiLSTM-CRF(레슨 06)는 라벨된 문장 1k–10k에서도 여전히 프로덕션급 NER 아키텍처입니다.

그 외에는 전부 트랜스포머로 갑니다.

## 결과물 내보내기 (Ship It)

`outputs/prompt-text-encoder-picker.md`로 저장하세요.

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 연습 문제 (Exercises)

1. **쉬움.** 3클래스 장난감 데이터셋(직접 만든 데이터)에서 TextCNN을 학습하세요. 필터 폭 (2, 3, 4)가 단일 폭 (3)보다 평균 F1에서 나은지 확인하세요.
2. **중간.** LSTM 분류기에 max-pool, mean-pool, last-state pooling을 구현하세요. 작은 데이터셋에서 비교하고, 어떤 풀링이 이기고 왜인지 가설을 적어 두세요.
3. **어려움.** BiLSTM-CRF NER 태거를 만드세요(레슨 06과 이 레슨을 결합). CoNLL-2003에서 학습하세요. 레슨 06의 CRF만 쓰는 베이스라인, BERT fine-tune과 비교하세요. 학습 시간, 메모리, F1을 보고하세요.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| TextCNN | 텍스트용 CNN | 단어 임베딩 위 1D 합성곱 스택 + 전역 max-pool. Kim (2014). |
| RNN | 순환 네트워크 | 각 시간 단계에서 은닉 상태 갱신: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | 게이트 RNN | 입력 / 망각 / 출력 게이트 + 셀 상태. 긴 시퀀스에서도 안정적으로 학습. |
| GRU | 더 단순한 LSTM | 게이트가 셋이 아니라 둘. 정확도는 비슷하고 파라미터가 더 적음. |
| Bidirectional | 양방향 | 순방향 + 역방향 RNN을 이어 붙임. 모든 토큰이 문맥의 양쪽을 봄. |
| Vanishing gradient | 학습 신호가 죽음 | 일반 RNN에서 <1 가중치의 반복 곱셈으로 초기 단계 기울기가 사실상 0이 됨. |

## 더 읽어보기 (Further Reading)

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 논문. 여덟 쪽. 읽기 쉽습니다.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 논문. 의외로 명료합니다.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM을 모두에게 이해시킨 다이어그램입니다.
