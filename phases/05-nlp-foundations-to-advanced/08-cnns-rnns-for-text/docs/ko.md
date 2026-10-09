# 텍스트를 위한 CNN과 RNN

> 합성곱은 n-gram을 학습합니다. 재귀 구조는 기억합니다. 둘 다 어텐션에 의해 대체되었지만, 제약된 하드웨어에서는 여전히 중요합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 3단계 · 11강 (PyTorch 입문), 5단계 · 03강 (단어 임베딩), 4단계 · 02강 (처음부터 합성곱 구현)
**시간:** 약 75분

## 문제점

TF-IDF와 Word2Vec은 단어 순서를 무시하는 평면 벡터를 생성했습니다. 이를 기반으로 만든 분류기는 `dog bites man`과 `man bites dog`을 구분할 수 없었습니다. 단어 순서는 때때로 신호를 담고 있습니다.

트랜스포머가 등장하기 전, 두 가지 아키텍처 계열이 그 공백을 채웠습니다.

**텍스트용 합성곱 신경망 (TextCNN).** 단어 임베딩 시퀀스에 1D 합성곱을 적용합니다. 너비 3인 필터는 학습 가능한 trigram 탐지기가 됩니다. 세 단어를 포함하여 점수를 출력합니다. 다양한 너비(2, 3, 4, 5)를 쌓아 다중 스케일 패턴을 탐지합니다. 최대 풀링을 통해 고정 크기의 표현으로 변환합니다. 평면적이고 병렬적이며 빠릅니다.

**재귀 신경망 (RNN, LSTM, GRU).** 토큰을 하나씩 처리하며 정보를 앞으로 전달하는 은닉 상태를 유지합니다. 순차적이며, 기억을 담고 있으며, 입력 길이가 유연합니다. 2014년부터 2017년까지 시퀀스 모델링을 지배했으나, 이후 어텐션이 등장했습니다.

이 강의에서는 두 가지를 모두 구축한 후, 어텐션을 동기 부여한 실패 사례를 명명합니다.

## 개념

**TextCNN** (Kim, 2014). 토큰이 임베딩됩니다. 너비 `k`인 1D 합성곱이 임베딩의 연속 `k`-gram 위에 필터를 슬라이딩하여 피처 맵을 생성합니다. 해당 맵에 대한 전역 최대 풀링이 가장 강한 활성화를 선택합니다. 여러 필터 너비에서 최대 풀링된 출력을 연결합니다. 분류기 헤드로 전달합니다.

작동 원리. 필터는 학습 가능한 n-gram입니다. 최대 풀링은 위치 불변이므로, 리뷰의 시작이나 중간에서 "not good"가 동일한 피처를 발화합니다. 세 가지 필터 너비에 각각 100개 필터를 사용하면 300개의 학습된 n-gram 탐지기가 됩니다. 학습은 병렬로 진행되며, 순차적 의존성이 없습니다.

**RNN.** 각 시간 단계 `t`에서, 은닉 상태는 `h_t = f(W * x_t + U * h_{t-1} + b)`입니다. `W`, `U`, `b`는 시간 전체에 걸쳐 공유됩니다. 시간 `T`에서의 은닉 상태는 전체 접두어(prefix)의 요약입니다. 분류를 위해 `h_1 ... h_T`에 대해 풀링(max, mean, 또는 last)을 수행합니다.

단순 RNN은 기울기 소실(vanishing gradients) 문제를 겪습니다. **LSTM**은 무엇을 잊을지, 무엇을 저장할지, 무엇을 출력할지 결정하는 게이트를 추가하여 긴 시퀀스를 통해 기울기를 안정화합니다. **GRU**는 LSTM을 두 개의 게이트로 단순화했으며, 더 적은 매개변수로 유사한 성능을 발휘합니다.

**양방향 RNN(Bidirectional RNNs)**은 하나의 RNN을 순방향으로, 다른 하나를 역방향으로 실행하여 은닉 상태를 연결합니다. 모든 토큰의 표현은 좌측 및 우측 컨텍스트를 모두 참조합니다. 태깅 작업에 필수적입니다.

```figure
rnn-unroll
```

## 구현하기

### 1단계: PyTorch에서 TextCNN

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

`transpose(1, 2)`는 `nn.Conv1d`이 중간 축을 채널로 취급하기 때문에 `[batch, seq_len, embed_dim]`를 `[batch, embed_dim, seq_len]`로 재형상합니다. 풀링된 출력은 입력 길이에 관계없이 고정 크기입니다.

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

마지막 상태 풀링이 아닌 시퀀스에 대한 최대 풀링(max-pool)을 수행합니다. 분류의 경우, 긴 시퀀스의 끝부분에 있는 정보가 마지막 상태를 지배하는 경향이 있으므로, 최대 풀링이 일반적으로 마지막 은닉 상태를 취하는 것보다 더 좋은 성능을 냅니다.

### 3단계: 기울기 소실 데모 (직관)

게이팅이 없는 단순 RNN은 장거리 의존성을 학습할 수 없습니다. 장난감 태스크를 고려해 보세요: 토큰 `A`이 시퀀스 내 어디에 나타났는지 예측하는 것입니다. `A`이 위치 1에 있고 시퀀스가 100 토큰 길이라면, 손실로부터의 기울기는 재귀 가중치(recurrent weight)의 99번 곱셈을 통해 역전파되어야 합니다. 가중치가 1보다 작으면 기울기가 소실됩니다. 1보다 크면 폭발합니다.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 가중치=0.9, 100 단계일 때:
#   0.9 ^ 100 ≈ 2.7e-5
# 100단계에서 1단계로의 기울기는 사실상 0입니다.
```

LSTM은 네트워크를 통해 가산 상호작용(additive interactions)만 수행하는 **셀 상태(cell state)**로 이를 해결합니다 (forget 게이트가 곱셈적으로 스케일링하지만, 기울기는 여전히 "고속도로(highway)"를 따라 흐릅니다). GRU는 더 적은 매개변수로 유사한 방식으로 수행합니다. 둘 다 100+ 단계 시퀀스를 통해 안정적인 학습을 제공합니다.

### 4단계: 왜 이것으로도 충분하지 않았는가

LSTM을 사용해도 세 가지 문제가 지속되었습니다.

1. **순차적 병목.** 길이 1000인 시퀀스에 대해 RNN을 학습하려면 1000개의 순차적 순방향/역방향 단계가 필요합니다. 시간 축을 따라 병렬화할 수 없습니다.
2. **인코더-디코더 구성의 고정 크기 컨텍스트 벡터.** 디코더는 인코더의 최종 은닉 상태만 보며, 이는 전체 입력에 대해 압축된 상태입니다. 긴 입력은 세부 정보를 잃습니다. 09강에서 이 문제를 직접 다룹니다.
3. **장거리 의존성 정확도 한계.** LSTM은 일반 RNN보다 성능이 뛰어나지만, 200+ 단계에 걸쳐 특정 정보를 전파하는 데는 여전히 어려움을 겪습니다.

어텐션은 이 세 가지 문제를 모두 해결했습니다. 트랜스포머는 재귀를 완전히 제거했습니다. 10강이 전환점입니다.

## 사용하기

PyTorch의 `nn.LSTM`, `nn.GRU`, `nn.Conv1d`는 프로덕션 준비가 완료되었습니다. 학습 코드는 표준입니다.

Hugging Face는 입력 레이어로 연결할 수 있는 사전 학습된 임베딩을 제공합니다:

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

제약 조건에 부합할 때 사용하는 체크리스트입니다.

- **엣지 / 온디바이스 추론.** GloVe 임베딩을 사용한 TextCNN은 트랜스포머보다 10-100배 작습니다. 배포 대상이 스마트폰이라면 이 스택이 적합합니다.
- **스트리밍 / 온라인 분류.** RNN은 토큰을 하나씩 처리하며, 트랜스포머는 전체 시퀀스가 필요합니다. 실시간으로 들어오는 텍스트의 경우 LSTM이 여전히 우세합니다.
- **베이스라인용 소형 모델.** 새로운 작업에 대한 빠른 반복. CPU에서 5분 안에 TextCNN을 학습할 수 있습니다.
- **제한된 데이터의 시퀀스 레이블링.** BiLSTM-CRF (06강)는 1k-10k개의 레이블이 지정된 문장에 대해 여전히 프로덕션급 NER 아키텍처입니다.

나머지 모든 것은 트랜스포머로 이동합니다.

## 출시하기

`outputs/prompt-text-encoder-picker.md`로 저장하세요:

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

## 연습 문제

1. **쉬움.** 3개 클래스의 장난감 데이터셋(직접 만든 데이터)에 대해 TextCNN을 학습하세요. 필터 폭(2, 3, 4)이 단일 폭(3)보다 평균 F1에서 더 나은 성능을 내는지 확인하세요.
2. **중간.** LSTM 분류기에 대해 최대 풀링, 평균 풀링, 마지막 상태 풀링을 구현하세요. 작은 데이터셋에서 비교하고, 어떤 풀링이 승리하는지 문서화하며 그 이유를 가설로 제시하세요.
3. **어려움.** BiLSTM-CRF NER 태거를 구축하세요 (06강과 이 강의를 결합). CoNLL-2003으로 학습하세요. 06강의 CRF 단독 베이스라인 및 BERT 미세 조정과 비교하세요. 학습 시간, 메모리, F1을 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| TextCNN | 텍스트용 CNN | 단어 임베딩에 대한 1D 합성곱의 스택과 전역 최대 풀링. Kim (2014). |
| RNN | 순환 신경망 | 각 시간 단계에서 숨겨진 상태가 업데이트됨: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | 게이트가 있는 RNN | 입력 / 망각 / 출력 게이트 + 셀 상태를 추가합니다. 긴 시퀀스를 통해 안정적으로 학습됩니다. |
| GRU | 더 단순한 LSTM | 세 개 대신 두 개의 게이트를 사용합니다. 유사한 정확도, 더 적은 매개변수. |
| Bidirectional | 양방향 | 순방향 + 역방향 RNN이 연결(concatenate)됩니다. 모든 토큰이 컨텍스트의 양쪽을 봅니다. |
| Vanishing gradient | 학습 신호 소멸 | 순수 RNN에서 <1인 가중치를 반복적으로 곱하면 초기 단계의 기울기가 사실상 0이 됩니다. |

## 추가 읽기

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 논문. 8페이지. 읽기 쉽습니다.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 논문. 예상외로 명쾌합니다.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM을 누구나 이해할 수 있게 만든 다이어그램.
