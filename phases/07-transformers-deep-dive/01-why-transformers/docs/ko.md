# 왜 트랜스포머(Transformers)인가 — RNN의 문제점

> RNN은 토큰을 한 번에 하나씩 처리합니다. 반면 트랜스포머는 모든 토큰을 한 번에 처리합니다. 이 단 하나의 구조적 선택이 2017년 이후 딥러닝의 모든 스케일링 곡선(scaling curve)을 바꾸어 놓았습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## 문제점 (The Problem)

2017년 이전에는 언어, 번역, 음성을 포함하여 지구상의 모든 최첨단 시퀀스 모델(state-of-the-art sequence model)이 순환 신경망(recurrent neural network)이었습니다. LSTM과 GRU는 수년간 ImageNet에 상응하는 번역 벤치마크를 석권했습니다. 당시에는 그것이 유일한 도구였습니다.

이 모델들은 세 가지 치명적인 약점을 가지고 있었습니다. 첫째, 순차적 계산(Sequential computation) 방식 때문에 시간 축을 따라 병렬화할 수 없었습니다. 토큰 `t+1`을 처리하려면 토큰 `t`의 은닉 상태(hidden state)가 반드시 필요했습니다. 1,024개의 토큰으로 구성된 시퀀스는 사이클당 1,000,000번의 부동 소수점 연산을 수행할 수 있는 GPU에서도 1,024번의 직렬 단계를 거쳐야 함을 의미했습니다. 병렬 처리를 위해 설계된 하드웨어에서 실제 학습 시간(wall-clock time)은 시퀀스 길이에 따라 선형적으로 증가했습니다.

둘째, 기울기 소실(Vanishing gradients) 문제로 인해 50개 토큰 이전의 정보는 이미 50번의 비선형 연산을 거치며 압축되어 버렸습니다. 게이트형 순환 유닛(LSTM, GRU)이 이 압박을 완화하기는 했으나 완전히 제거하지는 못했습니다. "지난여름 교토로 가는 비행기에서 읽었던 책은..."과 같은 장기 의존성(Long-range dependencies)을 처리하는 데는 일상적으로 실패했습니다.

셋째, 고정된 너비의 은닉 상태(Fixed-width hidden states)로 인해 인코더는 디코더가 정보를 확인하기도 전에 전체 소스 시퀀스를 단 하나의 벡터로 압축해야 했습니다. 소스 시퀀스가 5개의 토큰이든 500개의 토큰이든 상관없이 병목 현상(bottleneck)의 크기는 동일했습니다.

2017년 논문 "Attention Is All You Need"는 급진적인 제안을 했습니다. 순환(recurrence)을 완전히 버리자는 것이었습니다. 모든 위치가 다른 모든 위치를 병렬적으로 참조(attend)하게 만드십시오. 1,024번의 순차적 행렬 곱셈 대신, 한 번의 거대한 행렬 곱셈으로 학습하게 하십시오.

그 결과, 트랜스포머는 모든 모달리티(modality)를 지배하게 되었습니다. 언어(GPT-5, Claude 4, Llama 4), 시각(ViT, DINOv2, SAM 3), 오디오(Whisper), 생물학(AlphaFold 3), 로보틱스(RT-2)에 이르기까지, 동일한 블록 구조를 사용하되 입력값만 다를 뿐입니다.

## 개념 (The Concept)

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**병목 현상으로서의 순환(Recurrence as a bottleneck).** RNN은 `h_t = f(h_{t-1}, x_t)`를 계산합니다. 각 단계는 이전 단계에 의존합니다. `h_4`를 계산하기 전에는 `h_5`를 계산할 수 없습니다. 10,000개 이상의 병렬 코어를 가진 현대적 GPU에서, 긴 시퀀스를 처리할 때 이는 실리콘 자원의 99%를 낭비하게 만듭니다.

**브로드캐스트로서의 어텐션(Attention as a broadcast).** 셀프 어텐션(Self-attention)은 모든 쌍 `(i, j)`에 대해 `output_i = sum_j(a_ij * v_j)`를 동시에 계산합니다. 전체 $N \times N$ 어텐션 행렬은 단 한 번의 배치 행렬 곱셈(batched matmul)으로 채워집니다. 어떤 단계도 다른 단계에 의존하지 않습니다. GPU는 이러한 방식을 매우 선호합니다.

**속도 향상은 상수가 아닙니다.** 이는 `O(N)`의 직렬 깊이(serial depth)와 `O(1)`의 직렬 깊이 사이의 차이입니다. 실제로 트랜스포머는 $N=512$인 동일 하드웨어 환경에서 에폭(epoch)당 5~10배 더 빠르게 학습하며, 어텐션의 `O(N²)` 메모리 벽(memory wall)에 부딪힐 때까지 시퀀스 길이가 길어질수록 그 격차는 더 벌어집니다(이 문제는 나중에 Flash Attention이 해결했습니다 — 레슨 12 참조).

**트랜스포머의 비용(What transformers cost).** 어텐션 메모리는 `O(N²)`로 확장됩니다. 2K 컨텍스트에서는 괜찮지만, 128K 컨텍스트를 위해서는 슬라이딩 윈도우(sliding windows), RoPE 외삽(extrapolation), Flash Attention 타일링(tiling) 또는 선형 어텐션(linear attention) 변형 모델이 필요합니다. 순환 구조는 시간과 메모리 모두에서 `O(N)`이었으나, 트랜스포머는 시간을 메모리와 맞바꾼 뒤 병렬성을 통해 다시 시간을 확보합니다.

**귀납적 편향의 변화(The inductive bias shift).** RNN은 국소성(locality)과 최신성(recency)을 가정합니다. 트랜스포머는 아무것도 가정하지 않습니다 — 모든 쌍이 어텐션의 후보가 됩니다. 이것이 트랜스포머가 제대로 학습하기 위해 더 많은 데이터를 필요로 하지만, 일단 데이터를 확보하면 더 크게 확장(scale)할 수 있는 이유입니다. Chinchilla(2022)는 이를 공식화했습니다: 충분한 토큰이 주어진다면, 트랜스포머는 항상 동일한 파라미터 수를 가진 RNN을 압도합니다.

```figure
rnn-vs-parallel
```

## 직접 구현해 보기 (Build It)

여기에는 신경망이 포함되지 않습니다 — 여러분의 노트북에서 그 격차를 직접 체감할 수 있도록 핵심 병목 현상을 수치적으로 시뮬레이션합니다.

### 1단계: 직렬 깊이(serial depth) 측정

`code/main.py`를 참조하세요. 두 가지 함수를 구축합니다. 하나는 시퀀스를 덧셈의 사슬로 인코딩하며(RNN과 같이 직렬적임), 다른 하나는 병렬 리덕션(parallel reduction)으로 인코딩합니다(Attention과 같이 브로드캐스트 방식). 수학적 원리는 동일하지만, 의존성 그래프(dependency graph)가 다릅니다.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 병렬화 불가: h가 이전의 h에 의존함
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 모든 x가 독립적임
```

최대 100,000개의 요소를 가진 시퀀스에 대해 두 방식의 시간을 측정합니다. RNN 버전은 $O(N)$이며 단일 CPU 파이프라인을 사용합니다. 순수 Python 환경에서도, Attention 방식의 리덕션은 길이가 1,000 이상일 때 RNN보다 빠릅니다. 이는 Python의 `sum()`이 C로 구현되어 있어 단계별 인터프리터 오버헤드 없이 반복을 수행하기 때문입니다.

### 2단계: 이론적 연산량 계산 (count theoretical operations)

두 알고리즘 모두 $N$번의 덧셈을 수행합니다. 차이점은 *의존성 깊이(dependency depth)*, 즉 다음 연산이 시작되기 전까지 얼마나 많은 연산이 순차적으로 수행되어야 하는가에 있습니다. RNN의 깊이는 $N$입니다. Attention의 깊이는 트리 리덕션(tree reduction)을 사용할 경우 $\log(N)$이며, 병렬 스캔(parallel scan)을 사용하면 $1$입니다. GPU 실행 시간을 결정하는 것은 연산 횟수가 아니라 바로 이 깊이입니다.

### 3단계: 긴 시퀀스에 대한 경험적 스케일링 (Empirical scaling on long sequences)

우리는 $O(N)$의 차이를 명확히 보여주는 타이밍 테이블을 출력합니다. 2026년형 Mac 노트북에서는 1,000개 미만의 요소로 구성된 시퀀스는 측정하기에 너무 빠릅니다. 100,000개의 시퀀스는 깔끔한 선형 스캔(linear scan)을 보여줍니다. 이를 12개 레이어의 LSTM에 상응하는 16,384개 토큰의 트랜스포머(transformer) 규모로 확장해 보면, 왜 2016년에 실제 학습 시간(wall-clock time)이 걸림돌이었는지 알 수 있습니다.

## 활용 방법 (Use It)

2026년에도 여전히 RNN을 선택해야 하는 경우:

| 상황 | 선택 |
|-----------|------|
| 스트리밍 추론, 한 번에 하나의 토큰씩 처리, 일정한 메모리 사용 | RNN 또는 상태 공간 모델 (Mamba, RWKV) |
| 어텐션 메모리가 폭발하는 매우 긴 시퀀스 (>1M 토큰) | 선형 어텐션(Linear attention), Mamba 2, Hyena |
| 행렬 곱셈(matmul) 가속기가 없는 엣지 디바이스 | Depthwise-separable RNN이 여전히 FLOPs/watt 측면에서 우세 |
| 그 외 모든 경우 (학습, 배치 추론, 128K 이하의 컨텍스트) | Transformer |

Mamba와 같은 상태 공간 모델(SSMs)은 본질적으로 구조화된 파라미터화를 가진 RNN이며, 이를 통해 두 방식의 장점을 모두 갖추고 있습니다: `O(N)` 스캔 메모리와 선택적 스캔(selective scan)을 통한 병렬 학습이 가능합니다. 이들은 더 나은 긴 컨텍스트 확장성(long-context scaling)을 보이면서도 Transformer 품질의 90%를 회복합니다. 2026년 대부분의 최첨단 연구소들은 하이브리드 SSM+Transformer 모델(예: Jamba, Samba)을 학습시키고 있습니다. 즉, 순환(recurrence)은 사라진 것이 아니라 하나의 구성 요소로 자리 잡았습니다.

## Ship It (실전 적용)

`outputs/skill-architecture-picker.md`를 참조하세요. 이 스킬은 길이(length), 처리량(throughput), 그리고 학습 예산(training-budget) 제약 조건을 바탕으로 새로운 시퀀스 문제에 적합한 아키텍처를 선택합니다. 1B(10억) 토큰 이상의 학습 실행에 대해 순수 RNN(pure RNN)을 추천할 경우, 반드시 그 트레이드오프(trade-off)를 명시해야 하며, 그렇지 않을 경우 추천을 거부해야 합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에서 `rnn_style`을 가져와 스칼라(scalar) 은닉 상태(hidden state)를 길이가 64인 은닉 상태 벡터로 교체해 보세요. 다시 측정해 보세요. 은닉 상태의 차원이 증가함에 따라 직렬 오버헤드(serial overhead)는 얼마나 증가하나요?
2. **중간 (Medium).** 순수 파이썬(pure Python)으로 병렬 접두사 합(parallel prefix-sum, Hillis-Steele scan)을 구현해 보세요. 길이가 1024일 때 직렬 스캔(serial scan)과 동일한 수치적 출력을 생성하는지 확인하세요. 연산 깊이(depth)를 측정해 보세요.
3. **어려움 (Hard).** 어텐션 스타일 리덕션(attention-style reduction)을 GPU 상의 PyTorch로 포팅해 보세요. 시퀀스 길이(sequence length)를 64에서 65,536까지 변화시키며 두 방식의 시간을 측정하세요. 곡선의 형태를 그래프로 그리고 설명해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 순환 (Recurrence) | "RNN은 순차적이다" | 단계 `t`가 단계 `t-1`에 의존하는 계산 방식으로, 시간 축을 따라 직렬 실행을 강제합니다. |
| 직렬 깊이 (Serial depth) | "그래프가 얼마나 깊은가" | 의존적인 연산들의 가장 긴 체인; 무한한 하드웨어가 있더라도 실제 실행 시간(wall-clock)을 제한하는 요소입니다. |
| 어텐션 (Attention) | "토큰들이 서로를 보게 한다" | 위치 $i$와 $j$ 사이의 유사도 점수에서 유도된 $a_{ij}$를 사용하는 가중합 `sum_j a_ij v_j`입니다. |
| 컨텍스트 창 (Context window) | "모델이 얼마나 많이 보는가" | 어텐션 레이어가 입력으로 받을 수 있는 위치의 수; 여기서 이차 함수적(quadratic) 메모리 비용이 발생합니다. |
| 귀납적 편향 (Inductive bias) | "아키텍처에 내장된 가정" | 데이터가 어떤 형태일지에 대한 사전 지식; CNN은 이동 불변성(translation invariance)을, RNN은 최신성(recency)을 가정합니다. |
| 상태 공간 모델 (State-space model) | "대수학이 적용된 RNN" | 구조화된 상태 공간 행렬을 통해 병렬 학습이 가능하도록 매개변수화된 순환 구조입니다. |
| 이차 병목 (Quadratic bottleneck) | "컨텍스트 비용이 왜 이렇게 비싼가" | 어텐션 메모리는 시퀀스 길이 $N$에 대해 `O(N²)`입니다; Flash Attention은 상수를 숨길 뿐, 스케일링 자체를 바꾸지는 못합니다. |

## 추가 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 주류 NLP 분야에서 순환(recurrence) 구조를 종식시킨 논문입니다.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — RNN에 결합된 형태의 어텐션(attention)이 처음 탄생한 논문입니다.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 참고용으로, LSTM의 원형이 되는 논문입니다.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 트랜스포머(transformers)에 대응하는 현대적인 순환 모델의 해답입니다.
