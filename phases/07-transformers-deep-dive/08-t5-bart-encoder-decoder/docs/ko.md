# T5, BART — 인코더-디코더 모델

> 인코더는 이해하고, 디코더는 생성합니다. 이 둘을 결합하면 입력 → 출력 작업에 특화된 모델을 만들 수 있습니다: 번역, 요약, 재작성, 전사.

**유형:** Learn
**언어:** Python
**선수 요건:** 7단계 · 05강 (전체 트랜스포머), 7단계 · 06강 (BERT), 7단계 · 07강 (GPT)
**시간:** 약 45분

## 문제점

디코더 전용 GPT와 인코더 전용 BERT는 각각 2017년 아키텍처를 서로 다른 목적에 맞게 단순화했습니다. 하지만 많은 작업은 본질적으로 입력-출력 형태를 띱니다:

- 번역: 영어 → 프랑스어.
- 요약: 5,000 토큰 기사 → 200 토큰 요약.
- 음성 인식: 오디오 토큰 → 텍스트 토큰.
- 구조화 추출: 산문 → JSON.

이러한 작업에는 인코더-디코더 구조가 가장 잘 맞습니다. 인코더는 소스의 밀집 표현(dense representation)을 생성합니다. 디코더는 이 표현에 매 단계마다 교차 어텐션(Cross-Attention)하며 출력을 생성합니다. 학습은 출력 쪽에서 한 칸씩 이동(shift-by-one)합니다. GPT와 동일한 손실 함수를 사용하되, 인코더 출력에 조건을 걸기만 합니다.

두 논문이 현대적 플레이북을 정의했습니다:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." 모든 NLP 작업을 텍스트 입력, 텍스트 출력으로 재정의합니다. 단일 아키텍처, 단일 어휘, 단일 손실 함수. 입력에서 손상된 스팬(corrupt spans)을 복원하는 마스크 스팬 예측(masked span prediction)으로 사전 학습합니다.
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." 디노이징 오토인코더(Denoising Autoencoder): 입력을 여러 방식으로 손상(shuffle, mask, delete, rotate)하고, 디코더가 원본을 복원하도록 요청합니다.

2026년 현재, 입력 구조가 중요한 곳에서 인코더-디코더 형식이 계속 사용되고 있습니다:

- Whisper (음성 → 텍스트).
- Google의 번역 스택.
- 명확한 컨텍스트 및 편집 구조를 가진 일부 코드 완성/수리 모델.
- 구조화된 추론 작업을 위한 Flan-T5 및 변형 모델.

디코더 전용 모델이 주목받았지만, 인코더-디코더 모델은 사라지지 않았습니다.

## 개념

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 순방향 루프

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

중요한 점은 인코더가 입력당 한 번만 실행된다는 것입니다. 디코더는 자기회귀적으로 실행되지만, 모든 단계에서 *동일한* 인코더 출력에 교차 어텐션을 수행합니다. 긴 입력의 경우 인코더 출력을 캐싱하면 무료 속도 향상을 얻을 수 있습니다.

### T5 사전 학습 — 스팬 손상(span corruption)

입력의 랜덤 스팬(평균 길이 3 토큰, 총 15%)을 선택합니다. 각 스팬을 고유한 센티널(sentinel)로 대체합니다: `<extra_id_0>`, `<extra_id_1>` 등. 디코더는 센티널 접두어가 붙은 손상된 스팬만 출력합니다:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

전체 시퀀스를 예측하는 것보다 더 저렴한 신호입니다. T5 논문의 Ablation 실험에서 MLM (BERT) 및 prefix-LM (UniLM)과 경쟁력이 있습니다.

### BART 사전 학습 — 다중 노이즈 디노이징(multi-noise denoising)

BART는 5가지 노이즈 함수를 시도합니다:

1. 토큰 마스킹(Token masking).
2. 토큰 삭제(Token deletion).
3. 텍스트 채우기(Text infilling) (스팬을 마스킹하고, 디코더가 올바른 길이를 삽입합니다).
4. 문장 순열(Sentence permutation).
5. 문서 회전(Document rotation).

텍스트 채우기 + 문장 순열을 결합하면 최상의 다운스트림 성능을 얻습니다. 디코더는 항상 원본을 재구성합니다. BART의 출력은 손상된 스팬뿐만 아니라 전체 시퀀스이므로, 사전 학습 연산량이 T5보다 높습니다.

### 추론

GPT와 동일한 자기회귀 생성입니다. Greedy / beam / top-p 샘플링이 적용됩니다. 번역 및 요약에는 Beam search (너비 4–5)가 표준입니다. 출력 분포가 채팅보다 좁기 때문입니다.

### 2026년, 각 변형을 선택하는 시점

| 작업 | 인코더-디코더? | 이유 |
|------|------------------|-----|
| 번역 | 예, 보통 | 명확한 소스 시퀀스; 고정된 출력 분포; Beam search가 잘 작동 |
| 음성-텍스트 변환 | 예 (Whisper) | 입력 모달리티가 출력과 다름; 인코더가 오디오 특징을 형성 |
| 채팅 / 추론 | 아니요, 디코더 전용 | 지속되는 "입력"이 없음 — 대화가 시퀀스임 |
| 코드 완성 | 보통 아니요 | 긴 컨텍스트를 가진 디코더 전용 모델이 승리; Qwen 2.5 Coder 같은 코드 모델은 디코더 전용 |
| 요약 | 둘 다 가능 | BART, PEGASUS는 이전 디코더 전용 기준선을 능가함; 현대적인 디코더 전용 LLM이 그들을 따라잡음 |
| 구조화된 추출 | 둘 다 | T5는 "텍스트 → 텍스트"가 모든 출력 형식을 흡수하므로 깔끔함 |

~2022년 이후의 추세는 다음과 같습니다. 인코더-디코더가 담당하던 작업을 디코더 전용 모델이 대체하고 있습니다. 그 이유는 (a) 지시문 미세 조정된 디코더 전용 LLM이 프롬프트를 통해 모든 작업에 범용적으로 적용될 수 있고, (b) 하나의 아키텍처가 두 개보다 확장하기 쉬우며, (c) RLHF가 디코더를 전제로 하기 때문입니다. 인코더-디코더는 입력 모달리티가 다른 경우(음성, 이미지)나 빔 검색 품질이 중요한 곳에서 여전히 사용되고 있습니다.

```figure
encoder-decoder
```

## 구현하기

`code/main.py`를 참고하세요. 장난감 코퍼스에 T5 스타일 스팬 손상을 구현합니다. 이는 이후 모든 인코더-디코더 사전 학습 레시피에서 등장하는 이 강의에서 가장 유용한 부분입니다.

### 1단계: 스팬 손상

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

타겟 형식은 T5 관례인 `<sent0> span0 <sent1> span1 ...`입니다. 손상된 입력은 스팬 위치에서 변경되지 않은 토큰과 센티널 토큰을 교대로 배치합니다.

### 2단계: 왕복 검증

손상된 입력과 타겟을 주어 원문 문장을 복원합니다. 손상이 가역적이라면 순방향 패스가 잘 정의됩니다. 이는 정상 동작 확인입니다. 실제 훈련에서는 이 과정을 수행하지 않지만, 테스트 비용이 저렴하고 스팬 기록의 오프 바이 원(off-by-one) 버그를 잡아낼 수 있습니다.

### 3단계: BART 노이징

다섯 가지 함수: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. 두 가지를 조합하여 결과를 보여 주세요.

## 사용하기

HuggingFace 참고 자료:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5의 비결은 작업 이름을 입력 텍스트에 넣는 것입니다. 모든 작업이 텍스트 입력, 텍스트 출력으로 처리되므로 동일한 모델이 수십 개의 작업을 처리할 수 있습니다. 2026년 현재 이 패턴은 지시문 미세 조정된 디코더 전용 모델에 의해 일반화되었지만, T5가 이를 최초로 정립했습니다.

## 출시하기

`outputs/skill-seq2seq-picker.md`를 참고하세요. 이 스킬은 입력-출력 구조, 지연 시간, 품질 목표를 고려하여 새로운 작업에 인코더-디코더와 디코더 전용 중 하나를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고, 30개 토큰 문장에 스팬 손상을 적용한 후, 센티널이 아닌 소스 토큰과 디코딩된 타겟 스팬을 연결하면 원문이 복원되는지 확인하세요.
2. **중간.** BART의 `text_infill` 노이징을 구현하세요. 랜덤 스팬을 단일 `<mask>` 토큰으로 교체하고, 디코더가 올바른 스팬 길이와 내용을 추론해야 합니다. 한 가지 예제를 보여 주세요.
3. **어려움.** `flan-t5-small`을 작은 영어 → pig-Latin 코퍼스(200 쌍)로 미세 조정합니다. 유지된 50쌍 세트에서 BLEU를 측정합니다. 동일한 데이터와 동일한 컴퓨팅 자원으로 `Llama-3.2-1B`을 미세 조정하는 것과 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 인코더-디코더 | "Seq2seq 트랜스포머" | 두 스택: 입력용 양방향 인코더, 출력용 교차 어텐션(Cross-Attention)을 가진 인과적 디코더. |
| 교차 어텐션(Cross-Attention) | "소스가 타겟과 대화하는 곳" | 디코더의 Q × 인코더의 K/V. 인코더 정보가 디코더로 들어오는 유일한 지점. |
| 스팬 손상 | "T5의 사전 학습 트릭" | 랜덤 스팬을 센티널 토큰으로 대체; 디코더가 스팬을 출력합니다. |
| 디노이징 목적 함수 | "BART의 게임" | 입력에 노이즈 함수를 적용하고, 디코더가 깨끗한 시퀀스를 재구성하도록 학습합니다. |
| 센티널 토큰 | "`<extra_id_N>` 자리 표시자" | 소스에서 손상된 스팬을 태그하고 타겟에서 다시 태그하는 특수 토큰. |
| Flan | "지시문 미세 조정된 T5" | 1,800개 이상의 작업으로 미세 조정된 T5; 지시문 따르기(Instruction Following)에서 인코더-디코더가 경쟁력 있게 만듭니다. |
| 빔 검색 | "디코딩 전략(Decoding Strategy)" | 각 단계에서 상위 k개의 부분 시퀀스를 유지; 번역/요약의 표준. |
| 티처 포싱 | "학습 시간 입력" | 학습 중, 디코더에 샘플링된 토큰이 아닌 실제 이전 출력 토큰을 공급합니다. |

## 추가 읽기

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, 2026년 표준 인코더-디코더.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 참조 구현.
