# T5, BART — 인코더-디코더 모델 (Encoder-Decoder Models)

> 인코더는 이해하고, 디코더는 생성합니다. 이 둘을 결합하면 번역, 요약, 재작성, 전사(transcribe)와 같은 입력 → 출력(input → output) 작업을 수행하는 모델을 구축할 수 있습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 문제점 (The Problem)

디코더 전용(Decoder-only) GPT와 인코더 전용(Encoder-only) BERT는 각각 서로 다른 목표를 위해 2017년의 아키텍처를 간소화했습니다. 하지만 많은 작업은 본질적으로 입력-출력(input-output) 구조를 가집니다.

- 번역(Translation): 영어 → 프랑스어.
- 요약(Summarization): 5,000토큰 기사 → 200토큰 요약본.
- 음성 인식(Speech recognition): 오디오 토큰 → 텍스트 토큰.
- 구조화된 추출(Structured extraction): 산문 → JSON.

이러한 작업에는 인코더-디코더(Encoder-decoder) 구조가 가장 적합합니다. 인코더는 소스(source)의 밀집된 표현(dense representation)을 생성합니다. 디코더는 매 단계마다 해당 표현에 교차 주의(cross-attention)를 수행하며 출력을 생성합니다. 학습은 출력 측에서 한 칸씩 밀린(shift-by-one) 방식으로 이루어집니다. GPT와 동일한 손실 함수(loss)를 사용하되, 인코더 출력에 조건화(conditioned)될 뿐입니다.

두 편의 논문이 현대적인 플레이북을 정의했습니다.

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." 모든 NLP 작업을 '텍스트 입력, 텍스트 출력' 방식으로 재구성했습니다. 단일 아키텍처, 단일 어휘집(vocabulary), 단일 손실 함수를 사용합니다. 마스크된 구간 예측(masked span prediction, 입력의 구간을 손상시키고 출력에서 이를 디코딩함) 방식으로 사전 학습되었습니다.
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." 노이즈 제거 오토인코더(Denoising autoencoder) 방식입니다. 입력을 여러 방식(셔플, 마스킹, 삭제, 회전)으로 손상시킨 후, 디코더가 원래의 입력을 재구성하도록 합니다.

2026년 현재, 인코더-디코더 형식은 입력 구조가 중요한 영역에서 여전히 활용되고 있습니다.

- Whisper (음성 → 텍스트).
- Google의 번역 스택.
- 문맥과 편집(context-and-edit) 구조가 뚜렷한 일부 코드 완성/수정 모델.
- 구조화된 추론 작업을 위한 Flan-T5 및 그 변형 모델들.

디코더 전용 모델이 주목을 받았지만, 인코더-디코더 모델은 결코 사라지지 않았습니다.

## 개념 (The Concept)

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 순방향 루프 (The forward loop)

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

결정적으로, 인코더(encoder)는 입력당 한 번만 실행됩니다. 디코더(decoder)는 자기회귀(autoregressively) 방식으로 실행되지만, 매 단계마다 *동일한* 인코더 출력에 대해 교차 주의(cross-attention)를 수행합니다. 인코더 출력을 캐싱(caching)하면 긴 입력에 대해 추가 비용 없이 속도를 높일 수 있습니다.

### T5 사전 학습(Pretraining) — 스팬 오염(Span Corruption)

입력값에서 무작위 스팬(span)을 선택합니다(평균 길이 3개 토큰, 전체의 15%). 각 스팬을 고유한 센티넬(sentinel) 토큰인 `<extra_id_0>`, `<extra_id_1>` 등으로 교체합니다. 디코더는 센티넬 접두사가 붙은 오염된 스팬만을 출력합니다.

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

이는 전체 시퀀스를 예측하는 것보다 효율적인 신호(signal)를 제공합니다. T5 논문의 절제 연구(ablation study)에 따르면 MLM(BERT) 및 prefix-LM(UniLM)과 대등한 성능을 보입니다.

### BART 사전 학습(Pretraining) — 다중 노이즈 제거(Multi-noise denoising)

BART는 다섯 가지 노이징 함수(noising functions)를 사용합니다.

1. 토큰 마스킹(Token masking).
2. 토큰 삭제(Token deletion).
3. 텍스트 인필링(Text infilling) (특정 구간을 마스킹하면, 디코더가 적절한 길이를 삽입함).
4. 문장 순열(Sentence permutation).
5. 문서 회전(Document rotation).

텍스트 인필링(text infilling)과 문장 순열(sentence permutation)을 결합했을 때 가장 좋은 다운스트림(downstream) 성능을 기록했습니다. 디코더는 항상 원본을 재구성합니다. BART의 출력은 손상된 구간만이 아니라 전체 시퀀스(full sequence)입니다. 따라서 사전 학습 계산량(pretraining compute)은 T5보다 높습니다.

### 추론 (Inference)

GPT와 동일한 자기회귀(autoregressive) 생성 방식을 사용합니다. Greedy / beam / top-p 샘플링이 적용됩니다. 번역 및 요약 작업의 경우, 출력 분포가 채팅보다 좁기 때문에 빔 서치(beam search, 너비 4–5)를 사용하는 것이 표준입니다.

### 2026년 현재, 변형(Variant) 선택 가이드

| 작업(Task) | Encoder-decoder 사용 여부 | 이유 |
|------|------------------|-----|
| 번역 (Translation) | 예, 일반적으로 사용 | 소스 시퀀스가 명확함; 고정된 출력 분포; 빔 서치(beam search)가 효과적임 |
| 음성-텍스트 변환 (Speech-to-text) | 예 (Whisper) | 입력 모달리티가 출력과 다름; 인코더가 오디오 특징을 형성함 |
| 채팅 / 추론 (Chat / reasoning) | 아니요, decoder-only 사용 | 지속적인 "입력"이 없음 — 대화 자체가 하나의 시퀀스임 |
| 코드 완성 (Code completion) | 일반적으로 아니요 | 긴 컨텍스트를 가진 decoder-only가 유리함; Qwen 2.5 Coder와 같은 코드 모델은 decoder-only임 |
| 요약 (Summarization) | 둘 다 가능 | BART, PEGASUS가 초기 decoder-only 베이스라인을 능가함; 최신 decoder-only LLM들도 이들과 대등한 성능을 보임 |
| 구조화된 추출 (Structured extraction) | 둘 다 가능 | T5는 "텍스트 → 텍스트" 구조가 모든 출력 형식을 수용하므로 깔끔함 |

~2022년 이후의 트렌드: (a) 지시어 튜닝(instruction-tuned)된 decoder-only LLM이 프롬프팅을 통해 무엇이든 일반화할 수 있고, (b) 두 개의 구조보다 하나의 구조가 확장(scale)하기 더 쉬우며, (c) RLHF가 decoder를 가정하기 때문에, encoder-decoder가 담당하던 작업들을 decoder-only가 점유하고 있습니다. 다만, 입력 모달리티가 다르거나(음성, 이미지), 빔 서치의 품질이 중요한 경우에는 여전히 encoder-decoder가 사용됩니다.

```figure
encoder-decoder
```

## 직접 구현해 보기 (Build It)

`code/main.py`를 확인해 보세요. 작은 규모의 코퍼스(toy corpus)를 위해 T5 스타일의 스팬 오염(span corruption) 방식을 구현합니다. 이는 이 레슨에서 가장 유용한 부분으로, 이후 등장하는 모든 인코더-디코더 사전 학습(pretraining) 레시피의 핵심입니다.

### 1단계: span corruption (스팬 오염)

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

타겟 형식은 T5 컨벤션을 따릅니다: `<extra_id_0> span0 <extra_id_1> span1 ...`. 오염된 입력(corrupted input)은 변경되지 않은 토큰들과 스팬(span) 위치에 있는 센티넬(sentinel) 토큰들을 교차하여 구성됩니다.

### 2단계: 왕복 검증 (Verify round-trip)

손상된 입력(corrupted input)과 타겟(target)이 주어졌을 때, 원래의 문장을 재구성해 보세요. 만약 손상 과정이 가역적(reversible)이라면, 순전파(forward pass)가 잘 정의된 것입니다. 이는 일종의 무결성 검사(sanity check)입니다. 실제 학습 과정에서는 이런 작업을 수행하지 않지만, 이 테스트는 비용이 저렴하며 구간 관리(span bookkeeping) 시 발생하는 오프바이원(off-by-one) 버그를 잡아낼 수 있습니다.

### 3단계: BART 노이징 (BART noising)

다섯 가지 함수: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. 이 중 두 가지를 조합하여 그 결과를 보여주세요.

## 사용 방법 (Use It)

HuggingFace 참조 예시:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5의 핵심 기법(trick): 작업 이름(task name)을 입력 텍스트에 포함하는 것입니다. 각 작업이 '텍스트 입력, 텍스트 출력'의 형태를 취하기 때문에, 동일한 모델이 수십 가지의 작업을 처리할 수 있습니다. 2026년 현재 이 패턴은 지시어 튜닝된 디코더 전용(instruction-tuned decoder-only) 모델들에 의해 일반화되었지만, T5가 이를 가장 먼저 체계화했습니다.

## Ship It (실행하기)

`outputs/skill-seq2seq-picker.md`를 참조하세요. 이 스킬은 입력-출력 구조, 지연 시간(latency), 품질 목표가 주어졌을 때 새로운 태스크를 위한 encoder-decoder 방식과 decoder-only 방식 중 하나를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하여 30개의 토큰으로 구성된 문장에 span corruption을 적용해 보세요. 센티넬(sentinel)이 아닌 소스 토큰들과 디코딩된 타겟 span을 결합했을 때 원래 문장이 재현되는지 확인합니다.
2. **중간 (Medium).** BART의 `text_infill` 노이즈를 구현해 보세요. 무작위 span을 단일 `<mask>` 토큰으로 교체하며, 디코더는 올바른 span의 길이와 내용을 추론해야 합니다. 예시를 하나 보여주세요.
3. **어려움 (Hard).** 아주 작은 규모의 영어 → 피그 라틴(pig-Latin) 코퍼스(200개 쌍)로 `flan-t5-small`을 미세 조정(fine-tune)해 보세요. 별도로 분리한 50개 쌍의 데이터셋에 대해 BLEU 점수를 측정합니다. 동일한 연산 자원을 사용하여 같은 데이터로 `Llama-3.2-1B`를 미세 조정했을 때와 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq 트랜스포머" | 두 개의 스택: 입력을 위한 양방향 인코더(bidirectional encoder)와 출력을 위한 교차 주의(cross-attention) 기능이 있는 인과적 디코더(causal decoder). |
| Cross-attention | "소스가 타겟과 대화하는 곳" | 디코더의 `Q` × 인코더의 `K/V`. 인코더의 정보가 디코더로 들어가는 유일한 지점. |
| Span corruption | "T5의 사전 학습 트릭" | 무작위 스팬(span)을 센티넬 토큰(sentinel tokens)으로 교체; 디코더는 해당 스팬을 출력함. |
| Denoising objective | "BART의 방식" | 입력에 노이즈 함수를 적용하고, 디코더가 깨끗한 시퀀스를 재구성하도록 학습함. |
| Sentinel token | "`<extra_id_N>` 플레이스홀더" | 소스에서 손상된 스팬을 표시하고 타겟에서 이를 다시 표시하는 특수 토큰. |
| Flan | "지시어 미세 조정된 T5" | 1,800개 이상의 태스크로 미세 조정된 T5; 인코더-디코더 모델이 지시어 이행(instruction-following)에서 경쟁력을 갖게 함. |
| Beam search | "디코딩 전략" | 각 단계에서 상위 k개의 부분 시퀀스를 유지; 번역/요약의 표준 방식. |
| Teacher forcing | "학습 시 입력 방식" | 학습 중, 샘플링된 토큰이 아닌 실제 이전 출력 토큰을 디코더에 공급함. |

## 추가 읽을거리 (Further Reading)

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, 2026년의 전형적인 인코더-디코더(encoder-decoder) 모델.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 참조 구현체(reference implementation).
