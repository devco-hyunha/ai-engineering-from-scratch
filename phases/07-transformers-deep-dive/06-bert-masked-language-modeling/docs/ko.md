# BERT — 마스킹 언어 모델링

> GPT는 다음 단어를 예측합니다. BERT는 빠진 단어를 예측합니다. 한 문장의 차이 — 그리고 임베딩 관련 모든 것의 반 세기.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 05 (전체 트랜스포머), 5단계 · 02 (텍스트 표현)
**시간:** 약 45분

## 문제점

2018년, 모든 NLP 작업 — 감정 분석, NER, QA, 함의 —은 자체 레이블 데이터로 자체 모델을 처음부터 학습했습니다. 미세 조정할 수 있는 "영어 이해" 사전 학습 체크포인트는 없었습니다. ELMo (2018)는 양방향 LSTM으로 컨텍스트 임베딩을 사전 학습할 수 있음을 보여 주었습니다. 도움이 되었지만 일반화되지는 않았습니다.

BERT (Devlin et al. 2018)는 이렇게 물었습니다: 트랜스포머 인코더를 가져와 인터넷의 모든 문장으로 학습하고, 양쪽 컨텍스트에서 빠진 단어를 예측하도록 강제하면 어떨까요? 그런 다음 다운스트림 작업에 하나의 헤드만 미세 조정합니다. 매개변수 효율성은 혁명이었습니다.

결과: 18개월 안에 BERT와 그 변형 (RoBERTa, ALBERT, ELECTRA)은 존재했던 모든 NLP 리더보드를 장악했습니다. 2020년까지 지구상의 모든 검색 엔진, 콘텐츠 Moderation 파이프라인, 시맨틱 검색 시스템은 내부에 BERT를 가지고 있었습니다.

2026년에도 인코더 전용 모델은 분류, 검색, 구조화된 추출에 여전히 올바른 도구입니다. 토큰당 디코더보다 5–10배 더 빠르게 실행되며, 임베딩은 모든 현대적 검색 스택의 백본입니다. ModernBERT (2024년 12월)는 Flash Attention + RoPE + GeGLU로 아키텍처를 8K 컨텍스트로 확장했습니다.

## 개념

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 학습 신호

문장을 가져오세요: `the quick brown fox jumps over the lazy dog`.

토큰의 15%를 무작위로 마스킹하세요:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

마스킹된 위치에서 원본 토큰을 예측하도록 모델을 학습하세요. 인코더는 양방향이기 때문에, 위치 1에서 `[MASK]`를 예측할 때 위치 2+의 `brown fox jumps`를 사용할 수 있습니다. 이것이 GPT가 할 수 없는 것입니다.

### BERT 마스킹 규칙

예측을 위해 선택된 토큰의 15% 중:

- 80%는 `[MASK]`로 대체됩니다.
- 10%는 무작위 토큰으로 대체됩니다.
- 10%는 변경되지 않습니다.

왜 항상 `[MASK]`를 사용하지 않을까요? `[MASK]`는 추론 시점에 나타나지 않기 때문입니다. 모델이 마스킹된 위치의 100%에서 `[MASK]`를 기대하도록 학습하면 사전 학습과 미세 조정 사이에 분포 이동(Distribution Shift)이 발생합니다. 10% 랜덤 + 10% 원본 유지 전략은 모델의 정직성을 유지합니다.

### 다음 문장 예측(NSP) — 그리고 폐기된 이유

원본 BERT는 NSP로도 학습했습니다: 두 문장 A와 B가 주어졌을 때, B가 A를 따르는지 예측하는 것입니다. RoBERTa (2019)는 이를 제거(ablation)하여 NSP가 도움이 되지 않고 해를 끼친다는 것을 입증했습니다. 현대의 인코더는 이를 생략합니다.

### 2026년에 바뀐 점: ModernBERT

2024년 ModernBERT 논문은 2026년 기반 요소(primitives)로 블록을 재구성했습니다:

| 구성 요소 | 원본 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 위치 인코딩 | 학습된 절대 위치 | RoPE |
| 활성화 함수 | GELU | GeGLU |
| 정규화(Normalization) | LayerNorm | 사전 정규화(pre-norm) RMSNorm |
| 어텐션(Attention) | 완전 밀집(Dense) | 교대로 적용되는 지역(128) + 전역 |
| 컨텍스트 윈도우(Context Window) 길이 | 512 | 8192 |
| 토크나이저 | WordPiece | 바이트 쌍 인코딩 (BPE)(Byte Pair Encoding (BPE)) |

그리고 2018년 스택과 달리 Flash-Attention 네이티브입니다. 시퀀스 길이 8K에서 DeBERTa-v3보다 GLUE 점수가 더 좋으면서 추론 속도가 2–3배 빠릅니다.

### 2026년에도 인코더를 선택하는 사용 사례

| 작업 | 인코더가 디코더를 이기는 이유 |
|------|---------------------------|
| 검색 / 시맨틱 검색 임베딩 | 양방향 컨텍스트 = 토큰당 더 나은 임베딩 품질 |
| 분류 (감정, 의도, 독성) | 한 번의 순방향 패스; 생성 오버헤드 없음 |
| NER / 토큰 라벨링 | 위치별 출력, 본질적으로 양방향 |
| 제로샷(Zero-Shot) 함의 (NLI) | 인코더 위에 분류기 헤드 |
| RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation))용 리랭커(Reranker) | 교차 인코더(Cross-encoder) 점수 매기기, LLM 리랭커보다 10배 빠름 |

```figure
transformer-residual
```

## 구현하기

### 1단계: 마스킹 로직

`code/main.py`를 참조하세요. 함수 `create_mlm_batch`는 토큰 ID 목록, 어휘 크기, 마스킹 확률을 받습니다. 입력 ID (마스크가 적용된)와 레이블 (마스킹된 위치에만, 나머지는 -100 — PyTorch의 ignore index 관례)을 반환합니다.

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 원본 유지
    return input_ids, labels
```

### 2단계: 작은 코퍼pus에서 MLM 예측 실행

어휘 20개, 문장 200개를 사용하여 2층 인코더 + MLM 헤드를 학습합니다. 기울기 계산은 하지 않으며, 순전파(sanity check)만 수행합니다. 전체 학습에는 PyTorch가 필요합니다.

### 3단계: 마스크 유형 비교

`[MASK]` 없이도 모델을 사용할 수 있도록 하는 3가지 규칙을 보여줍니다. 마스크되지 않은 문장과 마스크된 문장에 대해 예측을 수행합니다. 학습 과정에서 두 패턴을 모두 보았으므로, 두 경우 모두 합리적인 토큰 분포가 생성되어야 합니다.

### 4단계: 헤드 미세 조정

장난감(sentiment) 데이터셋에서 MLM 헤드를 분류 헤드로 교체합니다. 헤드만 학습하고 인코더는 고정(frozen)합니다. 모든 BERT 애플리케이션이 따르는 패턴입니다.

## 사용하기

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**임베딩 모델은 미세 조정된 BERT입니다.** `all-MiniLM-L6-v2`와 같은 `sentence-transformers` 모델은 대조 학습(loss)으로 학습된 BERT입니다. 인코더는 동일하며, 손실 함수만 변경되었습니다.

**교차 인코더 리랭커도 미세 조정된 BERT입니다.** `[CLS] query [SEP] doc [SEP]`에서 쌍(pair) 분류를 수행합니다. 쿼리와 문서 간의 양방향 어텐션은 교차 인코더가 바이인코더보다 품질 우위를 갖는 이유입니다.

**2026년에 BERT를 선택하지 않는 경우.** 생성(gen) 작업은 제외합니다. 인코더는 토큰을 자기회귀적으로 생성할 합리적인 방법이 없습니다. 또한: 10억 매개변수 미만인 경우, 더 유연한 작은 디코더(Phi-3-Mini, Qwen2-1.5B)가 품질을 맞출 수 있습니다.

## 출시하기

`outputs/skill-bert-finetuner.md`을 참조하세요. 이 스킬은 새로운 분류 또는 추출 작업을 위해 BERT 미세 조정(백본 선택, 헤드 사양, 데이터, 평가, 중단)의 범위를 정의합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하고 10,000개 토큰에 대한 마스크 분포를 출력합니다. 약 15%가 선택되며, 그 중 약 80%가 `[MASK]`이 되는지 확인하세요.
2. **중간.** 전체 단어 masking을 구현하세요. 단어가 하위 단어(subword)로 토큰화되면, 모든 하위 단어를 함께 mask하거나 전혀 mask하지 않습니다. 500개 문장 코퍼스에서 MLM 정확도가 향상되는지 측정하세요.
3. **어려움.** 공개 데이터셋의 10,000개 문장을 사용하여 작은 BERT(2층, d=64)를 학습합니다. SST-2 sentiment에 대해 `[CLS]` 토큰을 미세 조정합니다. 매개변수가 동일한 디코더 전용(baseline)과 비교했을 때, 어느 쪽이 이길까요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | 학습 신호: 토큰의 15%를 무작위로 `[MASK]`로 교체하고 원본을 예측합니다. |
| Bidirectional | "양방향" | 인코더 어텐션에는 인과적 마스크가 없으므로 모든 위치가 다른 모든 위치를 참조합니다. |
| `[CLS]` | "풀링 토큰" | 모든 시퀀스 앞에 붙이는 특수 토큰으로, 최종 임베딩은 문장 수준 표현으로 사용됩니다. |
| `[SEP]` | "세그먼트 구분자" | 쌍을 이루는 시퀀스(예: 쿼리/문서, 문장 A/B)를 구분합니다. |
| NSP | "Next sentence prediction" | BERT의 두 번째 사전 학습 작업으로, RoBERTa에서는 쓸모없는 것으로 확인되어 2019년 이후 제거되었습니다. |
| Fine-tuning | "작업에 적응" | 인코더는 대부분 동결하고, 하위 작업용 작은 헤드를 상단에 학습합니다. |
| Cross-encoder | "리랭커" | 쿼리와 문서를 모두 입력으로 받아 관련성 점수를 출력하는 BERT입니다. |
| ModernBERT | "2024년 리뉴얼" | RoPE, RMSNorm, GeGLU, 지역/전역 어텐션 교대, 8K 컨텍스트 윈도우로 인코더를 재구축했습니다. |

## 추가 읽기

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 원 논문.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERT를 올바르게 학습하는 방법; NSP를 제거합니다.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 같은 연산량에서 MLM보다 교체된 토큰 검출이 더 우수합니다.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 논문.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 표준 인코더 참고 자료.
