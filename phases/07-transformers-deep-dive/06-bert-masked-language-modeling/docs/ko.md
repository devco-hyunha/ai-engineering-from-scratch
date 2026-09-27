# BERT — 마스크 언어 모델링 (Masked Language Modeling)

> GPT는 다음 단어를 예측합니다. BERT는 누락된 단어를 예측합니다. 단 한 문장의 차이가 모든 임베딩(embedding) 기술의 지난 5년을 결정지었습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## 문제점 (The Problem)

2018년에는 감성 분석(sentiment), 개체명 인식(NER), 질의응답(QA), 함의(entailment) 등 모든 NLP 작업이 각자의 레이블링된 데이터를 사용하여 처음부터(from scratch) 모델을 직접 학습시켜야 했습니다. 미세 조정(fine-tune)할 수 있는 "영어를 이해하는" 사전 학습된 체크포인트라는 개념이 없었습니다. ELMo (2018)는 양방향 LSTM을 사용하여 문맥적 임베딩(contextual embeddings)을 사전 학습할 수 있음을 보여주었지만, 이는 도움이 되었을 뿐 일반화(generalize)되지는 않았습니다.

BERT (Devlin et al. 2018)는 다음과 같은 질문을 던졌습니다: 만약 트랜스포머 인코더(transformer encoder)를 가져와 인터넷상의 모든 문장으로 학습시키고, 양방향 문맥을 통해 누락된 단어를 예측하도록 강제한다면 어떻게 될까? 그런 다음 하나의 헤드(head)만 다운스트림 작업(downstream task)에 맞춰 미세 조정한다면? 파라미터 효율성(parameter efficiency)은 가히 혁명적이었습니다.

그 결과: 18개월 이내에 BERT와 그 변형 모델들(RoBERTa, ALBERT, ELECTRA)은 존재하는 모든 NLP 리더보드를 장악했습니다. 2020년 무렵에는 지구상의 모든 검색 엔진, 콘텐츠 모더레이션 파이프라인, 시맨틱 검색 시스템 내부에 BERT가 탑재되었습니다.

2026년 현재도 인코더 전용(encoder-only) 모델은 분류(classification), 검색(retrieval), 구조화된 추출(structured extraction)을 위한 적절한 도구로 사용되고 있습니다. 이 모델들은 디코더(decoder)보다 토큰당 5~10배 더 빠르게 작동하며, 이들의 임베딩은 모든 현대적 검색 스택의 중추 역할을 합니다. ModernBERT (2024년 12월)는 Flash Attention + RoPE + GeGLU를 통해 아키텍처를 8K 컨텍스트까지 확장하며 한계를 밀어붙였습니다.

## 개념 (The Concept)

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 학습 신호 (The training signal)

다음 문장을 예로 들어 보겠습니다: `the quick brown fox jumps over the lazy dog`.

토큰의 15%를 무작위로 마스킹(Mask)합니다:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

모델이 마스킹된 위치에 원래의 토큰을 예측하도록 학습시킵니다. 인코더는 양방향(bidirectional)이기 때문에, 1번 위치의 `[MASK]`를 예측할 때 2번 이후 위치에 있는 `brown fox jumps`를 사용할 수 있습니다. 이것이 바로 GPT가 할 수 없는 기능입니다.

### BERT 마스킹 규칙 (The BERT mask rules)

예측을 위해 선택된 15%의 토큰 중:

- 80%는 `[MASK]`로 대체됩니다.
- 10%는 무작위 토큰(random token)으로 대체됩니다.
- 10%는 변경되지 않고 그대로 유지됩니다.

왜 항상 `[MASK]`를 사용하지 않을까요? 그 이유는 추론(inference) 시에는 `[MASK]`가 전혀 나타나지 않기 때문입니다. 마스킹된 위치의 100%에서 `[MASK]`가 나타날 것이라고 모델을 학습시키면, 사전 학습(pretraining)과 미세 조정(fine-tuning) 사이에 분포 변화(distribution shift)가 발생하게 됩니다. 10%의 무작위 대체와 10%의 유지 방식은 모델이 실제 데이터 환경에서도 올바르게 작동하도록 유지해 줍니다.

### 다음 문장 예측 (Next Sentence Prediction, NSP) — 그리고 이것이 폐기된 이유

오리지널 BERT 또한 NSP를 통해 학습되었습니다. 즉, 두 문장 A와 B가 주어졌을 때, B가 A의 뒤를 잇는 문장인지 예측하는 방식입니다. 하지만 RoBERTa (2019) 연구를 통해 이 과정을 제거(ablation)해 본 결과, NSP는 성능에 도움이 되기는커녕 오히려 해를 끼친다는 것이 밝혀졌습니다. 따라서 현대적인 인코더(encoders)들은 이 단계를 생략합니다.

### 2026년에 무엇이 바뀌었는가: ModernBERT

2024년 ModernBERT 논문은 2026년형 기본 요소(primitives)를 사용하여 블록을 재구축했습니다:

| 구성 요소 (Component) | 기존 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 위치 인코딩 (Positional) | 학습된 절대 위치 (Learned absolute) | RoPE |
| 활성화 함수 (Activation) | GELU | GeGLU |
| 정규화 (Normalization) | LayerNorm | Pre-norm RMSNorm |
| 어텐션 (Attention) | 전체 밀집 (Full dense) | 교차 로컬(128) + 글로벌 (Alternating local + global) |
| 컨텍스트 길이 (Context length) | 512 | 8192 |
| 토크나이저 (Tokenizer) | WordPiece | BPE |

또한 2018년의 스택과 달리, Flash-Attention이 네이티브로 적용되었습니다. 시퀀스 길이 8K에서 DeBERTa-v3보다 2~3배 더 빠르며, 더 높은 GLUE 점수를 기록합니다.

### 2026년에도 여전히 인코더(Encoder)를 선택하는 유스케이스

| 작업(Task) | 인코더가 디코더보다 우수한 이유 |
|------|---------------------------|
| 검색 / 의미론적 검색 임베딩 (Retrieval / semantic search embeddings) | 양방향 문맥(Bidirectional context) = 토큰당 더 나은 임베딩 품질 |
| 분류 (Classification: 감성, 의도, 유해성) | 단일 순전파(Forward pass); 생성 오버헤드 없음 |
| 개체명 인식 / 토큰 레이블링 (NER / token labeling) | 위치별 출력, 네이티브 양방향성 |
| 제로샷 함의 (Zero-shot entailment, NLI) | 인코더 상단의 분류기 헤드(Classifier head) 활용 |
| RAG를 위한 재순위화 (Reranker for RAG) | 크로스 인코더(Cross-encoder) 스코어링, LLM 재순위화 모델보다 10배 빠름 |

```figure
transformer-residual
```

## 직접 구현해 보기 (Build It)

### 1단계: 마스킹 로직 (masking logic)

`code/main.py`를 참조하세요. `create_mlm_batch` 함수는 토큰 ID 리스트, 어휘 사전 크기(vocab size), 그리고 마스크 확률(mask probability)을 인자로 받습니다. 이 함수는 입력 ID(마스크가 적용된 상태)와 레이블(마스킹된 위치에만 토큰 ID가 있고, 그 외에는 -100인 상태 — PyTorch의 `ignore_index` 관례를 따름)을 반환합니다.

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

### 2단계: 소규모 코퍼스(tiny corpus)에서 MLM 예측 실행하기

20개의 단어와 200개의 문장으로 구성된 어휘 집합(vocabulary)을 사용하여 2계층 인코더(2-layer encoder) + MLM 헤드(MLM head)를 학습시켜 보세요. 그래디언트(gradient) 계산은 수행하지 않으며, 순전파(forward-pass)를 통한 무결성 검사(sanity checks)만 진행합니다. 전체 학습을 위해서는 PyTorch가 필요합니다.

### 3단계: 마스크 유형 비교 (Compare mask types)

3방향 규칙(three-way rule)이 `[MASK]` 없이도 모델을 어떻게 사용 가능한 상태로 유지하는지 보여줍니다. 마스크가 없는 문장과 마스크가 있는 문장에 대해 각각 예측을 수행해 보세요. 모델은 학습 과정에서 두 패턴을 모두 접했기 때문에, 두 경우 모두 합리적인 토큰 분포를 생성해야 합니다.

### 4단계: 헤드 미세 조정(fine-tune head)

장난감 감성 분석 데이터셋(toy sentiment dataset)을 사용하여 MLM 헤드를 분류 헤드(classification head)로 교체해 보세요. 오직 헤드만 학습하며, 인코더는 동결(frozen)됩니다. 이것이 모든 BERT 애플리케이션이 따르는 패턴입니다.

## 사용 방법 (Use It)

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**임베딩 모델(Embedding models)은 파인튜닝된 BERT입니다.** `all-MiniLM-L6-v2`와 같은 `sentence-transformers` 모델은 대조 학습 손실(contrastive loss)을 사용하여 학습된 BERT입니다. 인코더는 동일하지만, 손실 함수(loss)가 변경되었습니다.

**크로스 인코더 리랭커(Cross-encoder rerankers) 또한 파인튜닝된 BERT입니다.** `[CLS] query [SEP] doc [SEP]` 구조를 사용하여 쌍 분류(pair-classification)를 수행합니다. 쿼리와 문서 사이의 양방향 어텐션(bidirectional attention)은 크로스 인코더가 바이인코더(bi-encoders)보다 품질 면에서 우위를 점할 수 있게 하는 핵심 요소입니다.

**2026년에 BERT를 선택하지 말아야 할 경우.** 생성형(generative) 작업은 무엇이든 해당됩니다. 인코더는 토큰을 자기회귀적(autoregressively)으로 생성할 수 있는 합리적인 방법이 없습니다. 또한, 작은 디코더가 더 높은 유연성으로 품질을 맞출 수 있는 1B(10억) 파라미터 미만의 모델(예: Phi-3-Mini, Qwen2-1.5B)을 사용하는 경우도 포함됩니다.

## Ship It (실전 적용)

`outputs/skill-bert-finetuner.md`를 참조하세요. 이 스킬은 새로운 분류(classification) 또는 추출(extraction) 작업을 위한 BERT 미세 조정(fine-tune) 범위(백본 선택, 헤드 사양, 데이터, 평가, 중단 조건)를 정의합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하고 10,000개의 토큰에 대한 마스크 분포를 출력해 보세요. 약 15%가 선택되었는지, 그리고 그중 약 80%가 `[MASK]`가 되는지 확인해 보세요.
2. **중간 (Medium).** 전체 단어 마스킹(whole-word masking)을 구현해 보세요: 만약 하나의 단어가 여러 개의 서브워드(subwords)로 토큰화된다면, 모든 서브워드를 함께 마스킹하거나 혹은 전혀 마스킹하지 않도록 합니다. 500개의 문장 코퍼스에서 이 방식이 MLM 정확도를 향상시키는지 측정해 보세요.
3. **어려움 (Hard).** 공개 데이터셋의 10,000개 문장을 사용하여 아주 작은(2개 계층, `d=64`) BERT를 학습시켜 보세요. SST-2 감성 분석을 위해 `[CLS]` 토큰을 미세 조정(fine-tune)해 보세요. 동일한 파라미터 수를 가진 디코더 전용(decoder-only) 베이스라인과 비교했을 때, 어느 쪽이 더 우수한 성능을 보이나요?

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| MLM | "마스크 언어 모델링(Masked language modeling)" | 학습 신호: 토큰의 15%를 무작위로 `[MASK]`로 교체하고, 원래 토큰을 예측합니다. |
| Bidirectional | "양방향을 본다" | 인코더 어텐션(Encoder attention)에 인과적 마스크(causal mask)가 없음 — 모든 위치가 다른 모든 위치를 참조합니다. |
| `[CLS]` | "풀러 토큰(The pooler token)" | 모든 시퀀스 앞에 추가되는 특수 토큰; 이 토큰의 최종 임베딩이 문장 수준의 표현(representation)으로 사용됩니다. |
| `[SEP]` | "세그먼트 구분자(Segment separator)" | 쌍으로 된 시퀀스(예: 질의/문서, 문장 A/B)를 구분합니다. |
| NSP | "다음 문장 예측(Next sentence prediction)" | BERT의 두 번째 사전 학습 작업; RoBERTa에서는 무용함이 증명되어 2019년 이후 제외되었습니다. |
| Fine-tuning | "태스크에 적응시키기" | 인코더는 대부분 동결(frozen) 상태로 유지하고, 다운스트림 태스크를 위한 작은 헤드(head)를 상단에 학습시킵니다. |
| Cross-encoder | "리랭커(A reranker)" | 질의(query)와 문서(doc)를 모두 입력으로 받아 관련성 점수를 출력하는 BERT입니다. |
| ModernBERT | "2024년형 리프레시" | RoPE, RMSNorm, GeGLU, 로컬/글로벌 어텐션 교차 적용, 8K 컨텍스트를 갖추도록 재설계된 인코더입니다. |

## 추가 읽을거리 (Further Reading)

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 원본 논문.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERT를 올바르게 학습하는 방법; NSP 제거.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 동일한 연산량 대비 교체된 토큰 탐지(replaced-token detection)가 MLM보다 우수함.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 논문.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 표준 인코더 참조 구현.
