# 다국어 NLP

> 하나의 모델, 100개 이상의 언어, 대부분에 대한 학습 데이터는 없습니다. 언어 간 전이는 2020년대 실용적 기적입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 04강 (GloVe, FastText, 하위 단어), 5단계 · 11강 (기계 번역)
**시간:** 약 45분

## 문제점

영어에는 수십억 개의 레이블이 지정된 예제가 있습니다. 우르두어에는 수천 개가 있습니다. 마이틸리어에는 거의 없습니다. 전 세계 청중을 위한 실용적인 NLP 시스템은 작업별 학습 데이터가 존재하지 않는 언어의 롱테일에서도 작동해야 합니다.

다국어 모델은 여러 언어로 동시에 하나의 모델을 학습하여 이 문제를 해결합니다. 공유된 표현은 모델이 고자원 언어에서 학습한 기술을 저자원 언어로 전이할 수 있게 합니다. 영어 감정 분석으로 모델을 미세 조정하면, 우르두어에 대해开箱(开箱) 상태로 놀라울 정도로 좋은 감정 예측을 생성합니다. 이것이 제로샷 언어 간 전이며, NLP가 세계에 배포되는 방식을 재편했습니다.

이 강의는 다국어 작업에 새로운 팀이 자주 실수하는 결정, 즉 전이를 위한 소스 언어 선택과 함께, 트레이드오프, 표준 모델, 그리고 그 결정을 명시합니다.

## 개념

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**공유 어휘.** 다국어 모델은 모든 대상 언어의 텍스트로 학습된 SentencePiece 또는 WordPiece 토크나이저를 사용합니다. 어휘는 공유됩니다: 동일한 하위 단어 단위는 관련 언어 간에 동일한 형태소를 나타냅니다. `anti-`는 영어와 이탈리아어에서 동일한 토큰을 받습니다.

**공유 표현.** 여러 언어에 걸쳐 마스킹된 언어 모델링으로 사전 학습된 트랜스포머는 다른 언어의 의미적으로 유사한 문장이 유사한 은닉 상태를 생성한다는 것을 학습합니다. mBERT, XLM-R, NLLB 모두 이 특성을 보입니다. 영어의 "cat" 임베딩은 프랑스어의 "chat" 및 스페인어의 "gato" 임베딩 근처에 클러스터링되며, 전체 문장 임베딩도 마찬가지입니다.

**제로샷 전이.** 한 언어(보통 영어)의 레이블이 지정된 데이터로 모델을 미세 조정합니다. 추론 시, 모델이 지원하는 다른 언어로 실행합니다. 대상 언어 레이블이 필요하지 않습니다. 유형론적으로 관련된 언어에 대해서는 결과가 강하고, 먼 언어에 대해서는 약합니다.

**소수 예시 미세 조정.** 대상 언어에 100-500개의 레이블이 지정된 예제를 추가하세요. 분류 작업에서 정확도가 영어 기준선의 95-98%까지 급상승합니다. 이는 다국어 NLP에서 가장 비용 효율적인 단일 수단입니다.

## 모델들

| 모델 | 연도 | 커버리지 | 비고 |
|-------|------|----------|-------|
| mBERT | 2018 | 104개 언어 | Wikipedia에서 학습. 최초의 실용적인 다국어 LM. 저자원 언어에 약함. |
| XLM-R | 2019 | 100개 언어 | CommonCrawl에서 학습 (Wikipedia보다 훨씬 큼). 교차 언어 기준선을 설정. Base 270M, Large 550M. |
| XLM-V | 2023 | 100개 언어 | 1M 토큰 어휘를 가진 XLM-R (vs 250k). 저자원 언어에 더 좋음. |
| mT5 | 2020 | 101개 언어 | 다국어 생성을 위한 T5 아키텍처. |
| NLLB-200 | 2022 | 200개 언어 | Meta의 번역 모델; 55개 저자원 언어 포함. |
| BLOOM | 2022 | 46개 언어 + 13개 프로그래밍 | 다국어로 학습된 오픈 176B LLM. |
| Aya-23 | 2024 | 23개 언어 | Cohere의 다국어 LLM. 아랍어, 힌디어, 스와힐리어에 강함. |

사용 사례에 따라 선택하세요. 분류는 XLM-R-base가 합리적인 기본값으로 잘 작동합니다. 생성 작업은 번역 대 오픈 생성에 따라 mT5 또는 NLLB가 필요합니다. LLM 스타일 작업은 명시적인 다국어 프롬프트를 사용하여 Aya-23 또는 Claude와 짝을 이룹니다.

## 소스 언어 결정 (2026 연구)

대부분의 팀은 미세 조정 소스로 영어를 기본값으로 설정합니다. 최근 연구 (2026)는 이것이 종종 틀렸음을 보여줍니다.

언어 유사성은 원시 코퍼스 크기보다 전이 품질을 더 잘 예측합니다. 슬라브어 대상의 경우, 독일어 또는 러시아어가 영어보다 종종 더 잘 작동합니다. 인도어 대상의 경우, 힌디어가 영어보다 종종 더 잘 작동합니다. **qWALS** 유사성 지표 (2026, World Atlas of Language Structures 기능 기반)는 이를 정량화합니다. **LANGRANK** (Lin et al., ACL 2019)는 언어적 유사성, 코퍼스 크기 및 유전적 관련성의 조합으로부터 후보 소스 언어를 순위를 매기는 별도의 초기 방법입니다.

실용적인 규칙: 대상 언어가 유형론적으로 가까운 고자원 친족 언어가 있다면, 먼저 그 언어로 미세 조정을 시도한 후 영어 미세 조정과 비교해 보세요.

```figure
n5-crosslingual-bridge
```

## 구현하기

### 1단계: 제로샷 교차 언어 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

하나의 모델, 세 개의 언어, 동일한 API. NLI 데이터로 학습한 XLM-R은 함의(trick)를 통해 분류로 잘 전이됩니다.

### 2단계: 다국어 임베딩 공간

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

번역은 임베딩 공간에서 가깝게 위치합니다. 다른 영어 문장은 더 멀리 위치합니다. 이것이 다국어 검색, 클러스터링 및 유사성이 작동하는 방식입니다.

### 3단계: 소수 예시 미세 조정 전략

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

100-500개의 대상 언어 예시에는 `num_train_epochs=5`와 `learning_rate=2e-5`가 안전한 기본값입니다. 더 높은 학습률은 다국어 정렬을 붕괴시켜 영어 전용 모델을 만들게 됩니다.

## 실제로 작동하는 평가

- **보유 세트에 대한 언어별 정확도.** 집계된 값이 아닙니다. 집계는 롱 테일(long tail)을 숨깁니다.
- **단일 언어 기준선과 벤치마크.** 충분한 데이터가 있는 언어의 경우, 처음부터 학습한 단일 언어 모델이 다국어 모델을 능가하는 경우가 있습니다. 테스트해 보세요.
- **엔티티 수준 테스트.** 대상 언어의 명명된 엔티티. 다국어 모델은 라틴 문자에서 먼 스크립트에 대해 토큰화가 약한 경우가 많습니다.
- **다국어 일관성.** 두 언어에서 동일한 의미는 동일한 예측을 산출해야 합니다. 격차를 측정하세요.

## 사용하기

2026년 스택:

| 작업 | 권장 사항 |
|-----|-------------|
| 분류, 100개 언어 | XLM-R-base (~270M) 미세 조정 |
| 제로샷 텍스트 분류 | `joeddav/xlm-roberta-large-xnli` |
| 다국어 문장 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 번역, 200개 언어 | `facebook/nllb-200-distilled-600M` (11강 참조) |
| 생성형 다국어 | Claude, GPT-4, Aya-23, mT5-XXL |
| 저자원 언어 NLP | XLM-V 또는 관련 고자원 언어에 대한 도메인 특화 미세 조정 |

성능이 중요하면 항상 대상 언어의 미세 조정 예산을 확보하세요. 제로샷은 시작점일 뿐, 최종 답이 아닙니다.

### 토큰화 세금 (저자원 언어에서 발생하는 문제)

다국어 모델은 모든 언어에 하나의 토크나이저를 공유합니다. 그 어휘는 영어, 프랑스어, 스페인어, 중국어, 독일어가 지배하는 코퍼스로 학습됩니다. 지배 세트 밖의 모든 언어에 대해 세 가지 세금이 조용히 누적됩니다:

- **생성 비용.** 저자원 언어의 텍스트는 영어보다 단어당 토큰이 훨씬 더 많이 토큰화됩니다. 힌디어 문장은 동등한 영어 문장 대비 토큰이 3~5배 더 필요할 수 있습니다. 이 3~5배는 컨텍스트 윈도우, 학습 효율성, 지연 시간을 소모합니다.
- **변형 복원 비용.** 모든 오타, 악센트 변형, 유니코드 정규화 불일치, 대소문자 변형은 임베딩 공간에서 콜드 스타트와 관련된 시퀀스가 됩니다. 모델은 원어민이 당연하게 여기는 철학적 대응 관계를 학습할 수 없습니다.
- **용량 유출 비용.** 비용 01강 2는 컨텍스트 위치, 레이어 깊이, 임베딩 차원을 소모합니다. 실제 추론에 남는 용량은 동일한 모델에서 고자원 언어가 얻는 용량보다 체계적으로 더 작습니다.

실제 증상: 모델은 힌디어에서 정상적으로 학습되고, 손실 곡선은 정상적으로 보이며, 평가 퍼플렉시티는 합리적으로 보이지만, 프로덕션 출력은 미묘하게 잘못됩니다. 문장 중간에서 형태론이 붕괴됩니다. 희귀한 굴절은 복원할 수 없습니다. **고장난 토크나이저를 데이터 스케일링으로 해결할 수 없습니다.**

완화책: 대상 언어에 대한 커버리지가 좋은 토크나이저를 선택하세요 (XLM-V의 100만 토큰 어휘는 직접적인 해결책입니다); 학습 전에 홀드아웃 대상 텍스트에서 토큰화 생성 비용을 검증하세요; 진정한 롱테일 스크립트에는 바이트 수준 폴백 (SentencePiece `byte_fallback=True`, GPT-2 스타일 바이트 수준 BPE)을 사용하여 OOV가 발생하지 않도록 하세요.

## 출시하기

`outputs/skill-multilingual-picker.md`로 저장하세요:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 연습 문제

1. **쉬움.** 영어, 프랑스어, 힌디어, 아랍어에서 언어당 10문장에 대해 제로샷 분류 파이프라인을 실행하세요. 각 언어의 정확도를 보고하세요. 프랑스어는 강하고, 힌디어는 적당하며, 아랍어는 변동성이 큰 결과를 볼 수 있어야 합니다.
2. **중간.** `paraphrase-multilingual-MiniLM-L12-v2`를 사용하여 작은 혼합 언어 코퍼스에 대해 언어 간 검색기를 구축하세요. 영어로 쿼리하고, 모든 언어의 문서를 검색하세요. recall@5를 측정하세요.
3. **어려움.** 힌디어 분류 작업에 대해 영어 소스 및 힌디어 소스 미세 조정을 비교하세요. 두 가지 환경 모두에서 소수 예시 미세 조정을 위해 500개의 대상 언어 예제를 사용하세요. 어떤 소스가 더 나은 힌디어 정확도를 생산하는지, 그리고 얼마나 더 나은지 보고하세요. 이는 LANGRANK 논문의 축소판입니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 다국어 모델 | 하나의 모델, 여러 언어 | 언어 간 공유 어휘 및 매개변수(Parameter). |
| 언어 간 전이 | 한 언어로 학습, 다른 언어로 실행 | 소스 언어로 미세 조정(Fine-tuning)하고, 타겟 언어 레이블 없이 타겟 언어로 평가(Evaluation (Eval)). |
| 제로샷(Zero-Shot) | 타겟 언어 레이블 없음 | 타겟 언어에 대한 미세 조정(Fine-tuning) 없이 전이. |
| 소수 예시(Few-Shot) | 소량의 타겟 레이블 | 미세 조정(Fine-tuning)에 사용된 100-500개의 타겟 언어 예시. |
| mBERT | 최초의 다국어 LM | 위키백과(Wikipedia)로 사전 학습된 104개 언어의 BERT. |
| XLM-R | 표준 언어 간 기준선 | CommonCrawl로 사전 학습된 100개 언어의 RoBERTa. |
| NLLB | Meta의 200개 언어 MT | No Language Left Behind. 55개 저자원 언어 포함. |

## 추가 읽기

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 논문.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 언어 간 전이 연구 라인을 시작한 분석 논문.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 논문.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Cohere의 다국어 LLM인 Aya.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 소스 언어 논문.
