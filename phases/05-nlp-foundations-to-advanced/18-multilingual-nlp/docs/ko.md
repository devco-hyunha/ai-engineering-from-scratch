# 다국어 NLP (Multilingual NLP)

> 하나의 모델, 100개 이상의 언어, 그중 대부분은 학습 데이터가 전무합니다. 교차 언어 전이(Cross-lingual transfer)는 2020년대가 낳은 실질적인 기적입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 문제 (The Problem)

영어에는 수십억 개의 라벨링된 예시가 존재합니다. 우르두어(Urdu)는 수천 개에 불과하며, 마이틸리어(Maithili)는 거의 전무합니다. 전 세계 사용자를 대상으로 하는 실용적인 NLP 시스템은 작업별 학습 데이터가 존재하지 않는 롱테일(long tail) 언어에서도 작동해야 합니다.

다국어 모델(Multilingual models)은 여러 언어를 동시에 하나의 모델로 학습하여 이 문제를 해결합니다. 공유된 표현(shared representation) 덕분에 모델은 고자원(high-resource) 언어에서 학습한 능력을 저자원(low-resource) 언어로 전이할 수 있습니다. 모델을 영어 감성 분석 데이터로 파인튜닝하면, 별도의 작업 없이도 우르두어에 대해 놀라울 정도로 훌륭한 감성 예측을 수행합니다. 이것이 바로 제로샷 교차 언어 전이(zero-shot cross-lingual transfer)이며, NLP가 전 세계에 서비스되는 방식을 완전히 바꾸어 놓았습니다.

이 레슨에서는 트레이드오프(tradeoffs), 표준 모델, 그리고 다국어 작업을 처음 접하는 팀들이 흔히 겪는 한 가지 결정적 실수인 '전이를 위한 소스 언어 선택'을 다룹니다.

## 개념 (The Concept)

![공유 다국어 임베딩 공간을 통한 교차 언어 전이](../assets/multilingual.svg)

**공유 어휘집(Shared vocabulary).** 다국어 모델은 모든 타깃 언어의 텍스트로 학습된 SentencePiece 또는 WordPiece 토크나이저를 사용합니다. 어휘집이 공유되므로, 동일한 서브워드(subword) 단위가 관련 언어 전반에서 동일한 형태소(morpheme)를 나타냅니다. 예를 들어 영어와 이탈리아어의 `anti-`는 동일한 토큰을 부여받습니다.

**공유 표현(Shared representation).** 여러 언어에 걸쳐 마스크드 언어 모델링(masked language modeling)으로 사전 학습된 트랜스포머는 서로 다른 언어의 의미적으로 유사한 문장이 유사한 은닉 상태(hidden states)를 생성하도록 학습합니다. mBERT, XLM-R, NLLB 모두 이러한 특성을 보입니다. 영어의 "cat" 임베딩은 프랑스어 "chat", 스페인어 "gato" 근처에 군집을 이루며, 전체 문장 임베딩도 마찬가지입니다.

**제로샷 전이(Zero-shot transfer).** 한 언어(대개 영어)의 라벨링된 데이터로 모델을 파인튜닝합니다. 추론 시점에는 모델이 지원하는 다른 어떤 언어로든 실행할 수 있습니다. 타깃 언어의 라벨은 전혀 필요하지 않습니다. 언어유형학적으로(typologically) 가까운 언어 간에는 강력한 성능을 보이지만, 거리가 먼 언어 사이에서는 성능이 떨어집니다.

**퓨샷 파인튜닝(Few-shot fine-tuning).** 타깃 언어로 된 100~500개의 라벨링된 예시를 추가합니다. 분류 작업에서 정확도가 영어 베이스라인의 95~98% 수준으로 급상승합니다. 이는 다국어 NLP에서 가장 비용 효율적인 핵심 레버입니다.

## 모델 (The models)

| 모델 | 출시 연도 | 지원 언어 수 | 특징 및 비고 |
|-------|------|----------|-------|
| mBERT | 2018 | 104개 언어 | Wikipedia로 학습된 최초의 실용적인 다국어 LM입니다. 저자원 언어 성능이 취약합니다. |
| XLM-R | 2019 | 100개 언어 | Wikipedia보다 훨씬 방대한 CommonCrawl로 학습되었습니다. 교차 언어 베이스라인을 정립했습니다. (Base 270M, Large 550M) |
| XLM-V | 2023 | 100개 언어 | 100만(1M) 토큰 어휘집(기존 25만 개 대비)을 갖춘 XLM-R입니다. 저자원 언어 성능이 크게 향상되었습니다. |
| mT5 | 2020 | 101개 언어 | 다국어 텍스트 생성을 위한 T5 아키텍처입니다. |
| NLLB-200 | 2022 | 200개 언어 | Meta의 번역 모델로, 55개의 저자원 언어를 포함합니다. |
| BLOOM | 2022 | 46개 언어 + 13개 프로그래밍 언어 | 다국어로 학습된 오픈 소스 176B LLM입니다. |
| Aya-23 | 2024 | 23개 언어 | Cohere의 다국어 LLM입니다. 아랍어, 힌디어, 스와힐리어에서 강력한 성능을 보입니다. |

사용 사례에 맞춰 모델을 선택하세요. 분류 작업에는 XLM-R-base가 합리적인 기본값입니다. 생성 작업은 단순 번역인지 개방형 텍스트 생성인지에 따라 mT5 또는 NLLB를 선택합니다. LLM 스타일의 작업에는 명시적인 다국어 프롬프팅을 적용한 Aya-23 또는 Claude를 활용합니다.

## 소스 언어 결정 (The source-language decision) (2026년 연구)

대부분의 팀은 파인튜닝 소스 언어로 영어를 기본 선택합니다. 하지만 최근 연구(2026년)에 따르면 이는 잘못된 선택일 때가 많습니다.

언어 간 유사성(Language similarity)은 원시 코퍼스 크기보다 전이 품질을 훨씬 정확하게 예측합니다. 슬라브어파(Slavic) 타깃의 경우 독일어나 러시아어가 영어보다 나은 결과를 내는 경우가 흔합니다. 인도어파(Indic) 타깃의 경우 힌디어가 종종 영어를 앞섭니다. **qWALS** 유사도 지표(2026년, World Atlas of Language Structures 특징 기반)는 이를 정량화합니다. **LANGRANK** (Lin et al., ACL 2019)는 언어적 유사성, 코퍼스 크기, 계통적 관련성(genetic relatedness)을 종합하여 후보 소스 언어의 순위를 매기는 이전의 대표적 방법론입니다.

실무 지침: 타깃 언어와 언어유형학적으로 가까운 고자원 언어가 있다면, 해당 언어로 먼저 파인튜닝을 시도한 다음 영어 파인튜닝 결과와 비교해 보세요.

```figure
n5-crosslingual-bridge
```

## 직접 구현하기 (Build It)

### 1단계: 제로샷 교차 언어 분류 (Step 1: zero-shot cross-lingual classification)

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

하나의 모델, 세 가지 언어, 동일한 API입니다. NLI 데이터로 학습된 XLM-R은 함의(entailment) 트릭을 통해 텍스트 분류로 훌륭하게 전이됩니다.

### 2단계: 다국어 임베딩 공간 (Step 2: multilingual embedding space)

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

번역문들은 임베딩 공간에서 서로 가깝게 배치됩니다. 의미가 다른 영어 문장은 더 먼 거리에 위치합니다. 이 특성 덕분에 교차 언어 검색, 군집화, 유사도 비교가 가능해집니다.

### 3단계: 퓨샷 파인튜닝 전략 (Step 3: few-shot fine-tuning strategy)

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

타깃 언어 예시가 100~500개인 경우, `num_train_epochs=5`와 `learning_rate=2e-5`가 안전한 기본값입니다. 학습률을 너무 높이면 다국어 정렬이 붕괴되어 영어 전용 모델로 전락하게 됩니다.

## 실질적으로 통하는 평가 (Evaluation that actually works)

- **홀드아웃 세트에 대한 언어별 정확도 측정.** 데이터를 집계하여 평균 내지 마세요. 평균치는 롱테일 언어의 실패를 감춥니다.
- **단일 언어 베이스라인과의 비교 벤치마크.** 데이터가 충분한 언어의 경우, 처음부터 학습한 단일 언어 모델이 다국어 모델을 능가하기도 합니다. 가정을 배제하고 직접 검증하세요.
- **엔티티 수준 테스트.** 타깃 언어의 개체명을 점검하세요. 다국어 모델은 라틴 문자에서 멀리 떨어진 문자 체계에 대해 토큰화 성능이 떨어지는 경우가 흔합니다.
- **교차 언어 일관성 측정.** 두 언어로 된 동일한 의미의 문장은 동일한 예측을 산출해야 합니다. 그 격차를 측정하세요.

## 실전 활용 (Use It)

2026년 추천 스택:

| 작업 | 추천 모델 / 기법 |
|-----|-------------|
| 100개 언어 대상 텍스트 분류 | 파인튜닝된 XLM-R-base (~270M) |
| 제로샷 텍스트 분류 | `joeddav/xlm-roberta-large-xnli` |
| 다국어 문장 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 200개 언어 대상 기계 번역 | `facebook/nllb-200-distilled-600M` (레슨 11 참조) |
| 생성형 다국어 작업 | Claude, GPT-4, Aya-23, mT5-XXL |
| 저자원 언어 NLP | XLM-V 또는 관련 고자원 언어로 파인튜닝된 도메인 특화 모델 |

성능이 중요하다면 타깃 언어에 대한 파인튜닝 예산을 항상 확보하세요. 제로샷은 출발점일 뿐 최종 해답이 아닙니다.

### 토큰화 비용: 저자원 언어에서 발생하는 문제 (The tokenization tax)

다국어 모델은 모든 언어에서 하나의 토크나이저를 공유합니다. 이 어휘집은 영어, 프랑스어, 스페인어, 중국어, 독일어가 주도하는 코퍼스로 학습됩니다. 이 주요 언어 집합에 속하지 않는 언어의 경우 세 가지 '비용(tax)'이 암묵적으로 누적됩니다:

- **토큰 분할률 비용(Fertility tax).** 저자원 언어 텍스트는 영어에 비해 단어당 훨씬 더 많은 토큰으로 분할됩니다. 힌디어 문장은 동등한 영어 문장보다 3~5배 더 많은 토큰이 필요할 수 있습니다. 이 3~5배의 팽창은 컨텍스트 윈도우, 학습 효율성, 추론 지연 시간을 잠식합니다.
- **변이 복구 비용(Variant recovery tax).** 모든 오타, 발음 구별 부호(diacritic) 변이, 유니코드 정규화 불일치, 대소문자 변화가 임베딩 공간에서 무관한 콜드 스타트 시퀀스로 취급됩니다. 모델은 원어민이 당연하게 여기는 철자 대응 관계를 학습하지 못합니다.
- **용량 잠식 비용(Capacity spillover tax).** 비용 1과 비용 2가 컨텍스트 위치, 레이어 깊이, 임베딩 차원을 소모합니다. 실제 추론에 남겨지는 모델 용량은 고자원 언어가 동일한 모델에서 얻는 용량보다 체계적으로 작아집니다.

실제 증상: 힌디어로 모델을 정상 학습시키고, 손실 곡선도 안정적이며, 평가 퍼플렉서티(eval perplexity)도 합리적으로 보이지만 프로덕션 출력물은 미묘하게 틀립니다. 문장 중간에서 형태소가 붕괴되고, 드문 굴절형은 복구되지 않은 채로 남습니다. **망가진 토크나이저는 데이터 규모 확장으로 해결할 수 없습니다.**

완화책: 타깃 언어에 대해 충분한 커버리지를 갖춘 토크나이저를 선택하세요(XLM-V의 100만 토큰 어휘집이 직접적인 해결책입니다). 학습 전 홀드아웃 텍스트에서 토큰 분할률(fertility)을 검증하세요. 극단적인 롱테일 문자 체계의 경우 어떤 토큰도 OOV(어휘집 외 단어)가 되지 않도록 바이트 수준 폴백(SentencePiece `byte_fallback=True`, GPT-2 스타일 바이트 수준 BPE)을 적용하세요.

## 배포하기 (Ship It)

`outputs/skill-multilingual-picker.ko.md`로 저장하세요:

```markdown
---
name: multilingual-picker
description: 다국어 NLP 작업을 위한 소스 언어, 대상 모델 및 평가 계획을 선택합니다.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

요구 사항(타깃 언어 목록, 작업 유형, 언어별 가용 라벨링 데이터)이 주어지면 다음을 출력합니다:

1. 파인튜닝용 소스 언어. 기본값은 영어이며, 타깃 언어와 언어유형학적으로 가까운 고자원 언어가 있다면 LANGRANK 또는 qWALS를 검토합니다.
2. 베이스 모델. XLM-R(분류), mT5(생성), NLLB(번역), Aya-23(생성형 LLM).
3. 퓨샷 예산. 가용한 경우 타깃 언어 예시 100~500개로 시작합니다. 제로샷은 라벨링이 불가능할 때만 사용합니다.
4. 평가 계획. 개별 언어별 정확도(통합 집계 금지), 교차 언어 일관성, 비라틴 문자의 개체명 F1 점수.

개별 언어별 평가 없는 다국어 모델 배포는 거부해야 합니다. 통합 지표는 롱테일 언어의 실패를 은폐합니다. 토큰화 커버리지가 낮은 문자 체계(암하라어, 티그리냐어, 다수의 아프리카 언어 등)는 바이트 폴백 지원 모델(SentencePiece `byte_fallback=True` 또는 GPT-2 스타일의 바이트 수준 토크나이저)이 필요함을 명시합니다.
```

## 연습 문제 (Exercises)

1. **초급.** 영어, 프랑스어, 힌디어, 아랍어에 대해 언어당 10개 문장으로 제로샷 분류 파이프라인을 실행해 보세요. 언어별 정확도를 보고하세요. 프랑스어는 우수하고, 힌디어는 양호하며, 아랍어는 가변적인 결과를 확인할 수 있습니다.
2. **중급.** `paraphrase-multilingual-MiniLM-L12-v2`를 활용하여 소규모 다국어 혼합 코퍼스를 대상으로 교차 언어 검색기(cross-lingual retriever)를 구축해 보세요. 영어로 질의하고 다양한 언어로 문서를 검색하여 recall@5를 측정하세요.
3. **고급.** 힌디어 분류 작업에서 영어 소스와 힌디어 소스 파인튜닝을 비교해 보세요. 두 체계 모두 500개의 타깃 언어 예시를 사용해 퓨샷 파인튜닝을 수행합니다. 어느 소스가 힌디어 정확도를 더 높이는지, 그 차이가 얼마인지 보고하세요. 이것이 바로 LANGRANK 논문의 축소판 실험입니다.

## 핵심 용어 (Key Terms)

| 용어 | 일반적인 인식 | 실제 기술적 의미 |
|------|-----------------|-----------------------|
| 다국어 모델 (Multilingual model) | 하나의 모델로 여러 언어를 지원하는 것 | 여러 언어 전반에 걸쳐 공유 어휘집과 가중치 파라미터를 갖춘 모델. |
| 교차 언어 전이 (Cross-lingual transfer) | 한 언어로 학습해 다른 언어에서 실행하는 것 | 타깃 언어 라벨 없이 소스 언어로 파인튜닝한 후 타깃 언어에서 평가하는 기법. |
| 제로샷 (Zero-shot) | 타깃 언어 라벨이 전혀 없는 상태 | 타깃 언어에 대한 파인튜닝 없이 곧바로 전이하여 추론하는 방식. |
| 퓨샷 (Few-shot) | 소량의 타깃 언어 라벨 활용 | 100~500개의 타깃 언어 예시를 사용하여 파인튜닝하는 방식. |
| mBERT | 최초의 실용 다국어 LM | Wikipedia로 사전 학습된 104개 언어 지원 BERT 모델. |
| XLM-R | 표준 교차 언어 베이스라인 | CommonCrawl로 사전 학습된 100개 언어 지원 RoBERTa 모델. |
| NLLB | Meta의 200개 언어 지원 MT | 'No Language Left Behind'. 55개의 저자원 언어를 포함한 기계 번역 모델. |

## 추가 참고 자료 (Further Reading)

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 원 논문.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 교차 언어 전이 연구의 포문을 연 분석 논문.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 논문.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Cohere의 다국어 오픈소스 LLM Aya 모델 논문.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS 및 LANGRANK 소스 언어 선정 연구 논문.
