# 기계 번역 (Machine Translation)

> 번역은 지난 30년 동안 NLP 연구를 지탱해 온 과제이며, 지금도 그러합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 문제 (The Problem)

모델은 한 언어로 된 문장을 읽고 다른 언어로 된 문장을 생성합니다. 길이는 제각각이며, 어순도 다릅니다. 일부 소스 단어는 여러 개의 타겟 단어와 매핑되거나 그 반대의 경우도 발생합니다. 관용구는 일대일 매핑이 되지 않습니다. 프랑스어의 "I miss you"는 "tu me manques"인데, 직역하면 "you are lacking to me"입니다. 어떤 단어 수준의 정렬(alignment)도 이를 견뎌내지 못합니다.

기계 번역은 NLP가 인코더-디코더, 어텐션(attention), 트랜스포머(transformer), 그리고 결국 LLM 패러다임 전체를 발명하게 만든 과제입니다. 번역 품질을 측정할 수 있었고 인간과 기계 사이의 격차가 완고했기 때문에 매 단계마다 발전이 가능했습니다.

이 레슨에서는 역사 공부는 건너뛰고 2026년의 실무 파이프라인을 배웁니다: 사전 학습된 다국어 인코더-디코더(NLLB-200 또는 mBART), 서브워드 토큰화(subword tokenization), 빔 서치(beam search), BLEU 및 chrF 평가, 그리고 여전히 프로덕션 환경에서 미처 발견되지 못한 채 배포되는 몇 가지 실패 사례들을 다룹니다.

## 개념 (The Concept)

![MT 파이프라인: 토큰화 → 인코딩 → 어텐션을 이용한 디코딩 → 디토큰화](../assets/mt-pipeline.svg)

현대적인 MT는 병렬 텍스트로 학습된 트랜스포머 인코더-디코더입니다. 인코더는 해당 언어의 토큰화 방식으로 소스를 읽습니다. 디코더는 크로스 어텐션(cross-attention, 레슨 10 참고)을 통해 인코더의 출력을 사용하여 타겟을 한 번에 하나의 서브워드씩 생성합니다. 디코딩 시에는 탐욕적 디코딩(greedy-decoding)의 함정을 피하기 위해 빔 서치를 사용합니다. 출력은 디토큰화(detokenized)되고, 대소문자가 복원(detruecased)되며, 참조 문장(reference)과 비교하여 점수가 매겨집니다.

세 가지 운영상의 선택이 실제 MT 품질을 결정합니다.

- **토크나이저(Tokenizer).** 혼합 언어 코퍼스로 학습된 SentencePiece BPE입니다. 언어 간 공유된 어휘(vocabulary)는 NLLB에서 제로샷(zero-shot) 쌍을 가능하게 하는 핵심입니다.
- **모델 크기(Model size).** NLLB-200 distilled 600M은 노트북에서도 돌아갑니다. NLLB-200 3.3B는 공개된 프로덕션 기본 모델입니다. 54.5B는 연구용 한계치입니다.
- **디코딩(Decoding).** 일반적인 콘텐츠에는 빔 너비(beam width) 4-5를 사용합니다. 너무 짧은 출력을 방지하기 위해 길이 페널티(length penalty)를 적용합니다. 용어 일관성이 필요한 경우 제약 디코딩(constrained decoding)을 사용합니다.

```figure
seq2seq-alignment
```

## 구현하기 (Build It)

### 1단계: 사전 학습된 MT 호출

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

여기서 세 가지가 중요합니다. `src_lang`은 토크나이저에게 어떤 스크립트와 세그멘테이션을 적용할지 알려줍니다. `forced_bos_token_id`는 디코더에게 어떤 언어를 생성할지 알려줍니다. 둘 다 NLLB 전용 트릭이며, mBART와 M2M-100은 자체적인 관례를 사용하므로 서로 호환되지 않습니다.

### 2단계: BLEU 및 chrF

BLEU는 출력과 참조 문장 사이의 n-gram 중첩도를 측정합니다. 네 가지 참조 n-gram 크기(1-4), 정밀도(precision)의 기하 평균, 너무 짧은 출력을 위한 간결성 페널티(brevity penalty)를 사용합니다. 점수는 [0, 100] 범위입니다. 흔히 사용되지만 해석하기는 까다롭습니다. BLEU 30은 "사용 가능", 40은 "좋음", 50은 "매우 뛰어남" 수준이며, 1 BLEU 미만의 차이는 노이즈로 간주합니다.

chrF는 문자 수준의 F-score를 측정합니다. BLEU가 매칭을 과소평가하는 형태론적으로 풍부한 언어(morphologically rich languages)에 더 민감합니다. 종종 BLEU와 함께 보고됩니다.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

항상 `sacrebleu`를 사용하세요. 토큰화를 정규화하여 논문 간에 점수를 비교할 수 있게 해줍니다. 직접 BLEU 계산 방식을 만드는 것은 오해의 소지가 있는 벤치마크를 만드는 지름길입니다.

### 3계층 평가 체계 (2026)

현대적인 MT 평가는 세 가지 상호 보완적인 지표군을 사용합니다. 최소 두 가지를 함께 사용하세요.

- **휴리스틱(Heuristic)** (BLEU, chrF). 빠르고 참조 기반이며 해석이 가능하지만, 의역(paraphrase)에는 민감하지 않습니다. 기존 방식과의 비교나 회귀(regression) 감지용으로 사용하세요.
- **학습 기반(Learned)** (COMET, BLEURT, BERTScore). 인간의 판단을 바탕으로 학습된 신경망 모델로, 번역의 의미적 유사성을 소스 및 참조 문장과 비교합니다. COMET은 2023년 이후 MT 연구와 가장 높은 상관관계를 보이며, 품질이 중요한 2026년 프로덕션의 기본 방식입니다.
- **LLM-as-judge** (참조 문장 없음). 대형 모델에 프롬프트를 주어 유창성, 적절성, 어조, 문화적 적절성을 기준으로 번역을 평가하게 합니다. 루브릭(rubric)이 잘 설계된 경우 GPT-4-as-judge는 인간의 판단과 약 80% 일치합니다. 참조 문장이 없는 개방형 콘텐츠에 사용하세요.

실무적인 2026년 스택: BLEU와 chrF를 위한 `sacrebleu`, COMET을 위한 `unbabel-comet`, 그리고 최종 사용자 피드백을 위한 프롬프트 기반 LLM을 사용합니다. 프로덕션 데이터에 적용하기 전에 모든 지표를 50~100개의 인간 라벨링 예시로 캘리브레이션(calibrate)하세요.

참조 문장이 없는 지표(COMET-QE, BLEURT-QE, LLM-as-judge)를 사용하면 참조 문장 없이도 번역을 평가할 수 있으며, 이는 참조 번역이 존재하지 않는 롱테일(long-tail) 언어 쌍에서 중요합니다.

### 3단계: 프로덕션에서 발생하는 문제

위의 실무 파이프라인은 80%의 확률로 유창하게 번역되지만, 나머지 20%는 조용히 실패합니다. 주요 실패 사례는 다음과 같습니다.

- **환각(Hallucination).** 모델이 소스에 없는 내용을 지어냅니다. 생소한 도메인 어휘에서 흔히 발생합니다. 증상: 출력은 유창하지만 소스에 명시되지 않은 사실을 주장합니다. 완화 방법: 도메인 용어에 대한 제약 디코딩, 규제 대상 콘텐츠에 대한 인간 검토, 입력보다 훨씬 긴 출력 모니터링.
- **대상 이탈 생성(Off-target generation).** 모델이 잘못된 언어로 번역합니다. NLLB는 희귀 언어 쌍에서 이 현상이 놀라울 정도로 자주 발생합니다. 완화 방법: `forced_bos_token_id`를 확인하고, 항상 출력에 대해 언어 ID 모델 체크를 수행하며 디코딩하세요.
- **용어 드리프트(Terminology drift).** 문서 1에서는 "Sign up"이 "s'inscrire"가 되고, 문서 2에서는 "créer un compte"가 됩니다. UI 텍스트와 사용자 대상 문자열의 경우, 원시 품질보다 일관성이 더 중요합니다. 완화 방법: 용어집 제약 디코딩(glossary-constrained decoding) 또는 사후 편집 사전(post-edit dictionary) 사용.
- **격식 불일치(Formality mismatch).** 프랑스어의 "tu" 대 "vous", 일본어의 경어 수준 등이 해당됩니다. 모델은 학습 데이터에서 더 흔했던 형태를 선택합니다. 고객 대상 콘텐츠의 경우 이는 대개 잘못된 선택입니다. 완화 방법: 모델이 지원한다면 격식 토큰으로 프롬프트 접두사를 붙이거나, 격식체 전용 코퍼스로 소형 모델을 미세 조정(fine-tune)하세요.
- **짧은 입력에 대한 길이 폭발.** 매우 짧은 입력 문장은 소스 토큰이 약 5개 미만일 때 길이 페널티가 급격히 떨어지기 때문에 종종 너무 긴 번역을 생성합니다. 완화 방법: 소스 길이에 비례하는 엄격한 최대 길이 제한(hard max-length cap)을 설정하세요.

### 4단계: 도메인 미세 조정 (Fine-tuning)

사전 학습된 모델은 범용적입니다. 법률, 의료 또는 게임 대화 번역은 도메인 병렬 데이터로 미세 조정할 때 눈에 띄는 이점을 얻을 수 있습니다. 방법은 복잡하지 않습니다.

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

수십만 개의 노이즈 섞인 웹 스크래핑 데이터보다 수천 개의 고품질 병렬 예시가 훨씬 낫습니다. 학습 데이터의 품질은 프로덕션에서 활용할 수 있는 가장 큰 레버(lever)입니다.

## 사용하기 (Use It)

MT를 위한 2026년 프로덕션 스택:

| 사용 사례 | 권장 시작점 |
|---------|---------------------------|
| 모든 언어 간, 200개 언어 | `facebook/nllb-200-distilled-600M` (노트북) 또는 `nllb-200-3.3B` (프로덕션) |
| 영어 중심, 고품질, 50개 언어 | `facebook/mbart-large-50-many-to-many-mmt` |
| 짧은 실행, 저렴한 추론, 영-불/독/스 | Helsinki-NLP / Marian 모델 |
| 지연 시간에 민감한 브라우저 측 | ONNX-quantized Marian (~50 MB) |
| 최대 품질, 비용 지불 의사 있음 | GPT-4 / Claude / Gemini (번역 프롬프트 사용) |

2026년 현재, LLM은 여러 언어 쌍, 특히 관용적 콘텐츠와 긴 컨텍스트에서 전문 MT 모델보다 성능이 뛰어납니다. 트레이드오프는 토큰당 비용과 지연 시간입니다. 처리량(throughput)보다 컨텍스트 길이, 스타일 일관성 또는 프롬프팅을 통한 도메인 적응이 더 중요하다면 LLM을 선택하세요.

## 배포하기 (Ship It)

`outputs/skill-mt-evaluator.md`로 저장하세요:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 연습 문제 (Exercises)

1. **쉬움.** `nllb-200-distilled-600M`을 사용하여 영어 문단 5문장을 프랑스어로 번역한 뒤 다시 영어로 번역해 보세요. 왕복 번역이 원문과 얼마나 유사한지 측정하세요. 단어 선택의 변화는 있더라도 의미는 보존되는 것을 확인할 수 있을 것입니다.
2. **중간.** `fasttext lid.176` 또는 `langdetect`를 사용하여 번역 출력에 대한 언어 ID 체크를 구현해 보세요. 대상 이탈 생성이 반환되기 전에 감지되도록 MT 호출에 통합하세요.
3. **어려움.** 원하는 도메인의 5,000쌍 코퍼스로 `nllb-200-distilled-600M`을 미세 조정해 보세요. 미세 조정 전후의 홀드아웃(held-out) 세트에 대한 BLEU를 측정하세요. 어떤 종류의 문장이 개선되었고 어떤 문장이 퇴보했는지 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 내용 | 실제 의미 |
|------|-----------------|-----------------------|
| BLEU | 번역 점수 | 간결성 페널티를 적용한 n-gram 정밀도. [0, 100]. |
| chrF | 문자 F-score | 문자 수준의 F-score. 형태론적으로 풍부한 언어에 더 민감함. |
| NMT | 신경망 MT | 병렬 텍스트로 학습된 트랜스포머 인코더-디코더. 2017년 이후의 기본 방식. |
| NLLB | No Language Left Behind | Meta의 200개 언어 MT 모델 제품군. |
| Constrained decoding | 제약 디코딩 | 출력에 특정 토큰이나 n-gram이 나타나거나 나타나지 않도록 강제함. |
| Hallucination | 환각 | 소스에 의해 뒷받침되지 않는 모델 출력. |

## 추가 읽을거리 (Further Reading)

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 논문.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — `sacrebleu`가 BLEU를 보고하는 유일하게 올바른 방법인 이유.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 논문.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 실무적인 미세 조정 가이드.
