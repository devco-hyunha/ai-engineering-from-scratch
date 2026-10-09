# 기계 번역

> 번역은 30년간 NLP 연구를 지원해 왔으며, 현재도 계속 지원하고 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 10강 (어텐션 메커니즘), 5단계 · 04강 (GloVe, FastText, 서브워드)
**시간:** 약 75분

## 문제점

모델은 한 언어로 된 문장을 읽고 다른 언어로 문장을 생성합니다. 길이가 다르고, 단어 순서가 다릅니다. 일부 원문 단어는 여러 대상 단어로 매핑되고, 그 반대도 가능합니다. 관용구는 일대일 매핑을 거부합니다. 프랑스어로 "I miss you"는 "tu me manques"이며, 문자 그대로는 "you are lacking to me"입니다. 단어 수준 정렬은 이를 견디지 못합니다.

기계 번역은 NLP가 인코더-디코더, 어텐션, 트랜스포머, 그리고 궁극적으로 전체 LLM 패러다임을 발명하도록 강제했습니다. 모든 진전은 번역 품질이 측정 가능했고, 인간과 기계 사이의 격차가 끈질겼기 때문에 이루어졌습니다.

이 강의는 역사적 배경을 생략하고 2026년의 실제 파이프라인을 가르칩니다: 사전 학습된 다국어 인코더-디코더 (NLLB-200 또는 mBART), 서브워드 토큰화, 빔 검색, BLEU 및 chrF 평가, 그리고 여전히 잡히지 않은 채 프로덕션으로 출시되는 몇 가지 실패 모드입니다.

## 개념

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

현대 MT는 병렬 텍스트로 학습된 트랜스포머 인코더-디코더입니다. 인코더는 원문 언어의 토큰화로 원문을 읽습니다. 디코더는 인코더의 출력에 대한 교차 어텐션 (10강)을 사용하여 한 번에 하나의 서브워드를 생성합니다. 디코딩은 탐욕 디코딩 함정을 피하기 위해 빔 검색을 사용합니다. 출력은 디토큰화되고, 대소문자 복원되며, 참조와 비교하여 점수화됩니다.

세 가지 운영적 선택이 실제 MT 품질을 결정합니다.

- **토크나이저.** 혼합 언어 코퍼스로 학습된 SentencePiece BPE. 언어 간 공유 어휘는 NLLB에서 제로샷 쌍을 가능하게 합니다.
- **모델 크기.** NLLB-200 증류 600M은 노트북에서 실행됩니다. NLLB-200 3.3B는 공개된 프로덕션 기본값입니다. 54.5B는 연구 상한선입니다.
- **디코딩.** 일반 콘텐츠에는 빔 폭 4-5를 사용합니다. 너무 짧은 출력을 피하기 위해 길이 페널티를 적용합니다. 용어 일관성이 필요할 때는 제약 디코딩을 사용합니다.

```figure
seq2seq-alignment
```

## 구현하기

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

여기서 세 가지가 중요합니다. `src_lang`은 토크나이저에 어떤 스크립트와 분할을 적용할지 알려줍니다. `forced_bos_token_id`은 디코더에 어떤 언어를 생성할지 알려줍니다. 둘 다 NLLB 전용 트릭입니다. mBART와 M2M-100은 자체 규약을 사용하며, 서로 호환되지 않습니다.

### 2단계: BLEU와 chrF

BLEU는 출력과 참조 간의 n-gram 중복을 측정합니다. 네 가지 참조 n-gram 크기(1-4), 정밀도의 기하 평균, 너무 짧은 출력에 대한 간결성 페널티(간결성 페널티)를 사용합니다. 점수는 [0, 100] 범위입니다. 널리 사용됩니다. 해석하기 어렵습니다. 30 BLEU는 "사용 가능", 40은 "좋음", 50은 "탁월"입니다. 1 BLEU 미만의 차이는 잡음입니다.

chrF는 문자 단위 F-점수를 측정합니다. BLEU가 매칭을 과소평가하는 형태론적으로 풍부한 언어에 더 민감합니다. BLEU와 함께 보고되는 경우가 많습니다.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

항상 `sacrebleu`을 사용하세요. 이는 토큰화를 정규화하여 논문 간 점수를 비교 가능하게 합니다. 자체 BLEU 계산은 오해의 소지가 있는 벤치마크가 발생하는 원인입니다.

### 3단계 평가 계층 구조 (2026)

현대 MT 평가는 세 가지 상호 보완적인 지표 계열을 사용합니다. 최소 두 가지를 함께 사용하세요.

- **휴리스틱** (BLEU, chrF). 빠르고, 참조 기반이며, 해석 가능하고, 패러프레이즈에 둔감합니다. 레거시 비교 및 회귀 감지에 사용하세요.
- **학습된 지표** (COMET, BLEURT, BERTScore). 인간 판단으로 학습된 신경망 모델로, 번역의 의미적 유사성을 원문 및 참조와 비교합니다. COMET는 2023년 이후 MT 연구와 가장 높은 연관성을 가지며, 품질이 중요한 2026년 생산 환경의 기본값입니다.
- **LLM-as-judge** (참조 없음). 대규모 모델에 프롬프트하여 유창성, 적절성, 어조, 문화적 적절성에 대해 번역을 점수화합니다. 평가 기준이 잘 설계된 경우 GPT-4-as-judge는 인간 일치율의 약 80%를 달성합니다. 참조가 없는 개방형 콘텐츠에 사용하세요.

실용적인 2026 스택: BLEU와 chrF에는 `sacrebleu`, COMET에는 `unbabel-comet`, 최종 인간-facing 신호에는 프롬프트된 LLM을 사용하세요. 생산 데이터에서 신뢰하기 전에 모든 지표를 50-100개의 인간 라벨링된 예제로 보정하세요.

참조 없는 지표(COMET-QE, BLEURT-QE, LLM-as-judge)는 참조 없이 번역을 평가할 수 있게 해주며, 이는 참조 번역이 존재하지 않는 롱테일 언어 쌍에서 중요합니다.

### 3단계: 프로덕션에서 문제가 되는 부분

위에서 설명한 파이프라인은 80%의 경우 유창하게 번역되지만, 나머지 20%는 조용히 실패합니다. 알려진 실패 모드:

- **환각(Hallucination).** 모델이 원문에 없는 내용을 만들어냅니다. 익숙하지 않은 도메인 어휘에서 흔합니다. 증상: 출력은 유창하지만 원문이 언급하지 않은 사실을 주장합니다. 완화책: 도메인 용어에 대한 제약 디코딩(제약 디코딩), 규제 대상 콘텐츠에 대한 인간 검토, 입력보다 출력이 훨씬 긴 경우 모니터링.
- **대상 이탈 생성(대상 이탈 생성).** 모델이 잘못된 언어로 번역합니다. NLLB는 드문 언어 쌍에서 이 문제가 놀라울 정도로 잘 발생합니다. 완화책: `forced_bos_token_id`을 확인하고, 출력에 대해 항상 언어 식별 모델 검사를 수행하며 디코딩하세요.
- **용어 드리프트(Terminology drift).** "Sign up"이 문서 1에서는 "s'inscrire"이 되고, 문서 2에서는 "créer un compte"가 됩니다. UI 텍스트와 사용자-facing 문자열의 경우, 원시적인 품질보다 일관성이 더 중요합니다. 완화책: 용어집 기반 제약 디코딩(glossary-제약 디코딩) 또는 사후 편집 사전(post-edit dictionary).
- **격식 불일치(Formality mismatch).** 프랑스어 "tu"와 "vous", 일본어 존댓말 수준. 모델은 학습 데이터에서 더 흔했던 형태를 선택합니다. 고객-facing 콘텐츠의 경우 이는 보통 잘못된 선택입니다. 완화책: 모델이 지원한다면 격식 토큰(formality token)을 프롬프트 접두어에 포함하거나, 격식 전용 코퍼스로 작은 모델을 미세 조정(fine-tuning)하세요.
- **짧은 입력에서의 길이 폭발(Length explosion on short input).** 매우 짧은 입력 문장은 길이 페널티(length penalty)가 약 5개 소스 토큰 이하에서 급격히 떨어지기 때문에 과도하게 긴 번역을 생성하는 경우가 많습니다. 완화책: 소스 길이에 비례하는 하드 최대 길이 상한(hard max-length cap)을 설정하세요.

### 4단계: 도메인 맞춤 미세 조정(fine-tuning)

사전 학습된 모델은 범용적입니다. 법률, 의료, 게임 대화 번역은 도메인 병렬 데이터에 대한 미세 조정(fine-tuning)을 통해 측정 가능한 이점을 얻습니다. 레시피는 특별하지 않습니다:

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

수천 개의 고품질 병렬 예제가 수십만 개의 잡음 많은 웹 스크래핑 예제보다 낫습니다. 학습 데이터의 품질이 프로덕션 성능을 좌우하는 가장 큰 단일 요인입니다.

## 사용하기

2026년 MT를 위한 프로덕션 스택:

| 사용 사례 | 권장 시작점 |
|---------|---------------------------|
| 모든 언어 간 번역, 200개 언어 | `facebook/nllb-200-distilled-600M` (노트북) 또는 `nllb-200-3.3B` (프로덕션) |
| 영어 중심, 고품질, 50개 언어 | `facebook/mbart-large-50-many-to-many-mmt` |
| 짧은 실행, 저렴한 추론, 영어-프랑스어/독일어/스페인어 | Helsinki-NLP / Marian 모델 |
| 지연이 중요한 브라우저 측 | ONNX 양자화 Marian (~50 MB) |
| 최대 품질, 비용 감수 | GPT-4 / Claude / Gemini 번역 프롬프트 |

2026년 현재, LLM은 여러 언어 쌍에서 전문 MT 모델을 능가하며, 특히 관용적인 콘텐츠와 긴 컨텍스트에서 성능이 우수합니다. 트레이드오프는 토큰당 비용과 지연입니다. 처리량보다 컨텍스트 길이, 스타일 일관성, 프롬프트를 통한 도메인 적응이 더 중요할 때 LLM을 선택하세요.

## 출시하기

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

## 연습 문제

1. **쉬움.** `nllb-200-distilled-600M`를 사용하여 5문장 영어 단락을 프랑스어로 번역하고 다시 영어로 번역하세요. 왕복 번역이 원본과 얼마나 가까운지 측정하세요. 의미 보존과 단어 선택의 편차를 확인할 수 있습니다.
2. **중간.** `fasttext lid.176` 또는 `langdetect`를 사용하여 번역 출력에 언어 식별(ID) 검사를 구현하세요. MT 호출에 통합하여 대상 이탈 생성이 반환되기 전에 감지되도록 하세요.
3. **어려움.** 선택한 5,000 쌍 도메인 코퍼스에 `nllb-200-distilled-600M`를 미세 조정하세요. 미세 조정 전후에 홀드아웃 세트에서 BLEU를 측정하세요. 어떤 종류의 문장이 개선되고 어떤 문장이 퇴보했는지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BLEU | 번역 점수 | 간결성 페널티가 포함된 N-gram 정밀도. [0, 100]. |
| chrF | 문자 F-점수 | 문자 수준 F-점수. 형태론적으로 풍부한 언어에 더 민감합니다. |
| NMT | 신경망 MT | 병렬 텍스트로 학습된 트랜스포머 인코더-디코더. 2017년 이후의 기본값. |
| NLLB | No Language Left Behind | Meta의 200개 언어 MT 모델 계열. |
| 제약 디코딩 | 제어된 출력 | 특정 토큰이나 n-gram이 출력에 나타나도록/나타나지 않도록 강제합니다. |
| 환각(Hallucination) | 발명된 콘텐츠 | 원문에 의해 지지되지 않는 모델 출력. |

## 추가 읽기

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 논문.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — `sacrebleu`가 BLEU를 보고하는 유일한 올바른 방법인 이유.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 논문.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 실용적인 미세 조정(fine-tuning) 절차입니다.
