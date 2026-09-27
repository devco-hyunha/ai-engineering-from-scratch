# 자연어 추론 (Natural Language Inference) — 텍스트 함의 (Textual Entailment)

> "$t$가 $h$를 함의한다($t$ entails $h$)"는 것은 $t$를 읽는 사람이 $h$가 참이라고 결론을 내릴 것임을 의미합니다. NLI는 함의(entailment) / 모순(contradiction) / 중립(neutral)을 예측하는 작업입니다. 겉보기에는 단순해 보일 수 있지만, 실제 서비스 환경(production)에서는 핵심적인 역할을 수행합니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## 문제 (The Problem)

요약기를 만들었습니다. 요약문이 생성되었습니다. 이 요약문에 환각(hallucination)이 포함되지 않았다는 것을 어떻게 알 수 있을까요?

챗봇을 만들었습니다. "예"라고 답변했습니다. 이 답변이 검색된 구절(passage)에 의해 뒷받침되는지 어떻게 알 수 있을까요?

10,000개의 뉴스 기사를 주제별로 분류해야 합니다. 학습용 레이블(label)이 없습니다. 모델을 재사용할 수 있을까요?

이 세 가지 문제는 모두 자연어 추론(Natural Language Inference, NLI) 문제로 귀결됩니다. NLI는 다음과 같이 질문합니다: 전제(premise) `t`와 가설(hypothesis) `h`가 주어졌을 때, `h`가 `t`에 의해 함의(entailed)되는지, 모순(contradicted)되는지, 아니면 중립(neutral, 관련 없음)인지 판단합니다.

- **환각 체크(Hallucination check):** `t` = 소스 문서, `h` = 요약문의 주장. 함의되지 않음 = 환각.
- **근거 기반 QA(Grounded QA):** `t` = 검색된 구절, `h` = 생성된 답변. 함의되지 않음 = 허구(fabrication).
- **제로샷 분류(Zero-shot classification):** `t` = 문서, `h` = 언어화된 레이블 ("이것은 스포츠에 관한 것입니다"). 함의됨 = 예측된 레이블.

하나의 작업으로 세 가지 실무 활용 사례를 해결할 수 있습니다. 이것이 바로 모든 RAG 평가 프레임워크가 내부적으로 NLI 모델을 탑재하고 있는 이유입니다.

## 개념 (The Concept)

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**세 가지 레이블.**

- **함의 (Entailment).** `t` → `h`. "고양이가 매트 위에 있다"는 "고양이가 있다"를 함의합니다.
- **모순 (Contradiction).** `t` → ¬`h`. "고양이가 매트 위에 있다"는 "고양이가 없다"와 모순됩니다.
- **중립 (Neutral).** 어느 쪽으로도 추론할 수 없습니다. "고양이가 매트 위에 있다"는 "고양이가 배고프다"에 대해 중립적입니다.

**논리적 함의(Logical entailment)가 아닙니다.** NLI는 *자연어* 추론(natural language inference)입니다. 즉, 엄격한 논리가 아니라 일반적인 인간 독자가 추론할 법한 내용을 다룹니다. NLI에서는 "존이 개를 산책시켰다"가 "존은 개를 가지고 있다"를 함의하지만, 엄격한 1차 논리(first-order logic)에서는 소유를 공리화해야만 이를 인정합니다.

**데이터셋 (Datasets).**

- **SNLI** (2015). 57만 개의 인간 주석 쌍, 이미지 캡션을 전제로 사용. 좁은 도메인.
- **MultiNLI** (2017). 10개 장르에 걸친 43.3만 개의 쌍. 현재 표준 학습 코퍼스.
- **ANLI** (2019). 적대적 NLI(Adversarial NLI). 기존 모델을 무너뜨리기 위해 설계된 예시를 인간이 직접 작성했습니다. 더 어렵습니다.
- **DocNLI, ConTRoL** (2020–21). 문서 길이의 전제. 멀티홉(multi-hop) 및 장거리 추론을 테스트합니다.

**아키텍처 (The architecture).** 트랜스포머 인코더(BERT, RoBERTa, DeBERTa)가 `[CLS] premise [SEP] hypothesis [SEP]`를 읽습니다. `[CLS]` 표현(representation)은 3-way 소프트맥스(softmax)로 전달됩니다. MNLI로 학습하고, 홀드아웃(held-out) 벤치마크에서 평가하여 분포 내(in-distribution) 쌍에 대해 90% 이상의 정확도를 얻습니다.

**NLI를 통한 제로샷 (Zero-shot via NLI).** 문서와 후보 레이블이 주어지면, 각 레이블을 하나의 가설로 변환합니다("이 텍스트는 스포츠에 관한 것이다"). 각 가설에 대한 함의 확률을 계산하고, 가장 높은 값을 선택합니다. 이것이 Hugging Face의 `zero-shot-classification` 파이프라인의 작동 원리입니다.

```figure
nli-router
```

## 직접 구현해 보기 (Build It)

### 1단계: 사전 학습된 NLI 모델 실행하기 (run a pretrained NLI model)

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 모든 레이블을 반환합니다; 지원 중단된 return_all_scores=True를 대체합니다

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

실제 서비스용(production) NLI를 위해서는 `facebook/bart-large-mnli`와 `microsoft/deberta-v3-large-mnli`가 오픈 소스 기본 모델로 사용됩니다. DeBERTa-v3는 리더보드 상위권을 차지하고 있습니다.

### 2단계: 제로샷 분류 (zero-shot classification)

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

기본 템플릿은 "This example is about {label}."입니다. `hypothesis_template`을 사용하여 커스텀할 수 있습니다. 학습 데이터가 필요하지 않습니다. 파인튜닝(fine-tuning)도 필요하지 않습니다. 별도의 설정 없이 즉시 사용할 수 있습니다.

### 3단계: RAG를 위한 충실도(faithfulness) 검사

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

이것이 RAGAS 충실도(faithfulness)의 핵심입니다. 생성된 답변을 원자적 주장(atomic claims)으로 분할합니다. 각 주장을 검색된 컨텍스트(context)와 대조하여 확인합니다. 함의(entail)되는 비율을 보고합니다.

### 4단계: 직접 구현한 NLI 분류기 (hand-rolled NLI classifier, 개념적)

표준 라이브러리(stdlib)만 사용한 토이 모델은 `code/main.py`를 참조하세요. 전제(premise)와 가설(hypothesis)을 어휘적 중첩(lexical overlap) 및 부정 탐지(negation detection)를 통해 비교합니다. 트랜스포머 모델(transformer models)만큼의 성능은 아니지만, 작업의 형태를 보여줍니다: 두 개의 텍스트가 입력되고, 3가지 레이블이 출력되며, 손실(loss)은 `{entail, contradict, neutral}`에 대한 교차 엔트로피(cross-entropy)로 계산됩니다.

## 주의 사항 (Pitfalls)

- **가설 전용 지름길 (Hypothesis-only shortcuts).** SNLI에서 "not", "nobody", "never"와 같은 단어들이 모순(contradiction)과 상관관계가 있기 때문에, 모델은 가설만으로도 약 60%의 정확도로 레이블을 예측할 수 있습니다. 이는 레이블 누출(label leakage)을 탐지하기 위한 강력한 베이스라인이 됩니다.
- **어휘 중첩 휴리스틱 (Lexical overlap heuristic).** 부분 수열 휴리스틱("모든 부분 수열은 함의된다")은 SNLI는 통과하지만 HANS/ANLI에서는 실패합니다. 적대적 벤치마크(adversarial benchmarks)를 사용하세요.
- **문서 길이 저하 (Document-length degradation).** 단일 문장 NLI 모델은 문서 길이의 전제(premises)가 주어지면 F1 점수가 20점 이상 하락합니다. 긴 문맥(long context)에는 DocNLI로 학습된 모델을 사용하세요.
- **제로샷 템플릿 민감도 (Zero-shot template sensitivity).** "This example is about {label}"와 "{label}", 그리고 "The topic is {label}" 사이의 차이로 인해 정확도가 10점 이상 변동될 수 있습니다. 템플릿을 튜닝해 보세요.
- **도메인 불일치 (Domain mismatch).** MNLI는 일반적인 영어로 학습됩니다. 법률, 의료 및 과학 텍스트에는 도메인 특화 NLI 모델(예: SciNLI, MedNLI)이 필요합니다.

## 활용 방법 (Use It)

2026년 스택:

| 사용 사례 (Use case) | 모델 (Model) |
|---------|-------|
| 범용 NLI (General-purpose NLI) | `microsoft/deberta-v3-large-mnli` |
| 빠른 속도 / 엣지 (Fast / edge) | `cross-encoder/nli-deberta-v3-base` |
| 제로샷 분류 (경량형) (Zero-shot classification (lightweight)) | `facebook/bart-large-mnli` |
| 문서 수준 NLI (Document-level NLI) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 다국어 (Multilingual) | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG에서의 환각 탐지 (Hallucination detection in RAG) | RAGAS / DeepEval 내부의 NLI 레이어 |

2026년 메타 패턴 (The 2026 meta-pattern): NLI는 텍스트 이해의 덕테이프(duct tape)와 같습니다. "A가 B를 지지하는가?" 또는 "A가 B와 모순되는가?"를 확인해야 할 때마다, 다른 LLM 호출을 고려하기 전에 NLI를 먼저 활용해 보세요.

## Ship It (실행해 보세요)

`outputs/skill-nli-picker.md`로 저장하세요:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 연습 문제

1. **쉬움(Easy).** 세 가지 클래스를 모두 포함하는 20개의 수동 제작된 (premise, hypothesis, label) 트리플에 대해 `facebook/bart-large-mnli`를 실행해 보세요. 정확도를 측정하세요. 적대적 "부분 수열 휴리스틱(subsequence heuristic)" 함정("I did not eat the cake" vs "I ate the cake")을 추가하여 모델이 무너지는지 확인해 보세요.
2. **중간(Medium).** 100개의 AG News 헤드라인을 대상으로 제로샷 템플릿 `"This text is about {label}"`을 `"The topic is {label}"` 및 `"{label}"`과 비교해 보세요. 정확도 변화(accuracy swing)를 보고하세요.
3. **어려움(Hard).** RAG 충실도(faithfulness) 체크 도구를 구축해 보세요: 원자적 주장 분해(atomic-claim decomposition) + 주장별 NLI. 정답 컨텍스트(gold context)가 포함된 50개의 RAG 생성 답변을 대상으로 평가하세요. 수동 레이블(hand labels) 대비 거짓 양성(false-positive) 및 거짓 음성(false-negative) 비율을 측정하세요.

## 주요 용어 (Key Terms)

| 용어 | 통상적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| NLI | 자연어 추론 (Natural Language Inference) | 전제(premise)-가설(hypothesis) 관계에 대한 3-way 분류입니다. |
| RTE | 텍스트 함의 인식 (Recognizing Textual Entailment) | NLI의 이전 명칭이며, 동일한 작업입니다. |
| Entailment | "t가 h를 함의함" | 전제 $t$가 주어졌을 때, 일반적인 독자는 $h$가 참이라고 결론짓습니다. |
| Contradiction | "t가 h를 배제함" | 전제 $t$가 주어졌을 때, 일반적인 독자는 $h$가 거짓이라고 결론짓습니다. |
| Neutral | "결정되지 않음" | $t$에서 $h$로의 어떠한 추론도 성립하지 않습니다. |
| Zero-shot classification | NLI를 분류기로 활용 | 레이블을 가설로 언어화(verbalize)한 뒤, 가장 높은 함의 값을 선택합니다. |
| Faithfulness | 답변이 근거를 갖추고 있는가? | (검색된 컨텍스트, 생성된 답변) 간의 NLI를 수행합니다. |

## 추가 학습 자료 (Further Reading)

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 벤치마크입니다.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier 방식입니다.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — NLI 작업의 핵심 모델(workhorse)입니다.
