# 자연어 추론 — 텍스트 함의

> "t가 h를 함의한다"는 t를 읽은 사람이 h가 참이라고 결론 내린다는 의미입니다. NLI는 함의 / 모순 / 중립을 예측하는 작업입니다. 표면적으로는 단순해 보이지만, 프로덕션 환경에서는 핵심적인 역할을 합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 05강 (감성 분석), 5단계 · 13강 (질문 답변)
**시간:** 약 60분

## 문제점

요약기를 구축했습니다. 요약이 생성되었습니다. 요약에 환각(Hallucination)이 포함되지 않았다는 것을 어떻게 알 수 있나요?

챗봇을 구축했습니다. "예"라고 답변했습니다. 그 답변이 검색된 지문에 의해 뒷받침된다는 것을 어떻게 알 수 있나요?

10,000개의 뉴스 기사를 주제별로 분류해야 합니다. 학습 레이블이 없습니다. 모델을 재사용할 수 있나요?

세 가지 문제 모두 자연어 추론(NLI)으로 귀결됩니다. NLI는 다음을 묻습니다: 전제 `t`와 가설 `h`가 주어졌을 때, `h`가 `t`에 의해 함의되는지, 모순되는지, 아니면 중립(무관한지)인지?

- **환각(Hallucination) 검사:** `t` = 원본 문서, `h` = 요약 주장. 함의되지 않으면 환각(Hallucination)입니다.
- **그라운딩된 QA:** `t` = 검색된 지문, `h` = 생성된 답변. 함의되지 않으면 조작된 내용입니다.
- **제로샷(Zero-Shot) 분류:** `t` = 문서, `h` = 언어화된 레이블 ("이것은 스포츠에 관한 것입니다"). 함의되면 예측된 레이블입니다.

하나의 작업, 세 가지 프로덕션 용도. 모든 RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) 평가 프레임워크가 내부적으로 NLI 모델을 탑재하는 이유입니다.

## 개념

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**세 가지 레이블.**

- **함의.** `t` → `h`. "고양이가 매트 위에 있다"는 "고양이가 있다"를 함의합니다.
- **모순.** `t` → ¬`h`. "고양이가 매트 위에 있다"는 "고양이가 없다"와 모순됩니다.
- **중립.** 어느 쪽으로 추론되지 않습니다. "고양이가 매트 위에 있다"는 "고양이가 배고프다"에 대해 중립입니다.

**논리적 함의가 아닙니다.** NLI는 *자연*어 추론입니다 — 엄격한 논리가 아니라 일반적인 인간 독자가 추론하는 것을 의미합니다. NLI에서는 "John이 개를 산책시켰다"가 "John은 개를 가지고 있다"를 함의하지만, 엄격한 1차 논리에서는 소유 관계를 공리화해야만 이를 인정할 것입니다.

**데이터셋.**

- **SNLI** (2015). 57만 개의 인간이 주석한 쌍, 이미지 캡션을 전제(premise)로 사용. 좁은 도메인.
- **MultiNLI** (2017). 10개 장르에 걸친 43만 3천 개의 쌍. 2026년 기준 표준 학습 코퍼스.
- **ANLI** (2019). 적대적 NLI. 인간이 기존 모델을 깨뜨리도록 특별히 설계된 예제를 작성. 더 어렵습니다.
- **DocNLI, ConTRoL** (2020–21). 문서 길이의 전제. 다중 홉(multi-hop) 및 장거리 추론을 테스트합니다.

**아키텍처.** 트랜스포머 인코더(BERT, RoBERTa, DeBERTa)가 `[CLS] premise [SEP] hypothesis [SEP]`를 읽습니다. `[CLS]` 표현은 3-way softmax로 전달됩니다. MNLI로 학습하고, 보존된 벤치마크로 평가하면, 분포 내(in-distribution) 쌍에서 90% 이상의 정확도를 얻습니다.

**NLI를 통한 제로샷.** 문서와 후보 레이블이 주어지면, 각 레이블을 가설(hypothesis)로 변환합니다("이 텍스트는 스포츠에 대한 것입니다"). 각 가설의 함의(entailment) 확률을 계산합니다. 최대값을 선택합니다. 이것이 Hugging Face의 `zero-shot-classification` 파이프라인의 메커니즘입니다.

```figure
nli-router
```

## 구현하기

### 1단계: 사전 학습된 NLI 모델 실행

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 모든 레이블을 반환; deprecated된 return_all_scores=True를 대체

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

프로덕션 NLI에서는 `facebook/bart-large-mnli`와 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`가 오픈 기본값입니다. DeBERTa-v3는 리더보드에서 최상위를 차지합니다.

### 2단계: 제로샷 분류

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01}]
```

기본 템플릿은 "This example is about {label}."입니다. `hypothesis_template`로 커스터마이즈하세요. 학습 데이터가 필요 없습니다. 미세 조정이 필요 없습니다. 바로 작동합니다.

### 3단계: RAG를 위한 충실성(faithfulness) 체크

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

이것은 RAGAS 충실성의 핵심입니다. 생성된 답변을 원자적 주장(atomic claims)으로 분할합니다. 각 주장을 검색된 컨텍스트와 대조합니다. 함의되는 비율을 보고합니다.

### 4단계: 직접 만든 NLI 분류기 (개념적)

`code/main.py`을 참고하여 stdlib 전용 장난감 코드를 보세요: 전제와 가설을 어휘 겹침 + 부정 탐지로 비교합니다. 트랜스포머 모델과 경쟁할 수준은 아니지만, 작업의 형태를 보여줍니다: 두 텍스트가 입력되고, 3-way 레이블이 출력되며, 손실은 `{entail, contradict, neutral}`에 대한 교차 엔트로피입니다.

## 함정

- **가설 전용 지름길.** 모델은 SNLI에서 "not", "nobody", "never"가 모순과 상관관계가 있어 가설만으로 레이블을 약 60% 예측할 수 있습니다. 레이블 누출을 탐지하기 위한 강력한 기준선입니다.
- **어휘 겹침 휴리스틱.** 부분 순서 휴리스틱("모든 부분 순서는 함의된다")은 SNLI를 통과하지만 HANS/ANLI에서는 실패합니다. 적대적 벤치마크를 사용하세요.
- **문서 길이 성능 저하.** 단일 문장 NLI 모델은 문서 길이의 전제에서 F1이 20 이상 떨어집니다. 긴 컨텍스트에는 DocNLI로 학습된 모델을 사용하세요.
- **제로샷 템플릿 민감도.** "This example is about {label}" vs "{label}" vs "The topic is {label}"는 정확도를 10 이상 변동시킬 수 있습니다. 템플릿을 튜닝하세요.
- **도메인 불일치.** MNLI는 일반 영어로 학습됩니다. 법률, 의료, 과학 텍스트는 도메인 특화 NLI 모델(예: SciNLI, MedNLI)이 필요합니다.

## 사용하기

2026년 스택:

| 사용 사례 | 모델 |
|---------|-------|
| 범용 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 빠른 / 엣지 | `cross-encoder/nli-deberta-v3-base` |
| 제로샷 분류 (경량) | `facebook/bart-large-mnli` |
| 문서 수준 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 다국어 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG에서의 환각 탐지 | RAGAS / DeepEval 내부의 NLI 레이어 |

2026년 메타 패턴: NLI는 텍스트 이해의 덕트 테이프입니다. "A가 B를 지지하는가?" 또는 "A가 B와 모순되는가?"가 필요할 때마다, 다른 LLM 호출을 하기 전에 NLI를 먼저 사용하세요.

## 출시하기

`outputs/skill-nli-picker.md`으로 저장하세요:

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

1. **쉬움.** 세 클래스를 모두 포함하는 20개의 수작업 (전제, 가설, 레이블) 삼중항에 `facebook/bart-large-mnli`을 실행하세요. 정확도를 측정하세요. 적대적 "부분 순서 휴리스틱" 함정("I did not eat the cake" vs "I ate the cake")을 추가하고, 이것이 깨지는지 확인하세요.
2. **중간.** 100개의 AG News 헤드라인에 대해 제로샷 템플릿 `"This text is about {label}"`을 `"The topic is {label}"` 및 `"{label}"`와 비교하세요. 정확도 변동 폭을 보고하세요.
3. **어려움.** RAG 충실성 체크어를 구축하세요: 원자적 주장 분해 + 주장별 NLI. 골든 컨텍스트가 있는 50개의 RAG 생성 답변에 대해 평가하세요. 수동 레이블 대비 오탐 및 미탐 비율을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NLI | 자연어 추론(Natural Language Inference) | 전제-가설 관계의 3-way 분류. |
| RTE | 텍스트 함의 인식(Recognizing Textual Entailment) | NLI의 이전 명칭; 동일한 작업. |
| 함의(Entailment) | "t가 h를 함의함" | 일반적인 독자는 t가 주어지면 h가 참이라고 결론 내립니다. |
| 모순(Contradiction) | "t가 h를 배제함" | 일반적인 독자는 t가 주어지면 h가 거짓이라고 결론 내립니다. |
| 중립(Neutral) | "미결정" | t에서 h로 어느 방향으로도 추론이 불가능합니다. |
| 제로샷 분류 | NLI를 분류기로 사용 | 레이블을 가설로 표현하고, 최대 함의를 선택합니다. |
| 충실성(Faithfulness) | 답변이 지원되는가? | (검색된 컨텍스트, 생성된 답변)에 대한 NLI. |

## 추가 읽기

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 벤치마크.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026년 NLI 주력 도구.
