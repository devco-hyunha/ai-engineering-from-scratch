# 질의응답 시스템 (Question Answering Systems)

> 세 가지 시스템이 현대의 QA를 형성했습니다. 추출형(Extractive)은 구간을 찾아냈고, 검색 증강형(Retrieval-augmented)은 문서에 근거를 두게 했으며, 생성형(Generative)은 답변을 만들어냈습니다. 모든 현대적 AI 어시스턴트는 이 세 가지의 혼합체입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## 문제 (The Problem)

사용자가 "첫 번째 아이폰은 언제 출시되었나요?"라고 입력하면 "2007년 6월 29일"이라는 답변을 기대합니다. "애플의 역사는 길고 다양합니다"와 같은 답변도, 문장 없이 "2007"만 덩그러니 있는 답변도 아닙니다. 직접적이고, 근거가 있으며, 정확한 답변을 원합니다.

지난 10년 동안 세 가지 아키텍처가 QA를 지배해 왔습니다.

- **추출형 QA (Extractive QA).** 질문과 정답이 포함된 것으로 알려진 지문이 주어졌을 때, 지문 내에서 정답 구간의 시작 및 종료 인덱스를 찾습니다. SQuAD가 표준 벤치마크입니다.
- **오픈 도메인 QA (Open-domain QA).** 지문이 주어지지 않습니다. 먼저 관련 지문을 검색한 다음, 정답을 추출하거나 생성합니다. 이는 오늘날 모든 RAG 파이프라인의 근간입니다.
- **생성형 / 폐쇄형 QA (Generative / Closed-book QA).** 대규모 언어 모델이 자신의 파라미터 메모리(parametric memory)를 통해 답변합니다. 검색 단계가 없습니다. 추론 속도는 가장 빠르지만, 사실 관계의 신뢰도는 가장 낮습니다.

2026년의 트렌드는 하이브리드 방식입니다. 가장 적절한 몇 개의 지문을 검색한 다음, 생성형 모델이 해당 지문에 근거하여 답변하도록 프롬프트를 작성합니다. 이것이 RAG이며, 14과에서 검색(retrieval) 부분을 심도 있게 다룹니다. 본 레슨에서는 QA 부분을 구축합니다.

## 개념 (The Concept)

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**추출형 (Extractive).** 트랜스포머(BERT 계열)를 사용하여 질문과 지문을 함께 인코딩합니다. 정답의 시작 및 종료 토큰 인덱스를 예측하는 두 개의 헤드를 학습시킵니다. 손실 함수는 유효한 위치에 대한 교차 엔트로피(cross-entropy)를 사용합니다. 출력은 지문 내의 특정 구간(span)입니다. 구조적으로 환각(hallucination)을 일으키지 않으며, 지문이 답할 수 없는 질문도 처리하지 않습니다.

**검색 증강형 (Retrieval-augmented (RAG)).** 두 단계로 나뉩니다. 첫째, 검색기(retriever)가 코퍼스에서 상위 `k`개의 지문을 찾습니다. 둘째, 리더(reader, 추출형 또는 생성형)가 해당 지문들을 사용하여 답변을 생성합니다. 검색기와 리더를 분리함으로써 각각 독립적으로 학습하고 평가할 수 있습니다. 현대적인 RAG는 종종 그 사이에 리랭커(reranker)를 추가합니다.

**생성형 (Generative).** 디코더 전용 LLM(GPT, Claude, Llama)이 학습된 가중치를 통해 답변합니다. 검색 단계가 없습니다. 일반 상식에는 뛰어나지만, 희귀하거나 최신 사실에 대해서는 치명적인 오류를 범할 수 있습니다. 환각 발생률은 사전 학습 데이터 내의 사실 빈도와 반비례합니다.

```figure
qa-span
```

## 구축하기 (Build It)

### 1단계: 사전 학습된 모델을 사용한 추출형 QA (extractive QA with a pretrained model)

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`는 답변 불가능한 질문을 포함하는 SQuAD 2.0으로 학습되었습니다. 기본적으로 `question-answering` 파이프라인은 모델의 null 점수가 더 높더라도 가장 점수가 높은 구간을 반환하며, 자동으로 빈 답변을 반환하지는 않습니다. 명시적인 "답변 없음" 동작을 얻으려면 파이프라인 호출 시 `handle_impossible_answer=True`를 전달하세요. 그러면 null 점수가 모든 구간 점수보다 높을 때만 파이프라인이 빈 답변을 반환합니다. 어떤 경우든 `score` 필드를 항상 확인하세요.

### 2단계: 검색 증강형 파이프라인 초안 (a retrieval-augmented pipeline (sketch))

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

2단계 파이프라인입니다. 밀집 검색기(Dense retriever, Sentence-BERT)가 의미적 유사성을 통해 관련 지문을 찾습니다. 추출형 리더(RoBERTa-SQuAD)가 결합된 상위 지문에서 정답 구간을 추출합니다. 작은 규모의 코퍼스에서 잘 작동합니다. 백만 개 문서 규모의 코퍼스라면 FAISS나 벡터 데이터베이스를 사용하세요.

### 3단계: RAG를 활용한 생성형 방식 (generative with RAG)

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

프롬프트 패턴이 중요합니다. 모델에게 문맥에 근거하여 답변하고, 문맥이 불충분할 경우 "모르겠습니다"라고 답하도록 명시적으로 지시하면, 단순한 프롬프트 방식에 비해 환각(hallucination) 발생률을 40-60% 줄일 수 있습니다. 더 정교한 패턴을 사용하면 인용, 신뢰도 점수, 구조화된 추출 기능을 추가할 수 있습니다.

### 4단계: 현실 세계를 반영하는 평가 (evaluation that reflects the real world)

SQuAD는 **정확 일치(Exact Match, EM)**와 **토큰 수준 F1(token-level F1)**을 사용합니다. EM은 정규화(소문자 변환, 문장 부호 제거, 관사 제거) 후의 엄격한 일치를 의미합니다. 예측값이 정확히 일치하거나 아니면 0점을 받습니다. F1은 예측값과 참조값 사이의 토큰 중첩을 기반으로 계산되며 부분 점수를 부여합니다. 두 지표 모두 패러프레이징(paraphrasing)에 대해 점수를 낮게 주는 경향이 있습니다. 예를 들어 "June 29, 2007"과 "June 29th, 2007"은 서수 표현 때문에 정규화가 깨져 EM은 0점을 받지만, 중첩된 토큰 덕분에 상당한 F1 점수를 얻습니다.

프로덕션 QA를 위한 고려 사항:

- **답변 정확도 (Answer accuracy):** (지표가 의미적 동등성을 포착하지 못하므로 LLM 또는 사람이 판단합니다).
- **인용 정확도 (Citation accuracy):** 인용된 지문이 실제로 답변을 뒷받침하는가? 생성된 인용과 검색된 지문 사이의 문자열 일치를 통해 자동으로 쉽게 확인할 수 있습니다.
- **거절 보정 (Refusal calibration):** 답변이 검색된 지문에 없을 때, 시스템이 올바르게 "모르겠습니다"라고 말하는가? 잘못된 확신율(false confidence rate)을 측정합니다.
- **검색 재현율 (Retrieval recall):** 리더를 평가하기 전에, 검색기가 올바른 지문을 상위 `k`개 안에 포함시켰는지 측정합니다. 리더는 누락된 지문을 해결할 수 없습니다.

#### RAGAS: 2026년 프로덕션 평가 프레임워크

`RAGAS`는 RAG 시스템을 위해 특화되어 설계되었으며, 2026년의 표준 프레임워크입니다. 정답(gold reference) 없이도 네 가지 차원을 점수화합니다:

- **충실도 (Faithfulness):** 답변의 각 주장이 검색된 문맥에서 나온 것인가요? NLI 기반의 함의(entailment)로 측정합니다. 주요 환각 지표입니다.
- **답변 관련성 (Answer relevance):** 답변이 질문에 적절히 답하고 있는가? 답변으로부터 가상의 질문을 생성하고 이를 실제 질문과 비교하여 측정합니다.
- **문맥 정밀도 (Context precision):** 검색된 청크 중 실제로 관련이 있는 비율은 얼마인가? 정밀도가 낮으면 프롬프트에 노이즈가 섞여 있다는 뜻입니다.
- **문맥 재현율 (Context recall):** 검색된 세트에 필요한 모든 정보가 포함되어 있는가? 재현율이 낮으면 리더가 성공할 수 없습니다.

정답이 없는(reference-free) 방식의 점수 산출을 통해, 정제된 정답 없이도 실제 운영 중인 트래픽에서 평가를 수행할 수 있습니다. 정확 일치 지표가 무용지물인 개방형 질문의 경우, 상단에 LLM-as-judge를 레이어로 추가하세요.

`pip install ragas`를 설치하고 검색기(retriever)와 리더(reader)를 연결하세요. 쿼리당 네 개의 스칼라 값을 얻을 수 있으며, 성능 저하가 발생하면 알림을 받으세요.

## 사용하기 (Use It)

2026년 스택.

| 사용 사례 | 권장 사항 |
|---------|-------------|
| 지문이 주어졌을 때, 정답 구간 찾기 | `deepset/roberta-base-squad2` |
| 고정된 코퍼스 대상, 폐쇄형 방식 사용 불가 | RAG: 밀집 검색기 + LLM 리더 |
| 문서 저장소에 대한 실시간 처리 | 하이브리드(BM25 + 밀집) 검색기 + 리랭커를 포함한 RAG (14과) |
| 대화형 QA (후속 질문) | 대화 기록을 포함한 LLM + 매 턴마다 RAG 적용 |
| 고도의 사실 관계가 중요한 규제 도메인 | 권위 있는 코퍼스에 대한 추출형 방식; 생성형 단독 사용 금지 |

2026년에는 LLM을 활용한 RAG가 더 많은 사례를 처리하기 때문에 추출형 QA는 인기가 없습니다. 하지만 법률 조사, 규제 준수, 감사 도구와 같이 문자 그대로의 인용이 필요한 문맥에서는 여전히 사용됩니다.

## 배포하기 (Ship It)

Save as `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 연습 문제 (Exercises)

1. **쉬움.** 위에서 설명한 SQuAD 추출형 파이프라인을 10개의 위키피디아 지문에 대해 설정해 보세요. 10개의 질문을 직접 만드세요. 답변이 얼마나 정확한지 측정합니다. 지문과 질문이 깨끗하다면 7~9개의 정답을 얻을 수 있을 것입니다.
2. **중간.** 거절 분류기(refusal classifier)를 추가해 보세요. 최상위 검색 점수가 임계값(예: 코사인 유사도 0.3) 미만일 경우, 리더(reader)를 호출하는 대신 "모르겠습니다"를 반환합니다. 홀드아웃(held-out) 데이터셋을 사용하여 임계값을 조정해 보세요.
3. **어려움.** 원하는 10,000개 문서 코퍼스에 대해 RAG 파이프라인을 구축해 보세요. RRF 퓨전(RRF fusion)을 사용한 하이브리드 검색(BM25 + 밀집 검색)을 구현해 보세요(14과 참조). 하이브리드 단계를 적용했을 때와 적용하지 않았을 때의 답변 정확도를 측정합니다. 어떤 유형의 질문이 가장 큰 이득을 보는지 기록하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 추출형 QA (Extractive QA) | 정답 구간 찾기 | 주어진 지문 내에서 정답의 시작 및 종료 인덱스를 예측합니다. |
| 오픈 도메인 QA (Open-domain QA) | 코퍼스 대상 QA | 주어진 지문이 없으며, 검색 후 답변해야 합니다. |
| RAG | 검색 후 생성 | 검색 증강 생성(Retrieval-augmented generation). 검색기 + 리더 파이프라인입니다. |
| SQuAD | 표준 벤치마크 | Stanford Question Answering Dataset. EM + F1 지표를 사용합니다. |
| 환각 (Hallucination) | 지어낸 답변 | 리더의 출력이 검색된 문맥에 의해 뒷받침되지 않는 경우입니다. |
| 거절 보정 (Refusal calibration) | 답변 불가 시점 인지 | 답변할 수 없을 때 시스템이 올바르게 "모르겠습니다"라고 말하는 능력입니다. |

## 추가 읽기 (Further Reading)

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 벤치마크 논문입니다.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — QA를 위한 표준 밀집 검색기인 DPR입니다.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG라는 이름을 붙인 논문입니다.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 포괄적인 RAG 서베이 논문입니다.
