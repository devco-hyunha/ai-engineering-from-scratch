# 질문 답변 시스템

> 세 가지 시스템이 현대 QA를 형성했습니다. 추출형은 스팬을 찾습니다. 검색 증강형은 이를 문서에 근거하게 합니다. 생성형은 답변을 만들어냅니다. 모든 현대 AI 어시스턴트는 이 세 가지의 혼합입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 11강 (기계 번역), 5단계 · 10강 (어텐션 메커니즘)
**시간:** 약 75분

## 문제점

사용자가 "첫 iPhone은 언제 출시되었나요?"라고 입력하면 "2007년 6월 29일"이라는 답변을 기대합니다. "Apple의 역사는 길고 다양합니다"라는 답변이 아닙니다. 문장 없이 고립된 "2007"도 아닙니다. 직접적이고, 근거가 있으며, 정확한 답변입니다.

지난 10년간 QA를 지배해 온 세 가지 아키텍처가 있습니다.

- **추출형 QA.** 질문과 답변이 포함된 것으로 알려진 지문을 주어지면, 지문 내에서 답변 스팬의 시작 및 끝 인덱스를 찾습니다. SQuAD가 표준 벤치마크입니다.
- **오픈 도메인 QA.** 지문이 주어지지 않습니다. 먼저 관련 지문을 검색한 후, 추출하거나 생성하여 답변을 만듭니다. 이는 오늘날 모든 RAG 파이프라인의 기초입니다.
- **생성형 / 클로즈드북 QA.** 대규모 언어 모델이 파라메트릭 메모리에서 답변합니다. 검색이 없습니다. 추론 속도가 가장 빠르지만, 사실에 대한 신뢰도는 가장 낮습니다.

2026년의 트렌드는 하이브리드입니다. 최상의 몇몇 지문을 검색한 후, 생성형 모델에 프롬프트하여 그 지문들에 근거해 답변하게 합니다. 이것이 RAG이며, 14강은 검색 부분을 심층적으로 다룹니다. 이 강의는 QA 부분을 구축합니다.

## 개념

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**추출형.** 질문과 지문을 트랜스포머 (BERT 계열)로 함께 인코딩합니다. 답변의 시작 및 끝 토큰 인덱스를 예측하는 두 헤드를 학습합니다. 손실은 유효한 위치에 대한 교차 엔트로피입니다. 출력은 지문에서의 스팬입니다. (구조적으로) 환각이 없으며, (구조적으로) 지문이 답변할 수 없는 질문을 처리하지 않습니다.

**검색 증강형 (RAG).** 두 단계입니다. 먼저, 리트리버가 코퍼스에서 상위 `k` 지문을 찾습니다. 둘째, 리더 (추출형 또는 생성형)가 그 지문들을 사용하여 답변을 생성합니다. 리트리버-리더 분리는 각각을 독립적으로 학습하고 평가할 수 있게 합니다. 현대 RAG는 종종 그 사이에 리랭커를 추가합니다.

**생성형.** 디코더 전용 LLM (GPT, Claude, Llama)은 학습된 가중치(Weight)에서 답변합니다. 검색 단계가 없습니다. 일반적인 지식에는 뛰어나지만, 희소하거나 최신 사실에는 치명적입니다. 환각(Hallucination)률은 사전 학습 데이터 내 사실의 빈도와 반비례합니다.

```figure
qa-span
```

## 구현하기

### 1단계: 사전 학습된 모델을 이용한 추출형 QA

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

`deepset/roberta-base-squad2`는 SQuAD 2.0으로 학습되었으며, 여기에는 답변할 수 없는 질문이 포함됩니다. 기본적으로 `question-answering` 파이프라인은 모델의 null 점수가 이기더라도 가장 높은 점수를 받은 스팬(span)을 반환합니다. 즉, 빈 답변을 *자동으로* 반환하지 *않습니다*. 명시적인 "no answer" 동작을 얻으려면 파이프라인 호출에 `handle_impossible_answer=True`를 전달하세요. 파이프라인은 null 점수가 모든 스팬 점수를 초과할 때만 빈 답변을 반환합니다. 어느 쪽이든 `score` 필드를 항상 확인하세요.

### 2단계: 검색 증강 파이프라인 (개략)

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

2단계 파이프라인. 밀집 검색기(Dense Retrieval)(Dense Retrieval)(Sentence-BERT)가 시맨틱 검색(Semantic Search)(semantic similarity)를 통해 관련 구문을 찾습니다. 추출형 리더(RoBERTa-SQuAD)가 결합된 상위 구문에서 답변 스팬(span)을 추출합니다. 작은 코퍼스에 잘 작동합니다. 백만 문서 코퍼스의 경우 FAISS나 벡터 데이터베이스(Vector Database)(vector database)를 사용하세요.

### 3단계: RAG를 이용한 생성형

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

프롬프트 패턴이 중요합니다. 컨텍스트에 근거하고 컨텍스트가 불충분할 때 "I don't know"를 반환하도록 모델에 명시적으로 지시하면, 단순한 프롬프트와 비교해 환각(Hallucination)률이 40-60% 감소합니다. 더 정교한 패턴은 인용문, 신뢰도 점수, 구조화된 추출을 추가합니다.

### 4단계: 실제 세계를 반영하는 평가

SQuAD는 **정확 일치(Exact Match (EM))**와 **토큰 단위 F1**을 사용합니다. EM은 정규화(소문자화, 구두점 제거, 관사 제거) 후의 엄격한 일치입니다. 예측이 정확히 일치하면 점수를 얻고, 그렇지 않으면 0점입니다. F1은 예측과 참조 간의 토큰 중복을 계산하며 부분 점수를 부여합니다. 둘 다 패러프레이즈(paraphrase)에 대해 점수를 낮게 매깁니다. "June 29, 2007"와 "June 29th, 2007"는 서수(ordinal)가 정규화를 깨뜨리므로 EM 점수는 보통 0이지만, 중복되는 토큰으로 상당한 F1 점수를 얻습니다.

프로덕션 QA의 경우:

- **답변 정확도** (지표가 의미적 동등성을 포착하지 못하므로 LLM이 판단하거나 사람이 판단).
- **인용 정확성.** 인용된 구문이 실제로 답변을 뒷받침하는가? 생성된 인용문과 검색된 구문 간의 문자열 일치로 자동으로 확인하기 쉽습니다.
- **거절 보정.** 검색된 구문에 답변이 없을 때, 시스템이 "모르겠습니다"라고 올바르게 말하는가? 거짓 확신율을 측정하세요.
- **검색 재현율.** 리더를 평가하기 전에, 검색기가 올바른 구문을 상위 `k`에 포함시키는가 확인하세요. 검색기가 놓친 구문은 리더가 복구할 수 없습니다.

### RAGAS: 2026년 프로덕션 평가 프레임워크

`RAGAS`는 RAG 시스템을 위해 특별히 제작되었으며, 2026년 기본 출시 옵션입니다. 골든 레퍼런스가 필요하지 않은 네 가지 차원을 점수화합니다:

- **충실성.** 답변의 각 주장이 검색된 컨텍스트에서 유래하는가? NLI 기반 함의(entailment)로 측정됩니다. 주요 환각(hallucination) 지표입니다.
- **답변 관련성.** 답변이 질문에 적절히 대응하는가? 답변에서 가상의 질문을 생성하고 실제 질문과 비교하여 측정합니다.
- **컨텍스트 정밀도.** 검색된 청크 중 실제로 관련 있는 비율은 얼마인가? 낮은 정밀도는 프롬프트 내 잡음을 의미합니다.
- **컨텍스트 재현율.** 검색된 집합에 필요한 모든 정보가 포함되었는가? 낮은 재현율은 리더가 성공할 수 없음을 의미합니다.

레퍼런스 없는 점수화(ref-free scoring)를 통해 큐레이션된 골든 답변 없이 라이브 프로덕션 트래픽에서 평가할 수 있습니다. 정확 일치(exact-match) 지표가 무용한 개방형 질문에는 LLM-as-judge를 상위에 계층화하세요.

`pip install ragas`. 검색기 + 리더를 연결하세요. 쿼리당 네 개의 스칼라 값을 얻습니다. 회귀(regression) 발생 시 알림을 설정하세요.

## 사용하기

2026년 스택입니다.

| 사용 사례 | 권장 사항 |
|---------|-------------|
| 주어진 구문에서 답변 스팬(span) 찾기 | `deepset/roberta-base-squad2` |
| 고정된 코퍼pus에서, closed-book이 허용되지 않는 경우 | RAG: 밀집 검색기(dense retriever) + LLM 리더 |
| 문서 저장소에서의 실시간 처리 | 하이브리드(BM25 + 밀집) 검색기 + 리랭커를 사용한 RAG (14강) |
| 대화형 QA (후속 질문) | 대화 히스토리를 가진 LLM + 각 턴(turn)에서의 RAG |
| 고도의 사실성, 규제 대상 도메인 | 권위 있는 코퍼pus에 대한 추출(extractive) 방식; 생성(generative) 단독 사용은 절대 금지 |

추출형 QA는 2026년 LLM 기반 RAG가 더 많은 케이스를 처리하기 때문에 유행이 아닙니다. 하지만 문자 그대로의 인용이 필요한 상황, 즉 법률 연구, 규제 준수, 감사 도구 등에서는 여전히 사용되고 있습니다.

## 출시하기

`outputs/skill-qa-architect.md`로 저장하세요:

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

## 연습 문제

1. **쉬움.** 위 추출형 SQuAD 파이프라인을 위키백과 passage 10개에 대해 설정하세요. 질문 10개를 직접 만들어 보세요. 정답이 얼마나 자주 맞는지 측정해 보세요. passage와 질문이 깔끔하다면 7~9개는 정답일 것입니다.
2. **중간.** 거절 분류기를 추가하세요. 상위 검색 점수가 임계값(예: 코사인 유사도 0.3) 미만일 때, reader를 호출하는 대신 "모르겠습니다"를 반환하도록 하세요. 임계값은 홀드아웃 세트에서 튜닝하세요.
3. **어려움.** 선택한 10,000개 문서 코퍼스에 대해 RAG 파이프라인을 구축하세요. 하이브리드 검색(BM25 + dense)을 RRF 융합(14강 참조)으로 구현하세요. 하이브리드 단계를 포함/미포함한 경우의 답변 정확도를 측정하세요. 어떤 질문 유형이 가장 큰 이점을 얻는지 문서화하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 추출형 QA | 답변 스팬 찾기 | 주어진 passage 내에서 답변의 시작 및 끝 인덱스를 예측합니다. |
| 오픈 도메인 QA | 코퍼스에 대한 QA | 주어진 passage가 없으며, 검색한 후 답변해야 합니다. |
| RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) | 검색 후 생성 | 검색 증강 생성입니다. Retriever + reader 파이프라인입니다. |
| SQuAD | 표준 벤치마크 | Stanford Question Answering Dataset입니다. EM + F1 지표입니다. |
| 환각(Hallucination) | 지어낸 답변 | Reader의 출력은 검색된 컨텍스트로 뒷받침되지 않습니다. |
| 거절 보정(Calibration) | 언제 침묵해야 하는지 알기 | 시스템이 답변할 수 없을 때 올바르게 "모르겠습니다"라고 말합니다. |

## 추가 읽기

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 벤치마크 논문입니다.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — QA용 표준 dense retriever인 DPR입니다.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG라는 이름을 붙인 논문입니다.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 종합적인 RAG 서베이입니다.
