# 텍스트 요약

> 추출형 시스템은 문서가 말한 내용을 알려줍니다. 생성형 시스템은 저자가 의도한 내용을 알려줍니다. 서로 다른 작업이며, 서로 다른 함정이 있습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 02강 (BoW + TF-IDF), 5단계 · 11강 (기계 번역)
**시간:** 약 75분

## 문제점

2,000단어짜리 뉴스 기사가 피드에 도착했습니다. 120단어로 내용을 파악해야 합니다. 기사에서 가장 중요한 세 문장을 선택(추출형)하거나, 내용을 자신의 말로 다시 작성(생성형)할 수 있습니다. 둘 다 요약이라고 불리지만, 완전히 다른 문제입니다.

추출형 요약은 순위 매기기 문제입니다. 모든 문장에 점수를 매기고, 상위 `k`개를 반환합니다. 원문에서 그대로 가져온 것이므로 출력은 항상 문법적으로 정확합니다. 위험 요소는 기사 전체에 분산된 내용을 놓칠 수 있다는 점입니다.

생성형 요약은 생성 문제입니다. 트랜스포머가 입력을 조건으로 새로운 텍스트를 생성합니다. 출력은 유창하고 압축적이지만, 원문에 없던 사실을 환각(Hallucination)할 수 있습니다. 위험 요소는 확신에 찬 조작입니다.

이 강의에서는 각 방식이 가진 실패 모드와 함께 두 방식을 모두 구축합니다.

## 개념

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**추출형.** 기사를 노드가 문장이고 간선이 유사성인 그래프로 취급합니다. 그래프에 PageRank (또는 유사한 알고리즘)를 실행하여 다른 모든 것과 얼마나 연결되어 있는지에 따라 문장에 점수를 매깁니다. 점수가 가장 높은 문장이 요약이 됩니다. 표준 구현은 **TextRank** (Mihalcea and Tarau, 2004)입니다.

**생성형.** 트랜스포머 인코더-디코더 (BART, T5, Pegasus)를 문서-요약 쌍으로 미세 조정(Fine-tuning)합니다. 추론 시 모델은 문서를 읽고 교차 어텐션(Cross-Attention)을 통해 토큰 단위로 요약을 생성합니다. 특히 Pegasus는 공백 문장(gap-sentence) 사전 학습 목표를 사용하므로, 많은 미세 조정 없이도 요약에 매우 우수합니다.

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)로 평가합니다. ROUGE-01강 ROUGE-2는 단음절(unigram) 및 이중음절(bigram) 겹침을 점수화합니다. ROUGE-L은 최장 공통 부분 수열을 점수화합니다. 점수가 높을수록 좋지만, ROUGE-L이 40이면 "좋음", 50이면 "탁월함"입니다. 모든 논문은 세 가지 점수를 모두 보고합니다. `rouge-score` 패키지를 사용해 보세요.

```figure
summarize-collapse
```

## 구현하기

### 1단계: TextRank (추출형)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

두 가지가 주목할 만합니다. 유사도 함수는 로그 정규화된 단어 겹침을 사용하며, 이는 원본 TextRank 변형입니다. TF-IDF 벡터의 코사인 유사도도 잘 작동합니다. 감쇠 계수 0.85와 반복 횟수는 PageRank의 기본값입니다.

### 2단계: BART를 사용한 생성형 요약

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN은 CNN/DailyMail 코퍼스로 미세 조정되었습니다.开箱(기본 설정)으로 뉴스 스타일 요약을 생성합니다. 다른 도메인(과학 논문, 대화, 법률)에서는 해당 Pegasus 체크포인트를 사용하거나 대상 데이터에 미세 조정하세요.

### 3단계: ROUGE 평가

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

항상 어간 추출(stemming)을 사용하세요. 이를 사용하지 않으면 "running"과 "run"이 서로 다른 단어로 간주되어 ROUGE 점수가 낮게 계산됩니다.

### ROUGE를 넘어서 (2026년 요약 평가)

ROUGE는 20년 동안 요약 평가의 지배적인 지표였으며, 2026년에는 단독으로는 충분하지 않습니다. NLG 논문에 대한 대규모 메타 분석은 다음을 보여줍니다:

- **BERTScore** (맥락적 임베딩 유사도)는 2023년에 지위를 확보했으며, 현재 대부분의 요약 논문에서 ROUGE와 함께 보고됩니다.
- **BARTScore**는 평가를 생성으로 간주합니다: 원본 텍스트가 주어졌을 때 사전 학습된 BART가 요약을 얼마나 높은 확률로 할당하는지 점수화합니다.
- **MoverScore** (맥락적 임베딩에 대한 Earth Mover's Distance)는 ROUGE보다 의미적 겹침을 더 잘 포착하기 때문에 2025년 요약 벤치마크에서 최상위에 올랐습니다.
- **FactCC**와 **QA 기반 충실도(faithfulness)**는 2021-2023년에 흔했으며, 현재는 **G-Eval** (사고의 연쇄(CoT) 추론으로 일관성, 정합성, 유창성, 관련성을 점수화하는 GPT-4 프롬프트 체인)로 자주 대체됩니다.
- **G-Eval** 및 유사한 LLM 심사자 접근법은 평가 기준(rubric)이 잘 설계되었을 때 인간 판단과 약 80% 일치합니다.

프로덕션 권장 사항: 레거시 비교를 위해 ROUGE-L을 보고하고, 의미적 겹침을 위해 BERTScore를, 일관성과 사실성을 위해 G-Eval을 보고하세요. 50-100개의 인간이 라벨링한 요약과 비교하여 보정하세요.

### 4단계: 사실성 문제

추상적 요약은 환각(Hallucination)에 취약합니다. 추출적 요약은 출력물이 원문에서 그대로 인용되기 때문에 환각 위험이 훨씬 낮지만, 원문 문맥이 제거되거나, 내용이 오래되었거나, 인용 순서가 뒤섞일 경우 여전히 오해를 불러일으킬 수 있습니다. 이는 컴플라이언스 관련 콘텐츠에서 프로덕션 시스템이 여전히 추출적 방법을 선호하는 가장 큰 이유입니다.

명명해야 할 환각 유형:

- **엔티티 교체.** 원문은 "John Smith."라고 말하지만, 요약은 "John Brown."라고 말합니다.
- **숫자 변형.** 원문은 "25,000."라고 말하지만, 요약은 "25 million."라고 말합니다.
- **극성 반전.** 원문은 "제안을 거절했다(rejected the offer)."라고 말하지만, 요약은 "제안을 받아들였다(accepted the offer)."라고 말합니다.
- **사실 발명.** 원문은 CEO를 언급하지 않지만, 요약은 CEO가 승인했다고 말합니다.

효과적인 평가 접근법:

- **FactCC.** 원문 문장과 요약 문장 간의 함의(entailment)에 대해 학습된 이진 분류기입니다. 사실적/비사실적 여부를 예측합니다.
- **QA 기반 사실성.** 답이 원문에 있는 질문을 QA 모델에 던집니다. 요약이 다른 답을 지지한다면 플래그를 지정합니다.
- **엔티티 수준 F1.** 원문과 요약의 named entities를 비교합니다. 요약에만 존재하는 엔티티는 의심스럽습니다.

사실성이 중요한 사용자-facing 영역(뉴스, 의료, 법률, 금융)에서는 추출적 방식이 더 안전한 기본값입니다. 추상적 방식은 루프 내의 사실성 체크가 필요합니다.

## 사용하기

2026년 스택:

| 사용 사례 | 권장 |
|---------|-------------|
| 뉴스, 3-5 문장 요약, 영어 | `facebook/bart-large-cnn` |
| 과학 논문 | `google/pegasus-pubmed` 또는 튜닝된 T5 |
| 다중 문서, 장문 | 32k+ 컨텍스트를 가진 모든 LLM, 프롬프트 적용 |
| 대화 요약 | `philschmid/bart-large-cnn-samsum` |
| 추출적, 구조적으로 낮은 환각 위험 | TextRank 또는 `sumy`의 LSA / LexRank |

2026년에는 컴퓨팅이 제약이 아닐 때, 긴 컨텍스트를 가진 LLM이 전문화된 모델을 능가하는 경우가 많습니다. 트레이드오프는 비용과 재현성입니다. 전문화된 모델은 더 일관된 출력을 제공합니다.

## 출시하기

`outputs/skill-summary-picker.md`로 저장:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 연습 문제

1. **쉬움.** 5개의 뉴스 기사에 TextRank를 실행하세요. 상위 3개 문장을 참조 요약과 비교하세요. ROUGE-L을 측정하세요. CNN/DailyMail 스타일 기사에서 30-45의 ROUGE-L 점수를 확인할 수 있습니다.
2. **중간.** 엔티티 수준의 사실성(factuality)을 구현하세요. 원문과 요약에서 named entities를 추출(spaCy)하고, 요약 내 원문 엔티티의 recall과 원문 대비 요약 엔티티의 precision을 계산하세요. 높은 precision과 낮은 recall은 안전하지만 간결함을 의미하며, 낮은 precision은 환각된(hallucinated) 엔티티를 의미합니다.
3. **어려움.** 50개의 CNN/DailyMail 기사에서 BART-large-CNN을 LLM(Claude 또는 GPT-4)과 비교하세요. ROUGE-L, 사실성(엔티티 F1 기준), 요약당 비용을 보고하세요. 각 모델이 승리하는 영역을 문서화하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 추출형(Extractive) | 문장 선택 | 원문에서 문장을 그대로 반환합니다. 절대 환각하지 않습니다. |
| 생성형(Abstractive) | 재작성 | 원문을 조건으로 새 텍스트를 생성합니다. 환각이 발생할 수 있습니다. |
| ROUGE | 요약 지표 | 시스템 출력과 참조 요약 간의 N-gram / LCS 겹침 정도. |
| TextRank | 그래프 기반 추출형 | 문장 유사성 그래프에 대한 PageRank. |
| 사실성(Factuality) | 정확한가 | 요약의 주장이 원문에 의해 지지되는지 여부. |
| 환각(Hallucination) | 지어낸 내용 | 원문이 지지하지 않는 요약 내의 내용. |

## 추가 읽기

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 추출형 요약의 표준 논문.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 논문.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus와 gap-sentence 목적 함수.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 논문.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 사실성 현황 논문.
