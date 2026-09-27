# 텍스트 요약 (Text Summarization)

> 추출적(Extractive) 시스템은 문서가 무엇을 말했는지 알려줍니다. 생성적(Abstractive) 시스템은 저자가 무엇을 의도했는지 알려줍니다. 작업이 다르면 문제점도 다릅니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## 문제 (The Problem)

2,000단어 분량의 뉴스 기사가 피드에 올라왔습니다. 내용을 파악할 수 있는 120단어의 요약이 필요합니다. 기사에서 가장 중요한 문장 세 개를 선택하거나(추출적), 내용을 자신의 언어로 다시 쓸 수 있습니다(생성적). 둘 다 요약(summarization)이라고 불리지만, 완전히 다른 문제입니다.

추출적 요약(extractive summarization)은 랭킹 문제입니다. 모든 문장에 점수를 매기고 상위 `k`개를 반환합니다. 출력은 원문을 그대로 가져오기 때문에 항상 문법적으로 정확합니다. 위험 요소는 기사 전체에 분산된 내용을 놓칠 수 있다는 점입니다.

생성적 요약(abstractive summarization)은 생성 문제입니다. 트랜스포머가 입력을 조건으로 새로운 텍스트를 생성합니다. 출력은 유창하고 압축적이지만, 원본 소스에 없는 사실을 환각(hallucinate)할 수 있습니다. 위험 요소는 자신 있게 사실을 조작하는 것입니다.

이 레슨에서는 각 방식이 가진 고유한 실패 모드와 함께 두 가지 모두를 구축해 봅니다.

## 개념 (The Concept)

![추출적 TextRank 대 생성적 트랜스포머](../assets/summarization.svg)

**추출적(Extractive).** 기사를 노드는 문장, 에지는 유사도로 구성된 그래프로 취급합니다. 그래프에 PageRank(또는 유사한 알고리즘)를 실행하여 다른 모든 문장과 얼마나 연결되어 있는지에 따라 문장의 점수를 매깁니다. 점수가 가장 높은 문장들이 요약문이 됩니다. 대표적인 구현체는 **TextRank**(Mihalcea and Tarau, 2004)입니다.

**생성적(Abstractive).** 문서-요약 쌍을 사용하여 트랜스포머 인코더-디코더(BART, T5, Pegasus)를 미세 조정(fine-tune)합니다. 추론 시 모델은 문서를 읽고 크로스 어텐션(cross-attention)을 통해 토큰 단위로 요약문을 생성합니다. 특히 Pegasus는 결손 문장(gap-sentence) 사전 학습 목표를 사용하여 많은 미세 조정 없이도 요약 작업에 탁월한 성능을 보입니다.

**ROUGE**(Recall-Oriented Understudy for Gisting Evaluation)를 이용한 평가. ROUGE-1과 ROUGE-2는 유니그램(unigram) 및 바이그램(bigram) 중첩도를 측정합니다. ROUGE-L은 최장 공통 부분 수열(longest common subsequence)을 측정합니다. 점수가 높을수록 좋지만, ROUGE-L 40은 "좋음", 50은 "매우 뛰어남" 수준입니다. 모든 논문은 이 세 가지를 모두 보고합니다. `rouge-score` 패키지를 사용하세요.

```figure
summarize-collapse
```

## 구축하기 (Build It)

### 1단계: TextRank (추출적)

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

언급할 만한 두 가지 사항이 있습니다. 유사도 함수는 원래 TextRank 변형인 로그 정규화된 단어 중첩(log-normalized word overlap)을 사용합니다. TF-IDF 벡터의 코사인 유사도도 작동합니다. 댐핑 계수(damping factor) 0.85와 반복 횟수는 PageRank의 기본값입니다.

### 2단계: BART를 이용한 생성적 요약 (Abstractive with BART)

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN은 CNN/DailyMail 코퍼스로 미세 조정되었습니다. 별도의 설정 없이도 뉴스 스타일의 요약을 생성합니다. 다른 도메인(과학 논문, 대화, 법률)의 경우, 해당하는 Pegasus 체크포인트를 사용하거나 대상 데이터로 미세 조정하세요.

### 3단계: ROUGE 평가 (ROUGE Evaluation)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

항상 어간 추출(stemming)을 사용하세요. 그렇지 않으면 "running"과 "run"을 서로 다른 단어로 간주하여 ROUGE 점수가 실제 중첩도보다 낮게 측정됩니다.

### ROUGE를 넘어서: 2026년 요약 평가 (Beyond ROUGE (2026 summarization eval))

ROUGE는 20년 동안 지배적인 요약 지표였지만, 2026년에는 그것만으로는 불충분합니다. 자연어 생성(NLG) 논문에 대한 대규모 메타 분석 결과는 다음과 같습니다:

- **BERTScore**(문맥 임베딩 유사도)는 2023년까지 입지를 넓혔으며, 현재 대부분의 요약 논문에서 ROUGE와 함께 보고됩니다.
- **BARTScore**는 평가를 생성 문제로 취급합니다. 사전 학습된 BART가 소스를 기반으로 요약문을 생성할 확률을 점수로 매깁니다.
- **MoverScore**(문맥 임베딩에 대한 Earth Mover's Distance)는 ROUGE보다 의미적 중첩을 더 잘 포착하기 때문에 2025년 요약 벤치마크에서 1위를 차지했습니다.
- **FactCC**와 **QA 기반 충실도(QA-based faithfulness)**는 2021-2023년에 흔히 사용되었으나, 현재는 종종 **G-Eval**(사고의 사슬(chain-of-thought) 추론을 통해 일관성, 일치성, 유창성, 관련성을 점수 매기는 GPT-4 프롬프트 체인)로 대체되고 있습니다.
- 루브릭(rubric)이 잘 설계된 경우, G-Eval 및 유사한 LLM-judge 방식은 인간의 판단과 약 80% 일치합니다.

프로덕션 권장 사항: 레거시 비교를 위해 ROUGE-L을, 의미적 중첩을 위해 BERTScore를, 일관성과 사실성을 위해 G-Eval을 보고하세요. 50~100개의 인간 라벨링 요약본을 기준으로 보정(calibrate)하세요.

### 4단계: 사실성 문제 (The Factuality Problem)

생성적 요약은 환각(hallucination)에 취약합니다. 추출적 요약은 출력이 소스에서 그대로 가져오기 때문에 환각 위험이 훨씬 낮지만, 소스 문장이 문맥에서 벗어나거나, 오래되었거나, 순서가 뒤바뀐 채 인용될 경우 오해를 불러일으킬 수 있습니다. 이것이 프로덕션 시스템이 규제와 밀접한 콘텐츠에 대해 여전히 추출적 방식을 선호하는 가장 큰 이유입니다.

주요 환각 유형:

- **개체 교체(Entity swap).** 소스에는 "John Smith"라고 되어 있으나 요약에는 "John Brown"이라고 나옵니다.
- **숫자 변동(Number drift).** 소스에는 "25,000"이라고 되어 있으나 요약에는 "2,500만(25 million)"이라고 나옵니다.
- **극성 반전(Polarity flip).** 소스에는 "제안을 거절했다(rejected the offer)"고 되어 있으나 요약에는 "제안을 수락했다(accepted the offer)"고 나옵니다.
- **사실 조작(Fact invention).** 소스에는 CEO에 대한 언급이 없는데, 요약에는 CEO가 승인했다고 나옵니다.

효과적인 평가 접근 방식:

- **FactCC.** 소스 문장과 요약 문장 간의 함의(entailment)에 대해 학습된 이진 분류기입니다. 사실임/사실 아님을 예측합니다.
- **QA 기반 사실성(QA-based factuality).** 정답이 소스에 있는 질문을 QA 모델에 던집니다. 요약문이 다른 답을 뒷받침한다면 플래그를 표시합니다.
- **개체 수준 F1(Entity-level F1).** 소스와 요약문의 개체명(named entity)을 비교합니다. 요약문에만 나타나는 개체는 의심 대상입니다.

사실성이 중요한 사용자 대면 콘텐츠(뉴스, 의료, 법률, 금융)의 경우 추출적 요약이 더 안전한 기본값입니다. 생성적 요약은 루프 안에 사실성 검증(factuality check) 단계가 반드시 필요합니다.

## 활용하기 (Use It)

2026년 스택 권장 사항:

| 사용 사례 | 권장 사항 |
|---------|-------------|
| 뉴스, 3-5문장 요약, 영어 | `facebook/bart-large-cnn` |
| 과학 논문 | `google/pegasus-pubmed` 또는 미세 조정된 T5 |
| 다중 문서, 장문 요약 | 32k 이상의 컨텍스트를 지원하는 프롬프트 기반 LLM |
| 대화 요약 | `philschmid/bart-large-cnn-samsum` |
| 추출적 요약, 구조적으로 낮은 환각 위험 | TextRank 또는 `sumy`의 LSA / LexRank |

2026년에는 컴퓨팅 자원이 제약되지 않는 한 긴 컨텍스트를 지원하는 LLM이 특화 모델보다 뛰어난 경우가 많습니다. 다만 비용과 재현성 간의 트레이드오프가 존재하며, 특화 모델이 더 일관된 출력을 제공합니다.

## 배포하기 (Ship It)

`outputs/skill-summary-picker.md`로 저장하세요:

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

## 연습 문제 (Exercises)

1. **기초 (Easy).** 5개의 뉴스 기사에 대해 TextRank를 실행해 보세요. 상위 3개 문장을 참조 요약과 비교하세요. ROUGE-L을 측정합니다. CNN/DailyMail 스타일 기사에서 30~45 수준의 ROUGE-L 점수가 나와야 합니다.
2. **중급 (Medium).** 개체 수준의 사실성(entity-level factuality) 검증을 구현해 보세요: 소스와 요약문에서 개체명을 추출(spaCy)하고, 요약문 내 소스 개체의 재현율(recall)과 소스 대비 요약문 개체의 정밀도(precision)를 계산합니다. 정밀도가 높고 재현율이 낮으면 안전하지만 간결함을 의미하며, 정밀도가 낮으면 환각된 개체가 존재함을 뜻합니다.
3. **고급 (Hard).** 50개의 CNN/DailyMail 기사를 대상으로 BART-large-CNN과 LLM(Claude 또는 GPT-4)을 비교해 보세요. ROUGE-L, 사실성(개체 F1 기준), 요약당 비용을 보고하세요. 각 모델이 우수한 영역을 문서화해 보세요.

## 핵심 용어 (Key Terms)

| 용어 | 일반적인 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 추출적(Extractive) | 문장 선택 | 소스에서 문장을 그대로 반환함. 환각이 발생하지 않음. |
| 생성적(Abstractive) | 재작성 | 소스를 바탕으로 새로운 텍스트를 생성함. 환각이 발생할 수 있음. |
| ROUGE | 요약 평가 지표 | 시스템 출력과 참조 요약 간의 N-gram 및 LCS 중첩도. |
| TextRank | 그래프 기반 추출 | 문장 유사도 그래프를 통한 PageRank 계산. |
| 사실성(Factuality) | 내용이 맞는지 | 요약문의 주장이 소스에 의해 뒷받침되는지 여부. |
| 환각(Hallucination) | 조작된 내용 | 소스가 뒷받침하지 않는 요약문 내의 내용. |

## 추가 참고 자료 (Further Reading)

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 추출적 방식의 표준 논문.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 논문.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 및 결손 문장(gap-sentence) 사전 학습 목표 논문.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 논문.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 사실성 환경 분석 논문.
