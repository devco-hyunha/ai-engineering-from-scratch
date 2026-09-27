# LLM 평가(Evaluation) — RAGAS, DeepEval, G-Eval

> Exact-match(정확히 일치) 및 F1 점수는 의미적 동등성(semantic equivalence)을 포착하지 못합니다. 사람이 직접 검토하는 방식은 확장성이 부족합니다. 프로덕션 환경의 해답은 LLM-as-judge(판사로서의 LLM)입니다. 단, 수치를 신뢰할 수 있을 만큼 충분한 보정(calibration)이 이루어져야 합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 문제점 (The Problem)

여러분의 RAG 시스템이 "June 29th, 2007."이라고 답변합니다.
정답(Gold reference)은 "June 29, 2007."입니다.
Exact Match(완전 일치) 점수는 0점입니다. F1 점수는 약 75%입니다. 하지만 사람이 채점한다면 100점을 줄 것입니다.

이제 이를 10,000개의 테스트 케이스로 확장해 보세요. 리트리버(retriever), 청킹(chunking), 프롬프트(prompt) 또는 모델을 변경할 때마다 이 과정을 반복해야 합니다. 여러분에게는 의미를 이해하고, 대규모 환경에서 저렴하게 실행되며, 성능 퇴보(regression)를 정확히 잡아내고, 올바른 실패 모드(failure modes)를 드러내는 평가자가 필요합니다.

2026년 현재, 이 문제를 해결하는 세 가지 프레임워크가 있습니다.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. NLI + LLM-judge 백엔드를 사용하는 네 가지 RAG 지표(faithfulness, answer-relevance, context-precision, context-recall)를 제공합니다. 연구로 검증되었으며 가볍습니다.
- **DeepEval.** LLM을 위한 Pytest입니다. G-Eval, 작업 완료(task-completion), 환각(hallucination), 편향(bias) 지표를 제공합니다. CI/CD 환경에 최적화되어 있습니다.
- **G-Eval.** 하나의 방법론이자 DeepEval의 지표입니다. 사고의 사슬(chain-of-thought), 사용자 정의 기준, 0-1 점수를 사용하는 LLM-as-judge 방식입니다.

세 가지 모두 LLM-as-judge 방식에 의존합니다. 이번 레슨에서는 이 방법론과 이를 둘러싼 신뢰 계층(trust layer)에 대한 직관을 기릅니다.

## 개념 (The Concept)

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** 정적인 지표를 대신하여, 루브릭(rubric)이 주어졌을 때 출력값에 점수를 매기는 LLM을 사용합니다. `(query, context, answer)`가 주어지면, 판사(judge) 역할을 하는 LLM에게 "충실도(faithfulness)를 0-1 사이로 점수 매겨줘"라고 프롬프트를 입력합니다. 그 후 점수를 반환받습니다.

작동 원리: LLM은 인간의 판단을 아주 적은 비용으로 근사할 수 있습니다. 사례당 약 $0.003인 GPT-4o-mini를 사용하면 5달러 미만으로 1,000개 샘플의 회귀 평가(regression eval)를 실행할 수 있습니다.

조용히 실패하는 이유(Why it fails silently):

1. **판사 편향(Judge bias).** 판사는 더 긴 답변, 자신이 속한 모델 제품군의 답변, 프롬프트 스타일과 일치하는 답변을 선호합니다.
2. **JSON 파싱 실패.** 잘못된 JSON 생성 → `NaN` 점수 → 집계에서 조용히 제외됨. RAGAS 사용자라면 이 고통을 잘 알고 있을 것입니다. `try/except`와 명시적인 실패 모드를 통해 방어하세요.
3. **모델 버전에 따른 드리프트(Drift).** 판사 모델을 업그레이드하면 모든 지표가 변합니다. 판사 모델과 버전을 고정(freeze)하세요.

**RAG 4대 지표 (The RAG four).**

| 지표 (Metric) | 질문 (Question) | 백엔드 (Backend) |
|--------|----------|---------|
| 충실도 (Faithfulness) | 답변의 각 주장이 검색된 컨텍스트에서 나온 것인가? | NLI 기반 함의(entailment) |
| 답변 관련성 (Answer relevance) | 답변이 질문에 적절히 답하고 있는가? | 답변으로부터 가상의 질문을 생성한 뒤, 실제 질문과 비교 |
| 컨텍스트 정밀도 (Context precision) | 검색된 청크 중 관련 있는 것은 어느 정도인가? | LLM-judge |
| 컨텍스트 재현율 (Context recall) | 검색이 필요한 모든 정보를 가져왔는가? | 정답(gold answer)을 기준으로 LLM-judge 수행 |

**G-Eval.** 커스텀 기준을 정의합니다: "답변이 올바른 출처를 인용했는가?" 이 프레임워크는 이를 사고 사슬(chain-of-thought) 평가 단계로 자동 확장한 뒤, 0-1 사이의 점수를 매깁니다. RAGAS가 다루지 않는 도메인 특화 품질 차원을 평가할 때 유용합니다.

**교정(Calibration).** 인간의 레이블과 상관관계가 확인되기 전까지는 판사의 원시 점수(raw score)를 절대 신뢰하지 마세요. 100개의 수동 레이블링 예시를 실행합니다. 판사 점수와 인간 점수를 플롯(plot)해 보세요. 스피어만 상관계수(Spearman rho)를 계산합니다. 만약 `rho < 0.7`이라면, 판사의 루브릭을 개선해야 합니다.

```figure
n5-judge-gauge
```

## 직접 구현해 보기 (Build It)

### 1단계: NLI를 이용한 충실도(Faithfulness) 측정 (RAGAS 방식)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm`은 prompt str -> generated str을 수행하는 호출 가능한(callable) 객체입니다.
# 예시: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

답변을 원자적 주장(atomic claims)으로 분해합니다. NLI를 사용하여 각 주장이 검색된 컨텍스트(context)에 의해 뒷받침되는지 확인합니다. 충실도(Faithfulness) = 뒷받침되는 주장의 비율입니다.

### 2단계: 답변 관련성 (answer relevance)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: .encode(texts, normalize_embeddings=True) -> ndarray를 구현하는 모든 모델
# 예: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

만약 답변이 질문한 내용과 다른 질문을 암시한다면, 관련성(relevance) 점수는 낮아집니다.

### 3단계: G-Eval 커스텀 메트릭(Custom Metric)

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

평가 단계(evaluation steps)는 루브릭(rubric) 역할을 합니다. 명시적인 단계(explicit steps)를 제공하는 것이 "0-1 사이로 점수를 매겨라"와 같은 암시적인 프롬프트보다 더 안정적입니다.

### 4단계: CI 게이트 (CI gate)

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

`pytest` 파일로 배포하세요. 모든 PR(Pull Request)에서 실행하세요. 성능 저하(regression)가 발생하면 머지(merge)를 차단하세요.

### 5단계: 기초적인 평가 구현 (toy eval from scratch)

`code/main.py`를 참조하세요. 충실도(faithfulness, 답변의 주장과 컨텍스트 간의 중첩) 및 관련성(relevance, 답변 토큰과 질문 토큰 간의 중첩)에 대해 표준 라이브러리(Stdlib)만을 사용한 근사치를 구현했습니다. 이는 프로덕션용이 아니며, 전체적인 구조를 보여주기 위한 용도입니다.

## 주의 사항 (Pitfalls)

- **보정 부재 (No calibration).** 인간의 라벨과 상관관계가 0.3인 판사 모델(Judge)은 노이즈에 불과합니다. 배포 전 반드시 보정(Calibration) 과정을 거치도록 하세요.
- **자기 평가 (Self-evaluation).** 동일한 LLM을 생성과 평가 판사에 모두 사용하면 점수가 10-20% 부풀려집니다. 판사에는 다른 모델 제품군(Model family)을 사용하세요.
- **쌍체 비교 판사의 위치 편향 (Positional bias in pairwise judging).** 판사는 먼저 제시된 옵션을 선호하는 경향이 있습니다. 항상 순서를 무작위로 섞고 두 가지 경우를 모두 실행하세요.
- **원시 집계 데이터의 실패 은닉 (Raw aggregate hides failures).** 평균 점수 0.85는 종종 5%의 치명적인 실패를 숨깁니다. 항상 하위 분위수(Bottom quantile)를 점검하세요.
- **골든 데이터셋의 부패 (Golden dataset rot).** 시간이 지남에 따라 변하는 버전 관리되지 않은 평가 세트는 종단적 비교(Longitudinal comparison)를 불가능하게 만듭니다. 데이터셋의 모든 변경 사항에 태그를 지정하세요.
- **LLM 비용 (LLM cost).** 대규모 운영 시 판사를 위한 호출 비용이 전체 비용을 압도합니다. 보정 임계값(Calibration threshold)을 충족하는 가장 저렴한 모델을 사용하세요. 예: `GPT-4o-mini`, `Claude Haiku`, `Mistral-small`.

## 활용 방법 (Use It)

2026년 기술 스택:

| 활용 사례 (Use case) | 프레임워크 (Framework) |
|---------|-----------|
| RAG 품질 모니터링 | RAGAS (4가지 지표) |
| CI/CD 회귀 테스트 게이트 | DeepEval + pytest |
| 커스텀 도메인 기준 | DeepEval 내의 G-Eval |
| 온라인 실시간 트래픽 모니터링 | 참조 없는 모드(reference-free mode)를 사용한 RAGAS |
| Human-in-the-loop 스팟 체크 | 어노테이션 UI를 갖춘 LangSmith 또는 Phoenix |
| 레드팀 / 안전성 평가 | Promptfoo + DeepEval |

전형적인 스택: 모니터링에는 RAGAS를, CI에는 DeepEval을, 새로운 차원의 평가에는 G-Eval을 사용합니다. 세 가지를 모두 실행해 보세요. 이들은 유용할 정도로 서로 다른 결과를 보여줍니다.

## Ship It

`outputs/skill-eval-architect.md`로 저장하세요:

```markdown
---
name: eval-architect
description: 보정된 판사(calibrated judge)와 CI 게이트를 포함한 LLM 평가 계획을 설계합니다.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

주어진 유스케이스(RAG / 에이전트 / 생성 작업)에 대해 다음을 출력하세요:

1. 지표(Metrics). Faithfulness / relevance / context-precision / context-recall + 기준(criteria)이 포함된 모든 커스텀 G-Eval 지표.
2. 판사 모델(Judge model). 모델명 + 버전, 비용 대비 정확도에 대한 근거.
3. 보정(Calibration). 수동 라벨링 데이터셋 크기, 목표 Spearman rho 값 및 인간과의 상관관계 > 0.7 여부.
4. 데이터셋 버전 관리(Dataset versioning). 태그 전략, 변경 로그, 층화(stratification).
5. CI 게이트(CI gate). 지표별 임계값(thresholds), 회귀 윈도우(regression-window) 로직, 하위 분위수(bottom-quantile) 알림.

50개 이상의 인간 라벨링 예시로 테스트되지 않은 판사 모델에 의존하는 것을 거부하세요. 자기 평가(동일 모델이 생성 및 판사 수행)를 거부하세요. 하위 10% 결과가 드러나지 않는 총계 위주의 보고를 거부하세요. 병렬 베이스라인 평가 없이 판사 모델 업그레이드가 적용되는 모든 파이프라인에 경고를 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 환각(hallucination)이 포함된 것으로 알려진 10개의 RAG 예제에 RAGAS를 사용해 보세요. 충실도(faithfulness) 지표가 각 사례를 잡아내는지 확인합니다.
2. **중간 (Medium).** 50개의 QA 답변에 대해 정답 여부를 0과 1로 직접 라벨링(hand-label)해 보세요. G-Eval을 사용하여 점수를 매깁니다. 판정 모델(judge)과 사람 사이의 스피어만 상관계수(Spearman rho)를 측정합니다.
3. **어려움 (Hard).** DeepEval을 사용하여 `pytest` CI 게이트(gate)를 구축해 보세요. 의도적으로 리트리버(retriever)의 성능을 퇴보(regress)시킨 후, 게이트가 실패하는지 확인합니다. 하위 10%에 대한 임계값 체크를 통해 하위 분위수(bottom-quantile) 알림 기능을 추가해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| LLM-as-judge | LLM을 이용한 점수 매기기 | 평가 루브릭(rubric)을 기반으로 LLM이 출력값에 0-1 사이의 점수를 부여하도록 프롬프트하는 방식입니다. |
| RAGAS | RAG 메트릭 라이브러리 | 참조 데이터가 필요 없는(reference-free) 4가지 RAG 메트릭을 제공하는 오픈 소스 평가 프레임워크입니다. |
| Faithfulness (충실도) | 답변이 근거를 갖추었는가? | 답변 내 주장이 검색된 컨텍스트에 의해 뒷받침되는 비율입니다. |
| Context precision (컨텍스트 정밀도) | 검색된 청크가 관련이 있는가? | 상위 K개의 청크 중 실제로 유효했던 청크의 비율입니다. |
| Context recall (컨텍스트 재현율) | 검색이 모든 것을 찾아냈는가? | 정답(gold-answer)의 주장 중 검색된 청크에 의해 뒷받침되는 비율입니다. |
| G-Eval | 맞춤형 LLM 평가자 | 루브릭 + 사고 사슬(chain-of-thought) 평가 단계 + 0-1 점수 체계를 사용합니다. |
| Calibration (교정) | 신뢰하되 검증하라 | 평가자 점수와 인간 점수 사이의 스피어만 상관계수(Spearman correlation)를 의미합니다. |

## 추가 읽을거리 (Further Reading)

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 논문입니다.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 논문입니다.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 오픈 소스 프로덕션 스택입니다.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 편향(biases), 보정(calibration), 한계점에 관한 내용입니다.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS, DeepEval, Phoenix를 통합하는 통합 프레임워크입니다.
