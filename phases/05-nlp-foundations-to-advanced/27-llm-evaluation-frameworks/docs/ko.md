# LLM 평가 — RAGAS, DeepEval, G-Eval

> 정확 일치(Exact Match)와 F1 점수는 의미적 등가성을 놓칩니다. 인간 검토는 확장성이 없습니다. LLM-as-judge는 생산 환경에서의 정답입니다 — 신뢰할 수 있는 수치로 만들기 위해 충분한 보정(Calibration)이 필요합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 5단계 · 13강 (질문 답변), 5단계 · 14강 (정보 검색)
**시간:** 약 75분

## 문제점

RAG 시스템이 "2007년 6월 29일"이라고 답변합니다.
정답 참조는 "2007년 6월 29일"입니다.
정확 일치 점수는 0입니다. F1 점수는 약 75%입니다. 인간은 100%로 평가할 것입니다.

이제 이를 10,000개의 테스트 케이스에 곱해 보세요. 검색기(retriever), 청킹(chunking), 프롬프트, 모델의 모든 변경 사항에 대해 다시 곱해 보세요. 의미를 이해하고, 대규모로 저렴하게 실행되며, 회귀(regression)에 대해 거짓말하지 않고, 올바른 실패 모드(failure mode)를 드러내는 평가기가 필요합니다.

2026년에는 이 문제를 다루는 세 가지 프레임워크가 있습니다.

- **RAGAS.** 검색 증강 생성(RAG) 평가. 네 가지 RAG 지표(충실성, 답변 관련성, 컨텍스트 정밀도, 컨텍스트 재현율)와 NLI + LLM-judge 백엔드를 사용합니다. 연구 기반이며 경량입니다.
- **DeepEval.** LLM을 위한 Pytest. G-Eval, 작업 완료, 환각(Hallucination), 편향(bias) 지표. CI/CD에 내장되어 있습니다.
- **G-Eval.** 방법론이자 DeepEval 지표입니다. 사고의 연쇄(CoT), 사용자 정의 기준, 0-1 점수를 사용하는 LLM-as-judge입니다.

세 가지 모두 LLM-as-judge에 의존합니다. 이 강의는 이 방법과 그 주변에 구축하는 신뢰 계층(trust layer)에 대한 직관을 기릅니다.

## 개념

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** 정적 지표를, 채점 기준(rubric)에 따라 출력에 점수를 매기는 LLM으로 대체합니다. `(query, context, answer)`을 사용하여 판정 LLM에 "충실성에 대해 0-1로 점수를 매기세요"라고 프롬프트합니다. 점수를 반환합니다.

이 방법이 작동하는 이유: LLM은 인간의 판단을 매우 낮은 비용으로 근사합니다. GPT-4o-mini는 약 $0.003 per scored case enables 1000-sample regression eval runs for under $5입니다.

이 방법이 조용히 실패하는 이유:

1. **판정 편향(Judge bias).** 판정자는 더 긴 답변, 자신의 모델 계열에서 나온 답변, 프롬프트 스타일과 일치하는 답변을 선호합니다.
2. **JSON 파싱 실패.** 잘못된 JSON → NaN 점수 → 집계에서 조용히 제외됩니다. RAGAS 사용자는 이 문제를 잘 알고 있습니다. try/except와 명시적인 실패 모드(failure mode)로 게이트를 설정하세요.
3. **모델 버전 간 드리프트.** 판정 모델을 업그레이드하면 모든 지표가 변경됩니다. 판정 모델과 버전을 고정하세요.

**RAG의 네 가지.**

| 지표 | 질문 | 백엔드 |
|--------|----------|---------|
| 충실도 | 답변의 각 주장이 검색된 컨텍스트에서 유래했나요? | NLI 기반 함의 |
| 답변 관련성 | 답변이 질문에 적절히 대응하나요? | 답변에서 가설 질문을 생성하여 실제 질문과 비교 |
| 컨텍스트 정밀도 | 검색된 청크 중 관련 있는 비율은 얼마인가요? | LLM 판정 |
| 컨텍스트 재현율 | 검색이 필요한 모든 것을 반환했나요? | 골든 답변에 대한 LLM 판정 |

**G-Eval.** 커스텀 기준을 정의하세요: "답변이 올바른 출처를 인용했나요?" 이 프레임워크는 사고의 연쇄(CoT) 평가 단계로 자동 확장된 후 0-1로 점수를 매깁니다. RAGAS가 다루지 않는 도메인 특화 품질 차원에 적합합니다.

**보정(Calibration).** 인간 라벨에 대한 상관관계가 확보될 때까지 원시 판정 점수를 신뢰하지 마세요. 100개의 수동 라벨 예제를 실행하세요. 판정 점수와 인간 라벨을 플롯하고 스피어만 rho를 계산하세요. rho가 0.7 미만이라면 판정 기준이 개선되어야 합니다.

```figure
n5-judge-gauge
```

## 구현하기

### 1단계: NLI를 통한 충실도 (RAGAS 스타일)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm`는 호출 가능한 객체입니다: 프롬프트 str -> 생성된 str.
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

답변을 원자적 주장으로 분해하세요. 각 주장을 검색된 컨텍스트에 대해 NLI로 검증하세요. 충실도는 지원되는 비율입니다.

### 2단계: 답변 관련성

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 인코더(encoder): .encode(texts, normalize_embeddings=True) -> ndarray를 구현하는 모든 모델
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

답변이 질문과 다른 질문을 함의한다면 관련성이 떨어집니다.

### 3단계: G-Eval 커스텀 지표

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

평가 단계가 기준입니다. 명시적인 단계는 암시적인 "0-1 점수" 프롬프트보다 더 안정적입니다.

### 4단계: CI 게이트

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

pytest 파일로 출시하세요. 모든 PR에서 실행하세요. 회귀 발생 시 병합을 차단하세요.

### 5단계: 처음부터 만드는 토이 평가

`code/main.py`를 참고하세요. 표준 라이브러리만 사용하는 충실도(답변 주장과 컨텍스트의 겹침) 및 관련성(답변 토큰과 질문 토큰의 겹침) 근사치입니다. 프로덕션용이 아닙니다. 형태를 보여줍니다.

## 함정

- **보정 없음.** 인간 레이블과의 상관관계가 0.3인 판정자는 잡음입니다. 출시 전에 보정 실행을 요구하세요.
- **자기 평가.** 동일한 LLM으로 생성 및 판정을 수행하면 점수가 10-20% 부풀려집니다. 판정자에는 다른 모델 계열을 사용하세요.
- **쌍대 판정에서의 위치 편향.** 판정자는 먼저 제시된 옵션을 선호합니다. 항상 순서를 랜덤화하고 두 경우 모두 실행하세요.
- **원시 집계는 실패를 숨깁니다.** 평균 점수 0.85는 종종 5%의 치명적 실패를 숨깁니다. 항상 하위 분위수를 검사하세요.
- **골든 데이터셋 부패.** 시간이 지남에 따라 드리프트하는 버전 관리되지 않은 평가 세트는 종적 비교를 깨뜨립니다. 모든 변경 사항에 대해 데이터셋에 태그를 지정하세요.
- **LLM 비용.** 대규모에서는 판정 호출이 비용을 지배합니다. 보정 임계값을 충족하는 가장 저렴한 모델을 사용하세요. GPT-4o-mini, Claude Haiku, Mistral-small.

## 사용하기

2026년 스택:

| 사용 사례 | 프레임워크 |
|---------|-----------|
| RAG 품질 모니터링 | RAGAS (4개 지표) |
| CI/CD 회귀 게이트 | DeepEval + pytest |
| 맞춤형 도메인 기준 | DeepEval 내의 G-Eval |
| 온라인 라이브 트래픽 모니터링 | 참조 없는 모드(ref-free mode)를 사용하는 RAGAS |
| 인간 개입 루프(HITL) 스팟 체크 | 주석 UI가 있는 LangSmith 또는 Phoenix |
| 레드 티밍 / 안전 평가 | Promptfoo + DeepEval |

일반적인 스택: 모니터링용 RAGAS, CI용 DeepEval, 새로운 차원용 G-Eval. 세 가지를 모두 실행하세요. 유용하게 불일치합니다.

## 출시하기

`outputs/skill-eval-architect.md`로 저장하세요:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 연습 문제

1. **쉬움.** 알려진 환각이 있는 10개 RAG 예제에 RAGAS를 사용하세요. 충실도 지표가 각각을 포착하는지 확인하세요.
2. **중간.** 50개 QA 답변에 정확도를 0-1로 수동 레이블링하세요. G-Eval로 점수를 매기세요. 판정자와 인간 간의 Spearman rho를 측정하세요.
3. **난이도: 상.** DeepEval로 pytest CI 게이트를 구축하세요. 의도적으로 리트리버를 퇴화시켜 보세요. 게이트가 실패하는지 확인하세요. 하위 10%의 임계값 체크를 통해 하위 분위수(bottom-quantile) 경보 기능을 추가하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| LLM-as-judge | LLM으로 채점하기 | 채점 모델에 루브릭을 주어 출력에 0-1 점수를 매기도록 프롬프트합니다. |
| RAGAS | RAG 지표 라이브러리 | 4가지 참조 없는(reference-free) RAG 지표를 제공하는 오픈소스 평가 프레임워크입니다. |
| Faithfulness (충실성) | 답변이 근거에 기반하고 있나요? | 검색된 컨텍스트에 의해 함의(entail)되는 답변 주장의 비율입니다. |
| Context precision (컨텍스트 정밀도) | 검색된 청크가 관련 있었나요? | 실제로 중요한 top-K 청크의 비율입니다. |
| Context recall (컨텍스트 재현율) | 검색이 모든 것을 찾았나요? | 검색된 청크에 의해 지원되는 정답(gold-answer) 주장의 비율입니다. |
| G-Eval | 맞춤형 LLM 채점자 | 루브릭 + 사고의 연쇄(CoT) 평가 단계 + 0-1 점수. |
| Calibration (보정) | 신뢰하되 검증하세요 | 채점자 점수와 인간 점수 간의 스피어만 상관계수입니다. |

## 추가 읽기

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 논문.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 논문.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 오픈 프로덕션 스택.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 편향, 보정, 한계.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS, DeepEval, Phoenix를 통합하는 통일된 프레임워크.
