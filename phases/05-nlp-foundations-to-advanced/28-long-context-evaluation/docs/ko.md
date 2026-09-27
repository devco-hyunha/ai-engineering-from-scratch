# 긴 문맥 평가 (Long-Context Evaluation) — NIAH, RULER, LongBench, MRCR

> Gemini 1.5 Pro는 1,000만 토큰의 문맥(context)을 제공한다고 광고합니다. 하지만 100만 토큰 환경에서 8-needle MRCR 성능은 26.3%로 급락합니다. 광고된 수치가 곧 사용 가능한 수치는 아닙니다. 긴 문맥 평가(Long-context evaluation)를 통해 여러분이 배포하려는 모델의 실제 역량을 파악할 수 있습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## 문제점 (The Problem)

200페이지 분량의 계약서가 있다고 가정해 봅시다. 모델은 1M 토큰의 컨텍스트 창을 지원한다고 주장합니다. 당신은 계약서를 붙여넣고 "해지 조항(termination clause)이 무엇인가요?"라고 질문합니다. 모델은 답변을 내놓지만, 실제 해지 조항은 120k 토큰 깊숙한 곳에 위치해 있어 모델이 실제로 주의(attend)를 기울이는 범위를 벗어났기 때문에 표지 부분의 내용으로 답변해 버립니다.

이것이 바로 2026년의 컨텍스트 용량 격차(context-capacity gap)입니다. 사양서(Spec sheets)에는 1M 또는 10M이라고 적혀 있지만, 현실은 그중 60~70%만이 사용 가능하며, 그 "사용 가능성"은 작업의 종류에 따라 달라집니다.

- **검색 (Retrieval, 건더미 속의 바늘 찾기):** 최첨단(frontier) 모델의 경우 광고된 최대치까지 거의 완벽하게 수행합니다.
- **멀티홉 / 집계 (Multi-hop / aggregation):** 대부분의 모델에서 약 128k를 넘어서면 성능이 급격히 저하됩니다.
- **분산된 사실에 대한 추론 (Reasoning over dispersed facts):** 가장 먼저 실패하는 작업입니다.

롱 컨텍스트(Long-context) 평가는 이러한 축들을 측정합니다. 이번 레슨에서는 벤치마크의 명칭, 각 벤치마크가 실제로 측정하는 것, 그리고 여러분의 도메인에 맞는 맞춤형 니들 테스트(needle test)를 구축하는 방법을 다룹니다.

## 개념 (The Concept)

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** 긴 문맥(long context) 내의 제어된 깊이에 특정 사실("마법의 단어는 파인애플이다")을 배치합니다. 모델에게 이를 검색하도록 요청합니다. 깊이(depth) × 길이(length)를 변화시키며 테스트합니다. 원조 롱 컨텍스트 벤치마크입니다. 최신 프런티어 모델들은 이제 이 테스트를 포화(saturate) 상태에 도달했으나, 이는 필수적이지만 충분하지는 않은 베이스라인입니다.

**RULER (Nvidia, 2024).** 4개 카테고리에 걸친 13가지 작업 유형을 포함합니다: 검색(retrieval; 단일/다중 키/다중 값), 멀티홉 추적(multi-hop tracing; 변수 추적), 집계(aggregation; 공통 단어 빈도), QA. 설정 가능한 문맥 길이(4k에서 128k 이상). NIAH는 통과하지만 멀티홉(multi-hop)에서 실패하는 모델들을 찾아냅니다. 2024년 출시 버전에서는 32k 이상의 문맥을 지원한다고 주장하는 17개 모델 중 절반만이 32k에서 품질을 유지했습니다.

**LongBench v2 (2024).** 503개의 객관식 질문, 8k~2M 단어 규모의 문맥, 6가지 작업 카테고리로 구성됩니다: 단일 문서 QA, 다중 문서 QA, 긴 인컨텍스트 학습(long in-context learning), 긴 대화, 코드 저장소(code repo), 긴 구조화된 데이터. 실제 환경의 롱 컨텍스트 동작을 측정하기 위한 프로덕션 벤치마크입니다.

**MRCR (Multi-Round Coreference Resolution).** 대규모 멀티턴 상호 참조 해결(coreference resolution) 테스트입니다. 8-needle, 24-needle, 100-needle 변형이 있습니다. 어텐션(attention)이 저하되기 전까지 모델이 얼마나 많은 사실을 동시에 다룰 수 있는지 드러냅니다.

**NoLiMa.** "비어휘적 바늘(Non-lexical needle)"입니다. 바늘(needle)과 쿼리(query) 사이에 문자 그대로의 겹침이 없으며, 검색을 위해 한 단계의 의미론적 추론(semantic reasoning)이 필요합니다. NIAH보다 어렵습니다.

**HELMET.** 많은 문서를 연결한 뒤, 그중 어느 하나에서 질문을 던집니다. 선택적 어텐션(selective attention)을 테스트합니다.

**BABILong.** 무관한 건더기(haystacks) 안에 bAbI 추론 체인을 삽입합니다. 단순 검색이 아닌, 건더기 속에서의 추론(reasoning-in-a-haystack)을 테스트합니다.

### 실제로 보고해야 할 항목 (What to actually report)

- **광고된 컨텍스트 창 (Advertised context window).** 사양서(spec-sheet)에 기재된 수치입니다.
- **유효 검색 길이 (Effective retrieval length).** 특정 임계값(예: 90%)에서의 NIAH 통과 기준입니다.
- **유효 추론 길이 (Effective reasoning length).** 해당 임계값에서의 멀티홉(Multi-hop) 또는 집계(Aggregation) 통과 기준입니다.
- **성능 저하 곡선 (Degradation curve).** 작업 유형별로 플로팅된 컨텍스트 길이에 따른 정확도 변화입니다.

사양서에는 두 가지 수치를 기재해야 합니다: 검색 유효(retrieval-effective) 및 추론 유효(reasoning-effective) 수치입니다. 일반적으로 추론 유효 길이는 광고된 컨텍스트 창의 25~50% 수준입니다.

```figure
gx-niah-decay
```

## 직접 구현해 보기 (Build It)

### 1단계: 도메인 맞춤형 NIAH(Needle In A Haystack) 구축

`code/main.py`를 참조하세요. 기본 골격은 다음과 같습니다:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 건초더미(haystack) 본문을 채울 만큼 충분히 길어질 때까지 filler를 반복합니다.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} 및 `total_tokens` ∈ {1k, 4k, 16k, 64k} 범위에 대해 스윕(sweep)을 수행하세요. 그 결과를 히트맵(heatmap)으로 시각화합니다. 이것이 대상 모델을 위한 NIAH 카드입니다.

### 2단계: 멀티 니들 변형(a multi-needle variant)

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"세 가지 마법의 단어는 무엇인가요?"와 같은 질문은 세 단어 모두를 찾아내야 합니다. 단일 니들(single-needle)에서의 성공이 멀티 니들(multi-needle)에서의 성공을 보장하지는 않습니다.

### 3단계: 멀티홉 변수 추적 (multi-hop variable tracing, RULER 스타일)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

이 질문에 답하기 위해서는 세 번의 할당(assignment) 과정을 체인처럼 연결해야 합니다. 128k 컨텍스트를 가진 프런티어 모델(Frontier models)들도 이 단계에서는 정확도가 50-70% 수준으로 떨어지는 경우가 많습니다.

### 4단계: 스택에서 LongBench v2 실행하기 (LongBench v2 on your stack)

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

카테고리별 정확도(accuracy)를 보고하세요. 점수를 합산하면 태스크 수준의 큰 차이를 가릴 수 있습니다.

## 주의 사항 (Pitfalls)

- **NIAH 전용 평가 (NIAH-only evaluation).** 1M 토큰에서 NIAH를 통과했다고 해서 멀티홉(multi-hop) 능력을 보장하지는 않습니다. 항상 RULER 또는 사용자 정의 멀티홉 테스트를 실행하세요.
- **균일한 깊이 샘플링 (Uniform depth sampling).** 많은 구현체가 `depth=0.5`만 테스트합니다. `depth=0`, `0.25`, `0.5`, `0.75`, `1.0`을 모두 테스트하세요. "중간에서 길을 잃는(lost in the middle)" 현상은 실제로 존재합니다.
- **채우기 텍스트와의 어휘적 중복 (Lexical overlap with filler).** 만약 바늘(needle)이 채우기 텍스트(filler)와 키워드를 공유한다면, 검색은 매우 쉬워집니다. NoLiMa 스타일의 비중복 바늘을 사용하세요.
- **지연 시간 무시 (Ignoring latency).** 1M 토큰 프롬프트는 프리필(prefill) 단계에서 30~120초가 소요됩니다. 정확도와 함께 첫 번째 토큰 생성 시간(time-to-first-token)을 측정하세요.
- **벤더 자체 보고 수치 (Vendor-self-reported numbers).** OpenAI, Google, Anthropic은 모두 자체 점수를 발표합니다. 항상 사용자의 유스케이스에 맞춰 독립적으로 재실행해 보세요.

## 사용 방법 (Use It)

2026년형 스택:

| 상황 (Situation) | 벤치마크 (Benchmark) |
|-----------|-----------|
| 빠른 무결성 검사 (Quick sanity check) | 3가지 깊이 × 3가지 길이의 커스텀 NIAH |
| 프로덕션용 모델 선정 (Model selection for production) | 목표 길이에 맞춘 RULER (13개 태스크) |
| 실전 QA 품질 (Real-world QA quality) | LongBench v2 single-doc-QA 서브셋 |
| 멀티홉 추론 (Multi-hop reasoning) | BABILong 또는 커스텀 변수 추적 (variable-tracing) |
| 대화 / 다이얼로그 (Conversational / dialogue) | 목표 길이에 맞춘 MRCR 8-needle |
| 모델 업그레이드 회귀 테스트 (Model upgrade regression) | 고정된 사내 NIAH + RULER 하네스, 모든 신규 모델에 실행 |

프로덕션을 위한 경험 법칙(Rule of thumb): 의도한 길이에서 NIAH와 1개의 추론 태스크를 수행하기 전까지는 컨텍스트 윈도우(context window)를 절대 신뢰하지 마세요.

## Ship It

`outputs/skill-long-context-eval.md`로 저장하세요:

```markdown
---
name: long-context-eval
description: 주어진 모델과 유스케이스에 적합한 롱 컨텍스트(long-context) 평가 세트를 설계합니다.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

대상 모델, 목표 컨텍스트 길이(target context length), 유스케이스가 주어지면 다음을 출력하세요:

1. 테스트(Tests). NIAH(Needle In A Haystack) 깊이 × 길이 그리드; RULER 멀티홉(multi-hop); 커스텀 도메인 태스크.
2. 샘플링(Sampling). 각 길이에서 0, 0.25, 0.5, 0.75, 1.0의 깊이 적용.
3. 지표(Metrics). 검색 통과율(retrieval pass rate); 추론 통과율(reasoning pass rate); 첫 번째 토큰 생성 시간(time-to-first-token); 쿼리당 비용(cost-per-query).
4. 컷오프(Cutoff). 유효 검색 길이(90% 통과 기준) 및 유효 추론 길이(70% 통과 기준). 두 가지 모두 보고할 것.
5. 회귀 테스트(Regression). 고정된 하네스(harness)를 사용하여 모델 업그레이드 시마다 재실행하고 차이점(deltas)을 도출할 것.

모델 카드(model card)에 명시된 컨텍스트 창 수치만을 신뢰하는 것을 거부하세요. 멀티홉(multi-hop) 작업에 대해 NIAH 전용 평가만을 수행하는 것을 거부하세요. 벤더가 자체 보고한 롱 컨텍스트 점수를 독립적인 증거로 사용하는 것을 거부하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 3가지 깊이(0.25, 0.5, 0.75) × 3가지 길이(1k, 4k, 16k)를 가진 NIAH를 구축해 보세요. 어떤 모델에서든 실행해 보세요. 통과율(pass rate)을 3×3 히트맵(heatmap)으로 시각화해 보세요.
2. **중간 (Medium).** 3-needle 변형 모델을 추가해 보세요. 각 길이에서 3개 모두를 검색(retrieval)하는지 측정해 보세요. 동일한 길이에서의 단일-needle 통과율과 비교해 보세요.
3. **어려움 (Hard).** 64k의 채우기 텍스트(filler) 내에 삽입된 변수 추적(variable-tracing) 작업(X1 → X2 → X3, 3단계 홉)을 구성해 보세요. 3가지 최첨단(frontier) 모델에 대해 정확도를 측정해 보세요. 모델별 유효 추론 길이(effective reasoning length)를 보고해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack (건초더미 속 바늘 찾기) | 채우기용 텍스트(filler) 사이에 사실을 심어두고, 모델이 이를 찾아내는지 테스트합니다. |
| RULER | NIAH on steroids (강화된 NIAH) | 검색(retrieval) / 멀티홉(multi-hop) / 집계(aggregation) / 질의응답(QA)을 아우르는 13가지 작업 유형입니다. |
| Effective context (유효 컨텍스트) | 실제 용량 (The real capacity) | 정확도가 임계값 이상으로 유지되는 컨텍스트의 길이를 의미합니다. |
| Lost in the middle (중간에서의 손실) | 깊이 편향 (Depth bias) | 모델이 긴 입력값의 중간에 위치한 콘텐츠에 대해 주의(attention)를 충분히 기울이지 못하는 현상입니다. |
| Multi-needle (멀티 니들) | 한 번에 여러 사실 찾기 | 여러 개의 사실을 심어두는 방식이며, 단순 검색이 아닌 주의력 조절(attention juggling) 능력을 테스트합니다. |
| MRCR | Multi-round coref (다회차 상호 참조) | 8개, 24개 또는 100개의 니들을 사용하는 상호 참조(coreference) 테스트로, 주의력 포화(attention saturation)를 드러냅니다. |
| NoLiMa | Non-lexical needle (비어휘적 니들) | 니들과 질의(query)가 문자 그대로의 토큰을 공유하지 않으며, 추론(reasoning)을 필요로 합니다. |

## 추가 읽을거리 (Further Reading)

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 원본 NIAH 저장소입니다.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 멀티태스크 벤치마크입니다.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 실세계의 긴 문맥(long-context) 평가입니다.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 더 어려운 형태의 '바늘(needles)'을 다룹니다.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 건초더미 속의 추론(reasoning-in-haystack)을 다룹니다.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 깊이 편향(depth-bias)에 관한 논문입니다.
