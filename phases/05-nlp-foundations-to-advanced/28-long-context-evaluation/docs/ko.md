# 긴 컨텍스트 평가 — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro는 10M 토큰의 컨텍스트를 지원합니다. 1M 토큰에서 8-니들 MRCR은 26.3%로 떨어집니다. 광고된 성능 ≠ 실제 사용 가능한 성능. 긴 컨텍스트 평가는 출시하는 모델의 실제 용량을 알려줍니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 13강 (질문 답변), 5단계 · 23강 (청킹 전략)
**시간:** 약 60분

## 문제점

200페이지 분량의 계약서가 있습니다. 모델은 1M 토큰 컨텍스트를 지원한다고 주장합니다. 계약서를 붙여넣고 "종료 조항이 무엇인가요?"라고 질문합니다. 모델은 답변하지만, 종료 조항이 모델이 실제로 어텐션하는 범위를 넘어 120k 토큰 깊이에 위치해 있기 때문에 표지 페이지에서 답변합니다.

이것이 2026년 컨텍스트 용량 격차입니다. 사양 시트에는 1M 또는 10M이라고 적혀 있습니다. 현실은 그 중 60-70%만 사용 가능하며, "사용 가능" 여부는 작업에 따라 다릅니다.

- **검색 (건초 더미 속 단일 니들):** 프런티어 모델에서는 광고된 최대 길이까지 거의 완벽합니다.
- **다중 홉 / 집계:** 대부분의 모델에서 약 128k를 넘으면 급격히 성능이 저하됩니다.
- **분산된 사실에 대한 추론:** 가장 먼저 실패하는 작업입니다.

긴 컨텍스트 평가는 이러한 축을 측정합니다. 이 강의에서는 벤치마크의 이름, 각 벤치마크가 실제로 측정하는 내용, 그리고 자체 도메인에 맞춘 커스텀 니들 테스트를 구축하는 방법을 다룹니다.

## 개념

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** 긴 컨텍스트의 제어된 깊이에 사실("마법의 단어는 pineapple이다")을 배치합니다. 모델이 이를 검색하도록 요청합니다. 깊이 × 길이를 스윕합니다. 최초의 긴 컨텍스트 벤치마크입니다. 프런티어 모델은 이제 이 벤치마크를 포화 상태에 도달했습니다. 이는 필요하지만 충분하지 않은 기준선입니다.

**RULER (Nvidia, 2024).** 4개 범주에 걸친 13가지 작업 유형: 검색(단일 / 다중 키 / 다중 값), 다중 홉 추적(변수 추적), 집계(공통 단어 빈도), QA. 컨텍스트 길이를 설정할 수 있습니다(4k부터 128k+). NIAH는 포화 상태에 도달했지만 다중 홉에서 실패하는 모델을 드러냅니다. 2024년 릴리스에서는 32k+ 컨텍스트를 주장하는 17개 모델 중 절반만 32k에서 품질을 유지했습니다.

**LongBench v2 (2024).** 503개의 객관식 질문, 8k~2M 단어 컨텍스트, 6가지 작업 범주: 단일 문서 QA, 다중 문서 QA, 긴 인컨텍스트 학습, 긴 대화, 코드 저장소, 긴 구조화된 데이터. 실제 세계의 긴 컨텍스트 동작을 위한 생산 벤치마크입니다.

**MRCR (Multi-Round Coreference Resolution).** 대규모 다중 턴 지시어 해석. 8-니들, 24-니들, 100-니들 변형. 어텐션이 저하되기 전에 모델이 몇 개의 사실을 처리할 수 있는지 드러냅니다.

**NoLiMa.** "비어휘 니들." 니들과 쿼리 사이에 문자 그대로의 중복이 없으며, 검색은 한 단계의 시맨틱 추론을 필요로 합니다. NIAH보다 어렵습니다.

**HELMET.** 많은 문서를 연결하고 그중 하나에서 질문을 던집니다. 선택적 어텐션을 테스트합니다.

**BABILong.** bAbI 추론 체인을 관련 없는 헤이스트랙 안에 포함시킵니다. 단순한 검색이 아닌 헤이스트랙 내 추론을 테스트합니다.

### 실제로 보고해야 할 사항

- **광고된 컨텍스트 윈도우.** 사양 시트 번호입니다.
- **효과적인 검색 길이.** 특정 임계값(예: 90%)에서의 NIAH 통과율입니다.
- **효과적인 추론 길이.** 해당 임계값에서의 다중 홉 또는 집계 통과율입니다.
- **저하 곡선.** 작업 유형별로 컨텍스트 길이에 따른 정확도를 플롯한 것입니다.

사양 시트용 두 가지 수치: 검색-효과적 길이와 추론-효과적 길이. 일반적으로 추론-효과적 길이는 광고된 윈도우의 25-50%입니다.

```figure
gx-niah-decay
```

## 구현하기

### 1단계: 도메인에 맞춘 커스텀 NIAH

`code/main.py`를 참조하세요. 골격은 다음과 같습니다:

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

    # 헤이스트랙 본문을 채울 만큼 충분히 길게 필러를 반복합니다.
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

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}를 스윕합니다. 히트맵을 플롯하세요. 이것이 대상 모델에 대한 NIAH 카드입니다.

### 2단계: 다중 니들 변형

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

"세 개의 마법 단어는 무엇인가요?"와 같은 질문은 세 개 모두를 검색해야 합니다. 단일 니들 성공은 다중 니들 성공을 예측하지 못합니다.

### 3단계: 다중 홉 변수 추적 (RULER 스타일)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

답변은 세 개의 할당을 체이닝해야 합니다. 128k의 프론티어 모델은 여기서 정확도가 50-70%로 떨어지는 경우가 많습니다.

### 4단계: 스택에서 LongBench v2 실행

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

카테고리별 정확도를 보고하세요. 집계 점수는 작업 단위에서의 큰 차이를 숨깁니다.

## 함정

- **NIAH 전용 평가.** 1M 토큰에서 NIAH를 통과하는 것은 다중 홉(multi-hop)에 대해 아무것도 말해 주지 않습니다. 항상 RULER 또는 맞춤형 다중 홉 테스트를 실행하세요.
- **균일한 깊이 샘플링.** 많은 구현이 depth=0.5만 테스트합니다. depth=0, 0.25, 0.5, 0.75, 1.0을 테스트하세요 — "중간 손실(lost in the middle)" 효과는 실재합니다.
- **필러와의 어휘적 겹침.** 니들(needle)이 필러와 키워드를 공유하면 검색이 자명해집니다. NoLiMa 스타일의 겹치지 않는 니들을 사용하세요.
- **지연 무시.** 1M 토큰 프롬프트는 프리필(prefill)에 30-120초가 걸립니다. 정확도와 함께 첫 토큰까지의 시간(TTFT)을 측정하세요.
- **벤더 자체 보고 수치.** OpenAI, Google, Anthropic 모두 자체 점수를 공개합니다. 항상 사용 사례에 대해 독립적으로 재실행하세요.

## 사용하기

2026년 스택:

| 상황 | 벤치마크 |
|-----------|-----------|
| 빠른 Sanity Check | 3가지 깊이 × 3가지 길이의 맞춤형 NIAH |
| 프로덕션용 모델 선택 | 목표 길이에서 RULER (13개 작업) |
| 실제 QA 품질 | LongBench v2 단일 문서 QA 하위 집합 |
| 다중 홉 추론 | BABILong 또는 맞춤형 변수 추적 |
| 대화/대사 | 목표 길이에서 MRCR 8-니들 |
| 모델 업그레이드 회귀 | 고정된 사내 NIAH + RULER 하네스, 모든 새 모델에서 실행 |

프로덕션에 대한 경험칙: 의도한 길이에서 NIAH + 1개 추론 작업을 수행하기 전까지 컨텍스트 윈도우를 신뢰하지 마세요.

## 출시하기

`outputs/skill-long-context-eval.md`로 저장하세요:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 연습 문제

1. **쉬움.** 3가지 깊이(0.25, 0.5, 0.75) × 3가지 길이(1k, 4k, 16k)로 NIAH를 구축하세요. 임의의 모델에서 실행하세요. 통과율을 3×3 히트맵으로 플롯하세요.
2. **중간.** 3-니들 변형을 추가하세요. 각 길이에서 3개 모두의 검색을 측정하세요. 같은 길이의 단일 니들 통과율과 비교하세요.
3. **어려움.** 64k 필러에 포함된 변수 추적 작업(X1 → X2 → X3, 3홉)을 구성하세요. 3개 프론티어 모델에서 정확도를 측정하세요. 모델별 유효 추론 길이를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| NIAH | Needles in Haystack | 채워진 텍스트 속에 사실을 심고, 모델이 이를 검색하도록 요청합니다. |
| RULER | NIAH의 확장판 | 검색 / 다중 홉 / 집계 / QA에 걸친 13가지 작업 유형. |
| 유효 컨텍스트 | 실제 용량 | 정확도가 임계값 이상을 유지하는 길이. |
| Lost in the middle | 깊이 편향 | 모델이 긴 입력의 중간 내용에 대한 어텐션을 낮게 유지합니다. |
| Multi-needle | 여러 사실을 동시에 | 여러 개의 심기; 검색만이 아닌 어텐션 관리 능력을 테스트합니다. |
| MRCR | 다중 라운드 지시어 | 8, 24, 또는 100개 니들 지시어; 어텐션 포화 현상을 드러냅니다. |
| NoLiMa | 비어휘 니들 | 니들과 쿼리가 문자열 토큰을 공유하지 않으며, 추론이 필요합니다. |

## 추가 읽기

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 원본 NIAH 저장소.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 다중 작업 벤치마크.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 실제 환경의 긴 컨텍스트 평가.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 더 어려운 니들.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — haystack에서의 추론.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 깊이 편향 논문.
