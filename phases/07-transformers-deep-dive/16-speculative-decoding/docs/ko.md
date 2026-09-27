# Speculative Decoding (추측적 디코딩) — 초안 작성, 검증, 반복

> 자기회귀(Autoregressive) 디코딩은 직렬 방식입니다. 각 토큰은 이전 토큰이 생성될 때까지 기다려야 합니다. 추측적 디코딩(Speculative decoding)은 이 사슬을 끊어냅니다. 가벼운 모델이 $N$개의 토큰 초안을 작성하면, 무거운 모델이 단 한 번의 순전파(forward pass)로 $N$개 토큰 전체를 검증합니다. 초안이 정확할 경우, $N$개의 토큰을 생성하는 데 단 한 번의 큰 순전파 비용만 지불하면 됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 문제점 (The Problem)

H100에서 70B LLM이 토큰 하나를 샘플링하는 데는 약 30ms가 소요됩니다. 3B 초안 모델(draft model)은 약 3ms가 소요됩니다. 만약 3B 모델이 5개의 토큰을 미리 생성하게 한 뒤, 70B 모델을 *단 한 번* 실행하여 5개 토큰 모두를 검증한다면, 최대 5개의 토큰을 수락하는 데 드는 총 시간은 `5×3 + 30 = 45 ms`입니다. 이는 순차적 생성 방식의 `5×30 = 150 ms`와 대조적입니다. 이것이 바로 추측적 디코딩(speculative decoding)의 핵심 논리입니다. 즉, 약간의 추가 GPU 메모리(초안 모델)를 사용하는 대신 디코딩 지연 시간(latency)을 2~4배 낮추는 것입니다.

핵심은 분포(distribution)를 보존해야 한다는 점입니다. Leviathan et al. (2023)과 Chen et al.이 동시에 도입한 추측적 샘플링(Speculative sampling)은 출력 시퀀스가 거대 모델이 단독으로 생성했을 때와 **동일한 분포(identically distributed)**를 갖도록 보장합니다. 품질 저하는 없습니다. 오직 속도만 빨라질 뿐입니다.

2026년 추론 기술을 주도하는 네 가지 초안-검증기(draft-verifier) 쌍의 체계는 다음과 같습니다:

1. **바닐라 추측 방식 (Vanilla speculative, Leviathan 2023).** 별도의 초안 모델(예: Llama 3 1B) + 검증기(예: Llama 3 70B)를 사용합니다.
2. **메두사 (Medusa, Cai 2024).** 검증기에 여러 개의 디코딩 헤드를 추가하여 `t+1..t+k` 위치를 병렬로 예측합니다. 별도의 초안 모델이 필요 없습니다.
3. **EAGLE 체계 (EAGLE family, Li 2024, 2025).** 검증기의 은닉 상태(hidden states)를 재사용하는 경량 초안 모델을 사용합니다. 바닐라 방식보다 수락률(acceptance rate)이 높으며, 일반적으로 3~4배 더 빠릅니다.
4. **룩어헤드 디코딩 (Lookahead decoding, Fu 2024).** Jacobi iteration 방식을 사용하며, 초안 모델이 전혀 필요하지 않습니다. 자체 추측(Self-speculation) 방식입니다. 특정 분야에 특화되어 있지만 의존성이 없다는 장점이 있습니다.

2026년의 모든 프로덕션 추론 스택은 기본적으로 추측적 디코딩을 탑재하여 출시됩니다. vLLM, TensorRT-LLM, SGLang, llama.cpp 모두 최소한 바닐라 방식과 EAGLE-2를 지원합니다.

## 개념 (The Concept)

### 핵심 알고리즘 (The core algorithm)

검증기(verifier) `M_q`와 더 저렴한 초안 모델(draft model) `M_p`가 주어졌을 때:

1. 이미 디코딩된 접두사(prefix)를 `x_1..x_k`라고 합시다.
2. **초안 생성(Draft)**: `M_p`를 사용하여 초안 확률 `p_1..p_N`과 함께 `d_{k+1}, d_{k+2}, ..., d_{k+N}`을 자기회귀(autoregressively) 방식으로 제안합니다.
3. **병렬 검증(Verify in parallel)**: `x_1..x_k, d_{k+1}, ..., d_{k+N}`에 대해 `M_q`를 한 번 실행하여, 위치 `k+1..k+N+1`에 대한 검증기 확률 `q_1..q_{N+1}`을 얻습니다.
4. **각 초안 토큰을 왼쪽에서 오른쪽으로 수락/거절(Accept/reject each draft token left to right)**: 각 `i`에 대해, `min(1, q_i(d_i) / p_i(d_i))`의 확률로 수락합니다.
5. 위치 `j`에서 첫 번째 거절이 발생하면: 정규화된 "잔차(residual)" 분포 `(q_j - p_j)_+`에서 `t_j`를 샘플링합니다. `j` 이후의 모든 초안은 폐기됩니다.
6. `N`개를 모두 수락하면: `q_{N+1}`에서 추가 토큰 하나 `t_{N+1}`을 샘플링합니다 (무료 보너스 토큰).

잔차 분포 트릭(residual distribution trick)은 출력이 마치 `M_q`가 처음부터 샘플링한 것과 정확히 동일한 분포를 유지하도록 만드는 수학적 통찰입니다.

### 속도 향상(Speedup)을 결정하는 요소

`α`를 초안 토큰(draft token)당 기대 수락률(expected acceptance rate)이라고 가정합니다. `c`를 초안 모델 대비 검증기(verifier)의 비용 비율(draft-to-verifier cost ratio)이라고 가정합니다. 단계별 계산은 다음과 같습니다.

- 나이브 생성(Naive generation)은 토큰당 1번의 거대 모델 호출을 수행합니다.
- 추측적 디코딩(Speculative decoding)은 `α`가 높을 때, 토큰당 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 번의 거대 모델 호출을 수행합니다.

`α = 0.75` 및 `N = 5`일 때의 일반적인 경험칙(rule of thumb): 거대 모델 호출 횟수가 3배 감소합니다. 초안 모델의 비용은 5배 저렴합니다. 결과적으로 전체 실행 시간(wall-clock time)은 약 2.5배 단축됩니다.

**`α`에 영향을 주는 요인:**

- 초안 모델이 검증기를 얼마나 잘 근사(approximate)하는가: 동일한 모델 계열(family)을 사용하거나 동일한 학습 데이터를 사용할 경우 `α`가 크게 향상됩니다.
- 디코딩 전략(Decoding strategy): 탐욕적(Greedy) 초안 모델과 탐욕적 검증기 조합은 `α`가 높습니다. 온도 샘플링(Temperature sampling)을 사용하면 일치시키기가 더 어려워져 수락률이 떨어집니다.
- 작업 유형(Task type): 코드 및 구조화된 출력(structured output)은 예측 가능성이 높아 수락률이 높고, 자유 형식의 창의적 글쓰기는 수락률이 낮습니다.

### Medusa — 초안 모델 없는 초안 생성 (drafts without a draft model)

Medusa는 검증기(verifier)에 추가적인 출력 헤드(output heads)를 배치하여 초안 모델(draft model)을 대체합니다. 위치 `t`에서의 구조는 다음과 같습니다:

```text
shared trunk → hidden h_t
├── head_0: t+1 위치의 토큰 예측 (표준 LM head)
├── head_1: t+2 위치의 토큰 예측
├── head_2: t+3 위치의 토큰 예측
├── head_3: t+4 위치의 토큰 예측
```

각 헤드는 고유한 로짓(logits)을 출력합니다. 추론 시에는 각 헤드에서 샘플링하여 후보 시퀀스(candidate sequence)를 얻은 다음, 모든 후보 연속(candidate continuations)을 한 번에 고려하는 트리 어텐션(tree-attention) 방식을 사용하여 단 한 번의 순전파(forward pass)로 검증합니다.

장점: 별도의 두 번째 모델이 필요 없습니다. 단점: 학습 가능한 파라미터가 추가됩니다. 지도 미세 조정(supervised fine-tuning) 단계(약 1B 토큰)가 필요하며, 수락률(acceptance rate)은 성능이 좋은 초안 모델을 사용하는 일반적인 추측적 디코딩(vanilla speculative decoding)보다 약간 낮습니다.

### EAGLE — 은닉 상태(hidden states) 재사용을 통한 더 나은 초안 생성

EAGLE-1/2/3 (Li et al., 2024–2025)는 검증기(verifier)의 마지막 레이어 은닉 상태를 입력받는 아주 작은 트랜스포머(일반적으로 1개 레이어)를 초안 모델(draft model)로 사용합니다. 초안 모델이 검증기의 특징 표현(feature representation)을 직접 확인하기 때문에, 초안 모델의 예측은 검증기의 출력 분포와 강력한 상관관계를 갖습니다. 이를 통해 수락률(acceptance rates)이 기존 방식(vanilla)의 ~0.6에서 0.85 이상으로 상승합니다.

EAGLE-3 (2025)는 후보 연속 문구(candidate continuations)에 대한 트리 탐색(tree search) 기능을 추가했습니다. vLLM과 SGLang은 Llama 3/4 및 Qwen 3를 위한 기본 사양 경로(default spec pathway)로 EAGLE-2/3를 탑재하여 제공합니다.

### KV 캐시 댄스 (The KV cache dance)

검증(Verification) 단계에서는 한 번의 순전파(forward pass)를 통해 `N`개의 초안 토큰(draft tokens)을 검증기에 전달합니다. 이는 검증기의 KV 캐시를 `N`개만큼 확장합니다. 만약 일부 초안이 거부되면, 캐시를 승인된 접두사(prefix) 길이로 롤백(roll back)해야 합니다.

실제 프로덕션 구현체(vLLM의 `--speculative-model`, TensorRT-LLM의 `LookaheadDecoder`)는 스크래치 KV 버퍼(scratch KV buffers)를 사용하여 이를 처리합니다. 먼저 기록한 뒤, 승인 시 커밋(commit)하는 방식입니다. 개념적으로 어렵지는 않지만, 구현 과정이 까다롭습니다.

```figure
draft-verify-tokens
```

## 구현하기 (Build It)

`code/main.py`를 확인해 보세요. 다음 요소들을 사용하여 핵심적인 추측적 샘플링(speculative-sampling) 알고리즘(거절 단계 + 잔차 분포)을 구현합니다:

- 수동으로 작성된 분포에 대해 결정론적 소프트맥스(deterministic-softmax)를 수행하는 "대형 모델(big model)" (이를 통해 수락 수학을 분석적으로 검증할 수 있습니다).
- 대형 모델을 섭동(perturbation)시킨 "초안 모델(draft model)".
- 직접 샘플링(direct sampling)과 동일한 주변 분포(marginal distribution)를 생성하는 수락/거절(acceptance / rejection) 루프.

### 1단계: 거절 단계 (the rejection step)

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`는 균등 분포 난수(uniform random number)입니다. `q_prob`는 검증기(verifier)가 초안 토큰(drafted token)에 대해 부여한 확률이며, `p_prob`는 초안 모델(draft model)의 확률입니다. Leviathan 정리에 따르면, 이러한 베르누이 결정(Bernoulli decision)을 수행하고 거절 시 잔차(residual)에서 샘플링을 진행하면 검증기의 분포를 정확하게 보존할 수 있습니다.

### 2단계: 잔차 분포 (residual distribution)

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

`q`에서 `p`를 요소별로 뺀 후, 음수 값은 0으로 고정(clamp)하고 다시 정규화(renormalize)합니다. 거절(rejection)이 발생할 때마다 이 분포에서 샘플링해 보세요.

### 3단계: 하나의 추측 단계 (one speculative step)

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5개의 토큰이 수락되면 → 1개의 보너스 토큰이 추가되어 → 검증기(verifier) 1회 통과 시 총 6개의 토큰이 생성됩니다.

### 4단계: 수락률(Acceptance Rate) 측정

다양한 초안 품질(draft-quality) 수준에서 10,000번의 추측 단계(speculative steps)를 실행합니다. 초안 분포와 검증기(verifier) 분포 사이의 KL 발산(KL divergence) 대비 수락률을 그래프로 그리세요. 두 지표 사이에서 명확한 단조 관계(monotone relationship)를 확인할 수 있을 것입니다.

### 5단계: 분포 등가성 검증 (Verify distribution equivalence)

경험적으로: 추측 루프(speculative loop)에 의해 생성된 토큰의 히스토그램은 검증기(verifier)에서 직접 샘플링하여 생성된 히스토그램과 일치해야 합니다. 이는 실무적인 Leviathan 정리입니다. 카이제곱 검정(chi-square test)을 통해 샘플링 오차 범위 내에서 일치함을 확인할 수 있습니다.

## 사용 방법 (Use It)

운영 환경(Production):

```bash
# EAGLE를 사용한 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# 일반(vanilla) 초안 모델을 사용한 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM은 2026년 중반 기준으로 가장 빠른 Medusa 경로를 제공합니다. `faster-whisper`는 작은 초안 모델을 사용하여 Whisper-large에 추측적 디코딩(speculative decoding)을 적용합니다.

**초안 모델 선택하기 (Picking a draft):**

| 전략 | 선택 시점 | 속도 향상 |
|----------|--------------|---------|
| 일반 초안 모델 (1B/3B Llama 계열) | 빠른 프로토타이핑, 별도 학습 없음 | 1.8–2.3× |
| Medusa 헤드 | 검증기(verifier)를 미세 조정할 수 있는 경우 | 2–3× |
| EAGLE-2 / 3 | 운영 환경, 최대 속도 필요 시 | 3–4× |
| Lookahead | 초안 모델 없음, 학습 없음, 추가 파라미터 없음 | 1.3–1.6× |

**추측적 디코딩을 사용하지 말아야 할 때 (When NOT to spec-decode):**

- 1~5개 토큰의 단일 시퀀스 생성 시. 오버헤드가 더 큽니다.
- 매우 창의적인 / 높은 온도의 샘플링(high-temperature sampling) 시 ($\alpha$ 값이 하락합니다).
- 메모리가 제한된 배포 환경 (초안 모델이 VRAM을 추가로 점유합니다).

## Ship It (실행하기)

`outputs/skill-spec-decode-picker.md`를 참조하세요. 이 스킬(skill)은 새로운 추론 워크로드에 대해 추측적 디코딩(speculative decoding) 전략(vanilla / Medusa / EAGLE / lookahead)과 튜닝 파라미터(`N`, `draft temperature`)를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하세요. 50,000개의 토큰에 대해 카이제곱 검정(chi-square test) 결과 $p > 0.05$를 만족하며, 추측된 토큰 분포(speculative token distribution)가 검증기(verifier)의 직접 샘플링 분포(direct-sample distribution)와 일치하는지 확인하세요.
2. **중간 (Medium).** $\alpha = 0.5, 0.7, 0.85$일 때, `N`의 함수로서 속도 향상(speedup, 빅 모델의 순전파당 토큰 수)을 그래프로 그리세요. 각 $\alpha$에 대한 최적의 `N`을 찾아보세요. (힌트: 검증 호출당 기대 토큰 수 = `(1 - α^{N+1}) / (1 - α)`입니다.)
3. **어려움 (Hard).** 아주 작은 규모의 Medusa를 구현해 보세요: Lesson 14의 캡스톤 GPT를 가져와서 $t+2, t+3, t+4$ 위치를 예측하는 3개의 추가 LM 헤드를 추가합니다. `tinyshakespeare` 데이터셋에서 공동 멀티헤드 손실(joint multi-head loss)을 사용하여 학습시키세요. 동일한 모델을 절단(truncating)하여 만든 일반적인 초안(vanilla draft) 모델과 수락률(acceptance rates)을 비교해 보세요.
4. **어려움 (Hard).** 롤백(rollback)을 구현해 보세요: 10개의 토큰 접두사(prefix) KV 캐시로 시작하여, 5개의 초안 토큰을 입력하고, 위치 3에서 거절(rejection)이 발생하는 상황을 시뮬레이션합니다. 다음 반복(iteration)에서 캐시 읽기(cache reads)가 "접두사 + 수락된 처음 2개의 초안"과 정확히 일치하는지 확인하세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| Draft model (초안 모델) | "저렴한 모델" | 후보 토큰을 제안하는 더 작은 모델; 일반적으로 검증기(verifier)보다 10~50배 저렴합니다. |
| Verifier (검증기) | "큰 모델" | 우리가 분포를 유지하고자 하는 대상 모델; 추측 단계(speculative step)당 한 번 실행됩니다. |
| Acceptance rate (α) (수락률) | "초안이 얼마나 자주 맞는지" | 검증기가 초안을 수락할 토큰당 확률. 일반적으로 0.7~0.9 사이입니다. |
| Residual distribution (잔차 분포) | "거절 시의 대체 수단" | 정규화된 `(q - p)_+`; 거절 시 이 분포에서 샘플링하면 검증기의 분포를 유지할 수 있습니다. |
| Bonus token (보너스 토큰) | "공짜 토큰" | N개의 초안이 모두 수락되었을 때, 검증기의 다음 단계 분포에서 하나를 더 샘플링합니다. |
| Medusa (메두사) | "초안 없는 추측" | 검증기에 여러 개의 LM 헤드를 두어 $t+1..t+k$ 위치를 병렬로 예측합니다. |
| EAGLE (이글) | "은닉 상태 초안" | 검증기의 마지막 레이어 은닉 상태(hidden states)를 조건으로 하는 아주 작은 트랜스포머 초안 모델입니다. |
| Lookahead decoding (룩어헤드 디코딩) | "자코비 반복법(Jacobi iteration)" | 고정점 반복(fixed-point iteration)을 사용하는 자기 투기(self-speculation) 방식; 초안 모델이 필요 없습니다. |
| Tree attention (트리 어텐션) | "여러 후보를 한 번에 검증" | 여러 초안 연속(continuations)을 동시에 고려하는 분기형 검증 방식입니다. |
| KV rollback (KV 롤백) | "거절된 초안 되돌리기" | 임시 KV 버퍼를 사용; 수락 시 커밋(commit)하고, 거절 시 폐기합니다. |

## 추가 읽을거리 (Further Reading)

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 핵심 알고리즘과 등가성 정리(equivalence theorem)를 다룹니다.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 동시에 발표된 논문이며, 깔끔한 베르누이 거절(Bernoulli-rejection) 증명을 제공합니다.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 논문; 트리 어텐션(tree-attention) 검증을 다룹니다.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; 은닉 상태 조건부 초안 생성(hidden-state-conditioned draft)을 다룹니다.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; 동적 트리 깊이(dynamic tree depth)를 다룹니다.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Lookahead 방식의 초안 없는(no-draft) 접근법을 다룹니다.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 네 가지 전략이 모두 구현된 표준적인 프로덕션 레퍼런스입니다.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3의 레퍼런스 코드입니다.
