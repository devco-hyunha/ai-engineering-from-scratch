# 추론적 디코딩(Speculative Decoding) — 초안 작성, 검증, 반복

> 자기회귀(Autoregressive) 디코딩은 순차적입니다. 각 토큰은 이전 토큰을 기다려야 합니다. 추론적 디코딩(Speculative Decoding)은 이 연쇄를 끊습니다: 값싼 모델이 N개의 토큰을 초안 작성하고, 값비싼 모델이 한 번의 순전파로 N개 전부를 검증합니다. 초안이 정확하다면 N번의 생성을 위해 한 번의 큰 순전파 비용을 지불한 셈입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 07강 (GPT 인과 LM), 7단계 · 12강 (KV 캐시 및 Flash Attention)
**시간:** 약 60분

## 문제점

70B LLM이 H100에서 토큰 하나를 샘플링하는 데 약 30ms가 걸립니다. 3B 초안 모델은 약 3ms가 걸립니다. 3B 모델이 5개 토큰을 미리 초안 작성한 후, 70B 모델을 *한 번* 실행하여 5개 전부를 검증하면, 최대 5개 토큰이 승인될 경우 총 시간은 `5×3 + 30 = 45 ms`입니다 — 직선 생성(straight-line generation)의 `5×30 = 150 ms`과 비교됩니다. 이것이 추론적 디코딩(Speculative Decoding)의 전체 제안입니다: 작은 양의 추가 GPU 메모리(초안 모델)를 사용하여 디코딩 지연 시간을 2–4배 낮추는 것입니다.

이 기법은 분포를 보존해야 합니다. Leviathan et al. (2023)과 Chen et al.이 동시에 도입한 추론적 샘플링(Speculative Sampling)은 출력 시퀀스가 큰 모델이 단독으로 생성했을 것과 **동일하게 분포**됨을 보장합니다. 품질 트레이드오프는 없습니다. 단지 더 빠를 뿐입니다.

2026년 추론에서는 초안-검증자 쌍의 네 가지 계열이 지배적입니다:

1. **바닐라 추론적 디코딩(Vanilla speculative, Leviathan 2023).** 분리된 초안 모델(예: Llama 3 1B) + 검증자(예: Llama 3 70B).
2. **Medusa (Cai 2024).** 검증자의 여러 디코딩 헤드가 `t+1..t+k` 위치를 병렬로 예측합니다. 분리된 초안 모델이 없습니다.
3. **EAGLE 계열 (Li 2024, 2025).** 검증자의 은닉 상태를 재사용하는 경량 초안; 바닐라보다 승인율이 높음; 일반적으로 3–4배.
4. **Lookahead 디코딩(Lookahead decoding, Fu 2024).** 야코비 반복(Jacobi iteration); 초안 모델이 전혀 필요하지 않습니다. 자기 추론(Self-speculation). 니치이지만 의존성이 없습니다.

2026년 모든 프로덕션 추론 스택은 추론적 디코딩(Speculative Decoding)을 기본으로 탑재합니다. vLLM, TensorRT-LLM, SGLang, llama.cpp는 모두 최소한 바닐라 + EAGLE-2를 지원합니다.

## 개념

### 핵심 알고리즘

검증자 `M_q`와 더 값싼 초안 `M_p`가 주어지면:

1. `x_1..x_k`를 이미 디코딩된 접두어(prefix)라고 합시다.
2. **초안 생성**: `M_p`를 사용하여 `d_{k+1}, d_{k+2}, ..., d_{k+N}`를 초안 확률 `p_1..p_N`으로 자기회귀적으로 제안합니다.
3. **병렬 검증**: `M_q`를 `x_1..x_k, d_{k+1}, ..., d_{k+N}`에 대해 한 번 실행하여, 위치 `k+1..k+N+1`에 대한 검증자 확률 `q_1..q_{N+1}`를 얻습니다.
4. **각 초안 토큰을 왼쪽에서 오른쪽으로 수락/거부**: 각 `i`에 대해 확률 `min(1, q_i(d_i) / p_i(d_i))`으로 수락합니다.
5. 위치 `j`에서 첫 번째 거부가 발생하면: 정규화된 "잔여" 분포 `(q_j - p_j)_+`에서 `t_j`를 샘플링합니다. `j` 이후의 모든 초안은 폐기됩니다.
6. 모든 `N`을 수락하면: `q_{N+1}`에서 추가 토큰 `t_{N+1}`을 샘플링합니다 (무료 보너스 토큰).

잔여 분포 트릭은 `M_q`가 처음부터 샘플링한 것과 정확히 동일한 분포로 출력이 유지되도록 하는 수학적 통찰입니다.

### 속도 향상을 결정하는 요소

`α` = 초안 토큰당 기대 수락률이라고 합시다. `c` = 초안 대 검증자 비용 비율이라고 합시다. 단계별:

- 소박한 생성은 토큰당 대형 모델 호출을 1번 수행합니다.
- 추론적 디코딩(Speculative Decoding)은 `α`이 높을 때 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 토큰마다 대형 모델 호출을 1번 수행합니다.

`α = 0.75` 및 `N = 5`에서의 일반적인 경험칙: 대형 모델 호출이 3배 적습니다. 초안 비용은 5배 저렴합니다. 전체 벽시계 시간이 약 2.5배 감소합니다.

**α는 다음에 따라 달라집니다:**

- 초안이 검증자를 얼마나 잘 근사하는지. 동일한 계열 / 동일한 학습 데이터는 α를 크게 높입니다.
- 디코딩 전략. 탐욕 초안 대 탐욕 검증자: 높은 α. 온도 샘플링: 매칭이 어려움; 수락률이 떨어집니다.
- 작업 유형. 코드 및 구조화된 출력은 더 많이 수락됩니다 (예측 가능함); 자유 형식 창의적 글쓰기는 더 적게 수락됩니다.

### Medusa — 초안 모델 없는 초안 생성

Medusa는 초안 모델을 검증자의 추가 출력 헤드로 대체합니다. 위치 `t`에서:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

각 헤드는 자체 로짓(Logits)을 출력합니다. 추론 시 각 헤드에서 샘플링하여 후보 시퀀스를 얻은 후, 모든 후보 연속성을 한 번에 고려하는 트리 어텐션(tree-attention) 방식을 사용하여 한 번의 순전파로 검증합니다.

장점: 두 번째 모델이 없음. 단점: 학습 가능한 매개변수(Parameter)가 추가됨; 지도 미세 조정 (SFT)(Supervised Fine-Tuning) 단계(~1B 토큰)가 필요함; 좋은 초안 모델을 사용한 순수 추론적 디코딩(Speculative Decoding)보다 수락률이 약간 낮음.

### EAGLE — 은닉 상태를 재사용하여 더 나은 초안 생성

EAGLE-1/2/3 (Li et al., 2024–2025)은 초안 모델을 초안 모델이 검증자의 마지막 레이어 은닉 상태를 입력으로 받는 초소형 트랜스포머(일반적으로 1 레이어)로 만듭니다. 초안 모델이 검증자의 특징 표현을 보기 때문에, 초안 모델의 예측은 검증자의 출력 분포와 강하게 상관됩니다. 수용률은 ~0.6 (바닐라)에서 0.85+로 상승합니다.

EAGLE-3 (2025)는 후보 연속성에 대한 트리 검색을 추가했습니다. vLLM과 SGLang은 Llama 3/4 및 Qwen 3에 대해 EAGLE-2/3를 기본 추론적 디코딩 경로로 제공합니다.

### KV 캐시 무브

검증은 `N` 초안 토큰을 한 번의 순방향 패스로 검증자에 입력합니다. 이는 검증자의 KV 캐시를 `N` 항목만큼 확장합니다. 일부 초안이 거부되면, 캐시를 수용된 접두어 길이로 되돌려야 합니다.

프로덕션 구현(vLLM의 `--speculative-config`, TensorRT-LLM의 LookaheadDecoder)은 이 문제를 스크래치 KV 버퍼로 처리합니다. 먼저 기록하고, 수용 시에 커밋합니다. 개념적으로 어렵지는 않지만, 세밀한 처리가 필요합니다.

```figure
draft-verify-tokens
```

## 구현하기

`code/main.py`를 참조하세요. 우리는 거부 단계 + 잔여 분포를 포함하는 핵심 추론적 샘플링 알고리즘을 다음을 사용하여 구현합니다:

- 수락 수학을 분석적으로 검증할 수 있도록, 수작업으로 코딩된 분포에 대한 결정적 소프트맥스인 "큰 모델".
- 큰 모델의 섭동인 "초안 모델".
- 직접 샘플링과 동일한 주변 분포를 생성하는 수용/거부 루프.

### 1단계: 거부 단계

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`는 균일한 난수입니다. `q_prob`은 초안 토큰에 대한 검증자의 확률입니다. `p_prob`는 초안 모델의 확률입니다. Leviathan 정리에 따르면, 이 베르누이 결정이 거부 시 잔여 분포에서 샘플링하는 것과 결합되면, 검증자의 분포를 정확히 보존합니다.

### 2단계: 잔여 분포

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

`q`에서 `p`를 요소별로 빼고, 음수 값을 0으로 클램프한 후, 재정규화합니다. 거부 시 이 분포에서 샘플링합니다.

### 3단계: 한 번의 추론적 단계

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

5개 수용 → 1개 보너스 → 한 번의 검증자 패스로 6개 토큰 생성.

### 4단계: 수용률 측정

다양한 초안 품질 수준에서 10,000개의 추론적 스텝을 실행해 보세요. 초안과 검증자 분포 간의 KL 발산에 대한 수용률을 플롯해 보세요. 명확한 단조 관계가 나타나는 것을 확인할 수 있습니다.

### 5단계: 분포 동등성 검증

경험적으로: 추론적 루프가 생성한 토큰의 히스토그램은 검증자에서 직접 샘플링한 히스토그램과 일치해야 합니다. 이는 Leviathan 정리의 실제 적용입니다. 카이제곱 검정은 샘플링 오차 범위 내에서 이를 확인합니다.

## 사용하기

프로덕션:

```bash
# EAGLE를 사용한 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# 바닐라 초안 모델을 사용한 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

TensorRT-LLM은 2026년 중반 기준으로 가장 빠른 Medusa 경로를 제공합니다. `faster-whisper`는 작은 초안 모델로 Whisper-large에 추론적 디코딩을 래핑합니다.

**초안 선택:**

| 전략 | 선택 시점 | 속도 향상 |
|----------|--------------|---------|
| 바닐라 초안 (1B/3B Llama 계열) | 빠른 프로토타입, 학습 불필요 | 1.8–2.3× |
| Medusa 헤드 | 검증자를 미세 조정할 수 있음 | 2–3× |
| EAGLE-2 / 3 | 프로덕션, 최대 속도 | 3–4× |
| Lookahead | 초안 없음, 학습 없음, 추가 매개변수 없음 | 1.3–1.6× |

**추론적 디코딩을 사용하지 말아야 할 경우:**

- 1–5개 토큰의 단일 시퀀스 생성. 오버헤드가 지배적입니다.
- 극도로 창의적인 / 고온도 샘플링 (α가 감소합니다).
- 메모리 제약이 있는 배포 환경 (초안 모델이 VRAM을 추가합니다).

## 출시하기

`outputs/skill-spec-decode-picker.md`를 참조하세요. 이 스킬은 새로운 추론 워크로드에 대해 추론적 디코딩 전략 (바닐라 / Medusa / EAGLE / lookahead) 및 튜닝 매개변수 (N, 초안 온도)를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 50,000개 토큰에서 추론적 토큰 분포가 검증자의 직접 샘플 분포와 카이제곱 p > 0.05 범위 내에서 일치하는지 확인하세요.
2. **중간.** `N`에 대한 속도 향상(큰 모델의 순방향 전파당 토큰 수)을 `α = 0.5, 0.7, 0.85`의 함수로 플롯하세요. 각 α에 대한 최적의 `N`를 식별하세요. (힌트: 검증 호출당 예상 토큰 수 = `(1 - α^{N+1}) / (1 - α)`.)
3. **난이도: 상.** 작은 Medusa를 구현해 보세요. 14강의 캡스톤 GPT를 가져와 t+2, t+3, t+4 위치를 예측하는 추가 LM 헤드 3개를 더합니다. tinyshakespeare로 결합 다중 헤드 손실로 학습합니다. 같은 모델을 잘라 만든 바닐라 초안과 수용률을 비교합니다.
4. **난이도: 상.** 롤백을 구현해 보세요. 10토큰 접두어 KV 캐시로 시작하여 5개의 초안 토큰을 입력하고, 위치 3에서 거절을 시뮬레이션합니다. 다음 반복에서 캐시 읽기가 "접두어 + 처음 2개 수용된 초안"과 정확히 일치하는지 확인합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 초안 모델 | "저렴한 모델" | 후보 토큰을 제안하는 더 작은 모델; 일반적으로 검증기보다 10–50배 저렴합니다. |
| 검증기 | "큰 모델" | 분포를 보존하는 대상 모델; 스펙큘레이티브 스텝마다 한 번 실행됩니다. |
| 수용률 (α) | "초안이 얼마나 자주 맞는지" | 토큰별 확률로, 검증기가 초안을 수용하는 빈도입니다. 일반적으로 0.7–0.9입니다. |
| 잔여 분포 | "거절 시 폴백" | `(q - p)_+` 정규화; 거절 시 이 분포에서 샘플링하면 검증기의 분포가 보존됩니다. |
| 보너스 토큰 | "무료 토큰" | N개의 초안이 모두 수용되면, 검증기의 다음 단계 분포에서 하나를 더 샘플링합니다. |
| Medusa | "초안 없는 스펙큘레이티브" | 검증기에 여러 LM 헤드를 두어 t+1..t+k 위치를 병렬로 예측합니다. |
| EAGLE | "은닉 상태 초안" | 검증기의 마지막 레이어 은닉 상태에 조건을 둔 작은 트랜스포머 초안입니다. |
| 룽어헤드 디코딩 | "야코비 반복" | 고정점 반복을 사용하는 자기 스펙큘레이션; 초안 모델이 없습니다. |
| 트리 어텐션 | "많은 후보를 한 번에 검증" | 여러 초안 연속을 동시에 고려하는 분기 검증입니다. |
| KV 롤백 | "거절된 초안 되돌리기" | 스크래치 KV 버퍼; 수용 시 커밋하고, 거절 시 폐기합니다. |

## 추가 읽기

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 핵심 알고리즘 및 동치 정리.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 동시 도입; 깔끔한 베르누이 거절 증명.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 논문; 트리 어텐션 검증.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; 은닉 상태 조건 초안.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; 동적 트리 깊이입니다.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3입니다.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 선제적 탐색, 초안 없는 접근 방식입니다.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 네 가지 전략이 모두 연결된 표준 생산 참조입니다.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3의 참조 코드입니다.
