# 추론적 디코딩(Speculative Decoding)과 EAGLE

> 프론티어 LLM이 토큰 하나를 생성하려면 수십억 개의 매개변수에 대한 완전한 순방향 전파가 필요합니다. 이 순방향 전파는 과도하게 자원 배분이 되어 있습니다: 대부분의 경우 훨씬 작은 모델이 다음 3-5개 토큰을 정확히 예측할 수 있으며, 큰 모델은 단지 그 예측을 *검증*하기만 하면 됩니다. 예측이 맞다면 토큰 5개를 토큰 1개의 비용으로 얻게 됩니다. 추론적 디코딩(Speculative Decoding)(Leviathan et al. 2023)은 이를 정밀하게 구현했으며, EAGLE-3 (2025)는 검증당 약 4.5개 토큰의 수용률로 끌어올려 — 일치하는 출력 분포에서 4-5배의 속도 향상을 달성했습니다.

**유형:** Build
**언어:** Python (numpy 포함)
**선수 요건:** 10단계 12강 (추론 최적화), 10단계 04강 (미니-GPT 사전 학습)
**시간:** 약 75분

## 문제점

H100에서의 70B급 모델의 디코딩 처리량은 일반적으로 초당 40-80 토큰입니다. 각 토큰은 HBM에서 모든 모델 가중치를 읽는 완전한 순방향 전파를 필요로 합니다. 모델의 출력을 변경하지 않고 모델을 더 작게 만들 수 없습니다. 메모리를 초과하여 배치 크기를 늘릴 수 없습니다. 막혀 있습니다 — 모델이 순방향 전파당 하나 이상의 토큰을 출력하도록 할 수 없는 한.

자기회귀(Autoregressive) 생성은 본질적으로 순차적인 것처럼 보입니다: `x_{t+1} = sample(p(· | x_{1:t}))`. 하지만 동시성 기회가 있습니다. "다음 4개 토큰은 아마도 [a, b, c, d]일 것이다"라고 말하는 값싼 예측기가 있다면, **큰 모델의 단일 순방향 전파**로 5개 위치를 모두 검증하고 가장 긴 일치 접두어를 수용할 수 있습니다.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding")는 타겟 모델의 샘플링 분포를 보존하는 영리한 수용/거절 규칙을 통해 이를 정밀하게 구현했습니다. 동일한 출력 분포, 2-4배 더 빠름.

## 개념

### 두 모델 설정

- **타겟 모델** `M_p`: 실제로 샘플링을 원하는 크고 느린 고품질 모델. 분포: `p(x)`.
- **초안 모델** `M_q`: 작고 빠르며 품질이 낮은 모델. 분포: `q(x)`. 5-30배 더 작음.

단계별:

1. 초안 모델이 `K`개의 토큰을 자기회귀(Autoregressive) 방식으로 제안합니다: `x_1, x_2, ..., x_K ~ q`.
2. 타겟 모델은 모든 `K+1` 위치에 대해 ONE forward pass를 병렬로 실행하여, 각 제안된 토큰에 대해 `p(x_k)`를 생성합니다.
3. 아래의 수정된 기각 샘플링 규칙을 통해 각 토큰을 왼쪽에서 오른쪽으로 순서대로 허용/기각합니다. 가장 긴 일치하는 접두어를 허용합니다.
4. 토큰이 기각되면, 수정된 분포에서 대체 토큰을 샘플링하고 중단합니다. 그렇지 않으면 `p(· | x_1...x_K)`에서 보너스 토큰 하나를 샘플링합니다.

초안(draft)이 타겟과 완벽하게 일치하면, 타겟 forward pass당 K+1개의 토큰을 얻습니다. 초안이 위치 1에서 틀리면, 토큰을 1개만 얻습니다.

### 정확성 규칙(The Exactness Rule)

추론적 디코딩(Speculative decoding)은 **p에서 샘플링하는 것과 분포상provably equivalent합니다.** 기각 규칙은 다음과 같습니다:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

여기서 `(p - q)+`는 pointwise 차이의 양수 부분을 나타냅니다. 초안과 타겟이 일치할 때(`p ≈ q`) 허용 확률은 거의 1입니다. 둘이 일치하지 않을 때, 잔여 분포는 전체 샘플이 정확히 `p`가 되도록 구성됩니다.

**Greedy case.** temperature=0 샘플링에서는 `argmax(p) == x_t`만 확인합니다. 일치하면 허용하고, 일치하지 않으면 `argmax(p)`를 출력하고 중단합니다.

### 예상 속도 향상(Expected Speedup)

초안 모델의 토큰 단위 허용률이 `α`라면, 타겟 forward pass당 생성되는 예상 토큰 수는 다음과 같습니다:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 초안 길이, α는 [0, 1]
```

`α = 0.8, K = 4`일 때: forward pass당 `(1 - 0.8^5)/(1 - 0.8) = 3.36`개의 토큰. 단일 타겟 forward pass 비용은 대략 `cost_q * K + cost_p`입니다 (K개의 초안 단계 + 타겟 검증 1회). `cost_p >> cost_q * K`이면 처리량(throughput)에 대한 속도 향상 비율은 `3.36× / 1 = 3.36×`입니다.

유일한 실제 매개변수는 `α`이며, 이는 전적으로 초안-타겟 정렬(alignment)에 의존합니다. 좋은 초안이 모든 것입니다.

### 초안 훈련: 증류(Distillation)

무작위 작은 모델은 초안으로 적합하지 않습니다. 표준 레시피는 타겟으로부터 증류(distill)하는 것입니다:

1. 작은 아키텍처를 선택합니다 (~1B는 70B 타겟용, ~500M는 7B 타겟용).
2. 타겟 모델을 대규모 텍스트 코퍼스에 대해 실행하고, 그 next-token 분포를 저장합니다.
3. 초안을 타겟의 분포에 대한 KL divergence로 훈련합니다 (ground-truth 토큰에 대한 것이 아닙니다).

결과: `α`는 코딩에서 보통 0.6-0.8, 자연어 채팅에서 0.7-0.85입니다. 프로덕션 환경에서 2-3배의 속도 향상을 얻습니다.

### EAGLE: 트리 초안 작성 + 기능 재사용

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")는 표준 추론적 디코딩(Speculative Decoding)에서 두 가지 비효율성을 관찰했습니다:

1. 초안 입력 = 위치 t에서의 타겟 최종 은닉 상태, 원시 토큰이 아님.
2. 초안 모델은 선형 체인을 출력합니다. 초안 모델이 후보의 *트리*를 출력할 수 있다면 (각 노드가 여러 추측을 가짐), 타겟 모델의 단일 순방향 패스는 트리 어텐션 마스크를 통해 여러 후보 경로를 병렬로 검증하고, 가장 긴 허용된 가지를 선택할 수 있습니다.

EAGLE-1의 변경 사항:
- 초안 입력 = 위치 t에서의 대상의 최종 은닉 상태이며, 원시 토큰이 아닙니다.
- 초안 아키텍처 = 트랜스포머 디코더 레이어 1개 (별도의 작은 모델이 아님).
- 출력 = 깊이 4-6, 각 깊이당 K = 4-8개의 후보 트리.

EAGLE-2 (2024)는 동적 트리 토폴로지를 추가합니다: 초안 모델이 불확실한 곳에서는 트리가 더 넓게 자라고, 확신하는 곳에서는 좁게 유지됩니다. 검증 비용을 증가시키지 않고 `α_effective`를 높입니다.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test")는 고정된 최상위 레이어 기능 의존성을 제거하고, 새로운 "테스트 시간 시뮬레이션" 손실로 초안 모델을 훈련합니다. 초안 모델은 교사 강제 훈련 분포가 아닌 타겟 모델의 테스트 시간 분포와 일치하는 출력으로 훈련됩니다. 허용률이 0.75 (EAGLE-2)에서 0.82 (EAGLE-3)로 상승하고, 평균 토큰/검증은 3.0에서 4.5로 증가합니다.

### 트리 어텐션 검증

초안 모델이 트리를 출력할 때, 타겟 모델은 **트리 어텐션 마스크**를 사용하여 단일 순방향 패스로 이를 검증합니다. 이는 순수한 선형이 아닌 트리 토폴로지를 인코딩하는 인과 마스크입니다. 각 토큰은 트리 내의 조상에만 어텐션합니다. 검증 패스는 여전히 하나의 순방향, 하나의 행렬 곱이며, 토폴로지 마스크는 몇 개의 추가 KV 항목 비용만 발생합니다.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

`a, b`가 경쟁하는 첫 번째 토큰 후보이고 `c, d, e, f`가 두 번째 토큰 후보라면, 모든 6개 위치가 하나의 순방향 패스로 검증됩니다. 출력은 허용된 경로 중 가장 긴 접두어입니다.

### 이 기법이 이길 때, 이기지 못할 때

**이기는 경우:**
- 예측 가능한 텍스트를 생성하는 채팅/완성 작업 (코드, 일반적인 영어, 구조화된 출력). `α`이 높습니다.
- 디코딩 중 GPU 연산이 남는 설정 (메모리 바운드 단계). 트리 초안 작성은 사용 가능한 FLOPs를 활용합니다.

**지는 경우 / 이득 없음:**
- 높은 확률적 출력 (고온에서의 창의적 글쓰기). `α`이 `1/|vocab|`로 떨어집니다.
- 매우 높은 동시성을 가진 배치 서빙 — 배치가 이미 FLOPs를 채우므로 트리 검증에 남는 여지가 거의 없습니다.
- 초안 모델이 타겟 모델보다 훨씬 작지 않은 매우 작은 타겟 모델.

프로덕션 환경에서는 일반적으로 채팅에서 2-3배의 벽시계(wall-clock) 속도 향상, 코드 생성에서 3-5배, 창의적 글쓰기에서는 거의 0배의 속도 향상을 보고합니다.

```figure
speculative-decoding
```

## 구현하기

`code/main.py`:

- 정확한 거부 규칙을 구현하고, 타겟의 분포를 보존하는지 검증하는 (평범한 타겟 샘플링 대비 경험적 KL < 0.01) 참고용 `speculative_decode(target, draft, prompt, K, temperature)`.
- top-p 분기로 깊이 K의 트리를 구축하는 EAGLE 스타일 트리 초안 작성기.
- 검증기를 위한 올바른 인과적 패턴을 생성하는 트리 어텐션 마스크 빌더.
- 초소형 LM (GPT-2-medium 타겟에서 GPT-2-small을 증류)에서 실행되는 수용률 테스트 하네스.

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. K개의 토큰 초안 작성
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 타겟이 모든 초안 위치 + 1개 추가 위치에서 p를 계산
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 좌에서 우로 순서대로 수용/거부
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. K개 모두 수용 → 타겟에서 보너스 토큰 샘플링
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 사용하기

- **vLLM**과 **SGLang**은 일급(first-class) 추론적 디코딩을 제공합니다. vLLM에서는 `--speculative-config`에 `method`, `model`, `num_speculative_tokens`를 포함한 JSON 객체를 전달하세요; EAGLE-3은 `"method": "eagle3"`입니다.
- **NVIDIA TensorRT-LLM**은 Medusa 및 EAGLE 트리를 네이티브로 지원합니다.
- **참고 초안 모델**: `Qwen/Qwen3-0.6B` (Qwen3-32B용 초안), `meta-llama/Llama-3.2-1B-Instruct` (Llama 3.x 70B용 초안).
- **Medusa 헤드** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): 초안 모델 대신 타겟 자체에 K개의 병렬 예측 헤드를 추가합니다. 배포가 더 간단하며, EAGLE보다 수용률이 약간 낮습니다.

## 출시하기

이 강의는 `outputs/skill-speculative-tuning.md`를 생성합니다. 이 스킬은 대상 모델의 워크로드를 프로파일링하고, 초안 모델, K (초안 길이), 트리 너비, 온도, 그리고 일반 디코딩으로 폴백할 시점을 선택합니다.

## 연습 문제

1. 정확한 기각 규칙을 구현하고 경험적으로 검증해 보세요. `speculative_decode`를 통해 10K 샘플을 실행하고 일반 대상 샘플링을 통해 실행한 후, 두 출력 분포 간의 TV 거리를 계산하세요. 값은 0.01 미만이어야 합니다.

2. 속도 향상 공식을 계산해 보세요. 고정된 `α`와 `K`에 대해 대상 순방향 전파당 예상 토큰 수를 플롯하세요. α ∈ {0.5, 0.7, 0.9}에 대한 최적의 K를 찾아보세요.

3. 작은 초안 모델을 훈련해 보세요. 124M GPT-2 대상 모델을 사용하여 100M 토큰에 대해 KL 손실로 30M GPT-2 초안 모델을 증류하세요. 홀드아웃 텍스트에서 `α`를 측정하세요. 예상 값은 0.6-0.7입니다.

4. EAGLE 스타일 트리 초안 생성을 구현해 보세요. 체인 대신 초안 모델이 각 깊이에서 상위 3개 분기를 출력하도록 하세요. 트리 어텐션 마스크를 구축하세요. 대상 모델이 가장 긴 올바른 분기를 수용하는지 검증하세요.

5. 실패 모드들을 측정해 보세요. temperature=1.5 (높은 확률적 성질)에서 추론적 디코딩을 실행하세요. α가 붕괴되고 초안 오버헤드 때문에 알고리즘이 일반 디코딩보다 느려지는 것을 보여주세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 대상 모델 | "큰 모델" | 샘플을 얻고자 하는 느리고 고품질의 모델 (p 분포) |
| 초안 모델 | "추론자" | 작고 빠른 예측자 (q 분포); 5-30배 더 작음 |
| K / 초안 길이 | "선제적 보기" | 검증 패스당 추론된 토큰 수 |
| α / 수용률 | "적중률" | 초안의 제안이 수용되는 토큰별 확률 |
| 정확한 기각 규칙 | "수용 테스트" | 대상의 분포를 보존하는 r < p/q 비교 |
| 잔여 분포 | "보정된 p-q" | (p - q)+ / ||(p - q)+||_1, 기각 시 샘플링할 분포 |
| 트리 초안 생성 | "분지 추론" | 초안이 후보 트리를 출력하며, 트리 구조 어텐션 마스크로 한 번의 패스에서 검증 |
| 트리 어텐션 마스크 | "위상학적 마스크" | 트리 위상을 인코딩하여 각 노드가 조상만 어텐션하도록 하는 인과 마스크 |
| Medusa 헤드 | "병렬 헤드" | 대상 모델 자체에 K개의 추가 예측 헤드; 별도 초안 모델 없음 |
| EAGLE 특징 재사용 | "은닉 상태 초안" | 초안 입력은 원시 토큰이 아닌 대상 모델의 마지막 은닉 상태이며, 초안 크기를 줄임 |
| 테스트 시간 시뮬레이션 손실 | "EAGLE-3 학습" | 교사 강제(teacher forcing)가 아닌, 대상 모델의 테스트 시간 분포와 일치하는 출력으로 초안을 학습 |

## 추가 읽기

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 정확한 거부 규칙 및 이론적 가속 분석
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind의 동시적 추론적 샘플링 논문
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 초안 모델의 대안인 병렬 헤드
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 특징 재사용 및 트리 초안 작성
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 동적 트리 토폴로지
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 학습 시간과 테스트 시간의 일치
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/선제적 디코딩, 스페크레이터가 필요 없는 대안
