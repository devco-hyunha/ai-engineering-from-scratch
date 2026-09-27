# 전문가 혼합 (Mixture of Experts, MoE)

> 밀집(Dense) 구조의 70B 트랜스포머는 모든 토큰마다 모든 파라미터를 활성화합니다. 반면 671B MoE는 토큰당 37B만 활성화하면서도 모든 벤치마크에서 이를 능가합니다. 희소성(Sparsity)은 이번 10년에서 가장 중요한 스케일링 아이디어입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 문제점 (The Problem)

Dense Transformer의 추론 시 FLOPs는 파라미터 수와 동일합니다(순전파를 위해 2배 적용). Dense 모델의 규모를 키우면 모든 토큰이 그 비용을 온전히 지불해야 합니다. 2024년경에 이르러 프런티어 모델들은 연산량의 벽(compute wall)에 부딪혔습니다. 즉, 유의미하게 더 똑똑해지기 위해서는 토큰당 기하급수적으로 더 많은 FLOPs가 필요했습니다.

Mixture of Experts(MoE)는 이 연결 고리를 끊어냅니다. 각 FFN을 `E`개의 독립적인 전문가(experts)와 토큰당 `k`개의 전문가를 선택하는 라우터(router)로 교체합니다. 총 파라미터 수 = `E × FFN_size`입니다. 토큰당 활성화되는 파라미터 수 = `k × FFN_size`입니다. 전형적인 2026년 구성은 `E=256`, `k=8`입니다. 저장 용량은 `E`에 따라 확장되지만, 연산량은 `k`에 따라 확장됩니다.

2026년의 프런티어 모델들은 거의 전적으로 MoE 방식입니다: DeepSeek-V3 (총 671B / 활성 37B), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss 등이 있습니다. Artificial Analysis의 독립 리더보드에서 상위 10개 오픈 소스 모델은 모두 MoE 모델입니다.

## 개념 (The Concept)

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 교체 (The FFN swap)

Dense 트랜스포머 블록(Dense transformer block):

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 블록(MoE block):

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 토큰당 E개 중 k개를 선택
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

각 전문가(expert)는 독립적인 FFN(일반적으로 SwiGLU)입니다. 라우터(router)는 단일 선형 레이어(linear layer)입니다. 각 토큰은 자신만의 `k`개 전문가를 선택하며, 선택된 전문가들의 출력값에 게이트(gate)가 적용된 혼합물(mixture)을 얻습니다.

### 부하 분산 문제 (The load-balancing problem)

만약 라우터가 토큰의 90%를 전문가 3(expert 3)으로 보낸다면, 나머지 전문가들은 사용되지 못하고 방치(starve)됩니다. 이를 해결하기 위해 세 가지 방법이 시도되었습니다:

1. **보조 부하 분산 손실 (Auxiliary load-balancing loss)** (Switch Transformer, Mixtral). 전문가 사용량의 분산에 비례하는 페널티를 추가합니다. 효과는 있지만, 하이퍼파라미터와 두 번째 그래디언트 신호(gradient signal)가 추가된다는 단점이 있습니다.
2. **전문가 용량 + 토큰 드롭 (Expert capacity + token dropping)** (초기 Switch). 각 전문가는 최대 `C × N/E`개의 토큰만 처리하며, 용량을 초과하는 토큰은 해당 레이어를 건너뜁니다. 이는 품질 저하를 야기합니다.
3. **보조 손실 없는 균형 조정 (Auxiliary-loss-free balancing)** (DeepSeek-V3). 라우터의 top-k 선택을 이동시키는 학습 가능한 전문가별 편향(per-expert bias)을 추가합니다. 편향은 학습 손실(training loss) 외부에서 업데이트됩니다. 메인 목적 함수(main objective)에 페널티를 주지 않습니다. 2024년의 중요한 돌파구(big unlock)입니다.

DeepSeek-V3의 접근 방식: 매 학습 단계 이후, 모든 전문가에 대해 사용량이 목표치보다 높은지 낮은지 확인합니다. 편향을 `±γ`만큼 조정합니다. 선택 시에는 `scores + bias`를 사용합니다. 게이팅(gating)에 사용되는 전문가 확률은 변경되지 않은 원본 `scores`를 그대로 사용합니다. 이를 통해 라우팅(routing)과 표현(expression)을 분리합니다.

### 공유 전문가 (Shared experts)

DeepSeek-V2/V3는 전문가(experts)를 *공유(shared)* 전문가와 *라우팅(routed)* 전문가로 분리합니다. 모든 토큰은 모든 공유 전문가를 통과합니다. 라우팅 전문가는 top-k 방식을 통해 선택됩니다. 공유 전문가는 공통 지식을 포착하며, 라우팅 전문가는 특정 분야에 특화됩니다. V3는 1개의 공유 전문가와 256개의 라우팅 전문가 중 상위 8개(top-8)를 실행합니다.

### 세밀한 전문가 (Fine-grained experts)

클래식 MoE (GShard, Switch): 각 전문가(expert)는 전체 FFN만큼의 너비를 가집니다. `E`는 작고 (8–64), `k`도 작습니다 (1–2).

현대적인 세밀한 MoE (DeepSeek-V3, Qwen-MoE): 각 전문가가 더 좁습니다 (FFN 크기의 1/8). `E`는 크고 (256+), `k`도 더 큽니다 (8+). 전체 파라미터 수는 동일하지만, 조합의 경우 훨씬 더 빠르게 확장됩니다. 토큰당 가능한 "전문가" 조합은 `C(256, 8) = 400 trillion`에 달합니다. 품질은 향상되면서 지연 시간(latency)은 일정하게 유지됩니다.

### 비용 프로필 (The cost profile)

토큰당, 레이어당:

| 설정 (Config) | 토큰당 활성 파라미터 (Active params / token) | 총 파라미터 (Total params) |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3는 **토큰당 더 적은 활성 FLOPs**를 수행하면서도 거의 모든 벤치마크에서 Llama 3 70B (dense)를 능가합니다. 더 많은 파라미터는 더 많은 지식을 의미합니다. 더 많은 활성 FLOPs는 토큰당 더 많은 연산을 의미합니다. MoE는 이 둘을 분리합니다.

### 주의 사항: 메모리 (The catch: memory)

모든 전문가(experts)는 활성화 여부와 관계없이 GPU 상에 상주합니다. 671B 모델의 경우 `fp16` 가중치를 위해 약 1.3 TB의 VRAM이 필요합니다. 최첨단(Frontier) MoE 배포에는 전문가 병렬화(expert parallelism)가 필수적입니다. 즉, 전문가들을 여러 GPU에 분산(shard)시키고, 네트워크를 통해 토큰을 라우팅해야 합니다. 이때 지연 시간(latency)은 행렬 곱셈(matmul)이 아닌, 올투올(all-to-all) 통신에 의해 결정됩니다.

```figure
expert-routing
```

## 구현하기 (Build It)

`code/main.py`를 참조하세요. 표준 라이브러리만을 사용하여 구현한 간결한 MoE 레이어이며, 다음을 포함합니다:

- `n_experts=8`개의 SwiGLU 스타일 전문가 (설명을 위해 각 전문가당 하나의 `linear` 레이어 사용)
- top-k=2 라우팅
- softmax로 정규화된 게이팅 가중치(gating weights)
- 전문가별 편향(per-expert bias)을 통한 보조 손실(auxiliary loss) 없는 균형 조절

### 1단계: 라우터 (the router)

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 선택된 전문가들의 원본(ORIGINAL) 점수에 대해 softmax 수행
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

`bias`는 게이트 가중치(gate weight)가 아닌 선택(selection)에 영향을 미칩니다. 이것이 바로 DeepSeek-V3의 트릭입니다. `bias`는 모델의 예측을 왜곡하지 않으면서 부하 불균형(load imbalance)을 교정합니다.

### 2단계: 라우터를 통해 100개의 토큰 실행하기

어떤 전문가(expert)가 얼마나 자주 활성화되는지 추적합니다. 편향(bias)이 없다면 사용량이 왜곡됩니다. 편향 업데이트 루프(과사용된 전문가는 `-γ`, 미사용된 전문가는 `+γ`)를 적용하면, 몇 번의 반복(iteration) 만에 사용량이 균등 분포(uniform distribution)로 수렴합니다.

### 3단계: 파라미터 수 비교 (param count comparison)

MoE 설정의 "dense equivalent(밀집 모델 대응치)"를 출력합니다. DeepSeek-V3 구조를 기준으로 합니다: 256개의 routed expert + 1개의 shared expert, 8개의 active expert, `d_model`=7168. 전체 파라미터 수는 엄청난 규모입니다. 반면 active 파라미터 수는 dense Llama 3 70B의 7분의 1 수준입니다.

## 사용 방법 (Use It)

HuggingFace 로드:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026년 프로덕션 추론(Production inference): vLLM은 MoE 라우팅을 기본적으로 지원합니다. SGLang은 가장 빠른 전문가 병렬화(expert-parallel) 경로를 제공합니다. 두 라이브러리 모두 `top-k` 선택과 전문가 병렬화를 자동으로 처리합니다.

**MoE를 선택해야 하는 경우:**
- 토큰당 추론 비용을 낮추면서 최첨단(frontier) 품질을 원하는 경우.
- VRAM 또는 전문가 병렬화(expert-parallel) 인프라를 갖춘 경우.
- 워크로드가 컨텍스트 중심(긴 문서)이 아닌 토큰 중심(채팅, 코드)인 경우.

**MoE를 선택하지 말아야 하는 경우:**
- 엣지 배포(Edge deployment) — 활성화된 FLOP에 대해 전체 저장 용량만큼의 비용을 지불해야 합니다.
- 지연 시간(Latency)이 중요한 단일 사용자 서빙 — 전문가 라우팅이 오버헤드를 추가합니다.
- 소형 모델(<7B) — MoE의 품질 이점은 계산 임계값(활성 파라미터 약 6B 이상)을 넘어야 나타납니다.

## Ship It (실전 적용)

`outputs/skill-moe-configurator.md`를 참조하세요. 이 스킬은 주어진 파라미터 예산(parameter budget), 학습 토큰(training tokens), 그리고 배포 대상(deployment target)을 바탕으로 새로운 MoE를 위한 `E`, `k`, 그리고 `shared-expert` 레이아웃을 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 50번의 반복(iteration) 동안 보조 손실이 없는 편향 업데이트(auxiliary-loss-free bias update)가 전문가 사용량(expert usage)을 어떻게 균등하게 만드는지 관찰해 보세요.
2. **중간 (Medium).** 학습된 라우터(learned router)를 해시 기반 라우터(hash-based router, 결정론적이며 학습하지 않음)로 교체해 보세요. 품질과 균형(balance)을 비교해 보세요. 왜 학습된 라우터가 더 나은 성능을 보일까요?
3. **어려움 (Hard).** GRPO 스타일의 "롤아웃 일치 라우팅(rollout-matched routing)"(DeepSeek-V3.2 트릭)을 구현해 보세요: 추론(inference) 중에 어떤 전문가가 활성화되는지 기록하고, 그래디언트 계산(gradient computation) 시 동일한 라우팅이 강제되도록 합니다. 토이 정책 경사(toy policy-gradient) 설정에서 그 효과를 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 전문가 (Expert) | "여러 개 중 하나의 FFN" | 독립적인 피드포워드 네트워크(FFN); FFN 연산의 희소한 슬라이스(sparse slice)에 할당된 파라미터입니다. |
| 라우터 (Router) | "게이트(gate)" | 각 토큰을 각 전문가와 대조하여 점수를 매기는 아주 작은 선형 레이어(linear layer)이며, top-k 선택을 수행합니다. |
| Top-k 라우팅 (Top-k routing) | "토큰당 k개의 활성 전문가" | 각 토큰의 FFN 연산은 게이트 가중치에 따라 정확히 k개의 전문가를 거칩니다. |
| 보조 손실 (Auxiliary loss) | "부하 분산 페널티(Load-balance penalty)" | 전문가 사용이 불균형할 경우 페널티를 부여하는 추가 손실 항입니다. |
| 보조 손실 없는 방식 (Auxiliary-loss-free) | "DeepSeek-V3의 트릭" | 추가적인 그래디언트 없이, 라우터의 선택 시 전문가별 편향(bias)만을 통해 균형을 맞춥니다. |
| 공유 전문가 (Shared expert) | "항상 켜져 있는 상태" | 모든 토큰이 통과하는 추가 전문가이며, 공통 지식을 포착합니다. |
| 전문가 병렬화 (Expert parallelism) | "전문가 단위 샤딩(Shard by expert)" | 서로 다른 전문가를 서로 다른 GPU에 분산 배치하며, 네트워크를 통해 토큰을 라우팅합니다. |
| 희소성 (Sparsity) | "활성 파라미터 < 전체 파라미터" | `k × expert_size / (E × expert_size)`의 비율을 의미합니다. DeepSeek-V3의 경우 37/671 ≈ 5.5%입니다. |

## 추가 학습 자료 (Further Reading)

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 핵심 아이디어.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — 클래식한 MoE 모델인 Switch Transformer.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 보조 손실 없는(auxiliary-loss-free) MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 편향(bias) 기반 균형 조정 논문.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 본 강의의 라우터가 사용하는 세밀한(fine-grained) 전문가 분할 및 공유 전문가(shared-expert) 방식.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 공유 전문가(shared-expert) 방식의 원천 논문.
