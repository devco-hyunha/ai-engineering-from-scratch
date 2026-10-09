# MoE (혼합 전문가)(MoE (Mixture of Experts))

> 70B 밀집 트랜스포머는 모든 토큰에 대해 모든 매개변수를 활성화합니다. 671B MoE는 토큰당 37B만 활성화하며 모든 벤치마크에서 이를 능가합니다. 희소성은 이 10년간 가장 중요한 확장 아이디어입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 7단계 · 05강 (전체 트랜스포머), 7단계 · 07강 (GPT)
**시간:** 약 45분

## 문제점

밀집 트랜스포머의 추론 시 FLOPs는 매개변수 수와 같습니다 (순전파의 경우 2배). 밀집 모델을 확장하면 모든 토큰이 전체 비용을 지불해야 합니다. 2024년경 프론티어는 컴퓨팅 한계에 부딪혔습니다. 의미 있게 더 똑똑해지려면 토큰당 FLOPs가 지수적으로 증가해야 했습니다.

MoE (혼합 전문가)(MoE (Mixture of Experts))는 이 연결을 끊습니다. 각 FFN을 `E`개의 독립적인 전문가와 토큰당 `k`개의 전문가를 선택하는 라우터로 교체합니다. 총 매개변수 = `E × FFN_size`. 토큰당 활성 매개변수 = `k × FFN_size`. 일반적인 2026년 구성: `E=256`, `k=8`. 저장소는 `E`에 따라 확장되고, 컴퓨팅은 `k`에 따라 확장됩니다.

2026년 프론티어는 거의 전적으로 MoE입니다: DeepSeek-V3 (총 671B / 활성 37B), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Artificial Analysis의 독립적인 리더보드에서 상위 10개 오픈소스 모델은 모두 MoE입니다.

## 개념

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 교체

밀집 트랜스포머 블록:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 블록:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 토큰당 E 중 k개 선택
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

각 전문가는 독립적인 FFN (일반적으로 SwiGLU)입니다. 라우터는 단일 선형 레이어입니다. 각 토큰은 자체 `k`개의 전문가를 선택하고, 그 출력의 게이트된 혼합을 받습니다.

### 부하 균형 문제

라우터가 토큰의 90%를 전문가 3으로 보내면, 다른 전문가들은 기아 상태가 됩니다. 세 가지 해결책이 시도되었습니다:

1. **보조 부하 균형 손실** (Switch Transformer, Mixtral). 전문가 사용량의 분산에 비례하는 페널티를 추가합니다. 작동하지만, 하이퍼파라미터와 두 번째 기울기 신호가 추가됩니다.
2. **전문가 용량 + 토큰 드롭아웃** (초기 Switch). 각 전문가는 최대 `C × N/E`개의 토큰을 처리합니다. 초과 토큰은 레이어를 건너뜁니다. 품질이 저하됩니다.
3. **보조 손실 없는 균형 조정**(Auxiliary-loss-free balancing) (DeepSeek-V3). 라우터의 top-k 선택을 이동시키는 학습된 전문가별 편향을 추가합니다. 편향은 학습 손실 외부에서 업데이트됩니다. 주요 목적 함수에 페널티가 없습니다. 2024년의 주요 돌파구입니다.

DeepSeek-V3의 접근 방식: 각 학습 단계 후, 모든 전문가에 대해 사용량이 목표보다 높은지 낮은지 확인합니다. 편향을 `±γ`만큼 조정합니다. 선택은 `scores + bias`을 사용합니다. 게이팅에 사용되는 전문가 확률은 원본 `scores`을 변경하지 않은 상태로 사용합니다. 라우팅과 표현을 분리합니다.

### 공유 전문가

DeepSeek-V2/V3는 또한 전문가를 *공유*와 *라우팅*으로 분리합니다. 모든 토큰은 모든 공유 전문가를 통과합니다. 라우팅된 전문가는 top-k로 선택됩니다. 공유 전문가는 공통 지식을 포착하고, 라우팅된 전문가는 특화됩니다. V3는 1개의 공유 전문가와 256개 라우팅 전문가 중 top-8을 실행합니다.

### 세분화된 전문가

고전적 MoE (GShard, Switch): 각 전문가는 전체 FFN만큼 넓습니다. `E`는 작습니다 (8–64), `k`는 작습니다 (1–2).

현대적 세분화된 MoE (DeepSeek-V3, Qwen-MoE): 각 전문가는 더 좁습니다 (FFN 크기의 1/8). `E`는 큽니다 (256+), `k`는 더 큽니다 (8+). 총 매개변수 수는 동일하지만, 조합이 훨씬 더 빠르게 확장됩니다. 토큰당 `C(256, 8) = 400 trillion`개의 "전문가"가 가능합니다. 품질은 향상되고 지연 시간은 평평하게 유지됩니다.

### 비용 프로필

토큰별, 레이어별:

| 구성 | 토큰당 활성 매개변수 | 총 매개변수 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (밀집) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3는 거의 모든 벤치마크에서 Llama 3 70B (밀집)를 능가하며 **토큰당 더 적은 활성 FLOP**를 수행합니다. 더 많은 매개변수 = 더 많은 지식. 더 많은 활성 FLOP = 토큰당 더 많은 연산. MoE는 이 둘을 분리합니다.

### 함정: 메모리

모든 전문가는 어떤 것이 발동하든 GPU에 상주합니다. 671B 모델은 fp16 가중치를 위해 약 1.3 TB의 VRAM이 필요합니다. 최전선 MoE 배포는 전문가 병렬화(Expert Parallelism)가 필요합니다 — 전문가를 GPU 전체에 샤드(shard)하고, 네트워크를 통해 토큰을 라우팅합니다. 지연 시간은 matmul가 아닌 all-to-all 통신에 의해 지배됩니다.

```figure
expert-routing
```

## 구현하기

`code/main.py`을 참조하세요. 순수 stdlib로 작성된 컴팩트한 MoE 레이어는 다음을 포함합니다:

- `n_experts=8` SwiGLU 계열 전문가 (설명용으로 각각 선형 변환 하나 사용)
- top-k=2 라우팅
- softmax로 정규화된 게이팅 가중치
- 전문가별 편향을 통한 보조 손실 없는 균형 조정

### 1단계: 라우터

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 선택된 전문가의 원본 점수에 대한 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

편향은 선택에 영향을 미치며, 게이팅 가중치에는 영향을 주지 않습니다. 이것이 DeepSeek-V3의 핵심 기법입니다. 편향은 모델의 예측을 조정하지 않으면서 로드 불균형을 보정합니다.

### 2단계: 100개 토큰을 라우터로 전달

각 전문가가 얼마나 자주 활성화되는지 추적합니다. 편향이 없으면 사용량이 편중됩니다. 편향 업데이트 루프(`-γ`는 과사용 전문가, `+γ`는 과소사용 전문가)를 적용하면 몇 번의 반복 후 사용량이 균일한 분포로 수렴합니다.

### 3단계: 매개변수 수 비교

MoE 구성의 "밀집(dense) 등가"를 출력합니다. DeepSeek-V3 형태: 라우팅된 전문가 256개 + 공유 전문가 1개, 활성 전문가 8개, d_model=7168. 총 매개변수 수는 매우 많습니다. 활성 매개변수 수는 Llama 3 70B 밀집 모델의 7분의 1입니다.

## 사용하기

HuggingFace 로딩:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026년 프로덕션 추론: vLLM은 MoE 라우팅을 네이티브로 지원합니다. SGLang는 가장 빠른 전문가 병렬화 경로를 제공합니다. 두 도구 모두 top-k 선택과 전문가 병렬화를 자동으로 처리합니다.

**MoE를 선택해야 하는 경우:**
- 토큰당 추론 비용을 낮추면서 프론티어 수준의 품질을 원할 때.
- VRAM 및 전문가 병렬화 인프라를 갖추고 있을 때.
- 워크로드가 토큰 중심(채팅, 코드)이고 컨텍스트 중심(긴 문서)이 아닐 때.

**MoE를 선택하지 말아야 하는 경우:**
- 엣지 배포 — 활성 FLOP에 대해 전체 스토리지 비용을 지불해야 합니다.
- 레이턴시 민감한 단일 사용자 서빙 — 전문가 라우팅이 오버헤드를 추가합니다.
- 소형 모델(<7B) — MoE의 품질 이점은 컴퓨팅 임계값(활성 매개변수 약 6B) 이상에서만 나타납니다.

## 출시하기

`outputs/skill-moe-configurator.md`를 참조하세요. 이 스킬은 매개변수 예산, 학습 토큰, 배포 대상을 고려하여 새로운 MoE의 E, k, 공유 전문가 레이아웃을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 보조 손실 없는 편향 업데이트가 50번의 반복 동안 전문가 사용량을 균등하게 만드는지 관찰해 보세요.
2. **중간 난이도.** 학습된 라우터를 해시 기반 라우터(결정적, 학습 없음)로 교체하세요. 품질과 균형이 어떻게 달라지는지 비교해 보세요. 학습된 라우터가 더 좋은 이유는 무엇일까요?
3. **고난이도.** GRPO 스타일의 "롤아웃 매칭 라우팅"(DeepSeek-V3.2 트릭)을 구현하세요. 추론 중 어떤 전문가가 발동되는지 기록하고, 기울기 계산 시 동일한 라우팅을 강제하세요. 장난감 정책 기울기 설정에서 미치는 영향을 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 전문가 | "여러 FFN 중 하나" | 독립적인 피드포워드 네트워크; FFN 연산의 희소 슬라이스에 전용된 매개변수. |
| 라우터 | "게이트" | 각 토큰을 각 전문가에 대해 점수화하는 작은 선형 레이어; top-k 선택. |
| Top-k 라우팅 | "토큰당 k개의 활성 전문가" | 각 토큰의 FFN 연산이 게이트 가중치를 적용하여 정확히 k개의 전문가를 거침. |
| 보조 손실 | "부하 균형 페널티" | 전문가 사용의 편향을 페널티하는 추가 손실 항. |
| 보조 손실 없음 | "DeepSeek-V3의 트릭" | 라우터의 선택에만 전문가별 편향을 적용하여 균형 유지; 추가 기울기 없음. |
| 공유 전문가 | "항상 켜짐" | 모든 토큰이 거치는 추가 전문가; 공통 지식을 포착. |
| 전문가 병렬화(Expert Parallelism) | "전문가별 샤딩" | 서로 다른 전문가를 서로 다른 GPU에 분배; 네트워크를 통해 토큰 라우팅. |
| 희소성 | "활성 매개변수 < 총 매개변수" | 비율 `k × expert_size / (E × expert_size)`; DeepSeek-V3의 경우 37/671 ≈ 5.5%. |

## 추가 읽기

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 아이디어.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, 고전적인 MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 보조 손실 없는 MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 편향 기반 균형 논문.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 이 강의의 라우터가 사용하는 세분화 + 공유 전문가 분할.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 원본 공유 전문가 논문.
