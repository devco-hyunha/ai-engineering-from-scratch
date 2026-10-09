# Proximal Policy Optimization (PPO)

> A2C는 한 번의 업데이트 후 각 롤아웃을 버립니다. PPO는 정책 그래디언트를 클리핑된 중요도 비율로 감싸므로, 정책이 폭발하지 않으면서 동일한 데이터로 10회 이상의 에포크를 수행할 수 있습니다. Schulman et al. (2017). 2026년에도 여전히 기본 정책 그래디언트 알고리즘입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 9단계 · 06강 (REINFORCE), 9단계 · 07강 (Actor-Critic)
**시간:** 약 75분

## 문제점

A2C (07강)는 온-폴리시입니다. 그래디언트 `E_{π_θ}[A · ∇ log π_θ]`는 *현재* `π_θ`에서 샘플링된 데이터를 필요로 합니다. 한 번 업데이트하면 `π_θ`가 변하고, 사용했던 데이터는 이제 오프-폴리시가 됩니다. 이를 재사용하면 그래디언트에 편향이 생깁니다.

롤아웃은 비용이 많이 듭니다. Atari에서는 8개 환경 × 128단계 = 1024개의 전이와 수십 초의 환경 시간이 필요합니다. 한 번의 그래디언트 단계 후 이를 버리는 것은 낭비입니다.

Trust Region Policy Optimization (TRPO, Schulman 2015)은 최초의 해결책이었습니다. 각 업데이트에서 이전 정책과 새 정책 간의 KL 발산이 `δ` 이하로 유지되도록 제한합니다. 이론적으로 깔끔하지만, 업데이트마다 공액 그래디언트 해가 필요합니다. 2026년에는 TRPO를 실행하는 사람이 없습니다.

PPO (Schulman et al. 2017)는 하드 트러스트 리전 제약 조건을 단순한 클리핑된 목적 함수로 대체합니다. 코드 한 줄이 추가됩니다. 롤아웃당 10개의 에포크. 공액 그래디언트 없음. 충분한 수준의 이론적 보장. 9년 후에도 MuJoCo부터 RLHF까지 모든 것에 대한 기본 정책 그래디언트 알고리즘으로 남아 있습니다.

## 개념

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**중요도 비율.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

이것은 새 정책과 데이터를 수집한 정책의 우도 비율입니다. `r_t = 1`는 변화가 없음을 의미합니다. `r_t = 2`는 새 정책이 `a_t`를 취할 가능성이 이전 정책의 두 배임을 의미합니다.

**클리핑된 대리 함수.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

두 항:

- 어드밴티지 `A_t > 0`가 `1 + ε`를 넘어서려고 시도하면, 클리핑이 그래디언트를 평평하게 만듭니다. 좋은 행동을 이전 확률보다 `+ε` 이상 더 밀어붙이지 마세요.
- 이점 `A_t < 0`과 비율이 `1 - ε`을 넘어서려 하면 (나쁜 행동을 클리핑된 감소량보다 더 높게 만들려는 경우), 클리핑이 기울기를 제한합니다 — 나쁜 행동을 `-ε` 아래로 내리지 마세요.

`min`은 반대 방향을 처리합니다: 비율이 *유익한* 방향으로 이동한 경우, 기울기를 그대로 얻습니다 (손해를 보는 쪽에는 클리핑이 적용되지 않습니다).

일반적인 `ε = 0.2`. `r_t`의 함수로 목적 함수를 플롯하면: "좋은 쪽"에 평평한 지붕, "나쁜 쪽"에 평평한 바닥을 가진 조각별 선형 함수가 됩니다.

**전체 PPO 손실.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2C와 동일한 액터-크리틱 구조입니다. 세 개의 계수, 보통 `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`입니다.

**학습 루프.**

1. `N`개의 병렬 환경에서 `T`단계 동안 `N × T`개의 전이를 수집합니다.
2. 이점(GAE)을 계산하고, 상수로 고정합니다.
3. `π_{θ_old}`을 현재 `π_θ`의 스냅샷으로 고정합니다.
4. `K` 에포크 동안, `(s, a, A, V_target, log π_old(a|s))`개의 미니배치마다:
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`을 계산합니다.
   - `L^{CLIP}` + 가치 손실 + 엔트로피를 적용합니다.
   - 기울기 단계.
5. 롤아웃을 버립니다. 1단계로 돌아갑니다.

`K = 10`과 64개의 미니배치는 표준 하이퍼파라미터 세트입니다. PPO는 강건합니다: ±50% 범위 내에서는 정확한 숫자가 거의 중요하지 않습니다.

**KL 페널티 변형.** 원 논문은 적응형 KL 페널티를 사용하는 대안을 제안했습니다: `L = L^{PG} - β · KL(π_θ || π_old)`, `β`은 관측된 KL에 따라 조정됩니다. 클리핑 버전이 지배적이게 되었으며, KL 변형은 RLHF에서 살아남습니다 (참조 정책과의 KL은 항상 원하는 별도의 제약 조건이므로).

```figure
ppo-clip
```

## 구현하기

### 1단계: 롤아웃 시점에 `log π_old(a | s)`을 캡처합니다

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

스냅샷은 롤아웃 시점에 한 번만 찍습니다. 업데이트 에포크 동안 변하지 않습니다.

### 2단계: GAE 이점 계산 (07강)

A2C와 동일합니다. 배치 전체에 대해 정규화합니다.

### 3단계: 클리핑된 대리 업데이트

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # -대리 손실의 역전파, 가치 손실 추가, 엔트로피 차감
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 클리핑
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"clipped → zero gradient" 패턴은 PPO의 핵심입니다. 새로운 정책이 유리한 방향으로 너무 많이 벗어났다면 업데이트가 중단됩니다.

### 4단계: 가치 및 엔트로피

크리틱 대상에 표준 MSE를 추가하고 액터에 엔트로피 보너스를 적용합니다. A2C와 동일합니다.

### 5단계: 진단

매 업데이트마다 확인해야 할 세 가지가 있습니다:

- **평균 KL** `E[log π_old - log π_θ]`. `[0, 0.02]` 범위 내에 유지해야 합니다. `0.1`를 초과하면 `K_EPOCHS` 또는 `LR`를 줄이세요.
- **클립 비율** — 비율이 `[1-ε, 1+ε]` 범위를 벗어난 샘플의 비율입니다. `~0.1-0.3`이어야 합니다. `~0`이면 클립이 발동되지 않으므로 `LR` 또는 `K_EPOCHS`를 높이세요. `~0.5+`이면 롤아웃에 과적합하고 있으므로 값을 낮추세요.
- **설명된 분산** `1 - Var(V_target - V_pred) / Var(V_target)`. 크리틱 품질 지표입니다. 크리틱이 학습함에 따라 1에 가까워져야 합니다.

## 함정

- **클립 계수 튜닝 실패.** `ε = 0.2`는 사실상 표준입니다. `0.1`로 설정하면 업데이트가 너무 소극적이 되고, `0.3+`는 불안정성을 초래합니다.
- **에포크가 너무 많음.** `K > 20`는 정책이 `π_old`에서 크게 벗어나므로 불안정성을 초래하는 경우가 많습니다. 특히 대규모 네트워크에서는 에포크 수를 제한하세요.
- **보상 정규화 누락.** 큰 보상 스케일은 클립 범위를 침식합니다. 어드밴티지를 계산하기 전에 보상(이동 표준 편차)을 정규화하세요.
- **어드밴티지 정규화 잊음.** 배치별 평균 0/표준 편차 1 정규화는 표준입니다. 이를 생략하면 대부분의 벤치마크에서 PPO가 망가집니다.
- **학습률 감쇠 없음.** PPO는 선형 LR 감쇠가 0으로 수렴할 때 이점을 얻습니다. 일정한 LR은 종종 더 나쁜 결과를 낳습니다.
- **중요도 비율 계산 오류.** 수치적 안정성을 위해 항상 `exp(log_new - log_old)`를 사용하세요. `new / old`를 사용하지 마세요.
- **그라디언트 부호 오류.** 대리 함수를 최대화하는 것은 `-L^{CLIP}`를 *최소화*하는 것과 같습니다. 부호 반전은 가장 흔한 PPO 버그입니다.

## 사용하기

PPO는 놀라울 정도로 많은 분야에서 2026년 기본 RL 알고리즘입니다:

| 사용 사례 | PPO 변형 |
|----------|-------------|
| MuJoCo / 로봇 제어 | 가우시안 정책, GAE(0.95)를 사용한 PPO |
| Atari / 이산 게임 | 범주형 정책, 롤링 128단계 롤아웃을 사용한 PPO |
| LLM을 위한 RLHF | 참조 모델에 대한 KL 페널티가 있는 PPO, 응답 끝에 RM으로부터 보상 |
| 대규모 게임 에이전트 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 추론 LLM | GRPO (12강) — 크리틱이 없는 PPO 변형 |
| 선호 데이터만 사용 | DPO — PPO+KL의 폐형(closed-form) 축소, 온라인 샘플링 없음 |

PPO의 *손실 형태* — 클리핑된 대리 손실 + 가치 + 엔트로피 —는 DPO, GRPO 및 거의 모든 RLHF 파이프라인의 뼈대입니다.

## 출시하기

`outputs/skill-ppo-trainer.md`로 저장하세요:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 연습 문제

1. **쉬움.** `ε=0.2, K=4`을 사용하여 4×4 GridWorld에서 PPO를 실행하세요. 동일한 환경 단계에서 A2C(롤아웃당 1에포크)와 샘플 효율성을 비교하세요.
2. **중간.** `K ∈ {1, 4, 10, 30}`을 스윕(sweep)하세요. 환경 단계 대비 리턴을 플롯하고 업데이트별 평균 KL을 추적하세요. 이 작업에서 `K`이 얼마일 때 KL이 폭발하나요?
3. **어려움.** 클리핑된 대리 손실을 적응형 KL 페널티(`β`, `KL > 2·target`이면 2배, `KL < target/2`이면 절반)로 교체하세요. 최종 리턴, 안정성 및 클리핑 없는 특성을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 중요도 비율 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; 데이터를 수집한 정책에서의 편차. |
| 클리핑된 대리 손실 | "PPO의 주요 트릭" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; 유리한 쪽에서 클리핑을 넘으면 기울기가 평평해짐. |
| 신뢰 영역 | "TRPO / PPO의 의도" | 각 업데이트의 KL을 제한하여 단조 개선(monotone improvement)을 보장. |
| KL 페널티 | "소프트 신뢰 영역" | 대안 PPO: `L - β · KL(π_θ \|\| π_old)`. 적응형 `β`. |
| 클리핑 비율 | "클리핑이 얼마나 자주 발동하는지" | 진단 지표 — 0.1-0.3이어야 하며, 그 밖이면 튜닝이 잘못됨. |
| 다중 에포크 학습 | "데이터 재사용" | 각 롤아웃에 대해 K 에포크; 샘플 효율성을 위해 분산 비용을 감수. |
| 온-폴리시-ish | "대부분 온-폴리시" | PPO는 명목상 온-폴리시지만 K>1 에포크는 약간 오프-폴리시 데이터를 안전하게 사용. |
| PPO-KL | "다른 PPO" | KL 페널티 변형; 참조 모델에 대한 KL이 이미 제약인 RLHF에서 사용. |

## 추가 읽기

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 논문.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO의 전신.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 모든 PPO 하이퍼파라미터를 제거한 실험입니다.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; RLHF 내 PPO 레시피입니다.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorch를 활용한 깔끔한 현대적 설명입니다.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 많은 논문에서 사용하는 단일 파일 PPO 참고 자료입니다.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 언어 모델에 PPO를 적용하기 위한 프로덕션 레시피입니다; 09강(RLHF)과 함께 읽어 보세요.
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37가지 코드 수준 최적화" 논문입니다; 어떤 PPO 트릭이 핵심이고 어떤 것이 구전 지식인지 알려줍니다.
