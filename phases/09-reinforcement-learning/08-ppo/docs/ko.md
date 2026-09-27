# 근사 정책 최적화 (Proximal Policy Optimization, PPO)

> A2C는 한 번의 업데이트 후에 각 롤아웃(rollout)을 폐기합니다. PPO는 정책 경사(policy gradient)를 클리핑된 중요도 비율(clipped importance ratio)로 감싸서, 정책이 폭발하지 않고도 동일한 데이터에 대해 10회 이상의 에포크(epoch)를 수행할 수 있게 합니다. Schulman et al. (2017). 2026년에도 여전히 기본 정책 경사 알고리즘으로 사용됩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## 문제점 (The Problem)

A2C (Lesson 07)는 온폴리시(on-policy) 방식입니다. 즉, 그래디언트 `E_{π_θ}[A · ∇ log π_θ]`를 계산하려면 *현재*의 `π_θ`로부터 샘플링된 데이터가 필요합니다. 업데이트를 한 번 수행하여 `π_θ`가 변하면, 이전에 사용했던 데이터는 이제 오프폴리시(off-policy) 데이터가 됩니다. 이 데이터를 재사용하면 그래디언트에 편향(bias)이 발생합니다.

롤아웃(Rollout)은 비용이 많이 듭니다. Atari 환경에서 8개의 환경 × 128 스텝을 수행하는 한 번의 롤아웃은 1024개의 전이(transition)를 생성하며, 환경 실행에 수 초가 소요됩니다. 단 한 번의 그래디언트 단계 후에 이 데이터를 버리는 것은 매우 낭비적입니다.

신뢰 영역 정책 최적화(Trust Region Policy Optimization, TRPO, Schulman 2015)가 첫 번째 해결책이었습니다. 이는 이전 정책과 새로운 정책 사이의 KL 발산(KL divergence)이 `δ` 미만이 되도록 각 업데이트를 제한하는 방식입니다. 이론적으로는 깔끔하지만, 업데이트마다 공액 경사법(conjugate-gradient)을 풀어야 합니다. 2026년 현재 TRPO를 사용하는 사람은 아무도 없습니다.

PPO (Schulman et al. 2017)는 엄격한 신뢰 영역 제약을 단순한 클리핑 목적 함수(clipped objective)로 대체했습니다. 코드 한 줄만 추가하면 됩니다. 롤아웃당 10번의 에포크(epoch)를 수행하며, 공액 경사법도 필요 없습니다. 이론적 보장 또한 충분히 훌륭합니다. 9년이 지난 지금도 PPO는 MuJoCo부터 RLHF에 이르기까지 모든 분야에서 기본 정책 경사(policy-gradient) 알고리즘으로 사용되고 있습니다.

## 개념 (The Concept)

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**중요도 비율 (The importance ratio).**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

이는 새로운 정책과 데이터를 수집했던 이전 정책 간의 가능도 비율(likelihood ratio)입니다. `r_t = 1`은 변화가 없음을 의미합니다. `r_t = 2`는 새로운 정책이 이전 정책보다 `a_t`를 선택할 확률이 두 배 높음을 의미합니다.

**클리핑된 대리 목적 함수 (The clipped surrogate).**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

두 가지 항이 있습니다:

- 만약 어드밴티지(advantage) `A_t > 0`이고 비율이 `1 + ε`을 넘어 커지려 한다면, 클리핑(clip)이 그래디언트를 평탄화합니다. 즉, 좋은 행동을 이전 확률보다 `+ε` 이상으로 더 밀어붙이지 않도록 합니다.
- 만약 어드밴티지 `A_t < 0`이고 비율이 `1 - ε`을 넘어 커지려 한다면(이는 클리핑된 감소치에 비해 나쁜 행동을 더 가능성 있게 만든다는 의미입니다), 클리핑이 그래디언트를 제한합니다. 즉, 나쁜 행동을 `-ε` 아래로 밀어내지 않도록 합니다.

`min` 함수는 반대 방향을 처리합니다. 만약 비율이 *이로운(beneficial)* 방향으로 움직였다면, 여전히 그래디언트를 얻을 수 있습니다(사용자에게 해가 되는 방향으로는 클리핑되지 않습니다).

일반적인 `ε = 0.2`를 사용합니다. `r_t`의 함수로서 목적 함수를 그래프로 그리면, "좋은 쪽"에는 평평한 지붕이 있고 "나쁜 쪽"에는 평평한 바닥이 있는 구간별 선형 함수(piecewise-linear function) 형태가 됩니다.

**전체 PPO 손실 함수 (The full PPO loss).**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2C와 동일한 액터-크리틱(actor-critic) 구조를 가집니다. 세 개의 계수가 있으며, 보통 `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`를 사용합니다.

**훈련 루프 (The training loop).**

1. `N`개의 병렬 환경에서 각각 `T` 스텝 동안 `N × T`개의 전이(transitions)를 수집합니다.
2. 어드밴티지(GAE)를 계산하고, 이를 상수로 고정합니다.
3. 현재 `π_θ`의 스냅샷으로서 `π_{θ_old}`를 고정합니다.
4. `K` 에포크(epoch) 동안, `(s, a, A, V_target, log π_old(a|s))`의 각 미니배치에 대해 다음을 수행합니다:
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`를 계산합니다.
   - `L^{CLIP}` + 가치 손실(value loss) + 엔트로피(entropy)를 적용합니다.
   - 그래디언트 단계를 수행합니다.
5. 롤아웃(rollout) 데이터를 버립니다. 1단계로 돌아갑니다.

`K = 10`과 미니배치 크기 64는 표준적인 하이퍼파라미터 설정입니다. PPO는 견고(robust)합니다. 정확한 수치가 ±50% 범위 내에서 변하더라도 결과에 큰 영향을 미치지 않는 경우가 많습니다.

**KL-페널티 변형 (KL-penalty variant).** 원본 논문에서는 관찰된 KL을 기반으로 `β`를 조정하는 적응형 KL 페널티를 사용하는 대안을 제안했습니다: `L = L^{PG} - β · KL(π_θ || π_old)`. 클리핑 방식이 주류가 되었지만, KL 변형은 RLHF(참조 정책에 대한 KL을 항상 별도의 제약 조건으로 두어야 하는 환경)에서 여전히 사용됩니다.

```figure
ppo-clip
```

## 구현하기 (Build It)

### 1단계: 롤아웃(rollout) 시점에 `log π_old(a | s)` 캡처하기

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

스냅샷은 롤아웃 시점에 단 한 번 촬영됩니다. 이는 업데이트 에포크(update epochs) 동안 변경되지 않습니다.

### 2단계: GAE 어드밴티지(advantages) 계산하기 (레슨 07)

A2C와 동일합니다. 배치(batch) 전체에 대해 정규화(Normalize)를 수행하세요.

### 3단계: 클리핑된 대리 목적 함수 업데이트 (Clipped Surrogate Update)

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
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"clipped → zero gradient(클리핑 시 그래디언트 0)" 패턴은 PPO의 핵심입니다. 만약 새로운 정책이 이로운 방향으로 이미 너무 멀리 벗어났다면, 업데이트를 중단합니다.

### 4단계: 가치(Value) 및 엔트로피(Entropy)

A2C와 동일하게, Critic 타겟에는 표준 MSE(Mean Squared Error)를 추가하고, Actor에는 엔트로피 보너스(Entropy bonus)를 추가해 보세요.

### 5단계: 진단 (diagnostics)

매 업데이트마다 다음 세 가지를 확인해야 합니다:

- **평균 KL (Mean KL)** `E[log π_old - log π_θ]`. `[0, 0.02]` 범위를 유지해야 합니다. 만약 `0.1`을 초과한다면, `K_EPOCHS` 또는 `LR`을 줄이세요.
- **클립 비율 (Clip fraction)** — 비율(ratio)이 `[1-ε, 1+ε]` 범위를 벗어나는 샘플의 비율입니다. `~0.1-0.3` 정도가 적당합니다. 만약 `~0`이라면 클립이 전혀 발생하지 않는 것이므로 `LR` 또는 `K_EPOCHS`를 높이세요. 만약 `~0.5+`라면 롤아웃(rollout)에 과적합(over-fitting)되고 있는 것이므로 이 값들을 낮추세요.
- **설명된 분산 (Explained variance)** `1 - Var(V_target - V_pred) / Var(V_target)`. 크리틱(Critic)의 품질을 나타내는 지표입니다. 크리틱이 학습됨에 따라 1을 향해 상승해야 합니다.

## 주의 사항 (Pitfalls)

- **클립 계수(Clip coefficient) 미조정.** `ε = 0.2`가 사실상의 표준(de-facto standard)입니다. `0.1`로 설정하면 업데이트가 너무 소극적이 되고, `0.3` 이상으로 설정하면 불안정성이 초래됩니다.
- **너무 많은 에포크(Epochs).** `K > 20`은 정책이 `π_old`로부터 너무 멀어지게 만들어 통상적으로 학습을 불안정하게 만듭니다. 특히 대규모 네트워크의 경우 에포크 수를 제한하세요.
- **보상 정규화(Reward normalization) 누락.** 보상 스케일이 너무 크면 클립 범위(clip range)를 침범하게 됩니다. 어드밴티지(advantages)를 계산하기 전에 보상을 정규화(running std 사용)하세요.
- **어드밴티지 정규화(Advantage normalization) 누락.** 배치별로 평균을 0, 표준편차를 1로 만드는 정규화가 표준입니다. 이를 생략하면 대부분의 벤치마크에서 PPO 성능이 망가집니다.
- **학습률(Learning rate) 미감소.** PPO는 학습률을 0으로 선형 감소(linear LR decay)시킬 때 이점이 있습니다. 고정된 학습률을 사용하는 것이 종종 더 나쁜 결과를 초래합니다.
- **중요도 비율(Importance ratio) 계산 오류.** 수치적 안정성을 위해 `new / old`가 아닌 항상 `exp(log_new - log_old)`를 사용하세요.
- **잘못된 그래디언트 부호(Gradient sign).** 대리 목적 함수(surrogate)를 최대화하는 것은 `-L^{CLIP}`을 *최소화*하는 것과 같습니다. 부호를 반대로 설정하는 것이 PPO에서 가장 흔히 발생하는 버그입니다.

## 활용하기 (Use It)

PPO는 놀라울 정도로 다양한 도메인에서 2026년의 기본 RL 알고리즘으로 자리 잡고 있습니다:

| 활용 사례 (Use case) | PPO 변형 (PPO variant) |
|----------|-------------|
| MuJoCo / 로보틱스 제어 | 가우시안 정책(Gaussian policy)을 사용하는 PPO, GAE(0.95) |
| Atari / 이산 게임 | 범주형 정책(categorical policy)을 사용하는 PPO, 128단계 롤아웃(rollouts) |
| LLM을 위한 RLHF | 참조 모델(reference model)에 대한 KL 페널티를 적용한 PPO, 응답 종료 시 RM으로부터 보상 수령 |
| 대규모 게임 에이전트 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 추론형 LLM | GRPO (레슨 12) — Critic이 없는 PPO 변형 |
| 선호도 전용 데이터 | DPO — PPO+KL의 폐쇄형(closed-form) 축소, 온라인 샘플링 없음 |

PPO의 *손실 형태(loss shape)* — 클리핑된 대리 함수(clipped surrogate) + 가치(value) + 엔트로피(entropy) — 는 DPO, GRPO, 그리고 거의 모든 RLHF 파이프라인의 기본 골격(scaffolding) 역할을 합니다.

## Ship It

`outputs/skill-ppo-trainer.md`로 저장하세요:

```markdown
---
name: ppo-trainer
description: 주어진 환경에 대한 PPO 학습 설정 및 진단 계획을 생성합니다.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

환경과 학습 예산(training budget)이 주어지면 다음을 출력하세요:

1. Rollout size. `N`개의 환경 × `T`개의 스텝.
2. Update schedule. `K` 에포크(epochs), 미니배치 크기(minibatch size), 학습률 스케줄(LR schedule).
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, 어드밴티지 정규화(advantage normalization) 활성화 여부.
4. Advantage. 명시적인 `γ` 및 `λ`를 포함한 GAE(`λ`).
5. Diagnostics plan. KL, clip fraction, 설명 분산(explained variance) 임계값 및 알림 설정.

`K > 30` 또는 `ε > 0.3` (안전하지 않은 신뢰 영역)인 경우 거부하세요. 어드밴티지 정규화나 KL/clip 모니터링이 없는 모든 PPO 실행은 거부하세요. clip fraction이 0.4 이상으로 지속되면 드리프트(drift)로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `ε=0.2, K=4` 설정으로 4×4 GridWorld에서 PPO를 실행해 보세요. 동일한 환경 스텝(env steps) 수에서 A2C(롤아웃당 1 에포크)와 샘플 효율성(sample efficiency)을 비교해 보세요.
2. **중간 (Medium).** `K ∈ {1, 4, 10, 30}` 범위를 탐색(Sweep)해 보세요. 환경 스텝 대비 리턴(return)을 그래프로 그리고, 업데이트당 평균 KL을 추적해 보세요. 이 작업에서 어떤 `K` 값일 때 KL이 폭발(explode)하나요?
3. **어려움 (Hard).** 클리핑된 대리 목적 함수(clipped surrogate)를 적응형 KL 페널티(adaptive KL penalty)로 교체해 보세요 (`KL > 2·target`이면 `β`를 두 배로, `KL < target/2`이면 절반으로 조정). 최종 리턴, 안정성, 그리고 클리핑 미사용(clip-free-ness) 측면을 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| 중요도 비율 (Importance ratio) | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; 데이터를 수집한 정책으로부터의 편차. |
| 클리핑된 대리 목적 함수 (Clipped surrogate) | "PPO의 핵심 트릭" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; 이득이 되는 방향으로 클리핑 범위를 벗어날 경우 그래디언트를 평탄화함. |
| 신뢰 영역 (Trust region) | "TRPO / PPO의 의도" | 단조적 개선(monotone improvement)을 보장하기 위해 각 업데이트의 KL 발산 값을 제한함. |
| KL 페널티 (KL penalty) | "소프트 신뢰 영역" | PPO의 대안: `L - β · KL(π_θ \|\| π_old)`. 적응형 `β` 사용. |
| 클리핑 비율 (Clip fraction) | "클리핑이 얼마나 자주 발생하는가" | 진단 지표 — 0.1~0.3 사이가 적절하며, 이 범위를 벗어나면 튜닝이 잘못된 것임. |
| 멀티 에포크 학습 (Multi-epoch training) | "데이터 재사용" | 각 롤아웃(rollout)에 대해 K 에포크를 수행; 샘플 효율성을 위해 분산(variance) 비용을 감수함. |
| 온-폴리시 유사 (On-policy-ish) | "대체로 온-폴리시" | PPO는 명목상 온폴리시(on-policy)이지만, K>1 에포크를 통해 약간의 오프폴리시(off-policy) 데이터를 안전하게 사용함. |
| PPO-KL | "또 다른 PPO" | KL-페널티 변형; 참조 모델(reference model)과의 KL 제약이 이미 존재하는 RLHF에서 사용됨. |

## 추가 학습 자료 (Further Reading)

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 논문 원문.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO의 전신.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 모든 PPO 하이퍼파라미터에 대한 절제 연구(ablation study).
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; RLHF에서의 PPO 활용 레시피.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorch를 사용한 깔끔하고 현대적인 설명.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 많은 논문에서 참조하는 단일 파일 형태의 PPO 구현체.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 언어 모델을 위한 PPO 프로덕션 레시피; Lesson 09 (RLHF)와 함께 읽어보세요.
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37가지 코드 수준 최적화"에 관한 논문; 어떤 PPO 트릭이 실제로 성능을 지탱하는지, 어떤 것이 단순한 설화(folklore)인지 다룹니다.
