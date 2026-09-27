# Sim-to-Real Transfer (시뮬레이션에서 실물로의 전이)

> 시뮬레이터에서 학습되었으나 하드웨어에서 실패하는 정책은 시뮬레이터를 암기해 버린 정책입니다. 도메인 무작위화(Domain randomization), 도메인 적응(Domain adaptation), 그리고 시스템 식별(System identification)은 학습된 제어기가 현실과의 격차(Reality gap)를 극복하게 만드는 세 가지 도구입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## 문제점 (The Problem)

실제 로봇을 학습시키는 것은 느리고, 위험하며, 비용이 많이 듭니다. 이족 보행 로봇이 걷는 법을 배우려면 수백만 번의 학습 에피소드가 필요하며, 실제 이족 보행 로봇은 단 한 번만 넘어지더라도 하드웨어가 파손될 수 있습니다. 시뮬레이션은 무제한의 리셋, 결정론적 재현성, 병렬 환경, 그리고 물리적 손상 없음이라는 이점을 제공합니다.

하지만 시뮬레이터는 부정확합니다. 베어링은 MuJoCo 모델보다 마찰이 더 심할 수 있습니다. 카메라는 시뮬레이터에 포함되지 않은 렌즈 왜곡을 가집니다. 모터에는 99%의 시뮬레이션 모델이 생략하는 지연(delay), 백래시(backlash), 포화(saturation) 현상이 존재합니다. 바람, 먼지, 가변적인 조명은 깨끗한 렌더링 환경에서 학습된 정책(policy)을 방해합니다. 시뮬레이션 분포와 실제 분포 사이의 체계적인 차이인 **리얼리티 갭(reality gap)**은 로보틱스에 배포된 강화학습(RL)의 핵심 문제입니다.

여러분에게는 *sim-to-real 분포 변화(distribution shift)에 강건한(robust)* 정책이 필요합니다. 역사적으로 세 가지 접근 방식이 있었습니다: 시뮬레이터를 무작위화하거나(도메인 무작위화, domain randomization), 약간의 실제 데이터를 사용하여 정책을 적응시키거나(도메인 적응 / 미세 조정, domain adaptation / fine-tuning), 실제 시스템의 파라미터를 식별하여 이를 일치시키는 방식(시스템 식별, system identification)입니다. 2026년 현재, 지배적인 방식은 대규모 병렬 시뮬레이션(Isaac Sim, Isaac Lab, GPU 기반의 Mujoco MJX)과 이 세 가지를 모두 결합하는 것입니다.

## 개념 (The Concept)

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**도메인 무작위화 (Domain Randomization, DR).** Tobin et al. 2017, Peng et al. 2018. 학습 과정에서 실제 로봇과 다를 수 있는 모든 시뮬레이션 파라미터를 무작위화합니다: 질량, 마찰 계수, 모터 PD 이득(gains), 센서 노이즈, 카메라 위치, 조명, 질감, 접촉 모델 등. 정책(policy)은 "오늘이 어떤 시뮬레이션 환경인지"에 대한 조건부 분포를 학습하며 전체 범위에 걸쳐 일반화됩니다. 실제 로봇이 학습된 범위(envelope) 내에 있다면 정책은 작동합니다.

- **장점:** 실제 데이터가 필요 없습니다. 하나의 레시피로 여러 로봇에 적용 가능합니다.
- **단점:** 과도하게 무작위화된(over-randomized) 학습은 "보편적"이지만 지나치게 조심스러운 정책을 생성합니다. 노이즈가 너무 많으면 과도한 규제(regularization)와 유사한 효과를 냅니다.

**시스템 식별 (System Identification, SI).** 학습 전 시뮬레이션의 파라미터를 실제 세계의 데이터에 맞춥니다. 실제 로봇의 팔 관절 마찰력을 측정할 수 있다면, 그 값을 시뮬레이션에 입력합니다. 그런 다음 해당 값을 예상하도록 정책을 학습시킵니다. 실제 시스템에 대한 접근이 필요하지만, 현실과의 격차(reality gap)를 직접적으로 줄여줍니다.

- **장점:** 정밀하고 노이즈가 적은 학습 목표를 제공합니다.
- **단점:** 잔여 모델 오차(residual model error)를 정책이 인지할 수 없습니다. 식별되지 않은 작은 효과(예: 모터 데드밴드)가 여전히 배포 시 문제를 일으킬 수 있습니다.

**도메인 적응 (Domain Adaptation).** 시뮬레이션에서 학습한 후, 소량의 실제 데이터로 미세 조정(fine-tune)합니다. 두 가지 방식이 있습니다:

- **Real2Sim2Real:** 실제 롤아웃(rollouts)을 사용하여 잔여 시뮬레이터 `f(s, a, z) - f_sim(s, a)`를 학습하고, 수정된 시뮬레이션에서 학습합니다. 많은 실제 데이터 없이도 격차를 좁힙니다.
- **관측 적응 (Observation adaptation):** 학습된 특징 추출기(예: GAN pixel-to-pixel)를 통해 실제 관측값(obs)을 시뮬레이션과 유사한 관측값으로 매핑하는 정책을 학습합니다. 컨트롤러는 시뮬레이션 상태를 유지합니다.

**특권 학습 / 교사-학생 (Privileged learning / teacher-student).** Miki et al. 2022 (ANYmal 사족 보행 로봇). 특권 정보(ground truth 마찰력, 지형 높이, IMU 드리프트)에 접근할 수 있는 *교사(teacher)*를 시뮬레이션에서 학습시킵니다. 이후 실제 센서 관측값만 보는 *학생(student)*에게 이를 증류(distill)합니다. 학생은 과거 이력으로부터 특권 특징을 추론하는 법을 배우며, 물리적 파라미터 변화에도 강건해집니다.

**대규모 병렬 시뮬레이션 (Massively parallel simulation).** 2024–2026. Isaac Lab, Mujoco MJX, Brax는 모두 단일 GPU에서 수천 대의 로봇을 병렬로 실행합니다. 4,096개의 병렬 휴머노이드를 사용하는 PPO는 수년 치의 경험을 단 몇 시간 만에 수집합니다. 학습 분포가 넓어짐에 따라 "현실 격차"는 줄어듭니다. 4,096개의 환경 각각이 서로 다른 무작위화된 파라미터를 가진다면 DR은 거의 비용이 들지 않게 됩니다.

**2026년 실전 레시피 (사족 보행 예시):**

1. 중력, 마찰력, 모터 이득, 페이로드를 도메인 무작위화한 대규모 병렬 시뮬레이션.
2. 특권 정보(지형 지도, 신체 속도 ground truth)를 사용하여 학습된 교사 정책.
3. 고유 수용 감각(proprioception, 다리 관절 인코더)만을 사용하여 교사로부터 증류된 학생 정책.
4. (선택 사항) 실제 IMU에 대한 오토인코더를 통한 관측 적응.
5. 배포. 10개 이상의 환경에서 제로샷(Zero-shot) 수행. 실패할 경우, 제약 조건이 있는 PPO(safety-constrained PPO)를 사용하여 몇 분간 실제 환경에서 미세 조정 수행.

```figure
f3-reality-gap
```

## 구축하기 (Build It)

이 레슨의 코드는 *노이즈가 있는(noisy)* 전이(transitions)가 존재하는 GridWorld에서 도메인 무작위화(domain randomization)를 보여주는 작은 데모입니다. 우리는 "시뮬레이션(sim)" 환경에서 무작위로 변하는 미끄러짐 확률(slip probabilities)을 경험하는 정책(policy)을 학습시키고, 학습 과정에서 한 번도 경험하지 못한 미끄러짐 수준을 가진 "실제(real)" 환경에서 이를 평가합니다. 이 구조는 MuJoCo에서 하드웨어로의 전이(transfer) 방식과 직접적으로 대응됩니다.

### 1단계: 매개변수화된 시뮬레이션 (parameterized sim)

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip`은 시뮬레이터가 노출하는 매개변수입니다. 실제 로보틱스에서는 마찰력(friction), 질량(mass), 모터 이득(motor gain) 등 시뮬레이션과 실제 환경 사이에서 차이가 발생할 수 있는 모든 요소가 될 수 있습니다.

### 2단계: DR을 이용한 학습

각 에피소드가 시작될 때, `slip ~ Uniform[0.0, 0.4]`를 샘플링합니다. PPO, Q-learning 또는 다른 알고리즘을 사용하여 학습을 진행하세요. 이 과정을 수많은 에피소드 동안 반복합니다.

### 3단계: "실제" 미끄러짐(slips)에 대한 제로샷(zero-shot) 평가

`slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`에 대해 평가를 수행합니다. 처음 네 가지 값은 학습 지원 범위(training support) 내에 있으며, `0.5`와 `0.7`은 범위 외(outside)에 있습니다. DR(Domain Randomization)로 학습된 정책은 지원 범위 내에서는 최적에 가깝게 유지되어야 하며, 범위 밖에서는 성능이 점진적으로 저하(degrade gracefully)되어야 합니다. 반면, 고정된 미끄러짐(fixed-slip)으로 학습된 정책은 학습된 미끄러짐 범위를 벗어나면 취약(brittle)한 모습을 보일 것입니다.

### 4단계: 좁은 범위의 학습(narrow training)과 비교하기

`slip = 0.0`인 상태로만 두 번째 정책(policy)을 학습시켜 보세요. 동일한 `slip` 스윕(sweep) 범위에서 평가를 진행합니다. 실제 `slip` 값이 0보다 커지는 즉시 성능이 급격히 떨어지는(catastrophic drop) 현상을 확인할 수 있을 것입니다.

## 주의 사항 (Pitfalls)

- **과도한 무작위화 (Too much randomization).** `slip ∈ [0, 0.9]` 범위로 학습하면 정책이 지나치게 위험 회피적(risk-averse)이 되어 최적의 경로를 시도조차 하지 않게 됩니다. "무엇이든 일어날 수 있다"는 식의 설정이 아니라, 실제 환경의 *기대(expected)* 분포에 맞춰야 합니다.
- **부족한 무작위화 (Too little randomization).** 좁은 범위의 데이터로만 학습하면 정책이 전혀 일반화(generalize)되지 못합니다. 정책이 개선됨에 따라 분포를 넓혀가는 적응형 커리큘럼(Adaptive Curriculum, 자동 도메인 무작위화)을 사용해 보세요.
- **잘못 식별된 파라미터 공간 (Misidentified parameter space).** 잘못된 요소를 무작위화하면(예: 실제 차이는 모터 지연인데 카메라 색조를 무작위화하는 경우) DR은 도움이 되지 않습니다. 먼저 실제 로봇의 특성을 프로파일링하세요.
- **특권 정보 유출 (Privileged info leakage).** 관측값(observation)뿐만 아니라 전역 상태(global state)를 사용하여 행동을 결정하는 교사(teacher) 모델은, 이를 따라잡을 수 없는 학생(student) 모델을 만들 수 있습니다. 교사의 정책이 관측 이력(observation history)을 가진 학생 모델에 의해 실행 가능한지 확인하세요.
- **Sim-to-sim 전이 실패 (Sim-to-sim transfer failure).** 정책이 더 어려운 시뮬레이션 변형(sim variant)에 대해 강건(robust)하지 않다면, 실제 환경에서도 강건하지 않을 것입니다. 배포하기 전에 항상 별도로 분리된 시뮬레이션 변형에서 테스트하세요.
- **실제 환경 안전 범위 부재 (No real-world safety envelope).** 시뮬레이션에서 작동하고 저수준 안전 보호막(low-level safety shield) 없이 "실제에서도 작동하는" 정책이라 할지라도 하드웨어를 파손시킬 수 있습니다. 학습되지 않은 컨트롤러(non-learned controller)에 속도 제한, 토크 제한, 관절 제한 등을 추가하세요.

## 활용하기 (Use It)

2026년형 sim-to-real 스택:

| 도메인 (Domain) | 스택 (Stack) |
|--------|-------|
| 보행 로코모션 (Legged locomotion: ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| 조작 (Manipulation: dexterous hands, pick-and-place) | Isaac Lab + DR + vision을 위한 DR-GAN |
| 자율 주행 (Autonomous driving) | CARLA / NVIDIA DRIVE Sim + DR + 실물 미세 조정 (real fine-tune) |
| 드론 레이싱 (Drone racing) | RotorS / Flightmare + DR + 온라인 적응 (online adaptation) |
| 손가락/인핸드 조작 (Finger/in-hand manipulation) | OpenAI Dactyl (전례 없는 규모의 DR) |
| 산업용 로봇 팔 (Industrial arms) | MuJoCo-Warp + SI + 소규모 실물 미세 조정 (small real fine-tune) |

모든 규모의 제어를 위한 워크플로우는 일관적입니다: 시뮬레이션을 최대한 정교하게 맞추고, 맞출 수 없는 부분은 무작위화(randomize)하며, 거대한 정책(policy)을 학습시킨 뒤, 이를 증류(distill)하고, 안전 보호막(safety shield)과 함께 배포하세요.

## Ship It

`outputs/skill-sim2real-planner.md`로 저장하세요:

```markdown
---
name: sim2real-planner
description: DR, SI 및 안전성을 포함하여 주어진 로봇 + 작업에 대한 sim-to-real 전이 파이프라인을 계획합니다.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

로봇 플랫폼, 작업, 그리고 실제 하드웨어 사용 시간이 주어졌을 때 다음을 출력하세요:

1. Reality gap 인벤토리. 예상되는 영향도 순으로 정렬된 의심되는 원인들 (접촉, 센싱, 구동 지연, 비전).
2. DR(Domain Randomization) 파라미터. 정확한 목록, 범위, 분포. 실제 측정값에 근거하여 각 범위를 정당화하세요.
3. SI(System Identification) 단계. 측정해야 할 파라미터와 측정 방법.
4. Teacher/student 분리. Teacher가 사용하는 특권 정보(privileged info)와 Student가 사용하는 관측값(obs).
5. Safety envelope(안전 영역). 저수준 제한(low-level limits), 비상 정지, 백업 컨트롤러.

다음 사항이 없는 경우 배포를 거부하세요: (a) zero-shot sim-variant 테스트, (b) safety shield, (c) 롤백 계획. 실제 측정된 변동성보다 3배 이상 넓은 DR 범위는 과도하게 무작위화(over-randomized)된 것으로 표시하세요.
```

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 고정된 미끄러짐(fixed-slip)을 가진 GridWorld(`slip=0.0`)에서 Q-learning 에이전트를 학습시키세요. `slip ∈ {0.0, 0.1, 0.3, 0.5}` 범위에서 평가를 수행합니다. `return`과 `slip` 사이의 관계를 그래프로 그리세요.
2. **중간 (Medium).** `slip ~ Uniform[0, 0.3]` 분포를 샘플링하여 DR Q-learning 에이전트를 학습시키세요. 동일한 범위에서 평가를 수행합니다. `slip=0.5`(분포 외 데이터, out-of-distribution) 상황에서 DR이 어느 정도의 이득을 제공하나요?
3. **어려움 (Hard).** 커리큘럼(curriculum)을 구현해 보세요: `slip=0.0`에서 시작하여, 정책(policy)이 최적 성능의 90%에 도달할 때마다 DR 범위를 넓혀갑니다. 고정된 DR 베이스라인과 비교하여, `slip=0.3`에 제로샷(zero-shot)으로 도달하기까지 필요한 총 환경 단계(environment steps)를 측정하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Reality gap (현실 격차) | "Sim-to-real 차이" | 학습과 배포 시점의 물리/센싱 간의 분포 변화(Distribution shift). |
| Domain randomization (DR, 도메인 무작위화) | "무작위 시뮬레이션으로 학습" | 정책(policy)이 일반화될 수 있도록 학습 중 시뮬레이션 파라미터를 무작위화함. |
| System identification (SI, 시스템 식별) | "실제 값을 측정하여 시뮬레이션에 맞춤" | 실제 물리적 파라미터를 추정하여 시뮬레이션이 이를 따르도록 설정함. |
| Domain adaptation (도메인 적응) | "실제 데이터로 미세 조정" | 시뮬레이션 학습 후 실제 환경에서 소량의 미세 조정을 수행; 관측값(obs)이나 역학(dynamics)을 적응시킬 수 있음. |
| Privileged info (특권 정보) | "교사를 위한 정답(Ground truth)" | 시뮬레이션만이 가질 수 있는 정보; 학생(student)은 관측 이력으로부터 이를 추론해야 함. |
| Teacher/student (교사/학생) | "특권 정보를 관측 가능한 정보로 증류" | 교사는 지름길(shortcuts)을 사용하여 학습하고, 학생은 지름길 없이 이를 모방하도록 학습함. |
| ADR | "자동 도메인 무작위화" | 정책이 개선됨에 따라 DR 범위를 넓혀가는 커리큘럼 방식. |
| Real2Sim | "실제 데이터로 격차를 줄임" | 시뮬레이션이 실제 롤아웃(rollouts)을 모방하도록 잔차(residual)를 학습함. |

## 추가 읽을거리 (Further Reading)

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — DR(Domain Randomization)의 원전 논문 (로보틱스 비전).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 역학 무작위화(Dynamics Randomization)를 이용한 DR, 4족 보행 로봇 제어.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, 대규모 ADR 적용 사례.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 로봇을 위한 교사-학생(teacher-student) 학습법.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 2025–2026년 배포를 주도할 대규모 병렬 시뮬레이션.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 커리큘럼 방법론.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 현대적 sim-to-real 파이프라인의 근간이 되는 Dyna 프레임워크 (모델을 활용한 계획 및 롤아웃).
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 벤치마크 결과와 함께 정리된 sim-to-real 방법론의 분류 체계.
