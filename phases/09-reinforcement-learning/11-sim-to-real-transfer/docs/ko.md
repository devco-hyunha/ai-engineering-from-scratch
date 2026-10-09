# 시뮬레이션에서 실물로의 전이

> 시뮬레이터에서 학습한 정책이 하드웨어에서 실패한다면, 그 정책은 시뮬레이터를 암기했을 뿐입니다. 도메인 랜덤화, 도메인 적응, 시스템 식별은 학습된 컨트롤러가 현실 격차(reality gap)를 넘어서게 하는 세 가지 도구입니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 9단계 · 08강 (PPO), 2단계 · 10강 (편향/분산)
**시간:** 약 45분

## 문제점

실제 로봇을 학습하는 것은 느리고, 위험하며, 비용이 많이 듭니다. 이족 로봇은 걷는 법을 배우기 위해 수백만 개의 학습 에포크가 필요하며, 실제 이족 로봇이 한 번만 넘어져도 하드웨어가 파손됩니다. 시뮬레이션은 무제한의 리셋, 결정적인 재현성, 병렬 환경, 그리고 물리적 손상의 부재라는 장점을 제공합니다.

그러나 시뮬레이터는 부정확합니다. 베어링은 MuJoCo 모델보다 마찰이 더 크고, 카메라는 시뮬레이터가 포함하지 않는 렌즈 왜곡을 가지며, 모터는 99%의 시뮬레이션 모델이 생략하는 지연, 백래시, 포화 현상을 포함합니다. 바람, 먼지, 변화하는 조명은 무균적인 렌더링으로 학습된 정책을 방해합니다. **현실 격차(reality gap)** — 시뮬레이션 분포와 실제 분포 간의 체계적인 차이 —는 로봇 공학에서 배포된 RL의 핵심 문제입니다.

시뮬레이션에서 실물로 넘어가는 분포 이동(sim-to-real distribution shift)에 *강인한* 정책이 필요합니다. 세 가지 역사적 접근 방식은 다음과 같습니다: 시뮬레이터를 랜덤화하는 것(도메인 랜덤화), 약간의 실제 데이터로 정책을 적응시키는 것(도메인 적응 / 미세 조정), 실제 시스템의 매개변수를 식별하고 일치시키는 것(시스템 식별). 2026년에는 대규모 병렬 시뮬레이션(Isaac Sim, Isaac Lab, GPU의 Mujoco MJX)과 세 가지 방법을 모두 결합하는 레시피가 지배적입니다.

## 개념

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**도메인 랜덤화(DR).** Tobin et al. 2017, Peng et al. 2018. 학습 중에 실제 로봇에서 다를 수 있는 모든 시뮬레이션 매개변수를 랜덤화합니다: 질량, 마찰 계수, 모터 PD 게인, 센서 잡음, 카메라 위치, 조명, 텍스처, 접촉 모델. 정책은 "오늘 어떤 시뮬레이터에 있는지"에 대한 조건부 분포를 학습하고 전체 범위에 걸쳐 일반화합니다. 실제 로봇이 학습 범주(training envelope) 내에 있다면 정책이 작동합니다.

- **장점:** 실제 데이터가 필요 없습니다. 하나의 레시피로 여러 로봇에 적용할 수 있습니다.
- **단점:** 과도한 무작위화 학습은 "보편적"이지만 지나치게 보수적인 정책을 생성합니다. 너무 많은 노이즈는 ≈ 너무 많은 정규화와 같습니다.

**시스템 식별(System Identification).** 학습 전에 시뮬레이터의 매개변수를 실제 세계 데이터에 맞춰 조정합니다. 실제 로봇에서 팔 관절 마찰을 측정할 수 있다면, 그 값을 시뮬레이터에 입력하세요. 그런 다음 그 값들을 예상하는 정책을 학습합니다. 실제 시스템에 접근할 필요가 있지만, 현실 격차(reality gap)를 직접 줄여줍니다.

- **장점:** 정밀하고 노이즈가 적은 학습 목표.
- **단점:** 잔여 모델 오차는 정책에 보이지 않습니다. 식별되지 않은 작은 효과(예: 모터 데드밴드)는 여전히 배포를 망가뜨립니다.

**도메인 적응(Domain Adaptation).** 시뮬레이션에서 학습한 후, 소량의 실제 데이터로 미세 조정합니다. 두 가지 방식이 있습니다:

- **Real2Sim2Real:** 실제 롤아웃을 사용하여 잔여 시뮬레이터 `f(s, a, z) - f_sim(s, a)`를 학습하고, 보정된 시뮬레이션에서 학습합니다. 많은 실제 데이터 없이 격차를 줄입니다.
- **관측 적응:** 학습된 기능 추출기(예: GAN pixel-to-pixel)를 통해 실제 관측(obs)을 시뮬레이션 유사 관측으로 매핑하는 정책을 학습합니다. 컨트롤러는 시뮬레이션에 남아 있습니다.

**특권 학습 / 교사-학생(Teacher-Student).** Miki et al. 2022 (ANYmal 사족보행 로봇). 특권 정보(정답 마찰, 지형 높이, IMU 드리프트)에 접근할 수 있는 *교사*를 시뮬레이션에서 학습합니다. 실제 센서 관측만 볼 수 있는 *학생*을 증류(distill)합니다. 학생은 과거 기록(history)로부터 특권 기능을 추론하는 법을 배우며, 물리적 매개변수에 대해 강건합니다.

**대규모 병렬 시뮬레이션.** 2024–2026. Isaac Lab, Mujoco MJX, Brax는 단일 GPU에서 수천 개의 병렬 로봇을 실행합니다. 4,096개의 병렬 휴머노이드를 사용하는 PPO는 몇 시간 안에 수년간의 경험을 수집합니다. 학습 분포가 넓어짐에 따라 "현실 격차"가 줄어들며, 4,096개 환경 각각이 서로 다른 무작위화 매개변수를 가질 경우 도메인 무작위화(DR)는 거의 무료가 됩니다.

**2026년 실제 세계 레시피 (사족보행 걷기 예시):**

1. 중력, 마찰, 모터 게인, 페이로드가 도메인 무작위화된 대규모 병렬 시뮬레이션.
2. 특권 정보(지형 맵, 신체 속도 정답)로 학습된 교사 정책.
3. 고유수용성(proprioception, 다리 관절 인코더)만 사용하여 교사로부터 증류된 학생 정책.
4. 실제 IMU에 대한 오토인코더를 통한 선택적 관측 적응.
5. 배포합니다. 10개 이상의 환경에서 제로샷(Zero-Shot)으로 실행합니다. 실패할 경우, 안전 제약이 있는 PPO를 사용하여 몇 분간의 실제 미세 조정(Fine-tuning)을 수행합니다.

```figure
f3-reality-gap
```

## 구현하기

이 강의의 코드는 *잡음*이 있는 전이(transitions)를 가진 GridWorld에서 도메인 랜덤화(Domain Randomization)를 시연하는 작은 예제입니다. "sim"에서는 랜덤화된 미끄러짐(slip) 확률을 경험하며 정책을 학습하고, 학습 중 보지 못한 미끄러짐 수준을 가진 "real" 환경에서 평가합니다. 이 형태는 MuJoCo에서 하드웨어로 전이하는 경우와 직접적으로 대응됩니다.

### 1단계: 매개변수화된 시뮬레이터

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip`는 시뮬레이터가 노출하는 매개변수입니다. 실제 로봇공학에서는 마찰, 질량, 모터 게인 등 시뮬레이션과 실제 환경 간에 변동하는 모든 것이 될 수 있습니다.

### 2단계: DR로 학습

각 에포크(Epoch)의 시작 시 `slip ~ Uniform[0.0, 0.4]`를 샘플링합니다. PPO / Q-learning / 기타 방법을 사용하여 학습합니다. 이를 여러 에포크에 걸쳐 수행합니다.

### 3단계: "real" 미끄러짐에 대해 제로샷으로 평가

`slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`에서 평가합니다. 처음 네 값은 학습 지원(support) 범위 내에 있으며, `0.5`과 `0.7`는 범위 밖에 있습니다. DR로 학습된 정책은 지원 범위 내에서 거의 최적 상태를 유지하고, 범위 밖에서는 우아한 저하(Graceful Degradation)를 보여야 합니다. 고정 미끄러짐으로 학습된 정책은 학습된 미끄러짐 밖에서는 취약(brittle)할 것입니다.

### 4단계: 좁은 학습과 비교

두 번째 정책을 `slip = 0.0`에서만 학습합니다. 동일한 `slip` 스윕(sweep)에서 평가합니다. 실제 미끄러짐이 0보다 커지는 즉시 catastrophic drop(치명적인 성능 저하)이 발생해야 합니다.

## 함정

- **너무 많은 랜덤화.** `slip ∈ [0, 0.9]`에서 학습하면 정책이 너무 위험 회피적이어서 최적 경로를 시도하지 않습니다. "무엇이든 일어날 수 있다"가 아니라 *기대되는* 실제 세계 분포에 맞춰야 합니다.
- **너무 적은 랜덤화.** 얇은 슬라이스(slice)에서 학습하면 정책이 전혀 일반화되지 못합니다. 정책이 개선됨에 따라 분포를 넓히는 적응형 커리큘럼(Automatic Domain Randomization)을 사용하세요.
- **잘못 식별된 매개변수 공간.** 잘못된 것을 랜덤화하면(예: 실제 격차가 모터 지연인데 카메라 색조를 랜덤화) DR이 도움이 되지 않습니다. 먼저 실제 로봇을 프로파일링하세요.
- **특권 정보 누출.** 전역 상태(global state)를 사용하여 행동하는 교사(teacher)는 학생(student)이 따라잡을 수 없는 결과를 낳을 수 있습니다. 관찰 기록(observation history)을 고려할 때 학생이 교사 정책을 실현(realizable)할 수 있는지 확인하세요.
- **시뮬레이션 간 전이 실패.** 정책이 더 어려운 시뮬레이션 변형에 대해 강인하지 않다면, 실제 세계에서도 강인하지 않을 것입니다. 배포하기 전에 항상 홀드아웃 시뮬레이션 변형으로 테스트해 보세요.
- **실제 세계 안전 엔벨로프 없음.** 시뮬레이션에서 작동하고 "실제에서도 작동"하는 정책이 저수준 안전 보호막(shield) 없이 하드웨어를 파괴할 수 있습니다. 학습되지 않은 컨트롤러에 속도 제한, 토크 제한, 관절 제한을 추가하세요.

## 사용하기

2026년 시뮬레이션-실제(sim-to-real) 스택:

| 도메인 | 스택 |
|--------|-------|
| 다족 보행 (ANYmal, Spot, 휴머노이드) | Isaac Lab + DR + 특권 교사/학생 |
| 조작 (다섯 손가락 손, 픽 앤 플레이스) | Isaac Lab + DR + 비전용 DR-GAN |
| 자율 주행 | CARLA / NVIDIA DRIVE Sim + DR + 실제 미세 조정 |
| 드론 레이싱 | RotorS / Flightmare + DR + 온라인 적응 |
| 손가락/손 내 조작 | OpenAI Dactyl (전례 없는 규모의 DR) |
| 산업용 로봇 팔 | MuJoCo-Warp + SI + 작은 실제 미세 조정 |

모든 규모의 제어에 대해 워크플로가 일관됩니다: 시뮬레이션을 최대한 맞추고, 맞출 수 없는 것은 랜덤화하고, 거대한 정책을 학습하고, 증류(distill)하고, 안전 보호막(shield)과 함께 배포하세요.

## 출시하기

`outputs/skill-sim2real-planner.md`로 저장하세요:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 연습 문제

1. **쉬움.** 고정 슬립 GridWorld (slip=0.0)에서 Q-learning 에이전트를 학습하세요. slip ∈ {0.0, 0.1, 0.3, 0.5}에서 평가하세요. slip 대비 리턴(return)을 플롯하세요.
2. **중간.** `slip ~ Uniform[0, 0.3]`을 샘플링하는 DR Q-learning 에이전트를 학습하세요. 동일한 스윕(sweep)을 평가하세요. slip=0.5 (분포 밖)에서 DR이 얼마나 도움이 됩니까?
3. **어려움.** 커리큘럼을 구현하세요: slip=0.0으로 시작하고, 정책이 최적의 90%에 도달할 때마다 DR 범위를 넓히세요. slip=0.3에서 제로샷(zero-shot)으로 도달하는 데 필요한 총 환경 스텝을 고정 DR 기준선과 비교하여 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 현실 격차 | "시뮬레이션-실제 차이" | 학습 및 배포 물리/센싱 간의 분포 이동(Distribution Shift). |
| 도메인 랜덤화 (DR) | "랜덤 시뮬레이션으로 학습" | 학습 중 시뮬레이션 매개변수를 랜덤화하여 정책이 일반화되도록 합니다. |
| 시스템 식별 (SI) | "실제 측정 후 시뮬레이션에 적용" | 실제 물리 매개변수를 추정하고, 시뮬레이션이 이를 따르도록 설정합니다. |
| 도메인 적응 | "실제 데이터로 미세 조정" | 시뮬레이션 학습 후 실제 데이터로 미세 조정합니다. 관측값이나 동역학을 적응시킬 수 있습니다. |
| 특권 정보 | "교사용 정답" | 시뮬레이션만 가진 정보입니다. 학생은 관측 기록에서 이를 추론해야 합니다. |
| 교사/학생 | "특권 정보 -> 관측 가능한 정보로 증류" | 교사는 지름길(shortcut)을 사용하여 학습합니다. 학생은 지름길 없이 이를 모방하는 법을 배웁니다. |
| ADR | "자동 도메인 랜덤화" | 정책이 개선됨에 따라 DR 범위를 넓히는 커리큘럼입니다. |
| Real2Sim | "실제 데이터로 간극을 줄이기" | 시뮬레이션이 실제 롤아웃을 모방하도록 잔차(residual)를 학습합니다. |

## 추가 읽기

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 원본 DR 논문 (로보틱스를 위한 비전).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 동역학 및 4족 보행 로보틱스를 위한 DR.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, 대규모 ADR.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal을 위한 교사-학생 학습.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 2025–2026년 배포를 주도하는 대규모 병렬 시뮬레이션.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 커리큘럼 방법.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 프레임워크 (계획 및 롤아웃에 모델 사용)로, 현대적인 시뮬레이션-실제(sim-to-real) 파이프라인의 기반이 됩니다.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 벤치마크 결과가 포함된 시뮬레이션-실제(sim-to-real) 방법 분류 체계.
