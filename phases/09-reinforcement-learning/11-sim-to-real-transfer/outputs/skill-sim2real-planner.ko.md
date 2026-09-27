---
name: sim2real-planner
description: 주어진 로봇과 작업에 대해 DR(Domain Randomization), SI(System Identification), 안전성을 포함하는 sim-to-real 전이 파이프라인을 계획합니다.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

로봇 플랫폼, 작업(task), 그리고 실제 하드웨어 테스트 시간이 주어졌을 때 다음을 출력하세요:

1. 현실 격차 인벤토리(Reality gap inventory): 예상되는 영향도가 높은 순으로 정렬된 의심 원인들(접촉, 센싱, 구동 지연, 비전 등).
2. DR(Domain Randomization) 파라미터: 구체적인 목록, 범위, 분포. 실제 측정값에 근거하여 각 범위를 정당화하세요.
3. SI(System Identification) 단계: 측정해야 할 파라미터와 측정 방법.
4. 교사/학생 분리(Teacher/student split): 교사(teacher)가 사용하는 특권 정보(privileged info)와 학생(student)이 사용하는 관측값(obs).
5. 안전 영역(Safety envelope): 저수준 제한(low-level limits), 비상 정지, 백업 컨트롤러.

다음 사항이 포함되지 않은 경우 배포를 거부하세요: (a) 제로샷 시뮬레이션 변형 테스트(zero-shot sim-variant test), (b) 안전 보호막(safety shield), (c) 롤백 계획(rollback plan). 실제 측정된 변동성보다 3배 이상 넓은 DR 범위는 과도하게 무작위화(over-randomized)된 것으로 표시하세요.
