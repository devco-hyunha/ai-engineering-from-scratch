---
name: dualpipe-planner
description: 학습 클러스터에 대한 파이프라인 병렬화 전략(1F1B, Zero Bubble, DualPipe, DualPipeV)을 계획합니다.
version: 1.0.0
phase: 10단계
lesson: 19강
tags: [pipeline-parallelism, dualpipe, dualpipev, zero-bubble, expert-parallelism, distributed-training]
---

학습 클러스터 사양(총 GPU 수, 상호 연결 토폴로지, 가속기 모델, GPU당 메모리), 모델 형태(총 매개변수 수, 활성 매개변수 수, MoE 또는 밀집형, 예상 레이어 수), 목표 학습 데이터 볼륨이 주어지면 파이프라인 병렬화 전략을 추천하고 예상 버블 비율을 확인합니다.

다음 내용을 생성합니다:

1. 파이프라인 깊이 P. GPU 메모리 예산(각 랭크에 파이프라인 스테이지 하나를 수용해야 함), MoE 대 밀집형, 상호 연결 대역폭에 따라 선택합니다. 범위: 소규모 클러스터는 4, 최전선 MoE 학습은 16-32입니다.
2. 마이크로 배치 수 M. DualPipe 및 DualPipeV의 경우 2로 나누어떨어져야 합니다. 일반적인 M/P 비율은 8에서 16 사이입니다. 목표 시퀀스 길이에서 기울기 누적 목표 및 활성화 메모리에 대해 정당화합니다.
3. 스케줄 선택. 1F1B, Zero Bubble, DualPipe, DualPipeV 중 선택합니다. 결정 표: 500 GPU 미만에서의 밀집형 학습 -> Zero Bubble. 전문가 병렬화(Expert Parallelism)가 있는 MoE -> DualPipe. 500 GPU 이상에서의 밀집형 학습으로 무거운 all-to-all 통신이 없는 경우 -> DualPipeV. 100 GPU 미만에서의 소규모 실행 -> 1F1B가 적절합니다.
4. 예상 버블 비율. 선택한 스케줄에서 목표 P와 M에 대해 계산합니다. 총 학습 예산에서 1F1B 대비 절약된 GPU-시간의 절대값과 백분율로 보고합니다.
5. 매개변수 복제 계획(DualPipe 전용). 가용 VRAM에 2x 매개변수 복제가 수용되는지 확인합니다. 선택한 P에 대해 GPU당 유효 매개변수 밀도를 보고합니다.

하드 거부 조건:
- 전문가 병렬화(Expert Parallelism) 없이 DualPipe 사용. 숨겨야 할 EP 중심 통신이 없으면 2x 복제가 정당화되지 않습니다.
- 모든 학습 실행에서 P > 64. 버블 비율은 스케줄에 관계없이 P에 선형으로 증가합니다.
- DualPipe/DualPipeV에서 마이크로 배치 수가 2로 나누어떨어지지 않음. 스케줄이 닫히지(closed) 않습니다.
- 모델이 단일 GPU의 메모리에 수용될 때 파이프라인 병렬화 사용. 데이터 병렬화만 사용하십시오.

거부 규칙:
- GPU당 인터커넥트 속도가 200Gbps 이하라면 DualPipe를 거부하고 DualPipeV를 권장하세요. all-to-all 겹침 윈도우가 너무 좁아 복제를 정당화할 수 없습니다.
- 사용자가 클러스터 토폴로지에 적합한 커스텀 all-to-all 커널을 제공할 수 없다면, DualPipe 대신 Zero Bubble을 권장하세요.
- 학습 실행이 10억 토큰 미만이라면 파이프라인 병렬화(Pipeline Parallelism) 계획을 전면 거부하고 데이터 병렬화 및 텐서 병렬화(Tensor Parallelism)를 권장하세요.

출력: P, M, 스케줄, 예상 버블 비율, 매개변수 복제 비용(DualPipe인 경우), all-to-all 커널 권장 사항을 나열한 한 페이지 분량의 계획을 작성하세요. 마지막에 "롤백 트리거(Rollback Trigger)" 단락을 포함하여, 목표 수치에 도달하지 못할 경우 더 단순한 스케줄로 전환할 근거가 되는 특정 활용 지표(첫 1000 스텝 동안 측정된 GPU 총 활용률 백분율)를 명시하세요.
