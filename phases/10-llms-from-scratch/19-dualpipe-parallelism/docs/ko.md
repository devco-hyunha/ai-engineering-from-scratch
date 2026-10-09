# DualPipe 병렬화

> DeepSeek-V3는 2,048개의 H800 GPU로 학습되었으며, MoE 전문가가 노드 전체에 분산되어 있습니다. 노드 간 전문가 all-to-all 통신은 계산 1 GPU-hour마다 통신 1 GPU-hour의 비용을 발생시켰습니다. GPU는 절반의 시간을 유휴 상태로 보냈습니다. DualPipe (DeepSeek, 2024년 12월)는 순방향 및 역방향 계산이 유발하는 all-to-all 통신과 이를 겹치는 양방향 파이프라인입니다. 버블이 줄어들고 처리량이 증가하며, 전문가 병렬화(Expert Parallelism)가 이미 랭크(ranks) 전체에 전문가를 분산하고 있으므로 두 개의 모델 매개변수 복사본("dual"이라는 이름의 유래)을 유지하는 비용은 저렴합니다. 이 강의는 DualPipe가 실제로 무엇을 수행하는지, 그리고 Sea AI Lab의 DualPipeV 개선이 버블을 약간 더 조이는 대가로 2배 매개변수 비용을 제거하는 이유를 설명하는 Learn 유형 안내입니다.

**유형:** Learn
**언어:** Python (stdlib, 스케줄 시뮬레이터)
**선수 요건:** 10단계 · 05강 (분산 학습, FSDP, DeepSpeed), 10단계 · 14강 (오픈 모델 아키텍처 및 MoE)
**시간:** 약 60분

## 학습 목표

- DualPipe 순방향-역방향 청크의 네 가지 구성 요소를 나열하고, 각 요소가 왜 자체 겹침 윈도우를 갖는지 설명해 보세요.
- 대규모에서의 파이프라인 버블 문제를 설명하고, "버블 프리(bubble-free)"가 마케팅에서의 의미와 실제에서의 의미가 어떻게 다른지 설명해 보세요.
- 8개의 PP 랭크와 16개의 마이크로 배치로 DualPipe 스케줄을 수동으로 추적하고, 순방향 및 역방향 스트림이 서로의 유휴 슬롯을 채우는지 확인해 보세요.
- DualPipeV (Sea AI Lab, 2025)가 취하는 트레이드오프를 명시해 보세요: 전문가 병렬화(Expert Parallelism)가 비활성화될 때 버블이 약간 커지는 대가로 2배 매개변수 복제를 제거합니다.

## 문제점

2k H800 GPU로 671B MoE 모델을 학습하면 세 가지 복합적인 병목 현상에 직면합니다:

1. **메모리 압력.** 각 GPU는 모델의 일부 슬라이스를 보유합니다. 61개 레이어와 128개 헤드에 걸쳐 시퀀스 8k의 활성화 메모리는 방대합니다.
2. **파이프라인 버블.** 전통적인 파이프라인 병렬화(GPipe, 1F1B)는 GPU가 해당 스테이지의 입력이나 기울기를 기다리는 동안 유휴 상태로 남습니다. 8개 스테이지에서는 1F1B 스케줄링을 사용하더라도 GPU 시간의 약 12%가 버블이 될 수 있습니다.
3. **노드 간 all-to-all.** 전문가 병렬화(Expert Parallelism)를 사용하는 MoE (혼합 전문가)(MoE (Mixture of Experts))는 전문가를 여러 노드에 분산합니다. 모든 순전파(forward pass)는 토큰을 해당 전문가로 전달하기 위해 all-to-all을 트리거하며, 결과를 합치기 위해 또 다른 all-to-all을 수행합니다. 2k GPU 환경에서는 연산 대 통신 비율이 1:1이 되기 쉽습니다.

이 각각은 별도의 해결책이 있습니다: 메모리 문제는 기울기 체크포인팅(Gradient Checkpointing)으로, 파이프라인 버블은 Zero Bubble (Sea AI Lab, 2023)으로, all-to-all은 전문가 병렬화 통신 커널로 해결합니다. DualPipe가 하는 일은 이들을 함께 작동하게 만드는 것입니다. 단일 순전파-역전파 청크(chunk) 내에서 연산과 통신을 겹치고, 파이프라인의 양쪽 끝에서 동시에 마이크로 배치를 주입하며, resulting schedule을 사용하여 all-to-all을 연산 윈도우 안에 숨깁니다.

보고된 결과: 파이프라인 버블의 거의 완전한 제거, DeepSeek-V3의 14.8T 토큰 학습 실행에서 95% 이상의 GPU 활용률.

## 개념

### 파이프라인 병렬화(Pipeline Parallelism) 복습

N층 모델을 P개의 장치로 분할합니다. 장치 `i`는 층 `i * N/P .. (i+1) * N/P - 1`를 담당합니다. 마이크로 배치는 장치 0에서 P-1까지 순전파로 흐른 후, P-1에서 0으로 역전파됩니다. 각 장치는 이전 장치가 출력값을 보낼 때만 순전파 단계를 시작할 수 있으며, 하류 장치가 상류 기울기를 보낼 때만 역전파를 시작할 수 있습니다.

GPipe (Huang et al., 2019)는 한 번에 하나의 마이크로 배치를 스케줄링하므로 GPU 시간의 대부분을 낭비합니다. 1F1B (Narayanan et al., 2021)는 여러 마이크로 배치의 순전파와 역전파를 인터리빙(interleave)합니다. Zero Bubble (Qi et al., 2023)은 역전파를 두 부분, 즉 입력에 대한 역전파(B)와 가중치에 대한 역전파(W)로 나누고, 버블을 채우도록 스케줄링합니다. Zero Bubble 이후 파이프라인은 거의 빈틈이 없습니다.

DualPipe는 그 다음 단계입니다. 두 가지 아이디어를 추가합니다:

### 아이디어 1: 청크 분해

각 순전파 청크는 네 가지 구성 요소로 나뉩니다:

- **어텐션(Attention).** Q/K/V 투사, 어텐션, 출력 투사.
- **All-to-all 전달.** 토큰을 해당 전문가로 보내는 노드 간 통신.
- **MLP.** MoE 전문가 연산.
- **All-to-all 합치기.** 전문가 출력값을 가져오는 노드 간 통신.

역방향 청크는 각각의 기울기 버전을 추가합니다. DualPipe는 all-to-all dispatch가 다음 청크의 어텐션 연산과 병렬로 실행되고, all-to-all combine이 그 다음 청크의 MLP 연산과 병렬로 실행되도록 스케줄링합니다.

### 아이디어 2: 양방향 스케줄링

대부분의 파이프라인 스케줄은 stage 0에서 micro-batch를 주입하여 stage P-1로 흐르게 합니다. DualPipe는 양쪽 끝에서 micro-batch를 주입합니다. Stage 0은 거기서 시작하는 순방향 micro-batch를 보고, stage P-1도 거기서 시작하는 순방향 micro-batch를 봅니다. 두 스트림은 중간에서 만납니다.

이것이 작동하려면 device `i`는 초기 파이프라인 레이어 `i`와 후기 파이프라인 레이어 `P - 1 - i`를 모두 보유해야 합니다. 이것이 DualPipe의 "dual" 부분입니다: 각 device는 서비스해야 하는 모델 레이어의 두 복사본(각 방향용)을 유지합니다. DeepSeek-V3의 규모에서는 이것이 2배의 파라미터 복제 비용입니다. Expert Parallelism이 MoE expert를 이미 매우 얇게 분산시켜 non-expert 레이어를 두 번 복제하는 비용은 미미하기 때문에 감당할 수 있습니다.

중요한 점은 한 방향의 순방향 스트림과 다른 방향의 역방향 스트림이 단방향 스케줄에서 버블이 생길 위치에서 정확히 겹친다는 것입니다. 버블은 사라집니다.

### 수동으로 추적한 스케줄

P = 4 ranks, 8 micro-batches, 순방향 4 / 역방향 4로 나누어 생각해 보세요. 시간은 왼쪽에서 오른쪽으로 흐르며, 행은 device rank입니다.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

"F4/F5R" 표기를 읽어 보세요: rank 1은 같은 시간 슬롯에서 micro-batch 4의 순방향(파이프라인에서 왼쪽에서 오른쪽으로 진행)과 micro-batch 5의 순방향(오른쪽에서 왼쪽으로 진행)을 실행하고 있습니다. 이것이 "양방향"의 운영적 의미입니다.

Rank 2에서는 교차 스트림이 더 빨리 겹치고, rank 00강 P-1에서는 가장 늦게 겹칩니다. 스케줄의 안정된 중간 단계에서 모든 rank는 X 방향의 순방향 연산이 Y 방향의 역방향 연산과 겹쳐 실행됩니다. 연산은 바쁩니다. 순방향 패스의 all-to-all dispatch는 역방향 연산 내부에 숨겨지고, all-to-all combine은 순방향 연산 내부에 숨겨집니다. 버블은 짜내어져 제거됩니다.

### 버블 계산

표준 1F1B 파이프라인 버블 (rank당 낭비 시간):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 개선은 버블을 줄이지만, 완전히 없애지는 못합니다. DualPipe는 안정 단계(stable phase)에서 마이크로 배치(micro-batch) 개수가 파이프라인 깊이의 2배로 나누어떨어질 때 버블이 0이 됩니다. 안정 단계 밖(워밍업 및 쿨다운)에서는 버블이 존재하지만, 마이크로 배치 개수가 증가해도 버블이 커지지 않습니다. 이는 논문이 강조하는 핵심 특성입니다.

마케팅 용어로는 "버블 프리(bubble-free)"입니다. 기술적 용어로는 버블이 마이크로 배치 개수에 따라 증가하지 않는다는 의미입니다. Sea AI Lab의 후속 분석(DualPipeV / Cut-in-half)에 따르면, 전문가 병렬화(Expert Parallelism)가 병목이 아닐 때만 완전한 제로 버블이 가능합니다. EP 기반의 all-to-all 통신이 있는 경우, 항상 스케줄링의 타협점이 존재합니다.

### DualPipeV — 개선안

Sea AI Lab (2025)은 EP 통신 겹침(overlap)이 핵심이 아닐 때 2배 매개변수 복제가 낭비적이라는 것을 관찰했습니다. 그들의 DualPipeV 스케줄은 양방향 주입(bidirectional injection)을 단일 매개변수 복사본에서 실행되는 "V자형" 스케줄로 접어 넣습니다. 버블은 DualPipe보다 약간 더 크지만, 메모리 절감 효과는 상당합니다. DeepSeek는 오픈소스 DualPipe 구현에서 EP-off 모드로 DualPipeV를 채택했습니다.

트레이드오프:

| 기능 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 장치당 매개변수 복사본 | 2 | 1 | 1 | 1 |
| 버블 vs 마이크로 배치 | 일정 | 작은 증가 | 증가 | 증가 |
| 연산-통신 겹침 | 완전 | 부분 | 최소 | 부분 |
| 사용 시점 | EP 중심 MoE | 밀집 모델 또는 EP-light | 기준선 | 모든 파이프라인 |

### 14.8T 토큰 학습에 미치는 의미

DeepSeek-V3의 사전 학습은 약 2.8M GPU-hours 동안 2,048개의 H800 GPU에서 14.8T 토큰을 소비했습니다. 단순한 1F1B를 사용했다면 파이프라인 버블로 인해 그 중 12-15%를 잃었을 것이며, 이는 340-420K GPU-hours에 해당합니다. 이는 70B 모델을 완전히 학습할 수 있는 양입니다. DualPipe는 이 대부분을 회복했습니다. 내부 로그 없이는 기여도를 직접 정량화하기 어렵지만, 논문에서는 학습 전체 평균 GPU 활용률이 95% 이상이라고 주장합니다.

더 작은 규모(1,000 GPU 미만)의 학습에서는 DualPipe가 과잉입니다. 파이프라인 버블은 총 비용에 비해 상대적으로 작으며, 밀집 모델 학습은 거의 all-to-all 병목에 도달하지 않습니다. 수천 GPU 규모의 프론티어 MoE 학습에서는 사실상 필수적입니다.

### 스택에서의 위치

- **FSDP**(10단계 · 05강)와 상호 보완적입니다. FSDP는 모델 매개변수를 랭크 간에 분할하고, DualPipe는 연산을 랭크 간에 스케줄링합니다. 두 기법을 결합할 수 있습니다.
- **ZeRO-3** 기울기 분할과 호환됩니다. 두 복사본 복제를 위한 기록(bookkeeping)은 ZeRO의 분할된 기울기와 협력해야 합니다.
- 특정 클러스터 토폴로지에 맞춰 튜닝된 **커스텀 all-to-all 커널**이 필요합니다. DeepSeek의 오픈소스 커널이 참조 구현입니다.

```figure
expert-capacity
```

## 사용하기

`code/main.py`는 파이프라인 스케줄링 시뮬레이터입니다. `(P, n_micro_batches, schedule)`를 입력으로 받아 1F1B, Zero Bubble, DualPipe, DualPipeV의 안정 단계(stable-phase) 활용도를 출력합니다. 이는 교육용 도구이며, 수치 값은 논문에서의 정성적 주장과 일치하지만, 실제 측정된 생산 환경의 속도 향상 주장은 아닙니다.

시뮬레이터의 가치: 다양한 P와 마이크로 배치 개수로 실행하여, 1F1B에서는 버블 비율이 증가하지만 DualPipe에서는 그렇지 않은 현상을 관찰해 보세요.

실제 훈련 실행을 위한 통합 고려 사항:

- 마이크로 배치 개수로 깔끔하게 나누어 떨어지는 파이프라인 병렬 깊이를 선택하세요.
- 전문가 병렬화(mesh)가 양방향 all-to-all을 지원하도록 하세요. DeepSeek의 커널이 참조입니다.
- 처음에는 스케줄링 자체에 디버깅 시간으로 일 주일을 투자할 것으로 예상하세요. 기록(bookkeeping)이 까다롭습니다.
- 집계된 지표가 아닌 랭크별 GPU 활용도를 모니터링하세요. DualPipe의 이점은 지연(stragglers)을 줄이는 데서 옵니다.

## 출시하기

이 강의는 `outputs/skill-dualpipe-planner.md`를 생성합니다. 훈련 클러스터 사양(GPU 개수, 토폴로지, 상호 연결, 모델 형태)이 주어지면, 파이프라인 병렬화 전략, 사용할 스케줄링 알고리즘, 목표 규모에서의 예상 버블 비율을 추천합니다.

## 연습 문제

1. `code/main.py`를 `(P=8, micro_batches=16, schedule=dualpipe)`와 `(P=8, micro_batches=16, schedule=1f1b)`에 대해 실행하세요. GPU 활용도 차이를 계산하고, 훈련 토큰 백만 개당 회복된 GPU 시간으로 표현하세요.

2. `(P=4, micro_batches=8, schedule=dualpipe)`의 스케줄링 표를 손으로 그려 보세요. 각 시간 슬롯에 마이크로 배치 ID와 방향을 표시하세요. 버블이 없는 첫 번째 시간 슬롯을 식별하세요.

3. DeepSeek-V3 기술 보고서(arXiv:2412.19437)의 그림 5를 읽어 보세요. DualPipe 순방향 청크 내부의 all-to-all dispatch 겹침 윈도우를 식별해 보세요. 계산 스케줄이 이를 어떻게 숨기는지 설명해 보세요.

4. P=8 파이프라인 스테이지를 가진 70B 밀집 모델과 P=16 파이프라인 스테이지를 가진 671B MoE 모델에 대해 DualPipe의 2x 매개변수 오버헤드를 계산해 보세요. MoE 경우의 오버헤드가 비례적으로 더 작은 이유를 보여 주세요(대부분의 매개변수는 전문가이며, 대규모 EP 그룹에 걸쳐 샤딩됩니다).

5. DualPipe를 Chimera(2021년 경쟁 양방향 스케줄러)와 비교해 보세요. 논문 섹션 3.4를 참조하여 Chimera에 없던 DualPipe가 추가한 두 가지 특정 속성을 식별해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 파이프라인 버블 | "랭크당 유휴 시간" | 파이프라인 스테이지가 입력이나 기울기를 기다리느라 낭비된 GPU 사이클 |
| 1F1B | "기본 파이프라인 스케줄" | 하나의 순방향 / 하나의 역방향 교대 스케줄링; DualPipe가 이기는 기준선 |
| Zero Bubble | "Sea AI Lab 2023" | 역방향을 B(입력 기울기)와 W(가중치 기울기)로 분할; 파이프라인을 거의 완전히 조임 |
| DualPipe | "DeepSeek-V3 스케줄" | 양방향 파이프라인 + 계산-통신 겹침; 버블이 마이크로 배치 수에 따라 증가하지 않음 |
| DualPipeV | "반으로 자르기" | V자형 개선으로, 약간 더 큰 버블을 대가로 2x 매개변수 복제를 제거 |
| 청크 | "파이프라인 작업 단위" | 하나의 마이크로 배치가 하나의 파이프라인 스테이지를 통과하는 순방향 또는 역방향 패스 |
| All-to-all dispatch | "토큰을 전문가에게 보내기" | 토큰을 할당된 MoE 전문가로 라우팅하는 노드 간 통신 |
| All-to-all combine | "전문가 출력 가져오기" | MLP 이후 전문가 출력을 수집하는 노드 간 통신 |
| 전문가 병렬화 (EP) | "GPU 간 전문가" | MoE 전문가를 랭크에 걸쳐 샤딩하여 서로 다른 GPU가 서로 다른 전문가를 보유하도록 함 |
| 파이프라인 병렬화 (PP) | "GPU 간 레이어" | 모델 레이어를 랭크에 걸쳐 샤딩; DualPipe가 스케줄링하는 차원 |
| 버블 비율 | "낭비된 GPU 시간" | (bubble_time / total_time); DualPipe가 0에 가깝게 만드는 비율 |

## 추가 읽기

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe의 주요 참고 자료
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — DualPipeV (Cut-in-half) 모드를 포함한 오픈소스 참고 구현
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble의 전신
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — DeepSeek의 EP-off 모드에 영향을 준 DualPipeV 분석
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe가 비교하는 1F1B 스케줄
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 파이프라인 병렬화(Pipeline Parallelism)의 원 논문 및 버블 문제
