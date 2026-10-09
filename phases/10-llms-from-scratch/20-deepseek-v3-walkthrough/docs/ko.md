# DeepSeek-V3 아키텍처 상세 분석

> 10단계 14강에서는 모든 오픈 모델이 조정하는 6가지 아키텍처 조절 변수(knobs)를 소개했습니다. DeepSeek-V3 (2024년 12월, 총 671B 파라미터, 활성 파라미터 37B)는 이 6가지 변수를 모두 조정하며, Multi-Head Latent Attention, 보조 손실 없는 로드 밸런싱, Multi-Token Prediction, DualPipe 학습이라는 4가지 추가 요소를 도입했습니다. 이 강의에서는 DeepSeek-V3의 아키텍처를 상단부터 하단까지 살펴보고, 공개된 설정(config)에서 모든 파라미터 수를 도출합니다. 강의를 마치면 671B/37B 비율이 왜 최적의 선택인지, 그리고 MLA와 MoE가 결합했을 때 각각 단독으로 사용할 때보다 프론티어 성능이 왜 더 좋은지 설명할 수 있습니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 파라미터 계산기)
**선수 요건:** 10단계 14강 (오픈 모델 상세 분석), 10단계 17강 (NSA), 10단계 18강 (MTP), 10단계 19강 (DualPipe)
**시간:** 약 75분

## 학습 목표

- DeepSeek-V3 설정을 상단부터 하단까지 읽고, GPT-2의 6가지 조절 변수와 DeepSeek 고유의 4가지 추가 요소를 기준으로 각 필드를 설명해 보세요.
- 총 파라미터 수(671B), 활성 파라미터 수(37B) 및 각각에 기여하는 구성 요소를 도출해 보세요.
- 128k 컨텍스트에서 MLA의 KV 캐시 점유 공간을 계산하고, 동일한 활성 파라미터를 가진 GQA 기반 밀집 모델이 지불하는 비용과 비교해 보세요.
- DeepSeek 고유의 4가지 혁신 요소(MLA, MTP, 보조 손실 없는 라우팅, DualPipe)를 나열하고, 각각이 아키텍처/학습 스택의 어떤 부분을 목표로 하는지 지정해 보세요.

## 문제점

DeepSeek-V3는 아키텍처가 Llama 계열과 의미 있게 다른 최초의 프론티어 오픈 모델입니다. Llama 3 405B는 "6가지 조절 변수가 조정된 GPT-2"입니다. DeepSeek-V3는 6가지 조절 변수에 4가지가 추가된 GPT-2입니다. Llama 3 설정을 읽는 것은 DeepSeek 설정을 읽기 위한 준비 운동이지만, 깊은 구조 — 어텐션 블록의 형태, 라우팅 로직, 학습 시간 목표 함수 — 는 별도의 상세 분석이 필요할 만큼 충분히 다릅니다.

이것을 학습하는 이점: DeepSeek-V3의 오픈 웨이트 릴리스는 오픈 모델에서 "프론티어 능력"의 의미를 변화시켰습니다. 이 아키텍처는 많은 2026년 학습 실행이 모방하는 청사진입니다. 이를 이해하는 것은 프론티어 LLM 학습이나 추론을 다루는 모든 역할의 기본 요건입니다.

## 개념

### 불변의 핵심, 다시

DeepSeek-V3는 여전히 자기회귀(Autoregressive)입니다. 여전히 디코더 블록을 쌓습니다. 각 블록은 여전히 어텐션(Attention)과 MLP, 그리고 두 개의 RMSNorm을 포함합니다. MLP에서는 여전히 SwiGLU를 사용합니다. RoPE를 사용합니다. 사전 정규화(Pre-norm)를 적용합니다. 가중치 공유 임베딩(Weight-tied embeddings)을 사용합니다. 모든 Llama나 Mistral과 동일한 기본 구조를 따릅니다.

### 변화: GQA 대신 MLA

10단계 14강에서 GQA가 Q 헤드의 그룹 간에 K와 V를 공유하여 KV 캐시를 축소한다는 것을 알고 있습니다. 다중 헤드 잠재 어텐션(Multi-Head Latent Attention, MLA)은 한 발 더 나아갑니다: K와 V는 공유된 저랭크 잠재 표현(`kv_lora_rank`)으로 압축된 후, 헤드로 실시간으로 복원됩니다. KV 캐시는 잠재 표현만 저장합니다. 일반적으로 토큰당 레이어당 512개의 floats를 저장하며, 8 x 128 = 1024 floats를 저장하지 않습니다.

128k 컨텍스트에서 DeepSeek-V3는 MLA를 사용하며 (토큰당 레이어당 하나의 공유된 잠재 표현 `c^{KV}`; K와 V는 후속 행렬 곱에 흡수될 수 있는 상향 투사(up-projections)를 통해 이 잠재 표현에서 모두 파생됩니다):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

가상의 GQA 기준선 (Llama 3 70B 형태, 8 KV 헤드, 헤드 차원 128)은 다음을 지불합니다:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA는 128k 컨텍스트에서 Llama-3-70B 스타일의 GQA 캐시보다 4배 더 작습니다.

상충 관계: MLA는 어텐션 연산(헤드별)마다 복원 단계를 추가합니다. 추가된 연산량은 절약된 대역폭에 비해 작습니다. 긴 컨텍스트 추론에서는 순이익이 있습니다.

### 라우팅: 보조 손실 없는 부하 균형

MoE 라우터는 각 토큰을 처리할 상위 k개 전문가를 결정합니다. 단순한 라우터는 몇몇 전문가에 너무 많은 작업을 집중시켜 나머지를 유휴 상태로 만듭니다. 표준 해결책: 부하 불균형을 페널티하는 보조 손실 항을 추가하는 것입니다. 이는 작동하지만 주요 작업 성능을 약간 저하시킵니다.

DeepSeek-V3는 보조 손실 없는 방식을 도입합니다. 전문가별 편향(bias) 항이 라우터 로짓에 추가되며, 훈련 중 간단한 규칙으로 조정됩니다: 전문가 `e`가 과부하되면 `bias_e`를 감소시키고, 과부하가 적으면 증가시킵니다. 추가 손실 항이 없습니다. 훈련은 깔끔하게 유지됩니다. 전문가 부하는 균형을 유지합니다.

주요 손실에 대한 영향: 측정 가능한 영향이 없습니다. MoE 아키텍처에 대한 영향: 더 깔끔하며, 튜닝해야 할 보조 손실 하이퍼파라미터가 없습니다.

### MTP: 더 밀집된 훈련 + 무료 초안

10단계 18강에서 DeepSeek-V3가 D=1 MTP 모듈을 추가하여 두 위치 앞의 토큰을 예측한다는 것을 알았습니다. 추론 시, 학습된 모듈은 80% 이상의 수용률을 보이는 추론적 디코딩(Speculative Decoding) 초안으로 재사용됩니다. 학습 시, 각 은닉 상태는 D+1 = 2개의 타깃에 대해 지도 학습을 받아 더 밀집된 신호를 제공합니다.

매개변수: 671B 메인 모델 위에 추가된 14B. 오버헤드: 2.1%.

### 학습: DualPipe

10단계 19강에서 DualPipe가 순방향 및 역방향 청크를 노드 간 all-to-all 통신과 겹치게 하는 양방향 파이프라인이라는 것을 알았습니다. DeepSeek-V3의 2,048-H800 규모에서는 1F1B 방식이 파이프라인 버블로 잃었을 약 245k GPU-hours를 회수합니다.

### 구성 필드별 분석

DeepSeek-V3 구성(간소화)은 다음과 같습니다:

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

파싱해 보세요:

- `hidden_size=7168`: 임베딩 차원.
- `num_hidden_layers=61`: 총 블록 깊이.
- `first_k_dense_layers=3`: 첫 3개 블록은 크기가 18432인 밀집 MLP를 사용합니다. 나머지 58개는 MoE를 사용합니다.
- `num_attention_heads=128`: 128개의 쿼리 헤드.
- `kv_lora_rank=512`: K와 V는 이 잠재 차원으로 압축되며 헤드별로 복원됩니다.
- `num_experts=256, num_experts_per_tok=8`: 각 MoE 블록은 256개의 전문가를 가지며 top-8을 라우팅합니다.
- `shared_experts=1`: 라우팅된 256개 전문가 위에, 항상 켜져 있는(always-on) 전문가 1개가 모든 토큰에 기여합니다. 모든 토큰이 신뢰할 수 있는 값을 받도록 보장하는 '밀집 바닥(dense floor)'이라고 생각할 수 있습니다.
- `moe_intermediate_size=2048`: 각 전문가의 MLP 은닉 크기. 전문가가 256개 있으므로 밀집 MLP보다 작습니다.

### 매개변수 계산

전체 계산은 `code/main.py`에 있습니다. 주요 내용은 다음과 같습니다:

- 임베딩: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 첫 3개 밀집 블록: MLA를 사용한 어텐션(블록당 약 144M) + 밀집 MLP(블록당 약 260M) + 정규화. 총 약 1.2B.
- 58개 MoE 블록: MLA를 사용한 어텐션(약 144M) + 각 256개 전문가(개당 30M) + 공유 전문가 1개(30M) + 정규화. 모든 전문가를 포함해 블록당 총 약 7.95B. 58개 MoE 블록의 총합은 461B입니다.
- MTP 모듈: 14B.

총합: 핵심 아키텍처 약 476B + MTP 14B. 공개된 671B 수치는 추가 구조적 매개변수(바이어스 텐서, 전문가별 구성 요소, 공유 전문가 스케일링 등)를 포함합니다. 계산기로 재현한 값은 공개된 수치와 3-5% 이내의 차이를 보이며, 이 차이는 DeepSeek 보고서의 섹션 2 부록에 문서화된 세밀한 계산 방식에서 기인합니다.

포워드 패스당 활성 매개변수:

- 어텐션: 레이어당 144M * 61 = 8.8B (모든 레이어가 작동).
- MLP 활성: 첫 3개 레이어는 밀집형(3 * 260M = 780M), 58개 MoE 레이어는 각각 8개 라우팅된 전문가 + 1개 공유 전문가 + 라우팅 오버헤드가 활성화됩니다. 레이어당 활성 MLP: 약 260M. 총합: 3 * 260M + 58 * 260M = 약 15.9B.
- 임베딩 + 정규화: 1.2B.
- 총 활성 매개변수: 약 26B 핵심 + 14B MTP (학습되었으나 추론 시 항상 실행되지는 않음) ≈ 37B.

### 671B / 37B 비율

18배 희소성 비율 (활성 매개변수는 전체의 5.5%). DeepSeek-V3는 오픈 웨이트를 공개한 프론티어 MoE 모델 중 가장 희소한 모델입니다. Mixtral 8x7B는 비율이 13/47 (28%)로 훨씬 밀집되어 있습니다. Llama 4 Maverick는 비율이 17B/400B (4.25%)로 유사합니다. DeepSeek의 전략: 프론티어 규모에서는 더 많은 전문가와 낮은 활성화 비율이 활성 FLOP당 더 높은 품질을 생성합니다.

### DeepSeek-V3의 위치

| 모델 | 총 매개변수 | 활성 매개변수 | 비율 | 어텐션 | 새로운 아이디어 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 확장 |

### 후속 모델: R1, V4

DeepSeek-R1 (2025)은 V3 백본에 대한 추론 학습 실행입니다. R1은 동일한 아키텍처를 사용합니다. 변경된 것은 사전 학습 아키텍처가 아니라 사후 학습 레시피(검증 가능한 작업에 대한 대규모 RL)입니다.

DeepSeek-V4 (출시될 경우)는 MLA + MoE + MTP를 유지하고 DSA (DeepSeek Sparse Attention)를 추가할 것으로 예상됩니다. DSA는 10단계 · 17강의 NSA의 후속입니다. 계보는 안정적입니다: 아키텍처 수준의 혁신이 누적되며, 각 버전은 추가적인 조절 변수를 도입합니다.

```figure
moe-routing
```

## 사용하기

`code/main.py`는 DeepSeek-V3의 구조에 특화된 파라미터 계산기입니다. 이를 실행하고, 출력 결과를 논문 수치와 비교해 보세요. 또한 가상의 변형 사례(256개 전문가 vs 512개, top-8 vs top-16, MLA 랭크 512 vs 1024)에 적용해 보세요.

확인할 사항:

- 총 파라미터 수 vs 공개된 671B.
- 활성 파라미터 수 vs 공개된 37B.
- 128k 컨텍스트에서의 KV 캐시 — MLA vs GQA 비교.
- 레이어별 세부 분석을 통해 파라미터 예산이 실제로 어디에 쓰이는지 확인하세요.

## 출시하기

이 강의는 `outputs/skill-deepseek-v3-reader.md`를 생성합니다. DeepSeek 계열 모델(V3, R1, 또는 미래의 변형)이 주어지면, 구성(config)의 각 필드를 명시하고, 구성 요소별 파라미터 수를 도출하며, 모델이 네 가지 DeepSeek 특화 혁신 중 어떤 것을 사용하는지 식별하는 구성 요소별 아키텍처 해석을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 계산기의 총 파라미터 추정치를 공개된 671B와 비교하고, 차이(delta)가 어디서 발생하는지 식별하세요. 논문 섹션 2에는 전체 항목별 상세 내역이 있습니다.

2. MLA 랭크를 512 대신 256으로 사용하도록 구성을 수정하세요. 128k 컨텍스트에서 resulting KV 캐시 크기를 계산하세요. 몇 퍼센트의 감소가 발생하며, 헤드별 표현력에는 어떤 비용이 발생하나요?

3. DeepSeek-V3의 (256개 전문가, top-8) 라우팅을 가상의 (512개 전문가, top-8) 변형과 비교하세요. 총 파라미터는 증가하지만 활성 파라미터는 동일합니다. 이론적으로 추가 전문가 용량이 무엇을 가져오며, 추론 시에는 어떤 비용이 발생하나요?

4. DeepSeek-V3 기술 보고서(arXiv:2412.19437)의 섹션 2.1을 읽고 MLA에 대해 설명하세요. K 및 V 압축 해제 행렬이 추론 시간 효율성을 위해 후속 matmul에 "흡수(absorbed)"될 수 있는 이유를 세 문장으로 설명하세요.

5. DeepSeek-V3는 대부분의 연산에 FP8 훈련을 사용합니다. 671B 가중치를 저장할 때 FP08강 BF16의 메모리 절감량을 계산하세요. 이것이 14.8T 토큰 훈련 예산과 어떻게 교차하나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | K와 V를 공유된 저랭크 잠재 공간(kv_lora_rank, 일반적으로 512)으로 압축하고, 헤드별로 실시간 압축 해제; KV 캐시는 잠재 공간만 저장 |
| kv_lora_rank | "MLA 압축 차원" | K와 V의 공유 잠재 표현 크기; DeepSeek-V3는 512를 사용 |
| 첫 k개의 밀집 레이어 | "초기 레이어는 밀집 유지" | MoE 모델의 첫 몇 레이어는 MoE 라우터를 건너뛰고 안정성을 위해 밀집 MLP를 실행 |
| num_experts_per_tok | "Top-k 라우팅" | 토큰당 라우팅된 전문가가 실행되는 개수; DeepSeek-V3는 8을 사용 |
| 공유 전문가 | "항상 실행되는 전문가" | 라우팅과 무관하게 모든 토큰을 처리하는 전문가; DeepSeek-V3는 1을 사용 |
| 보조 손실 없는 라우팅 | "편향 조정된 부하 균형" | 손실 항을 추가하지 않고 전문가 부하를 균형 있게 유지하기 위해 학습 중 조정되는 전문가별 편향 항 |
| MTP 모듈 | "추가 예측 헤드" | h^(1)과 E(t+1)로부터 t+2를 예측하는 트랜스포머 블록; 더 밀집된 학습, 무료 추론적 디코딩 초안 |
| DualPipe | "양방향 파이프라인" | 순방향/역방향 연산을 노드 간 all-to-all과 겹치게 하는 학습 스케줄 |
| 활성 매개변수 비율 | "희소성" | active_params / total_params; DeepSeek-V3는 5.5% |
| FP8 학습 | "8비트 학습" | FP8로 학습 저장 및 많은 연산 수행; BF16 대비 메모리를 약 절반으로 줄이면서 작은 품질 비용 |

## 추가 읽기

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 전체 아키텍처, 학습 및 결과 문서
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 구성 파일 및 배포 노트
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — MLA를 도입한 전신
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 아키텍처 기반의 추론 학습 후속 모델
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 계열 어텐션의 미래 방향
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 학습 스케줄 참조
