# 캡스톤 07 — 엔드투엔드 미세 조정 파이프라인 (데이터부터 SFT, DPO, 서빙까지)

> 자체 데이터로 학습한 8B 모델, 자체 선호도로 DPO 정렬, 양자화, 추론적 디코딩(Speculative Decoding) 적용, 측정 가능한 $/1M 토큰 비용으로 서빙. 2026년 오픈 스택은 Axolotl v0.8, TRL 0.15, 반복 작업용 Unsloth, 양자화용 GPTQ/AWQ/GGUF, 서빙용 vLLM 0.7 및 EAGLE-3입니다. 캡스톤의 목표는 전체 파이프라인을 재현 가능하게 실행하는 것입니다 — YAML 입력, 서빙 엔드포인트 출력 — 그리고 2026년 모델 개방성 프레임워크(Model Openness Framework)에 따라 모델 카드(Model Card)를 게시하는 것입니다.

**유형:** Capstone
**언어:** Python (파이프라인), YAML (설정), Bash (스크립트)
**선수 요건:** 2단계 (ML), 3단계 (DL), 7단계 (트랜스포머), 10단계 (LLM (대규모 언어 모델) 기초부터), 11단계 (LLM 엔지니어링), 17단계 (인프라), 18단계 (안전)

**활용 단계:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**시간:** 35시간

## 문제점

2026년 모든 진지한 AI 팀은 미세 조정 파이프라인을 항상 준비해 두고 있습니다. 프론티어 기반 모델을 출시하기 때문이 아니라, 도메인 SFT (지도 미세 조정)(Supervised Fine-Tuning), 라벨이 붙은 선호도에 대한 DPO (직접 선호 최적화)(Direct Preference Optimization), 추론적 디코딩(Speculative Decoding)용 증류된 초안, EAGLE-3를 활용한 서빙 등 다운스트림 적응이 측정 가능한 성과가 있는 곳이기 때문입니다. Axolotl v0.8은 다중 GPU SFT 설정을 처리합니다. TRL 0.15는 DPO와 GRPO를 처리합니다. Unsloth는 빠른 단일 GPU 반복 작업을 가능하게 합니다. vLLM 0.07강 EAGLE-3는 품질 손실 없이 디코딩 처리량을 2-3배 높입니다. 도구는 잘 작동합니다; 핵심은 YAML, 데이터 위생, 평가 규율에 있습니다.

8B 기반 모델(Llama 3.3, Qwen3, Gemma 3)을 태스크별 데이터로 SFT한 후 DPO를 수행하고, 서빙을 위해 양자화하며, lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 사용하여 성능 향상을 측정합니다. 2026년 모델 개방성 프레임워크에 따라 모델 카드(Model Card)를 작성합니다. 핵심은 재현성입니다 — 한 명령으로 전체 파이프라인을 처음부터 끝까지 다시 실행할 수 있습니다.

## 개념

파이프라인은 5단계로 구성됩니다. **데이터**: 중복 제거(MinHash / Datatrove), 품질 필터(Nemotron-CC 스타일 분류기), PII 제거, 공개 벤치마크 오염에 대한 분할 위생 검사. **SFT**: Axolotl YAML, 8xH100에서 ZeRO-3, 코사인 스케줄, 패킹된 시퀀스, 2-3 에포크. **DPO 또는 GRPO**: TRL 구성, 1 에포크, 선호 쌍은 인간이 라벨링하거나 모델이 판단하며, beta 튜닝. **양자화**: 배포 유연성을 위해 GPTQ + AWQ + GGUF. **서빙**: EAGLE-3 추론적 헤드를 사용한 vLLM 0.7 (또는 SpecForge를 사용한 SGLang), K8s 배포, 큐 대기 기반 HPA.

Ablation은 산출물입니다: 세 가지 작업 특화 벤치마크에서 SFT 전용 vs SFT+DPO vs SFT+GRPO. 서빙 지표: 배치 1 / 8 / 32에서의 토큰/초, EAGLE-3 수용률, $/1M 토큰. 안전성 평가: Llama Guard 4 통과율. 모델 카드: 편향 평가, 재현성 시드, 데이터 라이선스.

## 아키텍처

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 스택

- 데이터: 중복 제거용 Datatrove, 품질용 Nemotron-CC 분류기, PII용 Presidio
- 베이스: Llama 3.3 8B, Qwen3 14B 또는 Gemma 3 12B
- SFT: ZeRO-3, Flash Attention 3, 패킹된 시퀀스를 사용하는 Axolotl v0.8
- 선호 튜닝: DPO 또는 GRPO용 TRL 0.15; 단일 GPU 반복용 Unsloth
- 양자화: GPTQ (Marlin), AWQ, llama.cpp를 통한 GGUF
- 서빙: EAGLE-3 추론적 디코딩을 사용하는 vLLM 0.7 (또는 SGLang 0.4 + SpecForge)
- 평가: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- 안전성 평가: Llama Guard 4, ShieldGemma-2
- 인프라: Kubernetes + NVIDIA 디바이스 플러그인, 큐 대기 지표 기반 HPA
- 관측 가능성: 학습용 W&B, 추론용 Langfuse

```figure
ce-finetune-stages
```

## 구현하기

1. **데이터 파이프라인.** 원시 코퍼스에 Datatrove 중복 제거를 실행하세요. Nemotron-CC 스타일 품질 분류기를 적용하세요. Presidio가 PII를 제거합니다. 명시적인 시드로 train/val 분할을 작성하세요.

2. **오염 검사.** 모든 검증 분할에 대해 MMLU-Pro, MT-Bench-v2, RewardBench-2 테스트 세트와 MinHash를 계산하세요. 겹치는 부분을 거부하세요.

3. **Axolotl SFT.** ZeRO-3, FA3, 시퀀스 패킹을 포함한 YAML. 8xH100에서 2-3 에포크. W&B에 기록하세요.

4. **TRL DPO / GRPO.** SFT 체크포인트를 가져와 선호 쌍에 대해 DPO를 한 에포크 실행하거나 (수학/코드에 검증 가능한 보상을 사용하는 GRPO를 실행) beta를 스윕해 보세요.

5. **양자화(Quantization).** 세 가지 양자화 버전 생성: GPTQ-INT4-Marlin, AWQ-INT4, llama.cpp용 GGUF-Q4_K_M. 크기와 명목 처리량을 기록하세요.

6. **추론적 디코딩(Speculative Decoding)으로 서빙하세요.** Red Hat Speculators로 훈련된 EAGLE-3 드래프트 헤드를 사용하는 vLLM 0.7 구성. 배치 1 / 8 / 32에서 수용률과 꼬리 지연(Tail Latency)을 측정하세요. 동일한 평가에서 Anthropic / OpenAI 대비 $/1M 토큰을 보고하세요.

7. **평가 매트릭스.** base, SFT 전용, SFT+DPO, SFT+GRPO에 대해 lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 실행하세요. 표를 생성하세요.

8. **안전 평가.** dev 세트에 대한 Llama Guard 4 통과율. ShieldGemma-2 출력 필터.

9. **모델 카드(Model Card).** MOF 2026 템플릿: 데이터, 훈련, 평가, 안전, 라이선스, YAML 및 커밋 SHA를 포함한 재현성 섹션.

## 사용하기

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 출시하기

`outputs/skill-finetuning-pipeline.md`는 산출물을 설명합니다. 단일 명령이 데이터를 SFT, DPO, 양자화, 서빙, 평가를 거쳐 모델 카드와 서빙된 엔드포인트를 생성합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | base 대비 평가 델타 | 목표 작업(MMLU-Pro, MT-Bench-v2, 작업별)에서 측정된 향상 |
| 20 | 파이프라인 재현성 | 동일한 시드로 끝에서 끝까지 한 명령으로 재실행 |
| 20 | 데이터 위생 | 중복 제거율, PII 제거 커버리지, 오염 검사 통과 |
| 20 | 서빙 효율 | bs=1/8/32에서의 토큰/s, EAGLE-3 수용률, $/1M 토큰 |
| 15 | 모델 카드 + 안전 평가 | 2026 MOF 완성도 + Llama Guard 4 통과율 |
| **100** | | |

## 연습 문제

1. 동일한 작업별 벤치마크에서 SFT 전용 vs SFT+DPO vs SFT+GRPO를 실행하세요. 어떤 선호 방법이 이기는지, 그리고 얼마나 이기는지 보고하세요.

2. Llama 3.3 8B를 Qwen3 14B로 교체하세요. 동일한 품질에서 $/1M 토큰을 측정하세요.

3. 도메인 데이터 vs 범용 ShareGPT에서 EAGLE-3 수용률을 측정하세요. 델타와 이것이 지연 예산에 미치는 영향을 보고하세요.

4. 오염(Contamination) 1%를 주입(MMLU-Pro 답변을 학습 데이터에 유출)하고 평가를 다시 실행해 보세요. MMLU-Pro 정확도가 비현실적으로 급증하는 것을 관찰하세요. 이를 포착하는 오염 검사 CI 게이트를 구축하세요.

5. 전체 미세 조정(Fine-tune)의 대안으로 LoRA SFT를 추가하세요. 메모리 사용량이 10배 낮은 상태에서 품질 격차를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Axolotl | "SFT 트레이너" | SFT, DPO, 증류(Distillation)를 위한 통합 YAML 기반 트레이너 |
| TRL | "선호 튜너" | LLM에 대한 DPO, GRPO, PPO를 위한 Hugging Face 라이브러리 |
| GRPO | "그룹 상대적 정책 최적화" | 검증 가능한 보상(Verifiable Rewards)을 사용하는 DeepSeek R1의 RL 레시피 |
| EAGLE-3 | "추론적 디코딩(Speculative Decoding) 초안" | N개 토큰을 미리 예측하는 초안 헤드(Draft Heads); vLLM이 타겟 모델로 검증 |
| MOF | "모델 개방성 프레임워크(Model Openness Framework)" | 데이터, 코드, 라이선스에 대해 모델 릴리스를 평가하는 2026 표준 |
| 오염 검사 | "분할 위생(Split Hygiene)" | 학습 데이터에 테스트 세트가 유출된 것을 감지하는 MinHash 기반 검출 |
| 수용률 | "EAGLE / MTP 지표" | 타겟 모델이 초안 토큰을 수용하는 비율 |

## 추가 읽기

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 참조 SFT / DPO 트레이너
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 및 GRPO 참조 구현
- [Unsloth](https://github.com/unslothai/unsloth) — 단일 GPU 반복 참조
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 방법론
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 참조 서빙 스택
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 대안 추론적 디코딩(Speculative Decoding) 트레이너
- [Model Openness Framework 2026](https://isocpp.org/) — 개방 릴리스 평가 표준
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 표준 평가 실행기
