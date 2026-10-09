---
name: finetuning-pipeline
description: Ablation, quantization, and a 2026 Model Openness Framework model card included, reproducible data-to-SFT-to-DPO-to-serve fine-tuning pipeline를 실행합니다.
version: 1.0.0
phase: 19단계
lesson: 07강
tags: [capstone, fine-tuning, axolotl, trl, dpo, grpo, vllm, eagle-3, mof]
---

기본 모델(Llama 3.3 8B, Qwen3 14B, Gemma 3 12B)과 작업 전용 데이터셋이 주어졌을 때, 서빙 엔드포인트와 재현 가능한 모델 카드를 생성하는 단일 명령 파이프라인을 구축해 보세요.

구축 계획:

1. 데이터 단계: Datatrove 중복 제거, Nemotron-CC 스타일 품질 필터, Presidio PII 제거, 시드 지정된 train/val 분할.
2. 오염 검사: MMLU-Pro, MT-Bench-v2, RewardBench-2에 대해 MinHashLSH 검사. 중복이 발견되면 거부합니다.
3. 지도 미세 조정 (SFT)(SFT): Axolotl v0.8, ZeRO-3, Flash Attention 3, 패킹된 시퀀스 사용, 8xH100에서 2-3 에포크(Epoch).
4. 선호 튜닝: TRL 0.15 DPO (직접 선호 최적화)(DPO) (또는 검증 가능한 보상을 사용하는 GRPO)를 1 에포크(Epoch) 동안 실행, beta 스윕.
5. 양자화(Quantization): GPTQ-INT4-Marlin + AWQ-INT4 + GGUF-Q4_K_M.
6. 서빙(Serving): vLLM 0.7, EAGLE-3 추론적 디코딩(Speculative Decoding) 사용 (Red Hat Speculators 또는 SGLang SpecForge를 통해 draft heads 생성). queue-wait에 대한 HPA가 있는 K8s 배포.
7. 평가(Evaluation (Eval)): lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 base/SFT-only/SFT+DPO/SFT+GRPO에 대해 평가합니다.
8. 안전성: Llama Guard 4 통과율, ShieldGemma-2 출력 필터.
9. 2026 Model Openness Framework에 따른 모델 카드로, 데이터, 훈련, 평가, 안전성, 재현성 섹션을 포함합니다.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 기본 모델 대비 평가 델타 | MMLU-Pro, MT-Bench-v2, 작업 전용 벤치마크에서의 측정된 향상 |
| 20 | 파이프라인 재현성 | 동일한 시드로 단일 명령 재실행 시 일치하는 해시 생성 |
| 20 | 데이터 위생 | 중복 제거율, PII 제거 커버리지, 오염 검사 통과 |
| 20 | 서빙 효율 | batch 1/8/32에서의 초당 토큰 수 (TPS)(Tokens per Second (TPS)), EAGLE-3 수용률, $/1M 토큰 |
| 15 | 모델 카드 + 안전성 평가 | 2026 MOF 완성도 + Llama Guard 4 통과율 |

하드 거부:

- MinHash 오염 검사(MinHash contamination check)를 건너뛰는 파이프라인. MMLU-Pro를 학습 데이터에 유출하는 것은 전형적인 평가 속임수 실패 모드입니다.
- 시드(seed)나 YAML 파일이 첨부되지 않은 학습 실행. 재현 가능성은 필수 요건입니다.
- EAGLE-3 또는 동등한 추론적 디코딩(Speculative Decoding) 구성 없이 서빙하는 경우. 기준 토큰/s는 2026년 기준이 아닙니다.
- 안전성 평가가 누락된 경우. 모든 미세 조정(Fine-tuning)은 Llama Guard 4 통과율과 함께 출시해야 합니다.

거부 규칙:

- lm-eval-harness 커밋 SHA를 첨부하지 않고 벤치마크 점수를 주장하는 모델 카드(Model Card)를 게시하는 것을 거부합니다.
- 파생 모델을 금지하는 라이선스를 가진 데이터로 미세 조정(Fine-tuning)하는 것을 거부합니다. MOF는 데이터 라이선스를 등급화합니다.
- 평가 매트릭스(eval matrix)에서 품질 손실을 측정하지 않고 양자화(Quantization)된 모델을 출시하는 것을 거부합니다.

출력: 파이프라인 오케스트레이터(orchestrator), Llama 3.3 8B 및 하나의 대체 기반 모델에 대한 YAML 파일, SFT 및 DPO W&B 실행 로그, 양자화(Quantization)된 아티팩트, 서빙된 엔드포인트, 3개 벤치마크 평가 매트릭스, 안전성 평가, 2026년 MOF 모델 카드, 그리고 발견하고 수정한 세 가지 가장 큰 데이터 위생(data-hygiene) 문제에 대한 보고서가 포함된 저장소(repo)를 생성합니다.
