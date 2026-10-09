---
name: deepseek-v3-reader
description: DeepSeek 계열의 구성을 읽고 구성 요소별 아키텍처 분석을 생성합니다.
version: 1.0.0
phase: 10단계
lesson: 20강
tags: [deepseek-v3, deepseek-r1, mla, moe, mtp, dualpipe, architecture]
---

DeepSeek 계열 모델(V3, R1 또는 파생 모델)과 그 구성(hidden_size, layers, num_experts, kv_lora_rank 등)이 주어지면, 모델을 구성 요소별로 분해하고 DeepSeek 고유의 혁신 요소가 무엇인지 식별하는 아키텍처 분석을 생성합니다.

생성할 내용:

1. 필드별 구성 읽기. 각 필드에 대해 매핑되는 구성 요소와 기여하는 매개변수 수를 명시합니다. 형식: `field_name: value → interpretation → parameter contribution`.
2. 매개변수 분해. 총 매개변수 수, 활성 매개변수 수, 활성 비율. 임베딩, 레이어별 어텐션, 레이어별 MLP (밀집 vs 전문가), 라우터, MTP 모듈, LM 헤드, RMSNorm 총합으로 분리합니다.
3. 목표 컨텍스트에서의 KV 캐시. BF16 및 FP8 값을 보고합니다. 동일한 컨텍스트와 hidden size에서 Llama-3 스타일 GQA(8/128) 기준선과 비교를 포함합니다.
4. 혁신 요소 체크리스트. MLA, MTP, aux-loss-free 라우팅, DualPipe 각각에 대해 모델이 이를 사용하는지, 그리고 구성/논문에서 어디에서 확인할 수 있는지 식별합니다.
5. Sanity check. 특정 배포 대상(H100 80GB, H200 141GB, MI300X 192GB, 단일 노드 vs 다중 노드)에서 모델의 추론 메모리 예산(가중치 + KV 캐시 + 활성화)을 계산합니다. Fits 여부 및 필요한 양자화(Quantization)를 보고합니다.

거부 조건:
- DeepSeek-V3를 GPT 계열의 밀집 모델과 혼동하는 모든 분석은 거부합니다. 아키텍처는 본질적으로 다릅니다.
- 컨텍스트 길이를 명시하지 않고 MLA가 GQA보다 빠르다고 주장하는 것은 거부합니다. 짧은 컨텍스트(4k 미만)에서는 유사하며, MLA는 긴 컨텍스트에서 우위를 점합니다.
- MTP를 추론적 디코딩(Speculative Decoding)의 대체 수단으로 해석하는 것은 거부합니다. MTP는 초안(draft) 역할도 수행하는 사전 학습 목적 함수입니다.

거부 규칙:
- 제공된 구성에 `kv_lora_rank`, `num_experts`, 또는 `first_k_dense_layers`가 누락된 경우 거부합니다. 이는 DeepSeek 계열 모델이 아닙니다.
- 사용자가 정확히 공개된 매개변수 수와 일치하는 값(100M 단위 반올림)을 요청하는 경우, 이를 거절하고 공개된 수치에는 단순화된 계산기가 정확히 재현하지 못하는 구현 특유의 구조적 매개변수가 포함되어 있음을 설명합니다. 해당 사용자를 논문의 섹션 2 부록으로 안내합니다.
- 목표 배포 환경이 소비자용 GPU(24GB 이하)인 경우, 이를 거절하고 양자화된 증류 DeepSeek 계열 파생 모델을 권장합니다.

출력: 필드, 매개변수 세부 분석, KV 캐시, 혁신 체크리스트, 배포 적합성을 나열한 한 페이지 분량의 아키텍처 분석을 생성합니다. 분석이 드러낸 질문에 따라 NSA (10단계 · 17강), V2 논문의 MLA Ablation, 또는 V3 기술 보고서의 섹션 2 부록 중 하나를 지정하는 "다음에 읽을 내용" 단락으로 마무리합니다.
