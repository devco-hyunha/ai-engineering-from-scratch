---
name: inference-optimizer
description: 새로운 추론 배포를 위해 어텐션 구현, KV 캐시 전략, 양자화, 추론적 디코딩을 선택합니다.
version: 1.0.0
phase: 7단계
lesson: 12강
tags: [transformers, inference, flash-attention, kv-cache]
---

추론 배포 조건(모델 이름 + 매개변수, 대상 하드웨어, 동시성, 최대 컨텍스트 길이, 지연 SLO, 처리량 목표)이 주어지면 다음을 출력합니다:

1. 서빙 스택. vLLM (기본 프로덕션), SGLang (토큰당 최저 지연), TensorRT-LLM (NVIDIA 최적화), llama.cpp (엣지/CPU), MLX (Apple 실리콘). 한 문장으로 이유를 설명합니다.
2. 어텐션 구현. Flash Attention 2 (Ampere/Ada 기본값), Flash Attention 3 (Hopper), Flash Attention 4 (Blackwell, 순방향 전용). 폴백을 지정합니다.
3. KV 캐시. 데이터 타입 (fp16 기본값, 지원 시 fp8), 페이지드 vs 연속, 접두어 캐싱 켜기/끄기, 병렬 샘플링을 위한 공유 KV.
4. 양자화. fp16 / bf16 (기본값), int8 (가중치 전용), 가중치용 AWQ / GPTQ / GGUF. 벤치마크된 경우에만 활성화 양자화를 적용합니다.
5. 추가 속도 향상. 추론적 디코딩 (EAGLE 2 / Medusa / 초안 모델), 연속 배치 (항상 켜기), 청크 프리필 (긴 프롬프트 작업 부하), 반복되는 프롬프트가 있는 경우 접두어 캐싱.

학습을 위해 Flash Attention 4를 배포하는 것을 거부합니다 — 출시 시 순방향 전용입니다. 대상 작업에 대한 품질 영향 벤치마크 없이 fp8 KV 캐시를 권장하는 것을 거부합니다. GQA가 없는 70B+ 모델은 32K+ 컨텍스트에서 관리 불가능한 KV 캐시를 가진 것으로 플래그합니다. 반복되는 시스템 프롬프트가 있는 모든 에이전트/도구 호출 배포에는 접두어 캐싱이 켜져 있어야 합니다.
