---
name: inference-optimizer
description: 새로운 추론 배포를 위해 어텐션 구현, KV 캐시 전략, 양자화 및 추측 디코딩(speculative decoding)을 선택합니다.
version: 1.0.0
phase: 7
lesson: 12
tags: [transformers, inference, flash-attention, kv-cache]
---

추론 배포 환경(모델 이름 및 파라미터 수, 대상 하드웨어, 동시성, 최대 컨텍스트 길이, 지연 시간 SLO, 처리량 목표)이 주어지면, 다음을 출력하세요:

1. **서빙 스택(Serving stack)**: vLLM(기본 프로덕션용), SGLang(토큰당 최저 지연 시간), TensorRT-LLM(NVIDIA 최적화), llama.cpp(에지/CPU), MLX(Apple 실리콘) 중 하나를 선택하고, 한 문장으로 그 이유를 설명하세요.
2. **어텐션 구현(Attention implementation)**: Flash Attention 2(Ampere/Ada 기본), Flash Attention 3(Hopper), Flash Attention 4(Blackwell, 순전파 전용) 중 선택하세요. 폴백(fallback) 방식을 반드시 명시하세요.
3. **KV 캐시(KV cache)**: 데이터 타입(`dtype`, 기본값 `fp16`, 지원 시 `fp8`), Paged vs Contiguous, Prefix caching On/Off 여부, 병렬 샘플링을 위한 Shared KV 사용 여부를 결정하세요.
4. **양자화(Quantization)**: `fp16` / `bf16`(기본값), `int8`(가중치 전용), 가중치용 AWQ / GPTQ / GGUF 중 선택하세요. 활성화(Activation) 양자화는 별도의 벤치마크를 수행한 경우에만 포함하세요.
5. **추가 가속(Extra speedups)**: 추측 디코딩(Speculative decoding: EAGLE 2 / Medusa / draft model), 연속 배칭(Continuous batching: 항상 활성화), 청크 프리필(Chunked prefill: 긴 프롬프트 워크로드), 반복되는 프롬프트가 있는 경우 Prefix caching을 고려하세요.

**제약 사항:**
- 출시 시점에 순전파 전용(forward-only)인 Flash Attention 4를 학습용으로 배포하는 것은 거부하세요.
- 대상 작업에 대한 품질 영향을 벤치마크하지 않고 `fp8` KV 캐시를 추천하는 것은 거부하세요.
- GQA(Grouped Query Attention)가 없는 70B 이상의 모델이 32K 이상의 컨텍스트를 가질 경우, KV 캐시 관리가 불가능하다고 표시하세요.
- 반복되는 시스템 프롬프트가 있는 에이전트 또는 도구 호출(tool-calling) 배포의 경우, 반드시 Prefix caching을 On으로 설정하도록 요구하세요.
