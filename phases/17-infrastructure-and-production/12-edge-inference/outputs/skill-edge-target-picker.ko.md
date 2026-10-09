---
name: edge-target-picker
description: 디바이스, 모델, 지연 시간 예산을 고려하여 엣지 추론 대상(Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson)과 이에 상응하는 양자화 형식을 선택합니다.
version: 1.0.0
phase: 17단계
lesson: 12강
tags: [edge, ane, hexagon, webgpu, webllm, jetson, core-ml, qnn, nvfp4]
---

배포 플랫폼(iOS, Android, 브라우저, 로보틱스/자동차/엣지 서버), 모델, 지연 시간/메모리 예산을 고려하여 엣지 대상 추천을 생성해 보세요.

다음 내용을 생성합니다:

1. 대상. 특정 NPU/GPU(ANE, Hexagon, WebGPU, Jetson Orin Nano / AGX / Thor)를 지정합니다. 플랫폼과 2026년 런타임 커버리지로 정당화합니다.
2. 대역폭 상한. 이론적 디코딩 상한을 계산합니다: bandwidth_GB_s / model_size_GB. 사용자의 tok/s 요구 사항과 비교합니다. 상한이 요구 사항보다 낮다면, 거부하거나 더 작은 모델 / 더 엄격한 양자화를 제안합니다.
3. 양자화 형식. Q4 GGUF(브라우저/엣지 CPU), Core ML INT4 + FP16(ANE), QNN INT8/INT4(Hexagon), 또는 NVFP4 + FP8 KV(Jetson Thor / Edge-LLM) 중 하나를 선택합니다.
4. 변환 파이프라인. 정확한 변환기(Core ML converter, Qualcomm AI Hub, WebLLM용 MLC-LLM, TensorRT-LLM Edge compiler)를 지정합니다.
5. 컨텍스트 예산. 디바이스 RAM에서 가중치와 함께 수용 가능한 최대 컨텍스트를 명시합니다. 긴 컨텍스트 사용 사례의 경우, KV 양자화(Q4 KV)를 지정하거나 거부합니다.
6. 폴백. 디바이스가 기능을 수행할 수 없거나 WebGPU가 사용 불가능한 경우(Firefox Android, 구형 브라우저), 동일한 OpenAI 호환 인터페이스를 사용하는 서버 측 API 폴백을 지정합니다.

거부 조건:
- 대역폭 상한을 초과하는 tok/s를 약속하는 경우. 거부합니다 — 물리 법칙입니다.
- 2026년에 Core ML이 아닌 런타임을 통해 ANE를 직접 타겟팅하는 경우. Core ML만 ANE를 네이티브로 노출합니다.
- 모든 브라우저에 WebGPU가 있다고 가정하는 경우. 2026년 모바일 커버리지는 약 70-75%입니다. 항상 폴백을 명시합니다.

거부 규칙:
- 모델이 6 GB 이상이고 대상이 폰(4-8 GB RAM)인 경우, 거부합니다 — 먼저 더 작은 모델이나 공격적인 양자화를 제안합니다.
- iPhone에서 7B 모델에 대해 128K 컨텍스트를 요청하는 경우, 거부합니다 — Q4 KV 및 슬라이딩 윈도우 어텐션 없이는 디바이스 RAM이 이를 수용할 수 없습니다.
- Android에서 WebGPU를 통해 긴 컨텍스트 스트리밍 배포가 필요하고 사용자가 Firefox 지원을 요구하는 경우, 이를 거부하고 Chrome 또는 서버 폴백을 요구해야 합니다.

출력: 대상, 상한, 양자화(Quantization), 변환기, 컨텍스트 예산, 폴백을 명시한 한 페이지 분량의 계획. 마지막에는 목표 기기 군집에서 가장 성능이 낮은 기기에서 관측된 초당 토큰 수(TPS)라는 단일 지표만 포함합니다.
