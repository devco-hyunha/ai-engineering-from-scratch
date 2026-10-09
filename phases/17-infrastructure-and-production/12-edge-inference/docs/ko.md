# 엣지 추론 — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 엣지의 핵심 제약은 연산이 아니라 메모리 대역폭입니다. 모바일 DRAM은 50-90 GB/s인 반면, 데이터센터 HBM3는 2-3 TB/s를 넘기며 30-50배의 차이가 있습니다. 디코딩은 메모리 바운드(memory-bound)이므로 이 격차는 결정적입니다. 2026년에는 상황이 네 가지로 나뉩니다. Apple M4/A18 Neural Engine은 통합 메모리(CPU↔NPU 복사 없음)로 최대 38 TOPS를 달성합니다. Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon은 45 TOPS를 기록합니다. WebGPU + WebLLM은 M3 Max에서 Llama 3.1 8B (Q4)를 약 41 tok/s로 실행합니다(네이티브의 약 70-80%); GitHub 스타 17.6k, OpenAI 호환 API, 모바일 커버리지 약 70-75%를 제공합니다. NVIDIA Jetson Orin Nano Super (8GB)는 Llama 3.2 3B / Phi-3를 수용하며, AGX Orin은 vLLM을 통해 gpt-oss-20b를 약 40 tok/s로 실행합니다. Jetson T4000 (JetPack 7.1)은 AGX Orin의 2배 성능입니다. TensorRT Edge-LLM은 EAGLE-3, NVFP4, 청크 프리필(chunked prefill)을 지원하며, CES 2026에서 Bosch, ThunderSoft, MediaTek가 시연했습니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 장난감 수준의 대역폭 바운드 디코딩 시뮬레이터)
**선수 요건:** 17단계 · 04강 (서빙 엔진 내부 구조), 17단계 · 09강 (프로덕션 양자화)
**시간:** 약 60분

## 학습 목표

- 모바일 LLM 추론이 메모리 대역폭 바운드이며 연산은 부차적인 이유를 설명해 보세요.
- 네 가지 엣지 타겟(Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson)을 나열하고 각각에 맞는 사용 사례를 매칭해 보세요.
- 2026년 WebGPU 커버리지 격차(Firefox Android가 따라잡는 중)와 Safari iOS 26의 도입을 언급해 보세요.
- 타겟별 양자화 형식을 선택해 보세요 (ANE용 Core ML INT4 + FP16, Hexagon용 QNN INT8/INT4, 브라우저용 WebGPU Q4, Jetson Thor용 NVFP4).

## 문제점

고객은 온디바이스(on-device) 챗봇을 원합니다: 음성 우선, 기본 프라이버시, 오프라인 작동. MacBook Pro M3 Max에서는 Llama 3.1 8B Q4가 약 55 tok/s로 실행되어 문제가 없습니다. iPhone 16 Pro에서는 같은 모델이 3 tok/s로 실행되어 문제가 있습니다. Snapdragon 8 Gen 3를 탑재한 중급 Android에서는 7 tok/s입니다. Chrome Android v121+의 WebGPU를 통해 브라우저에서 실행하면 장치에 따라 4-8 tok/s입니다.

처리량 변동은 포팅(porting) 문제가 아닙니다. 이는 대역폭 격차, 양자화 형식, 그리고 사용자 공간(user-space)에서 NPU에 접근할 수 있는지의 곱입니다. 2026년 엣지 추론은 네 가지 다른 문제이며 네 가지 다른 해결책이 필요합니다.

## 개념

### 대역폭이 실제 상한선입니다

디코딩은 모든 토큰에 대해 전체 가중치를 읽습니다. Q4 양자화된 7B 모델은 3.5 GB입니다. 50 GB/s로 3.5 GB를 읽는 데 70 ms가 걸리며, 이는 이론적 상한선인 약 14 tok/s에 해당합니다. 90 GB/s (고급 모바일 DRAM)에서는 상한선이 약 25 tok/s로 이동합니다. 이 수치 아래에서는 연산량이 아무리 많아도 도움이 되지 않습니다.

데이터센터의 HBM3는 3 TB/s로 동일한 3.5 GB를 1.2 ms에 처리하며, 상한선은 830 tok/s입니다. 동일한 모델, 동일한 가중치입니다. 메모리 하위 시스템이 다릅니다.

### Apple Neural Engine (M4 / A18)

- 최대 38 TOPS. 통합 메모리(CPU와 ANE가 동일한 풀을 공유)로 복사 오버헤드가 없습니다.
- Core ML + `.mlmodel` 컴파일된 모델을 통해, 또는 PyTorch를 통해 Metal Performance Shaders (MPS)를 사용하여 접근합니다.
- Llama.cpp Metal 백엔드는 MPS를 사용하며, ANE를 직접 사용하지 않습니다. 네이티브 ANE는 Core ML 변환이 필요합니다.
- 2026년 iOS 앱에 가장 실용적인 경로: INT4 가중치 + FP16 활성화를 사용하는 Core ML입니다.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 최대 45 TOPS. SoC 내에서 CPU 및 GPU와 통합되지만, 메모리 도메인은 분리되어 있습니다.
- QNN (Qualcomm Neural Network) SDK와 AI Hub는 PyTorch/ONNX로부터의 변환을 제공합니다.
- 채팅 템플릿, Llama 3.2, Phi-3는 모두 AI Hub에서 일급 아티팩트로 제공됩니다.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. 소프트웨어는 Apple/Qualcomm보다 뒤처져 있습니다. OpenVINO는 개선되고 있지만 니치입니다.
- Windows ARM 코파일럿 앱에 가장 적합합니다. AMD/Intel 데스크톱에서는 로컬 우선(local-first)으로 네이티브로 실행됩니다.

### WebGPU + WebLLM

- WebGPU 컴퓨트 셰이더를 통해 브라우저에서 모델을 실행합니다. 설치가 필요 없습니다.
- M3 Max에서 Llama 3.1 8B Q4는 약 41 tok/s로 실행되며, 동일한 백엔드를 통해 네이티브 성능의 약 70-80%를 달성합니다.
- WebLLM은 GitHub에서 17.6k 스타를 기록했습니다. OpenAI 호환 JS API를 제공하며, Apache 2.0 라이선스를 따릅니다.
- 2026년 커버리지: Chrome Android v121+, Safari iOS 26 GA, Firefox Android는 아직 따라잡는 중입니다. 전체 모바일 커버리지는 약 70-75%입니다.

### NVIDIA Jetson 시리즈

- Orin Nano Super (8GB): Llama 3.2 3B와 Phi-3를 좋은 tok/s로 수용합니다.
- AGX Orin: vLLM을 통해 gpt-oss-20b를 약 40 tok/s로 실행합니다.
- Thor / T4000 (JetPack 7.1): AGX Orin의 2배 성능을 제공하며, EAGLE-3 및 NVFP4를 지원합니다.
- TensorRT Edge-LLM (2026)은 EAGLE-3 추론적 디코딩(Speculative Decoding), NVFP4 가중치, 청킹 프리필(Chunked Prefill)을 지원하며, 데이터센터 최적화 기법을 엣지 환경으로 이식한 것입니다.

### 타겟별 양자화(Quantization) 선택

| 타겟 | 형식 | 비고 |
|--------|--------|-------|
| Apple ANE | INT4 가중치 + FP16 활성화 | Core ML 변환 경로 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 변환기 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | `mlc_llm convert_weight` + 컴파일된 `.wasm`를 사용하세요; GGUF는 지원되지 않습니다 |
| Jetson Orin Nano | Q4 GGUF 또는 TRT-LLM INT4 | 메모리 바운드 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 경로 |

### 엣지에서의 긴 컨텍스트 함정

Llama 3.1의 128K 컨텍스트는 데이터센터 기능입니다. 8 GB RAM을 가진 폰에서는 4 GB 모델 + 32K 토큰용 2 GB KV 캐시 + OS 오버헤드 = OOM(메모리 부족)이 발생합니다. 엣지 배포는 공격적인 KV 양자화(Q4 KV)를 수용하지 않는 한 컨텍스트를 4K-8K로 유지합니다.

### 음성은 킬러 애플리케이션입니다

음성 에이전트(Agent)는 지연 시간에 민감합니다(첫 토큰까지의 시간(TTFT) < 500 ms). 로컬 추론(Inference)은 네트워크 지연을 완전히 제거합니다. 음성 인식(Whisper Turbo 변형은 엣지에서 실행됨)과 결합하면 엣지 추론이 생산 품질의 음성 루프가 됩니다.

### 기억해야 할 수치

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: Llama 3.1 8B Q4에서 초당 토큰 수(TPS) ~41.
- AGX Orin: vLLM을 통해 gpt-oss-20b에서 초당 토큰 수(TPS) ~40.
- 데이터센터-엣지 대역폭 격차: 30-50배.
- WebGPU 모바일 커버리지: ~70-75% (Firefox Android는 뒤처짐).

```figure
edge-bandwidth-pipe
```

## 사용하기

`code/main.py`는 엣지 타겟 전반에 걸쳐 대역폭 바운드 연산으로 이론적 디코딩 처리량 상한을 계산합니다. 관측된 벤치마크와 비교하며, 연산이 아닌 대역폭이 병목인 부분을 강조합니다.

## 출시하기

이 강은 `outputs/skill-edge-target-picker.md`를 생성합니다. 플랫폼(iOS/Android/브라우저/Jetson), 모델, 지연 시간/메모리 예산이 주어지면 양자화 형식과 변환 파이프라인을 선택합니다.

## 연습 문제

1. `code/main.py`을 실행해 보세요. Snapdragon 8 Gen 3 (~77 GB/s 대역폭)에서 Q4 양자화된 7B 모델의 디코딩 상한을 계산해 보세요. 관측된 6-8 tok/s와 비교하여 런타임이 효율적인지 판단해 보세요.
2. Android에서 WebGPU는 Chrome v121+가 필요합니다. 구형 브라우저를 위한 폴백을 설계해 보세요 — 동일한 OpenAI 호환 API를 통해 서버 측에서 처리하는 방식입니다.
3. iOS 앱이 4K 컨텍스트 스트리밍을 필요로 합니다. iPhone 16에서 활성 메모리 4 GB 미만으로 유지할 수 있는 모델/포맷 조합은 무엇인가요?
4. Jetson AGX Orin은 gpt-oss-20b를 40 tok/s로 실행합니다. Jetson Nano는 3B 모델만 수용할 수 있습니다. 두 제품을 모두 타겟팅한다면 추론 스택을 어떻게 통합할까요?
5. "2026년 WebLLM은 프로덕션 준비가 되었는가?"에 대해 논증해 보세요. 커버리지, 성능, 그리고 Firefox Android의 공백을 인용하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ANE | "Apple Neural Engine" | M-series 및 A-series의 온디바이스 NPU; 통합 메모리 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; 접근을 위한 QNN SDK |
| WebGPU | "브라우저 GPU" | W3C 표준화 브라우저 GPU API; Chrome/Safari 2026 |
| WebLLM | "브라우저 LLM 런타임" | MLC-LLM 프로젝트; Apache 2.0; OpenAI 호환 JS |
| Jetson | "NVIDIA 엣지" | Orin Nano / AGX / Thor / T4000 계열 |
| TRT Edge-LLM | "엣지 TensorRT" | TensorRT-LLM의 2026년 엣지 포트; EAGLE-3 + NVFP4 |
| 통합 메모리 | "공유 풀" | CPU와 NPU가 동일한 RAM을 참조; 복사 오버헤드 없음 |
| 대역폭 제한 | "메모리 제한" | 가중치를 읽는 bytes/sec에 의해 디코딩이 제한됨 |
| Core ML | "Apple 변환" | ANE 네이티브 모델을 위한 Apple 프레임워크 |
| QNN | "Qualcomm 스택" | Qualcomm Neural Network SDK |

## 추가 읽기

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 현황 및 벤치마크.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026년 엣지 포트 발표.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 설계 및 벤치마크.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 네이티브 변환.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon용 사전 변환된 모델.
