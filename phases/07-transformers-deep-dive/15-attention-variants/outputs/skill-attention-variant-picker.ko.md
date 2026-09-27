---
name: attention-variant-picker
description: 컨텍스트 길이, 검색 요구 사항 및 연산 프로필을 기반으로 새로운 모델에 적합한 전체(full) / 슬라이딩 윈도우(sliding-window) / 희소(sparse) / 차분(differential) 어텐션 토폴로지를 선택합니다.
version: 1.0.0
phase: 7
lesson: 15
tags: [attention, transformer, long-context, inference, memory]
---

# Attention Variant Picker (어텐션 변형 선택기)

개발자가 새로운 트랜스포머를 설계하거나, 기존 모델을 더 긴 컨텍스트(longer context)로 확장할 때 적절한 어텐션 토폴로지(attention topology)를 선택하고 그 근거를 제시할 수 있도록 도와줍니다.

## 수집해야 할 입력 정보 (Inputs to gather)

1. **목표 컨텍스트 길이 (Target context length)**: 학습 시와 추론 시의 길이를 각각 파악해야 합니다 (종종 서로 다릅니다. 많은 모델이 16K에서 학습하고 추론 시에 확장합니다).
2. **검색 요구도 (Retrieval demand)**: 1~5 척도로 측정합니다. 1 = 순수 채팅, 5 = needle-in-haystack / RAG / 긴 저장소 컨텍스트를 포함한 코드 작업.
3. **추론 메모리 예산 (Inference memory budget)**: 요청당 KV 캐시 허용 범위 (레이어당 토큰당 바이트(`bytes per token per layer`)가 적절한 단위입니다).
4. **학습 비용 허용치 (Training cost tolerance)**: SWA를 처음부터 학습하는 것은 저렴하지만, 사전 학습된 모델에 차분 어텐션(differential attention)을 맞춤형으로 적용하는 것은 비용이 많이 듭니다.
5. **하드웨어 타겟 (Hardware target)**: Hopper+는 FlashAttention-3를 완벽히 지원하며, Ada는 FA2를 지원하고, 구형 GPU는 마스크(mask) 제한이 있습니다.

## 결정 규칙 (Decision rules)

- **컨텍스트(Context) ≤ 16K 이고 검색(retrieval) ≤ 3인 경우**: FlashAttention을 사용한 전체 어텐션(full attention)을 적용하세요. 성급하게 최적화하지 마세요.
- **컨텍스트 16–128K 이고 검색 ≤ 3인 경우**: SWA와 글로벌(global) 어텐션을 5:1 비율로 혼합하고, 윈도우 크기는 1024로 설정하세요 (Gemma 3 구조). 이는 KV 캐시를 압축하면서도 검색 성능을 유지할 수 있게 합니다.
- **컨텍스트 > 128K인 경우**: 4~6개 레이어마다 글로벌 레이어를 배치하는 전체 SWA(full SWA)를 적용하고, 위치 보간(position interpolation) 또는 YaRN 스케일링(Lesson 04)을 병행하세요.
- **검색(Retrieval) = 5 이고 학습 예산이 허용되는 경우**: 상위 4개 레이어에만 차분 어텐션(differential attention) 적용을 고려하세요 (KV 배증을 절반으로 줄이면서, 싱크 취소(sink-cancellation) 이득을 대부분 챙길 수 있습니다).
- **공개 API를 출시하는 경우**: 안정적인 패턴(full, SWA, Gemma-3 혼합 방식)을 선호하세요. 커널 엔지니어가 없다면 네이티브 희소(native-sparse) 또는 차분(differential) 방식은 건너뛰세요.
- **베이스 모델을 변경할 수 없는 경우**: SWA는 마스킹(masking)을 통해 추론 시점에 사후 적용(retrofit)이 가능하지만, 차분(differential) 및 희소(sparse) 방식은 불가능합니다.

## 항상 주의할 점 (Always flag)

- 7B 미만의 순수 SWA(Pure-SWA) 모델은 추론 벤치마크에서 측정 가능한 성능 저하가 발생하는 경우가 많습니다. 사용을 권장하지 않습니다.
- 윈도우 크기(Window size)가 512 미만인 설정은 거의 항상 부적절합니다. 더 크게 설정하거나 다른 토폴로지(topology)를 사용해 보세요.
- 논문에 보고된 차분 어텐션(Differential attention) 결과는 소형 모델(3–7B) 기준입니다. 2026년 초 현재, 스케일업(Scale-up)에 대한 증거는 부족합니다.
- 모든 변형 모델은 RoPE / YaRN 스케일링(Lesson 04)과 상호작용합니다. 위치 인코딩 방식(position scheme)을 명시적으로 기술하세요.

## 출력 형식 (Output format)

반환 내용:

1. **권장 사항 (Recommendation)** — 단일 명명된 토폴로지 (예: "Gemma-3 mix, W=1024, 5:1 SWA:global").
2. **근거 (Justification)** — 각 입력을 위의 결정 규칙에 매핑합니다.
3. **KV 캐시 추정치 (KV cache estimate)** — 대상 컨텍스트에서, 레이어당 토큰당 바이트(`bytes per token per layer`) 및 배치 크기 1일 때의 GB 단위.
4. **마이그레이션 경로 (Migration path)** — 베이스 모델이 이미 학습된 경우, 어떻게 개조(retrofit)할 것인지에 대한 방법.
5. **알려진 리스크 (Known risks)** — 어떤 벤치마크 또는 워크로드에서 성능 저하(regress)가 발생할 수 있는지.
