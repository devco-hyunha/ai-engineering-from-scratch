---
name: skill-quantization
description: 하드웨어, 품질, 지연 시간 제약에 기반하여 LLM 배포를 위한 최적의 양자화 전략을 선택합니다
version: 1.0.0
phase: 10단계
lesson: 11강
tags: [양자화, 추론, 배포, 최적화, fp8, int4, int8, gptq, awq, gguf]
---

# 양자화 결정 프레임워크

언어 모델을 배포할 때, 이 프레임워크를 사용하여 올바른 숫자 형식, 양자화 방법 및 품질 검증 전략을 선택해 보세요.

## 입력 요구 사항

다음 정보를 제공하세요:
- **모델** (이름, 매개변수 수, 원본 정밀도)
- **타겟 하드웨어** (GPU 모델/VRAM, CPU, Apple Silicon, 엣지 디바이스)
- **지연 시간 목표** (초당 토큰 수, 첫 토큰까지의 시간)
- **품질 하한선** (허용되는 최대 퍼플렉시티 증가량, 벤치마크 차이)
- **서빙 패턴** (배치 크기, 최대 컨텍스트 길이, 동시 사용자 수)

## 빠른 선택

| 상황 | 형식 | 방법 | 예상 품질 손실 |
|---------------|--------|--------|----------------------|
| H100 GPU, 최대 처리량 | FP8 E4M3 | H100 네이티브 캐스팅 | < 0.1% |
| A100/A10, 2배 처리량 필요 | INT8 | LLM.int8() 또는 SmoothQuant | < 0.5% |
| 단일 24GB GPU, 70B 모델 | INT4 | AWQ 또는 GPTQ | 1-3% |
| MacBook / Apple Silicon | INT4 GGUF | llama.cpp를 통한 Q4_K_M | 1-2% |
| 모바일 / 엣지 디바이스 | INT4 또는 INT3 | QAT + 디바이스 특화 | 2-5% |
| 최대 압축, 일부 손실 허용 | INT2 | QuIP# 또는 AQLM | 5-15% |
| 학습 (혼합 정밀도) | BF16 + FP32 누적 | 네이티브 프레임워크 지원 | 0% |

## 구성 요소별 정밀도 선택

모든 텐서에 동일한 처리를 적용해서는 안 됩니다.

| 구성 요소 | 안전한 최소값 | 권장값 | 피해야 할 값 |
|-----------|-------------|-------------|-------|
| FFN 가중치 | INT4 | INT4 (AWQ/GPTQ) | QAT 없는 INT2 |
| 어텐션 가중치 | INT4 | INT8 또는 FP8 | INT2 |
| 임베딩 레이어 | INT8 | FP16 (원본 유지) | INT4 |
| 출력 헤드 | INT8 | FP16 (원본 유지) | INT4 |
| KV 캐시 | FP8 | FP8 또는 INT8 | 긴 컨텍스트에서 INT4 |
| 어텐션 로짓 | FP16 | FP16 또는 BF16 | INT8 |
| 활성화 (추론) | INT8 | FP8 또는 INT8 | INT4 |

## 방법 비교

### GPTQ
- **언제:** GPU 추론, Hugging Face 호환 모델을 원할 때
- **보정 데이터:** 예시 128개, 각 2048 토큰
- **시간:** A100에서 70B 모델 기준 30-60분
- **도구:** `gptqmodel`, `exllama`, `exllamav2`
- **장점:** 잘 검증됨, Hugging Face에 방대한 모델 저장소
- **단점:** AWQ보다 적용 속도가 느림, 일부 모델에서 AWQ보다 품질이 약간 낮음

### AWQ
- **언제:** GPU 추론, 비트당 최고의 품질을 원할 때
- **보정 데이터:** 예시 128개
- **시간:** A100에서 70B 모델 기준 15-30분
- **도구:** `llmcompressor`, `vLLM` (네이티브 지원)
- **장점:** 최고의 INT4 품질, 빠른 적용 속도, vLLM 통합
- **단점:** GPTQ보다 모델 저장소가 작음

### GGUF
- **언제:** CPU 추론, Apple Silicon, llama.cpp 생태계
- **변형:** Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0, F16
- **권장 기본값:** Q4_K_M (최적의 품질/크기 균형)
- **도구:** `llama.cpp`, `ollama`, `LM Studio`
- **장점:** 자체 완결된 파일, 혼합 정밀도, 거대한 생태계
- **단점:** GPU에 최적화되지 않음 (CPU/Metal용으로 설계됨)

### SmoothQuant
- **언제:** GPU에서 INT8, 가중치와 활성화 모두 양자화해야 할 때
- **핵심 아이디어:** 채널별 스케일링을 통해 양자화 난이도를 활성화에서 가중치로 이동
- **도구:** `smoothquant`, `TensorRT-LLM`
- **장점:** W8A8 (가중치와 활성화 모두 INT8)를 가능하게 하여 2배 속도 향상
- **단점:** INT8 전용, INT4로 확장되지 않음

## 품질 검증 프로토콜

양자화 후, 배포하기 전에 검증하세요:

1. **퍼플렉시티 테스트.** WikiText-2 또는 도메인 코퍼스로 계산하세요. 델타가 0.5 미만이면 우수, 0.5-1.0이면 양호, 2.0 초과이면 문제입니다.

2. **벤치마크 스윕.** MMLU (일반), GSM8K (수학), HumanEval (코드)를 실행하세요. 수학 및 코드는 정밀도 손실에 가장 민감합니다.

3. **출력 비교.** 원본 모델과 양자화 모델에서 각각 100개의 응답을 생성하세요. LLM-as-judge를 사용하여 승률을 계산하세요. 목표: 양자화 모델이 프롬프트의 90% 이상에서 승리하거나 무승부를 기록해야 합니다.

4. **레이턴시 측정.** 배치 크기 1 및 목표 배치 크기에서 초당 토큰 수를 측정하세요. 속도 향상이 품질 비용을 정당화하는지 확인하세요.

5. **긴 컨텍스트 테스트.** 긴 컨텍스트 (> 4K 토큰)를 서빙하는 경우, 최대 컨텍스트 길이에서 테스트하세요. KV 캐시 양자화 오류는 시퀀스 길이에 따라 누적됩니다.

## 메모리 예산 계산기

```
Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB (relatively constant, depends on batch size)
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)
```

Llama 3 70B의 INT4, 32K 컨텍스트 예시:
- 가중치: 70B * 4 / 8 / 1.07 = 32.6 GB
- KV 캐시 (FP16): 2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 = ~40 GB
- KV 캐시 (FP8): ~20 GB
- FP8 KV 포함 총계: ~55 GB (80GB A100 하나에 fits)

## 공통 실수

| 실수 | 실패 원인 | 해결책 |
|---------|-------------|-----|
| 임베딩 레이어를 INT4로 양자화 | 첫 번째 레이어가 전체 모델에 걸쳐 오류를 증폭시킴 | 임베딩은 FP16 또는 INT8로 유지 |
| INT4에 텐서 단위 스케일 사용 | 하나의 이상치 행이 모든 행의 정밀도를 파괴함 | 채널 단위 또는 그룹 단위 스케일 사용 |
| GPTQ/AWQ 보정(calibration) 생략 | 대표 데이터가 없으면 스케일 팩터가 부정확함 | 도메인에서 128개 예제 사용 |
| 모든 레이어에 동일한 비트 폭 사용 | 첫 번째/마지막 레이어가 더 민감함 | 혼합 정밀도: 첫 번째/마지막 레이어에 더 높은 비트 사용 |
| 매우 긴 컨텍스트에서 KV 캐시 양자화 | 오류가 시퀀스 길이에 따라 2차적으로 누적됨 | KV 캐시에는 INT4가 아닌 FP8 사용 |
| 품질 검증 생략 | 일부 모델은 양자화 품질이 낮음 (특히 경계에서) | 항상 퍼플렉시티 + 작업 평가 실행 |

## 배포 레시피

### 레시피 1: vLLM + AWQ (GPU 서버)
```
pip install vllm
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

### 레시피 2: llama.cpp + GGUF (MacBook)
```
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99
```

### 레시피 3: TensorRT-LLM + FP8 (H100)
```
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
```
