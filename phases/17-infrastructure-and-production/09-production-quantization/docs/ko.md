# 프로덕션 양자화 — AWQ, GPTQ, GGUF K-quant, FP8, MXFP4/NVFP4

> 양자화 형식은 보편적인 선택이 아닙니다. 하드웨어, 서빙 엔진, 워크로드에 따라 결정되는 함수입니다. GGUF Q4_K_M 또는 Q5_K_M은 CPU 및 엣지 환경에서 `llama.cpp`와 `Ollama`를 통해 제공됩니다. vLLM 내부에서는 동일한 기반 모델에 다중 LoRA를 적용해야 할 때 GPTQ가 유리합니다. AWQ는 Marlin-AWQ 커널을 사용하여 7B급 모델에서 약 741 tok/s를 제공하며, INT4에서 가장 높은 Pass@1을 달성합니다. 이는 2026년 데이터센터 프로덕션의 기본값입니다. FP8은 Hopper, Ada, Blackwell에서 중간 지점을 유지합니다. 거의 손실 없이 광범위하게 지원됩니다. NVFP4와 MXFP4 (Blackwell 마이크로 스케일링)는 공격적이며 블록 단위 검증을 필요로 합니다. 두 가지 함정이 팀을 괴롭힙니다. 캘리브레이션 데이터셋은 배포 도메인과 일치해야 하며, KV 캐시는 가중치 양자화와 별개입니다. AWQ의 "내 모델은 이제 4 GB다"라는 교훈은 프로덕션 배치 크기에서 10-30 GB의 KV 캐시를 잊고 있습니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 형식 간 메모리 및 처리량 비교용 장난감 예제)
**선수 요건:** 10단계 · 13강 (양자화 기초), 17단계 · 04강 (서빙 엔진 내부 구조)
**시간:** 약 75분

## 학습 목표

- 2026년 프로덕션 양자화 형식 6가지와 각각의 최적 사용처를 나열해 보세요.
- 하드웨어 (CPU 대 GPU, Hopper 대 Blackwell), 엔진 (vLLM, TRT-LLM, llama.cpp), 워크로드 (루틴 채팅, 추론, 다중 LoRA)에 따라 형식을 선택해 보세요.
- 선택한 형식에서 절약된 가중치 메모리와 건드리지 않은 KV 캐시를 계산해 보세요.
- 양자화된 모델이 도메인 트래픽에서 성능이 저하되는 캘리브레이션 데이터셋의 함정을 나열해 보세요.

## 문제점

양자화는 메모리와 HBM 대역폭을 줄입니다. 이는 디코딩에 정확히 필요한 부분입니다. FP16 70B 모델은 가중치가 140 GB입니다. 가중치를 INT4 (AWQ 또는 GPTQ)로 양자화하면 모델은 35 GB가 되어 하나의 H100에 KV 캐시를 위한 공간을 남기며 fits합니다. 이는 중요합니다. 왜냐하면 128개의 동시 시퀀스와 2k 컨텍스트에서는 KV 캐시만으로도 20-30 GB가 필요하기 때문입니다.

양자화는 공짜가 아닙니다. 공격적인 양자화는 품질을 저하시키며, 특히 추론이 많은 작업에서 두드러집니다. 형식마다 엔진이 다르고, 하드웨어마다 네이티브로 지원하는 정밀도가 다릅니다. 2026년 형식 동물원은 실재하며, 남의 선택을 복사할 수 없습니다. 스택에 기반해 선택해야 합니다.

## 개념

### 여섯 가지 형식

| 형식 | 비트 | 최적 영역 | 엔진 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, 엣지, 노트북 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM에서의 다중 LoRA | vLLM, TGI |
| AWQ | 4 | 데이터센터 GPU 프로덕션 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 데이터센터 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 다중 사용자 | TRT-LLM |
| NVFP4 | 4 | Blackwell 다중 사용자 | TRT-LLM |

### GGUF — CPU/엣지 기본값

GGUF는 양자화 방식이라기보다 파일 형식입니다. K-quant 변형(Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)을 하나의 컨테이너에 번들링합니다. Q4_K_M과 Q5_K_M이 프로덕션 기본값이며, 4-5비트에서 BF16에 가까운 품질을 제공합니다. llama.cpp가 압도적으로 빠른 CPU 추론 엔진이므로 CPU 또는 엣지 서빙에 가장 좋은 선택입니다.

vLLM에서의 처리량 페널티: 7B에서 약 93 tok/s — 형식이 GPU 커널에 최적화되지 않았습니다. 배포 대상이 CPU/엣지일 때만 GGUF를 사용하세요. 그 외에는 사용하지 마세요.

### GPTQ — vLLM에서의 다중 LoRA

GPTQ는 캘리브레이션 패스를 포함하는 학습 후 양자화 알고리즘입니다. Marlin 커널은 GPU에서 빠르게 동작합니다(Marlin이 아닌 GPTQ 대비 2.6배 속도 향상). 7B에서 약 712 tok/s입니다.

독특한 강점: GPTQ-Int4는 vLLM에서 LoRA 어댑터를 지원합니다. 기본 모델과 10-50개의 미세 조정 변형(각각 LoRA)을 서빙한다면 GPTQ가 경로입니다. NVFP4는 2026년 초 기준 LoRA를 지원하지 않습니다.

### AWQ — 데이터센터 GPU 기본값

활성화 인식 가중치 양자화(Activation-aware Weight Quantization)입니다. 양자화 중 가장 중요한 가중치 약 1%를 보호합니다. Marlin-AWQ 커널: 단순 방식 대비 10.9배 속도 향상. 7B에서 약 741 tok/s, INT4 형식 중 최상의 Pass@1.

다중 LoRA(GPTQ)나 공격적인 Blackwell FP4(NVFP4)가 필요하지 않은 한, 새로운 GPU 서빙에는 AWQ를 선택하세요.

### FP8 — 신뢰할 수 있는 중간 지점

8비트 부동 소수점. 거의 손실 없음. 널리 지원됨. Hopper Tensor Cores는 FP8를 네이티브로 가속화합니다. Blackwell은 이를 상속받습니다. 품질이 타협할 수 없는 경우(추론, 의료, 코드 생성) FP8은 2026년의 안전한 기본값입니다. 메모리 절감은 INT4의 절반이지만 품질 위험은 훨씬 낮습니다.

### MXFP4 / NVFP4 — Blackwell의 공격적 양자화

마이크로스케일링 FP4. 가중치 블록마다 자체 스케일 팩터를 가집니다. 공격적이지만 Blackwell Tensor Cores에서 하드웨어 가속됩니다. FP8 대비 토큰당 바이트 수를 절반으로 줄입니다 — 17단계 · 07강에서의 경제적 이점입니다.

주의 사항:
- LoRA 지원이 아직 없습니다 (2026년 초).
- 추론 중심 워크로드에서 품질 저하가 눈에 띄게 나타납니다.
- 모델별로 평가 세트에서 검증해 보세요.

### 보정(calibration)의 함정

AWQ와 GPTQ는 보정(calibration) 데이터셋이 필요합니다 — 일반적으로 C4 또는 WikiText를 사용합니다. 도메인 모델(코드, 의료, 법률)의 경우, 일반적인 웹 텍스트로 보정하면 알고리즘이 어떤 가중치를 보호해야 할지 잘못된 결정을 내릴 수 있습니다. HumanEval의 Pass@1 점수가 몇 포인트 떨어질 수 있습니다.

해결책: 도메인 내 데이터로 보정하세요. 수백 개의 도메인 샘플이면 보통 충분합니다. 출시하기 전에 평가 세트에서 테스트해 보세요.

### KV 캐시의 함정

AWQ는 가중치를 4비트로 줄입니다. KV 캐시는 분리되어 FP16/FP8을 유지합니다. AWQ를 적용한 70B 모델의 경우:

- 가중치: 약 35 GB (140 GB에서 INT4로).
- 128 동시 연결 × 2k 컨텍스트의 KV 캐시: 약 20 GB.
- 활성화: 약 5 GB.
- 총합: 약 60 GB — H100 80GB에 fits합니다.

단순히 "모델을 4 GB로 양자화했다"라고 말하면 나머지 30-50 GB를 잊게 됩니다. HBM 예산을 전체적으로 고려하세요.

별도로, KV 캐시 양자화(FP8 KV 또는 INT8 KV)는 자체적인 트레이드오프를 가진 다른 선택입니다 — 어텐션 정확도에 직접적인 영향을 미치며 무료 이점이 아닙니다.

### AWQ INT4는 추론에 위험합니다

사고의 연쇄(CoT), 수학, 긴 컨텍스트의 코드 생성 — 이러한 작업들은 공격적인 양자화로 눈에 띄게 피해를 입습니다. AWQ INT4는 MATH에서 약 3-5 포인트를 잃습니다. 추론 중심 워크로드의 경우, FP8 또는 BF16을 출시하세요; 메모리 비용을 감수해야 합니다.

### 2026년 선택 가이드

- CPU/엣지 서빙: GGUF Q4_K_M. 끝.
- GPU 서빙, 일상적인 채팅, LoRA 없음: AWQ.
- GPU 서빙, 다중 LoRA: Marlin을 사용한 GPTQ.
- 추론 작업 부하: FP8.
- Blackwell 데이터센터, 검증된 품질: NVFP4 + FP8 KV.
- 모호한 경우: 각 후보 형식에 대해 1,000개 샘플 평가를 실행해 보세요.

```figure
gpu-memory-breakdown
```

## 사용하기

`code/main.py`는 다양한 모델 크기에 대해 6가지 형식의 메모리 점유율(가중치 + KV + 활성화)과 상대적 처리량을 계산합니다. KV 캐시가 지배적인 위치, 가중치 압축이 효율적인 위치, FP8이 안전한 선택인 위치를 보여줍니다.

## 출시하기

이 강의는 `outputs/skill-quantization-picker.md`를 생성합니다. 하드웨어, 모델 크기, 작업 부하 유형, 품질 허용 범위를 고려하여 형식을 선택하고 보정/검증 계획을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행해 보세요. 128개 동시 연결 및 2k 컨텍스트에서 70B 모델의 각 형식별 총 HBM을 계산해 보세요. 어떤 형식으로 H100 80GB에 적합하게 맞출 수 있나요?
2. 7B 코딩 모델이 있습니다. 형식을 선택하고 정당화해 보세요. 품질 허용 범위에 대해 판단이 틀렸다면, 복구 경로는 무엇인가요?
3. 의료 도메인 모델에 대해 AWQ를 보정하는 데 필요한 보정 데이터셋 크기를 계산해 보세요. 더 많은 데이터가 항상 더 좋은 결과를 가져오지 않는 이유는 무엇인가요?
4. Marlin-AWQ 커널 논문이나 릴리스 노트를 읽어 보세요. AWQ가 7B 모델에서 741 tok/s를 달성하는 반면, 원시 GPTQ는 약 712 tok/s에 머무르는 이유를 세 문장으로 설명해 보세요.
5. AWQ 가중치를 FP8 KV 캐시와 결합하는 것이 KV를 BF16으로 유지하는 것보다 합리적인 경우는 언제인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| GGUF | "llama.cpp 형식" | K-quant 변형을 번들링하는 파일 형식; CPU/엣지 기본값 |
| Q4_K_M | "Q4 K M" | 4비트 K-quant 중간; 생산용 GGUF 기본값 |
| GPTQ | "gee pee tee q" | 보정을 통한 학습 후 INT4; vLLM에서 LoRA 지원 |
| AWQ | "a w q" | 활성화 인식 INT4; Marlin 커널; INT4에서 최상의 Pass@1 |
| Marlin 커널 | "빠른 INT4 커널" | Hopper용 INT4 커스텀 CUDA 커널; 10배 속도 향상 |
| FP8 | "8비트 부동 소수점" | Hopper/Ada/Blackwell의 안전한 정밀도 기본값 |
| MXFP4 / NVFP4 | "microscaling four" | 블록별 스케일 팩터를 사용하는 Blackwell 4비트 FP |
| 보정 데이터셋 | "cal data" | 양자화 매개변수를 선택하는 데 사용된 입력 텍스트; 도메인과 일치해야 합니다 |
| KV 캐시 양자화 | "KV INT8" | 가중치와 분리된 선택; 어텐션 정확도에 영향을 미칩니다 |

## 추가 읽기

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 비교 벤치마크.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 형식별 처리량 수치.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 형식별 선택.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 지원되는 형식 및 플래그.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ의 원형 공식화.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ의 원형 공식화.
