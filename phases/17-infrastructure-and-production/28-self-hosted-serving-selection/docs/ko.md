# 셀프 호스팅 서빙 선택 — 엔진을 하드웨어 및 규모에 매칭하기

> 엔진 선택은 하드웨어, 규모, 생태계의 함수이지 리더보드 읽기가 아닙니다. 2026년 셀프 호스팅 추론을 지배하는 네 가지 엔진은 llama.cpp, Ollama, vLLM, SGLang이며, TGI는 유지보수 모드에 있습니다. **llama.cpp**는 CPU에서 가장 빠릅니다 — 가장 넓은 모델 지원, 양자화 및 스레딩에 대한 완전한 제어. **Ollama**는 개발자 노트북용 원커맨드 설치이며, llama.cpp보다 약 15-30% 느립니다 (Go + CGo + HTTP 직렬화), 프로덕션 유사 부하에서는 처리량 격차가 3배입니다. **TGI는 2025년 12월 11일에 유지보수 모드에 진입했습니다** — 버그 수정만 수행하며, vLLM보다 원시 처리량이 약 10% 느리지만 역사적으로 최고의 관측 가능성 및 HF 생태계 통합을 제공했습니다. 이 유지보수 상태는 장기적으로 위험한 베팅이 되며, 신규 프로젝트에는 SGLang이나 vLLM이 더 안전한 기본값입니다. **vLLM**은 범용 프로덕션 기본값입니다 — v0.15.1 (2026년 2월)은 PyTorch 2.10, RTX Blackwell SM120, H200 최적화를 추가했습니다. **SGLang**은 에이전트형 멀티턴 / 접두어 중심의 전문가입니다 — 프로덕션에서 400,000개 이상의 GPU를 사용합니다 (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). 하드웨어 제약 조건: CPU 우선 → llama.cpp. AMD / 비-NVIDIA → vLLM이 가장 잘 지원되는 경로입니다 (TRT-LLM은 NVIDIA 전용). 2026년 파이프라인 패턴: dev = Ollama, staging = llama.cpp, prod = vLLM 또는 SGLang. 엔진은 서로 다른 가중치 형식을 사용합니다 — llama.cpp 계열은 GGUF, GPU 엔진은 HF safetensors — 따라서 형식 변환이 단계 사이에 위치할 수 있습니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 엔진 결정 트리 워커)
**선수 요건:** 엔진을 다루는 모든 17단계 강의를 포함 (04, 06, 07, 09, 18)
**시간:** 약 45분

## 학습 목표

- 하드웨어 (CPU / AMD / NVIDIA Hopper / Blackwell), 규모 (1명 사용자 / 100명 / 10,000명), 워크로드 (일반 채팅 / 에이전트 / 긴 컨텍스트)를 고려하여 엔진을 선택하세요.
- 2026년 TGI 유지보수 모드 상태 (2025년 12월 11일)를 명시하고, 이것이 신규 프로젝트를 vLLM이나 SGLang으로 편향시키는 이유를 설명하세요.
- dev/staging/prod 파이프라인을 설명하고, GGUF에서 safetensors 형식 변환이 단계 사이에 위치하는 부분을 포함하세요.
- "CPU 우선"이 llama.cpp를 가리키고 "AMD"가 TRT-LLM을 제외하는 이유를 설명하세요.

## 문제점

팀이 새로운 자체 호스팅 LLM 프로젝트를 시작합니다. 한 엔지니어는 Ollama를, 다른 엔지니어는 vLLM을, 세 번째 엔지니어는 "TGI가 바로开箱(开箱即出)으로 작동하지 않나요?"라고 말합니다. 세 가지 모두 서로 다른 맥락에서 옳습니다. 모두에게 옳은 것은 없습니다.

2026년에는 선택 트리가 중요합니다: 하드웨어가 먼저, 규모가 두 번째, 워크로드가 세 번째입니다. 그리고 2025년의 특정 사건인 TGI가 12월 11일에 유지보수 모드로 진입한 것은 새로운 프로젝트의 기본값을 변경합니다.

## 개념

### 5가지 엔진

| 엔진 | 최적 용도 | 비고 |
|--------|----------|-------|
| **llama.cpp** | CPU / 엣지 / 최소 의존성 / 가장 넓은 모델 지원 | CPU에서 가장 빠름, 완전한 제어 |
| **Ollama** | 개발용 노트북, 단일 사용자, 한 명령어 설치 | llama.cpp보다 15-30% 느림; 프로덕션 처리량 격차 3배 |
| **TGI** | HF 생태계, 규제 산업 | **2025년 12월 11일 유지보수 모드** |
| **vLLM** | 범용 프로덕션, 100명 이상 사용자 | 넓은 프로덕션 기본값; v0.15.1 2026년 2월 |
| **SGLang** | 에이전트형 다중 턴, 접두어 중심 워크로드 | 프로덕션에서 400,000+ GPU |

### 하드웨어 우선 결정

**CPU 우선** → llama.cpp. Ollama도 작동하지만 느립니다. CPU에서는 다른 엔진이 경쟁력이 없습니다.

**AMD GPU** → vLLM이 가장 잘 지원되는 경로입니다 (AMD ROCm 지원). SGLang도 작동합니다. TRT-LLM은 NVIDIA 전용이므로 제외됩니다.

**NVIDIA Hopper (H100 / H200)** → vLLM, SGLang 또는 TRT-LLM. 세 가지 모두 최상위 수준입니다.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM이 처리량 리더입니다 (17단계 · 07). vLLM과 SGLang이 뒤를 closely 따릅니다.

**Apple Silicon (M 시리즈)** → llama.cpp (Metal). Ollama가 이를 래핑합니다.

### 규모 두 번째 결정

**1명 사용자 / 로컬 개발** → Ollama. 한 명령어로, 몇 초 만에 첫 토큰이 나옵니다.

**10-100명 사용자 / 소규모 팀** → vLLM 단일 GPU.

**100-10,000명 사용자 / 프로덕션** → vLLM 프로덕션 스택 (17단계 · 18) 또는 SGLang.

**10,000명 이상 사용자 / 엔터프라이즈** → vLLM 프로덕션 스택 + 분산 서빙 (17단계 · 17) + LMCache (17단계 · 18).

### 워크로드 세 번째 결정

**일반 채팅 / Q&A** → vLLM이 넓은 기본값으로 승리합니다.

**에이전트형 멀티턴 (도구, 계획, 메모리)** → SGLang의 RadixAttention (17단계 · 06)이 압도적입니다.

**무거운 접두어 재사용을 사용하는 RAG** → SGLang입니다.

**코드 생성** → vLLM도 잘 작동합니다; 캐시 측면에서는 SGLang이 약간 더 좋습니다.

**긴 컨텍스트 (128K+)** → vLLM + 청크 프리필; SGLang + 계층형 KV입니다.

### TGI 유지보수 함정

Hugging Face TGI는 2025년 11월 11일에 유지보수 모드에 진입했습니다 — 앞으로는 버그 수정만 이루어집니다. 역사적으로: 최상급 관측 가능성, 최고 수준의 HF 생태계 통합 (모델 카드, 안전 도구), 순수 처리량 측면에서는 vLLM보다 약간 뒤처졌습니다.

2026년 신규 프로젝트의 경우: TGI를 기본 선택에서 제외하세요. 기존 TGI 배포는 계속 사용할 수 있지만, 결국 마이그레이션해야 합니다. SGLang과 vLLM이 더 안전한 기본 선택입니다.

### 파이프라인 패턴

개발 (Ollama) → 스테이징 (llama.cpp) → 프로덕션 (vLLM). 엔진들은 서로 다른 가중치 형식을 사용합니다 — llama.cpp 계열은 GGUF, GPU 엔진은 HF safetensors를 사용하므로 — 형식 변환이 단계 사이에 위치할 수 있습니다. 엔지니어는 노트북에서 빠르게 반복 작업합니다; 스테이징은 프로덕션 양자화를 반영합니다; 프로덕션은 서빙 대상입니다.

### Ollama 주의사항

Ollama는 개발에 적합합니다. 공유 프로덕션에는 적합하지 않습니다: Go HTTP 직렬화가 오버헤드를 추가하고, 동시성 관리가 vLLM보다 단순하며, OpenTelemetry 지원이 뒤처져 있습니다. Ollama가 빛나는 곳에서 사용하세요 — 한 사용자, 한 명령 — 그리고 공유 환경에서는 vLLM으로 전환하세요.

### 셀프 호스팅 vs 관리형은 별개의 결정입니다

17단계 · 01 (관리형 하이퍼스케일러), · 02 (추론 플랫폼)이 관리형을 다룹니다. 이 강의는 이미 셀프 호스팅을 결정했다고 가정합니다. 셀프 호스팅의 이유: 데이터 거주지, 맞춤형 미세 조정, 대규모에서의 총 소유 비용, 호스팅된 모델에서 제공되지 않는 도메인 모델.

### 기억해야 할 숫자

- TGI 유지보수 모드: 2025년 11월 11일.
- vLLM v0.15.1: 2026년 2월; PyTorch 2.10; Blackwell SM120 지원.
- SGLang 프로덕션 규모: 400,000+ GPU.
- llama.cpp 대비 Ollama 처리량 격차: 15-30% 느림; 프로덕션 부하에서 3배 느림.

```figure
data-parallel
```

## 사용하기

`code/main.py`는 결정 트리 워커입니다: 하드웨어 + 규모 + 작업 부하가 주어지면 엔진을 선택하고 이유를 설명합니다.

## 출시하기

이 강의는 `outputs/skill-engine-picker.md`를 생성합니다. 주어진 제약 조건을 고려하여 엔진을 선택하고 마이그레이션 계획을 작성합니다.

## 연습 문제

1. 하드웨어 / 규모 / 워크로드에 맞춰 `code/main.py`를 실행해 보세요. 결과가 직관과 일치하나요?
2. 인프라가 H100 12대와 AMD MI300X 8대입니다. 어떤 엔진을 선택해야 할까요? TRT-LLM이 제외되는 이유는 무엇인가요?
3. 한 팀이 "우리가 잘 아는 도구"라는 이유로 2026년에 TGI를 사용하려고 합니다. 마이그레이션의 필요성을 논증해 보세요.
4. Ollama 개발 환경에서 vLLM 프로덕션 환경으로 전환할 때, 양자화, 구성 및 관측 가능성에서 어떤 변화가 발생하나요?
5. P99 접두어 길이가 8K이고 테넌트 간 재사용이 높은 RAG 제품입니다. 엔진을 선택하고 17단계 · 11강 + 18강과 함께 스택을 구성해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| llama.cpp | "CPU용" | 가장 넓은 모델 지원, CPU에서 가장 빠름 |
| Ollama | "노트북용" | 한 명령어로 설치, 개발급 처리량 |
| TGI | "HF의 서빙" | 2025년 12월부터 유지보수 모드 |
| vLLM | "기본 선택지" | 2026년 광범위한 프로덕션 기준선 |
| SGLang | "에이전트용" | 접두어 중심, RadixAttention |
| TRT-LLM | "NVIDIA 전용" | Blackwell 처리량 리더, NVIDIA 전용 |
| GGUF | "llama.cpp 형식" | 번들된 K-quant 변형 |
| 프로덕션 스택 | "vLLM K8s" | 17단계 · 18강 참조 배포 |
| 파이프라인 패턴 | "dev→stage→prod" | Ollama → llama.cpp → vLLM; 엔진마다 가중치 형식이 다름 |

## 추가 읽기

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 릴리스 노트.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
