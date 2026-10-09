---
name: engine-picker
description: 하드웨어, 규모, 워크로드를 고려하여 자가 호스팅 LLM 엔진(llama.cpp, Ollama, TGI, vLLM, SGLang)을 선택합니다. 2026년 TGI 유지보수 모드를 마이그레이션 트리거로 명시합니다.
version: 1.0.0
phase: 17
lesson: 28
tags: [self-hosted, vllm, sglang, llama-cpp, ollama, tgi, trt-llm, engine-selection]
---

하드웨어(CPU / Apple Silicon / AMD / NVIDIA Hopper / NVIDIA Blackwell), 규모(단일 사용자 / 소규모 팀 / 프로덕션 / 엔터프라이즈), 워크로드(일반 채팅 / 에이전틱 / RAG (검색 증강 생성)(RAG (Retrieval-Augmented Generation)) / 긴 컨텍스트 / 코드)를 고려하여 엔진 권장 사항을 생성합니다.

생성 내용:

1. 엔진. 특정 엔진을 명시합니다. 하드웨어 우선, 규모 차선, 워크로드 최후의 트리 구조를 인용합니다.
2. 대안 엔진이 선택되지 않은 이유. 각 대안 엔진에 대해 선택되지 않은 이유를 명시합니다(TGI 유지보수 모드, AMD는 TRT-LLM 제외, Ollama는 개발 전용).
3. 파이프라인. 프로덕션 환경이라면 파이프라인 패턴(개발 Ollama → 스테이징 llama.cpp → 프로덕션 vLLM/SGLang)을 명시하고 가중치 형식(GGUF 또는 HF)이 흐름을 통해 전달되는지 확인합니다.
4. 프로덕션 스택 구성. 프로덕션 규모에서는 구성을 위해 17단계 · 18강(프로덕션 스택), · 17강(분산 서빙(Disaggregated Serving)), · 11강(캐시 인식 라우터)을 참조합니다.
5. TGI 마이그레이션. 현재 엔진이 TGI라면 마이그레이션 계획과 타임라인을 명시합니다. 긴급하지는 않지만 6개월 내에 시작해야 합니다.
6. 하드웨어 주의 사항. 두 가지 하드 제약 조건을 명시합니다: CPU 전용 → llama.cpp; AMD → TRT-LLM 사용 불가.

하드 리젝트(Hard Rejects):
- 2026년 신규 프로젝트를 TGI로 기본 설정하는 것. 거부합니다 — 유지보수 모드입니다.
- 동시 사용자 1명 이상인 공유 프로덕션 환경에서 Ollama를 사용하는 것. 거부합니다 — 처리량 격차 때문입니다.
- NVIDIA 전용임을 확인하지 않고 TRT-LLM을 제안하는 것. 거부합니다 — AMD / 비-NVIDIA는 하드 블록입니다.

거부 규칙:
- 하드웨어가 혼합된 경우(일부는 AMD, 일부는 NVIDIA), 클러스터별 엔진 결정을 요구합니다. 단일 엔진을 강제하지 마세요.
- 프로덕션 규모에서 워크로드가 "알 수 없음/일반적"인 경우, vLLM을 기본으로 설정하고 3개월간의 트래픽 데이터 수집 후 재평가 계획을 세우세요.
- 팀이 "Blackwell 가용성 없이 GPU당 가장 빠른 속도"를 원하며 Hopper 전용을 고집하는 경우, 확인합니다. TRT-LLM과 vLLM 모두 허용됩니다.

출력: 엔진 선택, 제외된 대안, 파이프라인, 프로덕션 스택 구성, TGI 마이그레이션 전략을 담은 한 장의 권장 사항. 분기별 리뷰로 마무리하세요: 워크로드 형태가 실질적으로 변할 때 엔진 선택을 재평가합니다.
