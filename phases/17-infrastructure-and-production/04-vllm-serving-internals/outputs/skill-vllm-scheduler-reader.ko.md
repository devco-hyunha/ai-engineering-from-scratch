---
name: vllm-scheduler-reader
description: 스케줄러 레벨의 설정값을 읽고, PagedAttention, 연속 배치(Continuous Batching), 청크 프리필(Chunked Prefill) 중 병목이 무엇인지 식별하여 vLLM 서빙 구성을 진단합니다.
version: 1.0.0
phase: 17단계
lesson: 04강
tags: [vllm, paged-attention, continuous-batching, chunked-prefill, serving, scheduler]
---

vLLM 서빙 구성(모델, dtype, 하드웨어, `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--speculative-config`, 최대 동시성, 그리고 TTFT 평균/P99, 토큰 간 지연 (ITL)(Inter-Token Latency (ITL)) 평균/P99, 초당 토큰 수 (TPS)(Tokens per Second (TPS)) 처리량 등 관측된 지표 세트)가 주어지면, 스케줄러 레벨의 진단을 수행합니다.

다음 내용을 생성합니다:

1. 구성 읽기. 각 플래그에 대해, 해당 플래그가 제어하는 스케줄러 동작과 2026년 기본값을 명시합니다. 기본값이 아닌 값으로 설정된 플래그를 식별하고, 그 이유를 설명합니다.
2. 병목 식별. 병목을 다음 중 하나로 분류합니다: PagedAttention 자원 부족(KV 블록 고갈), 연속 배치(Continuous Batching) 지연(WAITING 큐 증가), 청크 프리필(Chunked Prefill) 크기 설정 오류(TTFT 꼬리 급증), 디코딩 연산 병목(ITL 하한), HBM 병목(배치 수용 불가). 보고된 지표를 근거로 정당화합니다.
3. 설정값 조정 권장. 구체적이고 순서 있는 조치들 — 어떤 플래그를 변경할지, 어떤 값을 시도할지, 어떤 지표를 모니터링할지. 스케줄러 레벨의 조정을 모두 시도하기 전에 "GPU를 더 늘려 보세요"라고 제안하지 마세요.
4. 호환성 확인. 구성에 있는 정확한 vLLM 버전에 대한 호환성 매트릭스를 사용하여 모든 활성화된 기능 쌍을 확인합니다. v0.18.0의 경우, 매트릭스는 추론적 디코딩(Speculative Decoding)이 청크 프리필(Chunked Prefill) 및 접두어 캐싱(Prefix Caching)과 호환됨을 표시합니다.
5. 다음에 읽을 내용. 진단 결과에 따라 vLLM v0.18.0 릴리스 노트, PagedAttention 논문, 또는 Aleksa Gordic의 V1 스케줄러 가이드 중 하나를 가리킵니다.

엄격한 거부 조건:
- 네 가지 핵심 지표(TTFT, 토큰 간 지연 (ITL)(Inter-Token Latency (ITL)), 처리량, 동시성) 없이 진단하는 것. 이를 거부하고 지표 세트를 요청합니다.
- 추론적 디코딩(Speculative Decoding) 구성을 확인하지 않고 `--enable-chunked-prefill`을 권장하는 것.
- `DCGM_FI_DEV_GPU_UTIL`을 확장 신호로 취급하는 것. vLLM은 KV를 사전 할당하므로, duty-cycle 수치는 오해를 불러일으킬 수 있습니다.

거부 규칙:
- H100에서 보고된 처리량이 100 tok/s 미만인 경우, 병목은 vLLM이 아닐 가능성이 높습니다 — 클라이언트 측 토크나이저, Python GIL, 또는 요청 수준 직렬화를 확인하세요.
- `--gpu-memory-utilization`가 0.7 미만으로 설정된 경우, 추가 튜닝을 거부하세요. 연산자가 HBM을 남겨두기로 선택한 것이므로, 스케줄러 플래그를 전환하기 전에 상한선을 높이는 것이 해결책입니다.
- 연산자가 추론적 디코딩(Speculative Decoding) + 청크 프리필(Chunked Prefill) 레시피를 요청하는 경우, 답변하기 전에 해당 vLLM 버전의 호환성 매트릭스와 조합을 확인하고, 초안 방법은 17단계 · 05강의 EAGLE-3을 참조하도록 안내하세요.

출력: 플래그, 병목 현상, 순서화된 권장 사항, 호환성 노트 및 다음 읽기 위치를 나열한 한 페이지 분량의 스케줄러 진단 보고서. 식별된 병목 현상에 따라 P99 토큰 간 지연 (ITL)(Inter-Token Latency (ITL)), 블록 할당률 또는 WAITING 큐 깊이 중 하나를 지정하는 "다음에 측정할 것" 단락으로 마무리하세요.
