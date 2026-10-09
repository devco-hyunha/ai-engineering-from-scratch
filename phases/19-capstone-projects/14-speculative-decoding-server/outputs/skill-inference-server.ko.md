---
name: inference-server
description: EAGLE-3 또는 P-EAGLE 초안, K8s 자동 확장, 전체 처리량/지연 시간/비용 보고서를 포함한 추론적 디코딩 추론 서버 출시하기.
version: 1.0.0
phase: 19단계
lesson: 14강
tags: [capstone, inference, vllm, sglang, eagle-3, p-eagle, speculative-decoding, quantization, hpa]
---

두 개의 오픈 소스 대상 모델(Llama 3.3 70B 및 Qwen3-Coder-30B MoE 또는 GPT-OSS-120B)을 사용하여 추론적 디코딩, 양자화, Kubernetes 자동 확장을 갖춘 프로덕션 서빙 스택을 출시하세요. 측정된 가속화 및 꼬리 지연(tail-latency) 수치를 공개하세요.

구축 계획:

1. FP8 Marlin 양자화(quantization)를 사용하여 vLLM 0.7(또는 SGLang 0.4)에서 대상 모델을 배포하세요.
2. Red Hat Speculators에서 정렬된 EAGLE-3 초안을 로드하거나 SpecForge를 통해 하나를 학습하세요.
3. 기본 수치: 초안(speculation) 없이 배치 1/8/32에서의 토큰/s 및 p50/p99 지연 시간.
4. EAGLE-3를 활성화하세요. 동일한 벤치마크를 다시 실행하세요. 가속화, 수용률, p99 꼬리 지연(tail-latency) 델타를 보고하세요.
5. P-EAGLE 병렬 초안을 활성화하세요. 더 깊은 트리가 도움이 되는 지점과 해가 되는 지점의 변곡점을 보고하세요.
6. 분포 전체에 걸쳐 벤치마크를 실행하세요: ShareGPT, HumanEval, 도메인 데이터. 수용률 드리프트를 공개하세요.
7. 두 번째 대상 모델(MoE)에서 반복하세요. 초안 수용에서 라우팅 잡음 민감성을 식별하세요.
8. `queue_wait_ms`을 추적하는 HPA를 사용하여 Kubernetes에 배포하세요. 부하가 3배가 될 때 스케일 아웃(scale-out)을 시연하세요.
9. 일치하는 평가에서 Anthropic Claude Sonnet 4.7 및 OpenAI GPT-5.4와 토큰 100만 개당($) 비용을 비교하세요.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 기본 대비 측정된 가속화 | 두 모델 모두에서 일치하는 품질로 2.5x+ 처리량 |
| 20 | 현실적인 트래픽에서의 수용률 | 분포별 수용률 보고서 |
| 20 | P99 꼬리 지연(tail-latency) 규율 | 초안 유무에 따른 배치 1/8/32에서의 p99 |
| 20 | 운영 | K8s 배포, 큐 대기 기반 HPA, 매끄러운 롤아웃, 드레인 우선 업그레이드 |
| 15 | 작성 및 방법론 | 지표의 명확한 유도, 일치하는 기본값 |

하드 리젝트:

- 꼬리 지연(tail latency) 없이 정상 상태 처리량을 보고하는 것.
- 큐 대기 대신 CPU 기반 HPA를 사용함. GPU 포화 상태에서 불안정해짐.
- 초안-타겟 버전 정렬을 무시함. drifted 초안은 추론적 디코딩(Speculative Decoding)을 하지 않는 것보다 비용이 더 많이 듦.
- 호스팅 API의 프롬프트 캐싱(Prompt Cache) 할인을 생략한 비용 비교.

거부 규칙:

- 롤아웃 드레인 없이 서빙하는 것을 거부함. 요청이 진행 중일 때 제자리 업그레이드(In-place Upgrade)를 하는 것은 자격 요건에 미달함.
- 분포 전체에 대해 집계된 수용률(Acceptance Rate)을 보고하는 것을 거부함. 분포별 보고는 필수임.
- bs=32에서 추론적 디코딩(Speculative Decoding)의 이점을 주장하는 것을 거부함. 비추론적(non-speculative) 수치와 일치하는 값이 없기 때문임.

출력물: vLLM / SGLang 설정, EAGLE-3 초안 다운로드 스크립트, K8s 배포 매니페스트, 큐 대기 기반 HPA 설정, ShareGPT / HumanEval / 도메인 데이터용 벤치마크 하네스, $/1M 토큰 비교 표, 그리고 추론적 디코딩(Speculative Decoding)이 도입한 세 가지 꼬리 지연(Tail Latency) 퇴보와 각각을 해결한 완화책(배치 게이팅, ngram 폴백, 양자화 조정)을 명시한 문서가 포함된 저장소.
