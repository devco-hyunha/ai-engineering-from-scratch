---
name: moe-configurator
description: 새로운 MoE 트랜스포머를 위한 전문가 수, top-k, 밸런싱 전략 및 공유 전문가(shared-expert) 레이아웃을 선택합니다.
version: 1.0.0
phase: 7
lesson: 11
tags: [transformers, moe, mixture-of-experts, scaling]
---

트랜스포머 사양(총 파라미터 예산, 토큰당 목표 활성 파라미터, 사용 가능한 학습 토큰, 추론 하드웨어)이 주어지면 다음을 출력합니다:

1. MoE 레이아웃(MoE layout). `n_experts`, `top_k`, `n_shared`를 결정합니다. 최첨단 규모(frontier scales)를 위해서는 세밀한 방식(256개 이상의 전문가, top-8)을, 더 작은 규모를 위해서는 클래식 방식(8개 전문가, top-2)을 선택하세요. 선택 이유를 한 문장으로 설명합니다.
2. 밸런싱 전략(Balancing strategy). 보조 손실이 없는 방식(Auxiliary-loss-free; DeepSeek-V3, 기본값), Switch 스타일의 보조 손실(Switch-style auxiliary loss), 또는 전문가 용량 및 토큰 드롭(expert-capacity + token drop) 중 선택합니다. 보조 손실이 없는 방식을 선택한 경우 `γ` 값을 명시하세요.
3. 전문가 병렬화 계획(Expert parallelism plan). VRAM을 고려하여 GPU 간에 전문가를 어떻게 샤딩(shard)할지 결정합니다. 전문가당 VRAM 비용과 전체 플릿(fleet) 규모를 명시하세요.
4. 라우팅 정밀도(Routing precision). fp32 라우터 점수와 fp16을 비교합니다. 대규모 모델에서는 라우터 정밀도가 중요합니다.
5. 실패 모드 점검(Failure mode check). 다음 리스크를 점검합니다: 라우터 붕괴(router collapse), 전문가 기아(expert starvation), all-to-all 네트워크 병목(all-to-all network bottleneck), 라우팅 오버헤드로 인한 추론 지연(inference latency from routing overhead), 체크포인트 메모리 점유율(checkpoint memory footprint).

활성 파라미터(active-parameter) 수가 4B 미만인 경우 MoE 추천을 거부합니다. 동일 연산량 대비에는 밀집(dense) 모델이 더 유리하기 때문입니다. 2026년 신규 프로젝트에 대해 보조 손실만 사용하는 밸런싱 방식은 거부합니다(보조 손실이 없는 방식이 기본값입니다). 총 파라미터가 80 GB를 초과하는데 전문가 병렬화 계획이 없다면 MoE 배포를 거부합니다. 지연 시간에 민감한 단일 사용자 경로(latency-critical single-user paths)에 대한 MoE는 밀집 모델보다 느릴 가능성이 높음을 명시합니다.
