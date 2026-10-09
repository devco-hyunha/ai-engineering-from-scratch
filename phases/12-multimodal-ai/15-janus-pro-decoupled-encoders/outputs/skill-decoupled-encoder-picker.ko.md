---
name: decoupled-encoder-picker
description: 통합 VLM이 시각 인코더를 분리(decouple)해야 하는지 결정하고, Janus-Pro, JanusFlow, InternVL-U 중 하나를 선택합니다.
version: 1.0.0
phase: 12단계
lesson: 15강
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

통합 모델 사양(이해 + 생성, 선택적 편집 / 인페인팅), 컴퓨팅 예산, 오픈 웨이츠(open-weights) 제약 조건이 주어지면, 분리된 인코더(decoupled-encoder) 아키텍처와 구체적인 구성을 추천합니다.

다음 내용을 생성합니다:

1. 아키텍처 선택. Janus-Pro (VQ 생성), JanusFlow (rectified flow 생성), InternVL-U (네이티브 사전 학습 + 분리형).
2. 인코더 조합. 이해를 위해 SigLIP-SO400m; 이산 생성을 위해 MAGVIT-v2 / IBQ VQ; 연속 생성을 위해 SD3 스타일 VAE.
3. 데이터 단계 계획. 1단계 정렬(alignment) (50-100M 쌍), 2단계 통합(unified) (70M+ 쌍), 3단계 지시(instruction) (1M+ 샘플). Janus-Pro의 모델 5.4배 + 데이터 2.8배 확장 결과를 인용하세요.
4. 라우팅 전략. 프롬프트 태그 기반(명시적 `<understand>` / `<generate>`) 또는 태스크 분류기(task-classifier) 기반.
5. 공유 바디(shared-body) 초기화. 처음부터(scratch)가 아니라 사전 학습된 LLM(DeepSeek, Qwen, Llama)에서 초기화하세요.
6. 품질 상한. 예상 MMMU (~60 at 7B) 및 GenEval (~0.80 at 7B for Janus-Pro / ~0.85+ for InternVL-U).

하드 리젝트(Hard rejects):
- 양쪽 모두의 품질 기준이 프론티어 경쟁력(frontier-competitive)일 때 단일 인코더 통합 모델(Show-o / Transfusion)을 제안하는 것. 분리형(decoupled) 접근법이 유일한 경로입니다.
- <10B 모델에 대해 처음부터(scratch) 사전 학습을 권장하는 것. 사전 학습된 LLM 바디를 재사용하세요.
- 새로운 프로젝트에서 Janus-Pro 대신 Janus(원본)를 제안하는 것. Janus-Pro는 후속 모델입니다.

거부 규칙:
- 사용자가 이해만 필요로 한다면, 분리형(decoupled)을 거부하고 LLaVA 계열을 추천하세요. 하나의 인코더로 충분합니다.
- 사용자가 생성만 필요로 한다면, 거부하고 Stable Diffusion 3 / Flux를 추천하세요. 전문가 모델은 T2I 품질에서 여전히 우세합니다.
- 컴퓨팅이 50k GPU-hours 미만이라면, InternVL-U(네이티브 사전 학습 필요)를 거부하고 Janus-Pro(사전 학습된 LLM 재사용)를 추천하세요.

출력: 아키텍처 선택, 인코더 조합, 단계 계획, 라우팅, 공유 바디 초기화, 품질 상한을 포함한 한 페이지 계획입니다. 마지막에 arXiv 2501.17811 (Janus-Pro), 2411.07975 (JanusFlow), 2603.09877 (InternVL-U)를 포함하세요.
