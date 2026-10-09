---
name: omni-streaming-budget
description: 목표 TTFAB 및 기능 세트에 맞춰 Thinker-Talker 스트리밍 음성 파이프라인(Qwen-Omni / Moshi / Mini-Omni)의 규모를 산정합니다.
version: 1.0.0
phase: 12단계
lesson: 20강
tags: [qwen-omni, moshi, mini-omni, streaming, ttfab, thinker-talker]
---

음성 우선 제품 사양(목표 TTFAB, 마이크 샘플링 속도, 비전 포함 여부, 이중 언어, 풀 듀플렉스)과 컴퓨팅 제약(GPU 등급, 예산)이 주어지면 Thinker-Talker 파이프라인의 규모를 산정해 보세요.

산출물:

1. 모델 계열 선택. Moshi(최적의 지연 시간), Qwen2.5-Omni(최적의 오픈 기능), Qwen3-Omni(최신 품질), Mini-Omni(가장 단순).
2. Thinker 및 Talker 크기. TTFAB가 400ms 미만인 경우 7B Thinker + 200-300M Talker. 품질을 우선시하는 경우 70B+ Thinker를 사용하되 더 높은 TTFAB를 허용해야 합니다.
3. TTFAB 세부 분석. 구성 요소별 지연 시간 추정.
4. 듀플렉스 모드. VAD(turn-taking)를 사용한 반 듀플렉스를 기본으로 설정; 제품이 백채널(backchannel)을 요구하는 경우 풀 듀플렉스 사용.
5. 비전 통합. 인터리브된(interleaved) 비디오 프레임을 위해 절대 타임스탬프를 사용하는 TMRoPE.
6. 배포 형태. 처리량 요구 사항에 따라 단일 GPU 방식과 분리 방식(Thinker는 A에, Talker는 B에) 중 하나를 선택합니다.

허용 불가 조건:
- 70B Talker 제안. Talker는 음성 토큰 속도를 따라잡기 위해 작아야 합니다.
- 비스트리밍(non-streaming) 음성 디코더 사용. TTFAB가 급증합니다.
- 풀 듀플렉스가 플러그 앤 플레이(plug-and-play)라고 주장. 이는 특수한 학습 데이터를 필요로 합니다.

거절 규칙:
- 목표 TTFAB가 200ms 미만인 경우, 단일 A100에서 Moshi급(7B fused)보다 큰 모델은 거절합니다.
- 제품이 스트림 내 음악 생성을 요구하는 경우, 이 아키텍처를 거절하고 별도의 음악 파이프라인을 권장합니다.
- 마이크 샘플링 속도가 48kHz이고 엄격한 품질을 요구하는 경우, 더 강력한 음성 인코더가 필요함을 표시합니다. 무분별하게 다운샘플링하지 마세요.

출력: 모델 선택, 크기, TTFAB 세부 분석, 듀플렉스 모드, 비전 전략, 배포를 포함한 한 장의 스트리밍 계획. arXiv 2503.20215 (Qwen2.5-Omni), 2410.00037 (Moshi)로 마무리합니다.
