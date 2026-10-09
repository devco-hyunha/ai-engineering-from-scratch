---
name: realtime-voice-pipeline
description: 목표 엔드투엔드 지연 시간을 위해 전송, VAD, 스트리밍 STT, LLM, 스트리밍 TTS, 오케스트레이션 선택
version: 1.0.0
phase: 6단계
lesson: 11강
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

목표 (지연 시간 P50/P95, 언어, 채널, 오프라인 vs 클라우드, 통화량)가 주어지면 다음을 출력합니다:

1. 전송. WebRTC (LiveKit / Daily) · WebSocket · SIP 트렁킹 (Twilio / Telnyx). 지터 허용 범위 + 사용 사례와 연결된 이유.
2. VAD + 턴 테이킹. Silero VAD (오픈, TPR 99.5%) · Cobra (상용) · LiveKit 턴 감지. 임계값, 최소 발화 지속 시간, 침묵 유지 시간.
3. 스트리밍 STT. Parakeet TDT (가장 빠른 오픈) · Kyutai STT (플러시 트릭 포함) · Deepgram Nova-3 (API, ~150 ms) · Whisper-streaming. 이유.
4. LLM + 스트리밍. TTS가 시작되기 전에 첫 20개 토큰을 고정합니다. 모델 + 스트리밍 구성 + 프롬프트 인젝션(Prompt Injection)에 대한 가드레일(Guardrails).
5. 스트리밍 TTS. Kokoro-82M (~100 ms TTFA) · Orpheus · Cartesia Sonic · ElevenLabs Turbo. 보이스 팩 또는 클로닝 가드 (8강).
6. 오케스트레이션. LiveKit Agents · Pipecat · Vapi · Retell · 커스텀 Rust. 팀 기술 + 규모와 연결된 이유.
7. 관측 가능성(Observability). 단계별 P50/P95/P99 히스토그램; 오탐 중단율; 통화 드롭율; 통화 샘플의 WER.

STT 전에 전체 발화를 버퍼링하는 배포를 거부합니다. 스트리밍하지 않는 TTS를 거부합니다. 평균 지연 시간으로 평가하는 것을 거부하고 P95를 요구합니다. 월 10만 분 이상 사용 시 자체 구축 비용 비교 없이 관리형 플랫폼(Vapi / Retell)을 거부합니다.

예시 입력: "자동차 보험 견적용 음성 에이전트. P95 &lt; 500 ms. 영어, 미국. 주간 5만 분. 컴플라이언스: HIPAA 관련 (로그에 PII 없음)."

예시 출력:
- 전송: LiveKit Agents + Twilio SIP. 콜센터 규모에서 검증됨, HIPAA 모드 옵트인.
- VAD: Silero VAD @ 임계값 0.45, 최소 발화 220 ms, 침묵 유지 시간 400 ms. LiveKit 턴 감지 오버레이.
- STT: Deepgram Nova-3 영어 (~150 ms P95); 온프레미스 감사 필요 시 Parakeet-TDT로 폴백.
- LLM: OpenAI realtime API를 통한 GPT-4o 스트리밍; 포스트 필터로 프롬프트 인젝션(Prompt Injection) 방지; TTS에 첫 20개 토큰 고정.
- TTS: Cartesia Sonic 2 (~150 ms TTFA, 음성 클로닝 미사용 — 사전 정의된 음성).
- 오케스트레이션: LiveKit Agents. 프로덕션 환경에서는 Hamming AI를 통해 관측 가능성(Observability)을 확보합니다.
- 로그: CVV / SSN / DOB를 정규식 + NER 패스로 제거한 후 저장합니다. 30일간 보관합니다.
