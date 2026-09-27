---
name: realtime-voice-pipeline
description: Pick transport, VAD, streaming STT, LLM, streaming TTS, and orchestration for a target end-to-end latency.
version: 1.0.0
phase: 6
lesson: 11
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

목표(지연 시간 P50/P95, 언어, 채널, 오프라인 vs 클라우드, 통화량)가 주어지면, 다음 항목을 출력하세요:

1. 전송(Transport): WebRTC (LiveKit / Daily), WebSocket, SIP trunking (Twilio / Telnyx). 지터 허용 범위(jitter tolerance) 및 사용 사례와 연계된 근거를 제시하세요.
2. VAD + 턴 테이킹(turn-taking): Silero VAD (오픈 소스, 99.5% TPR), Cobra (상용), LiveKit turn-detector. 임계값(Threshold), 최소 발화 지속 시간(min speech duration), 침묵 유지 시간(silence hang-over)을 포함하세요.
3. 스트리밍 STT(Streaming STT): Parakeet TDT (가장 빠른 오픈 소스), Kyutai STT (flush trick 적용), Deepgram Nova-3 (API, ~150 ms), Whisper-streaming. 선정 근거를 제시하세요.
4. LLM + 스트리밍(LLM + streaming): TTS가 시작되기 전 첫 20개 토큰을 고정(pin)하세요. 모델, 스트리밍 설정, 프롬프트 인젝션 방지를 위한 가드레일을 포함하세요.
5. 스트리밍 TTS(Streaming TTS): Kokoro-82M (~100 ms TTFA), Orpheus, Cartesia Sonic, ElevenLabs Turbo. 보이스 팩 또는 클로닝 가드(Lesson 8 참조)를 포함하세요.
6. 오케스트레이션(Orchestration): LiveKit Agents, Pipecat, Vapi, Retell, custom Rust. 팀의 기술 스택 및 확장성과 연계된 근거를 제시하세요.
7. 관측 가능성(Observability): 단계별 P50/P95/P99 히스토그램, 오탐(false-positive) 중단율, 통화 드롭률, 통화 샘플에 대한 WER(Word Error Rate).

STT 이전에 전체 발화를 버퍼링하는 방식은 제외하세요. 스트리밍을 지원하지 않는 TTS는 제외하세요. 평균 지연 시간(average latency) 기준의 평가는 제외하고, 반드시 P95를 기준으로 하세요. 월 10만 분 이상의 통화량에 대해 직접 구축(build-your-own) 방식과의 비용 비교 없이 관리형 플랫폼(Vapi / Retell)만 사용하는 것은 제외하세요.

입력 예시: "자동차 보험 견적용 음성 에이전트. < 500 ms P95. 영어, 미국. 주당 5만 분. 컴플라이언스: HIPAA 준수 필요 (로그에 PII 포함 금지)."

출력 예시:
- 전송(Transport): LiveKit Agents + Twilio SIP. 콜센터 규모에서 검증됨, HIPAA 모드 옵트인 가능.
- VAD: Silero VAD @ 임계값 0.45, 최소 발화 220 ms, 침묵 유지 400 ms. LiveKit turn-detector 오버레이 적용.
- STT: Deepgram Nova-3 영어 (~150 ms P95); 온프레미스 감사가 필요한 경우 Parakeet-TDT로 폴백(fall-back).
- LLM: OpenAI realtime API를 통한 GPT-4o 스트리밍; 포스트 필터(post-filter)로 프롬프트 인젝션 방지; 첫 20개 토큰을 TTS로 고정.
- TTS: Cartesia Sonic 2 (~150 ms TTFA, 보이스 클로닝 미사용 — 사전 정의된 보이스 사용).
- 오케스트레이션(Orchestration): LiveKit Agents. 프로덕션 환경을 위해 Hamming AI를 통한 관측 가능성 확보.
- 로그(Logs): 영구 저장 전 정규식(regex) + NER 패스를 통해 CVV / SSN / DOB 제거. 30일간 보관.
