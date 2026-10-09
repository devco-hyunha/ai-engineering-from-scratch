---
name: voice-agent
description: 800ms 미만의 첫 오디오 출력, 바지인(barge-in) 처리, 대화 중 도구 사용을 지원하는 실시간 음성 에이전트를 구축합니다.
version: 1.0.0
phase: 19단계
lesson: 03강
tags: [capstone, voice, webrtc, livekit, pipecat, asr, tts, streaming]
---

도메인(고객 지원, 예약, 리테일 어시스턴트)을 고려하여, 바지인(barge-in), 도구 호출, 패킷 손실을 처리하면서도 엔드투엔드 첫 오디오 출력을 800ms 미만으로 유지하는 WebRTC 음성 에이전트를 배포합니다.

구축 계획:

1. 마이크 오디오를 스트리밍하는 웹 클라이언트와 함께 LiveKit Agents 1.0 룸을 설정합니다. 전화 커버리지를 위해 Twilio PSTN 게이트웨이를 추가합니다.
2. 스트리밍 ASR (Deepgram Nova-3 호스트 또는 g5.xlarge에서 실행되는 faster-whisper Whisper-v3-turbo)을 실행합니다. 부분 및 최종 전사(transcript)를 구독합니다.
3. 20ms 프레임에 Silero VAD v5를 실행합니다. 발화 종료 시, LiveKit turn-detector로 최신 부분 전사를 점수화합니다. VAD 침묵이 500ms 이상이고 완료 점수가 0.6 이상일 때만 턴 완료(turn-complete)로 확정합니다.
4. LLM (GPT-4o-realtime, Gemini 2.5 Flash Live, 또는 캐스케이드된 Claude Haiku 4.5)을 스트리밍합니다. 첫 토큰을 200ms 이내에 TTS로 전달합니다.
5. TTS (Cartesia Sonic-2 또는 ElevenLabs Flash v3)를 스트리밍합니다. 첫 오디오 청크는 첫 LLM 토큰 생성 후 200ms 이내에 서버를 떠나야 합니다.
6. 바지인(barge-in): SPEAKING 또는 THINKING 중에 VAD가 새로운 사용자 발화를 감지하면, TTS를 취소하고 남은 LLM 출력을 버리며 ASR을 재장전(re-arm)합니다. `tts_canceled` span을 게시합니다.
7. 도구 사이드 채널: 함수 호출을 병렬로 실행합니다. 지연이 300ms를 초과하면, 오디오 스트림이 멈추지 않도록 인정 필러(acknowledgment filler)를 방출합니다.
8. 100건의 통화를 기록합니다. 홀드아웃(held-out) 전사와의 WER, Hamming VAD 벤치마크에서의 false-cutoff율, 첫 오디오 출력 p50, NISQA MOS, 3% 패킷 드롭에서의 거동을 측정합니다.
9. 단일 g5.xlarge에서 합성 호출자를 사용하여 50건의 동시 통화를 부하 테스트합니다. 지속되는 첫 오디오 출력 p95를 보고합니다.

평가 루브릭:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 엔드투엔드 지연 | 100건의 기록된 통화에서 p50 첫 오디오 출력이 800ms 미만 |
| 20 | 턴 테이킹(turn-taking) 품질 | Hamming VAD 벤치마크에서 false-cutoff율이 3% 미만 |
| 20 | 도구 사용 정확성 | 대화 중 도구 호출이 오디오를 지연시키지 않고 올바른 데이터를 반환합니다 |
| 20 | 패킷 손실 환경에서의 신뢰성 | 3% 패킷 드롭이 주입된 상태에서 WER 및 턴 테이킹(turn-taking) 안정성 |
| 15 | 평가 하네스 완성도 | 공개된 설정으로 재현 가능한 측정 |

하드 리젝트(Hard Rejects):

- 비스트리밍 파이프라인(배치 ASR, 배치 TTS)은 지연 시간(latency) 목표를 달성할 수 없습니다.
- TTS 버퍼를 즉시 취소하지 않는 모든 바지인(barge-in) 정책은 거부됩니다. 지연된 취소는 최악의 사용자 경험(UX) 퇴행을 유발합니다.
- LLM 스트림을 동기적으로 차단하는 도구 호출은 거부됩니다. 도구 호출은 사이드 채널(side channel)에서 실행되어야 합니다.

거부 규칙:

- VAD 또는 턴 디텍터(turn-detector) 없이 배포하는 것을 거부합니다. 고정 시간아웃(turn-taking) 방식은 용인할 수 없는 컷오프(cutoff)율을 발생시킵니다.
- MOS가 인간이 평가한 것인지 NISQA로 근사한 것인지 문서화하지 않은 채 MOS를 보고하는 것을 거부합니다.
- 최소 100건의 기록된 호출과 호출 추적(call traces)을 공개하지 않은 채 "X 이하의 p50 지연 시간"을 보고하는 것을 거부합니다.

출물: LiveKit 에이전트 워커, PSTN 게이트웨이 설정, 100건 호출 평가 하네스, 공개된 Langfuse 음성 대시보드, 호스팅된 경쟁사(Retell, Vapi, 또는 OpenAI Realtime API 직접)와의 나란 비교(side-by-side comparison), 그리고 관찰된 세 가지 가장 큰 턴 테이킹(turn-taking) 실패와 각각을 해결한 디텍터 튜닝에 대한 보고서가 포함된 저장소입니다.
