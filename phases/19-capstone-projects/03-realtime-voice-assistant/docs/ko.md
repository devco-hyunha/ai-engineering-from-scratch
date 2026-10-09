# 캡스톤 03 — 실시간 음성 어시스턴트 (ASR에서 LLM, TTS로)

> 자연스럽게 느껴지는 음성 에이전트는 엔드투엔드 지연이 800ms 미만이어야 하며, 사용자가 말을 멈추는 시점을 파악하고, 바지인(barge-in)을 처리하며, 도구 호출 시 멈추지 않아야 합니다. Retell, Vapi, LiveKit Agents, Pipecat는 2026년 모두 이 기준을 충족합니다. 이들은 동일한 구조로 이를 달성합니다: 스트리밍 ASR, 턴 감지(turn-detector), 스트리밍 LLM, 스트리밍 TTS를 WebRTC로 연결하고 각 단계에서 공격적인 지연 예산을 적용합니다. 하나를 구축하고 WER, MOS, 오탐 컷오프(false-cutoff)율을 측정하며, 패킷 손실 환경에서 실행해 보세요.

**유형:** Capstone
**언어:** Python (에이전트 + 파이프라인), TypeScript (웹 클라이언트)
**선수 요건:** 6단계 (음성 및 오디오), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 13단계 (도구), 14단계 (에이전트), 17단계 (인프라)

**활용 단계:** P6 · P7 · P11 · P13 · P14 · P17
**시간:** 30시간

## 문제점

음성은 2025-2026년 AI UX 카테고리 중 가장 빠르게 변화하는 분야입니다. 기술적 상한선은 매 분기마다 낮아졌습니다. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, Pipecat 0.0.70는 모두 첫 오디오 출력까지의 시간을 800ms 미만으로 만들 수 있습니다. 기준은 지연 시간만이 아닙니다. 상호작용의 느낌입니다: 사용자를 끊지 않고, 끊기지 않으며, 문장 중의 중단에서 회복하고, 오디오를 멈추지 않고 대화 중 도구를 호출하며, 불안정한 모바일 네트워크를 견디는 것입니다.

세 개의 REST 호출을 이어 붙여서는 도달할 수 없습니다. 아키텍처는 엔드투엔드 파이프라인 스트리밍이어야 합니다. 이를 구축하면 실패 모드들이 드러납니다: 전화 오디오에 맞춰 튜닝된 VAD가 배경 TV 소리에 반응하는 경우, 영구히 오지 않는 문장 부호를 기다리는 턴 감지(turn-detector), 방출하기 전 400ms를 버퍼링하는 TTS. 캡스톤은 부하 하에서 이러한 문제들을 하나씩 수정하고 지연 및 품질 보고서를 공개하는 것입니다.

## 개념

파이프라인은 다섯 개의 스트리밍 단계로 구성됩니다: **오디오 입력** (브라우저나 PSTN에서 WebRTC로), **ASR** (Deepgram Nova-3나 faster-whisper에서 스트리밍 부분 전사), **턴 감지** (VAD와 부분 전사에서 완료 단서를 읽는 작은 턴 감지(turn-detector) 모델), **LLM** (턴이 완료된 것으로 판단되면 즉시 토큰을 스트리밍), **TTS** (첫 LLM 토큰 후 약 200ms 내에 오디오를 스트리밍 출력).

세 가지 교차 관심사. **바지인(Barge-in)**: 에이전트가 말하는 동안 사용자가 말하기 시작하면, TTS가 취소되고 ASR이 즉시 음성을 인식합니다. **도구 사용**: 대화 중 함수 호출(날씨, 캘린더)은 오디오를 지연시키지 않도록 사이드 채널에서 실행되어야 합니다. 지연이 300ms를 초과하면 에이전트가 확인 토큰("1초만...")을 미리 생성합니다. **백프레셔(Backpressure)**: 패킷 손실 시 부분 전사(transcript)가 유지되고, VAD는 음성 게이트 임계값을 높이며, 에이전트는 확인되지 않은 메시지 위에 말하지 않도록 합니다.

측정 기준은 정량적입니다. 15 dB SNR에서 Hamming VAD 벤치마크 기준 WER가 8% 미만이어야 합니다. 측정된 100건의 통화에서 첫 오디오 출력(first-audio-out) p50이 800ms 미만이어야 합니다. 오타 컷오프(false-cutoff)율이 3% 미만이어야 합니다. TTS의 MOS가 4.2 이상이어야 합니다. 단일 g5.xlarge 인스턴스에서 50건의 동시 통화를 처리해야 합니다. 이 수치들이 산출물입니다.

## 아키텍처

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 스택

- 전송: LiveKit Agents 1.0 (WebRTC) 및 Twilio PSTN 게이트웨이; 대안 프레임워크로 Pipecat 0.0.70
- ASR: Deepgram Nova-3 (스트리밍, 첫 부분 인식까지 300ms 미만) 또는 자가 호스팅된 faster-whisper Whisper-v3-turbo
- VAD: Silero VAD v5 및 LiveKit 턴 디텍터(부분 전사를 읽는 소형 트랜스포머)
- LLM: 긴밀한 통합을 위한 OpenAI GPT-4o-realtime, Gemini 2.5 Flash Live, 또는 캐스케이드된 Claude Haiku 4.5 (스트리밍 완료, 별도 오디오 경로)
- TTS: Cartesia Sonic-2 (최저 첫 바이트 지연), ElevenLabs Flash v3, 또는 자가 호스팅용 오픈소스 Orpheus
- 도구: 날씨/캘린더/예약용 FastMCP 사이드 채널; 도구가 300ms 이상 소요되면 에이전트가 필러(filler)를 미리 방출합니다
- 관측 가능성: OpenTelemetry 음성 스팬, 오디오 재생이 포함된 Langfuse 음성 추적
- 배포: 자가 호스팅된 Whisper + Orpheus용 단일 g5.xlarge (24GB VRAM); 최저 지연을 위한 호스팅 API

```figure
ce-voice-latency
```

## 구현하기

1. **WebRTC 세션.** LiveKit 방과 마이크 오디오를 스트리밍하는 웹 클라이언트를 설정합니다. 서버에서는 방에 참여하는 에이전트 워커를 연결합니다.

2. **ASR 스트리밍.** 20ms PCM 프레임을 Deepgram Nova-3 (또는 GPU의 faster-whisper)에 공급합니다. 부분 및 최종 전사를 구독하고 부분 전사별 지연을 기록합니다.

3. **VAD 및 턴 감지.** 프레임 스트림에서 Silero VAD v5를 실행하세요. 발화 종료 이벤트가 발생하면 최신 부분 전사(transcript)에 대해 LiveKit 턴 감지기를 실행하세요. VAD가 500ms 동안 침묵을 감지하고 턴 감지기가 완료 점수 0.6 이상을 기록할 때만 "턴 완료"로 확정하세요.

4. **LLM 스트림.** 턴이 완료되면 진행 중인 대화와 최종 전사를 포함하여 LLM 호출을 시작하세요. 토큰을 스트리밍하세요. 첫 번째 토큰이 나오면 TTS로 핸드오프하세요.

5. **TTS 스트림.** Cartesia Sonic-2가 오디오 청크를 스트리밍하여 반환합니다. 첫 번째 청크는 첫 번째 LLM 토큰이 나온 후 200ms 이내에 서버를 떠나야 합니다. 청크를 LiveKit 룸으로 전송하세요. 클라이언트는 WebRTC 지터 버퍼를 통해 재생합니다.

6. **바지인(Barge-in).** TTS가 재생되는 동안 VAD가 새로운 사용자 발화를 감지하면 TTS 스트림을 즉시 취소하고, 남은 LLM 출력을 버리며, ASR을 다시 준비하세요. `tts_canceled` 스팬(span)을 발행하세요.

7. **도구 사이드 채널.** 날씨와 캘린더를 함수 호출 도구로 등록하세요. 호출이 발생하면 동시에 실행하세요. 300ms 내에 해결되지 않으면 LLM이 "잠깐만요, 확인해 볼게요"라는 필러(filler)를 출력하도록 하세요. 도구가 반환되면 재개하세요.

8. **평가 하네스.** 100건의 호출을 기록하세요. WER(보유 전사와 비교), 거짓 컷오프율(사용자가 문장 중간에 있을 때 TTS가 취소된 비율), 첫 오디오 출력 p50, TTS MOS(인간 또는 NISQA), 지터 손실 테스트(패킷 3% 드롭)를 계산하세요.

9. **부하 테스트.** 단일 g5.xlarge 인스턴스에서 합성 호출자를 통해 50건의 동시 호출을 구동하세요. 지속되는 첫 오디오 출력 p95를 측정하세요.

## 사용하기

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 출시하기

`outputs/skill-voice-agent.md`가 산출물입니다. 도메인(고객 지원, 예약, 키오스크)이 주어지면 ASR/VAD/LLM/TTS 파이프라인이 측정 기준에 맞춰 조정된 LiveKit 에이전트를 구축하세요. 채점 기준:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 엔드투엔드 지연 | 100건의 기록된 호출에서 첫 오디오 출력 p50이 800ms 미만 |
| 20 | 턴 테이킹 품질 | Hamming VAD 벤치마크에서 거짓 컷오프율 3% 미만 |
| 20 | 도구 사용 정확성 | 오디오가 멈추지 않고 올바른 데이터를 반환하는 대화 중 도구 호출 |
| 20 | 패킷 손실 시 신뢰성 | 3% 패킷 드롭이 주입된 상태에서 WER 및 턴 테이킹 안정성 |
| 15 | 평가 하네스 완성도 | 공개된 설정으로 재현 가능한 측정 |
| **100** | | |

## 연습 문제

1. g5.xlarge에서 Deepgram Nova-3을 faster-whisper v3 turbo로 교체해 보세요. 지연 시간과 WER 격차를 측정하고, CPU 대 GPU 결정이 중요한 부분을 식별해 보세요.

2. 인터럽션 중재 정책을 추가해 보세요. 도구 호출 중 사용자가 끼어들 때 에이전트가 어떻게 동작하는지 비교해 보세요. 세 가지 정책(하드 취소, 도구 완료 후 정지, 다음 턴 큐잉)을 비교합니다.

3. 적대적 턴 감지 테스트를 실행해 보세요. 사용자가 문장 중간에 긴 휴식을 취하도록 설정합니다. VAD 침묵 임계값과 턴 감지 점수 임계값을 조정하여 900ms를 초과하지 않으면서 가장 낮은 오탐(cutoff)을 달성하도록 튜닝해 보세요.

4. Twilio를 통해 PSTN에 동일한 에이전트를 배포해 보세요. PSTN의 첫 오디오 출력(first-audio-out)을 WebRTC와 비교하고, 지터 버퍼와 코덱의 차이점을 설명해 보세요.

5. 비영어 언어(일본어, 스페인어)에 대한 음성 활동 감지(VAD)를 추가해 보세요. 언어별 미세 조정(fine-tune)과 비교하여 Silero VAD v5의 오탐(false-trigger)률을 측정해 보세요.

## 핵심 용어

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 턴 감지(Turn detection) | "발화 종료" | VAD 침묵과 부분 전사(transcript)를 입력으로 받아 사용자가 발화를 완료했는지 결정하는 분류기 |
| 끼어들기(Barge-in) | "인터럽션 처리" | VAD가 새로운 사용자 음성을 감지했을 때 TTS 재생을 중단하는 것 |
| 첫 오디오 출력(First-audio-out) | "지연 시간(Latency)" | 사용자가 발화를 멈추는 시점부터 서버에서 첫 오디오 패킷이 나가는 시점까지의 시간 |
| VAD | "음성 게이트" | 오디오 프레임을 음성 대 침묵으로 분류하는 모델; Silero VAD v5는 2026년 기본값 |
| 지터 버퍼(Jitter buffer) | "오디오 스무딩" | 네트워크 변동을 흡수하기 위해 패킷을 잠시 보관하는 클라이언트 측 버퍼 |
| 필러(Filler) | "확인 토큰" | 도구가 느릴 때 침묵을 피하기 위해 에이전트가 출력하는 짧은 문구 |
| MOS | "평균 의견 점수(Mean opinion score)" | 지각된 음성 품질 평가; NISQA는 자동화된 대리 지표 |

## 추가 읽기

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 참조 WebRTC 에이전트 프레임워크
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 대체 Python 우선 스트리밍 에이전트 프레임워크
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 통합 음성 모델 참조
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 스트리밍 ASR 참조
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 참조 모델
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 저지연 TTS 참조
- [Retell AI architecture](https://docs.retellai.com) — 프로덕션 음성 에이전트 아키텍처
- [Vapi.ai production stack](https://docs.vapi.ai) — 대체 프로덕션 참조
