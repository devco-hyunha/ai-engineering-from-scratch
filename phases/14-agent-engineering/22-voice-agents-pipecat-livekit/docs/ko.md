# 음성 에이전트: Pipecat과 LiveKit

> 음성 에이전트는 2026년 생산 환경에서 일급 생산 카테고리로 자리 잡았습니다. Pipecat은 Python 기반의 프레임 파이프라인(VAD → STT → LLM → TTS → 전송)을 제공합니다. LiveKit Agents는 WebRTC를 통해 AI 모델과 사용자를 연결합니다. 프리미엄 스택의 경우 종단 간(end-to-end) 생산 지연 목표는 450–600ms 수준입니다.

**유형:** Learn
**언어:** Python (stdlib)
**선수 요건:** 14단계 · 01강 (에이전트 루프), 14단계 · 12강 (워크플로우 패턴)
**시간:** 약 60분

## 학습 목표

- Pipecat의 프레임 기반 파이프라인을 설명하세요: DOWNSTREAM (소스→싱크) 및 UPSTREAM (제어).
- 표준 음성 파이프라인 단계와 Pipecat이 지원하는 전송 방식을 나열하세요.
- LiveKit Agents의 두 가지 음성 에이전트 클래스(MultimodalAgent, VoicePipelineAgent)를 설명하고 각각이 적합한 상황을 서술하세요.
- 2026년 생산 환경의 지연(latency) 기대치를 요약하고, 이것이 아키텍처 선택에 어떻게 영향을 미치는지 설명하세요.

## 문제점

음성 에이전트는 TTS가 붙은 텍스트 루프가 아닙니다. 지연 예산은 매우 엄격하며(~600ms), 부분 오디오가 기본이고, 턴(turn) 감지는 모델이 담당하며, 전송 방식은 전화 SIP부터 WebRTC까지 다양합니다. 프레임 기반 파이프라인(Pipecat)을 구축하거나 플랫폼(LiveKit)에 의존해야 합니다.

## 개념

### Pipecat (pipecat-ai/pipecat)

- Python 기반의 프레임 파이프라인 프레임워크입니다.
- `Frame` → `FrameProcessor` 체인.
- 두 가지 흐름 방향:
  - **DOWNSTREAM** — 소스 → 싱크 (오디오 입력, TTS 출력).
  - **UPSTREAM** — 피드백 및 제어 (취소, 지표, 바지인(barge-in)).
- `PipelineTask`는 이벤트(`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`)와 지표/추적/RTVI용 옵저버를 통해 생명주기를 관리합니다.

일반적인 파이프라인:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

전송 방식: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows는 구조화된 대화(상태 머신)를 추가합니다. Pipecat Cloud는 관리형 런타임입니다.

### LiveKit Agents (livekit/agents)

- WebRTC를 통해 AI 모델과 사용자를 연결합니다.
- 핵심 개념: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- 두 가지 음성 에이전트 클래스:
  - **MultimodalAgent** — OpenAI Realtime 또는 동등한 기술을 통해 오디오를 직접 처리합니다.
  - **VoicePipelineAgent** — STT → LLM → TTS 캐스케이드; 텍스트 수준 제어 기능을 제공합니다.
- 트랜스포머 모델을 통한 시맨틱 턴 감지.
- 네이티브 MCP (모델 컨텍스트 프로토콜)(MCP (Model Context Protocol)) 통합.
- SIP를 통한 텔레포니.
- LiveKit Inference를 통해 API 키 없이 50개 이상의 모델을 사용하며, 플러그인을 통해 200개 이상을 추가로 사용할 수 있습니다.

### 상용 플랫폼

Vapi (최적화된 프리미엄 스택에서 약 450–600ms) 및 Retell (180개 테스트 호출에서 엔드투엔드 약 600ms)는 이러한 기술 위에 구축됩니다. WebRTC 팀 없이 관리형 음성 스택을 원할 경우 플랫폼을 선택하세요.

### 이 패턴이 잘못되는 지점

- **바지인(barge-in) 처리 없음.** 사용자가 중단해도 에이전트가 계속 말합니다. Pipecat의 UPSTREAM 취소 프레임 및 LiveKit의 동등한 기능이 필요합니다.
- **STT 신뢰도 무시.** 낮은 신뢰도의 전사(transcript)가 마치 진리인 것처럼 LLM에 전달됩니다. 신뢰도로 게이트를 설정하거나 확인을 요청하세요.
- **TTS 문장 중간 차단.** 파이프라인이 발화 중간에 취소되면 TTS가 이를 인지하거나 오디오를 잘라야 합니다.
- **레이턴시 예산 무시.** 모든 구성 요소가 50–200ms를 추가합니다. 출시 전에 전체 체인의 합계를 계산하세요.

### 2026년 전형적인 레이턴시

- VAD: 20–60ms
- STT 부분 처리: 100–250ms
- LLM 첫 토큰: 150–400ms
- TTS 첫 오디오: 100–200ms
- 전송 RTT: 30–80ms

엔드투엔드 450–600ms는 프리미엄입니다. 800–1200ms는 일반적입니다. 1500ms 이상은 고장 난 것처럼 느껴집니다.

```figure
voice-pipeline
```

## 구현하기

`code/main.py`는 다음을 포함하는 프레임 기반 장난감 파이프라인입니다:

- `Frame` 타입 (audio, transcript, text, tts_audio, control).
- `process(frame)`을 사용하는 `Processor` 인터페이스.
- 스크립트된 프로세서로서의 5단계 파이프라인 (VAD → STT → LLM → TTS → transport).
- 바지인(barge-in)을 시연하기 위한 UPSTREAM 취소 프레임.

실행하세요:

```
python3 code/main.py
```

추적(trace)은 정상 흐름과 발화 중간에 TTS를 멈추는 바지인(barge-in) 취소를 보여줍니다.

## 사용하기

- **Pipecat**는 완전한 제어 기능을 제공합니다 — 커스텀 프로세서, Python 우선, 플러그형 제공자.
- **LiveKit Agents**는 WebRTC 우선 배포 및 전화 통신에 적합합니다.
- **Vapi / Retell**는 WebRTC 팀 없이 호스팅된 음성 에이전트를 구축할 때 유용합니다.
- **OpenAI Realtime / Gemini Live**는 직접 오디오 입력/오디오 출력(MultimodalAgent)을 처리할 때 사용합니다.

## 출시하기

`outputs/skill-voice-pipeline.md`는 VAD + STT + LLM + TTS + 전송 및 바지인(barge-in) 처리를 포함하는 Pipecat 형태의 음성 파이프라인을 스캐폴딩합니다.

## 연습 문제

1. 장난감 파이프라인에 메트릭 옵저버를 추가해 보세요. 초당 각 단계의 프레임 수를 세어 보세요. 지연은 어디에서 누적되나요?
2. 신뢰도 게이트가 적용된 STT를 구현해 보세요. 임계값 미만일 경우 "다시 말씀해 주시겠어요?"를 요청합니다.
3. 시맨틱 턴 감지를 추가해 보세요. 간단한 규칙: 전사(transcript)가 "?"로 끝나면 턴이 종료됩니다.
4. Pipecat의 전송(docs) 문서를 읽어 보세요. stdlib 전송을 SmallWebRTCTransport 구성(스텁)으로 교체해 보세요.
5. 동일한 쿼리에서 OpenAI Realtime과 STT+LLM+TTS 캐스케이드를 측정해 보세요. 텍스트 수준 제어는 어떤 지연 비용을 발생시키나요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Frame | "이벤트" | 파이프라인 내의 타입이 지정된 데이터 단위 (오디오, 전사, 텍스트, 제어) |
| Processor | "파이프라인 단계" | process(frame)을 가진 핸들러 |
| DOWNSTREAM | "순방향 흐름" | 소스에서 싱크로: 오디오 입력, 음성 출력 |
| UPSTREAM |
| VAD | "음성 활동 감지" | 사용자가 말할 때를 감지합니다 |
| Semantic turn detection |
| MultimodalAgent |
| VoicePipelineAgent |

## 추가 읽기

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 프레임 기반 파이프라인, 프로세서, 전송
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + 음성 프리미티브
- [Vapi](https://vapi.ai/) — 관리형 음성 플랫폼
- [Retell AI](https://www.retellai.com/) — 관리형 음성, 지연 벤치마크됨
