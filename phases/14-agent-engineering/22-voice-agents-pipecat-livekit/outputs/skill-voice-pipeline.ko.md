---
name: voice-pipeline
description: 바지인(barge-in), 신뢰도 게이팅(confidence gating), 지연 시간 예산 준수를 포함하여 VAD + STT + LLM + TTS + 전송(trans) 구조의 Pipecat형 음성 파이프라인을 스캐폴딩합니다.
version: 1.0.0
phase: 14단계
lesson: 22강
tags: [voice, pipecat, livekit, webrtc, latency]
---

음성 제품 사양(언어, 전송, 제공자)이 주어지면 프레임 기반 파이프라인을 스캐폴딩합니다.

생성물:

1. `Frame` 타입과 `kind`, `payload`, `direction` (다운스트림 / 업스트림)를 생성합니다.
2. 프로세서: `VAD`, `STT`, `LLM`, `TTS`, `Transport`. 각 프로세서에 `process(frame)`가 포함됩니다.
3. `link()` 헬퍼는 프로세서를 순방향 및 역방향으로 연결합니다.
4. 취소 프레임 처리: 전송(trans)에서 TTS, LLM, STT로 이어지는 UPSTREAM 경로를 통해 각 단계에서 대기 중인 작업을 드롭(dropping)합니다.
5. 관찰자: 단계별 지연 시간 지표; 프로세서를 통과하는 각 프레임에 대해 OTel span을 방출합니다(23강).
6. STT의 신뢰도 게이트: 임계값 미만일 경우, 전사(transcript) 대신 "다시 말해 주세요" 텍스트 프레임을 방출합니다.

하드 리젝트(Hard rejects):

- UPSTREAM 처리가 없는 파이프라인. 음성에서는 바지인(barge-in)이 선택 사항이 아닙니다.
- 스트리밍 없는 LLM 호출. 첫 토큰 지연 시간이 지배적이므로 반드시 스트리밍해야 합니다.
- 신뢰도-blind STT. 잘못된 전사를 LLM에 전달하면 잘못된 응답이 생성됩니다.

거부 규칙:

- 콜드 런(cold run)에서 엔드투엔드 지연 시간이 1500ms를 초과하면 출시를 거부합니다. 체인을 최적화하거나 MultimodalAgent(LiveKit direct-audio)를 사용하세요.
- 제품이 텔레포니 우선이고 파이프라인에 SIP 어댑터가 없으면 거부하세요. LiveKit SIP 또는 플랫폼(Vapi/Retell)을 통해 라우팅하세요.
- 제품이 전송 중 암호화되지 않은 PII 오디오를 포함하면 거부하세요.

출력: `frames.py`, `processors.py`, `pipeline.py`, `observers.py`, `README.md`. 지연 시간 예산, 바지인(barge-in) 설계, 전송 선택에 대해 설명합니다. "다음에 읽을 내용"으로 끝내며, OTel(23강), 관측 가능성 백엔드(24강), 또는 WebRTC 세부 사항에 대한 LiveKit 문서를 가리킵니다.
