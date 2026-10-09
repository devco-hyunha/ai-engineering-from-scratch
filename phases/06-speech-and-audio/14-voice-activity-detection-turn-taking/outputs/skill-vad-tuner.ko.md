---
name: vad-tuner
description: 음성 에이전트(Agent)를 위해 VAD 모델, 임계값, 무음 유지 시간, 프리롤, 턴 감지 전략을 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 14강
tags: [vad, silero, cobra, turn-detection, flush-trick]
---

작업 부하(소비자 / 콜센터 / 엣지 / 접근성; 잡음 프로필; 언어 혼합; 지연)를 고려하여 다음을 출력합니다:

1. VAD. Silero VAD (기본값) · Cobra (상용 정확도) · pyannote segmentation (화자 분리 등급) · WebRTC VAD (레거시 / 소형). 한 문장 이유를 포함합니다.
2. 매개변수(Parameter). 임계값(0.3-0.5), 최소 발화 시간(200-300 ms), 무음 유지 시간(400-800 ms), 프리롤(250-500 ms).
3. 시맨틱 턴 감지. 활성화(LiveKit turn-detector 또는 사용자 정의 MLP) 또는 비활성화. 예상 사용자 발화 패턴과 연결된 이유를 포함합니다.
4. 플러시 트릭. 활성화(STT가 지원하는 경우 — Kyutai / Deepgram) 또는 비활성화. 예상 지연 절감량을 포함합니다.
5. 가드레일(Guardrails). 최소 시간보다 짧은 발화를 거부합니다. 항상 프리롤을 유지합니다. 사용자별 무음 유지 시간 오버라이드를 제한합니다. VAD 서비스가 다운되면 fail-open합니다(모든 것을 발화로 취급합니다).

프로덕션 환경에서는 에너지 기반 VAD만 사용하는 것을 거부합니다 — 잡음이 너무 많습니다. 무음 유지 시간을 0으로 설정하는 것을 거부합니다 — 사용자를 끊게 됩니다. 전용 Silero가 사용 가능한 경우 Whisper 기반 VAD를 거부합니다(더 느리고 정확도가 낮습니다).

예시 입력: "항공사 재예약용 콜센터 IVR. 잡음 많은 배경(공항). 영어 + 스페인어. &lt; 500 ms 턴 감지."

예시 출력:
- VAD: 잡음 저항성 이점 때문에 Cobra (상용)를 선택합니다. 비용이 prohibitive(너무 높)인 경우 Silero로 폴백합니다.
- 매개변수(Parameter): 임계값 0.4 (공항 잡음 바닥이 높음); 최소 발화 시간 300 ms; 무음 유지 시간 600 ms (사용자는 IVR 중 비행편 번호를 읽기 위해 자주 멈추는 경우가 많음); 프리롤 400 ms.
- 시맨틱 턴: LiveKit turn-detector 활성화 — 문장 중 멈춤이 흔합니다("내 비행편을 바꿔야 해... 내일로").
- 플러시 트릭: Deepgram 스트리밍에서 활성화. 예상 절감량: 턴 종료 지연 400 ms → 150 ms.
- 가드레일(Guardrails): Cobra/Deepgram에 연결할 수 없는 경우 fail-open합니다. 튜닝을 위해 모든 VAD 발화 이벤트를 감사 로그(Audit Log)에 기록합니다.
