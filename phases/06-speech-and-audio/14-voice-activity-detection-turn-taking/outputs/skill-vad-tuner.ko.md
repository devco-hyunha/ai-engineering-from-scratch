---
name: vad-tuner
description: Pick VAD model, threshold, silence hangover, pre-roll, and turn-detection strategy for a voice agent.
version: 1.0.0
phase: 6
lesson: 14
tags: [vad, silero, cobra, turn-detection, flush-trick]
---

워크로드(소비자용 / 콜센터 / 에지 / 접근성; 소음 프로필; 언어 혼합; 지연 시간)를 고려하여 다음을 출력하세요:

1. VAD: Silero VAD (기본값) · Cobra (상용 수준의 정확도) · pyannote segmentation (화자 분리 수준) · WebRTC VAD (레거시 / 초경량). 선택 이유를 한 문장으로 설명하세요.
2. 파라미터(Parameters): 임계값/Threshold (0.3-0.5), 최소 음성 지속 시간/min speech (200-300 ms), 침묵 유예 시간/silence hangover (400-800 ms), 프리롤 버퍼/pre-roll (250-500 ms).
3. 의미론적 발화 전환 감지(Semantic turn detection): 활성화(LiveKit turn-detector 또는 커스텀 MLP) 또는 비활성화. 예상되는 사용자 발화 패턴과 연관된 이유를 제시하세요.
4. 플러시 트릭(Flush trick): 활성화(STT가 지원하는 경우 — Kyutai / Deepgram) 또는 비활성화. 예상되는 지연 시간 절감 효과를 명시하세요.
5. 가드레일(Guards): 최소 지속 시간보다 짧은 발화는 거부; 프리롤(pre-roll)은 항상 유지; 사용자별 침묵 유예 시간(silence hangover) 오버라이드 상한 설정; VAD 서비스 다운 시 fail-open(모든 입력을 음성으로 처리).

프로덕션 환경에서 에너지 기반(energy-only) VAD는 노이즈에 너무 취약하므로 거부하세요. 침묵 유예 시간(silence hangover)이 0인 설정은 사용자의 말을 끊을 수 있으므로 거부하세요. 전용 Silero를 사용할 수 있는 상황에서 Whisper 기반 VAD를 사용하는 것은 거부하세요(더 느리고 정확도가 낮음).

입력 예시: "항공권 재예약을 위한 콜센터 IVR. 소음이 있는 배경(공항). 영어 + 스페인어. 500ms 미만의 발화 전환 감지."

출력 예시:
- VAD: 소음 저항성 이점을 위해 Cobra(상용) 선택. 비용이 너무 높을 경우 Silero로 폴백(Fall-back).
- 파라미터: threshold 0.4 (공항 소음 레벨이 높음); min speech 300 ms; silence hangover 600 ms (사용자가 IVR 도중 항공편 번호를 읽기 위해 자주 멈춤); pre-roll 400 ms.
- 의미론적 발화 전환(Semantic turn): LiveKit turn-detector 활성화 — 문장 중간의 일시 정지가 흔함 ("I need to change my flight... to tomorrow").
- 플러시 트릭(Flush trick): Deepgram 스트리밍에서 활성화. 예상 절감 효과: 발화 전환 지연 시간 400 ms → 150 ms.
- 가드레일: Cobra/Deepgram에 접속할 수 없는 경우 fail-open; 튜닝을 위해 모든 VAD 트리거 이벤트에 대한 감사 로그(audit log) 기록.
