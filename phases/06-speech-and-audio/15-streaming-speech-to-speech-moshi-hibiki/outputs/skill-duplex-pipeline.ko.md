---
name: duplex-pipeline
description: 음성 에이전트 워크로드에 대해 전이중(Moshi) vs 파이프라인(VAD + STT + LLM + TTS) 아키텍처를 선택합니다.
version: 1.0.0
phase: 6
lesson: 15
tags: [moshi, hibiki, full-duplex, voice-agent, streaming]
---

워크로드(지연 시간 목표, 도구 호출 필요성, 언어 지원 범위, 하드웨어 예산, 클라우드 vs 에지)가 주어지면 다음을 출력합니다:

1. 아키텍처(Architecture). 전이중(Moshi / GPT-4o Realtime / Gemini Live) vs 파이프라인(LiveKit + STT + LLM + TTS, 레슨 12). 한 문장으로 이유 제시.
2. 모델(Model). Moshi · Hibiki · Hibiki-Zero · Sesame CSM · GPT-4o Realtime · Gemini 2.5 Live · 전통적 파이프라인. 이유 제시.
3. 규모(Scale). 세션당 GPU 비용(Moshi는 슬롯을 점유함), 최대 동시 세션 수, 콜드 스타트 영향.
4. 도구 호출 경로(Tool-calling path). 필요한 경우 — 하이브리드 파이프라인(전이중 + 도구 호출용 외부 LLM) 또는 순수 파이프라인. 트레이드오프 설명.
5. 언어 지원 범위(Language coverage). 전이중 모델은 언어 지원 범위가 좁으며, 파이프라인은 LLM의 다국어 능력을 그대로 상속받음.

도구 호출 / 검색(retrieval)이 필요한 엔터프라이즈 에이전트에는 전이중 전용 아키텍처를 거부하세요 — Moshi는 대화 모델이며 에이전트 프레임워크가 아닙니다. 250 ms 미만의 대화형 에이전트에는 파이프라인 전용을 거부하세요 — 단계별 지연 시간이 누적됩니다. 단일 GPU에서 4개를 초과하는 동시 세션에는 Moshi를 거부하세요 — 자원 경합이 발생합니다.

입력 예시: "언어 학습을 위한 음성 동반자 — 회화 유창성 연습. 영어 + 프랑스어. < 250 ms 응답성. 일일 활성 사용자(DAU) 1만 명."

출력 예시:
- Architecture: 전이중(Moshi). 250 ms 미만 지연 시간 요구사항 + 회화 유창성이 Moshi의 강점과 부합함.
- Model: Moshi. EN + FR 모두 잘 지원됨. CC-BY 4.0 라이선스.
- Scale: 동시 세션 4-6개당 L4 GPU 1장 → 동시 접속률 10% 기준 1만 DAU에 대해 피크 시 약 1,500장의 GPU 필요. 조용한 경로(quiet path)에는 Kyutai Pocket TTS + 로컬 Whisper를 사용하는 온디바이스 라이트 모드 계획.
- Tool calling: 최소 — "문법 힌트 표시" 및 "이 구문 번역"은 소형 LLM 사이드카를 통해 라우팅 가능; 상호작용의 대부분은 Moshi가 강점을 보이는 개방형 대화임.
- Language coverage: EN + FR (네이티브); Hibiki-Zero 적응을 통한 ES / DE / JP (새 언어당 1,000시간의 오디오 필요).
