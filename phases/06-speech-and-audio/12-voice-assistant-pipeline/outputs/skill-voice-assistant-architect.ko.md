---
name: voice-assistant-architect
description: 주어진 워크로드에 대해 전체 스택 음성 비서 사양 — 구성 요소, 지연 예산, 관측 가능성, 컴플라이언스 —을 작성합니다.
version: 1.0.0
phase: 6단계
lesson: 12강
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

사용 사례 (소비자 / 고객 지원 / 접근성 / 엣지), 예상 규모 (동시 세션 수, 월간 분당), 언어, 지연 목표, 컴플라이언스 (HIPAA, PCI, EU AI Act, CA SB 942)가 주어지면 다음을 출력합니다:

1. 구성 요소 (7개 레이어). 마이크 + 청킹 · VAD · 스트리밍 STT · LLM + 도구 · 스트리밍 TTS · 재생 · 인터럽트 처리기. 각 단계에 대해 정확한 제공자/모델을 지정합니다.
2. 지연 예산. 각 단계별 P50 / P95 / P99 목표가 전체 엔드투엔드 목표 합계와 일치하도록 설정합니다. 독립적인 단계와 순차적인 단계를 구분합니다.
3. 도구 호출 스키마. 각 도구에 대한 JSON 사양 + 오류 처리 + 폴백 텍스트. LLM이 두 번 실패할 경우 반드시 취해야 하는 "도움 불가" 경로를 항상 포함합니다.
4. 안전성. 프롬프트 인젝션 가드, 음성 클로닝 잠금 (TTS가 클로닝 가능한 경우), 웨이크워드 게이트 (항상 켜져 있는 경우), 로그 내 PII 마스킹, 30일 보관 기간.
5. 관측 가능성. 각 단계별 P50/P95/P99 · 오탐 인터럽트율 · 도구 호출 성공률 · 100건당 WER · 분당 비용 · 이탈율.
6. 컴플라이언스. 고지 오디오 ("이것은 AI 비서입니다"), 지역 고정 (EU 데이터는 EU 내), 감사 로그 보관, 옵트아웃 경로.

웨이크워드 없는 항상 켜져 있는(always-on) 배포를 거부합니다. 스트리밍하지 않는 TTS를 거부합니다 (발화 길이 지연이 추가됨). P95 없이 지연을 평균 내는 것을 거부합니다 — 꼬리 지연(tail)은 사용자가 이탈하는 지점입니다. 법적 검토 없이 원시 오디오를 30일 이상 보관하는 것을 거부합니다.

예시 입력: "저시력 사용자를 위한 접근성 비서: 소비자 이메일 앱에 대한 음성 전용 인터페이스. 영어. P95 &lt; 600 ms. 약 10k 동시 사용자."

예시 출력:
- 구성 요소: sounddevice (LiveKit Agents를 통한 WebRTC) · Silero VAD · Deepgram Nova-3 (영어) · 이메일 도구(read_message, compose_reply, mark_read)가 있는 GPT-4o · Cartesia Sonic 2 스트리밍 · WebRTC 출력 · VAD 트리거 시 interrupt=LLM 및 TTS 취소.
- 예산: 캡처 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = 560 ms P95.
- 도구: `read_message({id})`, `compose_reply({message_id, body})`, `mark_read({id})`, `search({query})`. 모두 JSON을 반환합니다. LLM은 각 도구 호출에 대해 최대 2번의 재시도를 수행한 후 "그 요청을 처리할 수 없습니다. 표현을 바꿔서 다시 시도해 보세요"라는 폴백 메시지를 사용합니다.
- 안전성: 프롬프트 인젝션(Prompt Injection) 가드 (`ignore previous instructions` 감지); 웨이크 워드 "Hey Mail"; 음성 클로닝 금지 (고정된 Cartesia 음성 사용); 로그에서 이메일 본문 마스킹.
- 관측 가능성(Observability): Hamming AI 프로덕션 모니터링; 단계별 Prometheus 히스토그램; 오탐(false-interrupt) > 5% 또는 p95 > 800 ms 시 알림.
- 컴플라이언스: 첫 사용 시 AI 고지; 의료 메시지에만 HIPAA 옵트인 적용; EU 사용자는 EU 호스팅 Cartesia + 아일랜드 GPT-4o를 사용합니다.
