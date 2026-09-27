---
name: voice-assistant-architect
description: Produce a full-stack voice-assistant spec — components, latency budget, observability, compliance — for a given workload.
version: 1.0.0
phase: 6
lesson: 12
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

주어진 유스케이스(소비자용 / 고객 지원 / 접근성 / 에지), 예상 규모(동시 세션 수, 월간 이용 시간), 언어, 지연 시간 목표, 컴플라이언스(HIPAA, PCI, EU AI Act, CA SB 942)를 바탕으로 다음을 출력하세요:

1. 구성 요소 (7개 계층). 마이크 + 청킹(chunking) · VAD · 스트리밍 STT · LLM + 도구(tools) · 스트리밍 TTS · 재생 · 중단 처리기(interruption handler). 각 단계에 사용할 정확한 제공업체/모델명을 명시하세요.
2. 지연 시간 예산(Latency budget). 전체 엔드 투 엔드(end-to-end) 목표 합계에 맞춘 단계별 P50 / P95 / P99 목표치를 설정하세요. 어떤 단계가 독립적인지 또는 순차적인지 표시하세요.
3. 도구 호출 스키마(Tool-call schema). 각 도구에 대한 JSON 명세 + 에러 처리 + 폴백(fallback) 텍스트. LLM이 두 번 실패했을 때 반드시 따라야 하는 "도움을 드릴 수 없습니다" 경로를 항상 포함하세요.
4. 안전성(Safety). 프롬프트 인젝션 방어, 음성 복제 차단(TTS가 복제 기능을 지원하는 경우), 웨이크 워드(wake-word) 게이트(상시 대기형의 경우), 로그 내 PII(개인정보) 비식별화, 30일 보관 정책.
5. 관측 가능성(Observability). 단계별 P50/P95/P99 · 오인식 중단율(false-interruption rate) · 도구 호출 성공률 · 100회 호출당 WER(단어 오류율) · 분당 비용 · 이탈률(abandon rate).
6. 컴플라이언스(Compliance). 고지 오디오("이 서비스는 AI 어시스턴트입니다"), 지역 고정(EU 데이터는 EU 내 보관), 감사 로그 보관, 옵트아웃(opt-out) 경로.

웨이크 워드가 없는 상시 대기형(always-on) 배포는 거부하세요. 스트리밍을 지원하지 않는 TTS(발화 길이에 따른 지연 시간 추가)는 거부하세요. P95를 제외한 평균 지연 시간만 제시하는 것은 거부하세요. 꼬리 지연 시간(tail latency)에서 사용자가 이탈하기 때문입니다. 법적 검토 없이 30일을 초과하는 원시 오디오(raw-audio) 보관은 거부하세요.

입력 예시: "저시력 사용자를 위한 접근성 어시스턴트: 소비자용 이메일 앱을 위한 음성 전용 인터페이스. 영어. P95 < 600 ms. 동시 사용자 약 1만 명."

출력 예시:
- 구성 요소: sounddevice (LiveKit Agents를 통한 WebRTC) · Silero VAD · Deepgram Nova-3 (English) · 이메일 도구(`read_message`, `compose_reply`, `mark_read`)를 포함한 GPT-4o · Cartesia Sonic 2 스트리밍 · WebRTC 출력 · VAD 작동 시 `interrupt=cancel-LLM-and-TTS` 설정.
- 예산: 캡처 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = 560 ms P95.
- 도구: `read_message({id})`, `compose_reply({message_id, body})`, `mark_read({id})`, `search({query})`. 모두 JSON 반환; LLM은 도구당 최대 2회 재시도 후 "해당 작업을 수행할 수 없습니다. 다시 말씀해 주시겠어요?"라고 폴백 응답.
- 안전성: 프롬프트 인젝션 방어 (`ignore previous instructions` 감지); 웨이크 워드 "Hey Mail"; 음성 복제 불가 (고정된 Cartesia 음성 사용); 로그 내 이메일 본문 비식별화.
- 관측 가능성: Hamming AI 프로덕션 모니터링; 단계별 Prometheus 히스토그램; 오인식 중단 > 5% 또는 p95 > 800 ms 발생 시 알림.
- 컴플라이언스: 첫 사용 시 AI 고지; 의료 메시지에 대해서만 HIPAA 옵트인; EU 사용자는 EU 호스팅 Cartesia + 아일랜드 지역 GPT-4o 연결.
