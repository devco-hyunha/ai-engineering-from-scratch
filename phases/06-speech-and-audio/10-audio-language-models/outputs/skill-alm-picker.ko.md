---
name: alm-picker
description: 오디오 이해 작업을 위해 오디오-언어 모델, 벤치마크 하위 집합, 출력 모달리티(텍스트 vs 음성), 가드레일을 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 10강
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

작업 조건(음성 / 소리 / 음악 / 다중 오디오 / 긴 오디오, 출력 모달리티, 지연 시간, 라이선스)이 주어지면 다음을 출력합니다:

1. 모델. Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro (API) · GPT-4o Audio (API). 한 문장 이유를 제시합니다.
2. 검증할 벤치마크 하위 집합. MMAU-Pro 음성 / 소리 / 음악 / 다중 오디오 · LongAudioBench · AudioCaps · ClothoAQA. 사용자 작업과 일치하는 축을 선택합니다.
3. 출력 모달리티. 텍스트 전용 · 텍스트 + 음성 (Qwen-Omni, GPT-4o Audio). 필요 시 추가 음성 디코더에 대한 예산을 고려합니다.
4. 가드레일. 모델의 다중 오디오 점수가 &lt; 30% (무작위 수준)인 경우 다중 오디오 비교를 요구하는 프롬프트를 거부합니다. &gt; 10분 입력의 경우 LALM 전에 화자 분리(diarization)를 수행합니다.
5. 상향 조정(Escalation). 이 작업이 특화 모델로 전환해야 하는 시점 — 전사(transcription)는 Whisper, 분류는 BEATs, 화자 분리(diarization)는 pyannote. LALM은 각 분야의 최적 모델이 아닙니다.

모델이 MMAU-Pro 다중 오디오 하위 집합에서 점수가 &gt; 40%임을 검증하지 않은 채 다중 오디오 비교 작업을 출시하는 것을 거부합니다. 상류 단계의 화자 분리(diarization) 없이 긴 오디오(&gt; 10분) 작업을 거부합니다. 독립적인 재검증 없이 벤더 보고 수치만 사용하는 배포는 모두 플래그 처리합니다.

예시 입력: "컴플라이언스 감사: 10분 은행 통화 녹음을 전사하고, 상담원이 필수 공시를 읽었는지 감지."

예시 출력:
- 모델: 전사(transcription)는 Whisper-large-v3-turbo, 전사문에 대한 공시 확인 QA는 Gemini 2.5 Pro (API 경유) 사용. 원시 오디오에 LALM을 직접 적용하는 것은 매력적이지만, 10분을 넘기면 긴 오디오 LALM의 정확도가 떨어집니다.
- 벤치마크 하위 집합: MMAU-Pro 음성 하위 집합 (Gemini 2.5 Pro = 73.4%) — 음성 추론 축을 포함합니다. 자체 50건 골든 세트(gold set)에서도 스팟 체크(spot-check)를 수행합니다.
- 출력 모달리티: 텍스트 전용. 감사 보고서에는 음성 출력이 필요하지 않습니다.
- 가드레일: pyannote 3.1로 먼저 화자 분리(diarization)를 수행합니다; 화자별 세그먼트를 별도로 전송합니다; 통화별 신뢰도 점수를 로깅합니다.
- 에스컬레이션: 호출이 공개 검사를 통과하지 못하면, 자율적 플래그 대신 인간 리뷰어로 라우팅합니다.
