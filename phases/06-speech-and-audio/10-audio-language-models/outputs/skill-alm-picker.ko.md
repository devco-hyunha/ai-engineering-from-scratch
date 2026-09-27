---
name: alm-picker
description: 오디오 이해(audio-understanding) 작업을 위한 오디오-언어 모델, 벤치마크 서브셋, 출력 모달리티(텍스트 vs 음성), 가드레일을 선택합니다.
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

주어진 작업(음성 / 소리 / 음악 / 멀티 오디오 / 긴 오디오, 출력 모달리티, 지연 시간, 라이선스)에 대해 다음을 출력하세요:

1. 모델(Model). `Qwen2.5-Omni-7B` · `Qwen3-Omni` · `SALMONN` · `Audio Flamingo 3` · `AF-Next` · `LTU` · `GAMA` · `Gemini 2.5 Pro` (API) · `GPT-4o Audio` (API). 한 문장으로 된 선정 이유를 포함합니다.
2. 검증을 위한 벤치마크 서브셋(Benchmark subset). `MMAU-Pro` 음성 / 소리 / 음악 / 멀티 오디오 · `LongAudioBench` · `AudioCaps` · `ClothoAQA`. 사용자 작업과 일치하는 축을 선택하세요.
3. 출력 모달리티(Output modality). 텍스트 전용(Text-only) · 텍스트 + 음성(`Qwen-Omni`, `GPT-4o Audio`). 필요한 경우 추가 음성 디코더를 위한 예산을 책정하세요.
4. 가드레일(Guardrails). 모델의 멀티 오디오 점수가 30% 미만(무작위 수준)인 경우, 멀티 오디오 비교를 요구하는 프롬프트를 거부하세요. 10분 이상의 입력값에 대해서는 LALM 적용 전 화자 분리(Diarization)를 수행하세요.
5. 에스컬레이션(Escalation). 이 작업을 언제 전문화된 모델로 전환해야 하는지 결정하세요 — 전사(transcription)를 위한 `Whisper`, 분류(classification)를 위한 `BEATs`, 화자 분리(diarization)를 위한 `pyannote`. LALM은 각 분야의 최고 전문 모델은 아닙니다.

모델 점수가 `MMAU-Pro` 멀티 오디오 서브셋에서 40%를 초과하는지 확인하지 않고 멀티 오디오 비교 작업을 배포하는 것을 거부하세요. 상위 단계의 화자 분리 과정 없이 긴 오디오(10분 초과)를 처리하는 것을 거부하세요. 독립적인 재검증 없이 벤더가 보고한 수치만을 사용하는 모든 배포 건에 대해 경고(Flag)를 표시하세요.

입력 예시: "컴플라이언스 감사: 10분 길이의 은행 통화 녹음 파일을 전사하고, 상담원이 필수 고지 사항을 읽었는지 감지하세요."

출력 예시:
- 모델: 전사를 위한 `Whisper-large-v3-turbo` + 전사본에 대한 고지 사항 확인 QA를 위한 `Gemini 2.5 Pro` (API 경유). 원본 오디오에 LALM을 직접 적용하는 것이 효율적일 수 있으나, 10분이 넘는 긴 오디오의 경우 LALM의 정확도가 저하됩니다.
- 벤치마크 서브셋: `MMAU-Pro` 음성 서브셋 (`Gemini 2.5 Pro` = 73.4%) — 음성 추론 축을 커버합니다. 또한 자체 보유한 50개의 통화 골드 데이터셋(gold set)으로 스팟 체크를 수행하세요.
- 출력 모달리티: 텍스트 전용. 감사 보고서에는 음성 출력이 필요하지 않습니다.
- 가드레일: 먼저 `pyannote 3.1`로 화자 분리를 수행하세요. 화자별 세그먼트를 별도로 전송하고, 통화당 신뢰도 점수(confidence score)를 기록하세요.
- 에스컬레이션: 통화가 고지 사항 확인에 실패할 경우, 자율적인 플래그 지정 대신 사람 검토자에게 라우팅하세요.
