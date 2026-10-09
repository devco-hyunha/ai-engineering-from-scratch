---
name: asr-picker
description: 배포 대상에 따라 ASR 모델, 디코딩 전략, 청킹, LM 융합을 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 04강
tags: [audio, asr, speech-recognition]
---

배포 대상(언어 목록, 도메인, 지연 예산, 하드웨어, 오프라인 / 스트리밍, 클립 길이)이 주어지면 다음을 출력합니다:

1. 모델. Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine. 한 문장으로 이유를 설명합니다.
2. 디코딩. Greedy / beam width / temperature fallback / LM fusion weight. 품질 예산과 연결된 이유를 설명합니다.
3. 청킹 및 VAD. 청크 길이, stride, Silero-VAD 또는 Whisper 자체 VAD로 게이트를 적용할지 여부.
4. 언어 정책. 언어 강제 vs auto-LID; 다국어 프레임(cross-lingual frames)을 처리하는 방법.
5. 평가 계획. 도메인 테스트 세트에 대한 WER, 화자별 커버리지, 무음 클립에서의 환각(Hallucination)률.

VAD 게이트가 없는 장문 Whisper 배포는 거부합니다(무음에서 환각(Hallucination)이 발생하기 쉬움). 텍스트 정규화(소문자화, 구두점 제거) 없이 WER을 보고하는 것도 거부합니다. LM이 없는 beam-width > 16은 플래그를 지정합니다. 빈(blank)에 대한 raw beams는 도움이 되지 않습니다.
