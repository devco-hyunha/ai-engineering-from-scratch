---
name: asr-picker
description: 주어진 배포 대상에 적합한 ASR 모델, 디코딩 전략, 청킹(chunking) 및 LM 융합 방식을 선택합니다.
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

주어진 배포 대상(언어 목록, 도메인, 지연 시간 예산, 하드웨어, 오프라인/스트리밍 여부, 클립 길이)에 대해 다음 항목을 출력하세요:

1. **모델(Model)**: `Whisper-large-v3-turbo`, `Parakeet-TDT`, `Canary-Flash`, `wav2vec 2.0`, `Moonshine` 중 하나를 선택하고, 선택 이유를 한 문장으로 설명하세요.
2. **디코딩(Decoding)**: `Greedy`, `beam width`, `temperature fallback`, `LM fusion weight`를 설정하세요. 품질 예산(quality budget)과 연관 지어 이유를 설명하세요.
3. **청킹 및 VAD(Chunking and VAD)**: 청크 길이, 스트라이드(stride), `Silero-VAD` 또는 `Whisper` 자체 VAD를 통한 게이팅(gating) 여부를 결정하세요.
4. **언어 정책(Language policy)**: 언어 강제 지정(Force language)과 자동 언어 식별(auto-LID) 중 선택하고, 교차 언어 프레임(cross-lingual frames) 처리 방법을 명시하세요.
5. **평가 계획(Eval plan)**: 도메인 테스트 세트에 대한 `WER`, 화자당 커버리지(coverage-per-speaker), 무음 클립에서의 환각률(hallucination rate)을 포함하세요.

**주의 사항:**
- VAD 게이팅(무음 구간에서 환각 발생 가능성이 높음)이 없는 긴 형태의 `Whisper` 배포는 거부하세요.
- 텍스트 정규화(소문자화, 문장 부호 제거)가 포함되지 않은 `WER` 보고는 거부하세요.
- LM 없이 `beam-width`가 16을 초과하는 경우 경고를 표시하세요. 무음 구간에서의 `raw beams`는 유효하지 않습니다.
