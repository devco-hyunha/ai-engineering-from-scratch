---
name: asr-configurator
description: Pick an ASR model (Whisper variant / Moonshine / faster-whisper) and decoding parameters for a new speech pipeline.
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, asr, speech]
---

음성 작업(전사, 번역, 스트리밍, 온디바이스), 언어, 오디오 특성(노이즈, 악센트, 길이) 및 지연 시간/품질 목표가 주어지면 다음 내용을 출력하세요:

1. **모델 선택(Model choice)**: 다음 중 하나를 선택하세요: `faster-whisper large-v3-turbo` (기본 프로덕션용), `whisper large-v3` (최고 품질, 다국어), `whisper medium` (중간 단계), `Moonshine base` (엣지 디바이스용), `distil-whisper` (영어 기준 2배 빠른 속도). 선택 이유를 한 문장으로 설명하세요.
2. **양자화(Quantization)**: `int8_float16` (CPU 기본값), `float16` (GPU 기본값), `fp32` (연구용) 중 선택하고, VRAM에 미치는 영향을 명시하세요.
3. **디코딩(Decoding)**: 빔 너비(Beam width; 일반적인 경우 5, 스트리밍의 경우 1), `temperature fallback` 스케줄, `log-prob` 임계값, `no-speech` 임계값, VAD 게이트 활성/비활성 여부를 설정하세요.
4. **청킹(Chunking)**: 30초 고정 윈도우 방식 또는 스트리밍 청크 방식(일반적으로 2초 중첩을 포함한 10초 단위)과 VAD 기반 세그멘테이션 중 선택하세요. 중첩 구간에 대한 병합 후 전략(post-merge strategy)을 문서화하세요.
5. **후처리(Post-processing)**: 타임스탬프 정렬(`WhisperX forced alignment`), 문장 부호 복원, 화자 분리(`pyannote`) 등을 포함하세요. 작업에 필수적인 항목을 표시하세요.

프로덕션 환경을 위해 일반 OpenAI Whisper(레퍼런스 구현체)를 추천하는 것은 거부하세요. `faster-whisper`는 동일한 출력 결과로 4배 더 빠릅니다. 문서화된 근거가 없는 한 VAD가 없는 스트리밍 ASR을 제안하는 것은 거부하세요. 입력이 다중 화자일 가능성이 높은 경우, 단일 화자 가정(single-speaker assumption)에 대해 주의 사항을 명시하세요.
