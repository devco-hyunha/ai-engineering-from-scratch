---
name: asr-configurator
description: 새로운 음성 파이프라인을 위해 ASR 모델 (Whisper 변형 / Moonshine / faster-whisper) 및 디코딩 매개변수를 선택합니다.
version: 1.0.0
phase: 7단계
lesson: 10강
tags: [transformers, whisper, asr, speech]
---

음성 작업 (전사 / 번역 / 스트리밍 / 온디바이스), 언어, 오디오 특성 (잡음, 억양, 길이) 및 지연/품질 목표가 주어지면 다음을 출력합니다:

1. 모델 선택. 다음 중 하나: faster-whisper large-v3-turbo (기본 프로덕션), whisper large-v3 (최고 품질, 다국어), whisper medium (중간 등급), Moonshine base (엣지), distil-whisper (영어 2배 빠름). 한 문장 이유를 제시합니다.
2. 양자화. int8_float16 (CPU 기본), float16 (GPU 기본), fp32 (연구용). VRAM 영향을 표시합니다.
3. 디코딩. 빔 너비 (일반적으로 5, 스트리밍은 1), 온도 폴백 스케줄, 로그 확률 임계값, 무음 임계값, VAD 게이트 켜기/끄기.
4. 청킹. 30초 고정 윈도우 대 스트리밍 청크 (일반적으로 2초 겹침이 있는 10초) + VAD 기반 분할. 겹침에 대한 병합 후 전략을 문서화합니다.
5. 후처리. 타임스탬프 정렬 (WhisperX 강제 정렬), 문장 부호 복원, 화자 분리 (pyannote). 작업에 필수적인 항목을 표시합니다.

프로덕션 환경에서는 순수 OpenAI Whisper (참구현)를 추천하지 마세요. `faster-whisper`는 동일한 출력으로 4배 더 빠릅니다. 문서화된 이유 없이 VAD가 없는 스트리밍 ASR을 출시하지 마세요. 입력이 다중 화자일 가능성이 높을 경우 단일 화자 가정을 표시합니다.
