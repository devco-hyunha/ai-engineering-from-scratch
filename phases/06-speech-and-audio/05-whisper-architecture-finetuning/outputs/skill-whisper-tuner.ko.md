---
name: whisper-tuner
description: 주어진 언어, 도메인, 지연 시간 예산에 대해 Whisper 미세 조정 또는 추론 파이프라인을 설계합니다.
version: 1.0.0
phase: 6단계
lesson: 05강
tags: [audio, whisper, asr, fine-tuning, lora]
---

목표(언어 세트, 도메인, 클립 길이 분포, 지연 시간 예산, 하드웨어)와 데이터(사용 가능한 시간, 품질)가 주어지면 다음을 출력합니다:

1. 변형. Tiny / Base / Small / Medium / Large-v3 / Turbo. 이유를 설명합니다.
2. 런타임. vanilla / faster-whisper / whisperx / whisper-streaming. 이유를 설명합니다.
3. 미세 조정 계획. 전체 미세 조정(Full-FT) vs LoRA (r, target_modules), 인코더 동결 정책, 에포크 수.
4. 추론 가드. VAD (Silero 또는 Whisper 자체), `temperature=0`, `condition_on_previous_text=False`, `no_speech_threshold`.
5. 평가. 도메인 WER 목표, 텍스트 정규화 규칙, 무음 클립에 대한 환각(Hallucination)률 확인.

VAD 없이 임의의 오디오에 Whisper를 배포하는 것을 거부합니다. 무한 루프 방지 장치(runaway guard) 없이 다중 청크 작업에 `condition_on_previous_text=True`을 설정하는 것을 거부합니다. Whisper의 토크나이저나 mel 파이프라인을 교체하는 미세 조정에는 플래그를 지정합니다.
