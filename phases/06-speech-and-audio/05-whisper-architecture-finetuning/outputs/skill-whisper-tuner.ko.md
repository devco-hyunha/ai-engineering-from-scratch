---
name: whisper-tuner
description: 주어진 언어, 도메인 및 지연 시간 예산(latency budget)에 맞는 Whisper 미세 조정(fine-tuning) 또는 추론 파이프라인을 설계합니다.
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

대상(언어 세트, 도메인, 클립 길이 분포, 지연 시간 예산, 하드웨어)과 데이터(가용 시간, 품질)가 주어지면 다음을 출력하세요:

1. 변형 모델(Variant). Tiny / Base / Small / Medium / Large-v3 / Turbo 중 선택하고 그 이유를 기술하세요.
2. 런타임(Runtime). vanilla / faster-whisper / whisperx / whisper-streaming 중 선택하고 그 이유를 기술하세요.
3. 미세 조정 계획(Fine-tune plan). Full-FT 대 LoRA (`r`, `target_modules`), 인코더 동결(freeze-encoder) 정책, 에포크(epoch) 수.
4. 추론 가드(Inference guards). VAD (Silero 또는 Whisper 자체 VAD), `temperature=0`, `condition_on_previous_text=False`, `no_speech_threshold`.
5. 평가(Evaluation). 도메인 WER 목표, 텍스트 정규화 규칙, 무음 클립에 대한 환각률(hallucination rate) 점검.

VAD 없이 임의의 오디오에 Whisper를 배포하는 설계는 거부하세요. 폭주 방지 가드(runaway guard) 없이 멀티 청크(multi-chunk) 작업에 대해 `condition_on_previous_text=True`로 설정하는 설계도 거부하세요. Whisper의 토크나이저(tokenizer)나 멜 파이프라인(mel pipeline)을 교체하려는 모든 미세 조정 시도는 반드시 플래그(flag)를 표시하세요.
