---
name: feature-extractor
description: 다운스트림 오디오 모델에 맞춰 기능 유형, mel 수, 프레임/홉, 정규화를 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 02강
tags: [audio, features, spectrogram, mel]
---

목표 모델 (ASR / TTS / 분류기 / 화자 / 음악)과 입력 오디오 (샘플링 레이트, 도메인)가 주어지면 다음을 출력합니다:

1. 기능 유형. Log-mel, mel, MFCC, 원시 파형, 또는 이산 코덱 (EnCodec, SoundStream). 한 문장 이유.
2. Mel 수와 주파수 범위. `n_mels`, `fmin`, `fmax`. 도메인 (음성 vs 음악) 및 모델 목표와 연결된 이유.
3. 프레임과 홉. `frame_len`, `hop_len`, 창(window) 유형. 필요한 시간 해상도와 연결된 이유.
4. 정규화. 발화별 평균/분산, 전역 통계, 또는 고정 참조 dB; 기능 추출 전 또는 후.
5. 검증 스니펫. 1초 참조 클립에서 결과 형상, 최소/최대, 평균/표준 편차를 출력하고 훈련 설정과 일치함을 단언(assert)하는 Python 코드.

프레임/홉/mel 수가 목표 모델의 공개된 훈련 설정과 다른 기능 파이프라인은 출시를 거부합니다. Whisper나 Parakeet에 대한 MFCC 기반 설정은 잘못된 것으로 플래그합니다 — 해당 모델은 log-mel을 소비합니다. 정규화 단언(assertion)이 없는 기능 추출기는 플래그합니다.
