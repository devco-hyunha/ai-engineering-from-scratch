---
name: feature-extractor
description: 대상 오디오 모델에 맞추기 위해 특징 유형(feature type), mel 개수, 프레임/홉(frame/hop), 정규화(normalization)를 선택합니다.
version: 1.0.0
phase: 6
lesson: 02
tags: [audio, features, spectrogram, mel]
---

대상 모델(ASR / TTS / 분류기 / 화자 / 음악)과 입력 오디오(샘플링 레이트, 도메인)가 주어지면 다음을 출력하세요:

1. 특징 유형(Feature type): Log-mel, mel, MFCC, raw waveform 또는 이산 코덱(discrete codec; EnCodec, SoundStream) 중 하나를 선택하고 그 이유를 한 문장으로 설명하세요.
2. Mel 개수 및 주파수 범위: `n_mels`, `fmin`, `fmax`를 지정하세요. 도메인(음성 vs 음악) 및 모델의 목표와 연관된 이유를 제시하세요.
3. 프레임 및 홉(Frame and hop): `frame_len`, `hop_len`, 윈도우 유형을 지정하세요. 요구되는 시간 해상도(temporal resolution)와 연관된 이유를 제시하세요.
4. 정규화(Normalization): 발화별 평균/분산(per-utterance mean/var), 전역 통계량(global stats), 또는 고정된 기준을 가진 dB 중 하나를 선택하고, 특징 추출 전(pre) 또는 후(post) 적용 여부를 명시하세요.
5. 검증 스니펫(Validation snippet): 1초 참조 클립에 대해 결과물의 형태(shape), 최솟값/최댓값, 평균/표준편차를 출력하고, 이 값이 학습 설정과 일치하는지 확인하는 Python 코드를 작성하세요.

대상 모델의 공개된 학습 설정(training config)과 프레임/홉/mel 개수가 일치하지 않는 특징 파이프라인(feature pipeline)은 생성을 거부하세요. Whisper 또는 Parakeet 모델에 대해 MFCC 기반 설정을 제안하면 잘못된 것으로 간주하여 지적하세요(해당 모델들은 log-mel을 사용합니다). 정규화 검증(normalization assertion)이 없는 특징 추출기는 모두 지적하세요.
