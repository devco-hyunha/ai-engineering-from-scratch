---
name: codec-picker
description: 주어진 생성 또는 압축 작업에 적합한 신경망 오디오 코덱(EnCodec / DAC / SNAC / Mimi)을 선택합니다.
version: 1.0.0
phase: 6
lesson: 13
tags: [codec, encodec, dac, snac, mimi, rvq, semantic-tokens]
---

주어진 작업(생성형 LM, 압축, 전이중 대화, 음악 편집, 충실도 목표)을 바탕으로 다음을 출력하세요:

1. 코덱 (Codec). EnCodec-24k · EnCodec-48k · DAC-44.1k · SNAC-24k · Mimi · (대체 방안: 비신경망 압축의 경우 Opus). 한 문장 이유.
2. 프레임 레이트 + 코드북 (Frame rate + codebooks). 비트레이트 예산, 코드북 개수(주로 4~12개), 목표 클립 길이에 따른 시퀀스 길이.
3. 토큰화 체계 (Tokenization scheme). 평면형(Flat) vs 계층형(hierarchical, SNAC) vs 의미+음향(semantic+acoustic, Mimi). LM이 토큰을 소비하는 방식.
4. 디코더 (Decoder). 코덱 내장 디코더 · 외부 보코더(HiFi-GAN) · LM 전용(보코더 없이 코덱 토큰 직접 예측). 이유 설명.
5. 학습 관련 고려사항 (Training implications). 인코더/디코더 학습 필요 여부, 도메인 오디오 미세조정 필요 여부(음성 전용 → 도메인 특화 음악), 기성 모델 동결(Frozen off-the-shelf) 사용 여부.

지연 시간 예산이 엄격한 AR-LM 워크로드에는 DAC를 거부하세요 — 86 Hz 프레임 레이트 × 8개 코드북 = 10초당 5,504개 토큰으로, 빠른 생성에 너무 깁니다. 음악 작업에는 Mimi를 거부하세요 — 음성에 맞춰 튜닝되어 있습니다. 의미 조건부 생성에는 EnCodec을 거부하세요 — 의미 코드북이 없어 텍스트에서 생성 시 음성이 흐릿해집니다.

입력 예시: "Build an AR LM for text-to-speech TTS. Target TTFA 200 ms. English only."

출력 예시:
- 코덱: Mimi. 의미+음향 분리를 통해 텍스트 → 코드북 0 → 코드북 1~7로 인수분해(factorization)할 수 있어 속도가 빠르고 음성 복제도 지원합니다.
- 프레임 레이트 + 코드북: 12.5 Hz · 8개 코드북 · 4.4 kbps. 10초 = 1,000개 토큰.
- 토큰화: 텍스트 + 화자 참조로부터 코드북 0을 먼저 예측한 뒤, 코드북 0 + 화자 참조가 주어진 상태에서 코드북 1~7을 예측합니다(depth-transformer 패턴).
- 디코더: Mimi의 내장 디코더 사용, 외부 보코더 불필요.
- 학습: 텍스트-코덱 LM을 학습하고, Mimi는 동결(freeze)합니다.
