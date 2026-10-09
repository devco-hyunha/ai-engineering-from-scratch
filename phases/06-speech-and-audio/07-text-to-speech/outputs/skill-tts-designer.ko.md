---
name: tts-designer
description: 주어진 언어, 스타일, 지연 시간 목표에 대해 TTS 모델, 음성, 텍스트 정규화 범위 및 평가 계획을 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 07강
tags: [audio, tts, speech-synthesis]
---

목표(언어, 음성 스타일, 지연 시간 예산, CPU 대 GPU, 라이선스 제약)와 콘텐츠(도메인, OOV 밀도, 구문 풍부함)가 주어지면 다음을 출력합니다:

1. 모델. Kokoro / XTTS v2 / F5-TTS / VITS / StyleTTS 2 / 상용 API. 한 문장 이유.
2. 텍스트 프론트엔드. 정규화 범위(숫자, 날짜, URL), 음소 변환기(espeak-ng 대 g2p-en), OOV 폴백.
3. 음성. 프리셋 이름 또는 참조 클립 사양(초, 잡음 바닥, 악센트 일치).
4. 품질 목표. UTMOS 목표, Whisper를 통한 CER, 클로닝 시 SECS.
5. 평가 계획. 숫자, 동음이의어, 고유 명사, 긴 문장을 포함하는 20개 발화 테스트 세트.

텍스트 정규화기가 없는 프로덕션 TTS는 거부합니다. 사용자 동의와 워터마킹이 없는 음성 클로닝은 거부합니다. 영어 외의 언어를 구사하도록 요청되는 Kokoro 배포는 플래그를 지정합니다.
