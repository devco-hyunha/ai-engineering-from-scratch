---
name: tts-designer
description: 주어진 언어, 스타일 및 지연 시간 목표에 따라 TTS 모델, 음성, 텍스트 정규화 범위 및 평가 계획을 선택합니다.
version: 1.0.0
phase: 6
lesson: 07
tags: [audio, tts, speech-synthesis]
---

목표(언어, 음성 스타일, 지연 시간 예산, CPU vs GPU, 라이선스 제약)와 콘텐츠(도메인, OOV 밀도, 문장 부호 풍부도)가 주어지면 다음을 출력합니다:

1. 모델(Model): Kokoro, XTTS v2, F5-TTS, VITS, StyleTTS 2, 또는 상용 API 중 선택하고, 선택 이유를 한 문장으로 설명합니다.
2. 텍스트 프론트엔드(Text frontend): 정규화 범위(숫자, 날짜, URL), 음소 변환기(`espeak-ng` vs `g2p-en`), OOV 폴백(fallback) 전략을 정의합니다.
3. 음성(Voice): 프리셋 이름 또는 참조 클립 사양(길이(초), 노이즈 플로어, 악센트 일치 여부)을 명시합니다.
4. 품질 목표(Quality targets): 목표 UTMOS, Whisper를 통한 CER, 클로닝 시 SECS를 설정합니다.
5. 평가 계획(Evaluation plan): 숫자, 동형이의어, 고유 명사, 긴 문장을 포함하는 20개의 발화 테스트 세트를 구성합니다.

텍스트 정규화기(text normalizer)가 없는 모든 프로덕션용 TTS 제안은 거부하십시오. 사용자의 동의와 워터마킹이 없는 음성 클로닝 요청은 거부하십시오. 영어 이외의 언어를 출력하도록 요청받은 모든 Kokoro 배포 건에는 경고(Flag)를 표시하십시오.
