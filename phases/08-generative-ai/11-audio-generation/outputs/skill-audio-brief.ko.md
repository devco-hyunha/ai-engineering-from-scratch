---
name: audio-brief
description: 오디오 브리프를 TTS, 음악, SFX 전반에 걸친 모델 + 프롬프트 + 평가 계획으로 번역합니다.
version: 1.0.0
phase: 8단계
lesson: 11강
tags: [audio, tts, music, sfx, codec]
---

오디오 브리프(작업: TTS / 음악 / SFX / 음성 클론, 길이, 스타일, 음성 또는 장르, 라이선스 제약, 실시간 또는 오프라인, 품질 기준)가 주어지면 다음을 출력합니다:

1. 모델 + 호스팅. ElevenLabs V3, OpenAI TTS, XTTS v2, Suno v4, Udio, Stable Audio 2.5, MusicGen 3.3B, AudioCraft 2, 또는 GPT-4o 실시간. 한 문장 이유를 제시합니다.
2. 프롬프트 형식. TTS: 텍스트 + 음성 프롬프트(3-10초 샘플 또는 음성 ID) + 감정 / 속도 태그. 음악: 장르 + 악기 구성 + 분위기 + BPM + 구조적 마커. SFX: 의성어 + 출처 + 길이 힌트.
3. 코덱 + 생성기 + 보코더 체인. 특정 코덱(Encodec 32 kHz, DAC 44 kHz, 커스텀)과 생성기 선택(토큰-AR vs 흐름 매칭)을 명시합니다.
4. 시드 + 재현성. 시드 고정, 버전 고정, 프롬프트 해시.
5. 평가. TTS의 경우 MOS(평균 의견 점수) 또는 A/B, 음악의 경우 CLAP 점수, TTS 전사(transcription)의 경우 CER, SFX의 경우 사용자 청취 테스트.
6. 가드레일(Guardrails). 음성 클론 동의 + 워터마크(PerTh / SynthID-audio), 음악 출력에 대한 저작권 스캔, 학습 데이터 정책 점검.

소유자의 검증된 동의 없이 어떤 음성도 클론하는 것을 거부합니다(카세트 시대의 "3초 프롬프트"는 동의가 아닙니다). 라이선스 없는 참조 자료로 음악을 출시하는 것을 거부합니다. 스트리밍 토큰-AR 모델을 사용하지 않는 200 ms 미만 실시간 목표를 플래그합니다 - 확산 기반(diffusion-based) 오디오는 2026년 기준 300 ms 미만 TTFB를 충족할 수 없습니다.
