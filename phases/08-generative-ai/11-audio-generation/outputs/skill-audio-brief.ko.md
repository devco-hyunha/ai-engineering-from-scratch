---
name: audio-brief
description: 오디오 브리프를 TTS, 음악, SFX 전반에 걸친 모델, 프롬프트 및 평가 계획으로 변환합니다.
version: 1.0.0
phase: 8
lesson: 11
tags: [audio, tts, music, sfx, codec]
---

오디오 브리프(작업 유형: TTS / 음악 / SFX / 음성 복제, 재생 시간, 스타일, 음성 또는 장르, 라이선스 제약, 실시간 또는 오프라인, 품질 기준)가 주어지면 다음 내용을 출력하세요:

1. **모델 + 호스팅(Model + hosting)**: ElevenLabs V3, OpenAI TTS, XTTS v2, Suno v4, Udio, Stable Audio 2.5, MusicGen 3.3B, AudioCraft 2 또는 GPT-4o realtime 중 선택하고, 선택 이유를 한 문장으로 설명하세요.
2. **프롬프트 형식(Prompt format)**: 
    - TTS: 텍스트 + 음성 프롬프트(3~10초 샘플 또는 `voice ID`) + 감정/속도 태그.
    - 음악: 장르 + 악기 구성 + 분위기 + BPM + 구조적 마커.
    - SFX: 의성어 + 소스 + 길이 힌트.
3. **코덱 + 생성기 + 보코더 체인(Codec + generator + vocoder chain)**: 특정 코덱(Encodec 32 kHz, DAC 44 kHz, 커스텀 등)과 생성 방식(token-AR vs flow-matching)을 명시하세요.
4. **시드 + 재현성(Seed + reproducibility)**: 시드 고정(`seed pin`), 버전 고정(`version pin`), 프롬프트 해시(`prompt hash`).
5. **평가(Eval)**: TTS의 경우 MOS(mean opinion score) 또는 A/B 테스트, 음악의 경우 CLAP 점수, TTS 전사(transcription)의 경우 CER, SFX의 경우 사용자 청취 테스트를 제안하세요.
6. **가드레일(Guardrails)**: 음성 복제 동의 여부 및 워터마크(PerTh / SynthID-audio), 음악 출력물에 대한 저작권 스캔, 학습 데이터 정책 확인.

소유자의 검증된 동의 없이 어떠한 음성도 복제하는 요청은 거부하세요(카세트 시대의 "3초 프롬프트"는 동의로 간주하지 않습니다). 라이선스가 없는 참조 자료가 포함된 음악의 출고를 거부하세요. 스트리밍 `token-AR` 모델을 사용하지 않으면서 실시간 목표가 200ms 미만인 경우 경고를 표시하세요. 2026년 기준으로 확산 기반(diffusion-based) 오디오는 300ms 미만의 TTFB(Time To First Byte)를 충족할 수 없습니다.
