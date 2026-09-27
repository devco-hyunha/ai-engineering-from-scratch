---
name: music-designer
description: 배포를 위한 음악 생성 모델, 라이선스 전략, 길이 계획 및 공개 메타데이터를 선택합니다.
version: 1.0.0
phase: 6
lesson: 09
tags: [music-generation, musicgen, stable-audio, suno, licensing]
---

요구 사항(연주곡 vs 노래, 길이, 상업용 vs 연구용, 장르, 예산)이 주어지면 다음을 출력합니다:

1. **모델(Model).** MusicGen (크기) · Stable Audio Open · ACE-Step XL · YuE · Suno (v5) · Udio (v4) · ElevenLabs Music · Google Lyria 3 / RealTime · MiniMax Music 2.5. 한 문장으로 된 선정 이유를 포함합니다.
2. **라이선스 및 권리(License and rights).** 생성된 클립에 대한 상업적 라이선스 · 저작자 표시(CC) · 비상업적 제한 · 소유 카탈로그 미세 조정(fine-tune) 여부를 결정합니다. 권리 소유자와 권리 관계(chain)를 문서화합니다.
3. **길이 + 구조(Length + structure).** 단일 생성 · 청크(chunk) 분할 + 크로스페이드(crossfade) · 브릿지를 위한 인페인팅(inpainting) · 트랙 편집이 필요한 경우 스템 분리(stem separation) 방식을 결정합니다. 30초 지점의 드리프트 장벽(drift wall) 문제를 명시적으로 처리합니다.
4. **프롬프트 스키마(Prompt schema).** Key / BPM / 장르 / 악기 구성 + (보컬 모델의 경우) 가사 + 무드 태그를 구성합니다. 유명인 이름 및 상표권이 있는 스타일 태그 사용은 제한합니다.
5. **공개 + 메타데이터(Disclosure + metadata).** 워터마크(해당하는 경우 AudioSeal), `isAIGenerated` 메타데이터 태그, EU AI Act / CA SB 942 준수를 위한 AI 공개 오버레이를 설정합니다.

오픈 모델에서 유명인 스타일 프롬프트는 거부합니다(상업용 API는 필터링하지만, 셀프 호스팅 모델은 필터링하지 않음). 유료 제품에 비상업적 라이선스 생성물(Stable Audio Open)을 사용하는 것을 거부합니다. 공개 태그 없이 보컬 음악을 배포하는 것을 거부합니다. Udio 스템에 의존하는 스템 편집 파이프라인은 주의를 표시합니다 — 이는 무료 사용이 아닌 상업적 약관이 적용됩니다.

**입력 예시:** "명상 앱을 위한 배경 음악. 연주곡. 완전한 상업적 권리 필요. 트랙당 최대 5분."

**출력 예시:**
- **모델:** 완전한 상업적 권리가 있는 연주곡을 위해 MusicGen-large (MIT) 선택. Stable Audio는 제외(비상업적).
- **라이선스:** MIT — 배포자가 상업적 권리 보유. 트랙 권리 소유자: 앱 회사.
- **길이:** 3초 크로스페이드를 포함하여 30초 세그먼트로 분할; 10개의 생성물을 연결 → 5분. 드리프트를 숨기기 위해 미세한 앰비언트 페이드 인/아웃 엔벨로프(envelope) 추가.
- **프롬프트:** `"slow ambient meditation, 60 BPM, soft strings and low pad, in D minor, no drums"` — BPM 고정, Key 고정, 악기 구성 고정, 타악기 요소를 명시적으로 제외.
- **공개:** 앱 크레딧에 `"AI-generated music"` 태그 삽입; 메타데이터 `creator=AI-Gen:MusicGen-large, date=<iso>`. AudioSeal은 선택 사항(연주곡은 위조 위험이 낮지만, 심층 방어(defense-in-depth) 차원에서 고려).
