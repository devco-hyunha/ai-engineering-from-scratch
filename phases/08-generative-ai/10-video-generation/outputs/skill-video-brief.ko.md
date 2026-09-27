---
name: video-brief
description: Translate a video brief into a model + prompt + shot plan for a 2026 video generator.
version: 1.0.0
phase: 8
lesson: 10
tags: [video, diffusion, sora, veo, kling]
---

비디오 브리프(영상 길이, 종횡비, 스타일, 피사체, 카메라 계획, 오디오 요구사항, 충실도 기준, 예산)가 주어지면 다음을 출력합니다:

1. **모델 + 호스팅(Model + hosting)**: Sora, Veo 3, Kling 2.1, Runway Gen-3, Pika 2.0, CogVideoX, HunyuanVideo, WAN 2.2 또는 Mochi-1 중 하나를 선택합니다. 영상 길이, 품질, 라이선스와 관련된 이유를 한 문장으로 포함합니다.
2. **프롬프트 스캐폴딩(Prompt scaffolding)**: (a) 카메라 언어(establishing, tracking, dolly, crane, handheld), (b) 피사체 + 동작, (c) 조명 + 스타일, (d) 부정 프롬프트(negative prompt) 또는 스타일 토글. Sora의 경우 50~150 토큰, Runway의 경우 20~60 토큰을 목표로 합니다.
3. **샷 플랜(Shot plan)**: 단일 클립(Single-clip) 대 스티칭된 멀티 샷(stitched multi-shot), 키프레임 또는 첫 프레임 앵커, 샷당 I2V(Image-to-Video) 대 T2V(Text-to-Video) 여부를 결정합니다.
4. **시드 + 재현성(Seed + reproducibility)**: 샷당 시드, 버전 고정(version pin), 툴링 저장소(tooling repo)를 정의합니다.
5. **QA 체크리스트(QA checklist)**: 깜빡임(flicker), 정체성 일관성(identity consistency), 물리 법칙 위반, 워터마크 준수 여부를 프레임 단위로 확인합니다.
6. **오디오(Audio)**: Veo 3의 네이티브 기능을 사용하거나, 그 외의 경우 별도 결합(ElevenLabs, Suno, 또는 라이선스 스템 + 립싱크 패스) 방식을 제안합니다.

무료 티어에서 1080p 해상도로 10초 이상의 연속 동작을 보장하라는 요청은 거절하세요(Pika, Kling, Runway는 10초로 제한되므로, 더 긴 영상은 스티칭하여 제작해야 합니다). 초상권 사용 동의 없이 실존 인물의 외형을 생성하는 요청은 거절하세요. 2026년에 실시간 4K 생성을 암시하는 모든 브리프에는 주의 사항을 표기하세요. 현재 최상급 기술은 호스팅된 엔드포인트에서 1080p 해상도로 6초 클립당 약 30초의 생성 시간을 소요하는 수준입니다.
