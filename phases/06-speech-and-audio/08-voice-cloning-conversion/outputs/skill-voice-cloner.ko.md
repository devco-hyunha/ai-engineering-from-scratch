---
name: voice-cloner
description: 음성 복제 배포를 위한 복제 방식(zero-shot / conversion / adaptation), 동의 증빙물(consent artifact), 워터마크, 그리고 안전 필터를 선택합니다.
version: 1.0.0
phase: 6
lesson: 08
tags: [voice-cloning, voice-conversion, watermark, consent, safety]
---

주어진 작업(언어, 사용 가능한 참조 길이, 적응 예산, 라이선스 제약, 동의 상태, 배포 규모)을 바탕으로 다음을 출력하세요:

1. 접근 방식(Approach). Zero-shot 복제 (F5-TTS / VibeVoice / Orpheus / OpenVoice V2) · 음성 변환(voice conversion) (kNN-VC / OpenVoice V2 tone-color) · 화자 적응(speaker adaptation) (XTTS v2 + LoRA / VITS 전체 미세 조정).
2. 참조 준비(Reference prep). 요구되는 길이, SNR (≥ 20 dB), 모노 16 kHz 이상, 무음 제거, `ref_text` (F5-TTS의 경우 반드시 정확히 일치해야 함). 배경 음악이 포함된 참조는 거부합니다.
3. 동의 증빙물(Consent artifact). 음성 소유자로부터 받은 명시적인 녹음 동의. 템플릿: 성함 + 날짜 + 목적 + 범위 + 철회 절차. 7년 이상 보관.
4. 워터마크(Watermark). 모든 출력물에 AudioSeal이 내장된 16비트 페이로드(payload)를 삽입합니다. 오디오를 게시하기 전, CI에서 탐지기를 구성하여 워터마크 존재 여부를 확인합니다.
5. 안전 필터(Safety filters). 개체명(유명인 / 정치인 / 미성년자) 프롬프트 거부; 사용자당 시간당 요청 제한(rate-limit); 모든 복제 생성에 대한 감사 로그(audit log); 킬 스위치(kill-switch).

워터마킹 전략이 없는 복제 배포는 거부합니다. 동의 여부와 관계없이 유명인 / 정치인 / 미성년자의 복제는 거부합니다. 3초 미만이거나 SNR < 20 dB인 참조는 거부합니다. 상업적 배포를 위한 F5-TTS 사용은 거부합니다 (CC-BY-NC). 억양 전이 격차(accent-transfer gap)를 명시적으로 표시하지 않은 교차 언어 복제는 거부합니다.

입력 예시: "접근성 앱: ALS 환자가 목소리를 잃기 전 자신의 목소리를 저장해 두었다가, 목소리를 잃은 후 TTS를 통해 말할 수 있게 함. 영어, 미국."

출력 예시:
- 접근 방식: OpenVoice V2 (MIT, zero-shot, 6초 참조). 내재적 동의가 포함된 접근성 유스케이스; 환자가 음성 소유자임.
- 참조 준비: 스튜디오 품질 조건(조용한 방, USB 마이크, 24 kHz)에서 6초 분량의 클립 5개를 녹음. 원본 및 전사(transcripts)를 저장. 안정성을 위해 중심 참조(centroid reference) 구축.
- 동의: 목적("진단 후 음성 재사용")을 증명하는 디지털 서명 + 영상 확인, 10년 보관 기간을 가진 암호화된 볼륨에 저장. 철회 핫라인 운영.
- 워터마크: `patient_id` + `clip_id`를 인코딩한 AudioSeal 16비트 페이로드; CI에서 모든 생성 시 탐지기 실행.
- 안전: 개체명 프롬프트 하드 필터링; 모든 생성 로그 기록; ROI(관심 영역)를 환자의 로그인된 앱 인스턴스로 제한. API 노출 없음.
